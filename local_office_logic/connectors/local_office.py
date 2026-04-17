import difflib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from ..core.interfaces import ArtifactFetcher, ArtifactPusher
from ..utils.document_adapter import document_to_html, html_to_document
from ..utils.html_builder import build_artifact_html
from ..utils.office_runtime import convert_with_office
from ..utils.sandbox import (
    artifact_id_from_path,
    classify_extension,
    ensure_within,
    get_allowed_extensions,
    get_sandbox_root,
    get_staging_root,
    normalize_relative_path,
    resolve_artifact_path,
    sanitize_filename,
    version_token_for_path,
)
from ..utils.spreadsheet_adapter import csv_to_html, html_to_csv, html_to_workbook, workbook_to_html


class LocalOfficeConnector(ArtifactFetcher, ArtifactPusher):
    def __init__(self):
        self.sandbox_root = get_sandbox_root()
        self.staging_root = get_staging_root()
        self.allowed_extensions = get_allowed_extensions()
        self.default_doc_format = (os.getenv("LOCAL_OFFICE_DEFAULT_DOC_FORMAT") or "docx").strip().lower()
        self.default_sheet_format = (os.getenv("LOCAL_OFFICE_DEFAULT_SHEET_FORMAT") or "xlsx").strip().lower()

    def list_artifacts(self, limit: int = 25) -> List[Dict[str, Any]]:
        artifacts = []
        for path in self._iter_allowed_files():
            stat = path.stat()
            ext = path.suffix.lower()
            artifacts.append(
                {
                    "artifact_id": artifact_id_from_path(path, self.sandbox_root),
                    "title": path.stem,
                    "relative_path": normalize_relative_path(path, self.sandbox_root),
                    "artifact_family": classify_extension(ext),
                    "file_format": ext.lstrip("."),
                    "version_token": version_token_for_path(path),
                    "sort_key": stat.st_mtime_ns,
                }
            )
        artifacts.sort(key=lambda item: item["sort_key"], reverse=True)
        return artifacts[: max(1, min(limit, 200))]

    def search_artifacts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        normalized_query = (query or "").strip().lower()
        if not normalized_query:
            return []

        scored = []
        for artifact in self.list_artifacts(limit=200):
            haystack = f"{artifact['title']} {artifact['relative_path']}".lower()
            substring_bonus = 1.0 if normalized_query in haystack else 0.0
            similarity = difflib.SequenceMatcher(None, normalized_query, haystack).ratio()
            score = similarity + substring_bonus
            artifact_copy = dict(artifact)
            artifact_copy["score"] = score
            scored.append(artifact_copy)
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, min(limit, 50))]

    def get_artifact_metadata(self, artifact_id: str) -> Dict[str, Any]:
        path = resolve_artifact_path(artifact_id, self.sandbox_root)
        ext = path.suffix.lower()
        return {
            "artifact_id": artifact_id,
            "title": path.stem,
            "relative_path": normalize_relative_path(path, self.sandbox_root),
            "artifact_family": classify_extension(ext),
            "file_format": ext.lstrip("."),
            "version_token": version_token_for_path(path),
            "absolute_path": str(path),
        }

    def fetch_artifact_html(self, artifact_id: str) -> Dict[str, Any]:
        metadata = self.get_artifact_metadata(artifact_id)
        path = Path(metadata["absolute_path"])
        snapshot = self._build_snapshot(path)
        metadata.update(snapshot)
        metadata.pop("absolute_path", None)
        return metadata

    def push_artifact_update(
        self,
        artifact_id: str,
        content_html: str,
        expected_version_token: str,
    ) -> Dict[str, Any]:
        metadata = self.get_artifact_metadata(artifact_id)
        if metadata["version_token"] != expected_version_token:
            raise ValueError(
                f"Version Conflict: Expected token {expected_version_token}, "
                f"but live token is {metadata['version_token']}."
            )

        original_path = resolve_artifact_path(artifact_id, self.sandbox_root)
        staging_dir = Path(tempfile.mkdtemp(prefix="office-edit-", dir=self.staging_root))
        ext = original_path.suffix.lower()
        family = metadata["artifact_family"]

        if family == "document":
            working_docx = self._ensure_document_working_copy(original_path, staging_dir)
            html_to_document(content_html, working_docx)
            produced = self._export_document(working_docx, ext, staging_dir, original_path.stem)
        else:
            produced = self._write_spreadsheet_update(original_path, content_html, ext, staging_dir)

        os.replace(produced, original_path)
        return {
            "success": True,
            "version_token": version_token_for_path(original_path),
            "relative_path": normalize_relative_path(original_path, self.sandbox_root),
        }

    def rename_artifact(
        self,
        artifact_id: str,
        expected_version_token: str,
        new_title: str,
    ) -> Dict[str, Any]:
        metadata = self.get_artifact_metadata(artifact_id)
        if metadata["version_token"] != expected_version_token:
            raise ValueError(
                f"Version Conflict: Expected token {expected_version_token}, "
                f"but live token is {metadata['version_token']}."
            )

        original_path = resolve_artifact_path(artifact_id, self.sandbox_root)
        safe_stem = sanitize_filename(new_title)
        new_path = ensure_within(original_path.parent, (original_path.parent / f"{safe_stem}{original_path.suffix}").resolve())
        if new_path.exists() and new_path != original_path:
            raise ValueError(f"An artifact named '{new_path.name}' already exists.")

        if metadata["artifact_family"] == "document":
            snapshot = self.fetch_artifact_html(artifact_id)
            adjusted_html = self._rewrite_visible_title_if_needed(snapshot["full_html"], original_path.stem, safe_stem)
            if adjusted_html != snapshot["full_html"]:
                self.push_artifact_update(artifact_id, adjusted_html, expected_version_token)

        os.replace(original_path, new_path)
        return {
            "success": True,
            "artifact_id": artifact_id_from_path(new_path, self.sandbox_root),
            "relative_path": normalize_relative_path(new_path, self.sandbox_root),
            "version_token": version_token_for_path(new_path),
        }

    def create_artifact(
        self,
        title: str,
        artifact_family: str,
        file_format: str,
        content_html: str,
    ) -> Dict[str, Any]:
        family = artifact_family.strip().lower()
        requested_format = file_format.strip().lower() if file_format else (
            self.default_doc_format if family == "document" else self.default_sheet_format
        )
        if not requested_format.startswith("."):
            requested_format = f".{requested_format}"
        if requested_format not in self.allowed_extensions:
            raise ValueError(f"Unsupported artifact format: {requested_format}")

        safe_stem = sanitize_filename(title)
        destination = ensure_within(
            self.sandbox_root,
            (self.sandbox_root / f"{safe_stem}{requested_format}").resolve(),
        )
        if destination.exists():
            raise ValueError(f"An artifact named '{destination.name}' already exists.")

        staging_dir = Path(tempfile.mkdtemp(prefix="office-create-", dir=self.staging_root))
        if family == "document":
            working_docx = staging_dir / f"{safe_stem}.docx"
            html_to_document(content_html, working_docx)
            produced = self._export_document(working_docx, requested_format, staging_dir, safe_stem)
        else:
            produced = self._write_spreadsheet_update(destination, content_html, requested_format, staging_dir, creating=True)

        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(produced, destination)
        return {
            "success": True,
            "artifact_id": artifact_id_from_path(destination, self.sandbox_root),
            "relative_path": normalize_relative_path(destination, self.sandbox_root),
            "title": destination.stem,
            "file_format": destination.suffix.lstrip("."),
            "version_token": version_token_for_path(destination),
        }

    def build_content_html(
        self,
        title: str,
        artifact_family: str,
        body_text: str = "",
        sections: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        return build_artifact_html(
            title=title,
            artifact_family=artifact_family,
            body_text=body_text,
            sections=sections,
        )

    def _iter_allowed_files(self):
        for path in self.sandbox_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.allowed_extensions and self.staging_root not in path.parents:
                yield path

    def _build_snapshot(self, original_path: Path) -> Dict[str, Any]:
        ext = original_path.suffix.lower()
        family = classify_extension(ext)
        staging_dir = Path(tempfile.mkdtemp(prefix="office-fetch-", dir=self.staging_root))

        if family == "document":
            working_path = self._ensure_document_working_copy(original_path, staging_dir)
            full_html, targets = document_to_html(working_path)
        else:
            working_path = self._ensure_spreadsheet_working_copy(original_path, staging_dir)
            if working_path.suffix.lower() == ".csv":
                full_html, targets = csv_to_html(working_path)
            else:
                full_html, targets = workbook_to_html(working_path)

        return {
            "full_html": full_html,
            "available_targets": targets,
            "working_path": str(working_path),
        }

    def _ensure_document_working_copy(self, original_path: Path, staging_dir: Path) -> Path:
        ext = original_path.suffix.lower()
        staged_input = staging_dir / original_path.name
        shutil.copy2(original_path, staged_input)
        if ext == ".docx":
            return staged_input
        return convert_with_office(staged_input, staging_dir, ".docx")

    def _ensure_spreadsheet_working_copy(self, original_path: Path, staging_dir: Path) -> Path:
        ext = original_path.suffix.lower()
        staged_input = staging_dir / original_path.name
        shutil.copy2(original_path, staged_input)
        if ext in {".xlsx", ".csv"}:
            return staged_input
        return convert_with_office(staged_input, staging_dir, ".xlsx")

    def _export_document(self, working_docx: Path, source_ext: str, staging_dir: Path, stem: str) -> Path:
        if source_ext == ".docx":
            return working_docx
        converted = convert_with_office(working_docx, staging_dir, source_ext)
        target = staging_dir / f"{stem}{source_ext}"
        if converted != target:
            shutil.copy2(converted, target)
        return target

    def _write_spreadsheet_update(
        self,
        original_path: Path,
        content_html: str,
        source_ext: str,
        staging_dir: Path,
        creating: bool = False,
    ) -> Path:
        if source_ext == ".csv":
            output_csv = staging_dir / (original_path.name if not creating else f"{original_path.stem}.csv")
            html_to_csv(content_html, output_csv)
            return output_csv

        if creating:
            working_xlsx = staging_dir / f"{original_path.stem}.xlsx"
        else:
            working_xlsx = self._ensure_spreadsheet_working_copy(original_path, staging_dir)

        html_to_workbook(content_html, working_xlsx)
        if source_ext == ".xlsx":
            return working_xlsx

        converted = convert_with_office(working_xlsx, staging_dir, source_ext)
        target = staging_dir / f"{original_path.stem}{source_ext}"
        if converted != target:
            shutil.copy2(converted, target)
        return target

    def _rewrite_visible_title_if_needed(self, full_html: str, old_stem: str, new_title: str) -> str:
        soup = BeautifulSoup(full_html, "html.parser")
        first_heading = soup.find(["h1", "h2"])
        if not first_heading:
            return full_html
        heading_text = first_heading.get_text(" ", strip=True)
        if heading_text.strip().lower() != old_stem.strip().lower():
            return full_html
        first_heading.string = new_title
        return str(soup)
