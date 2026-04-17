import hashlib
import os
from pathlib import Path
from typing import Optional


DOCUMENT_FORMATS = {".docx", ".doc", ".odt", ".rtf"}
SPREADSHEET_FORMATS = {".xlsx", ".xls", ".ods", ".csv"}
DEFAULT_ALLOWED_EXTENSIONS = DOCUMENT_FORMATS | SPREADSHEET_FORMATS


def get_sandbox_root() -> Path:
    configured = (os.getenv("LOCAL_OFFICE_SANDBOX_ROOT") or "").strip()
    if not configured:
        raise ValueError("LOCAL_OFFICE_SANDBOX_ROOT is not set.")
    root = Path(configured).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_staging_root() -> Path:
    configured = (os.getenv("LOCAL_OFFICE_STAGING_ROOT") or "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = get_sandbox_root() / ".local_office_staging"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_allowed_extensions() -> set[str]:
    configured = (os.getenv("LOCAL_OFFICE_ALLOWED_EXTENSIONS") or "").strip()
    if not configured:
        return set(DEFAULT_ALLOWED_EXTENSIONS)
    return {
        ext if ext.startswith(".") else f".{ext}"
        for ext in (piece.strip().lower() for piece in configured.split(","))
        if ext
    }


def classify_extension(extension: str) -> str:
    normalized = extension.lower()
    if normalized in DOCUMENT_FORMATS:
        return "document"
    if normalized in SPREADSHEET_FORMATS:
        return "spreadsheet"
    raise ValueError(f"Unsupported local office format: {extension}")


def is_document_extension(extension: str) -> bool:
    return extension.lower() in DOCUMENT_FORMATS


def is_spreadsheet_extension(extension: str) -> bool:
    return extension.lower() in SPREADSHEET_FORMATS


def ensure_within(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path escapes the configured local office sandbox.")
    return candidate


def normalize_relative_path(path: Path, root: Optional[Path] = None) -> str:
    root = root or get_sandbox_root()
    resolved = ensure_within(root, path)
    return resolved.relative_to(root).as_posix()


def artifact_id_from_path(path: Path, root: Optional[Path] = None) -> str:
    return normalize_relative_path(path, root=root)


def resolve_artifact_path(artifact_id: str, root: Optional[Path] = None) -> Path:
    root = root or get_sandbox_root()
    if not artifact_id or artifact_id.startswith("/") or artifact_id.startswith("\\"):
        raise ValueError("Artifact id must be a sandbox-relative path.")
    candidate = ensure_within(root, (root / artifact_id).resolve())
    if not candidate.exists():
        raise FileNotFoundError(f"Artifact '{artifact_id}' does not exist in the sandbox.")
    return candidate


def sanitize_filename(title: str) -> str:
    text = (title or "").strip() or "Untitled"
    cleaned = "".join(ch if ch.isalnum() or ch in {" ", "-", "_"} else " " for ch in text)
    collapsed = " ".join(cleaned.split()).strip(" .")
    return collapsed or "Untitled"


def version_token_for_path(path: Path) -> str:
    stat = path.stat()
    digest = hashlib.sha1()
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    with path.open("rb") as handle:
        digest.update(handle.read(65536))
    return digest.hexdigest()
