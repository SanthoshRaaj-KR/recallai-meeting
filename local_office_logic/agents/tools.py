import concurrent.futures
import difflib
import threading
from contextvars import ContextVar
from typing import Callable, List, Optional

from agents import function_tool

from ..connectors.local_office import LocalOfficeConnector
from ..core.schemas import (
    ArtifactCommitResponse,
    ArtifactPreviewResponse,
    ArtifactSearchResponse,
    ArtifactSectionInput,
    CandidateArtifact,
    CreateArtifactResponse,
    LiveArtifactResponse,
)
from ..db.vector_store import PineconeStore
from ..utils.html_parser import delete_content_in_section, edit_block_in_section, get_section_html
import logging


logger = logging.getLogger(__name__)

_store = None
_connector = None


def _fresh_tool_state() -> dict:
    return {"last_action": None, "success": None, "message": "", "html_cache": {}, "version_cache": {}}


_tool_run_state: ContextVar[dict] = ContextVar("tool_run_state")
_mutation_observer: ContextVar[Optional[Callable[[str], None]]] = ContextVar(
    "mutation_observer",
    default=None,
)


def get_store():
    global _store
    if _store is None:
        _store = PineconeStore()
    return _store


def get_connector():
    global _connector
    if _connector is None:
        _connector = LocalOfficeConnector()
    return _connector


def reset_tool_state() -> None:
    _tool_run_state.set(_fresh_tool_state())


def get_tool_state() -> dict:
    try:
        return _tool_run_state.get()
    except LookupError:
        state = _fresh_tool_state()
        _tool_run_state.set(state)
        return state


def set_mutation_observer(observer: Optional[Callable[[str], None]]) -> None:
    _mutation_observer.set(observer)


def _emit_mutation_started(action: str) -> None:
    observer = _mutation_observer.get()
    if observer is None:
        return
    try:
        observer(action)
    except Exception as exc:
        logger.warning("Mutation observer failed: %s", exc)


def _similarity_score(query: str, title: str) -> float:
    return difflib.SequenceMatcher(None, (query or "").lower(), (title or "").lower()).ratio()


def _reindex_in_background(artifact_id: str) -> None:
    def _run():
        try:
            from ..ingestion.doc_pipeline import IngestionPipeline

            IngestionPipeline().process_artifact(artifact_id)
        except Exception as exc:
            logger.error("Background re-indexing failed for %s: %s", artifact_id, exc)

    threading.Thread(target=_run, daemon=True).start()


def _candidate_from_metadata(meta: dict) -> CandidateArtifact:
    return CandidateArtifact(
        artifact_id=meta.get("artifact_id", ""),
        title=meta.get("title", "Unknown"),
        relative_path=meta.get("relative_path", ""),
        artifact_family=meta.get("artifact_family", "document"),
        file_format=meta.get("file_format", ""),
        section_label=meta.get("section_label") or None,
        snippet=(meta.get("text_summary") or meta.get("snippet") or meta.get("relative_path") or meta.get("title") or "")[:160],
    )


def format_artifact_titles_for_user(candidates: List[CandidateArtifact]) -> str:
    if not candidates:
        return "I couldn't find any local office files."

    title_counts: dict[str, int] = {}
    for candidate in candidates:
        title_counts[candidate.title] = title_counts.get(candidate.title, 0) + 1

    rendered = []
    seen = set()
    for candidate in candidates:
        text = candidate.title
        if title_counts[candidate.title] > 1:
            qualifier = candidate.relative_path or candidate.section_label or candidate.file_format
            if qualifier:
                text = f"{candidate.title} - {qualifier}"
        if text not in seen:
            rendered.append(text)
            seen.add(text)
    return ", ".join(rendered)


@function_tool
def list_sandbox_artifacts(limit: int = 100) -> ArtifactSearchResponse:
    try:
        items = get_connector().list_artifacts(limit=limit)
        return ArtifactSearchResponse(
            candidates=[_candidate_from_metadata(item) for item in items if item.get("artifact_id")],
            message="Success",
        )
    except Exception as exc:
        logger.error("List sandbox artifacts failed: %s", exc)
        return ArtifactSearchResponse(candidates=[], message=f"Error: {exc}")


@function_tool
def search_sandbox_artifacts(query: str) -> ArtifactSearchResponse:
    try:
        connector = get_connector()
        store = get_store()

        def _live_search():
            try:
                return connector.search_artifacts(query, limit=8)
            except Exception as exc:
                logger.error("Live sandbox search failed: %s", exc)
                return []

        def _recent():
            try:
                return connector.list_artifacts(limit=100)
            except Exception as exc:
                logger.warning("Recent sandbox artifact listing failed: %s", exc)
                return []

        def _semantic():
            try:
                return store.search(query, top_k=5)
            except Exception as exc:
                logger.warning("Local office Pinecone unavailable, continuing with live results: %s", exc)
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            live_results = executor.submit(_live_search).result()
            recent_results = executor.submit(_recent).result()
            semantic_results = executor.submit(_semantic).result()

        candidates_by_artifact: dict[str, CandidateArtifact] = {}
        scores: dict[str, float] = {}

        for item in live_results + recent_results:
            artifact_id = item.get("artifact_id", "")
            if not artifact_id:
                continue
            candidate = _candidate_from_metadata(item)
            score = _similarity_score(query, candidate.title)
            if artifact_id not in candidates_by_artifact or score > scores.get(artifact_id, float("-inf")):
                candidates_by_artifact[artifact_id] = candidate
                scores[artifact_id] = score

        for match in semantic_results:
            meta = match.get("metadata", {})
            artifact_id = meta.get("artifact_id", "")
            if not artifact_id:
                continue
            existing = candidates_by_artifact.get(artifact_id)
            if existing is None:
                candidates_by_artifact[artifact_id] = _candidate_from_metadata(meta)
                scores[artifact_id] = _similarity_score(query, candidates_by_artifact[artifact_id].title)
            else:
                if not existing.section_label and meta.get("section_label"):
                    existing.section_label = meta.get("section_label")
                if meta.get("text_summary"):
                    existing.snippet = meta["text_summary"][:160]

        candidates = sorted(
            candidates_by_artifact.values(),
            key=lambda cand: (-scores.get(cand.artifact_id, _similarity_score(query, cand.title)), cand.title.lower()),
        )
        if not candidates:
            return ArtifactSearchResponse(candidates=[], message="No matching local office files found.")
        return ArtifactSearchResponse(candidates=candidates[:5], message="Success")
    except Exception as exc:
        logger.error("Search sandbox artifacts failed: %s", exc)
        return ArtifactSearchResponse(candidates=[], message=f"Error: {exc}")


@function_tool
def fetch_live_artifact(artifact_id: str, section_label: Optional[str] = None) -> LiveArtifactResponse:
    try:
        snapshot = get_connector().fetch_artifact_html(artifact_id)
        state = get_tool_state()
        state.setdefault("html_cache", {})[artifact_id] = snapshot["full_html"]
        state.setdefault("version_cache", {})[artifact_id] = snapshot["version_token"]
        section_html = None
        if section_label:
            section_html = get_section_html(snapshot["full_html"], section_label)
        return LiveArtifactResponse(
            artifact_id=artifact_id,
            expected_version_token=snapshot["version_token"],
            available_targets=snapshot["available_targets"],
            section_html=section_html,
            message="Artifact loaded.",
        )
    except Exception as exc:
        logger.error("Fetch live artifact failed: %s", exc)
        return LiveArtifactResponse(
            artifact_id=artifact_id,
            expected_version_token="",
            available_targets=[],
            section_html=None,
            message=str(exc),
        )


@function_tool
def preview_artifact_edit(
    artifact_id: str,
    section_label: str,
    old_block_html: str = "",
    new_block_html: str = "",
) -> ArtifactPreviewResponse:
    try:
        state = get_tool_state()
        live_html = state.get("html_cache", {}).get(artifact_id) or get_connector().fetch_artifact_html(artifact_id)["full_html"]
        preview_html = edit_block_in_section(live_html, section_label, old_block_html, new_block_html)
        diff = "".join(
            difflib.unified_diff(
                live_html.splitlines(keepends=True),
                preview_html.splitlines(keepends=True),
                fromfile="Current",
                tofile="Preview",
            )
        )
        if not diff:
            return ArtifactPreviewResponse(success=False, diff="", message="No visible artifact changes detected.")
        return ArtifactPreviewResponse(success=True, diff=diff, message="Preview generated.")
    except Exception as exc:
        return ArtifactPreviewResponse(success=False, diff="", message=f"Failed artifact preview: {exc}")


@function_tool
def preview_artifact_delete(
    artifact_id: str,
    section_label: str,
    target_html_or_text: str = "",
    delete_entire_section: bool = False,
) -> ArtifactPreviewResponse:
    try:
        state = get_tool_state()
        live_html = state.get("html_cache", {}).get(artifact_id) or get_connector().fetch_artifact_html(artifact_id)["full_html"]
        preview_html = delete_content_in_section(
            live_html,
            section_label,
            target_html_or_text=target_html_or_text,
            delete_entire_section=delete_entire_section,
        )
        diff = "".join(
            difflib.unified_diff(
                live_html.splitlines(keepends=True),
                preview_html.splitlines(keepends=True),
                fromfile="Current",
                tofile="PreviewDelete",
            )
        )
        if not diff:
            return ArtifactPreviewResponse(success=False, diff="", message="No visible delete changes detected.")
        return ArtifactPreviewResponse(success=True, diff=diff, message="Delete preview generated.")
    except Exception as exc:
        return ArtifactPreviewResponse(success=False, diff="", message=f"Failed delete preview: {exc}")


@function_tool
def rename_local_artifact(artifact_id: str, expected_version_token: str, new_title: str) -> ArtifactCommitResponse:
    try:
        _emit_mutation_started("rename")
        result = get_connector().rename_artifact(artifact_id, expected_version_token, new_title)
        _reindex_in_background(result["artifact_id"])
        message = f"Artifact renamed successfully to {new_title}"
        _tool_run_state.set({"last_action": "commit", "success": True, "message": message})
        return ArtifactCommitResponse(success=True, version_token=result["version_token"], message=message)
    except Exception as exc:
        logger.error("Artifact rename failed: %s", exc)
        message = str(exc)
        _tool_run_state.set({"last_action": "commit", "success": False, "message": message})
        return ArtifactCommitResponse(success=False, version_token=None, message=message)


@function_tool
def commit_artifact_delete(
    artifact_id: str,
    expected_version_token: str,
    section_label: str,
    target_html_or_text: str = "",
    delete_entire_section: bool = False,
) -> ArtifactCommitResponse:
    try:
        _emit_mutation_started("delete")
        snapshot = get_connector().fetch_artifact_html(artifact_id)
        new_html = delete_content_in_section(
            snapshot["full_html"],
            section_label,
            target_html_or_text=target_html_or_text,
            delete_entire_section=delete_entire_section,
        )
        result = get_connector().push_artifact_update(artifact_id, new_html, expected_version_token)
        _reindex_in_background(artifact_id)
        message = f"Delete successful on artifact {artifact_id}"
        _tool_run_state.set({"last_action": "commit", "success": True, "message": message})
        return ArtifactCommitResponse(success=True, version_token=result["version_token"], message=message)
    except Exception as exc:
        logger.error("Commit artifact delete failed: %s", exc)
        message = str(exc)
        _tool_run_state.set({"last_action": "commit", "success": False, "message": message})
        return ArtifactCommitResponse(success=False, version_token=None, message=message)


@function_tool
def commit_artifact_edit(
    artifact_id: str,
    expected_version_token: str,
    section_label: str,
    old_block_html: str = "",
    new_block_html: str = "",
) -> ArtifactCommitResponse:
    try:
        _emit_mutation_started("edit")
        snapshot = get_connector().fetch_artifact_html(artifact_id)
        new_html = edit_block_in_section(snapshot["full_html"], section_label, old_block_html, new_block_html)
        result = get_connector().push_artifact_update(artifact_id, new_html, expected_version_token)
        _reindex_in_background(artifact_id)
        message = f"Commit successful on artifact {artifact_id}"
        _tool_run_state.set({"last_action": "commit", "success": True, "message": message})
        return ArtifactCommitResponse(success=True, version_token=result["version_token"], message=message)
    except Exception as exc:
        logger.error("Commit artifact edit failed: %s", exc)
        message = str(exc)
        _tool_run_state.set({"last_action": "commit", "success": False, "message": message})
        return ArtifactCommitResponse(success=False, version_token=None, message=message)


@function_tool
def create_local_artifact(
    title: str,
    artifact_family: str = "document",
    file_format: str = "",
    body_text: str = "",
    sections: Optional[List[ArtifactSectionInput]] = None,
) -> CreateArtifactResponse:
    try:
        connector = get_connector()
        html_payload = connector.build_content_html(
            title=title,
            artifact_family=artifact_family,
            body_text=body_text,
            sections=[section.model_dump() if isinstance(section, ArtifactSectionInput) else section for section in (sections or [])],
        )
        _emit_mutation_started("create")
        result = connector.create_artifact(
            title=title,
            artifact_family=artifact_family,
            file_format=file_format or "",
            content_html=html_payload,
        )
        _reindex_in_background(result["artifact_id"])
        message = "Local office artifact created and indexed."
        _tool_run_state.set({"last_action": "create", "success": True, "message": message})
        return CreateArtifactResponse(
            success=True,
            artifact_id=result["artifact_id"],
            title=title,
            relative_path=result["relative_path"],
            file_format=result["file_format"],
            version_token=result["version_token"],
            message=message,
        )
    except Exception as exc:
        logger.error("Local office artifact creation failed: %s", exc)
        message = str(exc)
        _tool_run_state.set({"last_action": "create", "success": False, "message": message})
        return CreateArtifactResponse(
            success=False,
            artifact_id=None,
            title=None,
            relative_path=None,
            file_format=None,
            version_token=None,
            message=message,
        )
