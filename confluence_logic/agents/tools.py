import difflib
from contextvars import ContextVar
from typing import List, Optional
from ..db.vector_store import PineconeStore
from ..connectors.confluence import ConfluenceConnector
from ..core.schemas import SearchResponse, CandidatePage, LivePageResponse, PreviewResponse, CommitResponse, CreatePageResponse, PageSectionInput
from ..utils.html_parser import edit_block_in_section, extract_headings, get_section_html
import logging
from agents import function_tool


logger = logging.getLogger(__name__)

_store = None
_connector = None
_tool_run_state: ContextVar[dict] = ContextVar(
    "tool_run_state",
    default={"last_action": None, "success": None, "message": ""},
)

def get_store():
    global _store
    if _store is None:
        _store = PineconeStore()
    return _store

def get_connector():
    global _connector
    if _connector is None:
        _connector = ConfluenceConnector()
    return _connector

def reset_tool_state() -> None:
    _tool_run_state.set({"last_action": None, "success": None, "message": ""})

def get_tool_state() -> dict:
    return _tool_run_state.get()

def _is_version_conflict(error: Exception) -> bool:
    return "Version Conflict" in str(error)

def _similarity_score(query: str, title: str) -> float:
    return difflib.SequenceMatcher(None, (query or "").lower(), (title or "").lower()).ratio()

def _candidate_from_metadata(meta: dict) -> CandidatePage:
    return CandidatePage(
        page_id=meta.get("page_id", ""),
        title=meta.get("title", "Unknown"),
        heading=meta.get("heading", "") or None,
        is_root_section=meta.get("is_root_section", False),
        space_key=meta.get("space_key", ""),
        snippet=(meta.get("text_summary") or meta.get("excerpt") or meta.get("title") or "")[:140],
    )

@function_tool
def list_workspace_pages(limit: int = 100) -> SearchResponse:
    """Lists recent Confluence pages so a resolver agent can reason over real workspace page options."""
    try:
        items = get_connector().list_pages(limit=limit)
        candidates = [_candidate_from_metadata(item) for item in items if item.get("page_id")]
        return SearchResponse(candidates=candidates, message="Success")
    except Exception as e:
        logger.error(f"List workspace pages failed: {e}")
        return SearchResponse(candidates=[], message=f"Error: {e}")

@function_tool
def search_workspace_knowledge(query: str) -> SearchResponse:
    """Searches live Confluence pages first, then supplements with Pinecone if available."""
    try:
        candidates_by_page: dict[str, CandidatePage] = {}
        scored_candidates: dict[str, float] = {}

        try:
            live_results = get_connector().search_pages(query, limit=8)
        except Exception as live_error:
            logger.error(f"Live Confluence search failed: {live_error}")
            live_results = []

        try:
            recent_pages = get_connector().list_pages(limit=100)
        except Exception as recent_error:
            logger.warning(f"Recent Confluence page listing failed: {recent_error}")
            recent_pages = []

        for item in live_results + recent_pages:
            page_id = item.get("page_id", "")
            if not page_id:
                continue

            candidate = _candidate_from_metadata(item)
            score = _similarity_score(query, candidate.title)
            existing = candidates_by_page.get(page_id)

            if existing is None or score > scored_candidates.get(page_id, float("-inf")):
                candidates_by_page[page_id] = candidate
                scored_candidates[page_id] = score

        try:
            pinecone_results = get_store().search(query, top_k=5)
        except Exception as pinecone_error:
            logger.warning(f"Pinecone search unavailable, continuing with live Confluence results: {pinecone_error}")
            pinecone_results = []

        for match in pinecone_results:
            meta = match.get("metadata", {})
            page_id = meta.get("page_id", "")
            if not page_id:
                continue

            existing = candidates_by_page.get(page_id)
            snippet = meta.get("text_summary", "")[:140]
            heading = meta.get("heading", "") or None

            if existing is None:
                candidates_by_page[page_id] = _candidate_from_metadata(meta)
                scored_candidates[page_id] = max(
                    scored_candidates.get(page_id, float("-inf")),
                    _similarity_score(query, candidates_by_page[page_id].title),
                )
            elif not existing.heading and heading:
                existing.heading = heading
                if snippet:
                    existing.snippet = snippet

        candidates = sorted(
            candidates_by_page.values(),
            key=lambda cand: (
                -scored_candidates.get(cand.page_id, _similarity_score(query, cand.title)),
                cand.title.lower(),
            ),
        )

        if not candidates:
            return SearchResponse(candidates=[], message="No matching Confluence pages found.")

        return SearchResponse(candidates=candidates[:5], message="Success")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return SearchResponse(candidates=[], message=f"Error: {e}")
@function_tool
def fetch_live_page(page_id: str, heading_string: Optional[str] = None) -> LivePageResponse:
    """Fetches the latest live Confluence page parsing out available headings. Provide heading_string to isolate raw section_html."""
    try:
        html = get_connector().fetch_page_html(page_id)
        metadata = get_connector().get_page_metadata(page_id)
        version = metadata.get("version", {}).get("number", 1)
        headings = extract_headings(html)
        
        section_html = None
        if heading_string:
            section_html = get_section_html(html, heading_string)
            
        return LivePageResponse(page_id=page_id, expected_version=version, available_headings=headings, section_html=section_html, message="Page loaded. Anchor to an available heading.")
    except Exception as e:
        logger.error(f"Fetch live page failed: {e}")
        return LivePageResponse(page_id=page_id, expected_version=-1, available_headings=[], section_html=None, message=str(e))
@function_tool
def preview_edit(page_id: str, heading_string: str, old_block_html: str = "", new_block_html: str = "") -> PreviewResponse:
    """Generates a DOM modified HTML preview applying differencing logic to preview the exact modification visually."""
    try:
        live_html = get_connector().fetch_page_html(page_id)
        new_document_html = edit_block_in_section(live_html, heading_string, old_block_html, new_block_html)
        
        diff_lines = list(difflib.unified_diff(
            live_html.splitlines(keepends=True),
            new_document_html.splitlines(keepends=True),
            fromfile='Current',
            tofile='Preview'
        ))
        
        diff_str = "".join(diff_lines)
        if not diff_str:
            return PreviewResponse(success=False, diff="", message="No visible DOM changes detected.")
            
        return PreviewResponse(success=True, diff=diff_str, message="Preview generated. Please verify diff string before executing commit.")
    except Exception as e:
        return PreviewResponse(success=False, diff="", message=f"Failed DOM manipulation: {str(e)}")

@function_tool
def update_page_title(page_id: str, expected_version: int, new_title: str) -> CommitResponse:
    """Updates the title of an existing Confluence page without creating a new page."""
    try:
        live_html = get_connector().fetch_page_html(page_id)
        try:
            success = get_connector().push_update(
                page_id,
                live_html,
                expected_version=expected_version,
                title_override=new_title,
            )
            committed_version = expected_version + 1 if success else None
        except ValueError as ve:
            if not _is_version_conflict(ve):
                raise
            logger.error(f"Version Lock conflict during title update: {ve}")
            refreshed_metadata = get_connector().get_page_metadata(page_id)
            refreshed_version = refreshed_metadata.get("version", {}).get("number", 1)
            refreshed_html = get_connector().fetch_page_html(page_id)
            success = get_connector().push_update(
                page_id,
                refreshed_html,
                expected_version=refreshed_version,
                title_override=new_title,
            )
            committed_version = refreshed_version + 1 if success else None

        if success:
            try:
                from ..ingestion.doc_pipeline import IngestionPipeline
                IngestionPipeline().process_page(page_id)
            except Exception as pipeline_err:
                logger.error(f"Post-title-update automatic re-indexing failed for {page_id}: {pipeline_err}")

            message = f"Title updated successfully on page {page_id}"
            _tool_run_state.set({"last_action": "commit", "success": True, "message": message})
            return CommitResponse(success=True, version=committed_version, message=message)

        message = "Page title update failed."
        _tool_run_state.set({"last_action": "commit", "success": False, "message": message})
        return CommitResponse(success=False, version=None, message=message)
    except Exception as e:
        logger.error(f"Title update failed: {e}")
        message = str(e)
        _tool_run_state.set({"last_action": "commit", "success": False, "message": message})
        return CommitResponse(success=False, version=None, message=message)

@function_tool
def commit_document_edit(page_id: str, expected_version: int, heading_string: str, old_block_html: str = "", new_block_html: str = "") -> CommitResponse:
    """Commits a parsed structural sub-section DOM replacement securely using Op-Locking bounds."""
    try:
        live_html = get_connector().fetch_page_html(page_id)
        new_document_html = edit_block_in_section(live_html, heading_string, old_block_html, new_block_html)

        try:
            success = get_connector().push_update(page_id, new_document_html, expected_version=expected_version)
            committed_version = expected_version + 1 if success else None
        except ValueError as ve:
            if not _is_version_conflict(ve):
                raise
            logger.error(f"Version Lock conflict: {ve}")
            refreshed_metadata = get_connector().get_page_metadata(page_id)
            refreshed_version = refreshed_metadata.get("version", {}).get("number", 1)
            refreshed_html = get_connector().fetch_page_html(page_id)
            refreshed_document_html = edit_block_in_section(refreshed_html, heading_string, old_block_html, new_block_html)
            success = get_connector().push_update(page_id, refreshed_document_html, expected_version=refreshed_version)
            committed_version = refreshed_version + 1 if success else None

        if success:
            try:
                from ..ingestion.doc_pipeline import IngestionPipeline
                IngestionPipeline().process_page(page_id)
            except Exception as pipeline_err:
                logger.error(f"Post-commit automatic re-indexing failed for {page_id}: {pipeline_err}")

            message = f"Commit successful on page {page_id}"
            _tool_run_state.set({"last_action": "commit", "success": True, "message": message})
            return CommitResponse(success=True, version=committed_version, message=message)

        message = "Push update failed."
        _tool_run_state.set({"last_action": "commit", "success": False, "message": message})
        return CommitResponse(success=False, version=None, message=message)
    except ValueError as ve:
        if _is_version_conflict(ve):
            logger.error(f"Version Lock conflict: {ve}")
            message = f"ConflictError: {str(ve)} Please re-fetch_live_page."
        else:
            logger.error(f"Commit edit target resolution failed: {ve}")
            message = str(ve)
        _tool_run_state.set({"last_action": "commit", "success": False, "message": message})
        return CommitResponse(success=False, version=None, message=message)
    except Exception as e:
        logger.error(f"Commit failed: {e}")
        message = str(e)
        _tool_run_state.set({"last_action": "commit", "success": False, "message": message})
        return CommitResponse(success=False, version=None, message=message)

@function_tool
def create_confluence_page(
    title: str,
    space_key: Optional[str] = None,
    body_text: str = "",
    sections: Optional[List[PageSectionInput]] = None,
    parent_page_id: Optional[str] = None,
) -> CreatePageResponse:
    """
    Creates a new blank or populated Confluence Document natively on the workspace and commits to Pinecone simultaneously.
    If providing structured sections, mapping should follow dicts natively {"heading": "Goals", "content": "- point 1"}.
    """
    from ..utils.html_builder import build_page_html
    
    try:
        html_payload = build_page_html(title=title, sections=sections, body_text=body_text)
        result = get_connector().create_page(
            space_key=space_key,
            title=title,
            content=html_payload,
            parent_page_id=parent_page_id
        )
        
        new_page_id = result.get('id')
        if new_page_id:
            try:
                from ..ingestion.doc_pipeline import IngestionPipeline
                IngestionPipeline().process_page(new_page_id)
            except Exception as pipe_e:
                logger.error(f"Post-creation indexing error: {pipe_e}")

            message = "Page generation and indexing achieved."
            _tool_run_state.set({"last_action": "create", "success": True, "message": message})
            return CreatePageResponse(
                success=True,
                page_id=new_page_id,
                title=result.get("title"),
                space_key=space_key,
                version=result.get("version", {}).get("number", 1),
                message=message
            )

        message = "Result ID not accessible."
        _tool_run_state.set({"last_action": "create", "success": False, "message": message})
        return CreatePageResponse(success=False, page_id=None, title=None, space_key=None, version=None, message=message)
        
    except Exception as e:
        logger.error(f"Creation failed: {e}")
        message = str(e)
        _tool_run_state.set({"last_action": "create", "success": False, "message": message})
        return CreatePageResponse(success=False, page_id=None, title=None, space_key=None, version=None, message=message)
