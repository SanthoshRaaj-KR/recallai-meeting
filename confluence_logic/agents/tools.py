import concurrent.futures
import asyncio
import difflib
import threading
import time
from contextvars import ContextVar
from typing import Callable, List, Optional
from ..db.vector_store import PineconeStore
from ..connectors.confluence import ConfluenceConnector
from ..core.schemas import SearchResponse, CandidatePage, LivePageResponse, PreviewResponse, CommitResponse, CreatePageResponse, PageSectionInput
from ..utils.html_parser import delete_content_in_section, edit_block_in_section, extract_headings, get_section_html
from ..utils.html_builder import markdown_to_html
import logging
from agents import function_tool
from confluence_logic import confluence_page_graph


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
        _connector = ConfluenceConnector()
    return _connector

_MAX_VERSION_RETRIES = 3


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
    except Exception as e:
        logger.warning(f"Mutation observer failed: {e}")

def _is_version_conflict(error: Exception) -> bool:
    return "Version Conflict" in str(error)

def _similarity_score(query: str, title: str) -> float:
    return difflib.SequenceMatcher(None, (query or "").lower(), (title or "").lower()).ratio()


def _reindex_in_background(page_id: str) -> None:
    """Fire-and-forget: re-index page in a daemon thread so commits return immediately."""
    def _run():
        try:
            from ..ingestion.doc_pipeline import IngestionPipeline
            IngestionPipeline().process_page(page_id)
        except Exception as exc:
            logger.error("Background re-indexing failed for %s: %s", page_id, exc)

    threading.Thread(target=_run, daemon=True).start()


def _commit_with_retry(
    page_id: str,
    apply_fn,
    expected_version: int,
    title_override: Optional[str] = None,
):
    """Fetch, transform, and push with bounded exponential-backoff retry on version conflicts.

    apply_fn: Callable[[str], str] — receives live HTML, returns new HTML to commit.
    Returns (success: bool, committed_version: Optional[int]).
    Raises ValueError after _MAX_VERSION_RETRIES consecutive version conflicts.
    """
    connector = get_connector()
    for attempt in range(_MAX_VERSION_RETRIES):
        try:
            live_html = connector.fetch_page_html(page_id)
            new_html = apply_fn(live_html)
            success = connector.push_update(
                page_id, new_html,
                expected_version=expected_version,
                title_override=title_override,
            )
            return success, (expected_version + 1 if success else None)
        except ValueError as ve:
            if not _is_version_conflict(ve):
                raise
            if attempt == _MAX_VERSION_RETRIES - 1:
                logger.error(
                    "Version conflict: max retries (%d) exceeded for page %s",
                    _MAX_VERSION_RETRIES, page_id,
                )
                raise
            logger.warning(
                "Version conflict on attempt %d/%d for page %s, retrying in %.1fs...",
                attempt + 1, _MAX_VERSION_RETRIES, page_id, 0.5 * (2 ** attempt),
            )
            meta = connector.get_page_metadata(page_id)
            expected_version = meta.get("version", {}).get("number", expected_version)
            time.sleep(0.5 * (2 ** attempt))
    return False, None

def _candidate_from_metadata(meta: dict) -> CandidatePage:
    return CandidatePage(
        page_id=meta.get("page_id", ""),
        title=meta.get("title", "Unknown"),
        heading=meta.get("heading", "") or None,
        is_root_section=meta.get("is_root_section", False),
        space_key=meta.get("space_key", ""),
        snippet=(meta.get("text_summary") or meta.get("excerpt") or meta.get("title") or "")[:140],
    )


def _run_async_blocking(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = {}

    def _runner():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _graph_candidates(query: str) -> List[CandidatePage]:
    user_id = confluence_page_graph.get_current_graph_user_id()
    if not user_id:
        return []
    try:
        built = _run_async_blocking(confluence_page_graph.ensure_user_confluence_graph(user_id))
        if not built:
            return []
        matches = _run_async_blocking(confluence_page_graph.query_user_confluence_graph(user_id, query, limit=8))
        return [
            CandidatePage(
                page_id=match.get("page_id") or "",
                title=match.get("title") or "Unknown",
                heading=match.get("heading") or None,
                is_root_section=(match.get("heading") or "") == "Root",
                space_key=match.get("space_key") or "",
                snippet=(match.get("relevant_content") or match.get("title") or "")[:140],
            )
            for match in matches
            if match.get("page_id")
        ]
    except Exception as exc:
        logger.warning("Confluence graph search unavailable, falling back to live/vector search: %s", exc)
        return []

def format_page_titles_for_user(candidates: List[CandidatePage]) -> str:
    """Renders user-facing page titles without leaking internal metadata by default."""
    if not candidates:
        return "I couldn't find any pages."

    title_counts: dict[str, int] = {}
    for candidate in candidates:
        title_counts[candidate.title] = title_counts.get(candidate.title, 0) + 1

    rendered_titles: List[str] = []
    for candidate in candidates:
        title = candidate.title
        if title_counts[title] > 1 and candidate.heading:
            rendered_titles.append(f"{title} - {candidate.heading}")
        else:
            rendered_titles.append(title)

    deduped: List[str] = []
    seen = set()
    for title in rendered_titles:
        if title not in seen:
            deduped.append(title)
            seen.add(title)

    return ", ".join(deduped)

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
    """Searches live Confluence pages and Pinecone in parallel, then merges results."""
    try:
        graph_results = _graph_candidates(query)
        connector = get_connector()
        store = get_store()

        def _live_search():
            try:
                return connector.search_pages(query, limit=8)
            except Exception as e:
                logger.error(f"Live Confluence search failed: {e}")
                return []

        def _recent_pages():
            try:
                return connector.list_pages(limit=100)
            except Exception as e:
                logger.warning(f"Recent Confluence page listing failed: {e}")
                return []

        def _pinecone_search():
            try:
                return store.search(query, top_k=5)
            except Exception as e:
                logger.warning(f"Pinecone search unavailable, continuing with live Confluence results: {e}")
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f_live = executor.submit(_live_search)
            f_recent = executor.submit(_recent_pages)
            f_pinecone = executor.submit(_pinecone_search)
            live_results = f_live.result()
            recent_pages = f_recent.result()
            pinecone_results = f_pinecone.result()

        candidates_by_page: dict[str, CandidatePage] = {}
        scored_candidates: dict[str, float] = {}

        for candidate in graph_results:
            candidates_by_page[candidate.page_id] = candidate
            scored_candidates[candidate.page_id] = 2.0

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
        connector = get_connector()
        html = connector.fetch_page_html(page_id)
        metadata = connector.get_page_metadata(page_id)
        version = metadata.get("version", {}).get("number", 1)

        # Cache HTML and version so preview_edit/preview_delete don't re-fetch
        state = get_tool_state()
        state.setdefault("html_cache", {})[page_id] = html
        state.setdefault("version_cache", {})[page_id] = version

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
        state = get_tool_state()
        live_html = state.get("html_cache", {}).get(page_id) or get_connector().fetch_page_html(page_id)
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
def preview_delete(
    page_id: str,
    heading_string: str,
    target_html_or_text: str = "",
    delete_entire_section: bool = False,
) -> PreviewResponse:
    """Generates a preview for a delete operation without changing the existing edit tool behavior."""
    try:
        state = get_tool_state()
        live_html = state.get("html_cache", {}).get(page_id) or get_connector().fetch_page_html(page_id)
        new_document_html = delete_content_in_section(
            live_html,
            heading_string,
            target_html_or_text=target_html_or_text,
            delete_entire_section=delete_entire_section,
        )

        diff_lines = list(difflib.unified_diff(
            live_html.splitlines(keepends=True),
            new_document_html.splitlines(keepends=True),
            fromfile='Current',
            tofile='PreviewDelete'
        ))

        diff_str = "".join(diff_lines)
        if not diff_str:
            return PreviewResponse(success=False, diff="", message="No visible delete changes detected.")

        return PreviewResponse(success=True, diff=diff_str, message="Delete preview generated.")
    except Exception as e:
        return PreviewResponse(success=False, diff="", message=f"Failed delete preview: {str(e)}")

@function_tool
def update_page_title(page_id: str, expected_version: int, new_title: str) -> CommitResponse:
    """Updates the title of an existing Confluence page without creating a new page."""
    try:
        _emit_mutation_started("update_title")
        success, committed_version = _commit_with_retry(
            page_id,
            apply_fn=lambda html: html,
            expected_version=expected_version,
            title_override=new_title,
        )

        if success:
            _reindex_in_background(page_id)
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
def commit_delete(
    page_id: str,
    expected_version: int,
    heading_string: str,
    target_html_or_text: str = "",
    delete_entire_section: bool = False,
) -> CommitResponse:
    """Commits a dedicated delete operation for sections or unique content blocks."""
    try:
        _emit_mutation_started("delete")
        success, committed_version = _commit_with_retry(
            page_id,
            apply_fn=lambda html: delete_content_in_section(
                html, heading_string,
                target_html_or_text=target_html_or_text,
                delete_entire_section=delete_entire_section,
            ),
            expected_version=expected_version,
        )

        if success:
            _reindex_in_background(page_id)
            message = f"Delete successful on page {page_id}"
            _tool_run_state.set({"last_action": "commit", "success": True, "message": message})
            return CommitResponse(success=True, version=committed_version, message=message)

        message = "Delete update failed."
        _tool_run_state.set({"last_action": "commit", "success": False, "message": message})
        return CommitResponse(success=False, version=None, message=message)
    except ValueError as ve:
        if _is_version_conflict(ve):
            logger.error(f"Version conflict max retries exceeded during delete for page {page_id}: {ve}")
            message = f"ConflictError: {str(ve)} Please re-fetch_live_page."
        else:
            logger.error(f"Delete target resolution failed: {ve}")
            message = str(ve)
        _tool_run_state.set({"last_action": "commit", "success": False, "message": message})
        return CommitResponse(success=False, version=None, message=message)
    except Exception as e:
        logger.error(f"Commit delete failed: {e}")
        message = str(e)
        _tool_run_state.set({"last_action": "commit", "success": False, "message": message})
        return CommitResponse(success=False, version=None, message=message)

@function_tool
def commit_document_edit(
    page_id: str,
    expected_version: int,
    heading_string: str,
    old_block_html: str = "",
    new_block_html: str = "",
    append: bool = False,
) -> CommitResponse:
    """Commits a parsed structural sub-section DOM replacement securely using Op-Locking bounds.

    When append=True, new_block_html is appended to the END of the existing section rather than
    replacing it. The apply_fn always reads the current section from live HTML before appending,
    so it is safe even when the page is modified by a concurrent commit in the same batch.
    Use append=True whenever you want to ADD content without removing anything.
    """
    try:
        new_html = markdown_to_html(new_block_html) if new_block_html else new_block_html
        _emit_mutation_started("edit")

        if append and heading_string:
            # Append mode: fetch the current section on every attempt so the anchor
            # never goes stale when other changes have already modified the same page.
            def _append_fn(live_html: str) -> str:
                current_section = get_section_html(live_html, heading_string)
                merged = (current_section.rstrip() + "\n" + new_html) if current_section.strip() else new_html
                return edit_block_in_section(live_html, heading_string, current_section, merged)
            apply_fn = _append_fn
        else:
            apply_fn = lambda html: edit_block_in_section(html, heading_string, old_block_html, new_html)

        success, committed_version = _commit_with_retry(
            page_id,
            apply_fn=apply_fn,
            expected_version=expected_version,
        )

        if success:
            _reindex_in_background(page_id)
            message = f"Commit successful on page {page_id}"
            _tool_run_state.set({"last_action": "commit", "success": True, "message": message})
            return CommitResponse(success=True, version=committed_version, message=message)

        message = "Push update failed."
        _tool_run_state.set({"last_action": "commit", "success": False, "message": message})
        return CommitResponse(success=False, version=None, message=message)
    except ValueError as ve:
        if _is_version_conflict(ve):
            logger.error(f"Version conflict max retries exceeded for page {page_id}: {ve}")
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
def delete_confluence_page(page_id: str) -> CommitResponse:
    """Permanently deletes an entire Confluence page by its page_id.

    Use this when the whole page should be removed (not just a section).
    Do NOT use commit_delete for whole-page removal — that only empties content.
    """
    try:
        _emit_mutation_started("delete_page")
        success = get_connector().delete_page(page_id)
        if success:
            message = f"Page {page_id} permanently deleted."
            _tool_run_state.set({"last_action": "commit", "success": True, "message": message})
            return CommitResponse(success=True, version=None, message=message)
        message = f"Page {page_id} delete returned unexpected status."
        _tool_run_state.set({"last_action": "commit", "success": False, "message": message})
        return CommitResponse(success=False, version=None, message=message)
    except Exception as e:
        logger.error("delete_confluence_page failed for %s: %s", page_id, e)
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
        _emit_mutation_started("create")
        result = get_connector().create_page(
            space_key=space_key,
            title=title,
            content=html_payload,
            parent_page_id=parent_page_id
        )
        
        new_page_id = result.get('id')
        if new_page_id:
            _reindex_in_background(new_page_id)

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
