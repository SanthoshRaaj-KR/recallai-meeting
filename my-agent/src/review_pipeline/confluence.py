from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

from .models import PageCandidate
from .text_utils import extract_sections, html_to_text

logger = logging.getLogger(__name__)


class RestConfluenceClient:
    """Small Confluence REST client used for execution and as a Rovo fallback."""

    def __init__(self) -> None:
        self.email = (os.getenv("ATLASSIAN_USER_EMAIL") or "").strip()
        self.api_token = (os.getenv("ATLASSIAN_API_TOKEN") or "").strip()
        self.domain = (os.getenv("ATLASSIAN_DOMAIN") or "").strip()
        self.default_space_key = (os.getenv("ATLASSIAN_SPACE_KEY") or "").strip()
        if not all([self.email, self.api_token, self.domain]):
            raise ValueError("Missing ATLASSIAN_USER_EMAIL, ATLASSIAN_API_TOKEN, or ATLASSIAN_DOMAIN.")
        self.auth = HTTPBasicAuth(self.email, self.api_token)
        self.base_url = f"https://{self.domain}/wiki/rest/api"
        self.site_url = f"https://{self.domain}/wiki"

    def search_pages(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        safe = (query or "").replace('"', '\\"').strip()
        if not safe:
            return []
        cqls = [
            f'type = page and text ~ "{safe}" order by lastmodified desc',
            f'type = page and title ~ "{safe}" order by lastmodified desc',
        ]
        found: dict[str, dict[str, Any]] = {}
        for cql in cqls:
            try:
                resp = requests.get(
                    f"{self.base_url}/content/search",
                    auth=self.auth,
                    headers={"Accept": "application/json"},
                    params={"cql": cql, "limit": limit, "expand": "space,version,_links"},
                    timeout=15,
                )
                resp.raise_for_status()
            except requests.HTTPError as exc:
                logger.warning("Confluence REST search failed for %r: %s", query, exc)
                continue  # try next CQL variant
            for item in resp.json().get("results", []):
                page_id = item.get("id")
                if not page_id or page_id in found:
                    continue
                found[page_id] = {
                    "page_id": page_id,
                    "title": item.get("title", ""),
                    "space_key": (item.get("space") or {}).get("key", ""),
                    "version": (item.get("version") or {}).get("number"),
                    "url": self._web_url(item),
                }
                if len(found) >= limit:
                    return list(found.values())
        return list(found.values())

    def list_pages(self, limit: int = 25) -> list[dict[str, Any]]:
        resp = requests.get(
            f"{self.base_url}/content/search",
            auth=self.auth,
            headers={"Accept": "application/json"},
            params={
                "cql": "type = page order by lastmodified desc",
                "limit": max(1, min(limit, 100)),
                "expand": "space,version,_links",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return [
            {
                "page_id": item.get("id", ""),
                "title": item.get("title", ""),
                "space_key": (item.get("space") or {}).get("key", ""),
                "version": (item.get("version") or {}).get("number"),
                "url": self._web_url(item),
            }
            for item in resp.json().get("results", [])
        ]

    def fetch_page(self, page_id: str) -> PageCandidate:
        resp = requests.get(
            f"{self.base_url}/content/{page_id}",
            auth=self.auth,
            headers={"Accept": "application/json"},
            params={"expand": "body.storage,space,version,_links"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        page = PageCandidate(
            page_id=data.get("id") or page_id,
            title=data.get("title") or "",
            space_key=(data.get("space") or {}).get("key", ""),
            version=(data.get("version") or {}).get("number"),
            url=self._web_url(data),
            html=((data.get("body") or {}).get("storage") or {}).get("value", "") or "",
            source="rest",
        )
        page.text = html_to_text(page.html)
        page.sections = extract_sections(page.html)
        return page

    def update_page(
        self,
        page_id: str,
        html_content: str,
        *,
        title: str | None = None,
        expected_version: int | None = None,
    ) -> bool:
        meta = self.get_page_metadata(page_id)
        current_version = (meta.get("version") or {}).get("number", 1)
        if expected_version is not None and expected_version != current_version:
            raise ValueError(f"Version conflict: expected {expected_version}, live is {current_version}.")
        payload = {
            "id": page_id,
            "type": "page",
            "title": title or meta.get("title", ""),
            "space": {"key": (meta.get("space") or {}).get("key", self.default_space_key)},
            "body": {"storage": {"value": html_content, "representation": "storage"}},
            "version": {"number": current_version + 1},
        }
        resp = requests.put(
            f"{self.base_url}/content/{page_id}",
            auth=self.auth,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.status_code == 200

    def create_page(self, title: str, html_content: str, space_key: str | None = None) -> dict[str, Any]:
        resolved_space = (space_key or self.default_space_key).strip()
        if not resolved_space:
            raise ValueError("No Confluence space key provided. Set ATLASSIAN_SPACE_KEY.")
        payload = {
            "type": "page",
            "title": title,
            "space": {"key": resolved_space},
            "body": {"storage": {"value": html_content, "representation": "storage"}},
        }
        resp = requests.post(
            f"{self.base_url}/content",
            auth=self.auth,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def get_page_metadata(self, page_id: str) -> dict[str, Any]:
        resp = requests.get(
            f"{self.base_url}/content/{page_id}",
            auth=self.auth,
            headers={"Accept": "application/json"},
            params={"expand": "space,version,_links"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def _web_url(self, item: dict[str, Any]) -> str | None:
        links = item.get("_links") or {}
        webui = links.get("webui")
        if webui:
            return f"{self.site_url}{webui}"
        return None


class RovoMCPConfluenceClient:
    """Thin dynamic wrapper around Atlassian's Rovo MCP Confluence tools.

    It is intentionally optional. If auth or transport fails, the pipeline falls
    back to Confluence REST instead of blocking proposal generation.
    """

    def __init__(self) -> None:
        self.url = (
            os.getenv("ROVO_MCP_URL")
            or os.getenv("ATLASSIAN_MCP_URL")
            or "https://mcp.atlassian.com/v1/mcp/authv2"
        ).strip()
        self.token = (os.getenv("ROVO_MCP_BEARER_TOKEN") or os.getenv("ATLASSIAN_MCP_BEARER_TOKEN") or "").strip()
        self.enabled = (os.getenv("ROVO_MCP_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"})
        # Serialize all MCP calls — Atlassian's MCP server rejects concurrent SSE connections
        # from the same token, causing TaskGroup errors when searches run in parallel.
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        # Lazily created so it binds to the running event loop.
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(1)
        return self._semaphore

    async def search_pages(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        cql = f'type = page and text ~ "{query.replace(chr(34), " ")}" order by lastmodified desc'
        raw = await self._call_tool(
            ["searchConfluenceUsingCql", "search_confluence_using_cql", "searchConfluence"],
            [
                {"cql": cql, "limit": limit},
                {"query": query, "limit": limit},
            ],
        )
        return self._coerce_search_results(raw)[:limit]

    async def fetch_page(self, page_id: str) -> PageCandidate | None:
        if not self.enabled:
            return None
        raw = await self._call_tool(
            ["getConfluencePage", "get_confluence_page"],
            [
                {"pageId": page_id},
                {"page_id": page_id},
                {"id": page_id},
            ],
        )
        data = self._first_json(raw)
        if not data:
            text = self._text_from_tool_result(raw)
            if not text:
                return None
            page = PageCandidate(page_id=page_id, title=page_id, text=text, source="rovo")
            return page
        html_value = (
            data.get("html")
            or data.get("body")
            or (((data.get("body") or {}).get("storage") or {}).get("value") if isinstance(data.get("body"), dict) else "")
            or ""
        )
        page = PageCandidate(
            page_id=str(data.get("id") or data.get("page_id") or page_id),
            title=str(data.get("title") or data.get("name") or page_id),
            space_key=str(data.get("spaceKey") or data.get("space_key") or ""),
            url=data.get("url") or data.get("webUrl") or data.get("web_url"),
            html=html_value,
            text=html_to_text(html_value) if html_value else self._text_from_tool_result(raw),
            source="rovo",
        )
        if page.html:
            page.sections = extract_sections(page.html)
        return page

    async def _call_tool(self, names: list[str], arg_options: list[dict[str, Any]]) -> Any:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        async with self._get_semaphore():
            try:
                async with streamablehttp_client(self.url, headers=headers, timeout=20) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        available = {tool.name for tool in tools.tools}
                        tool_name = next((name for name in names if name in available), None)
                        if not tool_name:
                            logger.debug("Rovo MCP tool unavailable. wanted=%s available=%s", names, sorted(available))
                            return None
                        last_error: Exception | None = None
                        for args in arg_options:
                            try:
                                return await session.call_tool(tool_name, args)
                            except Exception as exc:  # noqa: BLE001
                                last_error = exc
                        if last_error:
                            raise last_error
            except (SystemExit, KeyboardInterrupt):
                raise
            except BaseException as exc:
                # anyio TaskGroup failures surface as ExceptionGroup (BaseException subclass).
                # Unwrap to get the real cause for logging, then re-raise as a plain Exception
                # so callers with `except Exception` can catch it.
                cause: BaseException = exc
                if hasattr(exc, "exceptions") and exc.exceptions:
                    cause = exc.exceptions[0]
                logger.warning("Rovo MCP transport error: %s: %s", type(cause).__name__, cause)
                raise RuntimeError(f"Rovo MCP: {cause}") from exc
        return None

    def _coerce_search_results(self, raw: Any) -> list[dict[str, Any]]:
        data = self._first_json(raw)
        if isinstance(data, dict):
            items = data.get("results") or data.get("pages") or data.get("items") or []
        elif isinstance(data, list):
            items = data
        else:
            items = []
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            page_id = item.get("id") or item.get("pageId") or item.get("page_id")
            title = item.get("title") or item.get("name") or ""
            if page_id or title:
                out.append(
                    {
                        "page_id": str(page_id or ""),
                        "title": str(title),
                        "space_key": str(item.get("spaceKey") or item.get("space_key") or ""),
                        "url": item.get("url") or item.get("webUrl") or item.get("web_url"),
                    }
                )
        return out

    def _first_json(self, raw: Any) -> Any:
        structured = self._structured_from_tool_result(raw)
        if structured is not None:
            return structured
        text = self._text_from_tool_result(raw)
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = next(re.finditer(r"[\[{]", text), None)
            if not match:
                return None
            try:
                return json.loads(text[match.start() :])
            except Exception:
                return None

    def _structured_from_tool_result(self, raw: Any) -> Any:
        if raw is None:
            return None
        for attr in ("structured_content", "structuredContent"):
            value = getattr(raw, attr, None)
            if value:
                return value
        dump_fn = getattr(raw, "model_dump", None)
        if callable(dump_fn):
            try:
                dumped = dump_fn()
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                return dumped.get("structuredContent") or dumped.get("structured_content")
        if isinstance(raw, dict):
            return raw.get("structuredContent") or raw.get("structured_content")
        return None

    def _text_from_tool_result(self, raw: Any) -> str:
        if raw is None:
            return ""
        parts = []
        for item in getattr(raw, "content", []) or []:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)


class HybridConfluenceClient:
    """Prefer Rovo MCP for discovery/fetch, then fall back to REST."""

    def __init__(self) -> None:
        self.rovo = RovoMCPConfluenceClient()
        self.rest = RestConfluenceClient()

    async def search_pages(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        try:
            rovo_results = await self.rovo.search_pages(query, limit)
            if rovo_results:
                return rovo_results
        except Exception as exc:  # noqa: BLE001
            logger.info("Rovo search failed; falling back to REST: %s", exc)
        return await asyncio.to_thread(self.rest.search_pages, query, limit)

    async def fetch_page(self, page_id: str) -> PageCandidate:
        try:
            page = await self.rovo.fetch_page(page_id)
            if page and (page.text or page.html):
                return page
        except Exception as exc:  # noqa: BLE001
            logger.info("Rovo fetch failed; falling back to REST: %s", exc)
        return await asyncio.to_thread(self.rest.fetch_page, page_id)

    async def list_pages(self, limit: int = 25) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.rest.list_pages, limit)

    async def update_page(self, page_id: str, html_content: str, *, title: str | None = None, expected_version: int | None = None) -> bool:
        return await asyncio.to_thread(
            self.rest.update_page,
            page_id,
            html_content,
            title=title,
            expected_version=expected_version,
        )

    async def create_page(self, title: str, html_content: str, space_key: str | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.rest.create_page, title, html_content, space_key)