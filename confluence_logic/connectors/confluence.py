import os
import requests
from requests.auth import HTTPBasicAuth
from typing import Optional, Dict, Any, List
from ..core.interfaces import DocumentFetcher, DocumentPusher
import logging

logger = logging.getLogger(__name__)

class ConfluenceConnector(DocumentFetcher, DocumentPusher):
    def __init__(self):
        self.email = (os.getenv("ATLASSIAN_USER_EMAIL") or "").strip()
        self.api_token = (os.getenv("ATLASSIAN_API_TOKEN") or "").strip()
        self.domain = (os.getenv("ATLASSIAN_DOMAIN") or "").strip() # e.g., your-domain.atlassian.net
        self.default_space_key = (os.getenv("ATLASSIAN_SPACE_KEY") or "").strip()
        if not all([self.email, self.api_token, self.domain]):
            raise ValueError("Missing Atlassian credentials in environment variables.")

        self.auth = HTTPBasicAuth(self.email, self.api_token)
        self.base_url = f"https://{self.domain}/wiki/rest/api"

    def get_page_metadata(self, page_id: str) -> Dict[str, Any]:
        """Fetch metadata including current version of the given page."""
        url = f"{self.base_url}/content/{page_id}"
        response = requests.get(url, auth=self.auth, headers={"Accept": "application/json"}, timeout=15)
        response.raise_for_status()
        return response.json()

    def fetch_page_html(self, page_id: str) -> str:
        """Get the HTML/Storage format of the associated page."""
        url = f"{self.base_url}/content/{page_id}?expand=body.storage"
        response = requests.get(url, auth=self.auth, headers={"Accept": "application/json"}, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("body", {}).get("storage", {}).get("value", "")

    def search_pages(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search Confluence pages live using CQL so page discovery does not depend on Pinecone freshness."""
        safe_query = (query or "").replace('"', '\\"').strip()
        if not safe_query:
            return []

        cql_candidates = [
            f'type = page and title ~ "{safe_query}" order by lastmodified desc',
            f'type = page and text ~ "{safe_query}" order by lastmodified desc',
        ]

        collected: Dict[str, Dict[str, Any]] = {}
        url = f"{self.base_url}/content/search"

        for cql in cql_candidates:
            response = requests.get(
                url,
                auth=self.auth,
                headers={"Accept": "application/json"},
                params={"cql": cql, "limit": limit, "expand": "space,version"},
                timeout=15,
            )
            response.raise_for_status()

            for item in response.json().get("results", []):
                page_id = item.get("id")
                if not page_id or page_id in collected:
                    continue

                collected[page_id] = {
                    "page_id": page_id,
                    "title": item.get("title", ""),
                    "space_key": item.get("space", {}).get("key", ""),
                    "version": item.get("version", {}).get("number"),
                    "excerpt": item.get("excerpt", ""),
                }

                if len(collected) >= limit:
                    return list(collected.values())

        return list(collected.values())

    def list_pages(self, limit: int = 25) -> List[Dict[str, Any]]:
        """List recent Confluence pages to help vague enterprise requests resolve to likely targets."""
        url = f"{self.base_url}/content/search"
        results = []
        requested_limit = max(1, limit)
        start = 0
        page_size = min(requested_limit, 100)

        while len(results) < requested_limit:
            response = requests.get(
                url,
                auth=self.auth,
                headers={"Accept": "application/json"},
                params={
                    "cql": "type = page order by lastmodified desc",
                    "limit": min(page_size, requested_limit - len(results)),
                    "start": start,
                    "expand": "space,version",
                },
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            items = payload.get("results", [])
            if not items:
                break

            for item in items:
                results.append(
                    {
                        "page_id": item.get("id", ""),
                        "title": item.get("title", ""),
                        "space_key": item.get("space", {}).get("key", ""),
                        "version": item.get("version", {}).get("number"),
                        "excerpt": item.get("excerpt", ""),
                    }
                )
                if len(results) >= requested_limit:
                    break

            if len(items) < page_size:
                break
            start += len(items)

        return results

    def push_update(self, page_id: str, content: str, expected_version: Optional[int] = None, title_override: Optional[str] = None) -> bool:
        """Update a confluence page safely."""
        metadata = self.get_page_metadata(page_id)
        current_version = metadata.get("version", {}).get("number", 1)
        
        if expected_version is not None and current_version != expected_version:
            raise ValueError(f"Version Conflict: Expected version {expected_version}, but live version is {current_version}.")
            
        title = title_override if title_override is not None else metadata.get("title", "")
        
        payload = {
            "id": page_id,
            "type": "page",
            "title": title,
            "space": {"key": metadata.get("space", {}).get("key", "")},
            "body": {
                "storage": {
                    "value": content,
                    "representation": "storage"
                }
            },
            "version": {
                "number": current_version + 1
            }
        }
        
        url = f"{self.base_url}/content/{page_id}"
        response = requests.put(
            url,
            auth=self.auth,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
        return response.status_code == 200

    def create_page(self, space_key: Optional[str], title: str, content: str, parent_page_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new page under the given space (and optional parent)."""
        resolved_space_key = (space_key or self.default_space_key).strip()
        if not resolved_space_key:
            raise ValueError("No Confluence space key provided. Set ATLASSIAN_SPACE_KEY in .env or pass space_key explicitly.")

        payload = {
            "type": "page",
            "title": title,
            "space": {"key": resolved_space_key},
            "body": {
                "storage": {
                    "value": content,
                    "representation": "storage"
                }
            }
        }
        if parent_page_id:
            payload["ancestors"] = [{"id": parent_page_id}]
            
        url = f"{self.base_url}/content"
        response = requests.post(
            url,
            auth=self.auth,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            logger.error("Confluence create_page failed: status=%s body=%s", response.status_code, response.text)
            raise
        return response.json()
