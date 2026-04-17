from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class ArtifactFetcher(ABC):
    @abstractmethod
    def list_artifacts(self, limit: int = 25) -> List[Dict[str, Any]]:
        """List recent sandboxed office artifacts."""

    @abstractmethod
    def search_artifacts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search sandboxed office artifacts by filename/title/content hints."""

    @abstractmethod
    def get_artifact_metadata(self, artifact_id: str) -> Dict[str, Any]:
        """Return metadata for one artifact."""

    @abstractmethod
    def fetch_artifact_html(self, artifact_id: str) -> Dict[str, Any]:
        """Return the normalized HTML working view plus metadata."""


class ArtifactPusher(ABC):
    @abstractmethod
    def push_artifact_update(
        self,
        artifact_id: str,
        content_html: str,
        expected_version_token: str,
    ) -> Dict[str, Any]:
        """Persist a normalized HTML update back to the native office file."""

    @abstractmethod
    def rename_artifact(
        self,
        artifact_id: str,
        expected_version_token: str,
        new_title: str,
    ) -> Dict[str, Any]:
        """Rename the artifact and optionally update the visible document title."""

    @abstractmethod
    def create_artifact(
        self,
        title: str,
        artifact_family: str,
        file_format: str,
        content_html: str,
    ) -> Dict[str, Any]:
        """Create a new sandboxed office artifact."""

