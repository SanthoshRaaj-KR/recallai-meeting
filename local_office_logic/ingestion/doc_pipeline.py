import re
from typing import List

from bs4 import BeautifulSoup, Tag

from ..connectors.local_office import LocalOfficeConnector
from ..core.models import ArtifactChunk
from ..db.vector_store import PineconeStore


class IngestionPipeline:
    def __init__(self):
        self.connector = LocalOfficeConnector()
        self.vector_store = PineconeStore()

    def process_artifact(self, artifact_id: str) -> None:
        snapshot = self.connector.fetch_artifact_html(artifact_id)
        version_token = snapshot["version_token"]
        cached = None
        try:
            cached = self.vector_store.get_artifact_version_token(artifact_id)
        except Exception:
            cached = None
        if cached and cached == version_token:
            return

        sections = _split_html_sections(snapshot["full_html"])
        chunks: List[ArtifactChunk] = []
        for index, (label, html) in enumerate(sections):
            text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
            summary = (
                f"Title: {snapshot['title']}\n"
                f"Path: {snapshot['relative_path']}\n"
                f"Section: {label}\n"
                f"Excerpt: {text[:500]}"
            )
            chunks.append(
                ArtifactChunk(
                    id=f"{artifact_id}::{index}",
                    artifact_id=artifact_id,
                    relative_path=snapshot["relative_path"],
                    title=snapshot["title"],
                    artifact_family=snapshot["artifact_family"],
                    file_format=snapshot["file_format"],
                    section_label=label,
                    section_order=index,
                    version_token=version_token,
                    text_summary=summary,
                    markdown_content=text,
                )
            )

        self.vector_store.upsert_chunks(chunks)
        self.vector_store.clear_stale_sections(artifact_id, len(chunks))


def _split_html_sections(html: str) -> List[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    container = soup.body if soup.body else soup
    sections: List[tuple[str, str]] = []
    current_label = "Document intro"
    current_nodes: List[str] = []
    for child in container.children:
        if isinstance(child, Tag) and re.fullmatch(r"h[1-6]", child.name or ""):
            if current_nodes:
                sections.append((current_label, "".join(current_nodes)))
                current_nodes = []
            current_label = child.get_text(" ", strip=True) or current_label
            current_nodes.append(str(child))
        else:
            current_nodes.append(str(child))
    if current_nodes:
        sections.append((current_label, "".join(current_nodes)))
    return sections or [("Document intro", html)]
