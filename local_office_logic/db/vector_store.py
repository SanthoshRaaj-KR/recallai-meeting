import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pinecone import Pinecone


class PineconeStore:
    def __init__(self, index_name: str = "local-office-kb"):
        api_key = (os.getenv("PINECONE_API_KEY") or "").strip()
        self.enabled = (os.getenv("LOCAL_OFFICE_ENABLE_PINECONE") or "false").strip().lower() == "true"
        self.index_name = (os.getenv("LOCAL_OFFICE_PINECONE_INDEX_NAME") or index_name).strip()
        self.embedding_model = (os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
        configured_dims = (os.getenv("OPENAI_EMBEDDING_DIMENSIONS") or "").strip()
        self.embedding_dimensions = int(configured_dims) if configured_dims else None
        if self.enabled and api_key:
            client = Pinecone(api_key=api_key)
            self.index = client.Index(self.index_name)
        self.openai_client = OpenAI()

    def get_artifact_version_token(self, artifact_id: str) -> Optional[str]:
        if not hasattr(self, "index"):
            return None
        response = self.index.fetch(ids=[f"{artifact_id}::0"])
        vectors = response.get("vectors") or {}
        first = vectors.get(f"{artifact_id}::0")
        if not first:
            return None
        return first.get("metadata", {}).get("version_token")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        request: Dict[str, Any] = {"input": texts, "model": self.embedding_model}
        if self.embedding_dimensions and self.embedding_model.startswith("text-embedding-3"):
            request["dimensions"] = self.embedding_dimensions
        response = self.openai_client.embeddings.create(**request)
        return [item.embedding for item in response.data]

    def clear_stale_sections(self, artifact_id: str, new_section_count: int) -> None:
        if not hasattr(self, "index"):
            return
        stale_ids = [f"{artifact_id}::{i}" for i in range(new_section_count, new_section_count + 100)]
        self.index.delete(ids=stale_ids)

    def upsert_chunks(self, chunks: List[Any]) -> None:
        if not chunks or not hasattr(self, "index"):
            return
        embeddings = self.get_embeddings([chunk.text_summary for chunk in chunks])
        vectors = []
        for index, chunk in enumerate(chunks):
            vectors.append(
                {
                    "id": chunk.id,
                    "values": embeddings[index],
                    "metadata": {
                        "artifact_id": chunk.artifact_id,
                        "relative_path": chunk.relative_path,
                        "title": chunk.title,
                        "artifact_family": chunk.artifact_family,
                        "file_format": chunk.file_format,
                        "section_label": chunk.section_label,
                        "section_order": chunk.section_order,
                        "version_token": chunk.version_token,
                        "markdown_content": chunk.markdown_content,
                        "text_summary": chunk.text_summary,
                    },
                }
            )
        self.index.upsert(vectors=vectors)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not hasattr(self, "index"):
            return []
        embedding = self.get_embeddings([query])[0]
        results = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return results.get("matches", [])
