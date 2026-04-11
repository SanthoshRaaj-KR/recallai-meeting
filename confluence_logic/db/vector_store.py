import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

class PineconeStore:
    def __init__(self, index_name: str = "confluence-kb"):
        api_key = (os.getenv("PINECONE_API_KEY") or "").strip()
        self.index_name = (os.getenv("PINECONE_INDEX_NAME") or index_name).strip()
        self.embedding_model = (os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
        configured_dims = (os.getenv("OPENAI_EMBEDDING_DIMENSIONS") or os.getenv("PINECONE_INDEX_DIMENSION") or "").strip()
        self.embedding_dimensions = int(configured_dims) if configured_dims else None
        if apiKey := api_key:
            pc = Pinecone(api_key=apiKey)
            self.index = pc.Index(self.index_name)
        self.openai_client = OpenAI()

    def get_page_version(self, page_id: str) -> Optional[int]:
        """Return the cached version for page_id, or None if not indexed. Raises on Pinecone errors."""
        if not hasattr(self, 'index'):
            return None
        try:
            resp = self.index.fetch(ids=[f"{page_id}_0"])
            if resp and resp.get('vectors') and f"{page_id}_0" in resp['vectors']:
                return resp['vectors'][f"{page_id}_0"].get("metadata", {}).get("version")
            return None  # page not in index — not an error
        except Exception as exc:
            logger.error("Pinecone get_page_version failed for %s: %s", page_id, exc)
            raise  # propagate so callers can distinguish "unknown" from "not present"

    def clear_stale_sections(self, page_id: str, new_section_count: int):
        """Delete stale section vectors after a page update to keep the index clean."""
        if not hasattr(self, 'index'):
            return
        stale_ids = [f"{page_id}_{i}" for i in range(new_section_count, new_section_count + 100)]
        try:
            self.index.delete(ids=stale_ids)
        except Exception as exc:
            logger.error(
                "Pinecone stale section cleanup failed for %s (sections %d+): %s",
                page_id, new_section_count, exc,
            )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        request: Dict[str, Any] = {
            "input": texts,
            "model": self.embedding_model,
        }
        if self.embedding_dimensions and self.embedding_model.startswith("text-embedding-3"):
            request["dimensions"] = self.embedding_dimensions

        response = self.openai_client.embeddings.create(**request)
        return [data.embedding for data in response.data]

    def upsert_chunks(self, chunks: List[Any]):
        if not chunks or not hasattr(self, 'index'):
            return
        
        texts = [chunk.text_summary for chunk in chunks]
        embeddings = self.get_embeddings(texts)
        
        vectors = []
        for i, chunk in enumerate(chunks):
            vectors.append({
                "id": chunk.id,
                "values": embeddings[i],
                "metadata": {
                    "page_id": chunk.page_id,
                    "space_key": chunk.space_key,
                    "title": chunk.title,
                    "heading": chunk.heading,
                    "is_root_section": chunk.is_root_section,
                    "section_order": chunk.section_order,
                    "version": chunk.version,
                    "markdown_content": chunk.markdown_content,
                    "text_summary": chunk.text_summary
                }
            })
            
        try:
            self.index.upsert(vectors=vectors)
        except Exception as exc:
            logger.error(
                "Pinecone upsert failed for index '%s'. Check OPENAI_EMBEDDING_DIMENSIONS/PINECONE_INDEX_DIMENSION. Error: %s",
                self.index_name,
                exc,
            )
            raise

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not hasattr(self, 'index'):
            return []
        query_embedding = self.get_embeddings([query])[0]
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
        except Exception as exc:
            logger.error(
                "Pinecone query failed for index '%s'. Check OPENAI_EMBEDDING_DIMENSIONS/PINECONE_INDEX_DIMENSION. Error: %s",
                self.index_name,
                exc,
            )
            raise
        return results.get('matches', [])
