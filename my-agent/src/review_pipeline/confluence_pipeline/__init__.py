"""Vendored Confluence proposal pipeline (Pinecone-hybrid retrieval).

The proposal agents (intent extraction, evaluation, editing, verification,
structural classification) run against live Confluence pages indexed by
``reindex_live``. Only the retrieval layer is Pinecone-native hybrid
(dense + sparse integrated inference + rerank).
"""

from .chunker import chunk_markdown_text
from .models import ChunkRecord, ConfluenceIntent, ConfluenceProposal
from .pipeline import PipelineConfig, propose
from .retrieval import PineconeHybridIndex

__all__ = [
    "ChunkRecord",
    "ConfluenceIntent",
    "ConfluenceProposal",
    "PipelineConfig",
    "PineconeHybridIndex",
    "propose",
    "chunk_markdown_text",
]
