"""Vendored Confluence proposal pipeline for Confluence (Pinecone-hybrid retrieval).

The proposal agents (intent extraction, evaluation, editing, verification,
structural classification) are copied byte-for-byte from the confluence branch's
``local_doc_change`` — the proven 6/1 logic. Only the retrieval layer is swapped
to Pinecone-native hybrid (dense + sparse integrated inference + rerank), so
nothing runs on a local vector store and the embeddings are Pinecone's own.
"""

from .chunker import chunk_file, chunk_folder, chunk_markdown_text
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
    "chunk_file",
    "chunk_folder",
    "chunk_markdown_text",
]
