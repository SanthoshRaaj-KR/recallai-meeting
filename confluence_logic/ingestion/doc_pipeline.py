import os
import tempfile
import re
from docling.document_converter import DocumentConverter
from ..connectors.confluence import ConfluenceConnector
from ..db.vector_store import PineconeStore
from ..core.models import DocumentChunk
import logging

logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(self):
        self.confluence = ConfluenceConnector()
        self.vector_store = PineconeStore()
        self.converter = DocumentConverter()

    def process_page(self, page_id: str):
        metadata = self.confluence.get_page_metadata(page_id)
        space_key = metadata.get("space", {}).get("key", "")
        version = metadata.get("version", {}).get("number", 1)
        title = metadata.get("title", f"Page {page_id}")
        
        cached_version = self.vector_store.get_page_version(page_id)
        if cached_version and cached_version == version:
            logger.info(f"Skipping embed logic: Page {page_id} matches cached Pinecone version ({version}).")
            return
            
        html_content = self.confluence.fetch_page_html(page_id)
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp.write(html_content.encode("utf-8"))
            tmp_path = tmp.name
            
        try:
            doc_result = self.converter.convert(tmp_path)
            doc = doc_result.document
            markdown_content = doc.export_to_markdown()
            
            sections = re.split(r'\n(?=#+ )', markdown_content)
            if not sections:
                sections = [markdown_content]

            chunks_to_upsert = []
            
            for index, sec in enumerate(sections):
                lines = sec.strip().split('\n')
                heading_match = re.match(r'^#+ (.*)', lines[0]) if lines else None
                
                if index == 0 and not heading_match:
                    heading = "Page intro"
                    is_root = True
                elif heading_match:
                    heading = heading_match.group(1).strip()
                    is_root = False
                else:
                    heading = "Page intro"
                    is_root = True
                
                excerpt = sec[:500].strip()
                summary = f"Title: {title}\nHeading: {heading}\nExcerpt: {excerpt}"
                
                chunk = DocumentChunk(
                    id=f"{page_id}_{index}",
                    page_id=page_id,
                    space_key=space_key,
                    title=title,
                    heading=heading,
                    is_root_section=is_root,
                    section_order=index,
                    version=version,
                    text_summary=summary,
                    markdown_content=sec
                )
                chunks_to_upsert.append(chunk)
                
            self.vector_store.upsert_chunks(chunks_to_upsert)
            self.vector_store.clear_stale_sections(page_id, len(chunks_to_upsert))
            logger.info(f"✅ Upserted {len(chunks_to_upsert)} section chunks and cleared stales for page {page_id}.")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
