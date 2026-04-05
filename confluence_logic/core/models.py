from pydantic import BaseModel
from typing import Optional

class DocumentChunk(BaseModel):
    id: str
    page_id: str
    space_key: str
    title: str
    heading: str
    is_root_section: bool
    section_order: int
    version: int
    text_summary: str
    markdown_content: str
