from pydantic import BaseModel


class ArtifactChunk(BaseModel):
    id: str
    artifact_id: str
    relative_path: str
    title: str
    artifact_family: str
    file_format: str
    section_label: str
    section_order: int
    version_token: str
    text_summary: str
    markdown_content: str

