from typing import Dict, List, Optional, Union

from ..core.schemas import ArtifactSectionInput


def build_artifact_html(
    title: str,
    artifact_family: str,
    sections: Optional[List[Union[Dict[str, str], ArtifactSectionInput]]] = None,
    body_text: Optional[str] = None,
) -> str:
    if artifact_family == "spreadsheet":
        return build_spreadsheet_html(title=title, sections=sections, body_text=body_text)
    return build_document_html(title=title, sections=sections, body_text=body_text)


def build_document_html(
    title: str,
    sections: Optional[List[Union[Dict[str, str], ArtifactSectionInput]]] = None,
    body_text: Optional[str] = None,
) -> str:
    html_out = ""
    if title:
        html_out += f"<h1>{title.strip()}</h1>"
    if body_text:
        html_out += _render_rich_text(body_text)
    if sections:
        for entry in sections:
            section = entry.model_dump() if isinstance(entry, ArtifactSectionInput) else entry
            label = (section.get("label") or "").strip()
            content = (section.get("content") or "").strip()
            if label:
                html_out += f"<h2>{label}</h2>"
            html_out += _render_rich_text(content)
    return html_out


def build_spreadsheet_html(
    title: str,
    sections: Optional[List[Union[Dict[str, str], ArtifactSectionInput]]] = None,
    body_text: Optional[str] = None,
) -> str:
    html_out = ""
    if sections:
        for index, entry in enumerate(sections, start=1):
            section = entry.model_dump() if isinstance(entry, ArtifactSectionInput) else entry
            label = (section.get("label") or f"Sheet{index}").strip()
            html_out += f"<h2>{label}</h2>"
            html_out += _render_table_like_text(section.get("content", ""))
    else:
        html_out += f"<h2>{title.strip() or 'Sheet1'}</h2>"
        html_out += _render_table_like_text(body_text or "")
    return html_out


def _render_rich_text(content: str) -> str:
    stripped = (content or "").strip()
    if not stripped:
        return ""
    if "<table" in stripped and "</table>" in stripped:
        return stripped

    html_out = ""
    lines = stripped.splitlines()
    in_list = False
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text.startswith("- "):
            if not in_list:
                html_out += "<ul>"
                in_list = True
            html_out += f"<li>{text[2:].strip()}</li>"
        else:
            if in_list:
                html_out += "</ul>"
                in_list = False
            html_out += f"<p>{text}</p>"
    if in_list:
        html_out += "</ul>"
    return html_out


def _render_table_like_text(content: str) -> str:
    rows = []
    for line in (content or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "|" in stripped:
            parts = [cell.strip() for cell in stripped.strip("|").split("|")]
        else:
            parts = [cell.strip() for cell in stripped.split(",")]
        rows.append(parts)

    if not rows:
        rows = [["Value"]]

    html_out = '<table class="officeTable"><tbody>'
    for row_index, row in enumerate(rows):
        tag = "th" if row_index == 0 else "td"
        html_out += "<tr>"
        for cell in row:
            html_out += f"<{tag}>{cell}</{tag}>"
        html_out += "</tr>"
    html_out += "</tbody></table>"
    return html_out
