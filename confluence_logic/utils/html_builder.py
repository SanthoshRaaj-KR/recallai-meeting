from typing import List, Dict, Optional, Union
from ..core.schemas import PageSectionInput

def build_page_html(
    title: str,
    sections: Optional[List[Union[Dict[str, str], PageSectionInput]]] = None,
    body_text: Optional[str] = None,
) -> str:
    """
    Constructs safe Confluence Storage Format HTML.
    sections: list of dicts with 'heading' and 'content' OR simply 'content' (e.g. paragraphs).
    body_text: basic string inserted at root.
    """
    html_out = ""
    
    if body_text:
        html_out += f"<p>{body_text.strip()}</p>"
        
    if sections:
        for sec in sections:
            section = sec.model_dump() if isinstance(sec, PageSectionInput) else sec

            if "heading" in section and section["heading"]:
                html_out += f"<h2>{section['heading'].strip()}</h2>"
            
            content = section.get("content", "").strip()
            
            # Simple heuristic turning lines starting with "-" into unordered lists
            lines = content.split("\n")
            in_list = False
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("- "):
                    if not in_list:
                        html_out += "<ul>"
                        in_list = True
                    html_out += f"<li>{line[1:].strip()}</li>"
                else:
                    if in_list:
                        html_out += "</ul>"
                        in_list = False
                    html_out += f"<p>{line}</p>"
            if in_list:
                html_out += "</ul>"

    return html_out
