from bs4 import BeautifulSoup, Tag
import re
from typing import List, Tuple, Optional

FULL_PAGE_SENTINELS = {"full_page", "full page", "__full_page__", "page"}

def _normalize(text: str) -> str:
    """Normalize whitespace and lowercase for reliable matching."""
    return re.sub(r'\s+', ' ', text.strip().lower())

def _get_heading_level(tag_name: str) -> int:
    if tag_name and tag_name.startswith('h') and len(tag_name) == 2 and tag_name[1].isdigit():
        return int(tag_name[1])
    return 99

def _is_full_page_mode(heading_string: Optional[str]) -> bool:
    return bool(heading_string and heading_string.strip().lower() in FULL_PAGE_SENTINELS)

def _looks_like_html(text: str) -> bool:
    return "<" in text and ">" in text

def _resolve_visible_text_target(section_html: str, target_text: str) -> str:
    """
    Resolve a plain-text target into a unique outer HTML block within the section.
    This lets the agent target visible text like "14" or "common mistakes" without
    having to guess the exact storage HTML wrapper.
    """
    if not target_text:
        return target_text

    section_soup = BeautifulSoup(section_html, "html.parser")
    normalized_target = _normalize(target_text)
    exact_matches = []
    partial_matches = []

    for tag in section_soup.find_all(True):
        tag_text = _normalize(tag.get_text(" ", strip=True))
        if not tag_text:
            continue
        if tag_text == normalized_target:
            exact_matches.append(tag)
        elif normalized_target in tag_text:
            partial_matches.append(tag)

    matches = exact_matches if exact_matches else partial_matches
    if not matches:
        raise ValueError("No unique visible-text match found for the requested replacement target.")
    if len(matches) > 1:
        raise ValueError(
            f"The visible text target '{target_text}' matches {len(matches)} blocks. "
            "Provide a more specific target."
        )

    return str(matches[0])

def find_bounded_section(html_content: str, heading_string: Optional[str]) -> Tuple[Optional[Tag], List[Tag], BeautifulSoup]:
    """
    Returns (heading_tag, list_of_content_tags, soup).
    If heading_string is 'Root' or empty, isolates strict nodes preceding the first formal heading.
    If heading_string is FULL_PAGE mode, targets the entire page body.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    container = soup.body if soup.body else soup

    if _is_full_page_mode(heading_string):
        return None, list(container.contents), soup
    
    if not heading_string or heading_string.lower() == "root":
        content_nodes = []
        first_heading = soup.find(re.compile('^h[1-6]$'))
        
        if first_heading:
            current = first_heading
            while current.parent and current.parent != container:
                current = current.parent
                
            for node in container.contents:
                if node == current:
                    break
                content_nodes.append(node)
        else:
            content_nodes = list(container.contents)
            
        return None, content_nodes, soup

    headings = soup.find_all(re.compile('^h[1-6]$'))
    matches = []
    normalized_target = _normalize(heading_string)
    
    for h in headings:
        if normalized_target in _normalize(h.get_text()):
            matches.append(h)
            
    if not matches:
        raise ValueError(f"Heading containing '{heading_string}' not found.")
    if len(matches) > 1:
        options = " | ".join([h.get_text() for h in matches])
        raise ValueError(f"Multiple headings match '{heading_string}'. Please disambiguate. Options: {options}")
        
    anchor = matches[0]
    anchor_level = _get_heading_level(anchor.name)
    
    content_nodes = []
    current_node = anchor.next_sibling
    
    while current_node:
        if isinstance(current_node, Tag) and _get_heading_level(current_node.name) <= anchor_level:
            break
        content_nodes.append(current_node)
        current_node = current_node.next_sibling
        
    return anchor, content_nodes, soup

def get_section_html(html_content: str, heading_string: Optional[str]) -> str:
    """Returns the raw HTML string bound by the specified heading anchor, root bucket, or full-page mode."""
    _, nodes, _ = find_bounded_section(html_content, heading_string)
    return "".join(str(n) for n in nodes)

def edit_block_in_section(html_content: str, heading_string: Optional[str], old_block_html: str, new_block_html: str) -> str:
    """
    Locates the heading bounds. Supports section edits, root edits, and full-page replacement.
    """
    anchor, nodes_to_remove, soup = find_bounded_section(html_content, heading_string)
    section_html = "".join(str(n) for n in nodes_to_remove)
    
    if old_block_html:
        resolved_old_block_html = old_block_html
        if not _looks_like_html(old_block_html):
            resolved_old_block_html = _resolve_visible_text_target(section_html, old_block_html)

        occurrences = section_html.count(resolved_old_block_html)
        if occurrences == 0:
            target_name = "FULL_PAGE" if _is_full_page_mode(heading_string) else ("heading" if heading_string else "Root")
            raise ValueError(f"The old_block_html was not found within the specified {target_name}'s bounded section.")
        if occurrences > 1:
            raise ValueError(f"The old_block_html appears {occurrences} times. Provide a more extensive unique old_block_html chunk string.")
        new_section_html = section_html.replace(resolved_old_block_html, new_block_html, 1)
    else:
        new_section_html = new_block_html
        
    for node in nodes_to_remove:
        node.extract()
        
    new_soup = BeautifulSoup(new_section_html, "html.parser")
    
    if anchor is None:
        container = soup.body if soup.body else soup
        for new_element in reversed(list(new_soup.contents)):
            container.insert(0, new_element)
    else:
        current_anchor = anchor
        for new_element in list(new_soup.contents):
            current_anchor.insert_after(new_element)
            current_anchor = new_element
        
    return str(soup)

def extract_headings(html_content: str) -> List[str]:
    soup = BeautifulSoup(html_content, "html.parser")
    headings = soup.find_all(re.compile('^h[1-6]$'))
    return [h.get_text().strip() for h in headings]
