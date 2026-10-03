import re

# Explicit requirement identifiers, e.g. "REQ-001: The system shall..."
REQ_ID_PATTERN = re.compile(
    r'^(REQ-\d{3,})\s*[:.\-]\s*(.+)$',
    re.IGNORECASE,
)

# Numbered section headings such as "1. User Authentication" / "2. Book Search"
SECTION_HEADING_PATTERN = re.compile(
    r'^\d{1,2}\.\s+[A-Za-z][A-Za-z0-9\s\-&,/]{0,60}$'
)

HEADING_KEYWORDS = {
    "introduction", "overview", "scope", "purpose", "references",
    "definitions", "acronyms", "abbreviations", "system overview",
    "functional requirements", "non-functional requirements",
    "non functional requirements", "software requirements specification",
    "srs", "appendix", "table of contents", "revision history", "glossary",
}


def _is_section_heading(line: str) -> bool:
    """Return True when a line is a section/title heading, not a requirement."""
    text = line.strip()
    if not text:
        return True

    if REQ_ID_PATTERN.match(text):
        return False

    if SECTION_HEADING_PATTERN.match(text):
        # Numbered headings are short noun phrases, not full requirement sentences.
        # Reject if the line looks like a real requirement (contains a modal + object clause).
        if re.search(r'\b(shall|should|must|will)\b', text, re.IGNORECASE) and len(text.split()) > 6:
            return False
        return True

    lowered = text.lower().strip(":- ")
    if lowered in HEADING_KEYWORDS:
        return True

    return False


def parse_srs_requirements(srs_text: str) -> list:
    """
    Extract requirement statements from an SRS document.

    - Preserves explicit REQ-XXX IDs when present.
    - Ignores section headings such as "1. User Authentication".
    - Falls back to generated REQ-NNN IDs only when no explicit ID is given.
    """
    if not srs_text or not str(srs_text).strip():
        return []

    raw_lines = [line.strip() for line in str(srs_text).splitlines() if line.strip()]

    # Prefer explicit REQ-ID lines when the document uses that convention.
    explicit = []
    for line in raw_lines:
        match = REQ_ID_PATTERN.match(line)
        if match:
            req_id = match.group(1).upper()
            body = match.group(2).strip()
            if body:
                explicit.append({"id": req_id, "text": body, "raw_line": line})

    if explicit:
        return explicit

    # Fallback: treat each non-heading line as a requirement.
    requirements = []
    counter = 1
    for line in raw_lines:
        if _is_section_heading(line):
            continue
        requirements.append({
            "id": f"REQ-{counter:03d}",
            "text": line,
            "raw_line": line,
        })
        counter += 1

    return requirements
