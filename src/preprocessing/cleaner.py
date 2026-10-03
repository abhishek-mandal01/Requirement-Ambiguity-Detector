import re
from bs4 import BeautifulSoup

# Common requirement typos / OCR-style noise for the prototype demo.
SPELLING_CORRECTIONS = {
    "recieve": "receive",
    "recieves": "receives",
    "recieved": "received",
    "seperate": "separate",
    "seperately": "separately",
    "occured": "occurred",
    "accomodate": "accommodate",
    "enviroment": "environment",
    "responce": "response",
    "perfomance": "performance",
}


def clean_text(raw_text: str) -> str:
    """
    Removes HTML/XML formatting noise and fixes whitespace.
    """
    if not isinstance(raw_text, str):
        return ""

    # 1. Remove XML/HTML tags only when markup is actually present
    if "<" in raw_text and ">" in raw_text:
        text = BeautifulSoup(f"<div>{raw_text}</div>", "lxml").get_text(separator=" ")
    else:
        text = raw_text

    # 2. Fix multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)

    # 3. Strip leading/trailing whitespace
    return text.strip()


def detect_spelling_issues(text: str) -> list:
    """
    Return list of (original, correction) pairs for known noisy spellings.
    """
    if not text:
        return []

    issues = []
    for word in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text):
        lower = word.lower()
        if lower in SPELLING_CORRECTIONS:
            issues.append((word, SPELLING_CORRECTIONS[lower]))
    return issues


def correct_spelling(text: str) -> tuple:
    """
    Apply known spelling corrections.
    Returns (corrected_text, list_of_change_strings).
    """
    if not text:
        return "", []

    changes = []

    def repl(match):
        word = match.group(0)
        lower = word.lower()
        if lower not in SPELLING_CORRECTIONS:
            return word
        fixed = SPELLING_CORRECTIONS[lower]
        if word.isupper():
            fixed_out = fixed.upper()
        elif word[0].isupper():
            fixed_out = fixed.capitalize()
        else:
            fixed_out = fixed
        changes.append(f"'{word}' -> '{fixed_out}'")
        return fixed_out

    corrected = re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?", repl, text)
    return corrected, changes
