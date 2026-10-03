from src.preprocessing.cleaner import correct_spelling


def normalize_text(text: str, apply_spelling: bool = True) -> str:
    """
    Normalizes case, optional spelling noise, and terminology for requirements.
    """
    if not text:
        return ""

    # 1. Case normalization
    text = text.lower()

    # 2. Spelling / noise correction (explicit, auditable)
    if apply_spelling:
        text, _ = correct_spelling(text)

    # 3. Terminology normalization (lightweight dictionary)
    replacements = {
        "user": "user",
        "users": "user",
        "system": "system",
        "app": "application",
        "ui": "user interface",
    }

    words = text.split()
    normalized_words = [replacements.get(word, word) for word in words]

    return " ".join(normalized_words)


def normalize_with_audit(text: str) -> dict:
    """
    Run normalization and return an audit trail of real operations performed.
    """
    steps = []
    current = text or ""

    lowered = current.lower()
    if lowered != current:
        steps.append("Case normalization: converted text to lowercase.")
    else:
        steps.append("Case normalization: text was already lowercase.")
    current = lowered

    corrected, spelling_changes = correct_spelling(current)
    if spelling_changes:
        steps.append(
            "Spelling/noise correction: mapped "
            + ", ".join(dict.fromkeys(spelling_changes))
            + "."
        )
        current = corrected
    else:
        steps.append("Spelling/noise correction: no known spelling issues corrected.")

    replacements = {
        "user": "user",
        "users": "user",
        "system": "system",
        "app": "application",
        "ui": "user interface",
    }
    words = current.split()
    normalized_words = [replacements.get(word, word) for word in words]
    final = " ".join(normalized_words)

    term_changes = []
    for before, after in zip(words, normalized_words):
        if before != after:
            term_changes.append(f"'{before}' -> '{after}'")
    if term_changes:
        steps.append(
            "Terminology normalization: mapped "
            + ", ".join(dict.fromkeys(term_changes))
            + "."
        )
    else:
        steps.append("Terminology normalization: no terminology mapped.")

    return {
        "final": final,
        "steps": steps,
        "spelling_changes": spelling_changes,
        "terminology_changes": term_changes,
        "ops_count": sum(
            1
            for s in steps
            if not s.endswith("already lowercase.")
            and "no known spelling" not in s
            and "no terminology mapped" not in s
            and "no formatting noise" not in s
        ),
    }
