import re
import spacy

# Load spaCy model for English.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: en_core_web_sm not found. Run 'python -m spacy download en_core_web_sm'")
    nlp = None

# Ambiguity-indicator vocabulary (reported when present).
# Includes problem-statement terms and those used in the test SRS.
VAGUE_WORDS = {
    "quick", "quickly", "fast", "user-friendly", "userfriendly", "easy",
    "easily", "robust", "reliable", "efficient", "efficiently",
    "appropriate", "appropriately", "seamless", "flexibly", "flexible",
    "adequate", "adequately", "various", "multiple", "several",
    "reasonable", "reasonably", "simple", "simply", "better", "best",
    "sufficient", "sufficiently", "normal", "normally",
}

# Linguistic indicators to REPORT even when they alone do not decide ambiguity.
MODAL_INDICATORS = {
    "should", "could", "might", "may",
}

# Context/quality words kept in the vocabulary and reported as indicators,
# but they do not by themselves force "Potentially Ambiguous".
WEAK_CONTEXT_INDICATORS = {
    "normal", "normally",
}

# Common Hindi / Hinglish tokens used in mixed-language demo requirements.
HINGLISH_TOKENS = {
    "ko", "apna", "apka", "karna", "chahiye", "hai", "hain", "kya",
    "nahi", "nahin", "mat", "se", "mein", "me", "par", "aur", "ki",
    "ka", "ke", "ho", "hoga", "hogi", "karo", "kare", "krna", "password",
}

# Keep password out of hinglish - it's English. Fix that.
HINGLISH_TOKENS.discard("password")
HINGLISH_TOKENS.discard("me")  # too common in English ("with me")


def _tokenize(text: str) -> list:
    return re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?|[^\x00-\x7F]+", text.lower())


def detect_mixed_language(text: str) -> dict:
    """
    Simple heuristic for non-English / mixed-language requirement text.
    """
    if not text or not text.strip():
        return {"is_mixed_or_non_english": False, "signals": []}

    signals = []
    tokens = _tokenize(text)

    # Non-ASCII / Devanagari (or other scripts)
    if re.search(r"[^\x00-\x7F]", text):
        signals.append("non-ascii script")

    hinglish_hits = sorted({t for t in tokens if t in HINGLISH_TOKENS})
    if len(hinglish_hits) >= 2:
        signals.append("hinglish tokens: " + ", ".join(hinglish_hits))

    # High ratio of tokens that look non-English (no vowels atypical, etc.) is brittle;
    # rely mainly on hinglish lexicon + script checks for this prototype.
    return {
        "is_mixed_or_non_english": bool(signals),
        "signals": signals,
    }


def extract_features(text: str) -> dict:
    """
    Extracts linguistic features from requirement text.
    Reports ambiguity indicators (including 'should') while keeping
    classification dependent on the combination of features.
    """
    features = {
        "word_count": 0,
        "vague_word_count": 0,
        "vague_word_ratio": 0.0,
        "passive_voice": False,
        "ambiguity_indicators": [],
        "modal_indicators": [],
        "weak_context_indicators": [],
        "mixed_language": False,
        "mixed_language_signals": [],
        "has_measurable_constraint": False,
    }

    if not text:
        return features

    words = text.split()
    features["word_count"] = len(words)

    found_vague = []
    found_modals = []
    found_weak = []
    for word in words:
        clean_word = word.strip(".,!?\"';:()[]{}").lower()
        if clean_word in VAGUE_WORDS:
            found_vague.append(clean_word)
            if clean_word in WEAK_CONTEXT_INDICATORS:
                found_weak.append(clean_word)
        if clean_word in MODAL_INDICATORS:
            found_modals.append(clean_word)

    # Classification uses strong vague terms only; weak context words stay as indicators.
    strong_vague = [w for w in found_vague if w not in WEAK_CONTEXT_INDICATORS]
    features["vague_word_count"] = len(strong_vague)
    features["weak_context_indicators"] = list(dict.fromkeys(found_weak))
    if features["word_count"] > 0:
        features["vague_word_ratio"] = round(
            features["vague_word_count"] / features["word_count"], 3
        )

    # Measurable constraint hints (numbers + time/count units)
    if re.search(
        r'\b\d+(\.\d+)?\s*(second|seconds|ms|millisecond|milliseconds|'
        r'sec|secs|minute|minutes|hour|hours|%|percent|user|users|'
        r'attempt|attempts|day|days)\b',
        text,
        re.IGNORECASE,
    ) or re.search(r'\bafter\s+\d+\b', text, re.IGNORECASE):
        features["has_measurable_constraint"] = True

    if nlp:
        doc = nlp(text)
        for token in doc:
            if token.dep_ == "auxpass":
                features["passive_voice"] = True
                found_vague.append(f"passive voice ({token.head.text})")
                break

    lang = detect_mixed_language(text)
    features["mixed_language"] = lang["is_mixed_or_non_english"]
    features["mixed_language_signals"] = lang["signals"]

    # Preserve order while deduplicating; report actual detected terms.
    indicators = []
    for item in found_modals + found_vague:
        if item not in indicators:
            indicators.append(item)
    if features["mixed_language"]:
        for sig in lang["signals"]:
            if sig not in indicators:
                indicators.append(sig)

    features["ambiguity_indicators"] = indicators
    features["modal_indicators"] = list(dict.fromkeys(found_modals))
    return features


def classify_ambiguity(features: dict) -> tuple:
    """
    Decide Clear / Potentially Ambiguous / Mixed-language.

    Detected ambiguity indicators (should, normal, ...) are reported separately
    from the Potentially Ambiguous classification. Classification uses the
    combination of stronger vague terms / underspecified passive voice /
    mixed-language signals — not indicator presence alone.
    """
    if features.get("mixed_language"):
        return (
            "Mixed/Non-English",
            "Mixed or non-English content was detected; treat as noisy input "
            "rather than a clear English requirement.",
        )

    vague_count = features.get("vague_word_count", 0)
    passive = features.get("passive_voice", False)
    measurable = features.get("has_measurable_constraint", False)
    indicators = features.get("ambiguity_indicators", [])
    modals = set(features.get("modal_indicators", []))
    weak = set(features.get("weak_context_indicators", []))
    non_decisive = MODAL_INDICATORS | WEAK_CONTEXT_INDICATORS

    strong_content_indicators = [
        i for i in indicators
        if i not in non_decisive
        and not str(i).startswith("hinglish")
        and i != "non-ascii script"
        and not str(i).startswith("passive voice")
    ]

    def indicators_clause():
        if indicators:
            return "Ambiguity indicators detected: " + ", ".join(indicators) + "."
        return "No ambiguity indicators detected."

    # Precise + measurable behavior stays relatively clear even with soft indicators
    if measurable and vague_count == 0 and not passive:
        explanation = indicators_clause()
        if modals or weak:
            explanation += (
                " Indicator presence is not the same as Potentially Ambiguous: "
                "the behavior is precisely / measurably specified, so classification remains Clear."
            )
        return "Clear", explanation

    is_ambiguous = vague_count > 0 or (passive and not measurable)

    if is_ambiguous:
        parts = [indicators_clause()]
        if strong_content_indicators:
            parts.append(
                "Classification is Potentially Ambiguous because stronger vague/"
                "unquantified terms (or underspecified passive voice) are present."
            )
        elif vague_count > 0:
            parts.append(
                "Classification is Potentially Ambiguous because vague/unquantified "
                "quality terms leave acceptance criteria unclear."
            )
        if passive and not measurable:
            parts.append(
                "Passive voice without a measurable constraint leaves the actor/criteria underspecified."
            )
        return "Potentially Ambiguous", " ".join(parts)

    explanation = indicators_clause()
    if indicators:
        explanation += (
            " Indicator detection is separate from classification: these indicators "
            "alone do not make the requirement Potentially Ambiguous."
        )
    return "Clear", explanation
