import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from src.features.extractor import extract_features, classify_ambiguity
from src.preprocessing.cleaner import clean_text
from src.preprocessing.normalizer import normalize_text

METHODOLOGY_NOTES = (
    "Ambiguity labels in this prototype are produced by deterministic vocabulary/"
    "linguistic rules (vague terms, passive voice, mixed-language heuristics), "
    "not by independent human annotation. A classifier trained to predict those "
    "same rule-derived labels is therefore a consistency check of the pipeline, "
    "not evidence of general ML performance. Perfect or near-perfect scores are "
    "expected when features leak the labeling rule or when duplicated texts appear "
    "in both train and test. The experiment compares raw vs preprocessed text "
    "representations; preprocessing is not guaranteed to improve accuracy."
)


def rule_label(text: str) -> int:
    """
    Deterministic ground-truth label used for the academic demo.
    Labels are derived from linguistic rules on the given text.
    """
    features = extract_features(text)
    prediction, _ = classify_ambiguity(features)
    # Mixed-language is treated as a positive (problematic) class for demo metrics
    return 0 if prediction == "Clear" else 1


def prepare_evaluation_corpus(texts):
    """
    Deduplicate and drop empty strings so train/test are not contaminated
    by identical duplicated rows.
    """
    cleaned = []
    seen = set()
    for t in texts:
        s = str(t).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
    return cleaned


def build_tfidf_datasets(texts, use_preprocessing=True):
    """
    Build TF-IDF features from raw or preprocessed text.

    Labels are computed once from the ORIGINAL text using the deterministic
    rule labeler, so Mode A and Mode B share the same y (fair comparison of
    representation, not of re-labeled targets).
    """
    y = np.array([rule_label(t) for t in texts])

    processed = []
    for t in texts:
        if use_preprocessing:
            processed.append(normalize_text(clean_text(t)))
        else:
            processed.append(t)

    return processed, y


def train_and_evaluate_texts(texts, use_preprocessing=True, test_size=0.25, random_state=42):
    """
    Train Logistic Regression on TF-IDF vectors of (raw|preprocessed) text.
    Returns metrics, confusion matrix, and methodology metadata.
    """
    corpus = prepare_evaluation_corpus(texts)
    meta = {
        "n_input_texts": len(texts),
        "n_unique_texts": len(corpus),
        "duplicates_removed": max(0, len(texts) - len(corpus)),
        "label_source": "deterministic_rules_on_original_text",
        "feature_source": "tfidf_preprocessed_text" if use_preprocessing else "tfidf_raw_text",
        "methodology_notes": METHODOLOGY_NOTES,
    }

    if len(corpus) < 8:
        return (
            {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "n_train": 0,
                "n_test": 0,
                "note": "Insufficient unique samples for a meaningful train/test split.",
            },
            None,
            meta,
        )

    processed, y = build_tfidf_datasets(corpus, use_preprocessing=use_preprocessing)

    # Need both classes for meaningful precision/recall
    if len(set(y.tolist())) < 2:
        return (
            {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "n_train": 0,
                "n_test": 0,
                "note": "Only one label class present after rule labeling; metrics withheld.",
            },
            None,
            meta,
        )

    try:
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if min(np.bincount(y)) >= 2 else None,
        )
    except ValueError:
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed, y, test_size=test_size, random_state=random_state
        )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    try:
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
    except ValueError as e:
        return (
            {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "note": f"TF-IDF vectorization failed: {e}",
            },
            None,
            meta,
        )

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 2),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 2),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 2),
            "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 2),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "note": (
                "Scores reflect TF-IDF Logistic Regression predicting rule-based "
                "ambiguity labels. They must not be read as general ML accuracy."
            ),
        }
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        meta["n_train"] = metrics["n_train"]
        meta["n_test"] = metrics["n_test"]
        return metrics, cm, meta
    except Exception as e:
        print(f"Error training model: {e}")
        return (
            {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "note": f"Training failed: {e}",
            },
            None,
            meta,
        )


# Backward-compatible wrappers (older callers / docs)
def bootstrap_labels(texts, use_preprocessing=True):
    """
    Deprecated path: previously leaked rule features into both X and y.
    Kept for import compatibility; prefer train_and_evaluate_texts.
    """
    corpus = prepare_evaluation_corpus(texts)
    X = []
    y = []
    for text in corpus:
        if use_preprocessing:
            processed = normalize_text(clean_text(text))
        else:
            processed = text
        features = extract_features(processed)
        label = rule_label(text)
        feature_vector = [
            features["word_count"],
            features["vague_word_count"],
            features["vague_word_ratio"],
            1 if features["passive_voice"] else 0,
        ]
        X.append(feature_vector)
        y.append(label)
    return np.array(X), np.array(y)


def train_and_evaluate(X, y):
    """Legacy numeric-feature trainer (avoid for primary evaluation)."""
    if len(X) < 2:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 2),
            "precision": round(precision_score(y_test, preds, zero_division=0), 2),
            "recall": round(recall_score(y_test, preds, zero_division=0), 2),
            "f1_score": round(f1_score(y_test, preds, zero_division=0), 2),
            "n_test": int(len(y_test)),
            "note": METHODOLOGY_NOTES,
        }
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        return metrics, cm
    except Exception as e:
        print(f"Error training model: {e}")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}, None
