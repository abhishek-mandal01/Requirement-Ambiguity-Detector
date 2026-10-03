from flask import Flask, render_template, jsonify, request
from src.preprocessing.cleaner import clean_text, detect_spelling_issues
from src.preprocessing.normalizer import normalize_text, normalize_with_audit
from src.features.extractor import extract_features, classify_ambiguity
from src.features.similarity import calculate_similarity
from src.model.classifier import train_and_evaluate_texts, METHODOLOGY_NOTES
from src.model.evaluator import generate_bar_chart, generate_confusion_matrix_chart
from src.utils.file_loader import load_promise_dataset, load_pure_dataset, get_dataset_stats
from src.utils.srs_parser import parse_srs_requirements
import os

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

app = Flask(__name__)


def _pipeline_for_requirement(text: str) -> dict:
    """Run cleaning/normalization and return audit + features + prediction."""
    steps = []

    cleaned = clean_text(text)
    if cleaned != text.strip():
        steps.append("Formatting cleanup: removed HTML/XML tags and normalized whitespace.")
        formatting_changed = True
    else:
        steps.append("Formatting cleanup: no formatting noise found.")
        formatting_changed = False

    spelling_issues = detect_spelling_issues(cleaned)
    if spelling_issues:
        steps.append(
            "Spelling/noise detection: found "
            + ", ".join(f"'{a}' (suggests '{b}')" for a, b in spelling_issues)
            + "."
        )

    audit = normalize_with_audit(cleaned)
    # Avoid duplicating the case/spelling/term steps already produced by audit
    steps.extend(audit["steps"])

    normalized = audit["final"]
    features = extract_features(normalized)
    # Also surface indicators from original for mixed-language phrases altered by lowercasing only
    raw_features = extract_features(cleaned)
    if raw_features.get("mixed_language") and not features.get("mixed_language"):
        features["mixed_language"] = True
        features["mixed_language_signals"] = raw_features["mixed_language_signals"]
        for sig in raw_features["ambiguity_indicators"]:
            if sig not in features["ambiguity_indicators"]:
                features["ambiguity_indicators"].append(sig)

    prediction, explanation = classify_ambiguity(features)

    ops_count = 0
    if formatting_changed:
        ops_count += 1
    ops_count += len(audit["spelling_changes"])
    ops_count += len(audit["terminology_changes"])
    # Extra weight for whitespace-heavy originals
    if text != text.strip() or "  " in text or "\t" in text:
        ops_count += 1

    return {
        "cleaned": normalized,
        "features": features,
        "prediction": prediction,
        "explanation": explanation,
        "pipeline_steps": steps,
        "ops_count": ops_count,
        "spelling_issues": [{"from": a, "to": b} for a, b in spelling_issues],
    }


def _select_demo_requirement(requirements: list) -> dict:
    """Prefer a meaningful noisy requirement for the preprocessing demo."""
    candidates = [r for r in requirements if not r.get("is_exact_duplicate")]
    if not candidates:
        return None

    # Prefer REQ-012 when it has real spelling/noise corrections (recieve demo).
    for r in candidates:
        if r.get("id") == "REQ-012" and r.get("spelling_issues"):
            return r

    # Prefer any requirement with verified spelling corrections.
    spelling_hits = [r for r in candidates if r.get("spelling_issues")]
    if spelling_hits:
        return max(spelling_hits, key=lambda r: (r.get("ops_count", 0), len(r.get("spelling_issues", []))))

    # Prefer REQ-015 when present and it has preprocessing signal (whitespace/terminology).
    for r in candidates:
        if r.get("id") == "REQ-015" and r.get("ops_count", 0) > 0:
            return r

    return max(candidates, key=lambda r: (r.get("ops_count", 0), len(r.get("pipeline_steps", []))))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze/srs", methods=["POST"])
def analyze_srs():
    srs_text = ""

    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            ext = os.path.splitext(file.filename)[1].lower()
            try:
                if ext == ".txt":
                    srs_text = file.read().decode("utf-8")
                elif ext == ".pdf":
                    if PdfReader:
                        reader = PdfReader(file)
                        srs_text = "\n".join(
                            [page.extract_text() for page in reader.pages if page.extract_text()]
                        )
                    else:
                        return jsonify({"error": "PDF parsing library not installed"}), 400
                elif ext == ".docx":
                    if Document:
                        doc = Document(file)
                        srs_text = "\n".join([p.text for p in doc.paragraphs if p.text])
                    else:
                        return jsonify({"error": "DOCX parsing library not installed"}), 400
                else:
                    return jsonify({"error": f"Unsupported file type {ext}"}), 400
            except Exception as e:
                return jsonify({"error": f"Error parsing file: {str(e)}"}), 500

    elif request.is_json:
        srs_text = request.json.get("srs", "")
    elif "srs" in request.form:
        srs_text = request.form.get("srs", "")

    if not srs_text.strip():
        return jsonify({"error": "No SRS text or file provided"}), 400

    parsed = parse_srs_requirements(srs_text)
    seen_exact = {}
    requirements = []

    exact_dupes_count = 0
    potentially_ambiguous_count = 0
    clear_count = 0
    mixed_count = 0

    for item in parsed:
        req_id = item["id"]
        text = item["text"]

        if text in seen_exact:
            exact_dupes_count += 1
            requirements.append({
                "id": req_id,
                "original": text,
                "is_exact_duplicate": True,
                "duplicate_of": seen_exact[text],
                "duplicate_status": f"Exact Duplicate of {seen_exact[text]}",
                "prediction": "Skipped (Duplicate)",
                "features": {
                    "word_count": "-",
                    "vague_word_count": "-",
                    "vague_word_ratio": "-",
                    "passive_voice": "-",
                },
                "ops_count": 0,
            })
            continue

        seen_exact[text] = req_id
        result = _pipeline_for_requirement(text)
        features = result["features"]
        prediction = result["prediction"]

        if prediction == "Potentially Ambiguous":
            potentially_ambiguous_count += 1
        elif prediction == "Mixed/Non-English":
            mixed_count += 1
        else:
            clear_count += 1

        requirements.append({
            "id": req_id,
            "original": text,
            "cleaned": result["cleaned"],
            "is_exact_duplicate": False,
            "duplicate_status": "Unique",
            "ambiguity_indicators": features["ambiguity_indicators"],
            "features": {
                "word_count": features["word_count"],
                "vague_word_count": features["vague_word_count"],
                "vague_word_ratio": features["vague_word_ratio"],
                "passive_voice": features["passive_voice"],
                "mixed_language": features["mixed_language"],
            },
            "prediction": prediction,
            "explanation": result["explanation"],
            "pipeline_steps": result["pipeline_steps"],
            "ops_count": result["ops_count"],
            "spelling_issues": result["spelling_issues"],
        })

    similarities = []
    near_dupes_count = 0
    unique_reqs = [r for r in requirements if not r.get("is_exact_duplicate")]

    for i in range(len(unique_reqs)):
        for j in range(i + 1, len(unique_reqs)):
            req_a = unique_reqs[i]
            req_b = unique_reqs[j]
            score = calculate_similarity(req_a["original"], req_b["original"])

            if score >= 0.75:
                # Exact string matches are handled earlier; similarity pairs are never identical raw text.
                classification = "Near Duplicate"
                near_dupes_count += 1
                similarities.append({
                    "req_a_id": req_a["id"],
                    "req_b_id": req_b["id"],
                    "req_a_text": req_a["original"],
                    "req_b_text": req_b["original"],
                    "score": round(score, 3),
                    "classification": classification,
                })
                req_a["duplicate_status"] = "Has Near Duplicates"
                req_b["duplicate_status"] = "Has Near Duplicates"

    demo = _select_demo_requirement(requirements)

    return jsonify({
        "summary": {
            "total_detected": len(requirements),
            "successful": len(requirements),
            "exact_duplicates": exact_dupes_count,
            "near_duplicates": near_dupes_count,
            "potentially_ambiguous": potentially_ambiguous_count,
            "clear": clear_count,
            "mixed_or_non_english": mixed_count,
        },
        "requirements": requirements,
        "similarities": similarities,
        "demo_requirement_id": demo["id"] if demo else None,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json
    raw_text = data.get("requirement", "")

    if not raw_text:
        return jsonify({"error": "No requirement text provided"}), 400

    result = _pipeline_for_requirement(raw_text)
    features = result["features"]

    return jsonify({
        "original": raw_text,
        "cleaned": result["cleaned"],
        "ambiguity_indicators": features["ambiguity_indicators"],
        "features": {
            "word_count": features["word_count"],
            "vague_word_count": features["vague_word_count"],
            "vague_word_ratio": features["vague_word_ratio"],
            "passive_voice": features["passive_voice"],
            "mixed_language": features["mixed_language"],
        },
        "prediction": result["prediction"],
        "explanation": result["explanation"],
        "pipeline_steps": result["pipeline_steps"],
        "spelling_issues": result["spelling_issues"],
    })


@app.route("/api/dataset/overview", methods=["GET"])
def dataset_overview():
    stats = get_dataset_stats()
    return jsonify(stats)


@app.route("/api/demonstrate/preprocessing", methods=["POST"])
def demonstrate_preprocessing():
    data = request.json
    raw_text = data.get("requirement", "")

    if not raw_text:
        return jsonify({"error": "No requirement text provided"}), 400

    result = _pipeline_for_requirement(raw_text)
    return jsonify({
        "original": raw_text,
        "final": result["cleaned"],
        "steps": result["pipeline_steps"],
        "spelling_issues": result["spelling_issues"],
    })


@app.route("/api/demonstrate/similarity", methods=["POST"])
def demonstrate_similarity():
    data = request.json
    req_a = data.get("req_a", "")
    req_b = data.get("req_b", "")

    if not req_a or not req_b:
        return jsonify({"error": "Both req_a and req_b must be provided"}), 400

    score = calculate_similarity(req_a, req_b)

    if req_a.strip() == req_b.strip():
        classification = "Exact Duplicate"
    elif score >= 0.75:
        classification = "Near Duplicate"
    elif score >= 0.40:
        classification = "Related Topic"
    else:
        classification = "Distinct / Unrelated"

    return jsonify({
        "similarity_score": score,
        "classification": classification,
    })


@app.route("/api/model/evaluation", methods=["GET"])
def evaluate_model():
    promise_data = load_promise_dataset()
    pure_data = load_pure_dataset()

    texts = []
    if promise_data:
        for r in promise_data:
            if not r:
                continue
            if "RequirementText" in r:
                texts.append(str(r["RequirementText"]))
            else:
                texts.append(str(list(r.values())[0]))

    # Optionally enrich with a capped sample of PURE texts (avoid huge XML dumps)
    if pure_data and len(texts) < 40:
        for r in pure_data:
            if r and "RequirementText" in r:
                val = str(r["RequirementText"]).strip()
                # Keep short requirement-like snippets only
                if val and 20 <= len(val) <= 300 and "\n" not in val:
                    texts.append(val)
            if len(texts) >= 120:
                break

    if not texts:
        # Diverse fallback corpus (NOT duplicated) for a non-leaky demo when datasets are empty.
        texts = [
            "The system should lock the account after 5 consecutive unsuccessful login attempts.",
            "The system should respond quickly.",
            "The system should provide a reasonable response time.",
            "The system shall display the requested page within 2 seconds.",
            "Users must login with a password.",
            "A quick and easy UI is required.",
            "Data is stored in the database.",
            "The application will be developed seamlessly.",
            "The user logs out.",
            "Robust security measures must be implemented.",
            "Clicking the button saves the file.",
            "The system should be fast.",
            "The interface should be user-friendly and appropriate.",
            "The search feature should work efficiently.",
            "Borrowers shall receive reminders before due dates.",
            "The user ko apna password reset karna chahiye.",
            "The system shall support 100 concurrent users.",
            "Response times shall remain adequate under normal load.",
            "Book records must be stored in the database.",
            "The librarian can update inventory after each return.",
        ]

    metrics_a, cm_a, meta_a = train_and_evaluate_texts(texts, use_preprocessing=False)
    metrics_b, cm_b, meta_b = train_and_evaluate_texts(texts, use_preprocessing=True)

    bar_chart = generate_bar_chart(metrics_a, metrics_b)
    cm_b_chart = generate_confusion_matrix_chart(cm_b, "Mode B Confusion Matrix")

    return jsonify({
        "mode_a": metrics_a,
        "mode_b": metrics_b,
        "methodology": {
            "notes": METHODOLOGY_NOTES,
            "mode_a": meta_a,
            "mode_b": meta_b,
            "known_limitations": [
                "Labels are deterministic rule outputs, not independent gold labels.",
                "Do not interpret 100% scores (if they occur) as general ML performance.",
                "Duplicate rows are removed before train/test to avoid contamination.",
                "Features are TF-IDF over text; they are not the same numeric rule features used to create labels.",
                "The goal is to observe preprocessing effects, not to guarantee accuracy gains.",
            ],
        },
        "charts": {
            "bar_chart_base64": bar_chart,
            "confusion_matrix_base64": cm_b_chart,
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
