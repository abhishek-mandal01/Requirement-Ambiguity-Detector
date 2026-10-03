# Cleaning and Structuring Noisy Requirements Text for Ambiguity Detection

## 1. Introduction
This project is a prototype that applies data mining techniques to Software Requirement Specifications (SRS). Natural-language requirements are often noisy, vague, duplicated, or mixed-language. The prototype demonstrates preprocessing, linguistic feature extraction, near-duplicate detection, and a comparative ML evaluation of raw vs preprocessed text.

## 2. SRS Parsing
Uploaded or pasted SRS text is split into requirement statements rather than treating every non-empty line as a requirement.

- Explicit IDs such as `REQ-001:` are preserved exactly; the ID is not left inside the requirement body.
- Numbered section headings (e.g. `1. User Authentication`, `2. Book Search`) are ignored.
- When no explicit IDs are present, non-heading lines receive generated `REQ-NNN` IDs.

A sample 16-requirement SRS is available at `data/test_srs_16.txt`.

## 3. Preprocessing & Normalization
Before analysis, each requirement passes through an auditable cleaning pipeline:

- **Formatting cleanup**: BeautifulSoup removes HTML/XML markup when present; whitespace is normalized.
- **Spelling / noise correction**: Known typos (e.g. `recieve` → `receive`) are detected and corrected only when the mapping actually runs; the audit trail reports real operations.
- **Case normalization**: Text is lowercased.
- **Terminology normalization**: Lightweight synonym mapping (e.g. `ui` → `user interface`, `users` → `user`).

The UI preprocessing demo automatically selects a meaningfully noisy requirement (preferring spelling corrections such as REQ-012, otherwise high-change examples such as REQ-015).

## 4. Linguistic Feature Extraction
The extractor reports **ambiguity indicators** and a separate **classification**. Indicator detection is not the same as labeling a requirement Potentially Ambiguous.

**Indicator vocabulary (minimum)** includes problem-statement and test-SRS terms such as:
`should`, `appropriate`, `user-friendly`, `quickly`, `efficiently`, `reasonable`, `fast`, `easy`, `adequate`, and related forms (including weak context terms such as `normal`).

Additional signals:

- **Passive voice** via spaCy (`auxpass` dependency).
- **Mixed / non-English content** via a simple Hinglish / non-ASCII heuristic (e.g. *"The user ko apna password reset karna chahiye."*).

**Classification rules (high level):**

- Modal or weak indicators such as `should` or `normal` are reported, but do **not** alone force Potentially Ambiguous.
- Stronger unquantified quality terms (e.g. `quickly`, `reasonable`, `adequate`) or underspecified passive voice can yield Potentially Ambiguous.
- Measurable constraints (e.g. *within 2 seconds*, *after 5 attempts*) keep a requirement relatively Clear even when `should` appears.
- Mixed/non-English content is flagged as `Mixed/Non-English` rather than Clear.

## 5. TF-IDF Near-Duplicate Detection
After extraction, requirements are compared automatically with TF-IDF + cosine similarity (threshold ≥ 0.75).

- Identical raw strings are Exact Duplicates.
- High-similarity but non-identical pairs are labeled **Near Duplicate**.

## 6. Modeling & Evaluation
Mode A uses TF-IDF features over **raw** text; Mode B uses TF-IDF over **preprocessed** text. Both modes predict the same deterministic rule-based ambiguity labels derived from the original text. Metrics reported: Accuracy, Precision, Recall, F1, plus a confusion matrix.

Datasets:

- **PROMISE**: loaded from `data/promise/nfr.arff` (also accepts CSV if present).
- **PURE**: XML documents under `data/pure/`.

## 7. Evaluation Methodology Limitations
Ambiguity labels are bootstrapped from deterministic vocabulary and linguistic rules, not independent human annotation. Classifier scores are therefore a pipeline-consistency / preprocessing comparison, **not** evidence of general ML performance.

Earlier leakage (feeding the same rule features into both `X` and `y`, and duplicating rows across train/test) could produce artificial 100% scores; that path has been removed. Preprocessing is not guaranteed to improve accuracy—the experiment measures its effect.
