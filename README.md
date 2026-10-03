# Requirement Ambiguity Detector

A prototype for **Cleaning and Structuring Noisy Requirements Text for Ambiguity Detection**.

## Project Overview

The app accepts a full SRS (paste or file), extracts real requirements (not section headings), cleans noisy text, reports ambiguity indicators separately from Potentially Ambiguous classification, detects near-duplicates with TF-IDF, and compares Logistic Regression performance on raw vs preprocessed text using the PROMISE and PURE datasets.

## Setup & Installation

1. **Install dependencies** (Python 3.8+):
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the spaCy English model** (passive-voice detection):
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Datasets** (already included for local use):
   - PROMISE NFR data: `data/promise/nfr.arff` (CSV also supported if you add `promise.csv`)
   - PURE XML docs: `data/pure/`
   - Sample 16-requirement SRS: `data/test_srs_16.txt`

## Running the Application

```bash
python app.py
```

Open `http://127.0.0.1:5000/`.

## Features

- **SRS Analysis**: Parses `REQ-XXX:` IDs, skips headings such as `1. User Authentication`, and analyzes each requirement.
- **Indicators vs Classification**: Shows detected ambiguity indicators (e.g. `should`, `normal`, `quickly`) separately from Clear / Potentially Ambiguous / Mixed/Non-English.
- **Preprocessing Demo**: Audits formatting cleanup, spelling correction (`recieve` → `receive`), case, and terminology normalization on a selected noisy requirement.
- **Near-Duplicate Check**: Automatic pairwise TF-IDF + cosine similarity; non-identical similar pairs are labeled Near Duplicate.
- **Mixed-language Heuristic**: Flags Hinglish / non-English content instead of treating it as Clear.
- **Model Evaluation**: Mode A (raw TF-IDF) vs Mode B (preprocessed TF-IDF) with Accuracy, Precision, Recall, F1, and methodology notes. Labels are rule-based; scores are not claimed as general ML performance.

## Quick Test

Paste the contents of `data/test_srs_16.txt` into the SRS input and click **Analyze Requirements**. You should see 16 requirements, REQ-012 as the spelling-correction demo (`recieve` → `receive`), and ambiguous/clear/mixed labels consistent with the methodology above.

For more academic detail, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).
