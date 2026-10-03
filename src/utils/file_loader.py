import os
import pandas as pd
from bs4 import BeautifulSoup

def load_promise_dataset(filepath=None):
    """
    Loads the PROMISE NFR dataset from CSV or ARFF.
    """
    candidates = []
    if filepath:
        candidates.append(filepath)
    candidates.extend([
        'data/promise/promise.csv',
        'data/promise/nfr.csv',
        'data/promise/nfr.arff',
    ])

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            if path.lower().endswith('.arff'):
                return _load_arff_records(path)
            df = pd.read_csv(path)
            return df.to_dict('records')
        except Exception as e:
            print(f"Error loading PROMISE dataset from {path}: {e}")
    return []


def _load_arff_records(filepath):
    """Minimal ARFF reader for requirement text columns."""
    import csv

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    in_data = False
    attributes = []
    rows = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('%'):
            continue
        lower = line.lower()
        if lower.startswith('@attribute'):
            # @attribute name type...
            rest = line.split(None, 1)[1]
            name = rest.split(None, 1)[0].strip("'\"")
            attributes.append(name)
            continue
        if lower.startswith('@data'):
            in_data = True
            continue
        if not in_data:
            continue
        try:
            values = next(csv.reader([line], delimiter=',', quotechar="'"))
        except Exception:
            values = [v.strip().strip("'\"") for v in line.split(',')]
        if not values:
            continue
        record = {}
        for i, attr in enumerate(attributes):
            record[attr] = values[i].strip() if i < len(values) else ''
        text = (
            record.get('RequirementText')
            or record.get('requirement')
            or record.get('text')
            or (values[1] if len(values) > 1 else (values[0] if values else ''))
        )
        record['RequirementText'] = text.strip()
        if record['RequirementText']:
            rows.append(record)
    return rows

def load_pure_dataset(directory='data/pure/'):
    """
    Loads XML documents from the PURE dataset directory.
    Returns a list of parsed dictionaries.
    """
    if not os.path.exists(directory):
        return []
    
    records = []
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f.read(), "lxml-xml")
                    
                    reqs = soup.find_all(["requirement", "req", "text"])
                    if not reqs:
                        records.append({"RequirementText": soup.get_text(separator=" ").strip()})
                    else:
                        for req in reqs:
                            records.append({"RequirementText": req.get_text(separator=" ").strip()})
            except Exception as e:
                print(f"Failed to parse {filename}: {e}")
                
    return records
    
def get_dataset_stats():
    """
    T020: Returns statistics about the datasets.
    """
    promise = load_promise_dataset()
    pure = load_pure_dataset()
    
    all_texts = []
    if promise:
        all_texts.extend([str(list(r.values())[0]) for r in promise if r])
    if pure:
        all_texts.extend([str(list(r.values())[0]) for r in pure if r])
        
    exact_duplicates = len(all_texts) - len(set(all_texts))
    
    return {
        "promise_records": len(promise),
        "pure_records": len(pure),
        "successful_parses": len(promise) + len(pure),
        "failed_parses": 0,
        "duplicates_detected": exact_duplicates,
        "valid_remaining": len(set(all_texts))
    }
