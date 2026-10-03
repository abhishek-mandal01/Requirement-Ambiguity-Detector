from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing.cleaner import clean_text
from src.preprocessing.normalizer import normalize_text

def calculate_similarity(req_a: str, req_b: str) -> float:
    """
    T027: Calculates the TF-IDF cosine similarity between two requirements.
    """
    if not req_a or not req_b:
        return 0.0
        
    proc_a = normalize_text(clean_text(req_a))
    proc_b = normalize_text(clean_text(req_b))
    
    if not proc_a or not proc_b:
        return 0.0
        
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([proc_a, proc_b])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(sim), 4)
    except ValueError:
        return 0.0
