import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Load models once
model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model)

def load_csv(path):
    """Load a CSV file as a pandas DataFrame."""
    return pd.read_csv(path)

def save_csv(df, path):
    """Save pandas DataFrame to a CSV file (overwrite)."""
    df.to_csv(path, index=False)

def expand_requirement(text):
    """Generate a simple plain-language paraphrase."""
    # You can replace this with a better NLP paraphrasing method later
    return f"This means that {text.lower()}"

def extract_tags(text, top_n=3):
    """Use KeyBERT to extract top keywords as tags."""
    try:
        keywords = kw_model.extract_keywords(text, top_n=top_n)
        return ", ".join([kw[0] for kw in keywords])
    except Exception:
        return ""

def refine_category(original_category):
    """Normalize and improve category labels."""
    mapping = {
        "data access": "Access Rights",
        "consent": "User Consent",
        "security": "Data Security",
        "breach": "Breach Notification",
    }
    key = original_category.lower().strip()
    return mapping.get(key, original_category.title())

def annotate_row(requirement_text, category):
    """Generate expanded_text, tags, and refined category."""
    expanded = expand_requirement(requirement_text)
    tags = extract_tags(requirement_text)
    refined_category = refine_category(category)
    return expanded, tags, refined_category
