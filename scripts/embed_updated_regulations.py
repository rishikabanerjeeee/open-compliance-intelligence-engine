import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

REG_DIR = "data/regulations"
EMBED_DIR = "data/embeddings"

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_and_save_embeddings():
    os.makedirs(EMBED_DIR, exist_ok=True)

    for filename in os.listdir(REG_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(REG_DIR, filename)
            df = pd.read_csv(filepath)

            # Use expanded_text if available, else fallback to requirement_text
            if "expanded_text" in df.columns:
                texts = df["expanded_text"].fillna("").tolist()
            else:
                texts = df["requirement_text"].fillna("").tolist()

            # Generate embeddings
            embeddings = model.encode(texts, show_progress_bar=True)

            # Optional: get tags and category if they exist
            tags = df["tags"].fillna("N/A").tolist() if "tags" in df.columns else ["N/A"] * len(texts)
            categories = df["category_refined"].fillna("N/A").tolist() if "category_refined" in df.columns else ["N/A"] * len(texts)

            # Save all required fields in a dictionary
            data_to_save = {
                "embeddings": embeddings,
                "expanded_text": texts,
                "tags": tags,
                "category_refined": categories
            }

            # Save to .pkl
            regulator_name = os.path.splitext(filename)[0]
            pkl_path = os.path.join(EMBED_DIR, f"{regulator_name}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(data_to_save, f)

            print(f"✅ Saved enriched embeddings for {regulator_name} → {pkl_path}")

if __name__ == "__main__":
    embed_and_save_embeddings()
