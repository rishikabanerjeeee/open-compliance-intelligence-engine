import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

REG_DIR = "data/regulations"
EMBED_DIR = "data/embeddings"

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_and_save_embeddings():
    for filename in os.listdir(REG_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(REG_DIR, filename)
            df = pd.read_csv(filepath)

            # Use expanded_text column if exists, else fallback to requirement_text
            if "expanded_text" in df.columns:
                texts = df["expanded_text"].fillna("").tolist()
            else:
                texts = df["requirement_text"].fillna("").tolist()

            # Generate embeddings for each regulation text
            embeddings = model.encode(texts, show_progress_bar=True)

            # Determine regulator name from filename (strip .csv)
            regulator_name = os.path.splitext(filename)[0]

            # Save embeddings as a pickle file
            pkl_path = os.path.join(EMBED_DIR, f"{regulator_name}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(embeddings, f)

            print(f"Saved embeddings for {regulator_name} at {pkl_path}")

if __name__ == "__main__":
    embed_and_save_embeddings()
