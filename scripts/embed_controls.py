# scripts/embed_controls.py

import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer

# Load controls CSV
csv_path = "data/controls/controls.csv"
df = pd.read_csv(csv_path)

# Check for the correct column
if 'control_statement' not in df.columns:
    raise ValueError("Expected column 'control_statement' in CSV")

# Get control texts
control_texts = df['control_statement'].dropna().tolist()

# Embed controls using SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
control_embeddings = model.encode(control_texts)

# Save to pickle
output_path = "data/control_embeddings.pkl"
os.makedirs("data", exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(control_embeddings, f)

print(f"✅ Successfully embedded {len(control_texts)} controls into: {output_path}")
