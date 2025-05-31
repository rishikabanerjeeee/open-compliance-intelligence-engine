from sentence_transformers import SentenceTransformer
import pandas as pd
from models import match_engine

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load controls from CSV
controls_df = pd.read_csv("data/controls/controls.csv")

control_texts = controls_df["control_statement"].fillna("").tolist()

# Embed all control texts
control_embeddings = model.encode(control_texts)

# Load regulation embeddings (including texts, tags, categories)
reg_embeds = match_engine.load_all_regulation_embeddings("data/embeddings")

# Perform matching
matches = match_engine.match_controls_to_regulations(control_embeddings, reg_embeds, top_n=3, threshold=0.5)

# Print results
for i, match_list in enumerate(matches):
    print(f"\n🔐 Control {i+1}: {control_texts[i]}")
    print("-" * 100)
    if not match_list:
        print("❌ No matches found.")
    for m in match_list:
        print(f"✅ Regulation: {m['regulation']}")
        print(f"   - Similarity: {m['similarity']:.2f}")
        print(f"   - Requirement: {m['requirement_text'][:100]}...")
        print(f"   - Tags: {m.get('tags', 'N/A')} | Category: {m.get('category_refined', 'N/A')}")
