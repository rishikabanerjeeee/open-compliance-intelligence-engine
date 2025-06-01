from sentence_transformers import SentenceTransformer
import pandas as pd
from models import match_engine

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load control statements from a CSV file
controls_df = pd.read_csv("data/controls/controls.csv")

# Get a list of control statements (fill NaN with empty string to avoid errors)
control_texts = controls_df["control_statement"].fillna("").tolist()

# Generate sentence embeddings for all control statements
control_embeddings = model.encode(control_texts)

# Load all pre-generated regulation embeddings
reg_embeds = match_engine.load_all_regulation_embeddings("data/embeddings")

# Perform control-to-regulation matching (corrected keyword: min_threshold)
matches = match_engine.match_controls_to_regulations(
    control_embeddings,
    reg_embeds,
    top_n=3,
    min_threshold=0.5  # ✅ Correct keyword
)

# Display matching results
for i, match_list in enumerate(matches):
    print(f"\n🔐 Control {i+1}: {control_texts[i]}")
    print("-" * 100)
    if not match_list:
        print("❌ No matches found.")
    for m in match_list:
        print(f"✅ Regulation: {m['regulation']}")
        print(f"   - Similarity: {m['similarity']:.2f}")
        print(f"   - Requirement: {m['requirement_text'][:100]}...")  # Show only first 100 chars
        print(f"   - Tags: {m.get('tags', 'N/A')} | Category: {m.get('category_refined', 'N/A')}")
