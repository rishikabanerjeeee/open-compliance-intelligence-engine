from models.sentence_encoder import embed_controls_csv
from models.match_engine import load_all_regulation_embeddings, match_controls_to_regulations

# Step 1: Embed control statements
control_embeddings, control_texts = embed_controls_csv('./data/controls/controls.csv')

# Step 2: Load regulation embeddings
reg_embeddings = load_all_regulation_embeddings('./data/embeddings')

# Step 3: Match controls to regulations
results = match_controls_to_regulations(control_embeddings, reg_embeddings, top_n=3, threshold=0.75)

# Step 4: Print results
for i, matches in enumerate(results):
    print(f"\n🔹 Control {i + 1}: {control_texts[i]}")
    if matches:
        for match in matches:
            print(f"   ✅ Matches {match['regulation']} requirement #{match['requirement_index']} with similarity {match['similarity']}")
    else:
        print("   ❌ No regulation matched above the threshold.")
