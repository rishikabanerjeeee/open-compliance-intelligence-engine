import pickle
import os
from models.match_engine import match_controls_to_regulations
import numpy as np

def load_all_regulation_embeddings(embeddings_dir):
    embeddings_dict = {}
    for file in os.listdir(embeddings_dir):
        if file.endswith('.pkl'):
            reg_name = file.replace('.pkl', '')
            with open(os.path.join(embeddings_dir, file), 'rb') as f:
                data = pickle.load(f)
                embeddings_dict[reg_name] = {
                    'embeddings': data.get('embeddings', np.array([])),
                    'texts': data.get('expanded_text', data.get('text', [])),
                    'tags': data.get('tags', []),
                    'categories': data.get('category_refined', []),
                }
    return embeddings_dict

# Load your control embeddings (adjust the path)
with open('data/control_embeddings.pkl', 'rb') as f:
    control_embeddings = pickle.load(f)

# Load regulation embeddings dictionary
regulations_embeddings_dict = load_all_regulation_embeddings('data/embeddings')

# Call matching with your chosen parameters
results = match_controls_to_regulations(control_embeddings, regulations_embeddings_dict, top_n=3, min_threshold=0.70)

# Print results with match levels
for i, control_matches in enumerate(results):
    print(f"\n🔐 Control {i + 1}:")
    print("-" * 40)
    if not control_matches:
        print("No matches found above threshold.")
        continue

    for match in control_matches:
        print(f"✅ Regulation: {match['regulation']}")
        print(f"   - Similarity: {match['similarity']:.2f}")
        print(f"   - Match Level: {match['match_level']}")
        print(f"   - Requirement: {match['requirement_text'][:100]}...")
        print(f"   - Tags: {match['tags']}")
        print(f"   - Category: {match['category_refined']}")
        print()
