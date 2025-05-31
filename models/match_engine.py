import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import csv

# --- Load embeddings functions ---

def load_embeddings(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_all_regulation_embeddings(embeddings_dir):
    embeddings_dict = {}
    for file in os.listdir(embeddings_dir):
        if file.endswith('.pkl'):
            reg_name = file.replace('.pkl', '')
            path = os.path.join(embeddings_dir, file)
            with open(path, 'rb') as f:
                data = pickle.load(f)
                print(f"Loading {file}: type={type(data)}")

                if isinstance(data, dict):
                    embeddings_dict[reg_name] = {
                        'embeddings': data.get('embeddings', np.array([])),
                        'texts': data.get('expanded_text', data.get('text', [])),
                        'tags': data.get('tags', []),
                        'categories': data.get('category_refined', []),
                    }
                else:
                    # If data is not dict, assume it's just embeddings array
                    embeddings_dict[reg_name] = {
                        'embeddings': data,
                        'texts': [],
                        'tags': [],
                        'categories': [],
                    }
    return embeddings_dict

# --- Matching function ---

def match_controls_to_regulations(control_embeddings, regulations_embeddings_dict, top_n=5, min_threshold=0.65):
    results = []

    for i, ctrl_emb in enumerate(control_embeddings):
        control_results = []

        for reg_name, reg_data in regulations_embeddings_dict.items():
            reg_embs = reg_data['embeddings']
            texts = reg_data['texts']
            tags = reg_data['tags']
            categories = reg_data['categories']

            if len(reg_embs) == 0:
                continue

            sims = cosine_similarity(ctrl_emb.reshape(1, -1), reg_embs)[0]
            top_indices = sims.argsort()[-top_n:][::-1]

            for idx in top_indices:
                score = sims[idx]
                if score < min_threshold:
                    continue

                if score >= 0.80:
                    level = "Strong"
                elif score >= 0.75:
                    level = "Possible"
                elif score >= 0.70:
                    level = "Weak"
                else:
                    level = "No Match"

                control_results.append({
                    'control_index': i,
                    'regulation': reg_name,
                    'requirement_index': idx,
                    'requirement_text': texts[idx] if idx < len(texts) else "N/A",
                    'similarity': float(score),
                    'match_level': level,
                    'tags': tags[idx] if idx < len(tags) else "N/A",
                    'category_refined': categories[idx] if idx < len(categories) else "N/A"
                })

        # Sort and keep top N matches per control
        control_results = sorted(control_results, key=lambda x: x['similarity'], reverse=True)[:top_n]
        results.append(control_results)

    return results

# --- Main script logic ---

# ... (rest of your code unchanged)

if __name__ == "__main__":

    import os
    print("Current working directory:", os.getcwd())

    # Paths - update as needed
    controls_emb_path = "data/control_embeddings.pkl"
    regulations_emb_dir = "data/embeddings"

    # Load control embeddings
    print("Loading control embeddings...")
    control_embeddings = load_embeddings(controls_emb_path)
    print(f"Loaded {len(control_embeddings)} control embeddings.")

    # Load regulations embeddings + metadata
    print("Loading regulations embeddings...")
    regulations_embeddings_dict = load_all_regulation_embeddings(regulations_emb_dir)
    print(f"Loaded embeddings for regulations: {list(regulations_embeddings_dict.keys())}")

    # Match controls to regulations
    print("Matching controls to regulations...")
    matches = match_controls_to_regulations(control_embeddings, regulations_embeddings_dict, top_n=5, min_threshold=0.65)

    # Print matches per control
    for i, control_matches in enumerate(matches):
        print(f"\n🔐 Control {i}:")
        print("-" * 40)
        if len(control_matches) == 0:
            print("No matches found above threshold.")
        else:
            for match in control_matches:
                print(f"✅ Regulation: {match['regulation']}")
                print(f"   - Similarity: {match['similarity']:.2f}")
                print(f"   - Match Level: {match['match_level']}")
                print(f"   - Requirement: {match['requirement_text']}")
                print(f"   - Tags: {match['tags']}")
                print(f"   - Category: {match['category_refined']}")
                print()

    # === Add summary and save results ===
    print("Preparing to save summary and results...")

    try:
        # 1. Total matched controls
        total_matched_controls = sum(1 for m in matches if len(m) > 0)
        print(f"\n✅ Total Controls Matched: {total_matched_controls} / {len(matches)}")

        # 2. Save matches to JSON
        os.makedirs("data", exist_ok=True)
        with open("data/match_results.json", "w") as f:
            json.dump(matches, f, indent=2)
        print("📁 Saved match results to data/match_results.json")

        # 3. Save matches to flattened CSV
        with open("data/match_results.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "control_index", "regulation", "requirement_index",
                "requirement_text", "similarity", "match_level",
                "tags", "category_refined"
            ])
            writer.writeheader()
            for control_matches in matches:
                for row in control_matches:
                    writer.writerow(row)
        print("📁 Saved match results to data/match_results.csv")

    except Exception as e:
        print(f"❌ Error saving results: {e}")
        # === Compute and print overall summary ===
        print("\n📊 Overall Match Summary:")

        summary_stats = {
            "Strong": 0,
            "Possible": 0,
            "Weak": 0,
            "No Match": 0
        }
        regulation_counts = {}

        for control_matches in matches:
            if len(control_matches) == 0:
                summary_stats["No Match"] += 1
            else:
                for match in control_matches:
                    level = match["match_level"]
                    summary_stats[level] += 1

                    reg = match["regulation"]
                    if reg not in regulation_counts:
                        regulation_counts[reg] = 0
                    regulation_counts[reg] += 1

        total_controls = len(matches)
        print(f"Total Controls: {total_controls}")
        print(f"Matched Controls: {total_matched_controls}")
        print(f"Unmatched Controls: {summary_stats['No Match']}")
        print("\nMatch Levels Distribution:")
        for level, count in summary_stats.items():
            print(f"  {level}: {count}")

        print("\nTop Regulations by Match Count:")
        sorted_regs = sorted(regulation_counts.items(), key=lambda x: x[1], reverse=True)
        for reg, count in sorted_regs:
            print(f"  {reg}: {count} matches")

