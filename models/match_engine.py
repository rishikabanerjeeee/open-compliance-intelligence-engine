import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import csv
import pandas as pd  # Added missing import

# --- Load embeddings functions ---

def load_embeddings(path):
    """Load embeddings from a pickle file with error handling."""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, np.ndarray):
                return data
            elif isinstance(data, dict) and 'embeddings' in data:
                return data['embeddings']
            else:
                raise ValueError(f"Unexpected data format in {path}")
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return None

def load_all_regulation_embeddings(embeddings_dir):
    """Load all regulation embeddings from a directory."""
    embeddings_dict = {}
    if not os.path.exists(embeddings_dir):
        print(f"Directory not found: {embeddings_dir}")
        return embeddings_dict

    for file in os.listdir(embeddings_dir):
        if file.endswith('.pkl'):
            reg_name = file.replace('.pkl', '')
            path = os.path.join(embeddings_dir, file)
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    print(f"Loading {file}: type={type(data)}")

                    if isinstance(data, dict):
                        embeddings_dict[reg_name] = {
                            'embeddings': np.array(data.get('embeddings', [])),
                            'texts': data.get('expanded_text', data.get('text', [])),
                            'tags': data.get('tags', []),
                            'categories': data.get('category_refined', []),
                        }
                    else:
                        embeddings_dict[reg_name] = {
                            'embeddings': np.array(data),
                            'texts': [],
                            'tags': [],
                            'categories': [],
                        }
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue

    return embeddings_dict

# --- Matching function ---

def match_controls_to_regulations(control_embeddings, regulations_embeddings_dict, top_n=5, min_threshold=0.65):
    """Match controls to regulations based on cosine similarity."""
    results = []

    if not isinstance(control_embeddings, (list, np.ndarray)):
        print("Error: control_embeddings must be a list or numpy array")
        return results

    for i, ctrl_emb in enumerate(control_embeddings):
        if not isinstance(ctrl_emb, (list, np.ndarray)):
            print(f"Skipping control {i}: invalid embedding format")
            continue

        try:
            ctrl_emb = np.array(ctrl_emb).reshape(1, -1)
        except Exception as e:
            print(f"Skipping control {i}: could not reshape embedding - {str(e)}")
            continue

        control_results = []

        for reg_name, reg_data in regulations_embeddings_dict.items():
            reg_embs = reg_data['embeddings']
            texts = reg_data['texts']
            tags = reg_data['tags']
            categories = reg_data['categories']

            if len(reg_embs) == 0:
                continue

            try:
                sims = cosine_similarity(ctrl_emb, reg_embs)[0]
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
                        'control_index': int(i),
                        'regulation': reg_name,
                        'requirement_index': int(idx),
                        'requirement_text': texts[idx] if idx < len(texts) else "N/A",
                        'similarity': float(score),
                        'match_level': level,
                        'tags': tags[idx] if idx < len(tags) else "N/A",
                        'category_refined': categories[idx] if idx < len(categories) else "N/A"
                    })
            except Exception as e:
                print(f"Error processing {reg_name} for control {i}: {str(e)}")
                continue

        control_results = sorted(control_results, key=lambda x: x['similarity'], reverse=True)[:top_n]
        results.append(control_results)

    return results

# --- Compliance Analysis Functions ---

def compute_compliance_score(matches_per_control, total_regulations):
    """
    Compute compliance score = number of matched regulations / total regulations
    matches_per_control: list of list of matched regs per control
    total_regulations: int, total number of unique regulations
    Returns a list of compliance scores per control (float 0 to 1)
    """
    scores = []
    for matches in matches_per_control:
        unique_regs = set(m['regulation'] for m in matches)
        score = len(unique_regs) / total_regulations if total_regulations > 0 else 0
        scores.append(score)
    return scores

def detect_gaps(matches_per_control, all_categories, all_regions):
    """
    For each control, detect missing categories and regions based on matched regulations.
    matches_per_control: list of list of matched regulations dicts per control
    all_categories: set of all possible categories
    all_regions: set of all possible regions
    Returns two lists:
      missing_categories_per_control: list of lists
      missing_regions_per_control: list of lists
    """
    missing_categories = []
    missing_regions = []

    for matches in matches_per_control:
        matched_cats = set()
        matched_regions = set()

        for m in matches:
            if 'category_refined' in m and m['category_refined']:
                matched_cats.add(m['category_refined'])
            if 'region' in m and m['region']:
                matched_regions.add(m['region'])

        missing_cat = list(all_categories - matched_cats) if all_categories else []
        missing_reg = list(all_regions - matched_regions) if all_regions else []

        missing_categories.append(missing_cat)
        missing_regions.append(missing_reg)

    return missing_categories, missing_regions

# --- Main script logic ---

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())

    # Paths
    controls_emb_path = "data/control_embeddings.pkl"
    regulations_emb_dir = "data/embeddings"

    # Load control embeddings
    print("Loading control embeddings...")
    control_embeddings = load_embeddings(controls_emb_path)
    if control_embeddings is None:
        raise SystemExit("Failed to load control embeddings - exiting")
    print(f"Loaded {len(control_embeddings)} control embeddings.")

    # Load regulation embeddings
    print("Loading regulations embeddings...")
    regulations_embeddings_dict = load_all_regulation_embeddings(regulations_emb_dir)
    if not regulations_embeddings_dict:
        raise SystemExit("No regulation embeddings loaded - exiting")
    print(f"Loaded embeddings for regulations: {list(regulations_embeddings_dict.keys())}")

    # Perform matching
    print("Matching controls to regulations...")
    matches = match_controls_to_regulations(control_embeddings, regulations_embeddings_dict, top_n=5, min_threshold=0.65)

    # Print results
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
                print(f"   - Category: {match['category_refined']}\n")

    print("Preparing to save summary and results...")

    try:
        # Count total matched controls
        total_matched_controls = sum(1 for m in matches if len(m) > 0)
        print(f"\n✅ Total Controls Matched: {total_matched_controls} / {len(matches)}")

        # Save to JSON
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64)): return int(obj)
            if isinstance(obj, (np.floating, np.float64)): return float(obj)
            if isinstance(obj, (np.ndarray)): return obj.tolist()
            return obj

        os.makedirs("data", exist_ok=True)
        with open("data/match_results.json", "w") as f:
            json.dump(matches, f, indent=2, default=convert)
        print("📁 Saved match results to data/match_results.json")

        # Save to CSV
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

    # Summary stats
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
                summary_stats[match["match_level"]] += 1
                reg = match["regulation"]
                regulation_counts[reg] = regulation_counts.get(reg, 0) + 1

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