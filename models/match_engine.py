import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_all_regulation_embeddings(embeddings_dir):
    """
    Load all embeddings + metadata from pickle files into a dictionary:
    {
        'gdpr': {
            'embeddings': numpy_array,
            'texts': [...],
            'tags': [...],
            'categories': [...],
        },
        ...
    }
    """
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

def match_controls_to_regulations(control_embeddings, regulations_embeddings_dict, top_n=3, min_threshold=0.70):
    """
    Match each control embedding to top N most similar regulation requirements above a minimum similarity threshold.
    Adds a 'match_level' field to each match dict based on similarity:
      - Strong: similarity >= 0.80
      - Possible: 0.75 <= similarity < 0.80
      - Weak: 0.70 <= similarity < 0.75
      - No Match: similarity < 0.70 (excluded)
    """
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

                # Determine match level
                if score >= 0.80:
                    level = "Strong"
                elif score >= 0.75:
                    level = "Possible"
                elif score >= 0.70:
                    level = "Weak"
                else:
                    level = "No Match"  # won't happen due to threshold

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

        control_results = sorted(control_results, key=lambda x: x['similarity'], reverse=True)[:top_n]
        results.append(control_results)

    return results
