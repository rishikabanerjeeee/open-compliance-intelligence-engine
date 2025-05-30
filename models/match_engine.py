# /models/match_engine.py

import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_all_regulation_embeddings(embeddings_dir):
    """
    Load all .pkl embeddings in the directory into a dict:
    {
        'gdpr': numpy_array_of_embeddings,
        'rbi': numpy_array_of_embeddings,
        ...
    }
    """
    embeddings_dict = {}
    for file in os.listdir(embeddings_dir):
        if file.endswith('.pkl'):
            reg_name = file.replace('.pkl', '')
            path = os.path.join(embeddings_dir, file)
            embeddings_dict[reg_name] = load_embeddings(path)
    return embeddings_dict

def match_controls_to_regulations(control_embeddings, regulations_embeddings_dict, top_n=3, threshold=0.75):
    """
    Args:
        control_embeddings: np.array of shape (num_controls, embedding_dim)
        regulations_embeddings_dict: dict {reg_name: np.array of embeddings}
        top_n: number of top matches to return per control
        threshold: minimum cosine similarity for valid match

    Returns:
        List of lists: for each control, a list of matching dicts:
        [
            [
                {'control_index': 0, 'regulation': 'gdpr', 'requirement_index': 3, 'similarity': 0.82},
                {...},
                {...}
            ],
            [...],
            ...
        ]
    """
    results = []

    for i, ctrl_emb in enumerate(control_embeddings):
        control_results = []
        for reg_name, reg_embs in regulations_embeddings_dict.items():
            # Compute cosine similarity between control embedding and all regulation req embeddings
            sims = cosine_similarity(ctrl_emb.reshape(1, -1), reg_embs)[0]  # shape: (num_reg_reqs,)

            # Get top N highest similarity indices
            top_indices = sims.argsort()[-top_n:][::-1]
            for idx in top_indices:
                score = sims[idx]
                if score >= threshold:
                    control_results.append({
                        'control_index': i,
                        'regulation': reg_name,
                        'requirement_index': idx,
                        'similarity': float(score),
                    })

        # Sort matches by similarity descending and keep only top N
        control_results = sorted(control_results, key=lambda x: x['similarity'], reverse=True)[:top_n]
        results.append(control_results)

    return results
