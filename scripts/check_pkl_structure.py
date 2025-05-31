import os
import pickle

EMBED_DIR = "data/embeddings"

for filename in os.listdir(EMBED_DIR):
    if filename.endswith(".pkl"):
        path = os.path.join(EMBED_DIR, filename)
        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            print(f"✅ {filename} → Keys: {list(data.keys())}")
            print(f"   Example requirement: {data['expanded_text'][0][:80]}...")
            print(f"   Example tag: {data['tags'][0]}")
            print(f"   Example category: {data['category_refined'][0]}")
        else:
            print(f"❗ {filename} is not a dict. Type: {type(data)}")

        print("—" * 80)
