import os
import pickle

EMBEDDINGS_DIR = "data/embeddings"

def inspect_pkl_file(filepath):
    print(f"\n📄 Inspecting: {os.path.basename(filepath)}")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        print("✅ Keys found:", list(data.keys()))

        for key in ['embeddings', 'expanded_text', 'text', 'tags', 'category_refined']:
            if key in data:
                value = data[key]
                length = len(value) if hasattr(value, '__len__') else 'n/a'
                print(f" - {key}: {type(value)} | length = {length}")
                # Show a preview
                if isinstance(value, list) and len(value) > 0:
                    print(f"   > Sample: {value[0]}")
            else:
                print(f"❌ Missing key: {key}")

    except Exception as e:
        print(f"❗ Error reading {filepath}: {e}")

def inspect_all():
    for filename in os.listdir(EMBEDDINGS_DIR):
        if filename.endswith(".pkl"):
            inspect_pkl_file(os.path.join(EMBEDDINGS_DIR, filename))

if __name__ == "__main__":
    inspect_all()
