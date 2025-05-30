# /models/sentence_encoder.py

from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd
import os

# Load the sentence-transformer model once at module load
# You can switch the model here if needed (e.g., legal-bert)
model = SentenceTransformer('all-MiniLM-L6-v2')

def encode_texts(texts):
    """
    Takes a list of text strings and returns their embeddings as numpy arrays.
    convert_to_tensor=False returns numpy arrays instead of torch tensors.
    """
    return model.encode(texts, convert_to_tensor=False)

def save_embeddings(texts, path):
    """
    Encodes a list of texts and saves the embeddings to a .pkl file at 'path'.
    """
    embeddings = encode_texts(texts)
    with open(path, 'wb') as f:
        pickle.dump(embeddings, f)

def process_regulation_csv(file_path, output_dir):
    """
    Reads a regulation CSV file with column 'requirement_text',
    encodes the texts, and saves embeddings as .pkl in output_dir.

    Arguments:
    - file_path: path to input CSV, e.g., './data/regulations/gdpr.csv'
    - output_dir: directory to save embeddings, e.g., './data/embeddings'

    Saves:
    - embeddings as a .pkl file with same basename as input CSV, e.g., 'gdpr.pkl'
    """

    # Read the CSV into a DataFrame
    df = pd.read_csv(file_path)

    # Safety check to make sure required column exists
    if 'requirement_text' not in df.columns:
        raise ValueError(f"'requirement_text' column not found in {file_path}")

    # Extract the requirement texts
    texts = df['requirement_text'].tolist()

    # Encode texts to embeddings
    embeddings = encode_texts(texts)

    # Prepare output filename and path
    base_name = os.path.basename(file_path).replace('.csv', '.pkl')
    output_path = os.path.join(output_dir, base_name)

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"[✔] Saved embeddings for '{file_path}' → '{output_path}'")
