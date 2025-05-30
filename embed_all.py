import os
from models.sentence_encoder import process_regulation_csv

# Folder paths
REGULATIONS_DIR = './data/regulations'
EMBEDDINGS_DIR = './data/embeddings'

def main():
    # Ensure embeddings output folder exists
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)
        print(f"Created embeddings directory at {EMBEDDINGS_DIR}")

    # Process all CSV files inside regulations folder
    for file in os.listdir(REGULATIONS_DIR):
        if file.endswith('.csv'):
            csv_path = os.path.join(REGULATIONS_DIR, file)
            print(f"Processing {csv_path} ...")
            process_regulation_csv(csv_path, EMBEDDINGS_DIR)

if __name__ == "__main__":
    main()
