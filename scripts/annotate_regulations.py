import os
from utils import parser

REG_DIR = "data/regulations"

def main():
    for filename in os.listdir(REG_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(REG_DIR, filename)
            print(f"🔄 Annotating: {filename}...")

            df = parser.load_csv(file_path)

            expanded_texts = []
            tags_list = []
            refined_categories = []

            for _, row in df.iterrows():
                requirement = row["requirement_text"]
                category = row["category"]

                expanded, tags, refined_cat = parser.annotate_row(requirement, category)

                expanded_texts.append(expanded)
                tags_list.append(tags)
                refined_categories.append(refined_cat)

            df["expanded_text"] = expanded_texts
            df["tags"] = tags_list
            df["category_refined"] = refined_categories

            parser.save_csv(df, file_path)
            print(f"✅ Annotated: {filename}")

    print("🎉 All files processed successfully!")

if __name__ == "__main__":
    main()
