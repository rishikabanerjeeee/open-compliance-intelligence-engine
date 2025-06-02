from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import json
from models import match_engine
from utils import plot_utils  # Make sure this module exists

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

def main():
    try:
        # Load the sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load control statements from a CSV file
        controls_df = pd.read_csv("data/controls/controls.csv")
        control_texts = controls_df["control_statement"].fillna("").tolist()

        # Generate sentence embeddings for all control statements
        control_embeddings = model.encode(control_texts)

        # Load all pre-generated regulation embeddings
        reg_embeds = match_engine.load_all_regulation_embeddings("data/embeddings")
        if not reg_embeds:
            raise ValueError("No regulation embeddings loaded")

        # Perform control-to-regulation matching
        matches = match_engine.match_controls_to_regulations(
            control_embeddings,
            reg_embeds,
            top_n=3,
            min_threshold=0.5
        )

        # Display matching results
        for i, match_list in enumerate(matches):
            print(f"\n🔐 Control {i+1}: {control_texts[i][:100]}...")  # Truncate long text
            print("-" * 100)
            if not match_list:
                print("❌ No matches found.")
            for m in match_list:
                print(f"✅ Regulation: {m['regulation']}")
                print(f"   - Similarity: {m['similarity']:.2f}")
                print(f"   - Requirement: {m['requirement_text'][:100]}...")
                print(f"   - Tags: {m.get('tags', 'N/A')} | Category: {m.get('category_refined', 'N/A')}")

        # Get unique regulation names
        reg_list = list(reg_embeds.keys())
        
        # Get all categories and regions from regulation embeddings
        all_categories = set()
        all_regions = set()
        for reg_data in reg_embeds.values():
            if 'categories' in reg_data and reg_data['categories']:
                all_categories.update(reg_data['categories'])
            if 'tags' in reg_data and reg_data['tags']:  # Assuming regions might be in tags
                # You might need more specific logic here depending on your data structure
                pass

        # Compute compliance metrics
        compliance_scores = match_engine.compute_compliance_score(matches, len(reg_list))
        missing_categories, missing_regions = match_engine.detect_gaps(matches, all_categories, all_regions)

        # Build comprehensive report
        report = []
        for i, (text, score) in enumerate(zip(control_texts, compliance_scores)):
            match_list = matches[i]
            matched_regs = list(set(m['regulation'] for m in match_list)) if match_list else []
            
            report.append({
                "control_id": i + 1,
                "control_text": text,
                "compliance_score": score,
                "matched_regulations": matched_regs,
                "matched_count": len(matched_regs),
                "missing_categories": missing_categories[i],
                "missing_regions": missing_regions[i],
                "matches": match_list
            })

        # Save reports
        with open("outputs/compliance_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Create simplified DataFrame for CSV
        csv_data = []
        for item in report:
            csv_data.append({
                "control_id": item["control_id"],
                "control_text": item["control_text"][:200],  # Truncate for CSV
                "compliance_score": item["compliance_score"],
                "matched_count": item["matched_count"],
                "matched_regulations": ", ".join(item["matched_regulations"]),
                "missing_categories": ", ".join(item["missing_categories"]),
                "missing_regions": ", ".join(item["missing_regions"])
            })
        
        pd.DataFrame(csv_data).to_csv("outputs/compliance_report.csv", index=False)
        print("\n✅ Compliance scores & gap analysis saved to outputs/ folder.")

        # Generate visualizations
        plot_utils.plot_compliance_bar(
            [r["control_text"][:50] + "..." for r in report],  # Truncate labels
            [r["compliance_score"] for r in report]
        )
        
        regulation_coverage = {}
        for item in report:
            for reg in item["matched_regulations"]:
                regulation_coverage[reg] = regulation_coverage.get(reg, 0) + 1
                
        plot_utils.plot_coverage_heatmap(matches, reg_list)
        plot_utils.plot_region_pie(regulation_coverage)

    except Exception as e:
        print(f"❌ Error in processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()