import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import os

def plot_compliance_bar(control_texts, compliance_scores, save_path=None):
    plt.figure(figsize=(12, 5))
    sns.barplot(x=list(range(len(compliance_scores))), y=compliance_scores)
    plt.xlabel("Control Index")
    plt.ylabel("Compliance Score")
    plt.title("Compliance Score per Control")
    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


def plot_coverage_heatmap(matches, regulation_list, save_path=None):
    matrix = []
    for control_matches in matches:
        matched = set(m['regulation'] for m in control_matches)
        row = [1 if r in matched else 0 for r in regulation_list]
        matrix.append(row)

    df_matrix = pd.DataFrame(matrix, columns=regulation_list)
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_matrix, cmap="YlGnBu", cbar=True, linewidths=0.3, linecolor='gray')
    plt.title("Heatmap: Control × Regulation Coverage")
    plt.xlabel("Regulations")
    plt.ylabel("Controls")
    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


def plot_region_pie(matches, save_path=None):
    regions = {
        "GDPR": "EU", "DPDP": "India", "RBI": "India", "PIPEDA": "Canada",
        "GLBA": "US", "UK DPA": "UK", "MSA": "Singapore", "BaFin": "Germany",
        "Australia Privacy": "Australia", "SEBI": "India"
    }

    all_matches = [m['regulation'] for control in matches for m in control]
    region_labels = [regions.get(reg, "Unknown") for reg in all_matches]
    region_counts = dict(Counter(region_labels))

    plt.figure(figsize=(7, 7))
    plt.pie(region_counts.values(), labels=region_counts.keys(), autopct='%1.1f%%', startangle=140)
    plt.title("Regulation Coverage per Region")
    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt

def plot_pie_coverage(coverage_dict, save_path=None):
    labels = list(coverage_dict.keys())
    sizes = list(coverage_dict.values())

    plt.figure(figsize=(7,7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Regulation Coverage Distribution")
    plt.axis('equal')  # Equal aspect ratio for a circle

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Pie chart saved to {save_path}")
    else:
        plt.show()
