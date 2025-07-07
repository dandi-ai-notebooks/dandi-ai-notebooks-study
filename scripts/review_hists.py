# Re-import necessary libraries after code execution state reset
import pandas as pd
import matplotlib.pyplot as plt
import json

# Reload the CSV and JSON files
csv_path = "reviews_export.csv"
json_path = "../dandi-notebook-review/src/data/questions.json"

df = pd.read_csv(csv_path)
with open(json_path, "r") as f:
    question_data = json.load(f)

# Define standard bins for consistent visual layout
standard_bins = {
    "positive": [-0.5, 0.5, 1.5, 2.5],
    "negative": [-2.5, -1.5, -0.5, 0.5]
}

# Extract ordered column IDs from the JSON file
ordered_ids = [q["id"] for q in question_data["questions"]]

# Filter only numeric and valid columns from the dataset
valid_ordered_cols = [col for col in ordered_ids if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

# Plot histograms in the hard-coded order
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 15))
axes = axes.flatten()

for idx, col in enumerate(valid_ordered_cols):
    ax = axes[idx]
    data = df[col].dropna()
    min_val = data.min()

    if min_val < 0:
        bins = standard_bins["negative"]
        ticks = [-2, -1, 0]
    else:
        bins = standard_bins["positive"]
        ticks = [0, 1, 2]

    ax.hist(data, bins=bins, align='mid', rwidth=0.6)
    ax.set_title(col, fontsize=10)
    ax.set_xticks(ticks)
    ax.set_xlim(bins[0], bins[-1])
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Remove any unused subplots
for j in range(len(valid_ordered_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('images/review_histograms.png', dpi=300, bbox_inches='tight')
plt.close()
