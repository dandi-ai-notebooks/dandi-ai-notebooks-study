import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Load the data
df = pd.read_csv("reviews_export.csv")

# Create figure with subplots - use width ratios to make bars roughly same width
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

# ===== LEFT PLOT: Overall Helpfulness by Dandiset ID =====

# Compute breakdown for dandiset plot
breakdown_dandiset = df.groupby("dandiset_id")["overall-helpfulness"].value_counts().unstack(fill_value=0)
breakdown_dandiset = breakdown_dandiset[[2, 1, 0]] if set([0, 1, 2]).issubset(breakdown_dandiset.columns) else breakdown_dandiset
breakdown_dandiset.reset_index(inplace=True)
breakdown_dandiset.set_index("dandiset_id", inplace=True)

# Plot dandiset breakdown
colors = ['#2E8B57', '#9370DB', '#555555']  # Very helpful (2), Moderately helpful (1), Not helpful (0)
breakdown_dandiset.plot(kind="bar", stacked=True, color=colors, ax=ax1)
ax1.set_title("Overall Helpfulness by Dandiset ID")
ax1.set_xlabel("Dandiset ID")
ax1.set_ylabel("Number of Responses")
ax1.set_xticklabels(breakdown_dandiset.index, rotation=45)
ax1.legend(title="Helpfulness Score", labels=["Very helpful (2)", "Moderately helpful (1)", "Not helpful (0)"])
ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# ===== RIGHT PLOT: Overall Helpfulness by NWB/DANDI Expertise =====

# Remove rows where either column has missing values
df_clean = df.dropna(subset=['nwb-dandi-expertise', 'overall-helpfulness'])

# Compute breakdown: group by nwb-dandi-expertise and count overall-helpfulness values
breakdown_expertise = df_clean.groupby('nwb-dandi-expertise')['overall-helpfulness'].value_counts().unstack(fill_value=0)

# Ensure we have all columns [0, 1, 2] and reorder them to match the color scheme
all_helpfulness_values = [0, 1, 2]
for val in all_helpfulness_values:
    if val not in breakdown_expertise.columns:
        breakdown_expertise[val] = 0

# Reorder columns to match the color scheme (2, 1, 0)
breakdown_expertise = breakdown_expertise[[2, 1, 0]]

# Create stacked bar chart for expertise
breakdown_expertise.plot(kind='bar', stacked=True, color=colors, ax=ax2, width=0.6)

# Customize the expertise plot
ax2.set_title("Overall Helpfulness by NWB/DANDI Expertise Level")
ax2.set_xlabel("NWB/DANDI Expertise Level")
ax2.set_ylabel("Number of Responses")

# Set x-axis labels for expertise
expertise_labels = {0: "No expertise (0)", 1: "Some expertise (1)", 2: "High expertise (2)"}
ax2.set_xticklabels([expertise_labels.get(int(x), str(int(x))) for x in breakdown_expertise.index], rotation=0)

# Set legend for expertise plot
ax2.legend(title="Overall Helpfulness", labels=["Very helpful (2)", "Moderately helpful (1)", "Not helpful (0)"])

# Ensure y-axis shows only integer values for both plots
ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Add grid for better readability on both plots
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig("images/combined_helpfulness_analysis.png", dpi=300, bbox_inches='tight')
plt.show()
