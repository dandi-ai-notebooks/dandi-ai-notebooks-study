import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Load the data
df = pd.read_csv("reviews_export.csv")

# Remove rows where either column has missing values
df_clean = df.dropna(subset=['nwb-dandi-expertise', 'overall-helpfulness'])

# Compute breakdown: group by nwb-dandi-expertise and count overall-helpfulness values
breakdown = df_clean.groupby('nwb-dandi-expertise')['overall-helpfulness'].value_counts().unstack(fill_value=0)

# Ensure we have all columns [0, 1, 2] and reorder them to match the color scheme
all_helpfulness_values = [0, 1, 2]
for val in all_helpfulness_values:
    if val not in breakdown.columns:
        breakdown[val] = 0

# Reorder columns to match the color scheme (2, 1, 0)
breakdown = breakdown[[2, 1, 0]]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Use the same color scheme as the overall helpfulness plot
colors = ['#2E8B57', '#9370DB', '#555555']  # Very helpful (2), Moderately helpful (1), Not helpful (0)

# Create stacked bar chart
breakdown.plot(kind='bar', stacked=True, color=colors, ax=ax, width=0.6)

# Customize the plot
ax.set_title("Overall Helpfulness by NWB/DANDI Expertise Level")
ax.set_xlabel("NWB/DANDI Expertise Level")
ax.set_ylabel("Number of Responses")

# Set x-axis labels
expertise_labels = {0: "No expertise (0)", 1: "Some expertise (1)", 2: "High expertise (2)"}
ax.set_xticklabels([expertise_labels.get(int(x), str(int(x))) for x in breakdown.index], rotation=0)

# Set legend
ax.legend(title="Overall Helpfulness", labels=["Very helpful (2)", "Moderately helpful (1)", "Not helpful (0)"])

# Ensure y-axis shows only integer values
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Add grid for better readability
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("images/nwb_dandi_expertise_by_helpfulness.png", dpi=300)
plt.show()
