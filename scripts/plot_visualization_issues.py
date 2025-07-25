
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the data
df = pd.read_csv("reviews_export.csv")

# Compute breakdown
breakdown = df.groupby("dandiset_id")["visualization-issues"].value_counts().unstack(fill_value=0)
breakdown = breakdown[[0, -1, -2]] if set([0, -1, -2]).issubset(breakdown.columns) else breakdown
breakdown.reset_index(inplace=True)
breakdown.set_index("dandiset_id", inplace=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
breakdown.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
ax.set_title("Breakdown of 'Visualization issues' by Dandiset")
ax.set_xlabel("Dandiset ID")
ax.set_ylabel("Number of Responses")
ax.set_xticklabels(breakdown.index, rotation=45)
ax.legend(title="Visualization issues score", labels=["No issues (0)", "Unclear or hard to interpret (-1)", "Misleading or substatial issues (-2)"])
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig("images/visualization_issues_by_dandiset.png", dpi=300)