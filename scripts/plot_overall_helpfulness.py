
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the data
df = pd.read_csv("reviews_export.csv")

# Compute breakdown
breakdown = df.groupby("dandiset_id")["overall-helpfulness"].value_counts().unstack(fill_value=0)
breakdown = breakdown[[2, 1, 0]] if set([0, 1, 2]).issubset(breakdown.columns) else breakdown
breakdown.reset_index(inplace=True)
breakdown.set_index("dandiset_id", inplace=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
breakdown.plot(kind="bar", stacked=True, color=['green', 'yellow', 'red'], ax=ax)
ax.set_title("Breakdown of 'Overall Helpfulness' by Dandiset ID")
ax.set_xlabel("Dandiset ID")
ax.set_ylabel("Number of Responses")
ax.set_xticklabels(breakdown.index, rotation=45)
ax.legend(title="Helpfulness Score", labels=["Very helpful (2)", "Moderately helpful (1)", "Not helpful (0)"])
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig("images/overall_helpfulness_by_dandiset.png", dpi=300)
