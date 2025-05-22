# %% [markdown]
# # Exploring Dandiset 001354: Hippocampal Neuronal Responses to Programmable Antigen-Gated GPCR Activation
#
# **Notebook generated with the assistance of AI. Please review the code, results, and interpretations with care.**

# %% [markdown]
# ## Overview
#
# This notebook provides an introduction to [Dandiset 001354](https://dandiarchive.org/dandiset/001354/0.250312.0036), a dataset containing single-cell, whole-cell patch-clamp electrophysiology recordings from mouse hippocampal CA1 neurons. The highly structured Neurodata Without Borders (NWB) files include both raw membrane voltage responses and injected current protocols during programmable receptor activation.
#
# The notebook will help researchers explore the Dandiset, inspect the available data, and visualize example voltage and current traces for a single neuron/cell.
#
# ---
#
# ## What This Notebook Covers
#
# - Overview and access methods for Dandiset 001354
# - Listing and exploring NWB files in this Dandiset using the DANDI API
# - Loading NWB data via remote streaming (no local download required)
# - Previewing the structure: acquisition and stimulus data groups
# - Visualizing a complete sweep: membrane potential response and injected current
# - Accessing sweep-level metadata (e.g., stimulus grouping/protocol)
#
# All steps below draw upon procedures demonstrated in our interactive chat above.
#
# ---
#
# ## Required Packages
#
# - `dandi`
# - `pynwb`
# - `remfile`
# - `h5py`
# - `matplotlib`
# - `numpy`
#
# Please ensure these packages are already installed in your environment.

# %% [markdown]
# ## Accessing Dandiset 001354 via the DANDI API
#
# We'll begin by connecting to the DANDI Archive and listing a few of the available NWB files.

# %%
from dandi.dandiapi import DandiAPIClient
from itertools import islice

# Change these if you want to use another version or Dandiset
DANDISET_ID = "001354"
DANDISET_VERSION = "0.250312.0036"

# Set up DANDI client and fetch the Dandiset
client = DandiAPIClient()
dandiset = client.get_dandiset(DANDISET_ID, DANDISET_VERSION)

# List the first 10 NWB files in the Dandiset
assets = dandiset.get_assets_by_glob("*.nwb")
print("First 10 NWB file paths in this Dandiset:\n")
for asset in islice(assets, 10):
    print(asset.path)

# %% [markdown]
# ## Exploring Data Structure: NWB Groups and Series
#
# Each NWB file contains structured datasets for both recorded voltage (CurrentClampSeries) and injected current (CurrentClampStimulusSeries), along with metadata and tables relating sweeps and experiments.
#
# Let's demonstrate this using the first file in the list above.

# %%
# Pick a sample NWB file path (first in the list above)
nwb_path = "sub-PK-109/sub-PK-109_ses-20240717T180719_slice-2024-07-17-0009_cell-2024-07-17-0009_icephys.nwb"

# Get the remote download URL for streaming access
# (Restart client/assets generator as needed)
client = DandiAPIClient()
dandiset = client.get_dandiset(DANDISET_ID, DANDISET_VERSION)
asset = next(dandiset.get_assets_by_glob(nwb_path))
remote_url = asset.download_url

# %% [markdown]
# We'll use `remfile` and `h5py` to stream the NWB data remotely, and `pynwb` for parsing content.

# %%
import pynwb
import h5py
import remfile

# Open the NWB file for streaming access
remote_file = remfile.File(remote_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwbfile = io.read()

# List a preview of CurrentClampSeries objects in acquisition
acq_preview = [k for k, v in nwbfile.acquisition.items() if v.neurodata_type == 'CurrentClampSeries']
print(f"Acquisition group CurrentClampSeries preview (first 5): {acq_preview[:5]}")

# List a preview of CurrentClampStimulusSeries objects in stimulus
stim_preview = [k for k, v in nwbfile.stimulus.items() if v.neurodata_type == 'CurrentClampStimulusSeries']
print(f"Stimulus group CurrentClampStimulusSeries preview (first 5): {stim_preview[:5]}")

# %% [markdown]
# ## Visualizing an Example Sweep: Voltage and Current Traces
#
# We'll examine the membrane voltage response (`current_clamp-response-01-ch-0`) and the corresponding injected current (`stimulus-01-ch-0`). Both traces are streamed and visualized in aligned units with appropriate scaling and axis labels.
#
# **Note:** The current is stored in SI units (amperes) and should be scaled to nanoamperes (nA) for physiological interpretation.

# %%
import numpy as np
import matplotlib.pyplot as plt

# Load voltage trace and associated metadata
ccs = nwbfile.acquisition['current_clamp-response-01-ch-0']
voltage = ccs.data[:] * ccs.conversion  # convert to Volts
sampling_rate = ccs.rate
time = np.arange(len(voltage)) / sampling_rate

# Load the corresponding injected current, properly scaled to nA
ccss = nwbfile.stimulus['stimulus-01-ch-0']
current_nA = ccss.data[:] * ccss.conversion * 1e9  # convert to nA
current_time = np.arange(len(current_nA)) / ccss.rate

# Joint plot: overlay voltage and scaled current using twin y-axes
fig, ax1 = plt.subplots(figsize=(10, 4))
color1 = 'tab:blue'
color2 = 'tab:orange'
ax1.plot(time, voltage, color=color1, label='Voltage (V)')
ax1.set_ylabel('Voltage (V)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
ax2.plot(current_time, current_nA, color=color2, label='Current (nA)', alpha=0.6)
ax2.set_ylabel('Injected Current (nA)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Voltage and Injected Current for Sweep 01 (proper scaling)')
plt.xlabel('Time (s)')
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation
# The plot above shows the synchronized time course of:
# - Membrane potential (blue): Baseline, ramp-driven depolarization, action potentials, and repolarization.
# - Injected current (orange): Step and ramp protocol in nanamperes.
#
# The protocol and cell's response are tightly coupled, reflecting the impact of the injected current on firing behavior.

# %% [markdown]
# ## Accessing Sweep Metadata: Protocol Annotation and Recording Tables
#
# NWB intracellular electrophysiology data are accompanied by tables that relate sweeps to electrodes, stimuli, and responses, and annotate protocol grouping. Here, the 'stimulus_type' associated with each grouped/sequential recording can be extracted from the `icephys_sequential_recordings` table.

# %%
# Access the sequential recordings table and show stimulus type annotation
df_seq = nwbfile.icephys_sequential_recordings.to_dataframe()
print("Stimulus type(s) for sequential recordings in this NWB file:")
print(df_seq[['stimulus_type']].head())

# %% [markdown]
# The `'stimulus_type'` annotation links each group of sweeps to the underlying protocol, e.g., a ramp current injection as shown above.
#
# Additional sweep metadata (e.g., electrode identity, dataset names, start/stop indices) can be extracted by inspecting the `intracellular_recordings` table.
#
# Example:
# 

# %%
# Show the first few rows and column names from the intracellular_recordings table
df_intracell = nwbfile.intracellular_recordings.to_dataframe()
print("First 5 rows of intracellular_recordings table:")
print(df_intracell.head())
print("\nColumns:", df_intracell.columns.tolist())

# %% [markdown]
# ## Summary
#
# This notebook introduced Dandiset 001354 using DANDI and NWB best practices:
# - Connecting to and exploring DANDI Dandisets/assets programmatically
# - Remotely loading structured NWB data for scalable, reproducible analysis
# - Visualizing both cell membrane response and corresponding injected current, with correct physiological scaling
# - Examining sweep metadata and protocol grouping for flexible, annotated reanalysis
#
# For more detailed analyses, you can build on these procedures to aggregate, compare, and quantify electrophysiology data across cells, stimuli, or experimental conditions. Be sure to review data attributes and units for consistency.
#
# Further resources:
# - [Dandiset 001354 landing page](https://dandiarchive.org/dandiset/001354/0.250312.0036)
# - [NWB Format documentation](https://www.nwb.org/)
# - [DANDI Archive documentation](https://www.dandiarchive.org/)
