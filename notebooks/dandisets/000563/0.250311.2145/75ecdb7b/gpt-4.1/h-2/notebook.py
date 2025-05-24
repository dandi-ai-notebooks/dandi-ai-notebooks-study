# %% [markdown]
# # Exploring Dandiset 000563: Allen Institute Openscope - Barcoding
#
# *A practical introduction for researchers interested in mouse visual system neural recordings from the DANDI Archive.*
#
# ---
#
# **Notebook generated with the assistance of AI.**
#
# Please use caution and verify all code, results, and interpretations—this notebook is intended as an exploratory starting point.

# %% [markdown]
# ## Overview
#
# This notebook provides an introduction to **Dandiset 000563: Allen Institute Openscope - Barcoding** ([DANDI Archive link](https://dandiarchive.org/dandiset/000563/0.250311.2145)), which contains large-scale Neuropixels electrophysiological, behavioral, and stimulus data from mouse visual cortex and related areas. It is designed to help researchers:
#
# - Explore the contents and structure of the Dandiset
# - Load NWB files directly from the archive for interactive data analysis
# - Visualize Local Field Potentials (LFP), spike-sorted units, and behavioral data
# - Examine visual/optogenetic stimulus presentation timing
#
# **What this notebook covers:**
#
# 1. Dandiset and NWB Asset overview  
# 2. Loading remote NWB files with the DANDI API and streaming them with RemFile  
# 3. Example visualizations:
#     - LFP (Local Field Potential) segment
#     - Raster plot of spike-sorted units
#     - Pupil area from eye tracking over time
#     - Stimulus presentation/raster plot
# 4. Guidance to help you extend the analyses for your own research

# %% [markdown]
# ## Requirements
#
# This notebook assumes you have the following packages installed:
#
# - `dandi`
# - `pynwb`
# - `remfile`
# - `h5py`
# - `matplotlib`
# - `numpy`
# - `pandas`
#
# The code cells below use only APIs and methods discussed directly in the chat above.

# %% [markdown]
# ## 1. Exploring the Dandiset and NWB Assets

# %%
from dandi.dandiapi import DandiAPIClient

# Connect to the DANDI Archive and fetch the Dandiset
dandiset_id = "000563"
dandiset_version = "0.250311.2145"

client = DandiAPIClient()
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# List the first 10 NWB file paths
print("First 10 NWB files in the Dandiset:")
assets = dandiset.get_assets_by_glob("*.nwb")
asset_list = [a.path for a in assets]
for fn in asset_list[:10]:
    print("  ", fn)

# %% [markdown]
# ### Example File Structure
#
# File names follow this general pattern:
# - `sub-<subjectID>/sub-<subjectID>_ses-<session_id>_probe-<N>_ecephys.nwb` (electrophysiology: usually LFP & spikes)
# - `sub-<subjectID>/sub-<subjectID>_ses-<session_id>_ogen.nwb` (optogenetics/behavior/stimulus/units/metadata aggregation)
#
# Let's pick one file from each: an *ecephys* (LFP) file and the corresponding *ogen* (with spikes, opto, behavior, stimuli).

# %% [markdown]
# ## 2. Loading NWB Data Remotely ("Streaming" from DANDI)
#
# Files will be streamed using RemFile, which allows analysis without needing to download entire NWB files.

# %%
import h5py
import pynwb
import remfile

# Example NWB file: LFP data for one probe
lfp_asset_path = "sub-699241/sub-699241_ses-1318772854_probe-1_ecephys.nwb"
asset = next(dandiset.get_assets_by_glob(lfp_asset_path))
lfp_url = asset.download_url

# Example NWB file: "ogen" file for same session (spikes, behavior, opto, stimuli)
ogen_asset_path = "sub-699241/sub-699241_ses-1318772854_ogen.nwb"
asset_ogen = next(dandiset.get_assets_by_glob(ogen_asset_path))
ogen_url = asset_ogen.download_url

# (Files are not downloaded, but streamed as needed.)

# %% [markdown]
# ## 3. LFP Data: Visualize a 10-Second Segment
#
# Let's load and plot 10 seconds of LFP data from channel 0 in the LFP-only NWB file.

# %%
import numpy as np
import matplotlib.pyplot as plt

# Open the LFP file remotely
remote_file = remfile.File(lfp_url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# Access LFP for probe 1 (shape: [samples, channels])
lfp = nwb.acquisition['probe_1_lfp'].electrical_series['probe_1_lfp_data']
lfp_timestamps = lfp.timestamps[:]  # In ms, from our exploration above
channel_idx = 0

# Select a segment for the first 10,000 ms = 10 s
mask = lfp_timestamps < 10000
seg_times = lfp_timestamps[mask] / 1000  # convert ms to s
seg_data = lfp.data[:len(seg_times), channel_idx]

# Plot the LFP segment
plt.figure(figsize=(10,4))
plt.plot(seg_times, seg_data * 1e3)
plt.xlabel('Time (s)')
plt.ylabel('LFP (mV)')
plt.title('LFP: Channel 0 (First 10 Seconds)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Spike-Sorted Units: Raster Plot for 10 Units
#
# The "ogen" NWB files contain spike-sorted units. We'll show a raster plot for the first 10 units in a 10-second window where spikes are present (20–30 s).

# %%
# Access the ogen NWB file remotely
remote_file_ogen = remfile.File(ogen_url)
h5_file_ogen = h5py.File(remote_file_ogen, 'r')
io_ogen = pynwb.NWBHDF5IO(file=h5_file_ogen, load_namespaces=True)
nwb_ogen = io_ogen.read()

# Correct way to get spike times: units.spike_times_index[i] gives array of spike times for unit i
num_units = 10
spike_times_list = [nwb_ogen.units.spike_times_index[i] for i in range(num_units)]

# Select a window (20–30 s) where spikes are present for these units
window_start, window_end = 20, 30
spikes_window = [st[(st >= window_start) & (st < window_end)] for st in spike_times_list]

plt.figure(figsize=(10,4))
for i, st in enumerate(spikes_window):
    plt.vlines(st, i + 0.5, i + 1.5, color='black')
plt.xlabel('Time (s)')
plt.ylabel('Unit')
plt.yticks(np.arange(1, num_units+1), np.arange(1, num_units+1))
plt.title('Spike Raster: Units 1–10, 20–30s')
plt.xlim(window_start, window_end)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Pupil Area (Eye Tracking) Over 30 Seconds
#
# The ogen file contains rich behavioral data, including pupil tracking. Let's plot pupil area for the first 30 seconds of recording.

# %%
pupil = nwb_ogen.acquisition['EyeTracking'].spatial_series['pupil_tracking']
pupil_area = pupil.area[:]
pupil_timestamps = pupil.timestamps[:]
# The data may start later than t=0. We'll just use the first 30 seconds of available timestamps.
mask = (pupil_timestamps >= pupil_timestamps[0]) & (pupil_timestamps < pupil_timestamps[0] + 30)
plot_times = pupil_timestamps[mask]
plot_area = pupil_area[mask]

plt.figure(figsize=(10,4))
plt.plot(plot_times, plot_area)
plt.xlabel('Time (s)')
plt.ylabel('Pupil Area')
plt.title('Pupil Area Over First 30 Seconds')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Exploring Unit Properties
#
# Spike-sorted units in the ogen file have many associated quality metrics. Here's a table and summary for some key unit properties.

# %%
import pandas as pd
units = nwb_ogen.units
to_show = ['firing_rate', 'quality', 'isolation_distance', 'snr', 'amplitude', 'waveform_duration']
avail = [c for c in units.colnames if c in to_show]
unit_df = units.to_dataframe()[avail]

# Show first 10 units and summary statistics
display(unit_df.head(10))
display(unit_df.describe(include='all').transpose())

# %% [markdown]
# ## 7. Stimulus Presentation Times: Event Raster
#
# Let's visualize when visual stimuli are presented, as captured in the stimulus processing module.

# %%
stimulus = nwb_ogen.processing['stimulus']
stim_timestamps = stimulus.data_interfaces['timestamps'].data[:]

# Select a 1-second window to see individual events
window_start = stim_timestamps[0]
window_end = window_start + 1
mask = (stim_timestamps >= window_start) & (stim_timestamps < window_end)
window_times = stim_timestamps[mask]

plt.figure(figsize=(10,2))
plt.eventplot(window_times, orientation='horizontal', colors='tab:blue', lineoffsets=0.5, linelengths=0.8)
plt.xlabel('Time (s)')
plt.yticks([])
plt.title('Stimulus Event Times (First 1 s Window)')
plt.xlim(window_start, window_end)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conclusion & Next Steps
#
# This notebook offered initial exploratory analyses for Dandiset 000563:
# - How to access and stream data files with the DANDI API, RemFile, and pynwb
# - How to examine LFP, spikes, behavioral tracking, and visual stimulus structure
#
# **You can now build on these examples to:**
# - Overlay behavioral, neural, and stimulus data
# - Quantify responses to specific events or stimuli
# - Examine higher-dimensional aspects (across probes, units, sessions)
#
# For further analyses, see also the interactive DANDI Archive and Neurosift NWB browser for this Dandiset.

# %% [markdown]
# ---
#
# *Notebook generated with the assistance of AI. Please check all analyses and code before using in your research.*