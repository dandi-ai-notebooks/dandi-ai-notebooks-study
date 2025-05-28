# %% [markdown]
# # Exploring Dandiset 001375: Septum GABA Disruption with DREADDs
#
# *Generated with the assistance of AI. Please review all code and results with caution. Researchers are responsible for verifying interpretations, analyses, and code against the source data and methods.*

# %% [markdown]
# ## Overview
#
# This notebook provides an interactive introduction to [Dandiset 001375 (version 0.250406.1855)](https://dandiarchive.org/dandiset/001375/0.250406.1855) in the DANDI Archive.
#
# **Dandiset 001375** contains large-scale neural recordings from mice in which medial septal GABAergic neurons were disrupted using DREADDs. Mice were running laps in a virtual hallway while hippocampal and neocortical activity was recorded using silicon probes.
#
# The data is provided in NWB format and includes:
# - Raw extracellular voltage from 256 channels
# - Waveform-based sorted units for spike times
# - Behavioral events: trial intervals for each lap
# - Electrode and recording device metadata
#
# You are encouraged to explore the files, understand their structure, and adapt the code below for your own reanalyses.

# %% [markdown]
# ## What this notebook will cover
#
# - Listing and exploring the NWB files in the Dandiset
# - Loading session and subject metadata
# - Summarizing and visualizing:
#   - Units (spike counts and rasters)
#   - Trial/behavioral intervals
#   - Electrode mappings and probe configuration
#   - Raw voltage traces (field/LFP data)
#   - Aligning spikes to trial events
# - How to stream large NWB files efficiently for quick inspection without full downloads
#
# **All code uses the DANDI API, `remfile` for file streaming, and `pynwb` for NWB access.**

# %% [markdown]
# ## Requirements
#
# The following Python packages are used in this notebook (assume these are already installed):
# - dandi
# - remfile
# - h5py
# - pynwb
# - pandas
# - numpy
# - matplotlib

# %% [markdown]
# ## 1. List and Preview NWB Files in the Dandiset
#
# First, let’s use the DANDI API to see which NWB files are available and get basic info about their size and organization.

# %%
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI API and get the dandiset
dandiset_id = "001375"
dandiset_version = "0.250406.1855"
client = DandiAPIClient()
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# List all NWB files and print paths/sizes
assets = list(dandiset.get_assets_by_glob("*.nwb"))
print(f"Number of NWB files: {len(assets)}\n")
for asset in assets:
    size_gb = asset.size / (1024 ** 3)
    print(f"- {asset.path} ({size_gb:.2f} GB)")

# %% [markdown]
# ## 2. Load an Example NWB File Remotely
#
# NWB files are large, so we stream a remote file segment for exploration. Here we use `remfile` with h5py and `pynwb`. Let’s start with the first NWB file.

# %%
import remfile
import h5py
import pynwb

# Get the first NWB asset and prepare for streaming
nwb_asset = assets[0]
print("Loading file:", nwb_asset.path)
remote_file = remfile.File(nwb_asset.download_url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# %% [markdown]
# ## 3. Inspect Session and Subject Metadata
#
# Let’s look at the experimental context, subject details, and summary of units, trials, and electrodes.

# %%
# Session and subject metadata
info = {
    "Session description": nwb.session_description,
    "Session start time": str(nwb.session_start_time),
    "Subject ID": nwb.subject.subject_id,
    "Subject sex": nwb.subject.sex,
    "Subject species": nwb.subject.species,
    "Subject age": nwb.subject.age,
    "Subject description": nwb.subject.description,
}

# Table counts
n_electrodes = nwb.electrodes.to_dataframe().shape[0]
n_trials = nwb.trials.to_dataframe().shape[0]
n_units = nwb.units.to_dataframe().shape[0]

info["Number of electrodes"] = n_electrodes
info["Number of units"] = n_units
info["Number of trials"] = n_trials

for k, v in info.items():
    print(f"{k}: {v}")

# %% [markdown]
# ## 4. Examine the Trial (Behavior) and Units (Spikes) Tables
#
# Preview the start and stop times of several trials (laps), and examine how the spike times are structured.

# %%
import pandas as pd

# First 5 trials: start/stop per lap
trials_df = nwb.trials.to_dataframe()
print("First 5 trials:\n", trials_df.head())

# First 5 units: show spike_times column
units_df = nwb.units.to_dataframe()
print("\nFirst 5 units (truncated):\n", units_df.head())

# %% [markdown]
# ## 5. Visualize Unit and Trial Summaries
#
# Two main data features to understand quickly are spike counts per unit and the distribution of behavioral trial durations.
# 
# **Important:** 
# - The `units` DynamicTable in this dataset uses a `spike_times` column, which is a list of spike time arrays, one per unit (as seen in the table preview, not an index in this file).
# - We'll use this column to calculate spike counts for visualization.

# %%
import matplotlib.pyplot as plt

# Get spike count for each unit
spike_counts = units_df['spike_times'].apply(len)

# Get duration for each trial/lap
trial_durations = trials_df['stop_time'] - trials_df['start_time']

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.bar(spike_counts.index, spike_counts.values)
plt.xlabel('Unit ID')
plt.ylabel('Spike Count')
plt.title('Spike Counts per Unit')

plt.subplot(1,2,2)
plt.hist(trial_durations, bins=30, color='skyblue', edgecolor='k')
plt.xlabel('Trial Duration (s)')
plt.ylabel('Count')
plt.title('Distribution of Trial Durations')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Examine Electrode and Probe Organization
#
# Access and summarize electrode metadata. This reveals which probe/shank/channel groups are present.

# %%
electrodes_df = nwb.electrodes.to_dataframe()
print("First 8 electrodes:\n", electrodes_df.head(8))

# Summarize by group and location
location_counts = electrodes_df['location'].value_counts()
group_counts = electrodes_df['group_name'].value_counts()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
group_counts.plot(kind='bar')
plt.title('Electrode Count per Group Name')
plt.xlabel('Group Name')
plt.ylabel('Electrode Count')

plt.subplot(1,2,2)
location_counts.plot(kind='bar')
plt.title('Electrode Count per Location')
plt.xlabel('Location')
plt.ylabel('Electrode Count')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Visualize Raw Voltage Data from a Few Channels
#
# To see what the extracellular field looks like, extract and plot 1 second from the first 6 channels. This is an efficient preview step.

# %%
import numpy as np

# Access time_series (contains the raw voltage)
ts = nwb.acquisition['time_series']

# Extract the first second (30000 samples @ 30 kHz) and 6 channels
segment = ts.data[:30000, :6]  # [time, channels]
time = np.arange(segment.shape[0]) / ts.rate  # seconds

plt.figure(figsize=(10,5))
for ch in range(6):
    plt.plot(time, segment[:, ch] + ch*2000, label=f'Ch {ch+1}')  # offset for visibility
plt.xlabel('Time (s)')
plt.ylabel('Voltage + offset (μV)')
plt.title('Raw Voltage Traces: First 6 Channels, First 1s')
plt.legend(loc='upper right')
plt.show()

# %% [markdown]
# ## 8. Spike Raster: Session Overview for Example Units
#
# Plot a raster for the first 6 units to quickly visualize the spiking pattern across the initial part of the session.

# %%
plt.figure(figsize=(10,5))
for idx, (unit_id, row) in enumerate(units_df.head(6).iterrows()):
    spike_times = np.array(row['spike_times'])
    plt.vlines(spike_times[spike_times <= 30], idx+0.5, idx+1.5)
plt.yticks(np.arange(1, 7), [f'Unit {uid}' for uid in units_df.head(6).index])
plt.xlabel('Time (s)')
plt.ylabel('Unit')
plt.title('Spike Raster for 6 Example Units (first 30 s)')
plt.xlim(0, 30)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Spike Raster Aligned to Trial Events
#
# Here we show how to align spike times for one unit (the first unit) to the start of each behavioral trial—critical for perievent analyses.

# %%
# Select the first unit's spike times
unit_id, unit_row = next(units_df.iterrows())
spike_times = np.array(unit_row['spike_times'])

# Align to first 20 trials
trial_starts = trials_df['start_time'].values[:20]
trial_durations = (trials_df['stop_time'] - trials_df['start_time']).values[:20]

peri_spike_times = []
for t_start, t_dur in zip(trial_starts, trial_durations):
    rel_spikes = spike_times[(spike_times >= t_start) & (spike_times <= t_start+t_dur)] - t_start
    peri_spike_times.append(rel_spikes)

plt.figure(figsize=(9, 5))
for i, spikes in enumerate(peri_spike_times):
    plt.vlines(spikes, i+0.5, i+1.5)
plt.xlabel('Time from trial start (s)')
plt.ylabel('Trial')
plt.title(f'Spike Raster (Unit {unit_id}) Aligned to Trial Start (first 20 trials)')
plt.xlim(0, np.max(trial_durations))
plt.show()

# %% [markdown]
# ## 10. Summary and Suggestions for Further Analysis
#
# This notebook demonstrated how to:
# - Locate and stream large NWB files from the DANDI Archive
# - Access and visualize session, subject, and probe metadata
# - Summarize and plot spike and behavioral data at a glance
# - Extract and preview field potential data
# - Align spikes to behavioral events for perievent and trial-based analyses
#
# You are now ready to:
# - Repeat these analyses with any NWB file from this dandiset (change the asset path)
# - Extend or adapt the code for your specific research question
# - Investigate other spike units, experiment with cross-unit or cross-trial statistics, or analyze longer sections of data
#
# **Refer to [the DANDI Archive page for this Dandiset](https://dandiarchive.org/dandiset/001375/0.250406.1855) for more details and related resources.**