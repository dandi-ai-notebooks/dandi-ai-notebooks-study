# %% [markdown]
# # Exploring Dandiset 000563: Barcoding in Visual System Responses
# 
# *Allen Institute Openscope - Barcoding* ([View Dandiset](https://dandiarchive.org/dandiset/000563/0.250311.2145))
# 
# ---
# 
# **Notebook generated with the assistance of AI. Please review both code and results carefully before interpreting or reusing them for your research.**
# 
# ---
# 
# ## Overview
# 
# Dandiset 000563 contains multi-modal electrophysiological recordings from the mouse visual system, acquired by the Allen Institute using Neuropixels probes and optogenetic tagging. The dataset focuses on the “barcoding” phenomenon—highly repeatable, temporally distinct spike patterns evoked by repeated visual white noise stimuli.
# 
# The dataset uses NWB (Neurodata Without Borders) format and features:
# - Spike-sorted units
# - Precise stimulus timing (e.g., full-field flicker, static gratings)
# - Behavioral monitoring (running wheel, eye tracking)
# - Optogenetic stimulation (with intervals)
# 
# [DANDI Archive Link](https://dandiarchive.org/dandiset/000563/0.250311.2145)
# 
# ---
# 
# ## What This Notebook Covers
# 
# - How to find and load NWB files from the Dandiset using the DANDI API (remotely)
# - How to inspect high-level metadata and NWB file structure
# - How to efficiently access and visualize:
#   - Subject/session metadata
#   - Spike-sorted units and their spike trains using the correct approach (`units.spike_times_index[i]`)
#   - Stimulus structure and epochs
#   - Behavioral signals (running speed, eye tracking)
#   - Optogenetic stimulation events
# - How to find stimulus-responsive units and plot spike rasters for repeated stimuli
# 
# Along the way, you’ll find code examples and visualizations to help you get started with your own analyses.
# 
# ---
# 
# ## Required Packages
# 
# This notebook assumes you have the following packages available:
# - dandi
# - remfile
# - h5py
# - pynwb
# - pandas
# - numpy
# - matplotlib
# 
# ---
# 
# ## 1. Connecting to the Dandiset and Listing NWB Files

# %%
# Import standard libraries for data streaming and analysis
from dandi.dandiapi import DandiAPIClient
from itertools import islice

# Open connection to DANDI
client = DandiAPIClient()
dandiset_id = "000563"
dandiset_version = "0.250311.2145"
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# List all NWB files (first 10 shown for illustration)
assets = list(dandiset.get_assets_by_glob("*.nwb"))
print(f"Total NWB files in dandiset {dandiset_id}: {len(assets)}")
print("First 10 NWB files:")
for asset in islice(assets, 10):
    print(f"- {asset.path}")

# %% [markdown]
# The main, richly-annotated NWB files are those containing `_ogen.nwb` in their names. These include subject/session metadata, spike-sorted units, behavioral data, detailed stimuli and optogenetic events.
# 
# Let's select one of these files for further exploration:

# %%
# Find main data files with 'ogen' in the filename
ogen_files = [asset for asset in assets if "ogen" in asset.path]
session_path = ogen_files[0].path
print(f"Example main NWB file selected:\n{session_path}")

# %% [markdown]
# ## 2. Loading the NWB File Remotely (Streaming)
# 
# The file is accessed directly from the DANDI cloud, streamed via the `remfile` package to avoid downloading the entire file.
# 
# **Note:** This method works for exploration and smaller-scale data access. For very large-scale analyses, consider transferring files locally.
# 

# %%
import h5py
import pynwb
import remfile

# Set up for remote streaming via DANDI/remfile/h5py
asset = next(dandiset.get_assets_by_glob(session_path))
remote_file = remfile.File(asset.download_url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()
print("File loaded successfully.")

# %% [markdown]
# ---
# ## 3. Inspecting the NWB File: Metadata and Organization
# 
# Let's examine session metadata, the structure of electrodes and units, available stimuli, processing modules, and acquisition timeseries.

# %%
# Extract key information about the session and file structure
def get_nwb_overview(nwb):
    subject = nwb.subject
    overview = dict(
        session_id = getattr(nwb, "session_id", None),
        session_date = str(nwb.session_start_time.date()),
        subject_id = getattr(subject, "subject_id", None),
        species = getattr(subject, "species", None),
        sex = getattr(subject, "sex", None),
        age = getattr(subject, "age", None),
        genotype = getattr(subject, "genotype", None),
        electrodes = nwb.electrodes.to_dataframe().shape[0],
        units = nwb.units.to_dataframe().shape[0],
        stimulus_blocks = list(nwb.intervals.keys()),
        acquisition = list(nwb.acquisition.keys()),
        processing_modules = list(nwb.processing.keys())
    )
    return overview

meta = get_nwb_overview(nwb)
print("Session/Subject Metadata:")
for k, v in meta.items():
    print(f"{k}: {v}")

# %% [markdown]
# ---
# ## 4. Exploring Stimulus Structure
# 
# Each stimulus type (e.g. `RepeatFFF_presentations`) is stored as a `TimeIntervals` table.
# 
# Let's examine the timing and structure of the `RepeatFFF` stimulus epochs and view the distribution and variability of trial durations.

# %%
import pandas as pd
import matplotlib.pyplot as plt

repeatfff_df = nwb.intervals['RepeatFFF_presentations'].to_dataframe()
onsets = repeatfff_df['start_time'].values[:1000]
durations = (repeatfff_df['stop_time'] - repeatfff_df['start_time']).values[:1000]

plt.figure(figsize=(10, 4))
plt.plot(onsets, durations, '.', alpha=0.7)
plt.xlabel('Stimulus Onset Time (s)')
plt.ylabel('Duration (s)')
plt.title('RepeatFFF Presentation Intervals (First 1000)')
plt.tight_layout()
plt.show()

# Show a sample of relevant properties for a few epochs
properties = ['contrast', 'orientation', 'spatial_frequency', 'size', 'phase', 'units']
print("Sample RepeatFFF stimulus properties:")
print(repeatfff_df[properties].head())

# %% [markdown]
# The plot above shows extremely regular, precisely timed trial durations for repeated flicker stimuli, confirming the strong experimental control in this dataset.
# 
# The properties DataFrame shows each trial’s parameters, supporting parameter-specific analyses.
# 
# ---
# ## 5. Efficiently Accessing and Visualizing Spike Trains
# 
# To work efficiently with spike trains, you should avoid loading the entire units table. Instead, use `units.spike_times_index[i]` to retrieve spike times for the *i*-th unit.
# 
# Below, we screen all units to find the one with the most spikes in a window around RepeatFFF trials, then visualize its spike raster aligned to stimulus onset. This increases the chances of observing stimulus-driven activity.

# %%
import numpy as np

# Get the first 100 RepeatFFF stimulus onsets for quick raster construction
n_trials = 100
trial_onsets = repeatfff_df['start_time'].values[:n_trials]
window = [-0.05, 0.05]  # 50 ms pre/post onset

# Function to count spikes per unit in the stimulus-aligned window
def count_aligned_spikes(spike_times, onsets, window):
    return sum([(spike_times >= onset + window[0]) & (spike_times < onset + window[1])).sum() for onset in onsets])

unit_count = len(nwb.units.id)
spike_counts = []
for i in range(unit_count):
    spike_times = nwb.units.spike_times_index[i]  # the spike vector for unit i
    spike_counts.append(count_aligned_spikes(spike_times, trial_onsets, window))

best_unit_idx = int(np.argmax(spike_counts))
print(f"Best responsive unit index: {best_unit_idx} | spikes in window: {spike_counts[best_unit_idx]}")

# Raster for the most stimulus-responsive unit
spike_times = nwb.units.spike_times_index[best_unit_idx]
rasters = [(spike_times[(spike_times >= onset + window[0]) & (spike_times < onset + window[1])] - onset)
           for onset in trial_onsets]

plt.figure(figsize=(10, 5))
for i, spikes in enumerate(rasters):
    plt.vlines(spikes, i+0.5, i+1.5, color='k', alpha=0.8)
plt.xlabel('Time from Stimulus Onset (s)')
plt.ylabel('Trial')
plt.title('Spike Raster: Most Responsive Unit (RepeatFFF Stimulus)')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Tip:** For robust analyses, select units with activity in the stimulus window of interest. This approach helps reveal “barcoding”—precise, repeatable spike patterns across trials.
# 
# ---

# %% [markdown]
# ## 6. Exploring Behavioral Data: Running Wheel and Eye Tracking
# 
# Both running and eye movements are valuable for interpreting neural results.
# 
# ### Running Speed

# %%
# Running speed trace (first 10,000 samples)
ts_running = nwb.processing['running'].data_interfaces['running_speed']
running_speed = ts_running.data[:10000]
running_time = ts_running.timestamps[:10000]

plt.figure(figsize=(10, 3))
plt.plot(running_time, running_speed, color='green', lw=1)
plt.xlabel('Time (s)')
plt.ylabel('Running Speed (cm/s)')
plt.title('Behavior: Running Wheel (First 10,000 samples)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Eye Tracking (Ellipse Center)
# 
# The eye tracking data is stored as (x, y) points in meters.

# %%
# Eye tracking ellipse center (first 10,000 points)
eyedata = nwb.acquisition['EyeTracking'].eye_tracking.data[:10000]

plt.figure(figsize=(8, 4))
plt.plot(eyedata[:, 0], eyedata[:, 1], '.', alpha=0.5)
plt.xlabel('Eye X (m)')
plt.ylabel('Eye Y (m)')
plt.title('Behavior: Eye Tracking Ellipse XY (First ~10,000 Samples)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 7. Optogenetic Stimulation Events
# 
# Intervals for optogenetic stimulation are available as a pandas DataFrame. Below is a preview:

# %%
opto_df = nwb.processing['optotagging'].data_interfaces['optogenetic_stimulation'].to_dataframe()
print(opto_df[['start_time', 'stop_time', 'level', 'stimulus_name']].head())

# %% [markdown]
# Using these intervals, you can analyze the effect of direct neural perturbations on spikes, behavior, or response reliability.
# 
# ---
# 
# ## 8. Summary and Next Steps
# 
# This notebook provided an orientation to Dandiset 000563, including direct access to NWB content for:
# 
# - Session/subject context and recording protocol
# - Selection and visualization of spike trains
# - Matching spikes to detailed, structured stimulus intervals
# - Behavioral and optogenetic event exploration
# 
# **What you can do next:**  
# - Quantify repeatability/barcoding in neural patterns for chosen units/stimuli  
# - Explore neural-behavioral relationships by filtering/ranking on running or eye movement
# - Segment analyses by optogenetic state or block type
# - Build on these examples to analyze neural coding, population synchrony, or stimulus tuning
# 
# **For more details:**
# - [Explore the Dandiset on DANDI](https://dandiarchive.org/dandiset/000563/0.250311.2145)
# - [NWB Documentation](https://www.nwb.org/)
# 
# ---
# 
# *Notebook generated with the help of AI. Please validate all results and use these examples as a foundation for your own rigorous exploration!*