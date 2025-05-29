# %% [markdown]
# # Exploring Dandiset 000563: Allen Institute Openscope - Barcoding

# %% [markdown]
# > This notebook was generated with the assistance of AI. Exercise caution when interpreting the code or results.

# %% [markdown]
# ## Dandiset Overview and Purpose

# %% [markdown]
# This Dandiset presents electrophysiological data from mouse visual system experiments designed to investigate the temporal precision of neuronal responses to visual white noise stimuli. The dataset focuses on the "barcoding" phenomenon where repeated white noise stimuli evoke highly precise and repeatable "barcode-like" patterns in the spike responses of cortical and subcortical neurons.

# %% [markdown]
# **Key points:**
# - **Purpose:** To study temporal precision and cell-type specific spike "barcodes" in the mouse visual system, from retina to cortex.
# - **Visual Stimuli:** Includes full-field flicker (luminance-modulated) and static gratings (contrast-modulated) driven by repeated white noise sequences.
# - **Techniques:** Multi-electrode (Neuropixels) extracellular recordings; includes optogenetic data.
# - **Dataset Size:** 94 NWB files, ~200 GB of data, 14 subjects.
# - **Data Content:** LFP, spike-sorted units, optogenetic stimulation, stimulus presentations.
# - **Special Note:** LFP timestamps are in milliseconds (not seconds).

# %% [markdown]
# View the Dandiset on DANDI Archive: [https://dandiarchive.org/dandiset/000563/0.250311.2145](https://dandiarchive.org/dandiset/000563/0.250311.2145)

# %% [markdown]
# ## What This Notebook Covers

# %% [markdown]
# This notebook provides an introduction to exploring the Allen Institute Openscope - Barcoding Dandiset, with a focus on:

# %% [markdown]
# 1. **Getting Started**: Connecting to the DANDI Archive and listing NWB files
# 2. **Data Structure Exploration**: Examining the organization of a main NWB file
# 3. **Accessing Spike Data**: Working with spike-sorted units and spike times
# 4. **Stimulus Intervals**: Understanding the stimulus presentation structure and how to align neural data
# 5. **Behavioral Data**: Exploring running speed and other behavioral measurements
# 6. **Optogenetic Information**: Accessing optogenetic stimulation parameters

# %% [markdown]
# ## Required Packages

# %% [markdown]
# The following Python packages are required to run this notebook:

# %%
import h5py
import pynwb
import remfile
from dandi.dandiapi import DandiAPIClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Connecting to the DANDI Archive

# %% [markdown]
# Let's first use the DANDI Python API to get information about the Dandiset:

# %%
# Connect to the dandiset
client = DandiAPIClient()
dandiset = client.get_dandiset("000563", "0.250311.2145")
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# %% [markdown]
# ## 2. Exploring the NWB Files

# %% [markdown]
# Now let's list the NWB files in the Dandiset. In this case, we're especially interested in finding the `*_ogen.nwb` files, which contain the main data:

# %%
from itertools import islice

# List all NWB files in the dandiset
assets = list(dandiset.get_assets_by_glob("*.nwb"))

# Compile file info and look for main data files (typically *_ogen.nwb)
file_list = []
for asset in assets:
    file_info = {
        'path': asset.path,
        'size_gb': round(asset.size / 1e9, 2)
    }
    file_list.append(file_info)

# Files with 'ogen' typically contain optogenetic or main data
ogen_files = [f for f in file_list if 'ogen' in f['path']]

# Display summary
print(f"Total NWB files: {len(file_list)}\n")
print("First 10 NWB files:")
for f in islice(file_list, 10):
    print(f"- {f['path']} ({f['size_gb']} GB)")

print("\nFiles containing 'ogen' (main data files):")
for f in islice(ogen_files, 5):  # Just show first 5 to save space
    print(f"- {f['path']} ({f['size_gb']} GB)")
print(f"... and {len(ogen_files) - 5} more")

# %% [markdown]
# We can see that there are multiple file types. Each recording session has an `*_ogen.nwb` file containing optogenetic data, spike times, stimulus info, and behavior data. There are also additional probe-specific files (`*_probe-N_ecephys.nwb`) that likely contain electrophysiology (LFP) data.

# %% [markdown]
# ## 3. Loading and Exploring NWB File Structure

# %% [markdown]
# Let's load one example NWB file and examine its structure. We'll use a typical `*_ogen.nwb` file:

# %%
# Choose a specific *_ogen.nwb file to examine
asset_path = "sub-699241/sub-699241_ses-1318772854_ogen.nwb"
asset = next(dandiset.get_assets_by_glob(asset_path))

# Load NWB file remotely (streaming without downloading)
remote_file = remfile.File(asset.download_url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# %% [markdown]
# Now, let's extract some basic metadata about this session to get a better understanding:

# %%
# Session metadata summary
meta = {
    'session_id': nwb.session_id,
    'subject_id': nwb.subject.subject_id,
    'species': nwb.subject.species,
    'age': nwb.subject.age,
    'sex': nwb.subject.sex,
    'genotype': nwb.subject.genotype,
    'date': str(nwb.session_start_time.date()),
    'institution': nwb.institution
}

# Number of electrodes and units
num_electrodes = nwb.electrodes.to_dataframe().shape[0]
num_units = nwb.units.to_dataframe().shape[0]

# Available data types
meta['stimulus_kinds'] = list(nwb.intervals.keys())
meta['processing_modules'] = list(nwb.processing.keys())
meta['acquisition_timeseries'] = list(nwb.acquisition.keys())

print("Session metadata:")
for k, v in meta.items():
    if isinstance(v, list):
        print(f"  {k}: {', '.join(map(str, v))}")
    else:
        print(f"  {k}: {v}")
print(f"\nNumber of electrodes: {num_electrodes}")
print(f"Number of spike-sorted units: {num_units}")

# %% [markdown]
# ## 4. Exploring The Spike-Sorted Units

# %% [markdown]
# Let's first look at the distribution of spike-sorted units, their quality metrics and firing rates:

# %%
# Access units through the NWB container
units_df = nwb.units.to_dataframe()

# Summarize key quality metrics
total_units = units_df.shape[0]
qc_counts = units_df['quality'].value_counts(dropna=False)

print(f"Total spike-sorted units: {total_units}")
print("\nUnit quality counts:")
print(qc_counts)

# Show a sample of units 
sample_indices = np.random.choice(units_df.index, size=5, replace=False)
sample_info = units_df.loc[sample_indices, ['cluster_id', 'firing_rate', 'quality', 'peak_channel_id']]
print("\nExample units:")
print(sample_info)

# Plot distribution of firing rates
plt.figure(figsize=(10, 5))
plt.hist(units_df['firing_rate'].dropna(), bins=50, color="#348ABD", alpha=0.7)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Unit Count')
plt.title('Distribution of Unit Firing Rates')
plt.tight_layout()
plt.show()

# %% [markdown]
# This histogram shows the distribution of unit firing rates (Hz) across all spike-sorted units in this session. Most neurons fire at low rates (1–5 Hz), though a subset of units exhibits higher activity.

# %% [markdown]
# ### Accessing Spike Times

# %% [markdown]
# Let's demonstrate how to efficiently access spike times for individual units, using direct access to spike time vectors:

# %%
# Get the quality labels for all units
unit_qualities = nwb.units['quality'][:]
good_idxs = [i for i, q in enumerate(unit_qualities) if q == 'good']

# Select first few good units
good_unit_idxs = good_idxs[:5]
unit_ids = nwb.units.id[:]

print(f"Accessing spike times for unit IDs: {[unit_ids[idx] for idx in good_unit_idxs]}")

# Get spike times for these units directly (more efficient than dataframe access)
spike_trains = [nwb.units.spike_times_index[i] for i in good_unit_idxs]

# Show basic info about each unit's spike train
for i, (idx, spikes) in enumerate(zip(good_unit_idxs, spike_trains)):
    unit_id = unit_ids[idx]
    print(f"Unit {unit_id}: {len(spikes)} spikes, range: [{min(spikes):.2f}, {max(spikes):.2f}] seconds")

# %% [markdown]
# ### Visualizing Spike Raster

# %% [markdown]
# We'll generate a raster plot of the selected "good" units over a time window centered on the earliest spike:

# %%
# Find the global earliest spike across these units
all_spikes = np.concatenate(spike_trains)
t_start = all_spikes.min()
t_stop = t_start + 10  # show a 10s window

# For each unit, collect spikes in [t_start, t_start+10)
plot_spikes = [st[(st >= t_start) & (st < t_stop)] for st in spike_trains]

plt.figure(figsize=(10, 3))
for i, st in enumerate(plot_spikes):
    plt.vlines(st, i + 0.5, i + 1.5, color='k')
plt.yticks(np.arange(1, 6), [f"Unit {unit_ids[idx]}" for idx in good_unit_idxs])
plt.xlabel('Time (s)')
plt.ylabel('Unit')
plt.title(f'Spike Raster: {t_start:.2f}-{t_stop:.2f} s, 5 Good Units')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Exploring Stimulus Presentations

# %% [markdown]
# A key aspect of the barcoding dataset is the structured stimulus presentations. Let's examine how the stimulus intervals are organized:

# %%
# Access RepeatFFF stimulus intervals as DataFrame
repeatfff_df = nwb.intervals['RepeatFFF_presentations'].to_dataframe()

# Visualize distribution of onset times and durations for first 1000 trials
onsets = repeatfff_df['start_time'].values[:1000]
durations = (repeatfff_df['stop_time'] - repeatfff_df['start_time']).values[:1000]

plt.figure(figsize=(10, 4))
plt.plot(onsets, durations, '.', alpha=0.7)
plt.xlabel('Stimulus Onset Time (s)')
plt.ylabel('Duration (s)')
plt.title('RepeatFFF Presentation Intervals (First 1000)')
plt.tight_layout()
plt.show()

# Show unique stimulus properties
properties = ['contrast', 'orientation', 'spatial_frequency', 'size', 'phase', 'units']
print('Sample of stimulus properties (first 5 rows):')
print(repeatfff_df[properties].head())

# %% [markdown]
# The figure above plots the onset times and durations of RepeatFFF stimulus presentations. These intervals are extremely precise and regular, showing very brief and consistent durations (on the order of 16.7 ms, i.e., one 60 Hz video frame).

# %% [markdown]
# ## 6. Searching for Stimulus-Responsive Units

# %% [markdown]
# When working with this dataset, you'll likely want to find units that respond reliably to stimuli. Here's a demonstration of how you might search for units that are active during a stimulus:

# %%
# Pseudocode for finding responsive units 
# (This would be computationally intensive to run on all units, 
# so we'll just demonstrate the approach)

print("Approach for finding stimulus-responsive units:")
print("1. Define a time window relative to stimulus onset")
print("2. For each unit, count spikes across multiple stimulus presentations")
print("3. Select units with the highest spike counts or most reliable responses")
print()

# Example for a small number of units
# Get RepeatFFF intervals 
repeatfff = nwb.intervals['RepeatFFF_presentations']
onsets = repeatfff.to_dataframe()['start_time'].values[:100]  # use first 100 trials

# Define window around stimulus
window = [-0.05, 0.05]  # 50 ms before/after onset

# Function to count spikes in window across trials
def count_aligned_spikes(spike_times, onsets, window):
    count = 0
    for onset in onsets:
        # Count spikes in [onset+window[0], onset+window[1])
        count += np.sum((spike_times >= onset + window[0]) & 
                         (spike_times < onset + window[1]))
    return count

# Demo with just 5 units for efficiency
sample_idxs = good_unit_idxs[:5]
for i, idx in enumerate(sample_idxs):
    spike_times = nwb.units.spike_times_index[idx]
    unit_id = unit_ids[idx]
    count = count_aligned_spikes(spike_times, onsets, window)
    print(f"Unit {unit_id}: {count} spikes in stimulus window across 100 trials")

# %% [markdown]
# ## 7. Exploring Behavioral Data

# %% [markdown]
# This dataset includes behavioral measurements like running speed and eye tracking. Let's explore the running speed data:

# %%
# Access running speed time series
ts_running = nwb.processing['running'].data_interfaces['running_speed']
running_speed = ts_running.data[:10000]  # first 10,000 samples
running_time = ts_running.timestamps[:10000]  # in seconds

# Plot
plt.figure(figsize=(10, 3))
plt.plot(running_time, running_speed, color='green', lw=1)
plt.xlabel('Time (s)')
plt.ylabel('Running speed (cm/s)')
plt.title('Behavior: Running Speed (First ~10,000 Samples)')
plt.tight_layout()
plt.show()

# %% [markdown]
# The running speed plot above shows a snippet of behavioral data from the experiment, capturing mouse locomotion across the first ~10,000 samples. Researchers can use this trace to segment neural analysis by behavioral state (e.g., running vs. stationary).

# %% [markdown]
# ## 8. Accessing Optogenetic Stimulation Events

# %% [markdown]
# Let's examine the optogenetic stimulation events, which are important for understanding the experimental protocols:

# %%
# Access optogenetic stimulation intervals
opto_df = nwb.processing['optotagging'].data_interfaces['optogenetic_stimulation'].to_dataframe()

# Show summary of the first few events
print('First 5 optogenetic stimulation intervals:')
print(opto_df[['start_time', 'stop_time', 'level', 'stimulus_name']].head())

# %% [markdown]
# These intervals allow you to relate neural or behavioral responses to optogenetic protocols and perform precise event-triggered analyses.

# %% [markdown]
# ## Summary and Next Steps

# %% [markdown]
# This notebook has introduced the Allen Institute Openscope - Barcoding dataset and demonstrated how to:

# %% [markdown]
# - **Access NWB file structure and metadata**: Session information, subject details
# - **Work with spike data**: Access and visualize spike times from sorted units
# - **Explore stimulus structure**: Understand timing and properties of visual stimuli
# - **Access behavioral data**: Running speed measurements
# - **Access optogenetic events**: Identify stimulation epochs

# %% [markdown]
# ### Next Steps for Analysis

# %% [markdown]
# - **Barcode Analysis**: Compare spike timing patterns across repeated trials
# - **Response Tuning**: Relate neural properties to stimulus features
# - **Behavioral State**: Separate analyses by running vs. stationary periods
# - **Multi-modal Analysis**: Integrate running/eye-tracking with neural responses

# %% [markdown]
# > Note: When working with LFP data in the probe-specific files, remember that the timestamps are in milliseconds and need to be divided by 1,000 to convert to seconds.