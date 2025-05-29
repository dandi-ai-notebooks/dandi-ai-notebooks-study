# %% [markdown]
# # Exploring Dandiset 001375: Septum GABA Disruption with DREADDs

# %% [markdown]
# > **Note**: This notebook was generated with the assistance of AI. Please be cautious when interpreting code or results.

# %% [markdown]
# ## Overview
# 
# This notebook explores Dandiset 001375, which contains data from a pilot study investigating the effects of disrupting septal GABAergic activity using DREADDs (Designer Receptors Exclusively Activated by Designer Drugs) on hippocampal and neocortical activity.
# 
# Dandiset URL: [https://dandiarchive.org/dandiset/001375/0.250406.1855](https://dandiarchive.org/dandiset/001375/0.250406.1855)
# 
# In this notebook, we will:
# 1. Explore the metadata of the Dandiset
# 2. List and examine the NWB files available
# 3. Load and analyze neural recording data
# 4. Visualize spike times and basic electrophysiological recordings
# 5. Examine trial structure and unit activity

# %% [markdown]
# ## Required Packages

# %%
# Import necessary libraries
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient
import pandas as pd

# %% [markdown]
# ## Exploring Dandiset Metadata

# %%
# Initialize DandiAPI client and access the Dandiset
client = DandiAPIClient()
dandiset = client.get_dandiset("001375", "0.250406.1855")
metadata = dandiset.get_raw_metadata()

# Display basic information about the Dandiset
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")
print(f"Description: {metadata['description']}")
print(f"License: {metadata['license'][0]}")
print(f"Version: {metadata['version']}")

# %% [markdown]
# ## Exploring Available NWB Files

# %%
# List the available NWB files in the Dandiset
assets = list(dandiset.get_assets())
nwb_files = [asset.path for asset in assets if asset.path.endswith('.nwb')]

print(f"Number of NWB files: {len(nwb_files)}")
for i, file in enumerate(nwb_files):
    print(f"{i+1}. {file}")

# %% [markdown]
# The Dandiset contains 3 NWB files from 2 subjects. Let's examine one of these files in more detail.

# %% [markdown]
# ## Loading and Exploring an NWB File

# %%
# We'll examine the first NWB file from subject MS13B
file_path = "sub-MS13B/sub-MS13B_ses-20240725T190000_ecephys.nwb"

# Get the download URL for the file
asset = next(dandiset.get_assets_by_glob(file_path))
url = asset.download_url

# Open the file using remfile to stream without downloading
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# ### Basic Information about the NWB File

# %%
# Extract and display basic metadata
print(f"NWB File: {file_path}")
print(f"Session Description: {nwb.session_description}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Identifier: {nwb.identifier}")

# Subject information
subject = nwb.subject
print("\nSubject Information:")
print(f"  ID: {subject.subject_id}")
print(f"  Age: {subject.age}")
print(f"  Sex: {subject.sex}")
print(f"  Species: {subject.species}")
print(f"  Description: {subject.description}")

# %% [markdown]
# ## Electrode and Device Information

# %%
# Examine device information
devices = nwb.devices
print("Devices:")
for device_name, device in devices.items():
    print(f"  {device_name}: {device.description}, Manufacturer: {device.manufacturer}")

# Examine electrode groups
electrode_groups = nwb.electrode_groups
print("\nElectrode Groups:")
for group_name, group in electrode_groups.items():
    print(f"  {group_name}: {group.description}")
    print(f"    Location: {group.location}")
    print(f"    Device: {group.device.description}")

# %% [markdown]
# ### Examining Electrodes Table

# %%
# Convert electrodes to dataframe for easier exploration
electrodes_df = nwb.electrodes.to_dataframe()

# Display basic information about electrodes
print(f"Number of electrodes: {len(electrodes_df)}")
print("First 5 electrodes:")
display(electrodes_df.head())

# Group information
print("\nElectrode group distribution:")
print(electrodes_df['group_name'].value_counts())

# %% [markdown]
# ## Neural Data Acquisition

# %%
# Examine acquisition data
acq = nwb.acquisition['time_series']
print(f"Data dimensions: {acq.data.shape}")
print(f"Sampling rate: {acq.rate} Hz")
print(f"Unit: {acq.unit}")
print(f"Data duration: {acq.data.shape[0] / acq.rate:.2f} seconds")

# %% [markdown]
# Let's visualize a short segment of the raw electrophysiology data from a few channels:

# %%
# Plot a short segment of raw data from 4 channels
fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# Define time window to plot (first 1 second)
time_window = 1  # seconds
sample_count = int(time_window * acq.rate)
time_vector = np.arange(sample_count) / acq.rate

# Select 4 channels to plot (from different areas if possible)
channels_to_plot = [0, 64, 128, 192]  # Example channels spread across the array

for i, channel in enumerate(channels_to_plot):
    # Get channel data
    channel_data = acq.data[:sample_count, channel]
    
    # Plot
    axs[i].plot(time_vector, channel_data, linewidth=1)
    axs[i].set_ylabel(f"Ch {channel}\n({acq.unit})")
    axs[i].grid(True, linestyle='--', alpha=0.7)
    
    # Add electrode location
    electrode_info = electrodes_df.iloc[channel]
    axs[i].set_title(f"Channel {channel}: Group {electrode_info['group_name']}")

axs[-1].set_xlabel("Time (seconds)")
plt.suptitle(f"Raw Electrophysiology Data - First {time_window} Second(s)", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Trials Information

# %%
# Examine trials information
trials_df = nwb.trials.to_dataframe()
print(f"Number of trials: {len(trials_df)}")
print("First 5 trials:")
display(trials_df.head())

# Calculate trial durations
trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']

# Plot trial durations
plt.figure(figsize=(12, 5))
plt.plot(trials_df.index, trials_df['duration'], 'o-', alpha=0.7)
plt.xlabel("Trial Number")
plt.ylabel("Duration (seconds)")
plt.title(f"Trial Durations (n={len(trials_df)})")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Distribution of trial durations
plt.figure(figsize=(10, 5))
plt.hist(trials_df['duration'], bins=30, alpha=0.7)
plt.xlabel("Duration (seconds)")
plt.ylabel("Count")
plt.title("Distribution of Trial Durations")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %% [markdown]
# ## Analyzing Unit Activity

# %%
# Examine units
units_df = nwb.units.to_dataframe()
print(f"Number of units: {len(units_df)}")
print("Unit information:")
display(units_df.head())

# %% [markdown]
# Let's examine the spike times for a few units:

# %%
# Plot spike raster for multiple units
fig, ax = plt.subplots(figsize=(14, 8))

# Get the first 10 units (or all if less than 10)
num_units_to_plot = min(10, len(units_df))
unit_ids = list(range(num_units_to_plot))

# Plot spikes for each unit
for i, unit_id in enumerate(unit_ids):
    # Get spike times for this unit
    spike_times = nwb.units.spike_times_index[unit_id]
    
    # Plot spike times as dots
    ax.plot(spike_times, np.ones_like(spike_times) * i, '|', markersize=5, 
            label=f"Unit {unit_id}")

# Customize plot
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Unit")
ax.set_yticks(range(num_units_to_plot))
ax.set_yticklabels([f"Unit {i}" for i in unit_ids])
ax.set_title("Spike Raster Plot")
ax.grid(True, axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Analyzing Spiking Activity During Trials

# %%
# Let's analyze the firing rate for one unit across trials
selected_unit = 0
spike_times = nwb.units.spike_times_index[selected_unit]

# Calculate firing rates for each trial
trial_rates = []
trial_midpoints = []

for _, trial in trials_df.iterrows():
    start = trial['start_time']
    end = trial['stop_time']
    duration = end - start
    
    # Count spikes in this trial
    spikes_in_trial = np.sum((spike_times >= start) & (spike_times < end))
    
    # Calculate firing rate
    rate = spikes_in_trial / duration
    trial_rates.append(rate)
    
    # Trial midpoint for plotting
    trial_midpoints.append((start + end) / 2)

# Plot firing rates across trials
plt.figure(figsize=(12, 5))
plt.plot(trial_midpoints, trial_rates, 'o-', alpha=0.7)
plt.xlabel("Time (seconds)")
plt.ylabel("Firing Rate (spikes/second)")
plt.title(f"Unit {selected_unit} Firing Rate Across Trials")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %% [markdown]
# ## Visualizing Spike Waveforms in Raw Data

# %%
# Let's extract raw data around spike times to visualize average waveforms
def extract_waveforms(data, spike_times, sampling_rate, window_ms=2, channel=0):
    """Extract waveforms around spike times"""
    samples_per_ms = sampling_rate / 1000
    window_samples = int(window_ms * samples_per_ms)
    
    spike_indices = (spike_times * sampling_rate).astype(int)
    waveforms = []
    
    for spike_idx in spike_indices:
        if spike_idx - window_samples >= 0 and spike_idx + window_samples < data.shape[0]:
            waveform = data[spike_idx - window_samples:spike_idx + window_samples, channel]
            waveforms.append(waveform)
            
    return np.array(waveforms)

# Select a unit and channel for visualization
selected_unit = 0
selected_channel = 0  # Assuming spikes are most visible on first channel

# Get spike times and extract waveforms
spike_times = nwb.units.spike_times_index[selected_unit]

# Limit to first 100 spikes to avoid memory issues
spike_times = spike_times[:100] 

waveforms = extract_waveforms(acq.data, spike_times, acq.rate, window_ms=2, channel=selected_channel)

# Plot average waveform with individual traces
plt.figure(figsize=(10, 6))
time_ms = np.linspace(-2, 2, waveforms.shape[1])

# Plot individual waveforms (semi-transparent)
for waveform in waveforms:
    plt.plot(time_ms, waveform, 'k-', alpha=0.1)

# Plot average waveform
mean_waveform = np.mean(waveforms, axis=0)
std_waveform = np.std(waveforms, axis=0)
plt.plot(time_ms, mean_waveform, 'r-', linewidth=2, label='Mean')

# Add confidence interval
plt.fill_between(
    time_ms, 
    mean_waveform - std_waveform, 
    mean_waveform + std_waveform, 
    color='r', alpha=0.2, label='±1 SD'
)

plt.axvline(x=0, color='gray', linestyle='--')
plt.xlabel('Time (ms)')
plt.ylabel(f'Amplitude ({acq.unit})')
plt.title(f'Spike Waveforms for Unit {selected_unit} (n={len(waveforms)})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %% [markdown]
# ## Summary and Next Steps

# %% [markdown]
# In this notebook, we've explored Dandiset 001375, which contains electrophysiology data from mice with DREADD-mediated disruption of septal GABAergic neurons. We've examined:
# 
# 1. The structure of the NWB files and metadata
# 2. Electrode configuration and recording details
# 3. Raw electrophysiological data
# 4. Trial structures during behavioral tasks
# 5. Unit firing patterns and waveform characteristics
# 
# Potential future analyses could include:
# - Cross-session comparisons of neural activity
# - More detailed spike-train analysis (e.g., ISI distributions)
# - Correlating neural activity with trial features
# - Comparing activity patterns across brain regions