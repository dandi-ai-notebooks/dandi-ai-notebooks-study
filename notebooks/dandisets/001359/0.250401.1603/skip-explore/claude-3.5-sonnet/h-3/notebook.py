# %% [markdown]
# # Exploring Dandiset 001359: Human Patch-seq Recordings from Allen Institute

# %% [markdown]
# **Note**: This notebook was generated with AI assistance. Please verify all code and results independently.

# %% [markdown]
# ## Overview
# This notebook explores Dandiset 001359 ([link](https://dandiarchive.org/dandiset/001359/0.250401.1603)), which contains patch-clamp recordings from human neurons collected at the Allen Institute for Brain Science. The data includes both voltage-clamp and current-clamp recordings, along with associated stimulus protocols.

# %% [markdown]
# This notebook demonstrates:
# - Accessing the Dandiset using the DANDI API
# - Loading and examining NWB file structure
# - Visualizing electrophysiology recordings
# - Analyzing spike timing data

# %% [markdown]
# ## Required Packages

# %%
import pynwb
import h5py
import remfile
from dandi.dandiapi import DandiAPIClient
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Loading Data from DANDI

# %%
# Initialize DANDI API client
client = DandiAPIClient()
dandiset = client.get_dandiset("001359", "0.250401.1603")

# Get example NWB file
url = next(dandiset.get_assets_by_glob("sub-1203384279/sub-1203384279_ses-1207984257_icephys.nwb")).download_url

# Open the file for streaming access
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# ## Data Structure Overview
# The NWB file contains:
# - Voltage-clamp recordings
# - Current-clamp recordings
# - Stimulus protocols
# - Spike timing data
# - Experimental epochs

# %% [markdown]
# ## Examining Recording Data

# %%
# Plot example voltage-clamp recording
def plot_voltage_clamp_data(series, title):
    data = series.data[:]
    time = np.arange(len(data)) / series.rate + series.starting_time
    
    plt.figure(figsize=(12, 4))
    plt.plot(time, data, 'b-', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel(f'Current ({series.unit})')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Get example voltage clamp recording
vc_series = nwb.acquisition['data_00000_AD0']
plot_voltage_clamp_data(vc_series, 'Example Voltage-Clamp Recording')

# %% [markdown]
# ## Analyzing Spike Timing
# The processing module contains spike timing information for different sweeps.

# %%
# Plot spike times from an example sweep
def plot_spike_times(sweep):
    spike_times = sweep.timestamps[:]
    
    plt.figure(figsize=(12, 2))
    plt.eventplot(spike_times, lineoffsets=0, linelengths=1, color='black')
    plt.xlabel('Time (s)')
    plt.title(f'Spike Times - {sweep.name}')
    plt.grid(True)
    plt.show()

# Get example sweep with spikes
spikes_module = nwb.processing['spikes']
sweep_19 = spikes_module.data_interfaces['Sweep_19']
plot_spike_times(sweep_19)

# %% [markdown]
# ## Experimental Epochs

# %%
# Convert epochs to DataFrame and show summary
epochs_df = nwb.epochs.to_dataframe()
print(f"Number of epochs: {len(epochs_df)}")
print("\nEpoch duration statistics (seconds):")
durations = epochs_df['stop_time'] - epochs_df['start_time']
print(durations.describe())

# Plot epoch durations
plt.figure(figsize=(12, 4))
plt.hist(durations, bins=50)
plt.xlabel('Epoch Duration (s)')
plt.ylabel('Count')
plt.title('Distribution of Epoch Durations')
plt.grid(True)
plt.show()

# %% [markdown]
# ## Subject Information

# %%
# Print subject metadata
print(f"Species: {nwb.subject.species}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")
print(f"Subject ID: {nwb.subject.subject_id}")

# %% [markdown]
# ## Next Steps
# Potential analyses with this dataset:
# - Compare spike patterns across different sweeps
# - Analyze response properties to different stimulus protocols 
# - Investigate cellular properties through voltage-clamp recordings
# - Study temporal patterns in experimental epochs