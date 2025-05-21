# %% [markdown]
# # Exploring Dandiset 000563: Allen Institute Openscope - Barcoding
# 
# *This notebook was generated with the assistance of AI. Please review all code and results carefully, and exercise caution before drawing any conclusions or reusing code in a research setting.*
# 
# ---
# 
# ## Overview
# 
# This notebook introduces and demonstrates how to begin working with data from **Dandiset 000563: Allen Institute Openscope - Barcoding**, available at  
# [https://dandiarchive.org/dandiset/000563/0.250311.2145](https://dandiarchive.org/dandiset/000563/0.250311.2145)
# 
# This Dandiset provides electrophysiological data (LFP, spike sorting, optogenetic, behavioral, and more) from mouse visual cortex and associated structures, with a focus on temporally precise neural coding of visual and optogenetic stimuli.
# 
# ---
# 
# ## What this notebook covers
# - DANDI API basics: Enumerating NWB files and exploring assets
# - Loading NWB files using remote streaming (without full download)
# - Exploring and visualizing LFP time series data
# - Exploring spike-sorting outputs with the `units` table, including spike raster plots
# - Viewing optogenetic stimulation intervals and metadata
# - Accessing and plotting behavioral data (running speed, eye tracking)
# 
# The examples use only the methods and techniques demonstrated in the AI-assisted Q&A above.
# 
# ---
# 
# ## Required packages
# 
# This notebook requires the following Python packages (assumed pre-installed):
# 
# - `dandi` (DANDI API client)
# - `pynwb` (for NWB data streaming and access)
# - `remfile` (for HTTP streaming)
# - `h5py` (backend for hdf5)
# - `matplotlib` and `numpy` (for data handling and plotting)
# - `pandas` (for table access)
# 
# Please make sure these are installed in your environment!
# 
# ---

# %%
# Import required packages for all examples
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## List available NWB files in the Dandiset
# 
# Use the DANDI API to enumerate some of the files available in this Dandiset.

# %%
# Connect to the DANDI archive and list NWB files (first 10 for brevity)
dandiset_id = "000563"
dandiset_version = "0.250311.2145"
client = DandiAPIClient()
dandiset = client.get_dandiset(dandiset_id, dandiset_version)
assets = dandiset.get_assets_by_glob("*.nwb")
file_list = []
for i, asset in enumerate(assets):
    if i >= 10:
        break
    file_list.append(asset.path)
print("Example NWB files in this Dandiset:")
for fname in file_list:
    print("-", fname)

# %% [markdown]
# ## Load and summarize a specific NWB file
# 
# We will focus on `sub-699241/sub-699241_ses-1318772854_ogen.nwb`, which contains both spike sorting and optogenetic data, as well as behavioral and stimulus metadata.

# %%
# Locate and prepare the NWB asset for streaming
nwb_asset_path = "sub-699241/sub-699241_ses-1318772854_ogen.nwb"
asset = next(dandiset.get_assets_by_glob(nwb_asset_path))
download_url = asset.download_url

# Stream the NWB file using remfile and h5py
remote_file = remfile.File(download_url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# Print some high-level info
print("Session description:", nwb.session_description)
print("Session start time:", nwb.session_start_time)
print("Subject:", vars(nwb.subject) if nwb.subject else None)
print("Acquisition keys:", list(nwb.acquisition.keys()))

# %% [markdown]
# ## Explore spike-sorted data: the units table
# 
# The `units` table contains spike times, amplitudes, waveform features, and quality metrics for each sorted unit. We'll display the available columns and show the first 5 rows.

# %%
# Access the units table
units_table = nwb.units
units_columns = units_table.colnames
units_df = units_table.to_dataframe().head()  # First 5 rows

print("Units table columns:")
print(units_columns)
print("\nFirst 5 rows of the units table:")
print(units_df)

# %% [markdown]
# ## Visualize spike times for a single unit
# 
# We'll plot the spike raster (event plot) for unit id 15, zoomed to a 30-second window starting from the first detected spike.
# 

# %%
# Extract spike times for unit id 15 from the units DataFrame
unit_id = 15
spike_times_all = nwb.units.to_dataframe().loc[unit_id, 'spike_times']

# Select a 30-second window after the first spike
window_start = float(np.min(spike_times_all))
window_end = window_start + 30
in_window = [t for t in spike_times_all if window_start <= t <= window_end]

# Plot a raster of spike events in this window
plt.figure(figsize=(10, 2))
plt.eventplot(in_window, orientation='horizontal', colors='k')
plt.xlabel('Time (s)')
plt.yticks([])
plt.title(f'Spike times for unit id {unit_id}: {window_start:.2f}–{window_end:.2f} s')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summarize optogenetic stimulation intervals
# 
# The optogenetic stimulation epochs are found in the `optotagging` processing module as the `optogenetic_stimulation` interface.

# %%
# Access the optogenetic stimulation table
optotag_mod = nwb.processing['optotagging']
optogenetic_stim = optotag_mod.data_interfaces['optogenetic_stimulation']
stim_cols = optogenetic_stim.colnames
stim_rows = optogenetic_stim.to_dataframe().head()
print("Optogenetic stimulation columns:")
print(stim_cols)
print("\nFirst 5 rows of the table:")
print(stim_rows)

# %% [markdown]
# ## Explore behavioral data: running speed
# 
# We can plot 60 seconds of running speed data from the `running_speed` interface in the `running` processing module.

# %%
# Access and plot running speed for the first 60 seconds
running_mod = nwb.processing['running']
running_speed = running_mod.data_interfaces['running_speed']
rs_timestamps = running_speed.timestamps[:]
rs_data = running_speed.data[:]
rs_start = rs_timestamps[0]
rs_end = rs_start + 60
rs_idx = np.where((rs_timestamps >= rs_start) & (rs_timestamps <= rs_end))[0]

plt.figure(figsize=(10, 4))
plt.plot(rs_timestamps[rs_idx], rs_data[rs_idx], lw=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Running speed (cm/s)')
plt.title('Running speed (first 60 seconds of acquisition)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Explore behavioral data: eye tracking coordinates
# 
# The `EyeTracking` group in acquisition contains several time series, including `corneal_reflection_tracking`. We'll plot the X and Y coordinates for the first 10 seconds from the start of this stream.

# %%
eye_tracking_ts = nwb.acquisition['EyeTracking'].spatial_series['corneal_reflection_tracking']
et_data = eye_tracking_ts.data[:]
et_timestamps = eye_tracking_ts.timestamps[:]
et_start = et_timestamps[0]
et_end = et_start + 10
et_idx = np.where((et_timestamps >= et_start) & (et_timestamps <= et_end))[0]

plt.figure(figsize=(10, 5))
plt.plot(et_timestamps[et_idx], et_data[et_idx, 0], label='X coordinate')
plt.plot(et_timestamps[et_idx], et_data[et_idx, 1], label='Y coordinate')
plt.xlabel('Time (s)')
plt.ylabel('Position (meters)')
plt.title('Corneal reflection coordinates (first 10 seconds)')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Next steps
# 
# This notebook gives you a head start in remote exploration of Dandiset 000563. You can follow this pattern to:
# - Visualize LFP or other extracellular traces by browsing acquisition keys
# - Drill down into any behavioral or stimulus intervals table, or relate events to spike/behavioral data
# - Explore additional files and sessions in this Dandiset (see the file listing above)
# 
# **Remember:** Inspect all NWB content and metadata before publication or further analysis; review the full Dandiset documentation for scientific context.
# 
# ---
# 
# *For more in-depth exploration, see [DANDI Dandiset 000563](https://dandiarchive.org/dandiset/000563/0.250311.2145).*