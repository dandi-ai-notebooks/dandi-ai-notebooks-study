# %% [markdown]
# # Exploring Dandiset 000563: Allen Institute Openscope - Barcoding
#
# *Notebook generated with the assistance of AI. Please use caution when interpreting code or results, and verify procedures as needed.*
#
# ---
#
# ## Overview
#
# [Dandiset 000563](https://dandiarchive.org/dandiset/000563/0.250311.2145) contains Neuropixels extracellular electrophysiology data from mouse visual system, focusing on temporal precision of neural responses to white noise visual stimuli ("barcoding"). This unique dataset includes LFP, spike data, and comprehensive metadata, spanning multiple brain areas and includes innovative visual stimulation protocols.
#
# **Key facts:**
# - Species: Mus musculus (mouse)
# - Data: Neuropixels probe recordings (LFP, spikes), optogenetics, metadata
# - Format: NWB (Neurodata Without Borders)
#
# This notebook demonstrates:
# 1. Browsing Dandiset assets using the DANDI API
# 2. Streaming and reading NWB files without downloading
# 3. Exploring electrode metadata and probe layout
# 4. Visualizing example LFP data
#
# ---
#
# ## Requirements
#
# This notebook assumes you have the following Python packages installed:
# - numpy
# - pandas
# - matplotlib
# - pynwb
# - h5py
# - remfile
# - dandi
#
# Let's get started!
#
# # %% [markdown]
# ## List Dandiset Assets with the DANDI API
#
# Here we use the DANDI API to enumerate some files within this Dandiset. This helps reveal the structure and options for analysis.

# %%
from dandi.dandiapi import DandiAPIClient

dandiset_id = "000563"
dandiset_version = "0.250311.2145"
dandi_url = f"https://dandiarchive.org/dandiset/{dandiset_id}/{dandiset_version}"

print(f"Dandiset: {dandiset_id} (Version: {dandiset_version})")
print(f"Link: {dandi_url}\n")

with DandiAPIClient() as client:
    dandiset = client.get_dandiset(dandiset_id, dandiset_version)
    assets = list(dandiset.get_assets())
    nwb_assets = [a for a in assets if a.path.endswith('.nwb')]
    print("Example NWB files from this Dandiset:")
    for asset in nwb_assets[:10]:
        print("-", asset.path)

# %% [markdown]
# ---
# ## Streaming and Exploring an NWB File
#
# NWB files in this Dandiset can be hundreds of gigabytes. We'll stream a remote file using `remfile` and inspect its contents with `h5py` and `pynwb`. This workflow lets you preview and analyze large files without full download.
#
# We'll use:
# > `sub-699241/sub-699241_ses-1318772854_probe-1_ecephys.nwb`

# %%
import remfile
import h5py
import pynwb

target_path = "sub-699241/sub-699241_ses-1318772854_probe-1_ecephys.nwb"

# Use DANDI API to find direct download URL for the target NWB file
with DandiAPIClient() as client:
    dandiset = client.get_dandiset(dandiset_id, dandiset_version)
    asset = next(dandiset.get_assets_by_glob(target_path))
    url = asset.download_url
print(f"Streaming file: {url}")

# Open remote file for streaming
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# %% [markdown]
# ### Basic NWB Metadata

# %%
print("Session description:", nwb.session_description)
print("Session start time:", nwb.session_start_time)
print("Session ID:", nwb.session_id)
print("Institution:", nwb.institution)
print("Stimulus notes:", nwb.stimulus_notes)
print("File identifier:", nwb.identifier)
print("File creation date(s):", [d.isoformat() for d in nwb.file_create_date])
print("\nSubject info:")
print("  Subject ID:", nwb.subject.subject_id)
print("  Genotype:", nwb.subject.genotype)
print("  Age:", nwb.subject.age, f"({nwb.subject.age_in_days} days)")
print("  Sex:", nwb.subject.sex)
print("  Strain:", nwb.subject.strain)
print("  Species:", nwb.subject.species)

# %% [markdown]
# ---
# ## Examine Probe and Electrode Metadata
#
# Each NWB file contains information about the Neuropixels probe (device and electrode group) and a detailed electrode metadata table.

# %%
import pandas as pd

# Print device/probe info
for name, dev in nwb.devices.items():
    print(f"Device: {name}")
    print("  Description:", getattr(dev, "description", ""))
    print("  Manufacturer:", getattr(dev, "manufacturer", ""))
    print("  Probe ID:", getattr(dev, "probe_id", ""))
    print("  Sampling rate [Hz]:", getattr(dev, "sampling_rate", ""))

# Print electrode group info
print("\nElectrode Groups:")
for name, group in nwb.electrode_groups.items():
    print(f"  Group: {name}")
    print("    Location:", group.location)
    print("    Has LFP data:", getattr(group, "has_lfp_data", None))
    print("    Probe ID:", getattr(group, "probe_id", None))
    print("    LFP sampling rate [Hz]:", getattr(group, "lfp_sampling_rate", None))

# %% [markdown]
# ### Inspect Electrode Table
#
# The `electrodes` DynamicTable contains per-channel metadata including brain region, position, valid_data, and hardware filtering.

# %%
elec_df = nwb.electrodes.to_dataframe()
print(f"Electrodes table shape: {elec_df.shape}")
elec_df.head()

# %% [markdown]
# ### Visualize Electrode Geometry (x, y, z)
#
# Let's plot the Neuropixels array geometry for this probe to visualize the spatial relationship of electrodes.

# %%
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 4.5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(elec_df['x'], elec_df['y'], elec_df['z'])
ax.set_xlabel('x (posterior)')
ax.set_ylabel('y (inferior)')
ax.set_zlabel('z (right)')
ax.set_title('Electrode 3D Positions')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Load and Plot Example LFP Data
#
# The `acquisition` group contains LFP (`probe_1_lfp_data`). We will extract a short time window from several channels and plot them.

# %%
import numpy as np

lfp = nwb.acquisition["probe_1_lfp_data"]
print("LFP Data shape:", lfp.data.shape)
print("LFP Timestamp shape:", lfp.timestamps.shape)
print("LFP unit:", lfp.unit)

# Grab first 1 second for 5 channels
lfp_sr = nwb.electrode_groups['probeC'].lfp_sampling_rate  # 625 Hz
nsamp = int(lfp_sr * 1.0)
channels = np.arange(5)
lfp_snippet = lfp.data[:nsamp, channels]
lfp_times = lfp.timestamps[:nsamp]

plt.figure(figsize=(10, 5))
offset = 0
for i, ch in enumerate(channels):
    plt.plot(lfp_times, lfp_snippet[:, i]*1e3 + i*2, label=f'Ch {ch}')
plt.xlabel('Time (s)')
plt.ylabel('LFP (mV, offset by ch)')
plt.title('LFP from 5 channels, first second')
plt.legend(loc='lower right', fontsize='small')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Summary and Next Steps
#
# In this notebook, we:
# - Explored the organization of Dandiset 000563 on DANDI
# - Streamed an NWB file and inspected its experimental metadata
# - Examined probe configuration and visualized electrode geometry
# - Loaded and plotted an LFP data snippet
#
# **You are now ready to:**
# - Analyze longer time segments, specific brain regions, or LFP features
# - Explore additional probe files or investigate optogenetic/units data
# - Leverage this workflow for your own spike/LFP analyses or comparisons across sessions and probes
#
# See the [Dandiset page](https://dandiarchive.org/dandiset/000563/0.250311.2145) for documentation, links, and further scientific details.
#
# ---
# *End of notebook*