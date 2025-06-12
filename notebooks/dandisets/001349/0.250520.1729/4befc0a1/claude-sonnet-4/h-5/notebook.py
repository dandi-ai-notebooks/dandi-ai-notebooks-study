# %% [markdown]
# # Exploring Dandiset 001349: From Initial Formation to Developmental Refinement: GABAergic Inputs Shape Neuronal Subnetworks in the Primary Somatosensory Cortex

# %% [markdown]
# **Note:** This notebook was generated with the assistance of AI. Please be cautious when interpreting the code or results and verify findings independently.

# %% [markdown]
# ## Overview
# 
# This Dandiset contains two-photon calcium imaging data acquired in the primary somatosensory cortex of developing mice (postnatal days P11–P21). The dataset includes synchronized behavioral video recordings, with imaging and behavior synchronized using LED signals. The data is stored in Neurodata Without Borders (NWB) format and includes processed fluorescence traces, calcium events, and ROI segmentation data.
# 
# **Dandiset Link:** https://dandiarchive.org/dandiset/001349/0.250520.1729
# 
# **Key Details:**
# - **Species:** Mus musculus (House mouse)
# - **Modalities:** Two-photon calcium imaging, synchronized behavioral video
# - **Number of subjects:** 32
# - **Number of files:** 361
# - **Total size:** ~9 GB

# %% [markdown]
# ## Required Packages
# 
# This notebook assumes the following packages are installed:
# - `dandi`
# - `pynwb` 
# - `h5py`
# - `remfile`
# - `numpy`
# - `matplotlib`
# - `pandas`

# %%
# Import necessary packages
import pynwb
import h5py
import remfile
from dandi.dandiapi import DandiAPIClient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# ## Explore Dandiset Metadata

# %%
# Connect to DANDI and get dandiset information
client = DandiAPIClient()
dandiset = client.get_dandiset("001349", "0.250520.1729")
metadata = dandiset.get_raw_metadata()

print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")
print(f"Description: {metadata['description']}")

# %% [markdown]
# ## Explore NWB Files in the Dandiset

# %%
# List the first 20 NWB files to understand file organization
from itertools import islice

assets = dandiset.get_assets_by_glob("*.nwb")
asset_list = list(islice(assets, 20))

# Create summary table of files
file_info = []
for asset in asset_list:
    file_info.append({
        "path": asset.path,
        "size_MB": round(asset.size / 1e6, 2)
    })

file_df = pd.DataFrame(file_info)
print("Sample of NWB files in the dandiset:")
print(file_df)

# %% [markdown]
# The files are organized by subject and session, with naming convention `sub-{subject_id}/sub-{subject_id}_ses-{session}_ophys.nwb`. File sizes range from ~19-45 MB, indicating substantial imaging datasets for each session.

# %% [markdown]
# ## Load and Examine a Representative NWB File

# %%
# Load a representative NWB file remotely
asset_path = "sub-C57-C2-2-AL/sub-C57-C2-2-AL_ses-2_ophys.nwb"
url = next(dandiset.get_assets_by_glob(asset_path)).download_url

# Stream the file remotely without downloading
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# ## Extract Session and Subject Metadata

# %%
# Collect key metadata from the NWB file
metadata_info = {
    'Session description': nwb.session_description,
    'Session start time': str(nwb.session_start_time),
    'Experimenter': nwb.experimenter,
    'Institution': getattr(nwb, 'institution', ''),
    'Data collection': getattr(nwb, 'data_collection', ''),
    'Subject ID': nwb.subject.subject_id,
    'Species': getattr(nwb.subject, 'species', ''),
    'Sex': getattr(nwb.subject, 'sex', ''),
    'Age': getattr(nwb.subject, 'age', ''),
    'Date of birth': str(getattr(nwb.subject, 'date_of_birth', '')),
    'Imaging rate': nwb.imaging_planes['ImagingPlane_1_chn1'].imaging_rate,
    'Imaging plane location': nwb.imaging_planes['ImagingPlane_1_chn1'].location,
    'Device': nwb.imaging_planes['ImagingPlane_1_chn1'].device.description,
}

# Display as table
metadata_df = pd.DataFrame(list(metadata_info.items()), columns=['Field', 'Value'])
print("Session and Subject Metadata:")
print(metadata_df.to_string(index=False))

# %% [markdown]
# ## Understand ROI Segmentation Structure

# %%
# Access ROI segmentation data
ophys = nwb.processing['ophys']
seg = ophys.data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation_1']

# Get number of segmented ROIs (cells)
num_rois = seg.id.data.shape[0]
print(f"Number of segmented ROIs (cells): {num_rois}")

# %% [markdown]
# This session contains **143 segmented ROIs (cells)**, each with associated pixel masks and fluorescence traces. The segmentation data is stored in NWB's PlaneSegmentation structure, allowing access to both spatial (pixel masks) and temporal (fluorescence traces) information.

# %% [markdown]
# ## Explore Available Signal Types

# %%
# List all available RoiResponseSeries (different signal types)
roi_response_series = ophys.data_interfaces['Fluorescence'].roi_response_series
print("Available signal types:")
for name, series in roi_response_series.items():
    print(f"- {name}: {series.description}")
    print(f"  Shape: {series.data.shape}, Rate: {series.rate} Hz")

# %% [markdown]
# The file contains multiple processed signal types:
# - **dff_chn0**: ΔF/F normalized fluorescence (standard for relative activity changes)
# - **ca_events_chn0**: Calcium event detections (discrete events)
# - **f_corrected_chn0**: Background-corrected fluorescence
# - **fluorescence_chn0**: Raw fluorescence traces
# - **neuropil_fluorescence_chn0**: Background/neuropil signal
# - **z_score_chn0**: Z-scored fluorescence traces

# %% [markdown]
# ## Visualize ΔF/F Fluorescence Traces

# %%
# Extract and plot ΔF/F traces for first 10 cells
dff = roi_response_series['dff_chn0']
dff_data = dff.data[:1000, :10]  # First 1000 time points, first 10 cells
rate = dff.rate

# Create time vector
t = np.arange(dff_data.shape[0]) / rate

# Plot traces
plt.figure(figsize=(10, 6))
for i in range(dff_data.shape[1]):
    plt.plot(t, dff_data[:, i], label=f'Cell {i+1}')

plt.xlabel('Time (s)')
plt.ylabel('ΔF/F (a.u.)')
plt.title('ΔF/F traces: first 10 cells, first 1000 time points')
plt.tight_layout()
plt.show()

print(f"Data shape: {dff_data.shape} (time points × cells)")

# %% [markdown]
# ## Visualize Calcium Event Traces

# %%
# Extract and plot calcium event traces
caev = roi_response_series['ca_events_chn0']
caev_data = caev.data[:1000, :10]  # First 1000 time points, first 10 cells

# Plot event traces with small vertical offset for clarity
plt.figure(figsize=(10, 6))
for i in range(caev_data.shape[1]):
    plt.plot(t, caev_data[:, i] + i*0.03, label=f'Cell {i+1}')

plt.xlabel('Time (s)')
plt.ylabel('Ca Event Value (offset per cell)')
plt.title('Calcium Events: first 10 cells, first 1000 time points')
plt.tight_layout()
plt.show()

# %% [markdown]
# The calcium event traces show discrete, event-like activity patterns derived from the fluorescence data. These processed signals are ideal for event rate analysis, synchronization studies, and population decoding approaches.

# %% [markdown]
# ## Data Access Summary

# %% [markdown]
# ### Key code patterns for data access:
# 
# ```python
# # Access different signal types
# roi_resp = nwb.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series
# dff = roi_resp['dff_chn0'].data           # ΔF/F traces
# ca_events = roi_resp['ca_events_chn0'].data  # Calcium events
# fluor = roi_resp['fluorescence_chn0'].data   # Raw fluorescence
# 
# # Get ROI segmentation info
# seg = nwb.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation_1']
# num_cells = seg.id.data.shape[0]
# 
# # Access behavioral sync signals (check acquisition)
# sync_signals = list(nwb.acquisition)  # May contain LED triggers
# ```

# %% [markdown]
# ## Next Steps for Analysis
# 
# This notebook provides the foundation for exploring this developmental two-photon imaging dataset. Key analysis directions include:
# 
# - **Developmental dynamics**: Compare activity patterns across ages (P11-P21)
# - **Population analysis**: Examine cell-cell correlations and network structure
# - **Behavioral alignment**: Use LED sync signals to relate neural activity to behavior
# - **Event analysis**: Study calcium event rates, timing, and population synchrony
# 
# The standardized NWB format allows these analysis approaches to be applied across all sessions and subjects in the dataset.