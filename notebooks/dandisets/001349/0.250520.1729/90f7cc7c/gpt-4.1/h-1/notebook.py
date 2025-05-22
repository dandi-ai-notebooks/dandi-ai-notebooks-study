# %% [markdown]
# # Exploring Dandiset 001349: GABAergic Inputs Shape Neuronal Subnetworks in Developing Somatosensory Cortex
#
# *This notebook was generated with the assistance of AI. Please be cautious when interpreting the code or results, and double-check all outputs and analyses before drawing scientific conclusions.*

# %% [markdown]
# ## Overview
#
# This notebook introduces **Dandiset 001349**:
#
# **Title**: "From Initial Formation to Developmental Refinement: GABAergic Inputs Shape Neuronal Subnetworks in the Primary Somatosensory Cortex"
#
# - **Scientific Focus**: Calcium imaging in the developing mouse somatosensory cortex, examining the roles of GABAergic inputs.
# - **Data**: Two-photon calcium imaging (raw and processed), segmentation masks for ROIs, and detailed experiment/session metadata.
# - **Subjects**: Developing mice, P11–P21.
# - **Link:** [View on DANDI Archive](https://dandiarchive.org/dandiset/001349/0.250520.1729)
#
# ## What This Notebook Covers
#
# - How to use the DANDI API to explore and list assets in this Dandiset
# - How to stream and load NWB files remotely
# - How to list and explore ROI-related time series (e.g., dff, raw fluorescence)
# - How to visualize traces for a single ROI
# - How to plot the spatial segmentation mask for a ROI
#
# This notebook demonstrates key steps to get started with a reanalysis of processed calcium imaging data from Dandiset 001349.

# %% [markdown]
# ## Required Packages
#
# This notebook assumes the following packages are available in your environment:
# - `dandi`
# - `remfile`
# - `h5py`
# - `pynwb`
# - `numpy`
# - `matplotlib`
# - `pandas`
#
# No package installation commands are included below.

# %% [markdown]
# ## 1. Using the DANDI API to Explore NWB Assets

# %%
from dandi.dandiapi import DandiAPIClient
from itertools import islice

# Connect to the DANDI API and access the Dandiset
dandiset_id = "001349"
dandiset_version = "0.250520.1729"
client = DandiAPIClient()
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# List the first 10 NWB files in this Dandiset
assets = dandiset.get_assets_by_glob("*.nwb")
first_10 = list(islice(assets, 10))
print("First 10 NWB files in Dandiset 001349:")
for asset in first_10:
    print(asset.path)

# %% [markdown]
# For the remainder of this notebook, we will focus on the file:
#
# `sub-C57-C2-2-AL/sub-C57-C2-2-AL_ses-2_ophys.nwb`
#
# This file contains multiple types of processed calcium imaging traces and segmentation data for 143 ROIs.

# %% [markdown]
# ## 2. Loading the NWB File Remotely and Listing ROI Response TimeSeries

# %%
import remfile
import h5py
import pynwb
from pprint import pprint

# Specify the NWB asset path of interest
asset_path = "sub-C57-C2-2-AL/sub-C57-C2-2-AL_ses-2_ophys.nwb"
asset = next(dandiset.get_assets_by_glob(asset_path))

# Open the NWB file by streaming from the remote URL
remote_file = remfile.File(asset.download_url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# List the available ROI response TimeSeries objects in 'Fluorescence'
processing = nwb.processing['ophys']
fluorescence = processing.data_interfaces['Fluorescence']
roi_response_series = fluorescence.roi_response_series
print("Available ROI Response TimeSeries in 'Fluorescence':")
pprint(list(roi_response_series.keys()))

# %% [markdown]
# The ROI response traces available for this session include:
# - `ca_events_chn0`: Calcium event traces
# - `dff_chn0`: ΔF/F (normalized fluorescence) traces
# - `f_corrected_chn0`: Motion/noise-corrected fluorescence
# - `fluorescence_chn0`: Raw fluorescence signals
# - `neuropil_fluorescence_chn0`: Neuropil-corrected fluorescence
# - `z_score_chn0`: Z-scored traces

# %% [markdown]
# ## 3. Plotting ΔF/F (dff) Trace for ROI 0

# %%
import numpy as np
import matplotlib.pyplot as plt

# Extract ΔF/F data for ROI 0 (index 0)
dff_chn0 = fluorescence.roi_response_series['dff_chn0']
dff_data = dff_chn0.data[:, 0]
rate = dff_chn0.rate  # Hz
times = np.arange(len(dff_data)) / rate

# Plot the trace
plt.figure(figsize=(10, 4))
plt.plot(times, dff_data, label='ROI 0')
plt.xlabel('Time (s)')
plt.ylabel('ΔF/F (a.u.)')
plt.title('ΔF/F Trace (dff_chn0) for ROI 0')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# The plot shows the normalized ΔF/F calcium activity for ROI 0 across the session, revealing prominent transients.

# %% [markdown]
# ## 4. Visualizing the Spatial Footprint (Mask) of ROI 0

# %%
# Access the segmentation (PlaneSegmentation) interface
img_seg = processing.data_interfaces['ImageSegmentation']
plane_seg = img_seg.plane_segmentations['PlaneSegmentation_1']

# Extract the pixel mask for ROI 0
rois_df = plane_seg.to_dataframe()
pixel_mask = rois_df.iloc[0]['pixel_mask']  # list of (x, y, value)

# Determine mask image dimensions and create mask array
x = [int(p[0]) for p in pixel_mask]
y = [int(p[1]) for p in pixel_mask]
mask_shape = (max(y) + 1, max(x) + 1)
mask_img = np.zeros(mask_shape, dtype=float)
for px, py, val in pixel_mask:
    mask_img[int(py), int(px)] = val

# Plot the spatial footprint
plt.figure(figsize=(5, 5))
plt.imshow(mask_img, cmap='viridis', origin='lower')
plt.colorbar(label='Mask Value')
plt.title('Spatial Footprint (Mask) of ROI 0')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.tight_layout()
plt.show()

# %% [markdown]
# The image above shows the spatial extent of ROI 0 in pixel coordinates within the imaging field.

# %% [markdown]
# ## 5. Plotting the Raw Fluorescence Trace for ROI 0

# %%
# Extract raw fluorescence time series for ROI 0
fluorescence_chn0 = fluorescence.roi_response_series['fluorescence_chn0']
fluor_data = fluorescence_chn0.data[:, 0]
fluor_times = np.arange(len(fluor_data)) / fluorescence_chn0.rate

# Plot the raw fluorescence trace
plt.figure(figsize=(10, 4))
plt.plot(fluor_times, fluor_data, label='ROI 0')
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.)')
plt.title('Raw Fluorescence Trace (fluorescence_chn0) for ROI 0')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# The plot displays the raw, unprocessed fluorescence for ROI 0, showing both baseline fluctuations and activity-related transients.

# %% [markdown]
# ## 6. Checking for Behavioral or Stimulus-Related Data

# %%
# Check for behavioral or stimulus-related data outside "ophys"
# We'll print informative non-empty top-level NWB fields (excluding 'processing' and file creation date)
reported = False
for key in nwb.fields:
    if key not in ["processing", "file_create_date"]:
        val = getattr(nwb, key)
        # Only report if it's a container with non-trivial content
        if hasattr(val, "__len__"):
            try:
                if len(val) > 0:
                    print(f"Non-ophys NWB field '{key}' contains data.")
                    reported = True
            except TypeError:
                # Object does not support len(), skip
                pass

if not reported:
    print("No behavioral or stimulus-related time series found outside the ophys processing module in this NWB file.")

# %% [markdown]
# **Summary:**  
# - This NWB file contains processed calcium imaging traces and segmentation masks, with detailed session and imaging metadata.
# - No behavioral or stimulus-related time series are present outside the ophys module.
# - The notebook showed how to list, load, and visualize calcium activity traces and ROI segmentation data for exploratory analysis.
#
# For more details and to explore other sessions or ROIs, see the full [Dandiset 001349 archive](https://dandiarchive.org/dandiset/001349/0.250520.1729).
#
# *Always refer to the original data documentation and publication when interpreting or reanalyzing shared neuroscience datasets!*