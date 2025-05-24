# %% [markdown]
# # Exploring Dandiset 000617: Allen Institute Openscope - Sequence Learning Project
#
# *Generated with the assistance of AI. Please review code and results carefully before drawing scientific conclusions.*
#
# ---
#
# This notebook provides an introduction and exploration guide for [Dandiset 000617 (version 0.250311.1615)](https://dandiarchive.org/dandiset/000617/0.250311.1615):  
# **Allen Institute Openscope - Sequence Learning Project**.
#
# This Dandiset contains extensive two-photon calcium imaging data from mouse neocortex acquired during a sequence learning task. Mice were presented with sequences of natural movie clips while imaging was performed across multiple visual cortical areas and layers. The data includes:
#
# - Processed calcium imaging (fluorescence, dF/F, events)
# - ROI segmentation and mask info
# - Raw and motion-corrected movies
# - Pupil and eye-tracking data
# - Running speed and behavioral events
# - Detailed stimulus presentation timing
#
# This notebook shows how to access and explore these datasets directly from the DANDI Archive, including a detailed walkthrough using a representative NWB file.
#
# ---
#
# **What this notebook covers:**
#
# 1. Quick summary of the Dandiset and experiment design.
# 2. Overview of required Python packages.
# 3. How to use the DANDI API to list and access available assets.
# 4. How to stream and load NWB data (without full download).
# 5. How to extract and visualize example data:
#     - Calcium imaging ROIs and fluorescence traces
#     - ROI geometric info and masks
#     - Running and behavioral data
#     - Stimulus intervals/excerpts
#     - Eye and pupil tracking
#
# Explanatory remarks and code comments are included throughout.  
# _Let's get started!_
# 
# ---
# %% [markdown]
# ## Required packages
#
# This notebook assumes you have the following packages installed:
#
# - `dandi` (for DANDI API access)
# - `remfile` (for remote file streaming)
# - `h5py` (to access NWB files as HDF5)
# - `pynwb` (to interact with NWB file contents)
# - `matplotlib` and `seaborn` (for plotting)
# - `pandas`, `numpy` (for data handling)
#
# No installation commands are included (see the [DANDI documentation](https://www.dandiarchive.org/) for more information).
#
# ---
# %% [markdown]
# ## 1. Dandiset Summary
#
# - **Project title:** Allen Institute Openscope - Sequence Learning Project
# - **Organisms:** Mus musculus (house mouse)
# - **Sample size:** 13 subjects, >1000 files, >13 TB of data
# - **Approaches:** Two-photon mesoscope calcium imaging, Eye tracking, Running wheel, Visual stimuli
# - **Areas:** V1 (VISp), LM (VISl), AM (VISam), PM (VISpm); Cortical layers 2/3 and 4
# - **Genotype:** Cux2-CreERT2;Camk2a-tTA;Ai93(TITL-GCaMP6f)
#
# _For more details, see the full Dandiset record:_  
# [https://dandiarchive.org/dandiset/000617/0.250311.1615](https://dandiarchive.org/dandiset/000617/0.250311.1615)
# 
# ---
# %% [markdown]
# ## 2. Connecting to the DANDI Archive and Locating Example NWB Files
# 
# We'll use the DANDI API to query the Dandiset, list files, and select one sample NWB file for exploration.
# 
# ---
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dandi.dandiapi import DandiAPIClient

# Connect to the DANDI archive and Dandiset 000617 (specific version)
dandiset_id = "000617"
dandiset_version = "0.250311.1615"

client = DandiAPIClient()
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# List a subset of files for subject 684475
print("First few NWB files for subject 684475:")
file_listing = list(dandiset.get_assets_by_glob("sub-684475/*.nwb"))
for f in file_listing[:5]:
    print("-", f.path)
# %% [markdown]
# _**Note:**_ File names in this dandiset follow the convention:  
# `sub-{SUBJECT_ID}/sub-{SUBJECT_ID}_ses-{SESSION_ID}-acq-{ACQ_ID}-raw-movies_ophys.nwb`
#
# In this tutorial, we'll use the file:
# ```
# sub-684475/sub-684475_ses-1294084428-acq-1294179945-raw-movies_ophys.nwb
# ```
# as an illustrative example, but you can use this workflow for any NWB file in the Dandiset.
# %% [markdown]
# ## 3. Streaming and Loading NWB Files from DANDI (no download needed)
# 
# Let's stream the selected file using `remfile` and examine its NWB structure with `pynwb` and `h5py`.
# 
# ---
# %%
import remfile
import h5py
import pynwb

# Get the download URL for the example file
nwb_path = "sub-684475/sub-684475_ses-1294084428-acq-1294179945-raw-movies_ophys.nwb"
asset = next(dandiset.get_assets_by_glob(nwb_path))
url = asset.download_url
print("Streaming from:", url)

# Open file via remote streaming
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)

# Open with NWBHDF5IO for inspection with pynwb (read mode)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()
# %% [markdown]
# ## 4. NWB Structure and High-level Metadata
# 
# Let's inspect general information about this NWB session:  
# - Session details
# - Subject info
# - Keywords and notes
# - Experimental description
# %% [markdown]
# ### Session Metadata
# %%
print("NWB session identifier:", nwb.identifier)
print("Session start:", nwb.session_start_time)
print("Experiment description:", nwb.experiment_description)
print("Institution:", nwb.institution)
print("Subject info:", nwb.subject)
print("Subject genotype:", nwb.subject.genotype)
print("Subject sex:", nwb.subject.sex)
print("Keywords:", nwb.keywords[:])
# %% [markdown]
# ## 5. Exploring Calcium Imaging Data: ROIs and Fluorescence Traces
# 
# The "ophys" processing module contains:
# - **corrected_fluorescence** (raw/neuropil-subtracted)
# - **dff** (ΔF/F)
# - **event_detection** (binary or weighted event times)
# - **neuropil_trace**
# 
# Let's extract ROI info and plot some example traces.
# %% [markdown]
# ### ROI Segmentation Table
# 
# Let's inspect the ROI table:  
# (Each row = cell, with properties and mask for image segmentation)
# %%
# Access the ophys processing module and get main tables
ophys = nwb.processing["ophys"]
roi_table = ophys.data_interfaces["image_segmentation"].plane_segmentations["cell_specimen_table"].to_dataframe()

print("ROI table shape:", roi_table.shape)
display(roi_table.head())
# %% [markdown]
# ### Show ROI locations over the Field-of-View
# 
# _Note:_ To avoid overflow/casting errors, we use an integer array with a sufficient dtype for mask accumulation.
# %%
field_of_view_shape = (512, 512)
roi_mask_array = np.zeros(field_of_view_shape, dtype=np.int32)

for idx, mask in enumerate(roi_table['image_mask']):
    # Ensure mask is cast to an int (bools), and add unique ROI number to nonzero locations
    roi_mask_array = np.where(mask > 0, idx+1, roi_mask_array)

plt.figure(figsize=(6, 6))
plt.imshow(roi_mask_array, cmap='nipy_spectral', interpolation='nearest')
plt.title('ROI Segmentation Masks (Color = ROI #)')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.colorbar(label='ROI #')
plt.tight_layout()
plt.show()
# %% [markdown]
# ### ROI Geometry Overview
# 
# Let's summarize the distribution of ROI (cell) sizes and locations.
# %%
plt.figure(figsize=(10, 4))
sns.histplot(roi_table["width"], bins=20, color='skyblue', label='Width', kde=True)
sns.histplot(roi_table["height"], bins=20, color='orange', label='Height', kde=True)
plt.legend()
plt.title("Distribution of ROI Widths and Heights")
plt.xlabel("Pixels")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(roi_table["x"], roi_table["y"], c="r", s=40)
plt.title("ROI centroid locations (top-left corners)")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.xlim(0, field_of_view_shape[1])
plt.ylim(field_of_view_shape[0], 0)
plt.tight_layout()
plt.show()
# %% [markdown]
# ### ΔF/F Traces for All Cells (First 2 Minutes)
# 
# Let's plot ΔF/F for all ROIs in the first 120 seconds.  
# This shows synchronous and individual cell activity over time.
# %%
# Get dF/F traces and timestamps
dff = ophys.data_interfaces["dff"]
dff_traces = dff.roi_response_series["traces"].data[:]
dff_timestamps = dff.roi_response_series["traces"].timestamps[:]

# Restrict to first 120 seconds
mask = dff_timestamps < 120
plt.figure(figsize=(12, 6))
plt.plot(dff_timestamps[mask], dff_traces[mask])
plt.title("ΔF/F Traces for All ROIs (first 120 seconds)")
plt.xlabel("Time (s)")
plt.ylabel("ΔF/F")
plt.tight_layout()
plt.show()
# %% [markdown]
# ## 6. Exploring Event Detection Output
# 
# The event detection module provides putative neural event times for each cell on a frame-by-frame basis.
# 
# Let's plot event traces for the first few ROIs.
# %%
event_detection = ophys.data_interfaces["event_detection"]
event_data = event_detection.data[:]
event_timestamps = event_detection.timestamps[:]

n_show = 6  # number of ROIs to plot
plt.figure(figsize=(12, 6))
for i in range(n_show):
    plt.plot(event_timestamps[:1000], event_data[:1000, i] + i*0.2, label=f'Cell {i}')
plt.xlabel("Time (s)")
plt.title("Detected Event Traces (first 1000 frames, 6 ROIs, vertically offset)")
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.show()
# %% [markdown]
# ## 7. Viewing the Mean and Max Projection Images
# 
# Useful for context—what does the FOV look like?
# %%
images = ophys.data_interfaces["images"].images
average_img = images["average_image"].data[:]
max_proj_img = images["max_projection"].data[:]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(average_img, cmap='gray')
plt.title("Average Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(max_proj_img, cmap='gray')
plt.title("Max Projection Image")
plt.axis("off")
plt.tight_layout()
plt.show()
# %% [markdown]
# ## 8. Behavioral and Running Speed Data
# 
# This session includes synchronized running wheel data (raw and processed speed).  
# Let's inspect running speed over time.
# %%
running = nwb.processing["running"].data_interfaces
speed = running["speed"]
speed_vals = speed.data[:]
speed_times = speed.timestamps[:]

plt.figure(figsize=(12, 4))
plt.plot(speed_times[:10000], speed_vals[:10000])  # Show a subset for clarity
plt.xlabel("Time (s)")
plt.ylabel("Speed (cm/s)")
plt.title("Running Speed (first ~10k samples)")
plt.tight_layout()
plt.show()
# %% [markdown]
# ## 9. Eye and Pupil Tracking Overview
# 
# Multiple eye features are tracked: pupil, corneal reflection, and blinks.  
# Let's show example pupil size and position, plus blink trace.
# %%
eye = nwb.acquisition["EyeTracking"].spatial_series
pupil_data = eye["pupil_tracking"].data[:]
pupil_area = eye["pupil_tracking"].area[:]
blink_trace = nwb.acquisition["EyeTracking"].likely_blink.data[:]
sample_rate = 30.0
times = np.arange(len(pupil_area)) / sample_rate  # approximate frame rate

plt.figure(figsize=(12, 4))
plt.plot(times[:3000], pupil_area[:3000])
plt.xlabel("Time (s)")
plt.ylabel("Pupil area")
plt.title("Pupil Area (first 3000 samples)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 2))
plt.plot(times[:3000], blink_trace[:3000], color='k')
plt.xlabel("Time (s)")
plt.title("Likely Blink Trace (first 3000 samples)")
plt.yticks([0, 1], ["open", "blink"])
plt.tight_layout()
plt.show()
# %% [markdown]
# ## 10. Visual Stimulus Intervals: Movie Presentations
# 
# Each session includes detailed interval tables for stimulus presentations.  
# Let's list and briefly inspect the movie clip A presentation intervals.
# %%
intervals = nwb.intervals
movie_a = intervals["movie_clip_A_presentations"].to_dataframe()
print("Movie Clip A presentations, first 5:")
display(movie_a.head())
print("Number of Movie A presentations:", movie_a.shape[0])

plt.figure(figsize=(8, 4))
sns.histplot(movie_a["start_time"].diff().dropna(), bins=30)
plt.xlabel("Inter-stimulus Interval (s)")
plt.title("Intervals Between Movie Clip A Presentations")
plt.tight_layout()
plt.show()
# %% [markdown]
# ## 11. Next Steps and Further Exploration
# 
# You can adapt and extend this notebook to:
# - Analyze stimulus-driven calcium responses
# - Compare behavioral state (running/engaged vs. passive)
# - Examine response reliability across trials and conditions
# - Explore sequence learning effects (across days / sessions)
# - Investigate different brain regions and layers
#
# The Dandiset's structure and organization allow for efficient programmatic access to all subjects and sessions.
#
# ---
#
# For more, see:  
# - [DANDI User Guide](https://www.dandiarchive.org/)  
# - [Allen Brain Observatory](https://portal.brain-map.org/explore/circuits/visual-coding-2p)  
#
# Happy exploring!
# %% [markdown]
# ---
#
# *This notebook was generated with the assistance of AI based on the provided Dandiset structure and metadata. Please cite the Dandiset and refer to the associated protocols and Allen Institute documentation for experimental and analysis details.*