# %% [markdown]
# # Exploring Dandiset 000617: Allen Institute Openscope Sequence Learning Project
#
# *This notebook was generated with the assistance of AI. Please review and interpret results with caution. Double-check code, outputs, and conclusions before using for research or analysis purposes.*
#
# ---
#
# ## Overview
#
# **Dandiset 000617** ("Allen Institute Openscope - Sequence Learning Project") is a large-scale open neurophysiology resource focused on predictive coding and sequence learning in mouse cortex. The project combines two-photon calcium imaging, behavioral tracking, and naturalistic movie stimuli to investigate how the neocortex learns and anticipates temporal sequences.
#
# - Direct link to the Dandiset: [https://dandiarchive.org/dandiset/000617/0.250311.1615](https://dandiarchive.org/dandiset/000617/0.250311.1615)
# - Data: NWB files with raw and processed neural, behavioral, and stimulus data from multiple mice, sessions, and brain areas.
#
# ---
#
# ## What This Notebook Covers
#
# This notebook walks you through:
#
# 1. Listing and selecting NWB file(s) from the Dandiset with the DANDI API.
# 2. Loading NWB files (by streaming remotely—no download required).
# 3. Visualizing:
#     - The two-photon imaging data and ROI segmentations.
#     - ΔF/F (dff) activity traces for example cells.
#     - Mouse running speed during the session.
#     - Stimulus presentation timelines.
#     - A sample of eye tracking (pupil area) data.
# 4. Gaining confidence in how to access, inspect, and visualize these multimodal datasets for reanalysis.
#
# ---
#
# ## Required Packages
#
# Make sure the following packages are installed in your environment:
# - dandi
# - pynwb
# - remfile
# - h5py
# - numpy
# - matplotlib
#
# No install commands are included here (assume packages are present).
#
# Let's get started!

# %%
# Import packages used in this notebook
import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynwb

# %% [markdown]
# ## Accessing Dandiset Assets with the DANDI API
#
# We'll start by listing NWB files in the Dandiset and picking one to examine. We'll use the DANDI API to access the remote data.

# %%
from dandi.dandiapi import DandiAPIClient
from itertools import islice

dandiset_id = '000617'
dandiset_version = '0.250311.1615'

# Connect to DANDI and list first 10 NWB files
with DandiAPIClient() as client:
    dandiset = client.get_dandiset(dandiset_id, dandiset_version)
    assets = dandiset.get_assets_by_glob('*.nwb')
    asset_list = [asset.path for asset in islice(assets, 10)]

print("Example NWB files in this Dandiset:")
for fname in asset_list:
    print(fname)

# %% [markdown]
# For this exploration, we will use the following NWB file (from the above list):
#
# ```
# sub-684475/sub-684475_ses-1294084428-acq-1294179945-raw-movies_ophys.nwb
# ```
# 
# We will show how to access and visualize key data modalities within this file.

# %%
# Helper function: Get remote download URL for a NWB asset
def get_download_url(asset_path, dandiset_id, dandiset_version):
    from dandi.dandiapi import DandiAPIClient
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id, dandiset_version)
        asset = next(dandiset.get_assets_by_glob(asset_path))
        return asset.download_url

asset_path = 'sub-684475/sub-684475_ses-1294084428-acq-1294179945-raw-movies_ophys.nwb'
nwb_url = get_download_url(asset_path, dandiset_id, dandiset_version)
print(f"Streaming from: {nwb_url}")

# %% [markdown]
# ## Opening the NWB File (Remote Streaming)
#
# We'll stream data directly from DANDI using `remfile` and `h5py`, then open the NWB structure using pynwb without downloading the file.

# %%
# Open the NWB file for streaming with remfile and h5py
remote_file = remfile.File(nwb_url)
h5_file = h5py.File(remote_file, "r")

# Load NWB structure (read-only streaming)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# %% [markdown]
# ## 1. Visualizing Two-Photon Imaging Data
#
# Let's look at a motion-corrected frame from the main two-photon imaging series: `raw_suite2p_motion_corrected`.
# We'll display frame 0, using percentile normalization to enhance contrast for faint signals.

# %%
# Access and normalize the first frame of the imaging data
tps = nwb.acquisition['raw_suite2p_motion_corrected']
frame = tps.data[0, :, :].astype(float)
frame_min = np.percentile(frame, 1)
frame_max = np.percentile(frame, 99)
frame_disp = np.clip((frame - frame_min) / (frame_max - frame_min), 0, 1)

plt.figure(figsize=(7,7))
plt.imshow(frame_disp, cmap='gray', aspect='auto')
plt.title('Enhanced contrast: Frame 0 (raw_suite2p_motion_corrected)')
plt.axis('off')
plt.show()

# %% [markdown]
# ## 2. Visualizing Processed dF/F Traces for Example Cells
#
# We'll plot ΔF/F (dff; normalized calcium activity) traces for 10 example cells from the processed output.
# These traces capture the temporal neural activity dynamics.

# %%
# Load dff traces (ΔF/F) for all cells and the timestamps
traces = nwb.processing['ophys'].data_interfaces['dff'].roi_response_series['traces'].data
timestamps = nwb.processing['ophys'].data_interfaces['dff'].roi_response_series['traces'].timestamps
num_cells = traces.shape[1]
plot_cells = min(10, num_cells)  # Plot up to 10

# Downsample if signals are very long (for plotting speed)
downsample = 5 if traces.shape[0] > 10000 else 1
traces_np = traces[::downsample, :plot_cells]
time = timestamps[::downsample]

plt.figure(figsize=(10, 6))
for i in range(plot_cells):
    plt.plot(time, traces_np[:, i], label=f'Cell {i}')
plt.xlabel('Time (s)')
plt.ylabel('ΔF/F (dff)')
plt.title('Sample ΔF/F traces for 10 cells')
plt.legend(loc='upper right', ncol=2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Visualizing ROI Masks: Linking Cells to the Imaging Field
#
# Each example cell has a spatial ROI (segmentation mask) in the imaging plane.
# We'll overlay the ROI masks for the first 10 cells, in color, atop the average image for this session.

# %%
# Load the average imaging frame as background
avg_img = nwb.processing['ophys'].data_interfaces['images'].images['average_image'].data[:]

# Access ROI info (segmentation mask) for the first 10 dff cells
roi_table = nwb.processing['ophys'].data_interfaces['dff'].roi_response_series['traces'].rois.table
num_rois = min(10, len(roi_table.id))
img_shape = avg_img.shape
masks = roi_table['image_mask'][:num_rois]  # Each mask has shape (img_shape)

# Create an RGB overlay: background + colored masks (one per ROI)
background = (avg_img - np.percentile(avg_img, 1)) / (np.percentile(avg_img, 99) - np.percentile(avg_img, 1))
background = np.clip(background, 0, 1)
color_overlay = np.stack([background] * 3, axis=2)

colors = plt.colormaps['tab10'](np.linspace(0, 1, num_rois))
for i, mask in enumerate(masks):
    color = colors[i][:3]
    mask_2d = mask.astype(float)
    color_overlay[mask_2d > 0.2] = color

plt.figure(figsize=(8,8))
plt.imshow(background, cmap='gray', alpha=0.5)
plt.imshow(color_overlay, alpha=0.5)
plt.title('ROI masks (first 10 cells) overlaid on average image')
plt.axis('off')
plt.show()

# %% [markdown]
# ## 4. Visualizing Running Speed (Behavior)
#
# The mouse's locomotion (running speed) is recorded throughout the session.
# Here we extract and plot the running speed in centimeters per second (cm/s).

# %%
# Prefer processed running speed; fallback to raw wheel signal if needed
run_proc = nwb.processing.get('running', None)
if run_proc and 'speed' in run_proc.data_interfaces:
    speed_ts = run_proc.data_interfaces['speed']
    speed = speed_ts.data[:]
    t_speed = speed_ts.timestamps[:]
    speed_unit = speed_ts.unit
else:
    v_sig_ts = nwb.acquisition['v_sig']
    speed = v_sig_ts.data[:]
    t_speed = v_sig_ts.timestamps[:]
    speed_unit = v_sig_ts.unit

plt.figure(figsize=(10, 4))
plt.plot(t_speed, speed, color='dodgerblue')
plt.xlabel('Time (s)')
plt.ylabel(f'Running speed ({speed_unit})')
plt.title('Running speed over time for this session')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Visualizing Stimulus Presentation Intervals ("movie_clip_A")
#
# The experiment presents repeated movie clips in temporal sequences.
# We visualize the intervals (start/stop times) for "movie_clip_A" as a raster plot across the recording session.

# %%
# Access intervals for 'movie_clip_A' stimulus presentations
clip_A_int = nwb.intervals["movie_clip_A_presentations"]
start_A = clip_A_int.start_time[:]
stop_A = clip_A_int.stop_time[:]
num_A = len(start_A)

plt.figure(figsize=(10, 1.5))
for i in range(num_A):
    plt.plot([start_A[i], stop_A[i]], [1, 1], lw=4, color='cornflowerblue', alpha=0.7)
plt.ylim(0.9, 1.1)
plt.yticks([])
plt.xlabel("Time (s)")
plt.title("Timing of 'movie_clip_A' stimulus presentations")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Visualizing Eye Tracking Data (Pupil Area)
#
# Pupil dynamics can reflect arousal, attention, and visual input. We'll plot a short segment of the pupil area trace from the eye tracking acquisition.

# %%
# Load and plot pupil area trace (first 5000 samples for visibility)
eyetrack = nwb.acquisition['EyeTracking']
pupil_area = eyetrack.spatial_series['pupil_tracking'].area[:]
pupil_time = eyetrack.spatial_series['pupil_tracking'].timestamps[:]

plt.figure(figsize=(10, 4))
plt.plot(pupil_time[:5000], pupil_area[:5000], color='seagreen')
plt.xlabel('Time (s)')
plt.ylabel('Pupil area (a.u.)')
plt.title('Eye Tracking: Pupil area trace (first 5000 points)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary: What Did We Explore?
#
# In this notebook, you have learned how to:
# - List and stream assets from a Dandiset using the DANDI API.
# - Load and access NWB files directly from the archive (without saving locally).
# - Visualize the main data modalities:
#   - Motion-corrected two-photon imaging frames
#   - ROI segmentation masks spatially overlaid on the imaging field
#   - ΔF/F (dff) activity traces for individual cells
#   - Running speed timecourses (cm/s)
#   - Interval timelines for stimulus presentations
#   - Example eye tracking signals (pupil area)
#
# You can use these code templates as a starting point for your own analyses—such as response alignment to stimuli, cell population analyses, behavioral correlations, and more.
#
# ---
#
# *Remember: This notebook was AI-generated. Please validate, adapt, and extend to suit your specific research needs.*
#

# %%
# Properly close NWB and HDF5 streaming handles
io.close()
h5_file.close()
remote_file.close()