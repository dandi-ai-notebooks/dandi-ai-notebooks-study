# %% [markdown]
# # Exploring Dandiset 001361: A flexible hippocampal population code for experience relative to reward
#
# **Generated with the assistance of AI. Please use caution when interpreting the code or results, and verify findings independently before drawing conclusions.**
#
# ---
#
# This notebook serves as a practical introduction to [Dandiset 001361 (v0.250406.0045) on DANDI Archive](https://dandiarchive.org/dandiset/001361/0.250406.0045). It walks researchers through accessing, reviewing, and visualizing core types of data from this resource, helping you get started with reanalysis.
#
# **Dandiset overview:**
#
# - **Title:** A flexible hippocampal population code for experience relative to reward  
# - **Summary:** This dataset contains two-photon calcium imaging and behavioral tracking from the hippocampal CA1 of mice navigating a virtual reality environment, focused on learning reward locations.
# - **Data types:** Includes raw and processed imaging traces, anatomical segmentation, and a variety of behavioral/virtual trial variables, all stored in the NWB format.
#

# %% [markdown]
# ## What this notebook covers:
#
# - How to discover and load NWB files from this Dandiset using the DANDI API.
# - How to stream and inspect the structure of an NWB file with `h5py` (no download needed).
# - How to visualize:
#   - the animal's position over time (behavioral tracking)
#   - deconvolved neural activity for a sample of cells
#   - the segmentation mask regions (ROIs) for several example cells
#
# Throughout, explanations and code comments help you build a foundation for your own exploratory analyses.

# %% [markdown]
# ## Required packages
#
# This notebook assumes the following are already installed:
#
# - `dandi`
# - `h5py`
# - `remfile`
# - `numpy`
# - `matplotlib`
# - `scikit-image`
#

# %% [markdown]
# ## 1. Listing NWB files in the Dandiset
#
# First, we discover available NWB files (each representing an imaging/behavioral session) using the DANDI API.

# %%
from dandi.dandiapi import DandiAPIClient
from itertools import islice

dandiset_id = "001361"
dandiset_version = "0.250406.0045"

client = DandiAPIClient()
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# List a few NWB files
print("A sample of NWB files in this Dandiset:")
n = 10
assets = dandiset.get_assets_by_glob("*.nwb")
filenames = [asset.path for asset in islice(assets, n)]
for fn in filenames:
    print(" -", fn)

# %% [markdown]
# ## 2. Selecting a session
#
# We'll focus on one file: `sub-m11/sub-m11_ses-05_behavior+ophys.nwb`. This file includes a complete session with imaging and behavior data.

# %%
nwb_asset_path = "sub-m11/sub-m11_ses-05_behavior+ophys.nwb"
asset = next(dandiset.get_assets_by_glob(nwb_asset_path))
url = asset.download_url  # We will stream from this URL

print("Streaming NWB file from:", url)

# %% [markdown]
# ## 3. Inspecting the NWB file structure
#
# We'll use `h5py` to explore the major groups and datasets of the NWB file. This helps you orient yourself to the organization and locate the data you'd like to analyze.
#

# %%
import remfile
import h5py

def print_group_structure(h5group, prefix=""):
    """
    Recursively print the structure of the HDF5/NWB file,
    showing groups and top-level datasets.
    """
    for key in h5group:
        item = h5group[key]
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/ [Group]")
            # For brevity, don't recurse into all subgroups here.
        else:
            print(f"{prefix}{key} [Dataset] shape={item.shape} dtype={item.dtype}")

with h5py.File(remfile.File(url), 'r') as f:
    print("Top-level groups and datasets:")
    print_group_structure(f)

# %% [markdown]
# ## 4. Behavioral Data Example: Mouse Position Over Time
#
# Let's plot the mouse's position on the virtual track over the course of this session. This uses the `/processing/behavior/BehavioralTimeSeries/position/` dataset.
#

# %%
import numpy as np
import matplotlib.pyplot as plt

with h5py.File(remfile.File(url), 'r') as f:
    base_path = 'processing/behavior/BehavioralTimeSeries/position'
    position = f[base_path + '/data'][:]
    timestamps = f[base_path + '/timestamps'][:]

plt.figure(figsize=(10, 4))
plt.plot(timestamps, position, lw=1)
plt.xlabel('Time (s)')
plt.ylabel('Position (cm)')
plt.title('Mouse Position Over Time — sub-m11_ses-05')
plt.tight_layout()
plt.show()

# %% [markdown]
# *Interpretation:*  
# Each ramp reflects a traversal of the mouse along the virtual track, with returns to zero indicating trial resets (each trial = one lap). Negative values at session start may be artifacts and could be excluded in careful analyses.
#

# %% [markdown]
# ## 5. Neural Data Example: Deconvolved Activity Traces for 10 Cells
#
# We next visualize the deconvolved (inferred spike) activity traces for the first 10 segmented cells, using `/processing/ophys/Deconvolved/plane0/data`.
#

# %%
with h5py.File(remfile.File(url), 'r') as f:
    data = f['processing/ophys/Deconvolved/plane0/data'][:, :10]
    starting_time = f['processing/ophys/Deconvolved/plane0/starting_time'][()]
    num_frames = data.shape[0]
    sampling_rate = 15.5078125  # known from prior info
    time = np.arange(num_frames) / sampling_rate + starting_time

plt.figure(figsize=(10, 6))
offset = 7
for i in range(10):
    plt.plot(time, data[:, i] + i * offset, label=f'Cell {i}')
plt.xlabel('Time (s)')
plt.ylabel('Deconvolved Activity (a.u., offset per cell)')
plt.title('Deconvolved Activity Traces for 10 Example Cells')
plt.yticks([])
plt.tight_layout()
plt.show()

# %% [markdown]
# *Interpretation:*  
# Each trace represents one cell (offset for visibility). You can see sparse, event-like transients typical of deconvolved calcium imaging.
#

# %% [markdown]
# ## 6. Example Cell Segmentation Masks (ROI Contours Only)
#
# Finally, we visualize the pixel masks for the first 10 segmented ROIs (cells) in their native coordinate space (no background image, to avoid alignment issues). This gives a sense of the segmentation quality and ROI positions.
#

# %%
from dandi.dandiapi import DandiAPIClient

with h5py.File(remfile.File(url), 'r') as f:
    pixel_mask = f['processing/ophys/ImageSegmentation/PlaneSegmentation/pixel_mask'][:]
    pixel_mask_index = f['processing/ophys/ImageSegmentation/PlaneSegmentation/pixel_mask_index'][:]
    num_rois = 10
    roi_xy = []
    for i in range(num_rois):
        start = pixel_mask_index[i-1] if i > 0 else 0
        end = pixel_mask_index[i]
        mask_entries = pixel_mask[start:end]
        xy = np.array([(entry[0], entry[1]) for entry in mask_entries])
        roi_xy.append(xy)

plt.figure(figsize=(7, 7))
colors = plt.cm.tab10(np.linspace(0, 1, num_rois))
for i, xy in enumerate(roi_xy):
    if xy.shape[0] > 0:
        plt.scatter(xy[:, 0], xy[:, 1], s=10, color=colors[i], label=f'ROI {i}')
plt.gca().invert_yaxis()
plt.xlabel('X (pixel)')
plt.ylabel('Y (pixel)')
plt.title('Contours of the First 10 ROIs (no background)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# %% [markdown]
# *Interpretation:*  
# Each colored cluster reveals the spatial form and coverage of a segmented ROI (cell body or process extraction) in pixel space.
#

# %% [markdown]
# ## 7. What's Next?
#
# - Use these examples as templates to plot or analyze other behavioral variables, other neural signals (raw or ΔF/F), or all ROIs.
# - Investigate trial alignment, rewards, or trial-by-trial dynamics by using other behavioral fields.
# - For anatomical mapping, consult the original paper's supplementary materials, and consider coregistration efforts if anatomical overlays are required.
#
# **Remember:**  
# This notebook provides a skeleton for working with Dandiset 001361. Interpret results with caution and consult both dataset documentation and source publications for further context.
#
# ---
#
# **[Return to Dandiset page](https://dandiarchive.org/dandiset/001361/0.250406.0045)**