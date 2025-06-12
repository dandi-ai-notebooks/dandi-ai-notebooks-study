# %% [markdown]
# # Exploring Dandiset 001174: Calcium imaging in SMA and M1 of macaques
# 
# **Note:** This notebook was generated with the assistance of AI. Please be cautious when interpreting the code or results and verify findings independently.

# %% [markdown]
# ## Overview
# 
# This Dandiset contains calcium imaging data from the supplementary motor area (SMA) and primary motor cortex (M1) of rhesus macaques. The study uses genetically encoded calcium indicators (GCaMP6f) and miniature microscopes through GRIN lenses to record neural activity while macaques were at rest and performing arm-reaching tasks.
# 
# **Dandiset Link:** https://dandiarchive.org/dandiset/001174/0.250331.2218
# 
# **Citation:**  
# Galvan, Adriana; Martel, Anne-Caroline; Pittard, Damien; Devergnas, Annaelle; Risk, Benjamin; Nassi, Jonathan J.; Yu, Waylin; Downer, Joshua D. ; Wichmann, Thomas (2025) Calcium imaging in SMA and M1 of macaques (Version 0.250331.2218) [Data set]. DANDI Archive. https://doi.org/10.48324/dandi.001174/0.250331.2218

# %% [markdown]
# ### Dataset Contents
# 
# - **Species:** Rhesus macaque (Macaca mulatta)
# - **Subjects:** 4 macaques  
# - **Data types:** One-photon calcium imaging videos, segmented cell traces, processed cell masks, behavioral task metadata
# - **Modality:** One-photon population calcium imaging using GCaMP6f
# - **Structure:** Data stored in NWB files, organized by session and subject

# %% [markdown]
# ## Required Packages
# 
# The following packages are required to run this notebook:
# - `dandi`
# - `pynwb`
# - `h5py`
# - `remfile`
# - `matplotlib`
# - `numpy`
# - `pandas`

# %%
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dandi.dandiapi import DandiAPIClient
from itertools import islice

# %% [markdown]
# ## Exploring the Dandiset Metadata

# %%
# Get basic information about the dandiset
client = DandiAPIClient()
dandiset = client.get_dandiset("001174", "0.250331.2218")
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# %% [markdown]
# ## Exploring NWB Files in the Dandiset

# %%
# List NWB files in the dandiset
assets = list(islice(dandiset.get_assets_by_glob("*.nwb"), 20))

print(f"Number of .nwb files (showing up to 20): {len(assets)}\n")
for i, asset in enumerate(assets):
    print(f"{i+1}. {asset.path} - {asset.size/1e9:.2f} GB")

# %% [markdown]
# The dataset is organized by subject (e.g., sub-Q, sub-V, sub-F, sub-U) with multiple sessions per subject. File sizes range from ~0.8 GB to 25 GB depending on session length and data recorded.

# %% [markdown]
# ## Loading and Exploring NWB File Structure
# 
# Let's examine a representative NWB file to understand its structure and contents.

# %%
# Select a representative file for exploration
asset_path = 'sub-Q/sub-Q_ses-20221206T121002_ophys.nwb'
asset = next(dandiset.get_assets_by_glob(asset_path))
url = asset.download_url

# Open the file remotely (streaming)
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# Display basic metadata
print('Session description:', nwb.session_description)
print('Session start time:', nwb.session_start_time)
print('Subject ID:', nwb.subject.subject_id)
print('Species:', nwb.subject.species)
print('Sex:', nwb.subject.sex)
print('Age:', getattr(nwb.subject, 'age', 'n/a'))

# %% [markdown]
# ### Data Structure Overview

# %%
# Examine the main data components
ophys = nwb.processing['ophys']
fluorescence = ophys.data_interfaces['Fluorescence']
rrs = fluorescence.roi_response_series['RoiResponseSeries']
event_amp = ophys.data_interfaces['EventAmplitude']
plane_seg = ophys.data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
onphs = nwb.acquisition['OnePhotonSeries']

print('--- Data shapes ---')
print('Raw movie:', onphs.data.shape)
print('Fluorescence traces:', rrs.data.shape)
print('Event amplitudes:', event_amp.data.shape)
print('Number of segmented cells:', len(plane_seg.to_dataframe()))

print('\n--- Technical details ---')
print('Imaging rate:', onphs.rate, 'Hz')
print('Device:', onphs.imaging_plane.device.description)
print('Manufacturer:', onphs.imaging_plane.device.manufacturer)
print('Excitation wavelength:', onphs.imaging_plane.excitation_lambda, 'nm')

# %% [markdown]
# ## Visualizing Cell Segmentation
# 
# Let's examine the spatial distribution of segmented cells (ROIs) in the field of view.

# %%
# Get all cell masks and create a spatial overview
roi_df = plane_seg.to_dataframe()
mask_arrays = [plane_seg['image_mask'].data[i][:] for i in range(len(roi_df))]

# Create heatmap showing all cell footprints
mask_stack = np.stack(mask_arrays, axis=0)
max_mask = np.max(mask_stack, axis=0)

plt.figure(figsize=(10, 7))
plt.imshow(max_mask, cmap='hot')
plt.title('Spatial distribution of all segmented cells')
plt.colorbar(label='Number of overlapping ROIs')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.tight_layout()
plt.show()

# Calculate ROI sizes
roi_sizes = [np.count_nonzero(mask) for mask in mask_arrays]
print(f'Number of cells detected: {len(roi_sizes)}')
print(f'Mean ± std ROI size: {np.mean(roi_sizes):.1f} ± {np.std(roi_sizes):.1f} pixels')

# %% [markdown]
# ## Examining Raw Calcium Imaging Data
# 
# Let's look at a few frames from the raw calcium imaging movie to understand the data structure.

# %%
# Preview a short segment of the raw movie (first 12 frames)
frames_to_show = 12
movie_segment = onphs.data[:frames_to_show]

# Create a montage of frames
n_cols = 4
n_rows = int(np.ceil(frames_to_show / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8))

for i in range(frames_to_show):
    ax = axes[i // n_cols, i % n_cols]
    ax.imshow(movie_segment[i], cmap='gray', aspect='auto')
    ax.set_title(f'Frame {i}')
    ax.axis('off')

# Hide unused subplots
for i in range(frames_to_show, n_rows * n_cols):
    axes.flat[i].axis('off')

plt.suptitle('Raw calcium imaging frames (first 1.2 seconds at 10 Hz)')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# %% [markdown]
# ## Analyzing Fluorescence Traces
# 
# Now let's examine the processed fluorescence traces extracted from the segmented cells.

# %%
# Plot fluorescence trace for the first cell
sampling_rate = rrs.rate
n_seconds_plot = 5 * 60  # 5 minutes
n_samples_plot = int(n_seconds_plot * sampling_rate)

time = np.arange(n_samples_plot) / sampling_rate
f_trace = rrs.data[:n_samples_plot, 0]  # First ROI

plt.figure(figsize=(10, 4))
plt.plot(time, f_trace)
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.)')
plt.title('Fluorescence trace (cell 0, first 5 minutes)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Comparing Raw Fluorescence vs Event Amplitudes
# 
# The dataset includes both raw fluorescence traces and processed event amplitudes. Let's compare them for the same cell.

# %%
# Plot both fluorescence and event amplitude for the first cell
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Raw fluorescence
ax1.plot(time, f_trace)
ax1.set_ylabel('Fluorescence (a.u.)')
ax1.set_title('Raw fluorescence trace (cell 0)')

# Event amplitude
ampl_trace = event_amp.data[:n_samples_plot, 0]
ax2.plot(time, ampl_trace)
ax2.set_ylabel('Event amplitude (a.u.)')
ax2.set_xlabel('Time (s)')
ax2.set_title('Event amplitude trace (cell 0)')

plt.tight_layout()
plt.show()

# %% [markdown]
# The event amplitude trace shows sparse, discrete events compared to the continuous fluorescence signal, indicating processed detection of calcium transients.

# %% [markdown]
# ## Cell-wise Activity Statistics
# 
# Let's calculate summary statistics across all cells to understand population activity patterns.

# %%
# Calculate event rates for all cells
num_cells = rrs.data.shape[1]
minutes = rrs.data.shape[0] / sampling_rate / 60

# Count events (nonzero amplitudes) for each cell
cell_event_counts = [(event_amp.data[:, i] > 0).sum() for i in range(num_cells)]
cell_event_rates = [count / minutes for count in cell_event_counts]

print('Event rates per cell (events/min):')
for i, rate in enumerate(cell_event_rates):
    print(f'Cell {i}: {rate:.2f}')

print(f'\nPopulation statistics:')
print(f'Mean event rate: {np.mean(cell_event_rates):.2f} ± {np.std(cell_event_rates):.2f} events/min')
print(f'Range: {np.min(cell_event_rates):.2f} - {np.max(cell_event_rates):.2f} events/min')

# Plot distribution of event rates
plt.figure(figsize=(8, 5))
plt.hist(cell_event_rates, bins=10, edgecolor='black')
plt.xlabel('Event rate (events/min)')
plt.ylabel('Number of cells')
plt.title('Distribution of event rates across cells')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualizing Individual Cell Properties
# 
# Let's examine both the spatial footprint and temporal activity for a specific cell.

# %%
# Select an active cell for detailed visualization
cell_idx = np.argmax(cell_event_rates)  # Most active cell
mask = plane_seg['image_mask'].data[cell_idx][:]
trace = rrs.data[:1000, cell_idx]  # First 100 seconds
time_short = np.arange(1000) / sampling_rate

plt.figure(figsize=(12, 5))

# Plot cell mask
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap='hot')
plt.title(f'Cell {cell_idx} spatial footprint\n({cell_event_rates[cell_idx]:.1f} events/min)')
plt.colorbar(label='Mask intensity')
plt.axis('off')

# Plot fluorescence trace
plt.subplot(1, 2, 2)
plt.plot(time_short, trace)
plt.title(f'Cell {cell_idx} fluorescence trace')
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary and Next Steps
# 
# This notebook demonstrates how to:
# - Access and explore NWB files in the Dandiset using the DANDI API
# - Load calcium imaging data, cell segmentation masks, and processed traces
# - Visualize spatial and temporal aspects of the neural activity data
# - Calculate basic statistics across the cell population
# 
# **Potential analyses to explore further:**
# - Cross-correlation analysis between cells
# - Population dynamics and synchrony patterns  
# - Comparison across sessions, subjects, or behavioral conditions
# - Integration with behavioral task data if available
# - Advanced signal processing of calcium transients