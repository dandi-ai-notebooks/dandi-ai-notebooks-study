# %% [markdown]
# # Exploring Dandiset 001349: GABAergic Inputs Shape Neural Networks in Developing Somatosensory Cortex

# %% [markdown]
# > **Note**: This notebook was generated with AI assistance. Please review all code and results carefully.

# %% [markdown]
# ## Overview
# This notebook introduces [Dandiset 001349](https://dandiarchive.org/dandiset/001349/0.250520.1729), which contains two-photon calcium imaging data from the primary somatosensory cortex of developing mice (P11-P21). The dataset includes synchronized behavioral video recordings and multiple processed signals derived from cellular calcium imaging.

# %% [markdown]
# ### What this notebook covers:
# - Loading and accessing NWB files from the Dandiset
# - Exploring session and subject metadata
# - Visualizing calcium imaging signals (ΔF/F and calcium events)
# - Understanding the organization of ROIs (cells) and their activity
# - Working with different signal types available in the dataset

# %% [markdown]
# ### Required Packages
# ```python
# dandi
# pynwb
# h5py
# remfile
# numpy
# matplotlib
# pandas
# ```

# %% [markdown]
# ## Accessing the Dandiset

# %%
from dandi.dandiapi import DandiAPIClient
from itertools import islice

# Initialize DANDI API client
client = DandiAPIClient()
dandiset = client.get_dandiset("001349", "0.250520.1729")

# List the first 20 NWB files
assets = dandiset.get_assets_by_glob("*.nwb")
asset_list = list(islice(assets, 20))

# Create summary table of available files
import pandas as pd
result = [{
    "path": asset.path,
    "size_MB": round(asset.size / 1e6, 2)
} for asset in asset_list]

# Display file information
summary_table = pd.DataFrame(result)
summary_table

# %% [markdown]
# ## Loading and Exploring an NWB File
# Let's examine a representative NWB file to understand its structure and contents.

# %%
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load a sample NWB file remotely
asset_path = "sub-C57-C2-2-AL/sub-C57-C2-2-AL_ses-2_ophys.nwb"
url = next(dandiset.get_assets_by_glob(asset_path)).download_url

# Stream the file without downloading
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract key metadata
info = {
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

# Display metadata as table
pd.DataFrame(list(info.items()), columns=['Field', 'Value'])

# %% [markdown]
# ## Visualizing Neural Activity
# The dataset contains multiple signal types for each cell/ROI. Here we'll visualize both ΔF/F and calcium events.

# %%
# Get dF/F traces for first 10 cells
dff = nwb.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['dff_chn0']
dff_data = dff.data[:1000, :10]  # first 1000 time points, first 10 cells
rate = dff.rate

# Plot dF/F traces
plt.figure(figsize=(10, 6))
t = np.arange(dff_data.shape[0]) / rate
for i in range(dff_data.shape[1]):
    plt.plot(t, dff_data[:, i], label=f'Cell {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('dF/F (a.u.)')
plt.title('ΔF/F traces: first 10 cells, first 1000 time points')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Calcium Events
# Now let's look at the detected calcium events, which represent more discrete activity.

# %%
# Get calcium event traces
caev = nwb.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['ca_events_chn0']
caev_data = caev.data[:1000, :10]  # first 1000 time points, first 10 cells

# Plot calcium events with small offset for visibility
plt.figure(figsize=(10, 6))
t = np.arange(caev_data.shape[0]) / rate
for i in range(caev_data.shape[1]):
    plt.plot(t, caev_data[:, i] + i*0.03, label=f'Cell {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Ca Event Value (offset per cell)')
plt.title('Calcium Events: first 10 cells, first 1000 time points')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Available Signal Types
# The dataset provides multiple processed signals for analysis:

# Signal Type | Description | Use Case
# ------------|-------------|----------
# dff_chn0 | ΔF/F traces | Standard metric for activity changes
# ca_events_chn0 | Calcium event times | Event detection and timing analysis
# f_corrected_chn0 | Corrected fluorescence | Background-corrected signals
# fluorescence_chn0 | Raw fluorescence | Original imaging data
# neuropil_fluorescence_chn0 | Background signal | Neuropil contamination assessment
# z_score_chn0 | Z-scored activity | Normalized across cells/sessions

# %% [markdown]
# ## Next Steps
# This dataset enables various analyses:
# - Developmental changes in neural activity (P11-P21)
# - Cell assembly dynamics and population coding
# - Correlation between neural activity and behavior
# - Network structure and its refinement
# 
# ### Practical Tips
# - Use the Neurosift NWB browser for interactive exploration
# - Look for LED triggers in acquisition/processing groups for behavioral synchronization
# - Consider batch processing across sessions/ages for developmental analyses