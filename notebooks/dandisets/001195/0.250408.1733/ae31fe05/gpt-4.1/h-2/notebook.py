# %% [markdown]
# # Exploring Dandiset 001195: Separable Dorsal Raphe Dopamine Projections Mediate the Facets of Loneliness-like State
#
# *Generated with the assistance of AI. Please review all code and outputs with care.*
#
# This notebook provides an interactive introduction to Dandiset 001195 ([link to archive](https://dandiarchive.org/dandiset/001195/0.250408.1733)), a neuroscience dataset exploring the role of dorsal raphe dopamine projections in various facets of loneliness-like states in mice.
#
# The notebook is designed to help researchers:
# - Understand the structure and content of this Dandiset
# - Get started with loading, inspecting, and plotting example data streams
# - Learn best practices for streaming and visualizing NWB data from DANDI via Python
#
# **Notebook coverage includes:**
# - Dandiset overview and key metadata
# - Connecting to the DANDI Archive with the DANDI API
# - Listing representative files
# - Exploring the structure and content of an electrophysiology NWB file (ex vivo patch clamp)
# - Loading and visualizing both current clamp and voltage clamp sweeps with matched protocol commands
# - Guidance on next steps for accessing other modalities (e.g., calcium imaging, if present elsewhere in the Dandiset)
#
# Please refer to the DANDI web portal for additional context, and always verify code and output carefully.
# 
# ## Required Packages
# - `dandi`
# - `pynwb`
# - `remfile`
# - `h5py`
# - `matplotlib`
# - `numpy`
#
# All packages are assumed to be installed.

# %% [markdown]
# ## Dandiset 001195 Overview
#
# **Title:** Separable Dorsal Raphe Dopamine Projections Mediate the Facets of Loneliness-like State  
#
# **Authors:** Keyes, Laurel; Lee, Christopher R.; Wichmann, Romy; Matthews, Gillian A.; Tye, Kay M.  
# **Data Types:** Ex vivo patch-clamp electrophysiology, in vivo calcium imaging, behavior video  
# **Subjects:** Mouse (Mus musculus, DAT::Cre, viral AAV5-DIO-ChR2-eYFP into DRN)  
# **Data Format:** NWB  
# **Archive:** [https://dandiarchive.org/dandiset/001195/0.250408.1733](https://dandiarchive.org/dandiset/001195/0.250408.1733)
#
# ### Key Concepts
# - Patch clamp and optogenetics for dissecting neural circuits involved in social behaviors
# - NWB structure: Each file = session/experiment, can contain multiple data streams per modality
#
# **This notebook will show you:**
# - How to connect to DANDI, browse contents, and stream NWB data
# - How to load and plot example voltage and current clamp sweeps with corresponding command signals
# - How to examine important metadata

# %% [markdown]
# ## 1. Connect to the DANDI Archive and List Assets
#
# We'll use the `dandi` Python API to connect to the archive, browse files, and select files to explore.
# 
# *All data loading is performed by remote streaming, not local download, where possible.*

# %%
from dandi.dandiapi import DandiAPIClient

# Connect to the DANDI Archive and access the desired Dandiset version
dandiset_id = "001195"
dandiset_version = "0.250408.1733"

client = DandiAPIClient()
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# List the first several NWB files as examples
print("Representative NWB files in this dandiset:")
nwb_asset_list = []
for i, asset in enumerate(dandiset.get_assets_by_glob("*.nwb")):
    nwb_asset_list.append(asset.path)
    print(f"{i+1}. {asset.path}")
    if i >= 9:
        break

# %% [markdown]
# For the rest of this notebook, we will explore a specific NWB file containing patch-clamp electrophysiology:
# 
# ```
# sub-23/sub-23_ses-20150324T134114_slice-slice-1_cell-C1_icephys.nwb
# ```

# %% [markdown]
# ## 2. Streaming and Inspecting a Patch Clamp NWB File
#
# This section demonstrates how to:
# - Stream an NWB file directly from DANDI without downloading
# - List the available acquisition (recorded) and stimulus (command) series
# - Inspect key metadata and file structure

# %%
import remfile
import h5py
import pynwb

# Select the NWB file path
asset_path = "sub-23/sub-23_ses-20150324T134114_slice-slice-1_cell-C1_icephys.nwb"

# Find and stream the file
asset = next(dandiset.get_assets_by_glob(asset_path))
download_url = asset.download_url

# Stream the file using remfile and open with h5py and pynwb
remote_file = remfile.File(download_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwbfile = io.read()

# List the acquisition and stimulus series (data and commands)
acq_list = list(nwbfile.acquisition.keys())
stim_list = list(nwbfile.stimulus.keys())

print("Acquisition series (responses):")
for name in acq_list:
    print("-", name)
print("\nStimulus series (command waveforms):")
for name in stim_list:
    print("-", name)

# %% [markdown]
# ## 3. Key Metadata and Organizational Features
#
# Let's extract some core metadata and show how ex vivo patch clamp NWB files in this dandiset are structured.
#

# %%
# Key session metadata
print("Session description:", nwbfile.session_description)
print("Session start time:", nwbfile.session_start_time)
print("Lab:", getattr(nwbfile, "lab", ""))
print("Institution:", getattr(nwbfile, "institution", ""))

# Subject metadata
subject = nwbfile.subject
print("Subject ID:", getattr(subject, "subject_id", ""))
print("Sex:", getattr(subject, "sex", ""))
print("Strain:", getattr(subject, "strain", ""))
print("Genotype:", getattr(subject, "genotype", ""))
print("Description:", getattr(subject, "description", ""))

# Specialized lab-specific experimental info (may be present in many files)
if "DandiIcephysMetadata" in nwbfile.lab_meta_data:
    meta = nwbfile.lab_meta_data["DandiIcephysMetadata"]
    print("Recorded cell ID:", getattr(meta, "cell_id", ""))
    print("Slice:", getattr(meta, "slice_id", ""))
    print("Targeted layer/structure:", getattr(meta, "targeted_layer", ""))

# %% [markdown]
# ## 4. Visualizing Electrophysiology Data: Current Clamp Example
#
# The NWB file contains many current clamp sweeps. Each is made up of:
# - A response series (voltage, typically in volts)
# - Matched stimulus command series (injected current, typically in amperes)
#
# Let's load and plot an example sweep.

# %%
import numpy as np
import matplotlib.pyplot as plt

# Select a sample current clamp response and its matching stimulus
resp = nwbfile.acquisition["current_clamp-response-01-ch-0"]
stim = nwbfile.stimulus["stimulus-01-ch-0"]

# Load all samples for the entire duration of the sweep
resp_data = resp.data[:] * resp.conversion  # voltage (V)
stim_data = stim.data[:] * stim.conversion  # current (A)
rate = resp.rate
N = len(resp_data)
t = (1.0 / rate) * np.arange(N)

# Plot both traces, aligned by time
fig, ax1 = plt.subplots(figsize=(10, 4))
color = 'tab:blue'
ax1.plot(t, resp_data, color=color, label='Membrane Potential (V)')
ax1.set_ylabel('Membrane Potential (V)', color=color)
ax1.set_xlabel('Time (s)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.plot(t, stim_data, color=color, label='Injected Current (A)', alpha=0.7)
ax2.set_ylabel('Injected Current (A)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_title('Current Clamp Sweep: Voltage and Stimulus\ncurrent_clamp-response-01-ch-0 / stimulus-01-ch-0 (Full Duration)')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation:**  
# The above plot shows the neuron's membrane potential (blue, left axis) and the injected current step protocol (red, right axis) for a single sweep. The protocol produces a step of negative current, with corresponding membrane hyperpolarization and recovery visible in the response.

# %% [markdown]
# ## 5. Visualizing Electrophysiology Data: Voltage Clamp Example
#
# The file contains multiple voltage clamp sweeps. Each consists of:
# - A response series (membrane current, in amperes)
# - Matched stimulus/command series (typically a command voltage waveform, in volts)
#
# Let's load and plot an example voltage clamp sweep and its command.

# %%
# Select a sample voltage clamp response and its matching stimulus
vc_response_key = "voltage_clamp-response-22-ch-0"
vc_stimulus_key = "stimulus-22-ch-0"

resp_vc = nwbfile.acquisition[vc_response_key]
stim_vc = nwbfile.stimulus[vc_stimulus_key]

resp_data_vc = resp_vc.data[:] * resp_vc.conversion  # current (A)
stim_data_vc = stim_vc.data[:] * stim_vc.conversion  # voltage (V)
rate_vc = resp_vc.rate
N_vc = len(resp_data_vc)
t_vc = (1.0 / rate_vc) * np.arange(N_vc)

fig, ax1 = plt.subplots(figsize=(10, 4))
color = 'tab:blue'
ax1.plot(t_vc, resp_data_vc, color=color, label='Membrane Current (A)')
ax1.set_ylabel('Membrane Current (A)', color=color)
ax1.set_xlabel('Time (s)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.plot(t_vc, stim_data_vc, color=color, label='Command Voltage (V)', alpha=0.7)
ax2.set_ylabel('Command Voltage (V)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_title('Voltage Clamp Sweep: Current Response and Command Waveform\nvoltage_clamp-response-22-ch-0 / stimulus-22-ch-0 (Full Duration)')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Observation:**  
# In some voltage clamp protocols, the command waveform may be flat (no voltage step), depending on the experimental design. The membrane current trace may still show spontaneous or baseline fluctuations.

# %% [markdown]
# ## 6. Exploring Protocol Diversity: Voltage Clamp Command Descriptions
#
# Let's inspect the available protocols for voltage clamp stimulus series in this file by listing their descriptions.

# %%
# List and print descriptions for all voltage clamp stimulus series
vc_stim_descriptions = {}
for stim_key in nwbfile.stimulus.keys():
    # Find relevant voltage clamp keys by matching range
    # The protocol descriptions help identify which key is optogenetic/laser, etc.
    if stim_key.startswith("stimulus-"):
        desc = nwbfile.stimulus[stim_key].description
        if "laser" in desc or "pulse" in desc or "steps" in desc:
            vc_stim_descriptions[stim_key] = desc

print("Voltage clamp command stimulus series and their descriptions:")
for k, desc in sorted(vc_stim_descriptions.items()):
    print(f"{k}: {desc}")

# %% [markdown]
# You can use these descriptions to match protocols of interest (e.g., laser pulses at specific voltages, 1Hz trains, patterned optogenetic stimulation).

# %% [markdown]
# ## 7. Summary and Next Steps
#
# In this notebook, you have learned how to:
# - Connect to a Dandiset and list available assets
# - Stream NWB files and inspect their content and metadata
# - Visualize example current clamp and voltage clamp sweeps, along with their command protocols
# - Navigate the diversity of protocols by reading stimulus descriptions
#
# **Key Takeaways:**
# - This NWB file provides patch clamp and metadata only; no calcium imaging or behavioral data are present here.
# - For calcium imaging or behavioral video, locate and explore a different file within the same Dandiset using the asset listing method above.
#
# **You now have a starting point for:**
# - Systematic sweep-by-sweep analysis
# - Grouping by protocol for reanalysis
# - Adaptation to other files, protocols, or data modalities
#
# Please consult DANDI documentation, the NWB format reference, and the Dandiset materials for additional details, and always review outputs critically.