**Dandiset 001349**
- Message 9: The problem is that the offset you applied is too large. Try without an offset.

**Dandiset 001354**
- Message 5: You should show the full duration for the sweeps.
- Message 10: This is taking too long.

**Dandiset 001433**
- Message 6: The data for inhalation and exhalation time is all 1's. The real information is in the timestamps, such as nwb.processing['behavior'].data_interfaces['inhalation_time'].timestamps[:]. These are in milliseconds. So the bottom plot is incorrect.
- Message 8: In the distribution of breathing intervals histogram (bottom plot) there is only one bar, presumably because all of the intervals have the same duration.

**Dandiset 000563**
- Message 8: The spike times may not start at zero. You should start the plot window at the earliest spike. The proper way to access the spike train is via nwb.units.spike_times_index[i], which gives a vector of spike times for the i^th unit. It is inefficient to load the entire units dataframe.
- Message 14: This is not a helpful plot. The raster looks too regular indicating something is wrong, and it's too sparse to provide a meaningful PSTH. Don't try a PSTH.

**Dandiset 001361**
- Message 11: Those ROIs don't line up with the image. I don't think you'll be able to get it to line up.

**Dandiset 001366**

**Dandiset 001359**
- Message 9: show the full duration of these sweeps
- Message 13: The detected spikes aren't showing up because they are relative to the sweep start time and then offset by +0.25
- Message 15: The vertical lines are still not showing up because trace has nan, so np.min(trace) and np.max(trace) are nan
- Message 19: it's not very helpful to show only the first 10

**Dandiset 001375**

**Dandiset 001174**

**Dandiset 000690**
- Message 8: This image is completely black

**Dandiset 001195**

**Dandiset 000617**
- Message 7: The ROIs don't line up at all with the background image.
- Message 9: They still do not line up at all. I don't think you are going to be able to get them to line up.
- Message 13: This was taking too long

