# Spike-Analysis

Contains scripts necessary for running downstream analysis on open-ephys data. 
Required external program installations: Spike Interface (https://github.com/SpikeInterface/spikeinterface.git)

1. PREPROCESSING
   Read open-ephys recording into python and perform (i) Bandpass Filtering, (ii) Common Median Referencing, and (iii) Drift Correction.
   Preprocessed recordings saved for each channel (experiments combined) for further analysis.

2. MULTI-UNIT ANALYSIS
   Spike onset detected after artifact removal from each good channel (preprocessed channel recordings); stored separately for each experiment.

3. SINGLE-UNIT ANALYSIS
   Pipeline for raw open-ephys recording data - to obtain spike onset timings for each experiment (within a penetration)
   


ARCHIVE:
1. Generate metadata per experiment (mean, variance, #ofdatasamples) - getStats.py
2. Combine metadata from multiple experiments - combineStats.py
3. Spike onset detection per experiment - detectOnsets.py

Additional helper scripts: readNPY.m and readNPYheader.m (from open-ephys-matlab-tools)
