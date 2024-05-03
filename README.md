# spike-analysis

This folder contains scripts necessary for running downstream analysis on open-ephys data
Required external program installations: Spike Interface

Scripts in order:
1. Generate metadata per experiment (mean, variance, #ofdatasamples) - getStats.py
2. Combine stats from multiple experiments
3. Spike onset detection per experiment

Additional helper scripts:
1. readNPY.m and readNPYheader.m (from open-ephys-matlab-tools)
