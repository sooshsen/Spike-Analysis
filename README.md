# spike-analysis

Contains scripts necessary for running downstream analysis on open-ephys data
Required external program installations: Spike Interface

Scripts in order:
1. Generate metadata per experiment (mean, variance, #ofdatasamples) - getStats.py
2. Combine metadata from multiple experiments - combineStats.py
3. Spike onset detection per experiment - detectOnsets.py

Additional helper scripts:
1. readNPY.m and readNPYheader.m (from open-ephys-matlab-tools)
