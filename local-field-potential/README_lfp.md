Below are the steps for LFP analysis:
 - Path to scripts: G:\[username]\spike_analysis\python_scripts\LFP) 
 - All raw channels saved in G:\Final_exps_spikes\LFP\... (corresponding to the macaque name, penetration, and experiment)



1. [Slow process - Run on server] Run lfp_savechannels.py to save the whole dataframe as individual channels (Note that these channels have not undergone artifact removal)

Note: The probe information will also be saved using this script

PS : Update path of recording to be analyzed; path to where outputs are to be saved

2. [Fast process - Run on local system] Run lfp_onset_response.py to clean the individual channels of artifacts and slice out the data few milliseconds before and after the trigger onset


3. [Fast process - Run on local system] Run lfp_plot_overview.py to plot out the trigger-onset-data (NOTE that these are plots to get an overview of the dataset, i.e. how the recorded activity is across channels)

- Further statistical analyses done using relevant scripts in this folder - Check them for comments!
