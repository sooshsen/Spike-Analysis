# -*- coding: utf-8 -*-


def get_inputs():
    # OPEN TXT FILE FOR PATH and save as numpy array
    # NOTE: save destinations.txt in folder with scripts
    paths_file = np.loadtxt("D:\spike_analysis\destinations.txt", skiprows=1, dtype='str')
    
    return paths_file
    

def si_analysis(oe_folder):
    # steps to bandpass filtering, and common median referencing
    raw_rec_AP = si.read_openephys(oe_folder, stream_id="0")
    recordingAP_f = bandpass_filter(recording=raw_rec_AP, freq_min=500, freq_max=5000)
    recordingAP_cmr = common_reference(recording=recordingAP_f, operator="median")
    
    return recordingAP_cmr



def artifact_removal(channel, locs):
    # identify locations before and after the artifact positions
    # 2.5s before, 2.5s after (2.5*30000 = 75000 datapoints)
    # set these locations in the channel data to NaN
    
    channel = channel.astype(float)     # np.nan is a float, so conversion of channel to float is necessary before boolean masking
    print('# of artifact locations:', len(locs))
    
    # no artifacts, then return channel
    if len(locs)==0:
        return channel
    
    for i in range(0,len(locs)):
        
        if i == len(locs)-1:
            startind = locs[i]-75000
            if startind < 0:
                startind = 0
            endind = locs[i]
            ind = np.arange(startind, endind, dtype=int)
            channel[ind] = np.nan
            
        else:
            startind = locs[i]-75000
            if startind < 0:
                startind = 0
            endind = locs[i]+75000
            ind = np.arange(startind, endind, dtype=int)
            channel[ind] = np.nan
          
    return channel
        


def artifact_detection(channel, ref_mean):
    # function to identify locations with possible artifacts based on tc criteria
    # ref here is from tuning curve data for channel
    
    uplim = 15*ref_mean
    lowlim = -15*ref_mean
     
    p_artifact_locs = list(np.where(channel >= uplim)[0])
    n_artifact_locs = list(np.where(channel <= lowlim)[0])

    all_artifact_locs = np.concatenate((p_artifact_locs, n_artifact_locs))
    all_artifact_locs = np.unique(all_artifact_locs)
    
    channel = artifact_removal(channel, all_artifact_locs)  
    
    return channel



def onset_detection(chan_array, lowlim, uplim):
    chan_df = pd.DataFrame(chan_array)
    
    p_spike_locs = list(np.where(chan_df>uplim)[0])
    n_spike_locs = list(np.where(chan_df<lowlim)[0])
    
    all_spikes_onsets_locs = np.concatenate((p_spike_locs, n_spike_locs))
    
    all_spikes_onsets_locs = np.sort(all_spikes_onsets_locs)
    
    return all_spikes_onsets_locs   



def name_folder(path):
    # create results folder based on original data file location
    a = str(path)
    while "\\" in a:
      a = a.replace('\\', '_')
      if ':' in a:
          a = a.replace(':', '')
          
    return a


def get_artifacts(rec, data_dir, stats):
    channelID = rec.get_channel_ids()
    
    home = Path().resolve()     # current directory : directory where this script is saved ('D:/spike_analysis')
    folder = name_folder(data_dir)
    
    # NOTE: this folders were created during getting the metadata files, so they exist
    savehere = Path(os.path.join(home, 'results'))
    save_files_here = Path(os.path.join(savehere, folder))
    
    if not os.path.exists(save_files_here):     # if in any it does not exist, create one
        os.mkdir(save_files_here)

    
    for i in range(0,384):      # number of channels is fixed, i.e. 384
    
        # if small datasets
        if rec.get_num_samples() < 10000000:
            chan = rec.get_traces(channel_ids=[channelID[i]])
        
        # if large datasets - EXCLUSIVELY FOR THE MUSIC DATASET
        else:
            chan = np.eye(1, dtype='int') 
            j = 0
            while j < rec.get_num_samples():    # need to check datasample size (small like tc or huge like music) and based on that if-else loop
                startP = j
                endP = j+2000000
                if endP > rec.get_num_samples():    # end correction
                    endP = rec.get_num_samples()
        
                sampl_block = rec.get_traces(return_scaled=False, start_frame=startP, end_frame=endP, channel_ids=[channelID[i]])
                chan = np.concatenate((chan, sampl_block))
        
                j = endP
                del sampl_block 
        
            chan = np.delete(chan, (0))     # get rid of 1st element in the channel array which is an initiation element 


        # artifact removal for this channel
        ref_mean = np.mean(abs(chan))
        chan_clean = artifact_detection(chan, ref_mean)
        
        # based on input combinedStats file
        const = 4.0
        
        lowlim = stats.iloc[i,0] - const*stats.iloc[i,1]
        uplim = stats.iloc[i,0] + const*stats.iloc[i,1]
        
        
        # find the datapoints in this channel which are beyond the lower- and upper-limit
        spike_onsets = onset_detection(chan_clean, lowlim, uplim)
        
        # filename = "spikeOnsets_musicData_withCombinedThresh_%s" %i
        np.save(os.path.join(save_files_here, 'spikeonsets_channel_%s' %i), spike_onsets)
        
        del chan
    
        # np.save(os.path.join(savehere, "clean_channel%s" %i), chan_clean)   # if want to save channel data



def main():
    # main function body
    
    paths_array = get_inputs()
    combinedStats = np.load(Path('D:/Sushimita/python_analysis/combined_limits_for_spikes/tc_music_combined_29_36_stats.npy'))
    combinedStats = pd.DataFrame(combinedStats)
    
    for i in range(0,paths_array.size):
        # root = Path(paths_array[i])
        root = Path(paths_array.tolist())     # when working with single path in destinations.txt file
        # check if folder is empty
        is_not_empty = any(root.iterdir())      # is_not_empty is True if files present and False if no files
        
        if is_not_empty:
            rec_obj = si_analysis(root)
            get_artifacts(rec_obj, root, combinedStats)
                       
        else:
            continue



# get all imports
import spikeinterface.full as si
from spikeinterface.preprocessing import bandpass_filter, common_reference

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from pathlib import Path
import os

main()
