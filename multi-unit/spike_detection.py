# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:25:36 2024

@author: ssenapat

WHOLE NEW SCRIPT FOR SPIKE ONSET DETECTION
"""

def destination_dir():
    # folder where preprocessed data is to be saved
    
    # destdir = "/home/ssenapat/groups/PrimNeu/Final_exps_spikes/MUA/Elfie/p2_test"
    destdir = Path('G:/Final_exps_spikes/MUA/Elfie/p2_test')
    if not os.path.exists(destdir):     # if the required folder does not exist, create one
        os.mkdir(destdir)
    
    return destdir




def get_input_chan():
    
    # OPEN ONE CHANNEL RECORDING (one channel with all recordings)
    # channel ="/home/ssenapat/groups/PrimNeu/Final_exps_spikes/preprocessed_data/Elfie/p2_test/AP2/traces_cached_seg0.raw"
    # tt = np.load("/home/ssenapat/groups/PrimNeu/Final_exps_spikes/preprocessed_data/Elfie/p2_test/experiment_details.npy", allow_pickle=True)
    channel = Path("G:/Final_exps_spikes/preprocessed_data/Elfie/p2_test/AP4/traces_cached_seg0.raw")
    tt = np.load(Path("G:/Final_exps_spikes/preprocessed_data/Elfie/p2_test/experiment_details.npy"), allow_pickle=True)
    
    exp_details = pd.DataFrame(tt)
    
    exp_len = exp_details[0]
    exp_path = exp_details[1]
    
    sampl_freq = 30000.0
    datatyp = np.int16
    
    recording = se.read_binary(channel, sampling_frequency=sampl_freq, dtype=datatyp, num_channels=1)
    
    return channel, recording, exp_len, exp_path
    



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
        
        startind = locs[i]-75000
        if startind <= 0:
            startind = 0
            
        endind = locs[i]+75000
        if endind >= len(channel)-1:
            endind = len(channel)-1
            
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
    
    return channel, all_artifact_locs
    
    



def onset_detection(chan_array, lowlim, uplim):
    chan_df = pd.DataFrame(chan_array)
    
    p_spike_locs = list(np.where(chan_df>uplim)[0])
    n_spike_locs = list(np.where(chan_df<lowlim)[0])
    
    all_spikes_onsets_locs = np.concatenate((p_spike_locs, n_spike_locs))
    
    all_spikes_onsets_locs = np.sort(all_spikes_onsets_locs)
    
    return all_spikes_onsets_locs






def name_folder(path):
    # create results folder based on original data file
    folder = str(path)
    while "/" in folder:
      folder = folder.replace('/', '.')
      if ':' in folder:
          folder = folder.replace(':', '')
          
    while "\\" in folder:
      folder = folder.replace('\\', '.')
      if ':' in folder:
          folder = folder.replace(':', '')

          
    folder = folder.lower()
    folder_name_loc = re.search(r".p[0-9].\w+.", folder) # for penetrations <10
    # folder_name_loc = re.search(r".p[0-9]\w+.\w+.", folder) # for penetrations >=10
    ind = folder_name_loc.span()
    folder = folder[ind[0]+1:ind[1]-1]
    folder = folder.replace('.', '_')
          
    return folder



def name_channel(channelpath):
    # get the name of the channel currently being investigated
    chan = str(channelpath)
    while "/" in chan:
      chan = chan.replace('/', '.')
      if ':' in chan:
          chan = chan.replace(':', '')
          
    while "\\" in chan:
      chan = chan.replace('\\', '.')
      if ':' in chan:
          chan = chan.replace(':', '')
          
    folder_name_loc = re.search(r"(?:AP)\d+", chan)
    ind = folder_name_loc.span()
    folder = chan[ind[0]:ind[1]]
    
    return folder



def detect_onsets(rec):


    # # home = Path().resolve()     # current directory : directory where this script is saved ('D:/spike_analysis')
    # home = Path('/home/yhuang/groups/PrimNeu/Final_exps_spikes/MUA/Elfie/p22')
    # folder = name_folder(data_dir)
    
    # # NOTE: this folders were created during getting the metadata files, so they exist
    # save_files_here = Path(os.path.join(home, folder))
    
    # if not os.path.exists(save_files_here):     # if in any it does not exist, create one
    #     os.mkdir(save_files_here) 

    # get amplitudes from this channel
    print('Obtaining the amplitudes for this channel ...')
    chan = rec.get_traces()

    # artifact removal for this channel
    ref_mean = np.mean(abs(chan))
    chan_clean, artifact_locs = artifact_detection(chan, ref_mean)
    
    
    # # to check the traces for optimal artifact removal
    # fig, axes = plt.subplots(2)
    # axes[0].plot(chan)
    # axes[1].plot(chan_clean)
        
    # generate criteria for spike detection limits  
    mean_chan = np.nanmean(chan_clean)
    std_chan = np.nanstd(chan_clean)
        
    const = 4.0
        
    lowlim = mean_chan - const*std_chan
    uplim = mean_chan + const*std_chan
        
    # find the datapoints in this channel which are beyond the lower- and upper-limit
    spike_onsets = onset_detection(chan_clean, lowlim, uplim)
        
    return spike_onsets




def split_into_exps(onsets, explen, exppath, channelpath):
    
    outputdir = str(destination_dir())
    
    # get the cumulative length of the whole concatenated recording
    cumulative_explen = list(accumulate(explen, operator.add))

    spiketiming_df = pd.DataFrame(onsets, columns=['spike_onsetpoint'])


    folder = outputdir + '/spikeonsets_perExp_server'
    if not os.path.exists(folder):     # if the required folder does not exist, create one
        os.mkdir(folder)
        
        
    # splitting the spike train into different experiments
    for ii in range(0,len(cumulative_explen)):
        if ii == 0:
            # pick out range of spiketiming_df with spike_timepoint < exps_length[ii]
            expmnt = spiketiming_df.loc[spiketiming_df['spike_onsetpoint'] <= cumulative_explen[ii]].copy()
            
            expmnt['spike_onsetpoint_exp'] = expmnt['spike_onsetpoint']
            expmnt = expmnt[['spike_onsetpoint_exp', 'spike_onsetpoint']]
            
            expname = name_folder(exppath[ii])
            channame = name_channel(channelpath)
            expmnt = expmnt.to_numpy()
            np.save( folder + '/channel'+ channame +'_spikeonset_'+ expname +'.npy' , expmnt.astype(int) )
            # np.save( folder + '/spiketiming_exp_'+ str(ii+1) +'.npy' , expmnt.astype(int) )
            
        else:
            expmnt = spiketiming_df.loc[(spiketiming_df['spike_onsetpoint'] > cumulative_explen[ii-1]) & 
                            (spiketiming_df['spike_onsetpoint'] <= cumulative_explen[ii])].copy()
            
            expmnt_st = expmnt['spike_onsetpoint']
            expmnt_st = expmnt_st.apply(lambda expmnt_st: expmnt_st-cumulative_explen[ii-1])
            expmnt['spike_onsetpoint_exp'] = expmnt_st
            expmnt = expmnt[['spike_onsetpoint_exp', 'spike_onsetpoint']]
            
            expname = name_folder(exppath[ii])
            channame = name_channel(channelpath)
            expmnt = expmnt.to_numpy()
            np.save( folder + '/channel'+ channame +'_spikeonset_'+ expname +'.npy' , expmnt.astype(int) )
            # np.save( folder + '/spiketiming_exp_'+ str(ii+1) +'.npy' , expmnt.astype(int) )



def main():
    
    job_kwargs = dict(n_jobs=4, chunk_duration='1s', progress_bar=True)
    si.set_global_job_kwargs(**job_kwargs)

    # obtain the saved channel recording
    path_chan, rec_chan, exp_len, exp_path = get_input_chan()

    # # check the channel trace
    # %matplotlib qt
    # si.plot_traces(rec_chan)


    # obtain the spike onset timings for all experiments
    onsets = detect_onsets(rec_chan)

    # split the onsets into individual experiments and save the spike onset times for each expeeriment

    split_into_exps(onsets, exp_len, exp_path, path_chan)
    
    
    # continued in Matlab ...
    



# get all imports
import spikeinterface.full as si
from spikeinterface.preprocessing import bandpass_filter, common_reference
import spikeinterface.extractors as se

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
from itertools import accumulate
import operator

main()

