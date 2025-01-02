# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:44:11 2024

@author: ssenapat

THIS IS CURRENTLY FOR MUSIC DATASET ONLY
"""

def load_trigger():
    '''
    Returns
    -------
    sorted_trigger_loc : TYPE
        Generates trigger information for music pieces

    '''
    # read the saved trigger channel info
    file = '/home/ssenapat/groups/PrimNeu/Final_exps_spikes/LFP/Elfie/p1/p1_15/trigger_onset_for_py.npy'
    # file = Path('G:/Final_exps_spikes/LFP/Elfie/p1/p1_15/trigger_onset_for_py.npy')
    triggers = pd.DataFrame(np.load(file))
    
    return triggers



def load_channel(channel):
    '''
    Function to read one csv file at a time from the path
    
    Returns
    -------
    Pandas series of amplitudes for one channel

    '''
    channel_csv = pd.read_csv(channel, header=None)        #NOTE: This will be saved as a column vector, i.e. #sample-by-1
    
    return channel_csv
    


def artifact_removal(channel, locs):
    '''
    Identify locations before and after the artifact positions:
    2.5s before, 2.5s after (2.5*2500 = 6250 datapoints)
    set these locations in the channel data to NaN
    '''
    
    channel = channel.astype(float)     # np.nan is a float, so conversion of channel to float is necessary before boolean masking
    # print('# of artifact locations:', len(locs))
    
    # no artifacts, then return channel
    if len(locs)==0:
        return channel
    
    for i in range(0,len(locs)):
        
        startind = locs[i]-6250
        if startind <= 0:
            startind = 0
            
        endind = locs[i]+6250
        if endind >= len(channel)-1:
            endind = len(channel)-1
            
        ind = np.arange(startind, endind, dtype=int)
        channel[ind] = np.nan
        
          
    return channel
        


def artifact_detection(channel, ref_mean):
    '''
    Function to identify locations with possible artifacts based on tc criteria
    Ref here is from tuning curve data for channel
    '''
    
    uplim = 25*ref_mean
    lowlim = -25*ref_mean
     
    p_artifact_locs = list(np.where(channel >= uplim)[0])
    n_artifact_locs = list(np.where(channel <= lowlim)[0])

    all_artifact_locs = np.concatenate((p_artifact_locs, n_artifact_locs))
    all_artifact_locs = np.unique(all_artifact_locs)
    
    channel = artifact_removal(channel, all_artifact_locs)  
    
    return channel


def identify_channel(filepath):
    
    folder = str(filepath)     

    folder_name_loc = re.search(r"LFP[0-9]\w+", folder)
    ind = folder_name_loc.span()
    channelID = folder[ind[0]+3:ind[1]]        # get rid of the 'LFP' at the start of the channel ID name
    
    return int(channelID)




def channeldata_per_trial_onset(channel, channel_num, trigger, savehere):
    
    '''
    NOTE: Sampling rate of LFP is 2.5kHz or 2500Hz
    For music data, inter-trigger-interval is ~60s 
    Hence, we will look at 5s after trigger onset (5*2500=12500), and 2s before trigger onset (2*2500=5000)
    
    For tuning curve,  we look at 100 millisec (100*2.5=250) before and 300 millisec (300*2.5=750) after trigger onset
    
    '''
    chan_matrix = np.zeros(17500)       # for music data currently (12500 + 5000)
    # chan_matrix = np.zeros(1000)       # for tc data currently (250 + 750)
    
    # for music
    dp_pre_onset = 5000
    dp_post_onset = 12500

    for ii in range(len(trigger)):
        # relevant_points = channel[trigger[1].iloc[ii]-250:trigger[1].iloc[ii]+750]
        relevant_start = trigger.iloc[ii] - dp_pre_onset
        relevant_end = trigger.iloc[ii] + dp_post_onset
        relevant_points = channel[relevant_start[0]:relevant_end[0]].T
        chan_matrix = np.vstack([chan_matrix, relevant_points])
    
    chan_matrix = np.delete(chan_matrix, [0], axis=0) # remove 1st row, which is not crucial
    
    save_loc = str(savehere) + '/onset_responses'
    if not os.path.exists(save_loc):     # if the required folder does not exist, create one
        os.mkdir(save_loc)
    
    # to save this matrix
    channel_df = pd.DataFrame(chan_matrix)
    channel_df.to_csv(str(save_loc) + '/channel' + str(channel_num) + '_alltones_allamps.csv')




import os
import numpy as np
import pandas as pd
from pathlib import Path
import re

import spikeinterface.full as si


# main body
trigger = load_trigger()

directory = '/home/ssenapat/groups/PrimNeu/Final_exps_spikes/LFP/Elfie/p1/p1_15/channels_preprocessed/'
# directory = Path('G:/Final_exps_spikes/LFP/Elfie/p1/p1_15/channels_preprocessed/')
  
# steps before getting rid of artifacts
for chan in os.listdir(directory):
    filepath = str(directory) + '/' + chan
    channel = load_channel(filepath)      # #samples-by-1
    chan_counter = identify_channel(filepath)       # identify current channel based on filepath
    
    # detect artifact and correct them for this channel
    ref_mean = np.mean(abs(channel))    
    channel_clean = artifact_detection(channel, ref_mean)
    
    # for each channel
    # savehere = Path('G:/Final_exps_spikes/LFP/Elfie/p1/p1_15/')
    savehere = '/home/ssenapat/groups/PrimNeu/Final_exps_spikes/LFP/Elfie/p1/p1_15/'

    channeldata_per_trial_onset(channel_clean, chan_counter, trigger, savehere)
