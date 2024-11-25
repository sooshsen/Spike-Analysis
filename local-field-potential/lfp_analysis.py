# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:16:18 2024

@author: ssenapat

Script for local field potential analysis of the music dataset 
NOTE: First have to finalise this pipeline for tuning curve data for obtaining auditory evoked potentials
"""

def load_trigger():
    '''
    Returns
    -------
    sorted_trigger_loc : TYPE
        Generates trigger information in increasing order of frequency
        thus, repeated freq are clustered together

    '''
    # read the saved trigger channel info
    file = Path('G:/Final_exps_spikes/LFP/Elfie/p2/p2_1_1/trigger_onset_for_py.npy')
    freq_order = Path('G:/Sushmita/spike_analysis/trigger_frequency_arrangements/for_p2_tc/et_400.npy')
    # file = '/home/ssenapat/groups/PrimNeu/Final_exps_spikes/LFP/Elfie/p1/p1_15/trigger_onset_for_py.npy'
    trigger_loc = np.load(file)
    trigger_freq_order = np.load(freq_order)
    triggers = pd.DataFrame(np.column_stack((trigger_freq_order,trigger_loc)))
    
    sorted_triggers = triggers.sort_values(by = [0], ignore_index = True)
    # sorted_trigger_loc = pd.DataFrame(sorted_triggers[1])
    sorted_trigger_loc = sorted_triggers[1].astype('int64')
    sorted_trigger_freq = np.unique(sorted_triggers[0].astype('int64'))
    
    return sorted_trigger_loc, sorted_trigger_freq



def load_recording():
    # load recording object using spike interface
    oe_folder = Path('G:/Aryo/copy_data/Elfie_final_exp_202303/p2/1_1/2023-03-20_23-38-09')
    # oe_folder = '/home/ssenapat/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p1/15/2023-03-20_21-42-51'

    raw_rec_lfp = si.read_openephys(oe_folder, 
                                    stream_id="1", 
                                    block_index=0)   # for music data
    recordinglfp_f = bandpass_filter(recording=raw_rec_lfp, 
                                     freq_min=3, 
                                     freq_max=250)
    recordinglfp_cmr = common_reference(recording=recordinglfp_f, 
                                        operator="median")
    
    return recordinglfp_cmr


def artifact_removal(channel, locs):
    # identify locations before and after the artifact positions
    # 2.5s before, 2.5s after (2.5*30000 = 75000 datapoints)
    # set these locations in the channel data to NaN
    
    channel = channel.astype(float)     # np.nan is a float, so conversion of channel to float is necessary before boolean masking
    # print('# of artifact locations:', len(locs))
    
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
    
    uplim = 25*ref_mean
    lowlim = -25*ref_mean
     
    p_artifact_locs = list(np.where(channel >= uplim)[0])
    n_artifact_locs = list(np.where(channel <= lowlim)[0])

    all_artifact_locs = np.concatenate((p_artifact_locs, n_artifact_locs))
    all_artifact_locs = np.unique(all_artifact_locs)
    
    channel = artifact_removal(channel, all_artifact_locs)  
    
    return channel


def save_plots(channel, channel_num, trigger_freq, savehere):
    
    chan_matrix = np.zeros(1000)

    for ii in range(len(trigger)):
        # relevant_points = channel[trigger[1].iloc[ii]-250:trigger[1].iloc[ii]+750]  # we look at 100 millisec before and 300 millisec after trigger onset
        relevant_points = channel[trigger[ii]-250:trigger[ii]+750]  # we look at 100 millisec before and 300 millisec after trigger onset
        chan_matrix = np.vstack([chan_matrix, relevant_points])
    
    chan_matrix = np.delete(chan_matrix, [0], axis=0) # remove 1st row, which is not crucial
    # to save this matrix
    DF = pd.DataFrame(chan_matrix)
    DF.to_csv(str(savehere) + '/channel' + str(channel_num) + '_allamps.csv')

    ### USE NUMPY ARRAY SPLIT() HERE
    # averaging across each datapoint in this matrix (40 plots per channel, since averaging to be done for every 10 consecutive tones)
    avg_every_tone = np.zeros(1000)
    
    for ii in range(40):
        mean_across_tone = np.mean(chan_matrix[ii:ii+10], axis=0)
        avg_every_tone = np.vstack([avg_every_tone, mean_across_tone])
        ii = ii+10
    
    avg_every_tone = np.delete(avg_every_tone, [0], axis=0) # remove 1st row, which is not crucial
        
        
    
    # #%matplotlib qt
   
    x = range(1000)
    fig, axs = plt.subplots(1, 2, figsize=(4, 10), dpi=80)
    
    # plot all avg amplitudes per channel
    for ii in range(40):
        axs[0].plot(x, avg_every_tone[ii] + ii*100, 'k')
    
    axs[0].axvline(x = 251, color = 'r', linestyle='dashed')
    
    x_ticks = np.arange(0, 1200, 250)
    x_ticklabels = ([-100, 0, 100, 200, 300])
    axs[0].set_xticks(x_ticks)
    axs[0].set_xticklabels(x_ticklabels)
    
    y_ticks = np.arange(0, 4000, 100)
    y_ticklabels = (trigger_freq)
    axs[0].set_yticks(y_ticks)
    axs[0].set_yticklabels(y_ticklabels)
    
    axs[0].set_ylim(-75, 4000)
    
    axs[0].set_xlabel('Time (in ms)')
    axs[0].set_ylabel('Frequency (in Hz)')
    # axs[0].set_title('Average amplitude per frequency')
    
    
    # heatmap
    # fig = plt.subplots(figsize=(5, 10), dpi=80)
    x_ticks = np.arange(0, 1200, 250)
    x_ticklabels = ([-100, 0, 100, 200, 300])
    
    axs[1] = sns.heatmap(np.flip(avg_every_tone, 0), yticklabels = np.flip(trigger_freq, 0), cmap="crest", vmax=200, vmin=-200)     # reorder the array for plotting purpose
    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels(x_ticklabels)
    axs[1].axvline(x = 251, color = 'w', linestyle='dashed')
    axs[1].set_xlabel('Time (in ms)')
    # axs[1].set_title('Average amplitude per frequency - heatmap')
    
    plt.savefig(str(savehere) + '/channel' + str(channel_num) + '.png')
    plt.close()
    
   

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import spikeinterface.full as si
from spikeinterface.preprocessing import bandpass_filter, common_reference



# def main():
    
recording = load_recording()
trigger, trigger_freq = load_trigger()
    
# steps before getting rid of artifacts
chans = recording.get_traces()      # #samples-by-#channels

chans_upd = np.zeros(len(chans))

for ii in range(len(chans.T)):
    ref_mean = np.mean(abs(chans[:,ii]))
        
    chans_clean = artifact_detection(chans[:,ii], ref_mean)
        
    chans_upd = np.vstack([chans_upd, chans_clean])
        
# remove top row and then transpose
chans_upd = (np.delete(chans_upd, [0], axis=0)).T


# plotting for each channel

savehere = Path('G:/Final_exps_spikes/LFP/Elfie/p2/p2_1_1/plots')

# for c in range(len(chans_upd.T)):
for c in range(384):
    
    chan = chans_upd[:,c]
    save_plots(chan, c+1, trigger_freq, savehere)





'''
#plot out a channel to set criteria for artifact detection
chans = recordinglfp_cmr.get_traces()
%matplotlib qt
plt.plot(chans[:,6][0:2500])
'''
