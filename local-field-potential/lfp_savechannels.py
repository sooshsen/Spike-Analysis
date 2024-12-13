# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:54:15 2024

@author: ssenapat

The channels are preprocessed : 1. Bandpass filter; 2. Common reference (with median)
Then saved as csv files for further processing in Python/Matlab.
"""

def destination_dir():
    # folder where preprocessed data is to be saved
    
    destdir = "/home/ssenapat/groups/PrimNeu/Final_exps_spikes/LFP/Elfie/p1/p1_15"
    # destdir = Path('G:/Final_exps_spikes/preprocessed_data/Elfie/p2_test')
    if not os.path.exists(destdir):     # if the required folder does not exist, create one
        os.mkdir(destdir)
    
    return destdir


def load_recording():
    # load recording object using spike interface
    # oe_folder = Path('G:/Aryo/copy_data/Elfie_final_exp_202303/p2/1_1/2023-03-20_23-38-09')
    oe_folder = '/home/ssenapat/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p1/15/2023-03-20_21-42-51'

    raw_rec_lfp = si.read_openephys(oe_folder, 
                                    stream_id="1", 
                                    block_index=0)   # for music data
    recordinglfp_f = bandpass_filter(recording=raw_rec_lfp, 
                                     freq_min=3, 
                                     freq_max=250)
    # recordinglfp_r = resample(recording=recordinglfp_f, resample_rate = 1000)
    recordinglfp_cmr = common_reference(recording=recordinglfp_f, 
                                        operator="median")
    
    return recordinglfp_cmr



def split_chans(recording):
    '''
    Gets preprocessed recording as input and saves each channel recording for all experiments

    Parameters
    ----------
    recording : preprocessed recording

    
    Saves the split recording, i.e. split into individual channels
    '''
    outputdir = destination_dir() + '/channels_preprocessed'
    if not os.path.exists(outputdir):     # if the required folder does not exist, create one
        os.mkdir(outputdir)
    
    length_of_rec = len(recording.get_channel_ids())
    channel_groups = np.repeat(range(0,length_of_rec),1)
    recording.set_property('group', channel_groups)
    
    split_rec = recording.split_by('group')
    
    ids = recording.get_channel_ids()
    
    # start_time = datetime.now()
    # save each channel for all experiments as a csv
    for ii in range(len(ids)):
        rec_chan = split_rec[ii].get_traces()
        np.savetxt( str(outputdir) + '/channelID_' + str(ids[ii]) + '.csv', rec_chan.astype(int), delimiter=",")
        
    # end_time = datetime.now()    
    # print('save channel - Duration: {}'.format(end_time - start_time))
    
    

import os
import numpy as np
import pandas as pd
# from pathlib import Path
# from datetime import datetime

import spikeinterface.full as si
from spikeinterface.preprocessing import bandpass_filter, common_reference


# main body
recording = load_recording()

# save the probe information for this penetration
savehere = destination_dir()
rec_probe = recording.get_probe().to_dataframe()
rec_probe.to_csv(str(savehere) + '/probe-info.csv', index=False)

# save each channel amplitudes in a folder
split_chans(recording)
