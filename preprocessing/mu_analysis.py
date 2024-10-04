#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:52:18 2024

@author: ssenapat

WHOLE NEW SCRIPT WITH DRIFT CORRECTION FOR MULTI-UNIT ANALYSIS
"""

def destination_dir():
    # folder where preprocessed data is to be saved
    
    # destdir = "/home/ssenapat/groups/PrimNeu/Final_exps_spikes/preprocessed_data/Elfie/p2"
    destdir = Path('G:/Final_exps_spikes/preprocessed_data/Elfie/p2_test')
    if not os.path.exists(destdir):     # if the required folder does not exist, create one
        os.mkdir(destdir)
    
    return destdir


def get_inputs():
    
    # OPEN TXT FILE FOR EXPERIMENT and save as numpy array
    # NOTE: save experiments.txt in folder with scripts
    # all_exps = np.loadtxt("/home/ssenapat/groups/PrimNeu/Sushmita/spike_analysis/metafiles_Server/destinations_elfie_p2.txt", 
    #                       comments='#',
    #                       dtype='str')
    all_exps = np.loadtxt(Path("G:/Sushmita/spike_analysis/metafiles_Local/destinations_elfie_p2.txt"), comments='#', dtype='str')
    
    return all_exps

'''
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
'''


def si_analysis_pps(exp_paths):
    '''
    Gets input from a user modified file and creates the recording object

    Parameters
    ----------
    exp_paths : list of path to all experiments for this penetration

    Returns
    -------
    recording_dc : preprocessed recording object

    '''
    
    # if length > 1 , concatenate, otherwise dont  # i.e. if there is only one path provided in the path file
    if exp_paths.size == 1:
        exp_path_list = exp_paths.tolist()
        recording = si.read_openephys(exp_path_list, 
                                       stream_id="0")
        
    else:    
        rec = {}
        for ii in range(0, exp_paths.size):
            rec["raw_rec{0}".format(ii)] =  si.read_openephys(exp_paths[ii], 
                                                              stream_id="0")
        
        # combining multiple experiments
        recording = si.concatenate_recordings( list(rec.values()) )
    
    print('Number of segments in this recording: ', recording.get_num_segments())
    
    # bandpass filtering
    recording_f = bandpass_filter(recording=recording, 
                                  freq_min=500, 
                                  freq_max=5000)
    # common-median referencing
    recording_cmr = common_reference(recording=recording_f, 
                                     operator="median")
    # drift correction
    # motion_dir = outputdir + '/motion/nonrigid_acc/'
    recording_dc = correct_motion(recording=recording_cmr, preset='nonrigid_accurate')
    # NOTE: After drift correction, some channels might be removed

    
    return recording_cmr, recording_dc




def split_chans(recording):
    '''
    Gets preprocessed recording as input and saves each channel recording for all experiments

    Parameters
    ----------
    recording : preprocessed recording
    
    Saves the split recording, i.e. split into individual channels
    '''
    outputdir = destination_dir()
    
    length_of_rec = len(recording.get_channel_ids())
    channel_groups = np.repeat(range(0,length_of_rec),1)
    recording.set_property('group', channel_groups)
    
    split_rec = recording.split_by('group')
    
    ids = recording.get_channel_ids()

    # save each channel for all experiments as a recording object
    for ii in range(0,len(recording.get_channel_ids())):
        rec = split_rec[ii]
        recording_pps = rec.save(folder=str(outputdir) + '/' + str(ids[ii]),
                                            format='binary', 
                                            overwrite=True)
    
    return split_rec
    



def main():
    all_exps = get_inputs() 

    # outputdir = destination_dir()
    # foldername = name_folder(all_exps[0])

    job_kwargs = dict(n_jobs=50, 
                      chunk_duration='1s', 
                      progress_bar=True)
    si.set_global_job_kwargs(**job_kwargs)


    print('\n')
    ## GET RECORDING OBJECT
    recording_cmr, recording_dc = si_analysis_pps(all_exps)

    print('\n')
    # split recording based on channel
    split_recording = split_chans(recording_dc)
    
    # Next step would be to detect spike onsets for each channel ...
    


# imports
import spikeinterface.full as si
from spikeinterface.preprocessing import bandpass_filter, common_reference, correct_motion

import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from pathlib import Path
import os
import re


# Run
main()
