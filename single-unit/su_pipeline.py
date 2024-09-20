# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:18:37 2024

@author: ssenapat

This script is for single unit analysis for sorting the units from all the 
experiments in a penetration and extract the waveforms
"""

def destination_dir():
    
    destdir = "/home/ssenapat/groups/PrimNeu/Final_exps_spikes/SUA/server/p2/exp_3_4/"
    # destdir = Path('G:/Final_exps_spikes/SUA/server/p2/test')
    if not os.path.exists(destdir):     # if the required folder does not exist, create one
        os.mkdir(destdir)
    
    return destdir

    

def get_inputs():
    
    # OPEN TXT FILE FOR EXPERIMENT and save as numpy array
    # NOTE: save experiments.txt in folder with scripts
    all_exps = np.loadtxt("/home/ssenapat/groups/PrimNeu/Sushmita/spike_analysis/metafiles_SU_Server/experiments_elfie_p2.txt", 
                          comments='#',
                          dtype='str')
    # all_exps = np.loadtxt(Path("G:/Sushmita/spike_analysis/metafiles_SU_local/experiments_elfie_p2.txt"), comments='#', dtype='str')
    
    return all_exps



def si_analysis_SU(exp_paths):
    '''
    Gets input from a user modified file and creates the recording object

    Parameters
    ----------
    exp_paths : list of path to all experiments for this penetration

    Returns
    -------
    recording_cmr : preprocessed recording object
    experiment_size : number of datasamples in each experiment

    '''
    
    outputdir = destination_dir()
    
    # if length > 1 , concatenate, otherwise dont
    if len(exp_paths) == 1:
        recording = si.read_openephys(outputdir, 
                                       stream_id="0")
        experiment_size = recording.get_num_samples()
        
    else:    
        rec = {}
        experiment_size = []
        for ii in range(0, exp_paths.size):
            rec["raw_rec{0}".format(ii)] =  si.read_openephys(exp_paths[ii], 
                                                              stream_id="0")
            experiment_size.append(rec["raw_rec{0}".format(ii)].get_num_samples())
        
        # combining multiple experiments
        recording = si.concatenate_recordings( list(rec.values()) )
    
    
    # save file
    exps = np.column_stack((experiment_size,exp_paths))     # col1 : number of samples in the experiment; col2 : path to the experiment
    np.save(outputdir + '/experiment_details.npy', exps)
    
    
    # bandpass filtering and common-median referencing
    recording_f = bandpass_filter(recording=recording, 
                                  freq_min=500, 
                                  freq_max=5000)
    recording_cmr = common_reference(recording=recording_f, 
                                     operator="median")
    
    # plot_traces(recording, recording_f, recording_cmr)  # uncomment if want to check the plots for the recordings
    
    return recording_cmr, experiment_size
    
    
def plot_traces(recording, recording_f, recording_cmr):
    ## Not mandatory
    
    fig, axs = plt.subplots(ncols=3, figsize=(20, 10))
    si.plot_traces(recording, 
                   backend='matplotlib',  
                   clim=(-50, 50), 
                   ax=axs[0])
    si.plot_traces(recording_f, 
                   backend='matplotlib',  
                   clim=(-50, 50), 
                   ax=axs[1])
    si.plot_traces(recording_cmr, 
                   backend='matplotlib',  
                   clim=(-50, 50), 
                   ax=axs[2])
    for i, label in enumerate(('raw','bp_filter', 'cmr')):
        axs[i].set_title(label)
        
    # plotting only some channels
    fig, ax = plt.subplots(figsize=(10, 10))
    some_chans = recording_cmr.channel_ids[[100, 150, 200, ]]
    si.plot_traces({'bp_filter':recording_f, 
                    'cmr': recording_cmr}, 
                   backend='matplotlib', 
                   mode='line', 
                   ax=ax, 
                   channel_ids=some_chans)
  



def sort_spikes(recording, outputdir):
    '''
    we are using Kilosort4 for spike sorting
    
    Parameters
    ----------
    recording : preprocessed recording object (combined recording from all experiments in a penetration)
    outputdir : path to directory where the sorting is to be saved

    Returns
    -------
    sorting : sorting object 

    '''
    
    k4_params = ss.Kilosort4Sorter.default_params()
    k4_params['Th_learned'] = 4      # modified threshold from 8 to 4
    
    sorting = ss.run_sorter('kilosort4', 
                            recording, 
                            verbose=True, 
                            output_folder=outputdir+'/sorting', 
                            remove_existing_folder=True, 
                            **k4_params)
    # remove excess spikes if any
    sorting = sc.remove_excess_spikes(sorting, recording)
    
    return sorting



def get_waveforms(recording, sorting):
    '''
    Paramters
    ---------
    recording : preprocessed recording object (combined recording from all experiments in a penetration)
    outputdir : path to directory where the sorting is to be saved
    sorting : 

    Returns
    -------
    None.

    '''
    outputdir = destination_dir()
    
    folder = outputdir + '/waveforms'
    we = extract_waveforms(recording, 
                           sorting, 
                           folder, 
                           ms_before=1.,
                           ms_after=1., 
                           max_spikes_per_unit=None, 
                           n_jobs=10, 
                           chunk_size=30000 , 
                           verbose=True
    )
    # print(we)
    
    return we




def check_quality(wave_forms):
    '''
    Quality checking to identify good units providing the most information

    Returns
    -------
    None.

    '''
    outputdir = destination_dir()
    
    print(qm.get_default_qm_params())      # check the default values before making any changes

    qm_of_interest = {
        'presence_ratio': {'bin_duration_s': 60},
        'snr': {'peak_sign': 'both',
         'peak_mode': 'extremum',
         'random_chunk_kwargs_dict': None},
        'isi_violation': {'isi_threshold_ms': 1, 'min_isi_ms': 0},
        'rp_violation': {'refractory_period_ms': 1.0, 'censored_period_ms': 0.0},
        'sliding_rp_violation': {'bin_size_ms': 0.25,
         'window_size_s': 1,
         'exclude_ref_period_below_ms': 0.5,
         'max_ref_period_ms': 10,
         'contamination_values': None},
        'amplitude_cutoff': {'peak_sign': 'neg',
         'num_histogram_bins': 100,
         'histogram_smoothing_value': 3,
         'amplitudes_bins_min_ratio': 5},
        'amplitude_median': {'peak_sign': 'neg'},
        'drift': {'interval_s': 60,
         'min_spikes_per_interval': 100,
         'direction': 'y',
         'min_num_bins': 2},
        'nearest_neighbor': {'max_spikes': 10000, 'n_neighbors': 5},
        'nn_isolation': {'max_spikes': 10000,
         'min_spikes': 10,
         'n_neighbors': 4,
         'n_components': 10,
         'radius_um': 100,
         'peak_sign': 'neg'},
        'nn_noise_overlap': {'max_spikes': 10000,
         'min_spikes': 10,
         'n_neighbors': 4,
         'n_components': 10,
         'radius_um': 100,
         'peak_sign': 'neg'}}

    metrics = qm.compute_quality_metrics(wave_forms, 
                                      metric_names=['num_spikes',
                                       'firing_rate',
                                       'presence_ratio',
                                       'snr',
                                       'isi_violation', # this will automatically compute : 'isi_violations_ratio' & 'isi_violations_count'.
                                       'amplitude_cutoff',
                                       'amplitude_median',
                                       'drift'],
                                      qm_params=qm_of_interest
                                      )
    
    # criteria for selecting good units
    keep = (metrics['snr'] > 2)
    good_metrics = metrics[keep]
    
    # save these two metrics files
    metrics.to_pickle( outputdir + '/qm.pkl')
    good_metrics.to_pickle( outputdir + '/qm_good.pkl')
    
    # save the IDs of the good units
    keep_uids = keep[keep].index.values
    np.save( outputdir + '/unitID_good.npy' , keep_uids )
    
    return metrics, good_metrics




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




def get_spiketrains(sorting, explen, expname):
    '''
    Timing information of spikes from all units, aiming to separate these spikes based on experiments they came from
    
    Parameters
    ----------
    sorting : sorting object obtained after running run_sorter()
    explen : list of number of datasamples in each experiment

    Returns
    -------
    None.

    '''
    
    outputdir = destination_dir()
    
    st_all = sorting.get_all_spike_trains()
    st_dim = np.hstack(st_all)      # row1: spike timing in terms of datapoint; row2: unit ID
    np.save(outputdir + '/spiketiming_allExp.npy' , st_dim.astype(int) )

    # np.amax(st_dim, axis=1)     # checking maximum values in each row
    
    # get the cumulative length of the whole concatenated recording
    cumulative_explen = list(accumulate(explen, operator.add))

    spiketiming_df = pd.DataFrame(np.transpose(st_dim), columns=['spike_timepoint', 'unitID'])


    folder = outputdir + '/spiketiming_perExp'
    if not os.path.exists(folder):     # if the required folder does not exist, create one
        os.mkdir(folder)
        
        
    # splitting the spike train into different experiments
    for ii in range(0,len(cumulative_explen)):
        if ii == 0:
            # pick out range of spiketiming_df with spike_timepoint < exps_length[ii]
            expmnt = spiketiming_df.loc[spiketiming_df['spike_timepoint'] <= cumulative_explen[ii]].copy()
            
            expmnt['spike_timepoint_exp'] = expmnt['spike_timepoint']
            expmnt = expmnt[['spike_timepoint_exp', 'spike_timepoint', 'unitID']]
            
            foldername = name_folder(expname[ii])
            expmnt = expmnt.to_numpy()
            np.save( folder + '/spiketiming_'+ foldername +'.npy' , expmnt.astype(int) )
            # np.save( folder + '/spiketiming_exp_'+ str(ii+1) +'.npy' , expmnt.astype(int) )
            
        else:
            expmnt = spiketiming_df.loc[(spiketiming_df['spike_timepoint'] > cumulative_explen[ii-1]) & 
                            (spiketiming_df['spike_timepoint'] <= cumulative_explen[ii])].copy()
            
            expmnt_st = expmnt['spike_timepoint']
            expmnt_st = expmnt_st.apply(lambda expmnt_st: expmnt_st-cumulative_explen[ii-1])
            expmnt['spike_timepoint_exp'] = expmnt_st
            expmnt = expmnt[['spike_timepoint_exp', 'spike_timepoint', 'unitID']]
            
            foldername = name_folder(expname[ii])
            expmnt = expmnt.to_numpy()
            np.save( folder + '/spiketiming_'+ foldername +'.npy' , expmnt.astype(int) )
            # np.save( folder + '/spiketiming_exp_'+ str(ii+1) +'.npy' , expmnt.astype(int) )
    



def main():
    
    outputdir = destination_dir()       # save all generated files in this directory
    
    all_exps = get_inputs()     # this array has information on paths to each experiment; can be used to create folder names
    
    recordingAP, exp_size = si_analysis_SU(all_exps)
    # save preprocessed recording 
    recordingAP_pps = recordingAP.save(folder=outputdir + '/preprocessing',  
                                       format='binary', 
                                       overwrite=True)
    
    job_kwargs = dict(n_jobs=50, chunk_duration='1s', progress_bar=True)
    si.set_global_job_kwargs(**job_kwargs)
    
    ## SORTING
    # ss.installed_sorters()  # check installed sorters
    sorting = sort_spikes(recordingAP_pps, outputdir)
    
    ## EXTRACT WAVEFORMS
    wfe = get_waveforms(recordingAP_pps, sorting)
    
    ## QUALITY METRICS
    metrics, good_metrics = check_quality(wfe)
    
    ## OBTAIN SPIKETRAINS PER EXPERIMENT
    get_spiketrains(sorting, exp_size, all_exps)
    
    
    # continued in Matlab after this ...
    
    
    


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.full as si
from spikeinterface.preprocessing import bandpass_filter, common_reference
import spikeinterface.sorters as ss
import spikeinterface.curation as sc
from spikeinterface import extract_waveforms
import spikeinterface.qualitymetrics as qm


from itertools import accumulate
import operator
# from pathlib import Path
import os
import re



# run the script for single unit sorting
main()
