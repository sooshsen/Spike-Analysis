# -*- coding: utf-8 -*-
# sampling rate : 30kHz

def get_inputs():
    # OPEN TXT FILE FOR PATH and save as numpy array
    # NOTE: save destinations.txt in folder with scripts
    paths_file = np.loadtxt("destinations.txt", comments='#', dtype='str')
    
    return paths_file

    

def si_analysis(oe_folder):
    # steps to bandpass filtering, and common median referencing
    raw_rec_AP = si.read_openephys(oe_folder, stream_id="0")    # for music data
    # raw_rec_AP = si.read_openephys(oe_folder, stream_id="0", block_index=0)  # for tuning curve
    recordingAP_f = bandpass_filter(recording=raw_rec_AP, 
                                    freq_min=500, 
                                    freq_max=5000)
    recordingAP_cmr = common_reference(recording=recordingAP_f, 
                                       operator="median")

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
    
    return channel



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



def get_artifacts(rec, data_dir):
    channelID = rec.get_channel_ids()
    
    home = Path().resolve()     # current directory : directory where this script is saved ('D:/spike_analysis')
    folder = name_folder(data_dir)
  
    save_files_here = Path(os.path.join(home, folder))
    if not os.path.exists(save_files_here):     # if the required file does not exist, create one
        os.mkdir(save_files_here)


    # empty lists for stats of importance
    mean_val =  []
    var_val = []
    num_datasample = []

    
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
        
                sampl_block = rec.get_traces(return_scaled=False, 
                                             start_frame=startP, 
                                             end_frame=endP, 
                                             channel_ids=[channelID[i]])
                chan = np.concatenate((chan, 
                                       sampl_block))
        
                j = endP
                del sampl_block 
        
            chan = np.delete(chan, (0))     # get rid of 1st element in the channel array which is an initiation element 


        # artifact removal for this channel
        ref_mean = np.mean(abs(chan))
        chan_clean = artifact_detection(chan, ref_mean)
    
        chan_clean_nan_ind = np.argwhere(np.isnan(chan_clean))  # identify indices of nan
        num_datasample.append(len(chan_clean) - len(chan_clean_nan_ind))
    
        mean_chan = np.nanmean(chan_clean)
        var_chan = np.nanvar(chan_clean)
    
        mean_val.append(mean_chan)
        var_val.append(var_chan)
    
        # np.save(os.path.join(savehere, "clean_channel%s" %i), chan_clean)   # if want to save channel data

    # saving statistics for this experiment to be used in spike onset detection
    stats = pd.DataFrame({'mean': mean_val, 
                          'variance': var_val, 
                          '#_valid_dataSamples': num_datasample})   
    np.save(os.path.join(save_files_here, 
                         'metaData'), stats)   # .npy file saved with the metadata for the experiment




def main():
    # main function body
    
    paths_array = get_inputs()
    
    for i in range(0,paths_array.size):
        root = Path(paths_array[i])
        # root = Path(paths_array.tolist())     # when working with single path in destinations.txt file
        # check if folder is empty
        is_not_empty = any(root.iterdir())      # is_not_empty is True if files present and False if no files
        # it is not yet checking the inside of the directories (directory inside directory) if it is empty 
        
        if is_not_empty:
            rec_obj = si_analysis(root)
            get_artifacts(rec_obj, root)
                       
        else:
            continue



# get all imports
import spikeinterface.full as si
from spikeinterface.preprocessing import bandpass_filter, common_reference

import numpy as np
import pandas as pd
from pathlib import Path
import os

main()
