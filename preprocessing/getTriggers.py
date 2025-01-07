# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:24:31 2024

@author: ssenapat

This script needs installation of open-ephys-python-tools module from Github

Can be used for saving MU, SU, LFP related triggers
Based on above, they are saved in different folders respectively

Note: This does not take a lot of time for computation, hence can be run in the local PC


"""

def get_inputs():
    
    # OPEN TXT FILE FOR EXPERIMENT and save as numpy array
    all_exps = np.loadtxt(Path("G:/Sushmita/spike_analysis/metafiles_MU_TC_only/benny_for_triggers.txt"), comments='#', dtype='str')
    
    return all_exps


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
    #folder_name_loc = re.search(r".p[0-9]\w+.\w+.", folder) # for penetrations >=10
    ind = folder_name_loc.span()
    folder = folder[ind[0]+1:ind[1]-1]
    folder = folder.replace('.', '_')
          
    return folder




def trigger_onset(trigger, savehere):
    # 
    trigger_diff = np.diff(trigger, n=1)   # trigger_clean[i+1] - trigger_clean[i] 
    
    # find indices where the values are 1 after the subtraction
    trigger_onset_loc = np.where(trigger_diff == 1)[0]
    trigger_offset_loc = np.where(trigger_diff == -1)[0]
    
    # check for possible artifacts
    trigger_off_on_diff = trigger_offset_loc - trigger_onset_loc
    threshold = 20
    
    trigger_onset_loc_clean = trigger_onset_loc[trigger_off_on_diff > threshold]
    
    
    # separate for further python-based and matlab-based analysis
    trigger_ind_for_py = trigger_onset_loc_clean + 1
    trigger_ind_for_matlab = trigger_onset_loc_clean + 2        # NOTE: In Matlab, index starts from 1 (not 0)
    
    # save the locations of the trigger onsets
    np.save(os.path.join(savehere, 'ss_trigger_onset_for_py'), trigger_ind_for_py)
    np.savetxt(os.path.join(savehere, 'ss_trigger_onset_for_matlab.csv'), trigger_ind_for_matlab)
    #np.save(os.path.join(savehere, 'trigger_onset_for_matlab'), trigger_ind_for_matlab)
    print('Corresponding trigger onset locations saved...')
    
    
    


def save_AP_trigger_MU(trigger, source):
    # save as npy
    # dest_dir = '/home/ssenapat/groups/PrimNeu/Final_exps_spikes/MU_TC_only/Benny'
    dest_dir = Path('G:/Final_exps_spikes/MU_TC_only/Benny')
    foldername = name_folder(source)
    
    savehere = Path(os.path.join(dest_dir, foldername))
    
    if not os.path.exists(savehere):     # if the required file does not exist, create one
        os.mkdir(savehere)
    
    np.save(os.path.join(savehere, 'ss_trigger_ap'), trigger)
    np.savetxt(os.path.join(savehere, 'ss_trigger_ap.csv'), trigger)
    print('\n')
    print('Trigger channel amplitudes (AP) saved for multi-unit analysis')
    
    trigger_onset(trigger, savehere)
 

   
def save_AP_trigger_SU(trigger, source):
    # save as npy
    # dest_dir = '/home/ssenapat/groups/PrimNeu/Final_exps_spikes/SUA/server/Elfie/p1/'
    dest_dir = Path('G:/Final_exps_spikes/SUA/server/Elfie/p2')
    foldername = name_folder(source)
    
    savehere = Path(os.path.join(dest_dir, foldername))
    
    if not os.path.exists(savehere):     # if the required file does not exist, create one
        os.mkdir(savehere)
    
    np.save(os.path.join(savehere, 'trigger_ap'), trigger)
    print('\n')
    print('Trigger channel amplitudes (AP) saved for single-unit analysis')
    
    trigger_onset(trigger, savehere)



def save_LFP_trigger(trigger, source):
    # save as npy
    # dest_dir = '/home/ssenapat/groups/PrimNeu/Final_exps_spikes/LFP/Elfie/p1/'
    dest_dir = Path('G:/Final_exps_spikes/LFP/Elfie/p2')
    foldername = name_folder(source)
    
    savehere = Path(os.path.join(dest_dir, foldername))
    
    if not os.path.exists(savehere):     # if the required file does not exist, create one
        os.mkdir(savehere)
    
    np.save(os.path.join(savehere, 'trigger_lfp'), trigger)
    print('\n')
    print('Trigger channel amplitudes (LFP) saved')
    
    trigger_onset(trigger, savehere)
    



def main():
    
    all_exps = get_inputs() 
    
    
    for i in range(0,all_exps.size):
        
        if all_exps.size == 1:
            directory = Path(all_exps.tolist())     # when working with single path in destinations.txt file
        else:
            directory = Path(all_exps[i])
        
        print(directory)
        session = Session(directory)
        recording = session.recordnodes[0].recordings[0]
        
        
        # Extract trigger channel - not required to run after saving the trigger channel
        
        ap = recording.continuous[0].samples
        ap_trigger = ap[:, 384]
        
        lfp = recording.continuous[1].samples
        lfp_trigger = lfp[:, 384]
        
        
        ## uncomment based on requirement
        save_AP_trigger_MU(ap_trigger, directory)
        
        # save_AP_trigger_SU(ap_trigger, directory)
        
        #save_LFP_trigger(lfp_trigger, directory)
        
        # trg = pd.DataFrame(music_trigger)
        
       



from open_ephys.analysis import Session
from pathlib import Path
import numpy as np
import os
import re


# run 
main()
