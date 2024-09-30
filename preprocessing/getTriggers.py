# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:24:31 2024

@author: ssenapat

This script needs installation of open-ephys-python-tools module from Github

Can be used for saving MU, SU, LFP related triggers
Based on above, they are saved in different folders respectively

NOTE: Next, to automate this extraction using a destinations.txt like file

"""

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




def trigger_onset(trigger, savehere):
    # original
    trigger_original = np.append(trigger, 0)
    # phase-shifted trigger
    trigger_phaseshifted = np.insert(trigger, 0, 0)
    
    trigger_in = np.subtract(trigger_phaseshifted, trigger_original)
    
    # find the loci with value == 1
    trigger_loci = np.where(trigger_in == 1)
    
    # save the locations of the trigger onsets
    np.save(os.path.join(savehere, 'trigger_loci'), trigger_loci)
    
    


def save_AP_trigger_MU(trigger, source):
    # save as npy
    # dest_dir = '/home/ssenapat/groups/PrimNeu/Final_exps_spikes/MUA/Elfie/p1/'
    dest_dir = Path('G:/Final_exps_spikes/MUA/Elfie/p2')
    foldername = name_folder(source)
    
    savehere = Path(os.path.join(dest_dir, foldername))
    
    if not os.path.exists(savehere):     # if the required file does not exist, create one
        os.mkdir(savehere)
    
    np.save(os.path.join(savehere, 'trigger_ap'), trigger)
    
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
    
    trigger_onset(trigger, savehere)
    



def main():   
    ## LOCAL
    directory = Path('G:/Aryo/copy_data/Elfie_final_exp_202303/p2/1_1/2023-03-20_23-38-09')  # tuning curve
    # directory = Path('G:/Aryo/copy_data/Elfie_final_exp_202303/p1/15/2023-03-20_21-42-51')  # music data
    
    # SERVER
    # get experimetn path whose trigger you want to extract
    # directory = '/home/ssenapat/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p2/1_1/2023-03-20_23-38-09'
    # directory = '/home/ssenapat/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p1/1/2023-03-20_15-38-11'
    
    
    session = Session(directory)
    
    recording = session.recordnodes[0].recordings[0]
    
    # Extract trigger channel - not required to run after saving the trigger channel
    
    ap = recording.continuous[0].samples
    ap_trigger = ap[:, 384]
    
    lfp = recording.continuous[1].samples
    lfp_trigger = lfp[:, 384]
    
    
    
    
    ## uncomment based on requirement
    save_AP_trigger_MU(ap_trigger, directory)
    
    save_AP_trigger_SU(ap_trigger, directory)
    
    save_LFP_trigger(lfp_trigger, directory)
    
    # trg = pd.DataFrame(music_trigger)
   



from open_ephys.analysis import Session
from pathlib import Path
import numpy as np
import os
import re


# run 
main()
