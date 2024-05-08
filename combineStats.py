# -*- coding: utf-8 -*-
"""

COMBINE STATS FROM MULTIPLE EXPERIMENTS
Note: Currently able to combine statistics from two experiments


user input all the metaData files from the experiments in StepI

save a file similar to destinations.txt with the path to results/... folders, where each of them has metaData.npy file
"""

def get_inputs():
    # OPEN TXT FILE FOR PATH and save as numpy array
    paths_file = np.loadtxt("D:\spike_analysis\metaFiles.txt", comments='#', dtype='str')
    
    return paths_file

def combine_stats(all_mean, all_var, sampls, N):
    
    # NOTE: these experiment files are the stats files which has the columns
    # Columns: Experiments ; Rows: Channels (384)
    # We'll look at one channel at a time, i.e. for loop
    # N: number of experiments
    
    combined_mean = []
    combined_std = []
    
    numerator_mean = 0
    total_sampl = 0
    net_q = 0
    
    for channel in range(0,len(all_mean)):
        for exp in range(0,N):
            
            # mean
            product_mean_sampl = sampls(channel,exp)*all_mean(channel,exp)
            
            numerator_mean = numerator_mean + product_mean_sampl
            total_sampl = total_sampl + sampls(channel,exp)
            
            # variance
            q = (sampls(channel,exp) - 1)*all_var(channel,exp) + sampls(channel,exp)*(all_mean(channel,exp)**2)
            
            net_q = net_q + q;
        
            
        # mean
        net_mean_per_chan = numerator_mean/total_sampl
        combined_mean.append(net_mean_per_chan)
        
        # variance - stddev
        net_stddev_per_chan = math.sqrt((net_q - total_sampl*net_mean_per_chan**2)/(total_sampl-1))
        combined_std.append(net_stddev_per_chan)
       
        
        
    savehere = Path('D:/sushmitaS16/python_analysis/combined_limits_for_spikes')
    stats = pd.DataFrame({'mean': combined_mean, 'standard_dev': combined_std})   
    np.save(os.path.join(savehere, 'combined_stats'), stats)   # .npy file saved with the limits for the tuning curve data

    return stats
    
  
def main():
    paths_array = get_inputs()
    
    # stats_all = pd.DataFrame(columns=['mean','variance','numSamp'])
    mean_all = pd.DataFrame()
    var_all = pd.DataFrame()
    num_sampl = pd.DataFrame()

    for i in range(0,len(paths_array)):
        file = pd.DataFrame(np.load(Path(paths_array[i])))
        
        mean_all = pd.concat([mean_all, file.iloc[:,0]], axis=1, ignore_index=True)
        var_all = pd.concat([var_all, file.iloc[:,1]], axis=1, ignore_index=True)
        num_sampl = pd.concat([num_sampl, file.iloc[:,2]], axis=1, ignore_index=True)
        
        
    # call function
    combine_stats(mean_all, var_all, num_sampl, len(mean_all.columns))
    

  
# Import all dependencies
import os
import math
import numpy as np
import pandas as pd
from pathlib import Path


main()
