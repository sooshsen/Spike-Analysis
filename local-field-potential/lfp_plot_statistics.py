# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:11:45 2025

@author: ssenapat

read pickle files with pvalues to generate heatmap
"""

def load_probe():
    
    file = Path('G:/Final_exps_spikes/LFP/Elfie/p1/p1_15/probe-info.csv')
    prb = pd.read_csv(file)
    
    prb_ylocs = prb['y']    # column representing the depths of channels
    
    return prb_ylocs


def for_each_order(model):
    # orderXspeed
    orderXspeed = model['orderXspeed']

    p_oXs_o1 = []       # o1 = original
    p_oXs_o2 = []       # o2 = global reversed
    p_oXs_o3 = []       # o3 = local reversed

    for tw in orderXspeed.keys():
        # print(key)
        window = orderXspeed[tw][0]
        
        p_oXs_o1 = np.append(p_oXs_o1, min(window['original'][0]))
        p_oXs_o2 = np.append(p_oXs_o2, min(window['global_reversed'][0]))
        p_oXs_o3 = np.append(p_oXs_o3, min(window['local_reversed'][0]))
        
    return p_oXs_o1, p_oXs_o2, p_oXs_o3



def for_each_speed(model):
    # speedXorder
    speedXorder = model['speedXorder']

    p_sXo_s1 = []       # s1 = 66Hz
    p_sXo_s2 = []       # s2 = 75Hz
    p_sXo_s3 = []       # s3 = 85Hz

    for tw in speedXorder.keys():
        # print(key)
        window = speedXorder[tw][0]
        
        p_sXo_s1 = np.append(p_sXo_s1, min(window['66'][0]))
        p_sXo_s2 = np.append(p_sXo_s2, min(window['75'][0]))
        p_sXo_s3 = np.append(p_sXo_s3, min(window['85'][0]))
        
    return p_sXo_s1, p_sXo_s2, p_sXo_s3


def combine_p_channels(df, p_list, index):
    
    df.loc[index,:] = p_list
    return df
    

def percent_significance(df):
    
    percent_sig = []
    
    for tw in df.columns:
        subset = df[tw]
        significant_rows_num = Counter(subset < 0.05)[True]
        total_rows_num = len(subset)
        
        percent_sig_for_tw = (significant_rows_num/total_rows_num)*100
        
        #print(percent_sig_for_tw)
        
        # combine % significance of all time windows in a list
        percent_sig = np.append(percent_sig, percent_sig_for_tw)
    
    
    return percent_sig
        
        
    
    
    


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
from pathlib import Path

directory =  Path('G:/Final_exps_spikes/LFP/Elfie/p1/p1_15/significance_tests/ttest/')


timewindows = ['tw1', 'tw2', 'tw3', 'tw4','tw5','tw6','tw7','tw8','tw9','tw10','tw11','tw12']

# left
p_oXs_o1_df = pd.DataFrame(columns=timewindows)
p_oXs_o2_df = pd.DataFrame(columns=timewindows)
p_oXs_o3_df = pd.DataFrame(columns=timewindows)

# right
p_sXo_s1_df = pd.DataFrame(columns=timewindows)
p_sXo_s2_df = pd.DataFrame(columns=timewindows)
p_sXo_s3_df = pd.DataFrame(columns=timewindows)



# get channels based on depth
# get probe info
probe_locs = load_probe()



for chan in range(1,385):

    # checking one channel for now
    with open(str(directory) + '/channel' + str(chan) + '_interaction.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        file.close()
    
    channel_num = chan
    
    # obtain the p values for this channel
    p_oXs_o1, p_oXs_o2, p_oXs_o3 = for_each_order(loaded_model)
    p_sXo_s1, p_sXo_s2, p_sXo_s3 = for_each_speed(loaded_model)

    # left
    combine_p_channels(p_oXs_o1_df, p_oXs_o1, channel_num)
    combine_p_channels(p_oXs_o2_df, p_oXs_o2, channel_num)
    combine_p_channels(p_oXs_o3_df, p_oXs_o3, channel_num)

    # right
    combine_p_channels(p_sXo_s1_df, p_sXo_s1, channel_num)
    combine_p_channels(p_sXo_s2_df, p_sXo_s2, channel_num)
    combine_p_channels(p_sXo_s3_df, p_sXo_s3, channel_num)





# add probe info to rest of the data
p_oXs_o1_df['probeinfo'] = np.array(probe_locs)
p_oXs_o2_df['probeinfo'] = np.array(probe_locs)
p_oXs_o3_df['probeinfo'] = np.array(probe_locs)
p_sXo_s1_df['probeinfo'] = np.array(probe_locs)
p_sXo_s2_df['probeinfo'] = np.array(probe_locs)
p_sXo_s3_df['probeinfo'] = np.array(probe_locs)

# sort the data based on depth of channels
p_oXs_o1_df = p_oXs_o1_df.sort_values(by=['probeinfo']).astype(float)
p_oXs_o2_df = p_oXs_o2_df.sort_values(by=['probeinfo']).astype(float)
p_oXs_o3_df = p_oXs_o3_df.sort_values(by=['probeinfo']).astype(float)
p_sXo_s1_df = p_sXo_s1_df.sort_values(by=['probeinfo']).astype(float)
p_sXo_s2_df = p_sXo_s2_df.sort_values(by=['probeinfo']).astype(float)
p_sXo_s3_df = p_sXo_s3_df.sort_values(by=['probeinfo']).astype(float)

# drop the probe location column
p_oXs_o1_df.drop(['probeinfo'], axis=1, inplace=True)
p_oXs_o2_df.drop(['probeinfo'], axis=1, inplace=True)
p_oXs_o3_df.drop(['probeinfo'], axis=1, inplace=True)
p_sXo_s1_df.drop(['probeinfo'], axis=1, inplace=True)
p_sXo_s2_df.drop(['probeinfo'], axis=1, inplace=True)
p_sXo_s3_df.drop(['probeinfo'], axis=1, inplace=True)


# for significance percent calculation
sigpercent_p_oXs_o1 = percent_significance(p_oXs_o1_df)
sigpercent_p_oXs_o2 = percent_significance(p_oXs_o2_df)
sigpercent_p_oXs_o3 = percent_significance(p_oXs_o3_df)
sigpercent_p_sXo_s1 = percent_significance(p_sXo_s1_df)
sigpercent_p_sXo_s2 = percent_significance(p_sXo_s2_df)
sigpercent_p_sXo_s3 = percent_significance(p_sXo_s3_df)


# change the index for ease during plotting
p_oXs_o1_df.set_index(np.unique(probe_locs), inplace=True)
p_oXs_o2_df.set_index(np.unique(probe_locs), inplace=True)
p_oXs_o3_df.set_index(np.unique(probe_locs), inplace=True)
p_sXo_s1_df.set_index(np.unique(probe_locs), inplace=True)
p_sXo_s2_df.set_index(np.unique(probe_locs), inplace=True)
p_sXo_s3_df.set_index(np.unique(probe_locs), inplace=True)






    



# plotting
%matplotlib qt

fig, axs = plt.subplots(nrows=2, ncols=6, sharex=True, figsize=(20, 10))


### HEATMAP
sns.heatmap(p_oXs_o1_df, cmap="coolwarm", vmax=0.034, vmin=0.0005, ax=axs[0,0], cbar=False)
axs[0, 0].set_title('Original - pairwiseSpeeds')
axs[0, 0].set_ylabel('Depth')

sns.heatmap(p_oXs_o2_df, cmap="coolwarm", vmax=0.034, vmin=0.0005, ax=axs[0,1], cbar=False, yticklabels=False)
axs[0, 1].set_title('Global rev - pairwiseSpeeds')

sns.heatmap(p_oXs_o3_df, cmap="coolwarm", vmax=0.034, vmin=0.0005, ax=axs[0,2], cbar=False, yticklabels=False)
axs[0, 2].set_title('Local rev - pairwiseSpeeds')

sns.heatmap(p_sXo_s1_df, cmap="coolwarm", vmax=0.034, vmin=0.0005, ax=axs[0,3], cbar=False, yticklabels=False)
axs[0, 3].set_title('66 - pairwiseOrders')

sns.heatmap(p_sXo_s2_df, cmap="coolwarm", vmax=0.034, vmin=0.0005, ax=axs[0,4], cbar=False, yticklabels=False)
axs[0, 4].set_title('75 - pairwiseOrders')

sns.heatmap(p_sXo_s3_df, cmap="coolwarm", vmax=0.034, vmin=0.0005, ax=axs[0,5], yticklabels=False)
axs[0, 5].set_title('85 - pairwiseOrders')


for axisnum in range(6):
    axs[0, axisnum].invert_yaxis()
    axs[0, axisnum].axvline(x = 6, color = 'w', linestyle = '--', linewidth = 1)
    #axs[0, axisnum].set_xticks([0,300,600],['-300','0','300'])
    
    
    



### SIGNIFICANCE PERCENTAGE

axs[1,0].plot(sigpercent_p_oXs_o1, 'o-')
axs[1, 0].set_ylabel('% significance in each time window')

axs[1,1].plot(sigpercent_p_oXs_o2, 'o-')

axs[1,2].plot(sigpercent_p_oXs_o3, 'o-')

axs[1,3].plot(sigpercent_p_sXo_s1, 'o-')

axs[1,4].plot(sigpercent_p_sXo_s2, 'o-')

axs[1,5].plot(sigpercent_p_sXo_s3, 'o-')

for axisnum in range(6):
    axs[1, axisnum].set_ylim(0, 100)
    axs[1, axisnum].set_xlabel('Time windows')


fig.suptitle('Pairwise Significance', fontsize=16)
