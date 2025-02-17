# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:11:45 2025

@author: ssenapat

Read pickle files with pvalues (results from lfp_ttest.py) to generate heatmap
Also, percentage of significance measured (directionality of effect not considered)
{For directionality effect in pairwise comparison, 
 check scripts lfp_pairwise_direction_effect.py and lfp_plot_pairwise_direction_effect.py}
"""


def load_probe():
    
    file = Path('G:/Final_exps_spikes/LFP/Elfie/p1/p1_15/probe-info.csv')
    prb = pd.read_csv(file)
    prb_ylocs = prb['y']    # column representing the depths of channels
    return prb_ylocs


def combine_channels(df, val_list, index):
    
    df.loc[index,:] = val_list
    return df


def add_depth_info(df, depths):
    
    # add probe info to rest of the data
    df['probeinfo'] = np.array(depths)
    
    # sort the data based on depth of channels
    df = df.sort_values(by=['probeinfo']).astype(float)
    
    # drop the probe location column
    df.drop(['probeinfo'], axis=1, inplace=True)
    
    # change the index for ease during plotting
    df.set_index(np.unique(depths), inplace=True)
    
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


'''
### FOR HEATMAP plots : Only picking up the minimum p-value without caring about which pairwise comparison it belongs to
'''
def minp_each_factor(model):
    
    p_factor1 = []       # o1 = original;           s1 = 66Hz
    p_factor2 = []       # o2 = global reversed;    s2 = 75Hz
    p_factor3 = []       # o3 = local reversed;     s3 = 85Hz

    for tw in model.keys():
        # print(key)
        window = model[tw][0]
        
        p_factor1 = np.append(p_factor1, min(window[list(window)[0]][0]))
        p_factor2 = np.append(p_factor2, min(window[list(window)[1]][0]))
        p_factor3 = np.append(p_factor3, min(window[list(window)[2]][0]))
        
    return p_factor1, p_factor2, p_factor3

        



'''
### FOR %SIGNIFICANCE plots : obtain a df for each pairwise comparison pvalue   
'''
# p_factor1 = {}
# p_factor2 = {}
# p_factor3 = {}
def everyp_each_factor(model):
    
    
    p_factor1_12 = []       # o1 = original; pair s1-s2
    p_factor1_13 = []       # pair s1-s3
    p_factor1_23 = []       # pair s2-s3
    
    p_factor2_12 = []       # o2 = global reversed
    p_factor2_13 = []       
    p_factor2_23 = []       
    
    p_factor3_12 = []       # o3 = local reversed
    p_factor3_13 = []       
    p_factor3_23 = []       
    
    for tw in model.keys():
        #print(tw)
        window = model[tw][0]
        
        # factor1
        p_factor1_12 =  np.append(p_factor1_12, window[list(window)[0]][0][0])
        p_factor1_13 =  np.append(p_factor1_13, window[list(window)[0]][0][1])
        p_factor1_23 =  np.append(p_factor1_23, window[list(window)[0]][0][2])
        
        # factor 2
        p_factor2_12 =  np.append(p_factor2_12, window[list(window)[1]][0][0])
        p_factor2_13 =  np.append(p_factor2_13, window[list(window)[1]][0][1])
        p_factor2_23 =  np.append(p_factor2_23, window[list(window)[1]][0][2])
        
        # factor 3
        p_factor3_12 =  np.append(p_factor3_12, window[list(window)[2]][0][0])
        p_factor3_13 =  np.append(p_factor3_13, window[list(window)[2]][0][1])
        p_factor3_23 =  np.append(p_factor3_23, window[list(window)[2]][0][2])
        
    # return the 6 lists (each list of length 12)
    return p_factor1_12, p_factor1_13, p_factor1_23, p_factor2_12, p_factor2_13, p_factor2_23, p_factor3_12, p_factor3_13, p_factor3_23





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
from pathlib import Path
import os


directory =  Path('G:/Final_exps_spikes/LFP/Elfie/p1/p1_15/')
directory_ttest = Path(str(directory) + '/significance_tests/ttest/')
directory_downsizedonsets = Path(str(directory) + '/donwsampled_onset_responses/')
savehere = Path('G:/Final_exps_spikes/LFP/Elfie/p1/p1_15/plots/')

timewindows = ['tw1', 'tw2', 'tw3', 'tw4','tw5','tw6','tw7','tw8','tw9','tw10','tw11','tw12']

# orderXspeed
p_o1_df = pd.DataFrame(columns=timewindows)
p_o2_df = pd.DataFrame(columns=timewindows)
p_o3_df = pd.DataFrame(columns=timewindows)

# order1
p_o1_12_df = pd.DataFrame(columns=timewindows)
p_o1_13_df = pd.DataFrame(columns=timewindows)
p_o1_23_df = pd.DataFrame(columns=timewindows)
# order2
p_o2_12_df = pd.DataFrame(columns=timewindows)
p_o2_13_df = pd.DataFrame(columns=timewindows)
p_o2_23_df = pd.DataFrame(columns=timewindows)
# order3
p_o3_12_df = pd.DataFrame(columns=timewindows)
p_o3_13_df = pd.DataFrame(columns=timewindows)
p_o3_23_df = pd.DataFrame(columns=timewindows)

# speedXorder
p_s1_df = pd.DataFrame(columns=timewindows)
p_s2_df = pd.DataFrame(columns=timewindows)
p_s3_df = pd.DataFrame(columns=timewindows)

# speed1
p_s1_12_df = pd.DataFrame(columns=timewindows)
p_s1_13_df = pd.DataFrame(columns=timewindows)
p_s1_23_df = pd.DataFrame(columns=timewindows)
# speed2
p_s2_12_df = pd.DataFrame(columns=timewindows)
p_s2_13_df = pd.DataFrame(columns=timewindows)
p_s2_23_df = pd.DataFrame(columns=timewindows)
# speed3
p_s3_12_df = pd.DataFrame(columns=timewindows)
p_s3_13_df = pd.DataFrame(columns=timewindows)
p_s3_23_df = pd.DataFrame(columns=timewindows)




# get probe info
probe_locs = load_probe()

for chan in range(1,385):
    
    channel_num = chan

    # checking one channel to get associated pvalues (output from lfp_ttest.py)
    with open(str(directory_ttest) + '/channel' + str(chan) + '_interaction.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        file.close()
    
    
    ### HEATMAP with minimum pvalue
    # obtain the minimum p-value for each factor for this channel
    p_o1, p_o2, p_o3 = minp_each_factor(loaded_model['orderXspeed'])
    p_s1, p_s2, p_s3 = minp_each_factor(loaded_model['speedXorder'])

    # orderXspeed
    combine_channels(p_o1_df, p_o1, channel_num)
    combine_channels(p_o2_df, p_o2, channel_num)
    combine_channels(p_o3_df, p_o3, channel_num)

    # speedXorder
    combine_channels(p_s1_df, p_s1, channel_num)
    combine_channels(p_s2_df, p_s2, channel_num)
    combine_channels(p_s3_df, p_s3, channel_num)
    
    
    ### % SIGNIFICANCE with all pvalues
    p_o1_12, p_o1_13, p_o1_23, p_o2_12, p_o2_13, p_o2_23, p_o3_12, p_o3_13, p_o3_23 = everyp_each_factor(loaded_model['orderXspeed'])
    p_s1_12, p_s1_13, p_s1_23, p_s2_12, p_s2_13, p_s2_23, p_s3_12, p_s3_13, p_s3_23 = everyp_each_factor(loaded_model['speedXorder'])
    
    # orderXspeed
    # order1
    combine_channels(p_o1_12_df, p_o1_12, channel_num)
    combine_channels(p_o1_13_df, p_o1_13, channel_num)
    combine_channels(p_o1_23_df, p_o1_23, channel_num)
    
    # order2
    combine_channels(p_o2_12_df, p_o2_12, channel_num)
    combine_channels(p_o2_13_df, p_o2_13, channel_num)
    combine_channels(p_o2_23_df, p_o2_23, channel_num)
    
    # order3
    combine_channels(p_o3_12_df, p_o3_12, channel_num)
    combine_channels(p_o3_13_df, p_o3_13, channel_num)
    combine_channels(p_o3_23_df, p_o3_23, channel_num)
    
    
    # speedXorder
    # speed1
    combine_channels(p_s1_12_df, p_s1_12, channel_num)
    combine_channels(p_s1_13_df, p_s1_13, channel_num)
    combine_channels(p_s1_23_df, p_s1_23, channel_num)
    
    # speed2
    combine_channels(p_s2_12_df, p_s2_12, channel_num)
    combine_channels(p_s2_13_df, p_s2_13, channel_num)
    combine_channels(p_s2_23_df, p_s2_23, channel_num)
    
    # speed3
    combine_channels(p_s3_12_df, p_s3_12, channel_num)
    combine_channels(p_s3_13_df, p_s3_13, channel_num)
    combine_channels(p_s3_23_df, p_s3_23, channel_num)
    
    

'''
DIRECTION OF EFFECT NOT CONSIDERED IN THIS CODE BLOCK
'''
# # for significance % calculation
# # orderXspeed
# sigpercent_p_o1_12 = percent_significance(p_o1_12_df)
# sigpercent_p_o1_13 = percent_significance(p_o1_13_df)
# sigpercent_p_o1_23 = percent_significance(p_o1_23_df)

# sigpercent_p_o2_12 = percent_significance(p_o2_12_df)
# sigpercent_p_o2_13 = percent_significance(p_o2_13_df)
# sigpercent_p_o2_23 = percent_significance(p_o2_23_df)

# sigpercent_p_o3_12 = percent_significance(p_o3_12_df)
# sigpercent_p_o3_13 = percent_significance(p_o3_13_df)
# sigpercent_p_o3_23 = percent_significance(p_o3_23_df)


# # speedXorder
# sigpercent_p_s1_12 = percent_significance(p_s1_12_df)
# sigpercent_p_s1_13 = percent_significance(p_s1_13_df)
# sigpercent_p_s1_23 = percent_significance(p_s1_23_df)

# sigpercent_p_s2_12 = percent_significance(p_s2_12_df)
# sigpercent_p_s2_13 = percent_significance(p_s2_13_df)
# sigpercent_p_s2_23 = percent_significance(p_s2_23_df)

# sigpercent_p_s3_12 = percent_significance(p_s3_12_df)
# sigpercent_p_s3_13 = percent_significance(p_s3_13_df)
# sigpercent_p_s3_23 = percent_significance(p_s3_23_df)


'''Data preparation for heatmap'''
# updating the channel order based on electrode depths
p_o1_df = add_depth_info(p_o1_df, probe_locs)
p_o2_df = add_depth_info(p_o2_df, probe_locs)
p_o3_df = add_depth_info(p_o3_df, probe_locs)

p_s1_df = add_depth_info(p_s1_df, probe_locs)
p_s2_df = add_depth_info(p_s2_df, probe_locs)
p_s3_df = add_depth_info(p_s3_df, probe_locs)




# plotting
%matplotlib qt

fig, axs = plt.subplots(nrows=2, ncols=6, sharex=True, figsize=(20, 10))


### HEATMAP 1 : alpha = 0.05/12 = 0.004
sns.heatmap(p_o1_df, cmap="coolwarm", vmax=0.008, vmin=0.0005, ax=axs[0,0], cbar=False)
axs[0, 0].set_title('Original - pairwiseSpeeds')
axs[0, 0].set_ylabel('Depth')

sns.heatmap(p_o2_df, cmap="coolwarm", vmax=0.008, vmin=0.0005, ax=axs[0,1], cbar=False, yticklabels=False)
axs[0, 1].set_title('Global rev - pairwiseSpeeds')

sns.heatmap(p_o3_df, cmap="coolwarm", vmax=0.008, vmin=0.0005, ax=axs[0,2], cbar=False, yticklabels=False)
axs[0, 2].set_title('Local rev - pairwiseSpeeds')

sns.heatmap(p_s1_df, cmap="coolwarm", vmax=0.008, vmin=0.0005, ax=axs[0,3], cbar=False, yticklabels=False)
axs[0, 3].set_title('66 - pairwiseOrders')

sns.heatmap(p_s2_df, cmap="coolwarm", vmax=0.008, vmin=0.0005, ax=axs[0,4], cbar=False, yticklabels=False)
axs[0, 4].set_title('75 - pairwiseOrders')

sns.heatmap(p_s3_df, cmap="coolwarm", vmax=0.008, vmin=0.0005, ax=axs[0,5], yticklabels=False)
axs[0, 5].set_title('85 - pairwiseOrders')



### HEATMAP 2 : alpha = 0.05/36 = 0.001
sns.heatmap(p_o1_df, cmap="coolwarm", vmax=0.002, vmin=0.0005, ax=axs[1,0], cbar=False)
axs[1, 0].set_title('Original - pairwiseSpeeds')
axs[1, 0].set_ylabel('Depth')

sns.heatmap(p_o2_df, cmap="coolwarm", vmax=0.002, vmin=0.0005, ax=axs[1,1], cbar=False, yticklabels=False)
axs[1, 1].set_title('Global rev - pairwiseSpeeds')

sns.heatmap(p_o3_df, cmap="coolwarm", vmax=0.002, vmin=0.0005, ax=axs[1,2], cbar=False, yticklabels=False)
axs[1, 2].set_title('Local rev - pairwiseSpeeds')

sns.heatmap(p_s1_df, cmap="coolwarm", vmax=0.002, vmin=0.0005, ax=axs[1,3], cbar=False, yticklabels=False)
axs[1, 3].set_title('66 - pairwiseOrders')

sns.heatmap(p_s2_df, cmap="coolwarm", vmax=0.002, vmin=0.0005, ax=axs[1,4], cbar=False, yticklabels=False)
axs[1, 4].set_title('75 - pairwiseOrders')

sns.heatmap(p_s3_df, cmap="coolwarm", vmax=0.002, vmin=0.0005, ax=axs[1,5], yticklabels=False)
axs[1, 5].set_title('85 - pairwiseOrders')


for row in range(2):
    for col in range(6):
        axs[row, col].invert_yaxis()
        axs[row, col].axvline(x = 6, color = 'w', linestyle = '--', linewidth = 1)
        #axs[1, axisnum].set_xticks([0,300,600],['-300','0','300'])
        axs[row, col].set_xticks(np.arange(0,15,3), ['-300','-150','0','150','300'], rotation=0)
        axs[row, col].set_xlabel('Time(in ms)')

fig.suptitle('Pairwise Significance', fontsize=16)


### SIGNIFICANCE PERCENTAGE

'''
DIRECTION OF EFFECT NOT CONSIDERED IN THIS CODE BLOCK
'''
# # orderXspeed
# axs[1,0].plot(sigpercent_p_o1_12, 'o-')
# axs[1,0].plot(sigpercent_p_o1_13, 'o-')
# axs[1,0].plot(sigpercent_p_o1_23, 'o-')
# axs[1,0].set_ylabel('% significance in each time window')

# axs[1,1].plot(sigpercent_p_o2_12, 'o-')
# axs[1,1].plot(sigpercent_p_o2_13, 'o-')
# axs[1,1].plot(sigpercent_p_o2_23, 'o-')

# axs[1,2].plot(sigpercent_p_o3_12, 'o-')
# axs[1,2].plot(sigpercent_p_o3_13, 'o-')
# axs[1,2].plot(sigpercent_p_o3_23, 'o-')

# for axisnum in range(3):
#     axs[1, axisnum].legend(['66Hz-75Hz', '66Hz-85Hz', '75Hz-85Hz'])

# # speedXorder
# axs[1,3].plot(sigpercent_p_s1_12, 'o-')
# axs[1,3].plot(sigpercent_p_s1_13, 'o-')
# axs[1,3].plot(sigpercent_p_s1_23, 'o-')

# axs[1,4].plot(sigpercent_p_s2_12, 'o-')
# axs[1,4].plot(sigpercent_p_s2_13, 'o-')
# axs[1,4].plot(sigpercent_p_s2_23, 'o-')

# axs[1,5].plot(sigpercent_p_s3_12, 'o-')
# axs[1,5].plot(sigpercent_p_s3_13, 'o-')
# axs[1,5].plot(sigpercent_p_s3_23, 'o-')

# for axisnum in range(3,6):
#     axs[1, axisnum].legend(['original-globalrev', 'original-localrev', 'localrev-globalrev'])

# for axisnum in range(6):
#     axs[1, axisnum].set_ylim(0, 100)
#     axs[1, axisnum].axvline(x = 6, color = 'r', linestyle = '--', linewidth = 1)
#     axs[1, axisnum].set_xlabel('Time windows')


# fig.suptitle('Pairwise Significance', fontsize=16)

save_loc = str(savehere) + '/interaction_significance_plots'
if not os.path.exists(save_loc):     # if the required folder does not exist, create one
    os.mkdir(save_loc)
    
plt.savefig(str(save_loc) + '/pairwise_significance_hm.png')
plt.close()
