# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:23:30 2025

@author: ssenapat

plotting script for significance percentage taking into account whether the pairwise effect is a positive or negative effect
"""
def get_alpha():
    
    alpha = 0.05
    # alpha = 0.00138     # this is corrected alpha value for 3 comparisons each in 12 time windows, i.e. 12*3=36, therefore alpha=0.05/36
    # alpha =  0.00416     # this is corrected alpha value for 3 comparisons in 12 time windows, therefore alpha=0.05/12
    return alpha


def percent_significance(df_p, df_amp, alpha):
    
    pos_percent_sig = []
    neg_percent_sig = []
    
    for tw in df_p.columns:
        subset_p = df_p[tw]
        subset_amp = df_amp[tw]
        subset = pd.concat([subset_p, subset_amp], axis=1)
        subset.columns = ['pval', 'relAmps']
        
        # check the amplitudes now for positive or negative effect
        significant_rows = subset[subset['pval'] < alpha]
        poseff_significant_rows_num = Counter(significant_rows['relAmps'] > 0)[True] 
        negeff_significant_rows_num = Counter(significant_rows['relAmps'] < 0)[True]
        
        total_rows_num = len(subset)
        
        pos_percent_sig_for_tw = (poseff_significant_rows_num/total_rows_num)*100
        neg_percent_sig_for_tw = (negeff_significant_rows_num/total_rows_num)*100
        
        #print(percent_sig_for_tw)
        
        # combine % significance of all time windows in a list
        pos_percent_sig = np.append(pos_percent_sig, pos_percent_sig_for_tw)
        neg_percent_sig = np.append(neg_percent_sig, neg_percent_sig_for_tw)
    
    return pos_percent_sig, neg_percent_sig



def combine_channels(df, val_list, index):
    
    df.loc[index,:] = val_list
    return df


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



'''
### FOR %SIGNIFICANCE plots : obtain a df for each pairwise comparison absolute amplitude (similar to the one for pvalue)   
'''

def everyamp_each_factor(model):
    # NOTE here that the indexing syntax is different compared to the pvalue dictionary
    # this is due to slight difference in the stored dictionary formats
    
    
    a_factor1_12 = []       # o1 = original; pair s1-s2
    a_factor1_13 = []       # pair s1-s3
    a_factor1_23 = []       # pair s2-s3
    
    a_factor2_12 = []       # o2 = global reversed
    a_factor2_13 = []       
    a_factor2_23 = []       
    
    a_factor3_12 = []       # o3 = local reversed
    a_factor3_13 = []       
    a_factor3_23 = []       
    
    for tw in model.keys():
        #print(tw)
        window = model[tw]
        
        # factor1
        a_factor1_12 =  np.append(a_factor1_12, window[list(window)[0]][0])
        a_factor1_13 =  np.append(a_factor1_13, window[list(window)[0]][0])
        a_factor1_23 =  np.append(a_factor1_23, window[list(window)[0]][0])
        
        # factor 2
        a_factor2_12 =  np.append(a_factor2_12, window[list(window)[1]][0])
        a_factor2_13 =  np.append(a_factor2_13, window[list(window)[1]][0])
        a_factor2_23 =  np.append(a_factor2_23, window[list(window)[1]][0])
        
        # factor 3
        a_factor3_12 =  np.append(a_factor3_12, window[list(window)[2]][0])
        a_factor3_13 =  np.append(a_factor3_13, window[list(window)[2]][0])
        a_factor3_23 =  np.append(a_factor3_23, window[list(window)[2]][0])
        
    # return the 6 lists (each list of length 12)
    return a_factor1_12, a_factor1_13, a_factor1_23, a_factor2_12, a_factor2_13, a_factor2_23, a_factor3_12, a_factor3_13, a_factor3_23




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
from pathlib import Path
import os



## main body
alpha = get_alpha()

directory =  Path('G:/Final_exps_spikes/LFP/Elfie/p1/p1_15/significance_tests/')
directory_ttest = Path(str(directory) + '/ttest/')
directory_directionalamps = Path(str(directory) + '/pairwise_direction_eff/')
savehere = Path('G:/Final_exps_spikes/LFP/Elfie/p1/p1_15/plots/')

timewindows = ['tw1', 'tw2', 'tw3', 'tw4','tw5','tw6','tw7','tw8','tw9','tw10','tw11','tw12']

'''p-VALUE dataframes'''
# orderXspeed
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


'''Relative amplitude dataframes'''
# orderXspeed
# order1
a_o1_12_df = pd.DataFrame(columns=timewindows)
a_o1_13_df = pd.DataFrame(columns=timewindows)
a_o1_23_df = pd.DataFrame(columns=timewindows)
# order2
a_o2_12_df = pd.DataFrame(columns=timewindows)
a_o2_13_df = pd.DataFrame(columns=timewindows)
a_o2_23_df = pd.DataFrame(columns=timewindows)
# order3
a_o3_12_df = pd.DataFrame(columns=timewindows)
a_o3_13_df = pd.DataFrame(columns=timewindows)
a_o3_23_df = pd.DataFrame(columns=timewindows)

# speedXorder
# speed1
a_s1_12_df = pd.DataFrame(columns=timewindows)
a_s1_13_df = pd.DataFrame(columns=timewindows)
a_s1_23_df = pd.DataFrame(columns=timewindows)
# speed2
a_s2_12_df = pd.DataFrame(columns=timewindows)
a_s2_13_df = pd.DataFrame(columns=timewindows)
a_s2_23_df = pd.DataFrame(columns=timewindows)
# speed3
a_s3_12_df = pd.DataFrame(columns=timewindows)
a_s3_13_df = pd.DataFrame(columns=timewindows)
a_s3_23_df = pd.DataFrame(columns=timewindows)



for chan in range(1,385):
    
    channel_num = chan

    # checking one channel to get associated pvalues (output from lfp_ttest.py)
    with open(str(directory_ttest) + '/channel' + str(chan) + '_interaction.pkl', 'rb') as file:
        loaded_model_pval = pickle.load(file)
        file.close()
        
    # checking one channel to get associated pvalues (output from lfp_ttest.py)
    with open(str(directory_directionalamps) + '/channel' + str(chan) + '_directionalInteraction_Amps.pkl', 'rb') as file:
        loaded_model_amps = pickle.load(file)
        file.close()
        
    ### % SIGNIFICANCE with all pvalues
    p_o1_12, p_o1_13, p_o1_23, p_o2_12, p_o2_13, p_o2_23, p_o3_12, p_o3_13, p_o3_23 = everyp_each_factor(loaded_model_pval['orderXspeed'])
    p_s1_12, p_s1_13, p_s1_23, p_s2_12, p_s2_13, p_s2_23, p_s3_12, p_s3_13, p_s3_23 = everyp_each_factor(loaded_model_pval['speedXorder'])
    
    ### % SIGNIFICANCE with all relative amplitudes
    a_o1_12, a_o1_13, a_o1_23, a_o2_12, a_o2_13, a_o2_23, a_o3_12, a_o3_13, a_o3_23 = everyamp_each_factor(loaded_model_amps['orderXspeed'])
    a_s1_12, a_s1_13, a_s1_23, a_s2_12, a_s2_13, a_s2_23, a_s3_12, a_s3_13, a_s3_23 = everyamp_each_factor(loaded_model_amps['speedXorder'])
    
    
    # orderXspeed
    '''pvalues'''
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
    
    '''relative amplitudes'''
    # order1
    combine_channels(a_o1_12_df, a_o1_12, channel_num)
    combine_channels(a_o1_13_df, a_o1_13, channel_num)
    combine_channels(a_o1_23_df, a_o1_23, channel_num)
    # order2
    combine_channels(a_o2_12_df, a_o2_12, channel_num)
    combine_channels(a_o2_13_df, a_o2_13, channel_num)
    combine_channels(a_o2_23_df, a_o2_23, channel_num)
    # order3
    combine_channels(a_o3_12_df, a_o3_12, channel_num)
    combine_channels(a_o3_13_df, a_o3_13, channel_num)
    combine_channels(a_o3_23_df, a_o3_23, channel_num)
    
    
    # speedXorder
    '''pvalues'''
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
    
    '''relative amplitudes'''
    # speed1
    combine_channels(a_s1_12_df, a_s1_12, channel_num)
    combine_channels(a_s1_13_df, a_s1_13, channel_num)
    combine_channels(a_s1_23_df, a_s1_23, channel_num)
    # speed2
    combine_channels(a_s2_12_df, a_s2_12, channel_num)
    combine_channels(a_s2_13_df, a_s2_13, channel_num)
    combine_channels(a_s2_23_df, a_s2_23, channel_num)
    # speed3
    combine_channels(a_s3_12_df, a_s3_12, channel_num)
    combine_channels(a_s3_13_df, a_s3_13, channel_num)
    combine_channels(a_s3_23_df, a_s3_23, channel_num)



'''
DIRECTION OF EFFECT CONSIDERED
'''
# for significance % calculation
# orderXspeed
pos_sigpercent_p_o1_12, neg_sigpercent_p_o1_12 = percent_significance(p_o1_12_df, a_o1_12_df, alpha)
pos_sigpercent_p_o1_13, neg_sigpercent_p_o1_13 = percent_significance(p_o1_13_df, a_o1_13_df, alpha)
pos_sigpercent_p_o1_23, neg_sigpercent_p_o1_23 = percent_significance(p_o1_23_df, a_o1_23_df, alpha)

pos_sigpercent_p_o2_12, neg_sigpercent_p_o2_12 = percent_significance(p_o2_12_df, a_o2_12_df, alpha)
pos_sigpercent_p_o2_13, neg_sigpercent_p_o2_13 = percent_significance(p_o2_13_df, a_o2_13_df, alpha)
pos_sigpercent_p_o2_23, neg_sigpercent_p_o2_23 = percent_significance(p_o2_23_df, a_o2_23_df, alpha)

pos_sigpercent_p_o3_12, neg_sigpercent_p_o3_12 = percent_significance(p_o3_12_df, a_o3_12_df, alpha)
pos_sigpercent_p_o3_13, neg_sigpercent_p_o3_13 = percent_significance(p_o3_13_df, a_o3_13_df, alpha)
pos_sigpercent_p_o3_23, neg_sigpercent_p_o3_23 = percent_significance(p_o3_23_df, a_o3_23_df, alpha)


# speedXorder
pos_sigpercent_p_s1_12, neg_sigpercent_p_s1_12 = percent_significance(p_s1_12_df, a_s1_12_df, alpha)
pos_sigpercent_p_s1_13, neg_sigpercent_p_s1_13 = percent_significance(p_s1_13_df, a_s1_13_df, alpha)
pos_sigpercent_p_s1_23, neg_sigpercent_p_s1_23 = percent_significance(p_s1_23_df, a_s1_23_df, alpha)

pos_sigpercent_p_s2_12, neg_sigpercent_p_s2_12 = percent_significance(p_s2_12_df, a_s2_12_df, alpha)
pos_sigpercent_p_s2_13, neg_sigpercent_p_s2_13 = percent_significance(p_s2_13_df, a_s2_13_df, alpha)
pos_sigpercent_p_s2_23, neg_sigpercent_p_s2_23 = percent_significance(p_s2_23_df, a_s2_23_df, alpha)

pos_sigpercent_p_s3_12, neg_sigpercent_p_s3_12 = percent_significance(p_s3_12_df, a_s3_12_df, alpha)
pos_sigpercent_p_s3_13, neg_sigpercent_p_s3_13 = percent_significance(p_s3_13_df, a_s3_13_df, alpha)
pos_sigpercent_p_s3_23, neg_sigpercent_p_s3_23 = percent_significance(p_s3_23_df, a_s3_23_df, alpha)



# plotting
%matplotlib qt

fig, axs = plt.subplots(nrows=3, ncols=6, sharex=True, figsize=(20, 10))

# orderXspeed

# Original - all speeds
axs[0,0].plot(pos_sigpercent_p_o1_12, 'o-')
axs[0,0].plot(-neg_sigpercent_p_o1_12, 'o-')
axs[0,0].set_title('Original : 66-75Hz')
axs[0,0].set_ylabel('% significance in each time window')

axs[1,0].plot(pos_sigpercent_p_o1_13, 'o-')
axs[1,0].plot(-neg_sigpercent_p_o1_13, 'o-')
axs[1,0].set_title('Original : 66-85Hz')
axs[1,0].set_ylabel('% significance in each time window')

axs[2,0].plot(pos_sigpercent_p_o1_23, 'o-')
axs[2,0].plot(-neg_sigpercent_p_o1_23, 'o-')
axs[2,0].set_title('Original : 75-85Hz')
axs[2,0].set_ylabel('% significance in each time window')


# Global reversal - all speeds
axs[0,1].plot(pos_sigpercent_p_o2_12, 'o-')
axs[0,1].plot(-neg_sigpercent_p_o2_12, 'o-')
axs[0, 1].set_title('Global rev : 66-75Hz')

axs[1,1].plot(pos_sigpercent_p_o2_13, 'o-')
axs[1,1].plot(-neg_sigpercent_p_o2_13, 'o-')
axs[1,1].set_title('Global rev : 66-85Hz')

axs[2,1].plot(pos_sigpercent_p_o2_23, 'o-')
axs[2,1].plot(-neg_sigpercent_p_o2_23, 'o-')
axs[2,1].set_title('Global rev : 75-85Hz')

# Local reversal - all speeds
axs[0,2].plot(pos_sigpercent_p_o3_12, 'o-')
axs[0,2].plot(-neg_sigpercent_p_o3_12, 'o-')
axs[0,2].set_title('Local rev : 66-75Hz')

axs[1,2].plot(pos_sigpercent_p_o3_13, 'o-')
axs[1,2].plot(-neg_sigpercent_p_o3_13, 'o-')
axs[1,2].set_title('Local rev : 66-85Hz')

axs[2,2].plot(pos_sigpercent_p_o3_23, 'o-')
axs[2,2].plot(-neg_sigpercent_p_o3_23, 'o-')
axs[2,2].set_title('Local rev : 75-85Hz')



# speedXorder

# 66Hz - all orders
axs[0,3].plot(pos_sigpercent_p_s1_12, 'o-')
axs[0,3].plot(-neg_sigpercent_p_s1_12, 'o-')
axs[0,3].set_title('66Hz : Original-Globalrev')

axs[1,3].plot(pos_sigpercent_p_s1_13, 'o-')
axs[1,3].plot(-neg_sigpercent_p_s1_13, 'o-')
axs[1,3].set_title('66Hz : Original-Localrev')

axs[2,3].plot(pos_sigpercent_p_s1_23, 'o-')
axs[2,3].plot(-neg_sigpercent_p_s1_23, 'o-')
axs[2,3].set_title('66Hz : Localrev-Globalrev')


# 75Hz - all orders
axs[0,4].plot(pos_sigpercent_p_s2_12, 'o-')
axs[0,4].plot(-neg_sigpercent_p_s2_12, 'o-')
axs[0,4].set_title('75Hz : Original-Globalrev')

axs[1,4].plot(pos_sigpercent_p_s2_13, 'o-')
axs[1,4].plot(-neg_sigpercent_p_s2_13, 'o-')
axs[1,4].set_title('75Hz : Original-Localrev')

axs[2,4].plot(pos_sigpercent_p_o2_23, 'o-')
axs[2,4].plot(-neg_sigpercent_p_o2_23, 'o-')
axs[2,4].set_title('75Hz : Localrev-Globalrev')


# 85Hz - all orders
axs[0,5].plot(pos_sigpercent_p_s3_12, 'o-')
axs[0,5].plot(-neg_sigpercent_p_s3_12, 'o-')
axs[0,5].set_title('85Hz : Original-Globalrev')

axs[1,5].plot(pos_sigpercent_p_s3_13, 'o-')
axs[1,5].plot(-neg_sigpercent_p_s3_13, 'o-')
axs[1,5].set_title('85Hz : Original-Localrev')

axs[2,5].plot(pos_sigpercent_p_s3_23, 'o-')
axs[2,5].plot(-neg_sigpercent_p_s3_23, 'o-')
axs[2,5].set_title('85Hz : Localrev-Globalrev')



    
for row in range(3):
    for col in range(6):
        axs[row, col].axvline(x = 6, color = 'r', linestyle = '--', linewidth = 1)
        axs[row, col].axhline(y = 0, color = 'k', linewidth = 1)
        axs[row, col].set_ylim(-15, 15)
        axs[row, col].grid(axis='both', color='0.95')
        
for col in range(6):
    axs[2, col].set_xticks(np.arange(0,15,3), ['-300','-150','0','150','300'], rotation=0)
    axs[2, col].set_xlabel('Time(in ms)')
    
fig.suptitle('All Pairwise Significance with Directional Effect', fontsize=16)


save_loc = str(savehere) + '/interaction_significance_plots'
if not os.path.exists(save_loc):     # if the required folder does not exist, create one
    os.mkdir(save_loc)
    
plt.savefig(str(save_loc) + '/directional_significance_alpha' + str(alpha) + '.png')
plt.close()
