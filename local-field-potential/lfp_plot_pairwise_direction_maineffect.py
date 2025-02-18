# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:42:40 2025

@author: ssenapat
"""

def get_alpha():
    
    # alpha = 0.05
    # alpha = 0.00138     # this is corrected alpha value for 3 comparisons each in 12 time windows, i.e. 12*3=36, therefore alpha=0.05/36
    alpha =  0.00416     # this is corrected alpha value for 3 comparisons in 12 time windows, therefore alpha=0.05/12
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
def p_each_factor(model):
    
    p_12 = []       # original-globalrev;   66Hz-75Hz
    p_13 = []       # original-localrev;    66Hz-85Hz
    p_23 = []       # localrev-globalrev;   75Hz-85Hz

    for tw in model.keys():
        # print(key)
        window = model[tw][0]
        
        p_12 = np.append(p_12, window[0])        
        p_13 = np.append(p_13, window[1])       
        p_23 = np.append(p_23, window[2])        
        
    return p_12, p_13, p_23


'''
### FOR %SIGNIFICANCE plots : obtain a df for each pairwise comparison absolute amplitude (similar to the one for pvalue)   
'''

def relamp_each_factor(model):
    # NOTE here that the indexing syntax is different compared to the pvalue dictionary
    # this is due to slight difference in the stored dictionary formats
    
    a_factor12 = []       # original-globalrev;   66Hz-75Hz
    a_factor13 = []       # original-localrev;    66Hz-85Hz
    a_factor23 = []       # localrev-globalrev;   75Hz-85Hz
         
    
    for tw in model.keys():
        #print(tw)
        window = model[tw]
        
        # factor combinations
        a_factor12 =  np.append(a_factor12, window[0])
        a_factor13 =  np.append(a_factor13, window[1])
        a_factor23 =  np.append(a_factor23, window[2])
        
    # return the 6 lists (each list of length 12)
    return a_factor12, a_factor13, a_factor23



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
directory_directionalamps = Path(str(directory) + '/pairwise_direction_eff_without_interaction/')
savehere = Path('G:/Final_exps_spikes/LFP/Elfie/p1/p1_15/plots/')

timewindows = ['tw1', 'tw2', 'tw3', 'tw4','tw5','tw6','tw7','tw8','tw9','tw10','tw11','tw12']



'''p-VALUE dataframes'''
# order effect
p_o12_df = pd.DataFrame(columns=timewindows)
p_o13_df = pd.DataFrame(columns=timewindows)
p_o23_df = pd.DataFrame(columns=timewindows)

# speed effect
p_s12_df = pd.DataFrame(columns=timewindows)
p_s13_df = pd.DataFrame(columns=timewindows)
p_s23_df = pd.DataFrame(columns=timewindows)



'''relative amplitude dataframes'''
# order effect
a_o12_df = pd.DataFrame(columns=timewindows)
a_o13_df = pd.DataFrame(columns=timewindows)
a_o23_df = pd.DataFrame(columns=timewindows)

# speed effect
a_s12_df = pd.DataFrame(columns=timewindows)
a_s13_df = pd.DataFrame(columns=timewindows)
a_s23_df = pd.DataFrame(columns=timewindows)


for chan in range(1,385):
    
    channel_num = chan

    # checking one channel to get associated pvalues (output from lfp_ttest.py)
    with open(str(directory_ttest) + '/channel' + str(chan) + '_mainEffect.pkl', 'rb') as file:
        loaded_model_pval = pickle.load(file)
        file.close()
        
    # checking one channel to get associated pvalues (output from lfp_ttest.py)
    with open(str(directory_directionalamps) + '/channel' + str(chan) + '_directionalInteraction_Amps_maineff.pkl', 'rb') as file:
        loaded_model_amps = pickle.load(file)
        file.close()
        
    ### % SIGNIFICANCE with all pvalues
    # obtain the p-value for each factor comparison for this channel
    p_o12, p_o13, p_o23 = p_each_factor(loaded_model_pval['main_order'])
    p_s12, p_s13, p_s23 = p_each_factor(loaded_model_pval['main_speed'])
    
    
    ### % SIGNIFICANCE with all relative amplitudes
    a_o12, a_o13, a_o23 = relamp_each_factor(loaded_model_amps['main_order'])
    a_s12, a_s13, a_s23 = relamp_each_factor(loaded_model_amps['main_speed'])
    
    # order
    '''pvalues'''
    combine_channels(p_o12_df, p_o12, channel_num)
    combine_channels(p_o13_df, p_o13, channel_num)
    combine_channels(p_o23_df, p_o23, channel_num)
    
    '''relative amplitudes'''
    combine_channels(a_o12_df, a_o12, channel_num)
    combine_channels(a_o13_df, a_o13, channel_num)
    combine_channels(a_o23_df, a_o23, channel_num)
    
    
    # speed
    '''pvalues'''
    combine_channels(p_s12_df, p_s12, channel_num)
    combine_channels(p_s13_df, p_s13, channel_num)
    combine_channels(p_s23_df, p_s23, channel_num)
    
    '''relative amplitudes'''
    combine_channels(a_s12_df, a_s12, channel_num)
    combine_channels(a_s13_df, a_s13, channel_num)
    combine_channels(a_s23_df, a_s23, channel_num)
    


'''
DIRECTION OF EFFECT CONSIDERED
'''
# for significance % calculation
# orderXspeed
pos_sigpercent_p_o12, neg_sigpercent_p_o12 = percent_significance(p_o12_df, a_o12_df, alpha)
pos_sigpercent_p_o13, neg_sigpercent_p_o13 = percent_significance(p_o13_df, a_o13_df, alpha)
pos_sigpercent_p_o23, neg_sigpercent_p_o23 = percent_significance(p_o23_df, a_o23_df, alpha)

# speedXorder
pos_sigpercent_p_s12, neg_sigpercent_p_s12 = percent_significance(p_s12_df, a_s12_df, alpha)
pos_sigpercent_p_s13, neg_sigpercent_p_s13 = percent_significance(p_s13_df, a_s13_df, alpha)
pos_sigpercent_p_s23, neg_sigpercent_p_s23 = percent_significance(p_s23_df, a_s23_df, alpha)



# plotting
%matplotlib qt

fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(20, 10))



# Order comparisons
axs[0,0].plot(pos_sigpercent_p_o12, 'o-')
axs[0,0].plot(-neg_sigpercent_p_o12, 'o-')
axs[0,0].set_title('Original - Global rev')
axs[0,0].set_ylabel('% significance in each time window')

axs[0,1].plot(pos_sigpercent_p_o13, 'o-')
axs[0,1].plot(-neg_sigpercent_p_o13, 'o-')
axs[0,1].set_title('Original - Local rev')

axs[0,2].plot(pos_sigpercent_p_o23, 'o-')
axs[0,2].plot(-neg_sigpercent_p_o23, 'o-')
axs[0,2].set_title('Local rev - Global rev')



# Speed comparisons
axs[1,0].plot(pos_sigpercent_p_s12, 'o-')
axs[1,0].plot(-neg_sigpercent_p_s12, 'o-')
axs[1,0].set_title('66-75Hz')
axs[1,0].set_ylabel('% significance in each time window')

axs[1,1].plot(pos_sigpercent_p_s13, 'o-')
axs[1,1].plot(-neg_sigpercent_p_s13, 'o-')
axs[1,1].set_title('66-85Hz')

axs[1,2].plot(pos_sigpercent_p_s23, 'o-')
axs[1,2].plot(-neg_sigpercent_p_s23, 'o-')
axs[1,2].set_title('75-85Hz')

    
for row in range(2):
    for col in range(3):
        axs[row, col].axvline(x = 6, color = 'r', linestyle = '--', linewidth = 1)
        axs[row, col].axhline(y = 0, color = 'k', linewidth = 1)
        axs[row, col].set_ylim(-20, 20)
        axs[row, col].grid(axis='both', color='0.95')
        
for col in range(3):
    axs[1, col].set_xticks(np.arange(0,15,3), ['-300','-150','0','150','300'], rotation=0)
    axs[1, col].set_xlabel('Time(in ms)')
    
fig.suptitle('Pairwise Significance (only main factor) with Directional Effect', fontsize=16)


save_loc = str(savehere) + '/maineffect_significance_plots'
if not os.path.exists(save_loc):     # if the required folder does not exist, create one
    os.mkdir(save_loc)
    
plt.savefig(str(save_loc) + '/directional_significance_maineff_alpha' + str(alpha) + '.png')
plt.close()
