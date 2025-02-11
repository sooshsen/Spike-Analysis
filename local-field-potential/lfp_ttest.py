# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:21:18 2025

@author: ssenapat

Pairwise t-test to check the significantly observed effects for pairwise difference
"""

def load_onset_response(file):

    # read all saved onset response per channel info - generated by lfp_onset_response.py
    onsets = pd.read_csv(file, header=None)
    
    return onsets


def load_pvals(file):

    # read all saved pvals per channel info - generated by lfp_ANOVA.py
    pvals = pd.read_csv(file)
    
    return pvals


def identify_channel(filepath):
    
    folder = str(filepath)     

    folder_name_loc = re.search(r"channel[0-9]+", folder)
    ind = folder_name_loc.span()
    channelID = folder[ind[0]+7:ind[1]]        # get rid of the 'LFP' at the start of the channel ID name
    
    return int(channelID)



def downsample_channel(chan_matrix, channel_num, savehere):
    
    ### 
    save_loc = str(savehere) + '/data_for_statistics'
    if not os.path.exists(save_loc):     # if the required folder does not exist, create one
        os.mkdir(save_loc)
     
    chan_matrix_selected = chan_matrix.iloc[:,500:2000]
    chan_matrix_selected.columns = range(chan_matrix_selected.columns.size)     # don't forget to handle the column names of this subset
    
    
    # downsampling to get one datapoint for every 50 ms, i.e. 1 datapoint for every 125 datapoints
    downsampled_chan_matrix = np.zeros(12) 
    
    for condition in range(len(chan_matrix_selected)):
        # for each row at a time
        rowval = chan_matrix_selected.iloc[condition,:]
        mean_values_for_condition = []
        w = 0
        while w < chan_matrix_selected.columns.size:
            mean_per_window = np.mean(rowval[w:w+125])
            mean_values_for_condition = np.append(mean_values_for_condition, mean_per_window)
            w = w + 125
            
        downsampled_chan_matrix = np.vstack([downsampled_chan_matrix, mean_values_for_condition])
        
    downsampled_chan_matrix = np.delete(downsampled_chan_matrix, [0], axis=0) # remove 1st row, which is not crucial
        
    # downsampled_chan_matrix_df = pd.DataFrame(downsampled_chan_matrix)
    # downsampled_chan_matrix_df.to_csv(str(save_loc) + '/channel' + str(channel_num) + '_downsampled.csv', header=False, index=False)
    
    return downsampled_chan_matrix


'''
def interaction_effect(pvals):
    # for a given channel check if interaction effect is significant
    
    p_interaction = pvals['p_orderXspeed']
    
    return p_interaction[p_interaction < 0.05].index.values
'''    

def ttest_by_order(data):
    
    orders = data['order'].unique()
    pval_order_pairwisespeed = {}
    
    for o in orders:
        subset_d = data[data['order'] == o]
        subset_d_s1 = subset_d[subset_d['speed'] == 66]['evoked_response'].reset_index(drop=True)
        subset_d_s2 = subset_d[subset_d['speed'] == 75]['evoked_response'].reset_index(drop=True)
        subset_d_s3 = subset_d[subset_d['speed'] == 85]['evoked_response'].reset_index(drop=True)
        
        # perform ttest
        pval_12 = stats.ttest_rel(subset_d_s1, subset_d_s2).pvalue
        pval_13 = stats.ttest_rel(subset_d_s1, subset_d_s3).pvalue
        pval_23 = stats.ttest_rel(subset_d_s2, subset_d_s3).pvalue
        
        pval_order_pairwisespeed[o] = []
        pval_order_pairwisespeed[o].append([pval_12,pval_13,pval_23])
        
    
    return pval_order_pairwisespeed
        
        

def ttest_by_speed(data):
    
    speeds = data['speed'].unique()
    pval_speed_pairwiseorder = {}
    
    for s in speeds:
        subset_d = data[data['speed'] == s]
        subset_d_o1 = subset_d[subset_d['order'] == 'original']['evoked_response'].reset_index(drop=True)
        subset_d_o2 = subset_d[subset_d['order'] == 'local_reversed']['evoked_response'].reset_index(drop=True)
        subset_d_o3 = subset_d[subset_d['order'] == 'global_reversed']['evoked_response'].reset_index(drop=True)
        
        # perform ttest
        pval_12 = stats.ttest_rel(subset_d_o1, subset_d_o2).pvalue
        pval_13 = stats.ttest_rel(subset_d_o1, subset_d_o3).pvalue
        pval_23 = stats.ttest_rel(subset_d_o2, subset_d_o3).pvalue
        
        pval_speed_pairwiseorder[s] = []
        pval_speed_pairwiseorder[s].append([pval_12,pval_13,pval_23])
        
    return pval_speed_pairwiseorder
    
    

def sample_prep(window_matrix):
    
    # based on behavioral experiement paradigm
    condition_order = ['original', 'global_reversed','original','original','global_reversed','local_reversed','local_reversed','global_reversed','local_reversed']
    condition_speed = [66,75,85,75,85,75,66,66,85]
    
    # combine all information in a dataframe
    data_df = pd.DataFrame(window_matrix)
    data_df.rename(columns={0 : 'evoked_response'}, inplace=True)
    
    # add two factors
    data_df['order'] = np.repeat(condition_order, 10)
    data_df['speed'] = np.repeat(condition_speed, 10)
    
    # based on order
    p_by_order = ttest_by_order(data_df)
    
    # based on speed
    p_by_speed = ttest_by_speed(data_df)
    
    return p_by_order, p_by_speed


def with_interaction_effect(data, channel_num, savehere):
    
    save_loc = str(savehere) + '/significance_tests'
    if not os.path.exists(save_loc):     # if the required folder does not exist, create one
        os.mkdir(save_loc)
    
    significance_by_order = {}
    significance_by_speed = {}
    
    for window in range(len(data.T)):      # this should be 12
        # print('Window '+ str(window+1))
        windowname = 'timewindow_' + str(window+1)
        pvals_from_order, pvals_from_speed = sample_prep(data[:,window])
        
        significance_by_order[windowname] = []
        significance_by_order[windowname].append(pvals_from_order)
        
        significance_by_speed[windowname] = []
        significance_by_speed[windowname].append(pvals_from_speed)
    
    
    # save to pickle
    destination_file_1 = str(save_loc) + '/channel' + str(channel_num) + '_order_interaction.pkl'
    with open(destination_file_1, 'wb') as fp:
        pickle.dump(significance_by_order, fp)
        print('dictionary saved successfully to file')
        fp.close()
      
        
    destination_file_2 = str(save_loc) + '/channel' + str(channel_num) + '_speed_interaction.pkl'
    with open(destination_file_2, 'wb') as fp:
        pickle.dump(significance_by_speed, fp)
        print('dictionary saved successfully to file')
        fp.close()
    
    
   

import numpy as np
import pandas as pd
import os
from pathlib import Path
import re

import scipy.stats as stats
import pickle
    

   
directory = Path('G:/Final_exps_spikes/LFP/Elfie/p1/p1_15/')

channel_order = []

for chan in os.listdir(directory):
    filepath = str(directory) + '/onset_responses/' + chan
    onset_mat = load_onset_response(filepath)
    channel_num = identify_channel(filepath)       # identify current channel based on filepath
    print('Channel '+ str(channel_num))
    
    channel_order.append(channel_num)
    
    # every tone per channel
    modified_matrix = downsample_channel(onset_mat, channel_num, directory)
    
    '''
    # from the ANOVA pvals, identify the cases with significant interaction effects
    pval_filepath = str(directory) + '/significance_tests/channel' + str(channel_num) + '_2way_ANOVA.csv'
    
    pvals = load_pvals(pval_filepath)
    windows_of_interest = interaction_effect(pvals)     # these are our columns of interest for ttest from the 90x12 modified matrix
    '''
    
    # ttest considering interaction effect
    with_interaction_effect(modified_matrix, channel_num, directory)
