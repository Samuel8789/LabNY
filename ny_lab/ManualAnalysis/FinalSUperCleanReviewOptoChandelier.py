# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:35:41 2024

@author: sp3660
"""
"""
 CLEANES ANALYSIS OF OPTO CHANDELIER DATASETS
 """
import numpy as np
from PIL import Image
import scipy
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr, ttest_ind, zscore, mode
from statsmodels.nonparametric.smoothers_lowess import lowess
import caiman as cm
from matplotlib.patches import Rectangle
import scipy.signal as sg
import scipy.stats as st
import pandas as pd
import shutil
from copy import deepcopy
import os
import seaborn as sns
from math import sqrt
import pickle
import glob
import matplotlib
import matplotlib.gridspec as gridspec
import random
from pathlib import Path
from pprint import pprint
import time


def check_temp_data(temp_data_path,data_analysis_name) :
    temp_data_list=[]
    temp_data_list=glob.glob(str(temp_data_path)+os.sep+f'**{data_analysis_name}**', recursive=False)
    return temp_data_list


def save_temp_data(multiple_analysis,datapath):
    if not os.path.isfile(datapath):
        with open(datapath, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(multiple_analysis, f, pickle.HIGHEST_PROTOCOL)
    return datapath
            


def load_temp_data(temp_data_list,dataindex):
    multiple_analysis={}
    if temp_data_list:
        selected_temp_data_path=temp_data_list[dataindex]
        with open( selected_temp_data_path, 'rb') as file:
            multiple_analysis=  pickle.load(file)
    return multiple_analysis

def get_stimulus_table(analysis,vis_stim_transitions,blank_opt):
    # grating_number=vis_stim_transitions.shape[0]
    grating_repetitions=vis_stim_transitions.shape[1]*vis_stim_transitions.shape[2]
    blnaksweepreps=4*vis_stim_transitions.shape[1]
    angles=np.linspace(0,360,9)[:-1]
    # angle_numbers=len(angles)
    frequencies=np.array([2])
    # frequency_numbers=len(frequencies)
    angles_xv, frequencies_yv = np.meshgrid(angles,frequencies)
    anglevalues = np.reshape(np.arange(1,9), (1, 8))
    
    all_rows=[]
    for ori in range(1,9):
        angled=angles_xv[:,np.where(anglevalues==ori)[1][0]][0]
        freq=float(frequencies[np.where(anglevalues==ori)[0][0]])
        indexes=list(zip(np.reshape(vis_stim_transitions[ori-1,:,:,0],(1,grating_repetitions)),         np.reshape(vis_stim_transitions[ori-1,:,:,1],(1,grating_repetitions))))[0]
        for i in range(grating_repetitions):
            opto=np.float32(0)

            if ori==analysis.acquisition_object.visstimdict['opto']['randomoptograting'] and i % 2 == 0:
                opto=np.float32(1)       
            all_rows.append((np.float32(freq),np.float32(angled), np.float32(0),opto,np.int32(indexes[0][i]), np.int32(indexes[1][i]) ))
             
    blankindexes=list(zip(np.reshape(analysis.signals_object.optodrift_info['Blank']['ArrayFinal_downsampled_LED_clipped'],(1,blnaksweepreps)), np.reshape(analysis.signals_object.optodrift_info['Blank']['ArrayFinalOffset_downsampled_LED_clipped'],(1,blnaksweepreps))))[0]
    for i in range(blnaksweepreps):
        opto=np.float32(0)

        if blank_opt and i in np.array([np.array([4*i, 4*i+1]) for i in range(20)]).ravel():
            opto=np.float32(1)
        
        all_rows.append((np.float32(np.nan), np.float32(np.nan), np.float32(1),opto,np.int32(blankindexes[0][i]), np.int32(blankindexes[1][i]) ))

    df = pd.DataFrame(all_rows, columns =['temporal_frequency','orientation', 'blank_sweep','opto','start', 'end',])
    sorted_df=df.sort_values(by=['start'])
    sorted_df= sorted_df.reset_index(drop=True)
    analysis.full_data['visstim_info']['OptoDrift']={}
    analysis.full_data['visstim_info']['OptoDrift']['stimulus_table']=sorted_df
    
    return sorted_df

#%% GET VIARABLE FROM TO DO SCRIPT  
all_analysis=selected_analysis

#%% ANALYSIS DECISIONS
full_parameter_dictionary={'evoked_period_start':_,# this is to defined which evoked period I'll use for analysis
                           'evoked_period_duration':_,# this is to defined which evoked period I'll use for analysis
                           'trial_pre_stim_time':500,# this is the time pre stimulus nonset to claculate the baseline in ms, keep this as default is for calculating frames mostly
                           'trial_post_stim_tim':2000,# this is the time post stimulus nonset to claculate the baseline in ms,keep this as default is for calculating frames mostly
                           'trial_number':20,
                           'grating_directions':8,
                           'evoked_period':_,
                           'evoked_period':_,
                           'evoked_period':_,
                           'evoked_period':_,
                           'evoked_period':_,
                           'evoked_period':_,
                           'evoked_period':_,
                           'evoked_period':_,
                           'evoked_period':_,
                           'evoked_period':_,
                           'evoked_period':_,}

# DECISION 1 Depedning oin the data i have to use the 0 to 1 second pereiod or the 1 to 2 for data with LED or 0 to 2 for full here I decide this
mean_Stim_decision_idx=1
mean_Stim_decision=['full','half','remove_led']
stim_dec_dict={'full':[0,2000],
               'half':[0,1000],
               'remove_led':[1000,1000]}       
full_parameter_dictionary['evoked_period_start']  =stim_dec_dict[mean_Stim_decision[mean_Stim_decision_idx]]
full_parameter_dictionary['evoked_period_duration']  = stim_dec_dict[mean_Stim_decision[mean_Stim_decision_idx]]

               




#%% folders and paths
timestr=time.strftime("%Y%m%d-%H%M%S")

# OBJECT SAVING PATH FOR RESULTS
tempprocessingpat= Path(os.path.join(os.path.expanduser('~'),r'Desktop/TempPythonObjects'))
mouse_loaded=sorted(list(set([i['analysis'].acquisition_object.mouse_imaging_session_object.mouse_object.mouse_name for i in all_analysis])))
if len(mouse_loaded)>1:
    datapath=os.path.join(tempprocessingpat,'_'.join(mouse_loaded+[timestr]))
else:
    datapath=os.path.join(tempprocessingpat,mouse_loaded[0])

#%% CHECK AND LOAD AVAILABLE ANALYSIS
# Here first checks if tehre ia analysis with the same mouse datasets. It wil list all dataset with the combination of mice and the with dataindex you select wich one wnat to load
dataindex=0
temp_data_list=check_temp_data(tempprocessingpat,'_'.join(mouse_loaded))
multiple_analysis=load_temp_data(temp_data_list,dataindex)
 

#%% PROCESS THE DATA
"""
THE OBJECTIVE OF THIS PROCESSING IS
    FIRST GET THE STIMULUS TABLE WITH THE APPROPIATE TRANSITIONS
        REVIEW CALCULATED TRANBSTION IN VoltageProcessing objects
        Rewvie that analysis object has the proper voltaheg transitions also
"""
full_review=True
plt.close('all')

if not multiple_analysis:
    # FIRST STEP IS PROCESING DATSET INDIVIDUAL AND EXTRACTING RELEVANT INFORMATION
    for single_dataset in all_analysis:   
        # Loading the two main objects
        analysis=single_dataset['analysis']
        full_data=single_dataset['full_data']

        # REVIEW TRANSITIONS
        if full_review:
            analysis.signals_object.plot_all_basics()
            # analysis.signals_object
            # Go to voltsignalsextractio object and review the plotting functions
            # FOr the moment trust that yhey are properly done. These were saved anyway somewhere
         
        # LOADING GRATING TRANSITIONS FROM VOLTAGE PROCESSING THIS VARIABLE IS USED WITH GET_STIMULUS_TABLE ONLY
        # this shape depend on the experiment parameter    directions, trial_number, onset-offset frames, adjusted
        # not include here are the blank trials fromsome reason or the separation betwen opto and control
        vis_stim_transitions=np.zeros((8,20,2,2)).astype(np.uint16)
        for i in range(1,9):
            vis_stim_transitions[i-1,:,:,0]=analysis.signals_object.optodrift_info[f'Grat_{i}']['ArrayFinal_downsampled_LED_clipped']
            vis_stim_transitions[i-1,:,:,1]=analysis.signals_object.optodrift_info[f'Grat_{i}']['ArrayFinalOffset_downsampled_LED_clipped']


        # IN THE EXPERIMENT I HAVE SELECTED A GRATING TO DO OPTO AND TEHRE COULS ALSO BE BLANKLS TO DO OPTO BUT THIS IS VARIABLE
        # HERE I CEHCK WHICH IS THE GRATIUNG WITH IOPTO THIS IS DIONE USING THE VARIABLE IN ACQUISITION GOTTEN FORM THE VIS STIM FILE PRODUCED BY MATLAB WHEN RUNNING BEHAVIOR
        # HERE STE SOM REVIEW FUNCTION TO CHECK VISSTIM INFO
        # THEN SELECT THE ONSET FOR THE OPTOGRATING AND ALSO GETS THE BLANK
        # I THINK THEY ARE HERE IN THE TRY BEAUSE SOKMME DATASET DONT HAVE BLANK WITH OPTO
        gratingcontrol=np.empty(0)
        try:
            optograting=analysis.acquisition_object.visstimdict['opto']['randomoptograting']
            optogratinfo=analysis.signals_object.optodrift_info[f'Grat_{optograting}']
            optoblankinfo=analysis.signals_object.optodrift_info['Blank']
        
            gratingcontrol=optogratinfo['ArrayFinal_downsampled_LED_clipped'][:,1]
            # opto_gratdifference=analysis.full_data['voltage_traces']['Full_signals_transitions']['PhotoTrig']['aligned_downsampled_LEDshifted']['Prairie']['up']-optogratinfo['ArrayFinal_downsampled_LED_clipped'][:,0]
            
        except:
            print('Non optograting')





    
    all_info={'activity_dict':activity_dict,
              'activity_dict_peak':activity_dict_peak,
              'scaled_activity_dict':scaled_activity_dict,
              'scaled_activity_dict_peak':scaled_activity_dict_peak,
              'peaks':peaks,
              'optocellindex_dict':optocellindex_dict,
              'traces_dict':traces_dict,
              'transition_array':transition_array,
              'chand_indexes':chand_indexes,
              'stimulated_cells_number':stimulated_cells_number,
              'nTrials':nTrials,
              'opto_repetitions':opto_repetitions,
              'led_opt':led_opt,
              'pretime':pretime,
              'posttime':posttime,
              'fr':fr,
              'gratingcontrol':gratingcontrol,
              'speed':speed,
              'speedtimestamps':speedtimestamps,
              'accepted_all_plane_distances':accepted_all_plane_distances,
              'vis_stim_transitions':vis_stim_transitions,
              'sweep_response':sweep_response,
              'mean_sweep_response':mean_sweep_response,
              'pval':pval,
              'response':response,
              'peak':peak,               
              'sweepscale':sweepscale,
              'sweepsmoothed':sweepsmoothed ,
              'dff':sweepdff,
              'blank_opt':blank_opt,
              'mean_start':mean_start,
              'meanstimtime':meanstimtime,
              'mean_Stim_decision':mean_Stim_decision[mean_Stim_decision_idx],
              'stim_table':sorted_df,
              'pre_frames_df':pre_frames,
              'post_frames_df':post_frames,
              'pre_time_df':pre_time,
              'post_time_df':post_time
              }
    
    multiple_analysis[analysis.acquisition_object.aquisition_name]=all_info

