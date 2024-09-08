# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 19:44:15 2024

@author: sp3660
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.signal import correlate
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm
from scipy.signal import gaussian, convolve,find_peaks
import time
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import scipy
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr, ttest_ind, zscore, mode
import scipy
from scipy.signal import gaussian, convolve,find_peaks
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
import scipy.stats as st
from pylab import *
import matplotlib
import matplotlib.patches as patches
from pathlib import Path

#%% REORGANIZING DATA
#pv depth 175 IDS
plt.close('all')

pv_175_ids=[{667772494:[667772496,668528146,671164733]},{667692761:[670395999,668526823,667692764]}]
chand_ids=[688083454, 688084148, 688084135, 688084552]

boc.get_cell_specimens(ids=[688083454,688084148,688084135])

for animal in pv_175_ids:
    if list(animal.keys())[0]==667772494:
        chand_ids=[688083454,688084148,688084135]
    elif list(animal.keys())[0]==667692761:
        chand_ids=[693254414, 693254426, 693254397]
        
        
    for i in animal.values():
        for exp in i:
            data_set = boc.get_ophys_experiment_data(exp)
            pprint(data_set.get_metadata())


            # % EXPLORE DATASET
            
            #number of cells recorded
            cell_n=data_set.get_cell_specimen_ids().shape[0]
            # plot max projection and amsks
            
            # all_roi_masks = data_set.get_roi_mask_array()
            all_roi_masks = data_set.get_roi_mask_array(chand_ids)

            combined_mask = all_roi_masks.max(axis=0)
            
            contours = measure.find_contours(combined_mask, 0.5)  # Threshold of 0.5 to get the contours
            
            f,ax=plt.subplots(1)
            ax.imshow(data_set.get_max_projection())
            # Overlay the contours
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='red')  # Contours are plotted in (row, col) format
            
            plt.axis('off')  # Hide axis
            plt.show()
            
            stim_epoch = data_set.get_stimulus_epoch_table()

            time, raw_traces = data_set.get_corrected_fluorescence_traces(cell_specimen_ids=chand_ids)
            # time, raw_traces = data_set.get_corrected_fluorescence_traces()

            dxcm, tsd = data_set.get_running_speed()
            fig,ax = plt.subplots(1,figsize=(14,8))
            for i in range(raw_traces.shape[0]):
                ax.plot(raw_traces[i,:]+(i*raw_traces.max()/2), color='gray')
                
            #for each stimulus, shade the plot when the stimulus is presented
            colors = ['blue','orange','green','red','yellow']
            for c,stim_name in enumerate(stim_epoch.stimulus.unique()):
                stim = stim_epoch[stim_epoch.stimulus==stim_name]
                for j in range(len(stim)):
                    ax.axvspan(xmin=stim.start.iloc[j], xmax=stim.end.iloc[j], color=colors[c], alpha=0.1)


#%% PLOT ONE CELL SLICED ACTIVTY AND TRIAL AVERAGED
plt.close('all')

all_datasets_tr_av_mean=[]
all_datasets_tr_av_sem=[]
all_res=[]
mean_similarity_across_trials=np.zeros([3,2,7])
mean_ta_similarity_across_cells=np.zeros([3,2])
mean_con_similarity_across_cells=np.zeros([3,2])
select_similarity=1
similarity=['cosine','pearson']


for dataset,datset_name in dtset_list:
    

    analysis=all_analysis[dataset]['analysis']
    full_data=all_analysis[dataset]['full_data']
    
    _, processed_speed, _,processed_speed_timestamps=compute_locomotion_bouts(analysis,pretime=1,posttime=1)


    
    res=analyzed_movie1_vs_spont(dataset,all_analysis)
    all_res.append(res)
    sliced_movie1=res[1]
    sliced_movie1_timestamps=res[5]
    sliced_spont=res[3]
    sliced_spont_timestamps=res[7]
    volt_spont_slices=res[8]
    no_speed_spont_trials=[i for i,trial in enumerate(volt_spont_slices) if not processed_speed[trial].any()]
    print(no_speed_spont_trials)


    

    
    all_para_tr_av_mean=[]
    all_para_tr_av_sem=[]
    for para,para_name in [(0,'Natural Movie'),(1,'Spontaneous')]: 
            
        if 'Spont' in para_name:
            if dataset==0:
                name='spont1_'
            else:    
                name='spont_'
        else:
                name='natural_movie_one_set_'
                voltage_slice=slice(analysis.volt_object.extraction_object.transitions_dictionary[f'{name}first'],analysis.volt_object.extraction_object.transitions_dictionary[f'{name}last'])
                movie1_volt_starts=analysis.volt_object.extraction_object.movie_one_trial_full_recording-voltage_slice.start
                speed_timestamps=processed_speed_timestamps[voltage_slice]/1000-processed_speed_timestamps[voltage_slice][0]/1000
                speed=processed_speed[voltage_slice]
                trials_without_locomotion=[trial  for trial,i in enumerate(movie1_volt_starts) if not speed[ i:i+min(np.diff(movie1_volt_starts))].any()]
                print(trials_without_locomotion)
   
        if 'Spont' not in para_name:
            activity=sliced_movie1
            activity_without_loc=sliced_movie1[:,trials_without_locomotion,:]
            corrected_timestamps=  sliced_movie1_timestamps - sliced_movie1_timestamps[:, 0][:, np.newaxis]

        else:
            activity=sliced_spont
            activity_without_loc=sliced_spont[:,no_speed_spont_trials,:]
            corrected_timestamps=  sliced_spont_timestamps - sliced_spont_timestamps[:, 0][:, np.newaxis]



        all_tr_av_mean=[]
        all_tr_av_sem=[]
        
        #remove locomotion trtails and create concatenated array
        
        
        trial_averaged= activity_without_loc.mean(axis=1)
        concatenated_trials = np.concatenate(np.split(activity_without_loc, activity_without_loc.shape[1], axis=1),axis=2).squeeze()
        
        # calculate sijmilarity measure , plot heatmaps and stroa averaged similarity across cells per recording
        if similarity[select_similarity]=='cosine':
            norms = np.linalg.norm(trial_averaged, axis=1, keepdims=True)
            normalized_X = trial_averaged / norms
            trial_averaged_similarity_across_cells= np.dot(normalized_X, normalized_X.T)
            norms = np.linalg.norm(concatenated_trials, axis=1, keepdims=True)
            normalized_X = concatenated_trials / norms
            vmin=0
            trial_concatenated_similarity_across_cells= np.dot(normalized_X, normalized_X.T) 
        elif similarity[select_similarity]=='pearson':
            trial_averaged_similarity_across_cells = np.corrcoef(trial_averaged)
            trial_concatenated_similarity_across_cells = np.corrcoef(concatenated_trials)
            vmin=-1

            
        mask = np.zeros_like(trial_averaged_similarity_across_cells, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mean_ta_similarity_across_cells[dataset,para]=np.mean(trial_averaged_similarity_across_cells[mask])  
        mean_con_similarity_across_cells[dataset,para]=np.mean(trial_concatenated_similarity_across_cells[mask])  
        
        
        #plot similarity across trial averaged cell responses
        for dtst,typ in zip([trial_averaged_similarity_across_cells,trial_concatenated_similarity_across_cells],['Trial Averaged','Trial Concatenated']):
            timestr = time.strftime("%Y%m%d-%H%M%S")
            ffuptit=f'{typ} for {datset_name}, {para_name}, {similarity[select_similarity]} Similarity Across Cells'
            plt.figure(figsize=(20, 20))
            sns.heatmap(dtst,vmin=vmin,vmax=1,cmap='viridis', annot=True, fmt='.2f', cbar=True)
            plt.title(ffuptit)
            plt.savefig(temppath /  Path(ffuptit+f'_{timestr}.png'), dpi=300, bbox_inches='tight')
            
            
        for cell in range(data['CellNumber']): 
            
            df = pd.DataFrame(activity_without_loc[cell,:,:].T, columns=[f'Trial_{i}' for i in range(activity_without_loc[cell,:,:].shape[0])])
            df = df.reset_index().melt(id_vars='index', var_name='Trial', value_name='Value')
            df.columns = ['Time (s)', 'Trial', 'Value']
            mean = df.groupby('Time (s)')['Value'].mean()
            sem = df.groupby('Time (s)')['Value'].sem()
            repeated_time_array=np.tile(corrected_timestamps[0,:], activity_without_loc.shape[1])
            df['Time (s)'] = repeated_time_array
            
            #PLot trial averaged activity and trial in same subplot
            timestr = time.strftime("%Y%m%d-%H%M%S")
            fuptit=f'Chandelier Cell {cell+1} for {datset_name}, {para_name}, Trial Averaged'
            fig,axs=plt.subplots(2,sharex=True,figsize=(20,20), constrained_layout=True)    
            fig.suptitle(fuptit)
            all_tr_av_mean.append(mean.values)
            all_tr_av_sem.append(sem.values)
            axs[0].plot(corrected_timestamps[0,:], mean.values, color='blue')
            axs[0].fill_between(corrected_timestamps[0,:], mean.values - sem.values, mean.values + sem.values, color='blue', alpha=0.3)
            sns.lineplot(data=df, x='Time (s)', y='Value', hue='Trial', palette='Greys', alpha=0.5,linewidth=1, ax=axs[1])
            for ax in axs:
                for spine in ax.spines.values():
                    if spine != ax.spines['bottom']:
                        spine.set_visible(False)
            plt.savefig(temppath /  Path(fuptit+f'_{timestr}.png'), dpi=300, bbox_inches='tight')

        
            #PLot trial averaged activity and trials in separeted subplots

            timestr = time.strftime("%Y%m%d-%H%M%S")
            fffuptit=f'Chandelier Cell {cell+1} for {datset_name}, {para_name}, Trial Responses'
            fig,axs=plt.subplots(11,sharex=True,figsize=(20,20), constrained_layout=True)
            fig.suptitle(fffuptit)
            for i in range(10):
                sns.lineplot(x=corrected_timestamps[0, :], y=activity[cell, i, :], ax=axs[i], color='gray', alpha=0.7,linewidth=1)
                for spine in axs[i].spines.values():
                    if spine != axs[i].spines['bottom']:
                        spine.set_visible(False)
            axs[10].plot(corrected_timestamps[0, :], mean.values, color='blue', label='Average Trial',linewidth=2)
            axs[10].fill_between(corrected_timestamps[0, :], mean.values - sem.values, mean.values + sem.values, color='blue', alpha=0.3)
            for spine in axs[10].spines.values():
                if spine != axs[i].spines['bottom']:
                    spine.set_visible(False)                   
            axs[-1].set_xlabel('Time (s)')
            plt.savefig(temppath /  Path(fffuptit+f'_{timestr}.png'), dpi=300, bbox_inches='tight')
            
                 
            # calculate similarity measure , plot heatmaps and stroa averaged similarity across trials per recoded cell
            trial_activity_no_loco=activity_without_loc[cell,:,:]
            if similarity[select_similarity]=='cosine':
                norms = np.linalg.norm(trial_activity_no_loco, axis=1, keepdims=True)
                normalized_X = trial_activity_no_loco / norms
                cell_similarity_across_trials= np.dot(normalized_X, normalized_X.T)

            elif similarity[select_similarity]=='pearson':
                cell_similarity_across_trials = np.corrcoef(trial_activity_no_loco)
                
            mask = np.zeros_like(cell_similarity_across_trials, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            mean_similarity_across_trials[dataset,para,cell]=np.mean(cell_similarity_across_trials[mask])
            
            timestr = time.strftime("%Y%m%d-%H%M%S")
            ffuptit=f'Chandelier Cell {cell+1} for {datset_name}, {para_name}, {similarity[select_similarity]} Similarity Across Trials'
            plt.figure(figsize=(20, 20))
            sns.heatmap(cell_similarity_across_trials,vmin=vmin,vmax=1,cmap='viridis', annot=True, fmt='.2f', cbar=True)
            plt.title(ffuptit)
            plt.savefig(temppath /  Path(ffuptit+f'_{timestr}.png'), dpi=300, bbox_inches='tight')


            
        all_para_tr_av_mean.append(all_tr_av_mean)
        all_para_tr_av_sem.append(all_tr_av_sem)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        ffuptit=f'All Chandelier Cells for {datset_name}, {para_name}, Trial Averaged'
        f,axes=plt.subplots(7,sharex=True,figsize=(20,20), constrained_layout=True)
        f.suptitle(ffuptit)
        for cell in range(data['CellNumber'])  :         
            axes[cell].plot(corrected_timestamps[0,:], all_tr_av_mean[cell], color='blue')
            axes[cell].fill_between(corrected_timestamps[0,:],all_tr_av_mean[cell] - all_tr_av_sem[cell], all_tr_av_mean[cell] + all_tr_av_sem[cell], color='blue', alpha=0.3)
            for spine in ax.spines.values():
                if spine != ax.spines['bottom']:
                    spine.set_visible(False)
        plt.savefig(temppath /  Path(ffuptit+f'_{timestr}.png'), dpi=300, bbox_inches='tight')
    all_datasets_tr_av_mean.append(all_para_tr_av_mean)
    all_datasets_tr_av_sem.append(all_para_tr_av_sem)
 
plt.close('all')
  
#%% COMPARE CELSS BETWEEN SESSIONS
plt.close('all')
cell_across=np.zeros([3,2,7,847])
cell_across=np.zeros([3,2,7,847])

mean_cell_similarity_across_sessions=np.zeros([2,7])

for dataset,datset_name in dtset_list:
    analysis=all_analysis[dataset]['analysis']
    _, processed_speed, _,processed_speed_timestamps=compute_locomotion_bouts(analysis,pretime=1,posttime=1)


    
    for j,(para,para_name) in enumerate( [(1,'Natural Movie'),(3,'Spontaneous')]): 
            
        if 'Spont' in para_name:
            if dataset==0:
                name='spont1_'
            else:    
                name='spont_'
                
            volt_spont_slices=all_res[dataset][8]
            no_speed_spont_trials=[i for i,trial in enumerate(volt_spont_slices) if not processed_speed[trial].any()]
            print(no_speed_spont_trials)
            trials_without_locomotion=no_speed_spont_trials

        else:
                name='natural_movie_one_set_'                 
                voltage_slice=slice(analysis.volt_object.extraction_object.transitions_dictionary[f'{name}first'],analysis.volt_object.extraction_object.transitions_dictionary[f'{name}last'])
                speed_timestamps=processed_speed_timestamps[voltage_slice]/1000-processed_speed_timestamps[voltage_slice][0]/1000
                speed=processed_speed[voltage_slice]
                movie1_volt_starts=analysis.volt_object.extraction_object.movie_one_trial_full_recording-voltage_slice.start
                trials_without_locomotion=[trial  for trial,i in enumerate(movie1_volt_starts) if not speed[ i:i+min(np.diff(movie1_volt_starts))].any()]
                print(trials_without_locomotion)


        trial_activity=all_res[dataset][para][:,trials_without_locomotion,:847]
        trial_averaged=trial_activity.mean(axis=1)
        trial_concatenated = np.concatenate(np.split(trial_activity, trial_activity.shape[1], axis=1),axis=2).squeeze()
        cell_across[dataset,para-j-1,:,:]=trial_averaged

      
for para ,para_name in [(0,'Natural Movie'),(1,'Spontaneous')]: 
    for cell in range(7):
        # calculate similarity measure , plot heatmaps and stroa averaged similarity across trials per recoded cell
        session_activity_no_loco=cell_across[:,para,cell,:]

        if similarity[select_similarity]=='cosine':
            norms = np.linalg.norm(session_activity_no_loco, axis=1, keepdims=True)
            normalized_X = session_activity_no_loco / norms
            trial_averaged_cell_similarity_across_sessions= np.dot(normalized_X, normalized_X.T)
            vmin=0


        elif similarity[select_similarity]=='pearson':
            trial_averaged_cell_similarity_across_sessions = np.corrcoef(session_activity_no_loco)
            vmin=-1

            
        mask = np.zeros_like(trial_averaged_cell_similarity_across_sessions, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mean_cell_similarity_across_sessions[para,cell]=np.mean(trial_averaged_cell_similarity_across_sessions[mask])

        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        ffuptit=f'Chandelier Cell {cell+1} for {para_name}, {similarity[select_similarity]} Similarity Across Sessions'
        plt.figure(figsize=(20, 20))
        sns.heatmap(trial_averaged_cell_similarity_across_sessions,vmin=0,vmax=1,cmap='viridis', annot=True, fmt='.2f', cbar=True,annot_kws={"fontsize":16})
        plt.title(ffuptit)
        plt.savefig(temppath /  Path(ffuptit+f'_{timestr}.png'), dpi=300, bbox_inches='tight')
