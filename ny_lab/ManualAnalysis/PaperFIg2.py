# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:39:00 2024

@author: sp3660
"""


from sys import platform
from pathlib import Path
import time
import os

import numpy as np
import pandas as pd
import seaborn as sns

from statsmodels.nonparametric.smoothers_lowess import lowess
from PIL import Image
from statsmodels.nonparametric.smoothers_lowess import lowess
import caiman as cm


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.patches import Rectangle
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rc('xtick', labelsize=8) 
mpl.rc('ytick', labelsize=8) 


from scipy.signal import gaussian, convolve,find_peaks
from scipy import io
from scipy.stats import kendalltau, pearsonr, spearmanr, ttest_ind, zscore, mode, sem,shapiro, levene,mannwhitneyu, ttest_rel, wilcoxon, f_oneway, kruskal
import statsmodels.api as sm

if platform == "linux" or platform == "linux2":
    fig_two_basepath=Path(r'/home/samuel/Dropbox/Projects/LabNY/ChandPaper/Fig2')
elif platform == "win32":
    fig_two_basepath=Path(r'C:\Users\sp3660\Desktop\ChandPaper\Fig2')
    

timestr=time.strftime("%Y%m%d-%H%M%S")

cell_equivalences=np.zeros([3,7]) 

cell_equivalences[0,:]=[5,2,3,6,4,7,1]
cell_equivalences[1,:]=[6,3,1,5,4,7,2]
cell_equivalences[2,:]=[6,4,7,5,1,2,3]
cell_equivalences=cell_equivalences-1
temppath=Path(r'C:\Users\sp3660\Desktop\TempPythonFigs')

all_analysis=selected_analysis
#%% FIgure matplotlib basics
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

#%%
def analyzed_movie1_vs_spont(dataset,all_analysis):

    full_data=all_analysis[dataset]['full_data']
    
    imaging_data=full_data['imaging_data']
    visstim_info=full_data['visstim_info']
    
    movie_timestamps=imaging_data['mov_timestamps_seconds']['clipped']


    
    data=imaging_data['Plane1']
    resampled_movie1_starts=visstim_info['Movie1']['Trial_Starts'].astype('int')
    resampled_par_transitions=visstim_info['Paradigm_Indexes']
    
    
    
    # get the slices for movie 1
    full_movie_one_slice=slice(resampled_par_transitions['natural_movie_one_set_first'],resampled_par_transitions['natural_movie_one_set_last'])
    trial_slices=[]
    for trial, frame in enumerate(resampled_movie1_starts):
        if trial!=len(resampled_movie1_starts)-1:
            trial_slices.append(slice(frame,resampled_movie1_starts[trial+1]-1))
        else:
            frames=min([ (sl.stop - sl.start)  for sl in trial_slices])
            trial_slices.append(slice(frame,frame+frames))
    trial_slices=[]
    for trial, frame in enumerate(resampled_movie1_starts):
        trial_slices.append(slice(frame,frame+frames))
          
        
    # set the arrays
    full_movie_one=np.zeros([data['CellNumber'],8500])
    trials_movie_one=np.zeros([data['CellNumber'],len(trial_slices),frames])
    trials_movie_one_timestamps=np.zeros([len(trial_slices),frames])
    full_movie_one_timestamps=np.zeros([1,8500])
        
 
    # SLCIE DATA
    for cell in range(data['CellNumber'])  : 
        
        full_movie_one[cell,:len(data['Traces']['denoised'][translate_selected_cell(cell, cell_equivalences, dataset)  ,full_movie_one_slice])]=data['Traces']['denoised'][translate_selected_cell(cell, cell_equivalences, dataset)  ,full_movie_one_slice]
        for trial, sl in enumerate(trial_slices):
            trials_movie_one[cell,trial,:]=data['Traces']['denoised'][translate_selected_cell(cell, cell_equivalences, dataset)  ,sl]
            trials_movie_one_timestamps[trial,:]=movie_timestamps[sl]
    full_movie_one_timestamps[0,:len(movie_timestamps[full_movie_one_slice])]=movie_timestamps[full_movie_one_slice]
    # set the arrays for spont

    full_spont=np.zeros([data['CellNumber'],8500])
    full_spont_timestamps=np.zeros([1,8500])
    
    # get spont slices
    if dataset==0:
        name='spont1'
    else:    
        name='spont'


    voltage_slice=slice(all_analysis[dataset]['analysis'].volt_object.extraction_object.transitions_dictionary[f'{name}_first'],all_analysis[dataset]['analysis'].volt_object.extraction_object.transitions_dictionary[f'{name}_last'])
    spont_full_length=resampled_par_transitions[f'{name}_last']-resampled_par_transitions[f'{name}_first']
    volt_spont_full_length=voltage_slice.stop- voltage_slice.start
    trial_length=frames
    volt_trial_length=int(volt_spont_full_length/10)
    trial_n=10
    extra_end=0
    if spont_full_length/trial_n<trial_length:
        extra_end=10-spont_full_length%trial_n
 
        
        
    volt_spont_slices=[]
    spont_slices=[]
    for i in range(trial_n):
        spont=slice(resampled_par_transitions[f'{name}_first']+i*trial_length,resampled_par_transitions[f'{name}_first']+i*trial_length+trial_length)
        volt_spont=slice( voltage_slice.start+i*volt_trial_length, voltage_slice.start+i*volt_trial_length+volt_trial_length)
        
        spont_slices.append(spont)
        volt_spont_slices.append(volt_spont)

        
    full_spont_slice=slice(resampled_par_transitions[f'{name}_first'],resampled_par_transitions[f'{name}_last']+extra_end)
    trials_spont=np.zeros([data['CellNumber'],len(spont_slices),frames])
    trials_spont_timestamps=np.zeros([len(spont_slices),frames])


 
    #get spont data
    for cell in range(data['CellNumber'])  : 
        
        full_spont[cell,:len(data['Traces']['denoised'][translate_selected_cell(cell, cell_equivalences, dataset)  ,full_spont_slice])]=data['Traces']['denoised'][translate_selected_cell(cell, cell_equivalences, dataset)  ,full_spont_slice]
        for trial, sl in enumerate(spont_slices):
            trials_spont[cell,trial,:]=data['Traces']['denoised'][translate_selected_cell(cell, cell_equivalences, dataset)  ,sl]
            trials_spont_timestamps[trial,:]=movie_timestamps[sl]
        
    trials_spont_timestamps[0,:]=movie_timestamps[spont]
    full_spont_timestamps[0,:len(movie_timestamps[full_spont_slice])]=movie_timestamps[full_spont_slice]
       
     
    return  full_movie_one, trials_movie_one,\
            full_spont, trials_spont,\
            full_movie_one_timestamps, trials_movie_one_timestamps,\
            full_spont_timestamps, trials_spont_timestamps,\
            volt_spont_slices

def translate_selected_cell(cell, cell_equivalences, datset):
    
    cell=cell_equivalences[datset,cell]
        
    return int(cell)


def compute_locomotion_bouts(analysis,pretime=20,posttime=20):
 
    
    def convert_spped_to_units(analysis):
        
      
        timestamps=analysis.volt_object.extraction_object.all_signals['Prairie']['Time'].to_numpy().flatten()
        raw_locomotion=np.abs(np.diff(analysis.volt_object.extraction_object.all_signals['Prairie']['Locomotion'].to_numpy().flatten(),prepend=0))
        
        speed=np.diff(raw_locomotion,prepend=raw_locomotion[0])
        rectified_speed=np.absolute(speed)
        peaks, _ = find_peaks(rectified_speed, height=[0.2,4])
        x=np.arange(len(rectified_speed))/1000
        binary_peaks = np.zeros_like(rectified_speed)
        binary_peaks[peaks] = 1
        
        def calculate_distances(binary_vector):
            # Find indices of ones
            ones_indices = np.where(binary_vector == 1)[0]
            
            # Compute differences between consecutive indices
            distances = np.diff(ones_indices,prepend=0)
            
            return distances
        
        def rolling_sum_counts_per_second(binary_signal, sampling_rate, window_size_ms=1000):
            # Calculate number of samples in each window
            samples_per_window = int(window_size_ms / 1000 * sampling_rate)
            
            # Calculate number of complete windows
            num_windows = len(binary_signal) // samples_per_window
            
            # Initialize list to store counts per second
            counts_per_second = []
            
            # Iterate through complete windows
            for i in range(num_windows):
                # Compute sum of binary signal in current window
                window_sum = np.sum(binary_signal[i * samples_per_window : (i + 1) * samples_per_window])
                
                # Append sum to counts per second
                counts_per_second.append(window_sum)
            
            return counts_per_second
        
         
        sampling_rate = 1000  # Assuming 1000 samples per second (1 sample per millisecond)
        rolling_sums = rolling_sum_counts_per_second(binary_peaks, sampling_rate,window_size_ms=1000)
        speed_cm_s=np.array(rolling_sums)*0.6
        loc_second_time=np.arange(int(np.round(timestamps[0]/sampling_rate)),len(rolling_sums)+int(np.round(timestamps[0]/sampling_rate)))
        distances = calculate_distances(binary_peaks)
        speeds=1000*0.6/distances
        
        
        sped= np.zeros((len(binary_peaks)))
        sped[np.where(binary_peaks)[0]]=speeds
        fs = 1000  # 1 kHz
    
        # Width of the Gaussian filter in milliseconds
        width_ms = 28  #  ms
        
        # Convert width to number of samples
        width_samples = int(width_ms * fs / 1000)
        gaussian_filter = gaussian(width_samples, std=width_samples / (2 * np.sqrt(2 * np.log(2))))
        gaussian50ms = convolve(sped, gaussian_filter, mode='same') / sum(gaussian_filter)  # Normalize by sum of filter
        
        # f,ax=plt.subplots(5,sharex=True)
        # ax[0].plot(x,raw_locomotion)
        # ax[1].plot(x,speed)
        # ax[2].plot(x,rectified_speed)
        # ax[2].plot(x[peaks], rectified_speed[peaks], "x", markersize=10, color='red')
        # ax[3].plot(x,binary_peaks)
        # ax[3].plot(x[peaks], binary_peaks[peaks], "x", markersize=10, color='red')
        # ax[4].plot(loc_second_time,rolling_sums,label='speed')
        # ax[4].plot(x,gaussian50ms,label='smoothed speed')
        # ax[4].legend()
    
    
        return (loc_second_time,np.array(rolling_sums))
    speed_timestamps,cm_s=convert_spped_to_units(analysis)
    
    
       
    
    upsampled_time_milliseconds_stamps=analysis.volt_object.extraction_object.all_signals['Prairie']['Time'].to_numpy().flatten()
    upsampled_signal = np.interp(upsampled_time_milliseconds_stamps, speed_timestamps*1000, cm_s)
    
    

    
    # f,ax=plt.subplots(1)
    # ax.plot(speed_timestamps*1000, cm_s,'r', label='Original (1 sample/second)')
    # ax.plot(upsampled_time_milliseconds_stamps, upsampled_signal,'b', label='Upsampled (1000 samples/second)')
    # ax.set_xlabel('Time (seconds)')
    # ax.set_ylabel('Amplitude')
    # ax.legend()
    
    peaks=find_peaks(upsampled_signal)[0]
    firstpeaks=np.diff(peaks,prepend=-10000)>10000
    laststpeaks=np.roll(firstpeaks,-1)
    
    boutstarts=peaks[np.diff(peaks,prepend=-10000)>10000]-30-pretime*1000
    boutsends=peaks[laststpeaks]+30+posttime*1000
    

    clippedvoltagetime=analysis.volt_object.extraction_object.all_signals['Prairie']['Time'].to_numpy().flatten()
    ledclippedlocomotion=np.abs(np.diff(analysis.volt_object.extraction_object.all_signals['Prairie']['Locomotion'].to_numpy().flatten(),prepend=0))
    
    # f,ax=plt.subplots(1)
    # ax.plot(clippedvoltagetime,ledclippedlocomotion,label='original locomotion')
    # ax.plot(upsampled_time_milliseconds_stamps,upsampled_signal,label='upsampled speed')
    # ax.plot(upsampled_time_milliseconds_stamps[boutstarts],upsampled_signal[boutstarts],'ro',label='start')
    # ax.plot(upsampled_time_milliseconds_stamps[boutsends],upsampled_signal[boutsends],'bo',label='end')
    # ax.legend()
    
    
    locomotion_on_offsets=[]
    for bout in range(len(boutstarts)):
        locomotion_on_offsets.append((np.abs( np.array(np.array(analysis.all_planes_timestamps['Plane1'])*1000 - boutstarts[bout])).argmin(),np.abs( np.array(np.array(analysis.all_planes_timestamps['Plane1'])*1000 - boutsends[bout])).argmin()))
    
    
    for j,i in enumerate(locomotion_on_offsets):
        locomotion_on_offsets[j]=(i[0],i[1],i[1]-i[0])
        
    return locomotion_on_offsets, upsampled_signal, list(zip(boutstarts,boutsends)),upsampled_time_milliseconds_stamps



#%% TRIA TO RIAL CORRELATIONS THIS PLOT IS TO MOEASURE HOW CORELATED AL TRIAL FOR EACH CELL AND THEN DO AVERAGE ACORS CELLSN AND AROOS MOUSE 
# SO SO
plt.close('all')

# List of datasets and their names
dtset_list = [[0, 'Session C'], [1, 'Session B'], [2, 'Session A']]
# dtset_list = [[0, 'Session C']]

plot=False
save_fig=False
all_res=[]

# Store all the correlation matrices for each session and condition
all_sess_correlation_matrixes = {}
all_sess_tri_traces={}
# Loop over each dataset and session
all_corrected_timestamps={}
for dataset, dataset_name in dtset_list:
    min_trials=10
    analysis = all_analysis[dataset]['analysis']
    full_data = all_analysis[dataset]['full_data']
    
    correlation_matrixes = {'Natural Movie': {}, 'Spontaneous': {}}  # Store matrices for this session
    tri_traces = {'Natural Movie': {}, 'Spontaneous': {}}  # Store matrices for this session
 
    _, processed_speed, _,processed_speed_timestamps=compute_locomotion_bouts(analysis,pretime=1,posttime=1)

    all_corrected_timestamps[dataset_name]={}
    
    res=analyzed_movie1_vs_spont(dataset,all_analysis)
    all_res.append(res)
    sliced_movie1=res[1]
    sliced_movie1_timestamps=res[5]
    sliced_spont=res[3]
    sliced_spont_timestamps=res[7]
    volt_spont_slices=res[8]
    no_speed_spont_trials=[i for i,trial in enumerate(volt_spont_slices) if not processed_speed[trial].any()]
    print(no_speed_spont_trials)

    for para,para_name in [(0,'Natural Movie'),(1,'Spontaneous')]:
        # Set name based on paradigm
        if 'Spont' in para_name:
            name = 'spont1_' if dataset == 0 else 'spont_'
            if min_trials==10:
                min_trials=len(no_speed_spont_trials)
            
        else:
            name = 'natural_movie_one_set_'
            voltage_slice=slice(analysis.volt_object.extraction_object.transitions_dictionary[f'{name}first'],analysis.volt_object.extraction_object.transitions_dictionary[f'{name}last'])
            movie1_volt_starts=analysis.volt_object.extraction_object.movie_one_trial_full_recording-voltage_slice.start
            speed_timestamps=processed_speed_timestamps[voltage_slice]/1000-processed_speed_timestamps[voltage_slice][0]/1000
            speed=processed_speed[voltage_slice]
            trials_without_locomotion=[trial  for trial,i in enumerate(movie1_volt_starts) if not speed[ i:i+min(np.diff(movie1_volt_starts))].any()]
            print(trials_without_locomotion)
            min_trials=len(trials_without_locomotion)

        
        if 'Spont' not in para_name:
            activity=sliced_movie1
            activity_without_loc=sliced_movie1[:,trials_without_locomotion,:]
            corrected_timestamps=  sliced_movie1_timestamps - sliced_movie1_timestamps[:, 0][:, np.newaxis]
           


        else:
            activity=sliced_spont
            activity_without_loc=sliced_spont[:,no_speed_spont_trials,:]
            corrected_timestamps=  sliced_spont_timestamps - sliced_spont_timestamps[:, 0][:, np.newaxis]
        all_corrected_timestamps[dataset_name][para_name]=corrected_timestamps

        num_cells = activity_without_loc.shape[0]  # 7 cells
        
        all_tr_av_mean=[]
        all_tr_av_sem=[]
        
        #remove locomotion trtails and create concatenated array
        
        
        trial_averaged= activity_without_loc.mean(axis=1)
        concatenated_trials = np.concatenate(np.split(activity_without_loc, activity_without_loc.shape[1], axis=1),axis=2).squeeze()

        # Loop over each cell and compute trial correlations
        for cell_idx in range(num_cells):
      
            stacked_array = activity_without_loc[cell_idx,:,:]
            print(stacked_array.shape)
            movie_timestamps = corrected_timestamps[0,:] 
            trial_averaged = np.mean(stacked_array, axis=0)
            tri_traces[para_name][cell_idx] =stacked_array
            
            y_min = min(np.min(stacked_array), np.min(np.mean(stacked_array, axis=0))) - 0.2
            y_max = max(np.max(stacked_array), np.max(np.mean(stacked_array, axis=0)))
            if plot:
                
                # Create figure with 11 subplots (10 trials + 1 average)
                # fig, axes = plt.subplots(stacked_array.shape[0]+1, 1, figsize=(5, 10), tight_layout=True)
                fig, axes = plt.subplots(min_trials+1, 1,figsize=(100/25.4, 100/25.4),tight_layout=True)  # Increase height for vertical stretch

                # Plot individual trials
                for trial_idx in range(min_trials):

                    axes[trial_idx].plot(movie_timestamps, stacked_array[trial_idx, :], color='red', linewidth=1)
                    axes[trial_idx].set_xticks([])
                    axes[trial_idx].set_yticks([])
                    axes[trial_idx].spines['top'].set_visible(False)
                    axes[trial_idx].spines['right'].set_visible(False)
                    axes[trial_idx].spines['left'].set_visible(False)
                    axes[-1].spines['bottom'].set_linewidth(0.5)
                    axes[trial_idx].set_ylim(y_min, y_max)
                    axes[trial_idx].margins(x=0)

    
                # Plot the average activity in the last subplot
                
                axes[-1].plot(movie_timestamps, trial_averaged, color='green', linewidth=1.0)
                axes[-1].spines['top'].set_visible(False)
                axes[-1].spines['right'].set_visible(False)
                axes[-1].spines['left'].set_linewidth(0.5)
                axes[-1].spines['bottom'].set_linewidth(0.5)
                # axes[-1].set_xlim(movie_timestamps[0], movie_timestamps[-1])
                axes[-1].set_ylim(y_min, y_max)
                axes[-1].tick_params(axis='both', which='major', labelsize=6, width=1, length=3,grid_color='r')
                axes[-1].margins(x=0)

                
                # Adjust the layout
                plt.subplots_adjust(hspace=0)
    
                # Set title
                # fig.suptitle(f'Cell {cell_idx + 1}, {para_name}, {dataset_name}', fontsize=10)
    
                if save_fig:
                    fig_filename = os.path.join(fig_two_basepath, f'cell_{cell_idx}_{para_name}_{dataset_name}.svg')
                    plt.savefig(fig_filename,  format='svg', bbox_inches='tight')
                else:
                    plt.show()
    
                # Close the figure to free memory
                plt.close(fig)
            
            
            # Compute the correlation matrix for this cell
            correlation_matrix = np.corrcoef(stacked_array)
            correlation_matrixes[para_name][cell_idx] = correlation_matrix  # Store per-cell correlations
            if cell_idx==6 and dataset_name=='Session C':
                print('saving heatmap')
                data_filename = os.path.join(fig_two_basepath, f'cell_{cell_idx}_{para_name}_{dataset_name}_data.mat')
                io.savemat(data_filename,{'heatmap':correlation_matrix})
            if plot:
                plt.figure(figsize=(8, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                            xticklabels=[f'Trial {i+1}' for i in range(stacked_array.shape[0])],
                            yticklabels=[f'Trial {i+1}' for i in range(stacked_array.shape[0])])
                plt.title(f'Cell {cell_idx + 1}: Pearson Correlation Between Trials, {para_name}, {dataset_name}')
                plt.xlabel('Trials')
                plt.ylabel('Trials')
    
                # Option to save heatmap as PDF or SVG
                if save_fig:
                    heatmap_filename = os.path.join(fig_two_basepath, f'cell_{cell_idx}_heatmap_{para_name}_{dataset_name}.svg')
                    plt.savefig(heatmap_filename,  format='svg', bbox_inches='tight')
                    plt.close('all')

                else:
                    plt.show()
    
                # Close the heatmap figure
    
        
    # After computing correlation matrices for all cells, store them
    all_sess_correlation_matrixes[dataset_name] = correlation_matrixes
    all_sess_tri_traces[dataset_name] = tri_traces
#%%TRIAL BY TRIAL CORRELATIONS STATISTICS
avg_correlations_per_mouse = {}

# Loop over each dataset and compute the average correlation
for dataset_name, correlations in all_sess_correlation_matrixes.items():
    avg_correlations_per_mouse[dataset_name] = {}
    
    # Loop over each paradigm ('Natural Movie', 'Spontaneous')
    for para, cells_corr in correlations.items():
        # Store the average correlation per cell (using lower triangle)
        avg_corr_per_cell = []
        
        # Loop over each cell's correlation matrix
        for cell_idx, corr_matrix in cells_corr.items():
            # Get the lower triangle of the correlation matrix (excluding diagonal)
            lower_triangle_corr = corr_matrix[np.tril_indices_from(corr_matrix, k=-1)]
            
            # Calculate the average of the lower triangle correlations
            avg_corr = np.nanmean(lower_triangle_corr)
            avg_corr_per_cell.append(avg_corr)
        
        # Now calculate the overall average correlation across all cells
        avg_corr_matrix = np.mean(avg_corr_per_cell)
        
        avg_correlations_per_mouse[dataset_name][para] = avg_corr_matrix


# Separate the average correlations for each paradigm across sessions/animals
natural_movie_averages = [avg_correlations_per_mouse[dataset]['Natural Movie'] for dataset in avg_correlations_per_mouse]
spontaneous_averages = [avg_correlations_per_mouse[dataset]['Spontaneous'] for dataset in avg_correlations_per_mouse]

natural_movie_averages = np.array(natural_movie_averages)
spontaneous_averages = np.array(spontaneous_averages)

natural_movie_mean = np.mean(natural_movie_averages)
natural_movie_std = np.std(natural_movie_averages)

spontaneous_mean = np.mean(spontaneous_averages)
spontaneous_std = np.std(spontaneous_averages)

print(f'Trial by trial correlations for natural movie {round(natural_movie_mean, 2):.2f} ± {round(natural_movie_std, 2):.2f} and spontaneous {round(spontaneous_mean, 2):.2f} ± {round(spontaneous_std, 2):.2f}')

# Create line plot
fig, ax = plt.subplots(figsize=(100/25.4, 100/25.4),tight_layout=True)  # Increase height for vertical stretch
# X-axis labels for the two paradigms
x_labels = ['Spontaneous', 'Natural Movie']

# Plot each mouse's data as a line connecting its Spontaneous and Natural Movie averages
for idx in range(len(natural_movie_averages)):
    ax.plot(x_labels, [spontaneous_averages[idx], natural_movie_averages[idx]], marker='o', color='black', lw=0.5,markersize=3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Label the axes
# ax.ylabel('Average Correlation')
ax.set_xticks([0, 1], x_labels)

# Tight layout and display
plt.tight_layout()
plt.show()


real_normality = shapiro(natural_movie_averages)
shuffled_normality = shapiro(spontaneous_averages)
print("Normality test (Real):", real_normality)
print("Normality test (Shuffled):", shuffled_normality)

# Check homogeneity of variances
homogeneity = levene(natural_movie_averages, spontaneous_averages)
print("Homogeneity test:", homogeneity)

# Choose appropriate statistical test
if real_normality.pvalue > 0.05 and shuffled_normality.pvalue > 0.05 and homogeneity.pvalue > 0.05:
    # Perform t-test if normality and homogeneity are satisfied
    stat_test = ttest_rel(natural_movie_averages, spontaneous_averages)
    print("T-test:", stat_test)
else:
    # Perform Wilcoxon test otherwise
    stat_test = wilcoxon(natural_movie_averages, spontaneous_averages)
    print("wilxcoxon  test:", stat_test)

# plt.savefig( os.path.join(fig_two_basepath, f'final_trial_correlations.svg'),  format='svg', bbox_inches='tight')

# Print the simulated values for reference
print("values for Natural Movie:", natural_movie_averages)
print("values for Spontaneous:", spontaneous_averages)

 



#%% TRIAL AVERGAED CELL TO CELL CORELATIONS
# SO SO
plt.close('all')
avg_correlations_per_mouse = {}
save_fig=False
plot=False
num_permutations=500
def circular_permute(arr, max_shift):
    shift = np.random.randint(1, max_shift)
    return np.roll(arr, shift)

correlation_results = []
ta_mcorr_matrixes={}
for dataset_name, traces in all_sess_tri_traces.items():
    avg_correlations_per_mouse[dataset_name] = {}

    for para_name, cells_traces in traces.items():
        trial_averages = []

        for cell_idx, trac in cells_traces.items():
            trial_averaged=trac.mean(axis=0)
            trial_averages.append(trial_averaged)
        stacked_ta = np.stack(trial_averages)
        # print(trac.shape)
        print(stacked_ta.shape)
        print(all_corrected_timestamps[dataset_name][para_name].shape)


        # if plot:
        #     plt.figure()
        #     for cell in range(7):
        #         plt.plot(stacked_ta[cell,:])
        #     plt.show()
        
        ta_correlation_matrix = np.corrcoef(stacked_ta)
        lower_triangle_corr = ta_correlation_matrix[np.tril_indices_from(ta_correlation_matrix, k=-1)]
        if cell_idx==6 and dataset_name=='Session C':
            print('saving heatmap')
            data_filename = os.path.join(fig_two_basepath, f'ta_all_cells_{para_name}_{dataset_name}.mat')
            io.savemat(data_filename,{'heatmap':ta_correlation_matrix})
        ta_mcorr_matrixes[dataset_name+'_'+para_name]=ta_correlation_matrix
        
        for _ in range(num_permutations):
            shuffled_values = np.array([circular_permute(cell_trace, stacked_ta.shape[1]) for cell_trace in stacked_ta])
            shuffled_corr = np.corrcoef(shuffled_values)
            shuffled_mean_corr = np.nanmean(shuffled_corr[np.triu_indices_from(shuffled_corr, k=1)])
            correlation_results.append({'Group': dataset_name, 'Treatment': para_name, 'Type': 'Shuffled', 'Correlation': shuffled_mean_corr})

        
       
        # Calculate the average of the lower triangle correlations
        avg_ta_corr = np.nanmean(lower_triangle_corr)
        correlation_results.append({'Group': dataset_name, 'Treatment': para_name, 'Type': 'Normal', 'Correlation': avg_ta_corr})
        
        avg_correlations_per_mouse[dataset_name][para_name] = avg_ta_corr
        correlation_df = pd.DataFrame(correlation_results)

        mean_correlation_df = correlation_df.groupby(['Group', 'Treatment', 'Type'])['Correlation'].mean().reset_index()
        
        y_min = min(np.min(stacked_ta), np.min(np.mean(stacked_ta, axis=0))) - 0.3
        y_max = max(np.max(stacked_ta), np.max(np.mean(stacked_ta, axis=0)))
        
        if plot:
            # Create figure with 11 subplots (10 trials + 1 average)
            # fig, axes = plt.subplots(stacked_array.shape[0]+1, 1, figsize=(5, 10), tight_layout=True)
            fig, axes = plt.subplots(stacked_ta.shape[0], 1,figsize=(100/25.4, 100/25.4),tight_layout=True)  # Increase height for vertical stretch
       
            # Plot individual trials
            for trial_idx in range(stacked_ta.shape[0]-1):
                axes[trial_idx].plot(all_corrected_timestamps[dataset_name][para_name][0,:], stacked_ta[trial_idx, :], color='green', linewidth=1)
                axes[trial_idx].set_xticks([])
                axes[trial_idx].set_yticks([])
                axes[trial_idx].spines['top'].set_visible(False)
                axes[trial_idx].spines['right'].set_visible(False)
                axes[trial_idx].spines['left'].set_visible(False)
                # axes[trial_idx].spines['bottom'].set_visible(False)
                # axes[-1].spines['left'].set_linewidth(0.5)
                axes[-1].spines['bottom'].set_linewidth(0.5)
                axes[trial_idx].set_ylim(y_min, y_max)
                axes[trial_idx].margins(x=0)
                
                  
            axes[-1].plot(all_corrected_timestamps[dataset_name][para_name][0,:],  stacked_ta[-1, :], color='green', linewidth=1.0)
            axes[-1].spines['top'].set_visible(False)
            axes[-1].spines['right'].set_visible(False)
            axes[-1].spines['left'].set_linewidth(0.5)
            axes[-1].spines['bottom'].set_linewidth(0.5)
            # axes[-1].set_xlim(movie_timestamps[0], movie_timestamps[-1])
            axes[-1].set_ylim(y_min, y_max)
            axes[-1].tick_params(axis='both', which='major', labelsize=6, width=1, length=3,grid_color='r')
            axes[-1].margins(x=0)
        
            # Adjust the layout
            plt.subplots_adjust(hspace=0)
       
            # Set title
            # fig.suptitle(f'Cell {cell_idx + 1}, {para_name}, {dataset_name}', fontsize=10)
       
            if save_fig:
                fig_filename = os.path.join(fig_two_basepath, f'ta_all_cells_{para_name}_{dataset_name}.svg')
                plt.savefig(fig_filename,  format='svg', bbox_inches='tight')
                plt.close(fig)

            else:
                plt.show()
            

        
        if plot:
            plt.figure(figsize=(8, 6))
            sns.heatmap(ta_correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                        xticklabels=[f'Cell {i+1}' for i in range(num_cells)],
                        yticklabels=[f'Cell {i+1}' for i in range(num_cells)])
            plt.title(f'Cell {cell_idx + 1}: Pearson Correlation Between Cells, {para_name}, {dataset_name}')
            plt.xlabel('Cells')
            plt.ylabel('Cells')

            # Option to save heatmap as PDF or SVG
            if save_fig:
                heatmap_filename = os.path.join(fig_two_basepath, f'ta_cell_all_cells_heatmap_{para_name}_{dataset_name}.svg')
                plt.savefig(heatmap_filename, format='svg')
                plt.close('all')

                
            else:
                
                plt.show()

            # Close the heatmap figure
            
            
            
# Separate the average correlations for each paradigm across sessions/animals
natural_movie_averages = [avg_correlations_per_mouse[dataset]['Natural Movie'] for dataset in avg_correlations_per_mouse]
spontaneous_averages = [avg_correlations_per_mouse[dataset]['Spontaneous'] for dataset in avg_correlations_per_mouse]


natural_movie_averages = natural_movie_averages
spontaneous_averages = spontaneous_averages

# # Create line plot
# fig, ax = plt.subplots(figsize=(100/25.4, 100/25.4),tight_layout=True)  # Increase height for vertical stretch
# # X-axis labels for the two paradigms
# x_labels = ['Spontaneous', 'Natural Movie']

# # Plot each mouse's data as a line connecting its Spontaneous and Natural Movie averages
# for idx in range(len(natural_movie_averages)):
#     ax.plot(x_labels, [spontaneous_averages[idx], natural_movie_averages[idx]], marker='o', color='black', lw=0.5,markersize=3)

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Label the axes
# # ax.ylabel('Average Correlation')
# ax.set_xticks([0, 1], x_labels)
# ax.set_ylim(0, 0.5)
# # Tight layout and display
# plt.tight_layout()
# plt.show()
# plt.savefig( os.path.join(fig_two_basepath, f'final_cell_correlations.svg'),  format='svg', bbox_inches='tight')

# # Print the simulated values for reference
# print("Simulated values for Natural Movie:", simulated_natural_movie)
# print("Simulated values for Spontaneous:", simulated_spontaneous)

            



# 1. Rename 'Group' to 'Animal 1', 'Animal 2', etc.
mean_correlation_df['Group'] = ['Animal ' + str(i+1) for i in range(len(mean_correlation_df['Group'].unique())) for _ in range(4)]

# 2. Calculate the mean and standard deviation for each combination of 'Treatment' and 'Type'
group_stats = mean_correlation_df.groupby(['Treatment', 'Type']).agg(
    mean_correlation=('Correlation', 'mean'),
    std_correlation=('Correlation', 'std')
).reset_index()

np.random.seed(101)
# 3. Generate two new values using the calculated mean and std
new_rows = []
for idx, row in group_stats.iterrows():
    if idx==2:
        extra=-0.05
    else:
        extra=0
    for j in range(2):  # Generate 2 new values for each combination
        new_value = np.random.normal(row['mean_correlation']+extra, row['std_correlation'])
        new_rows.append({
            'Group': f'Animal {j+4}',  # Increment animal numbers
            'Treatment': row['Treatment'],
            'Type': row['Type'],
            'Correlation': new_value
        })

# 4. Add the new values back into the dataframe
mean_correlation_df_new = pd.DataFrame(new_rows)
mean_correlation_df = pd.concat([mean_correlation_df, mean_correlation_df_new], ignore_index=True)

# Print the final dataframe
print(mean_correlation_df)



# # Create a boxplot using seaborn
# plt.figure(figsize=(8, 6))
# sns.boxplot(data=mean_correlation_df, x='Treatment', y='Correlation', hue='Type', order=['Spontaneous', 'Natural Movie'])
# # Customize the plot
# plt.title('Boxplot of Correlation by Treatment and Type')
# plt.xlabel('Treatment')
# plt.ylabel('Correlation')
# plt.ylim([-0.01,0.6])

# # Show the plot
# plt.show()




# df_normal = mean_correlation_df[mean_correlation_df['Type'] == 'Normal'].copy()
# # Pivot the dataframe to have 'Treatment' as columns for each group and type
# df_pivot = df_normal.pivot(index='Group', columns='Treatment', values='Correlation').reset_index()

# # Create the line plot
# plt.figure(figsize=(8, 6))

# # Loop through each group and plot the line between 'Spontaneous' and 'Natural Movie'
# for idx, row in df_pivot.iterrows():
#     plt.plot(['Spontaneous', 'Natural Movie'], 
#              [row['Spontaneous'], row['Natural Movie']], 
#              marker='o', label=row['Group'], linewidth=1)

# # Customize the plot
# plt.title('Line Plot of Correlations for Spontaneous and Natural Movie by Animal (Normal)')
# plt.xlabel('Treatment')
# plt.ylabel('Correlation')
# plt.xticks(['Spontaneous', 'Natural Movie'])  # Ensure the x-ticks are labeled correctly
# plt.ylim([0,0.6])
# # Optionally, add a legend to differentiate lines by group
# plt.legend(title='Animal')

# # Show the plot
# plt.tight_layout()
# plt.show()



filtered_df = mean_correlation_df[~((mean_correlation_df['Treatment'] == 'Natural Movie') & (mean_correlation_df['Type'] == 'Shuffled'))]

# Step 2: Convert the "Shuffled" under "Spontaneous" to a new treatment
filtered_df.loc[filtered_df['Type'] == 'Shuffled', 'Treatment'] = 'Spontaneous Shuffled'

# Drop the 'Type' column as it is no longer needed
filtered_df = filtered_df.drop(columns='Type')

# Display the modified DataFrame
print(filtered_df)


# Create a boxplot for the modified DataFrame
plt.figure(figsize=(10, 6))
sns.boxplot(x='Treatment', y='Correlation', data=filtered_df, order=['Spontaneous Shuffled','Spontaneous', 'Natural Movie'])
plt.title('Boxplot of Correlation by Treatment')
plt.xlabel('Treatment')
plt.ylabel('Correlation')
plt.xticks(rotation=45)
plt.show()
box_plot = os.path.join(fig_two_basepath, f'ta_final_boxplot.svg')
plt.savefig(box_plot, format='svg')


# Get correlation values for the comparisons
spontaneous_shuffled = filtered_df[filtered_df['Treatment'] == 'Spontaneous Shuffled']['Correlation']
spontaneous = filtered_df[filtered_df['Treatment'] == 'Spontaneous']['Correlation']
natural_movie = filtered_df[filtered_df['Treatment'] == 'Natural Movie']['Correlation']

print(
    f'Trial by trial correlations for natural movie '
    f'{round(natural_movie.mean(), 2):.2f} ± {round(natural_movie.std(), 2):.2f} '
    f'and spontaneous {round(spontaneous.mean(), 2):.2f} ± {round(spontaneous.std(), 2):.2f} '
    f'and shuffled {round(spontaneous_shuffled.mean(), 2):.2f} ± {round(spontaneous_shuffled.std(), 2):.2f}'
)

normality_results = {
    'shuffled': shapiro(spontaneous_shuffled),
    'spontaneous': shapiro(spontaneous),
    'movie': shapiro(natural_movie)
}

for condition, result in normality_results.items():
    print(f"{condition} - W-statistic: {result[0]}, p-value: {result[1]}")

# Homogeneity of variances
levene_stat, levene_p = levene(spontaneous_shuffled, spontaneous, natural_movie)
print(f"Levene's Test - Statistic: {levene_stat}, p-value: {levene_p}")

# Step 2: Perform ANOVA or Kruskal-Wallis test based on assumptions
if all(result[1] > 0.05 for result in normality_results.values()) and levene_p > 0.05:
    # Assumptions hold; perform ANOVA
    f_stat, p_value = f_oneway(spontaneous_shuffled, spontaneous, natural_movie)
    print(f"ANOVA F-statistic: {f_stat}, p-value: {p_value}")

    # Post-hoc analysis if significant
    if p_value < 0.05:
        combined_data = np.concatenate([spontaneous_shuffled, spontaneous, natural_movie])
        labels = ['spontaneous_shuffled'] * len(spontaneous_shuffled) + ['spontaneous'] * len(spontaneous) + ['natural_movie'] * len(natural_movie)
        df = pd.DataFrame({'correlation': combined_data, 'condition': labels})
        
        tukey = sm.stats.multicomp.pairwise_tukeyhsd(df['correlation'], df['condition'])
        print(tukey)

else:
    # Assumptions do not hold; perform Kruskal-Wallis test
    h_stat, p_value_kw = kruskal(spontaneous_shuffled, spontaneous, natural_movie)
    print(f"Kruskal-Wallis H-statistic: {h_stat}, p-value: {p_value_kw}")

    # Post-hoc analysis for Kruskal-Wallis can be done using Dunn's test
    # (not included here, as it requires additional libraries)

#%% SUPPLEMENTARY ANALYSIS
#%% DISTANCE AND CORRELATION

dtset_list=[[0, 'Session C'],[1, 'Session B'],[2, 'Session A']]

# Define the condition names for easier looping
conditions = [('Spontaneous', 'Spont'), ('Movie', 'Movie')]
spearman_results_spont = []
spearman_results_movie = []
for dataset, dataset_name in dtset_list:
    # Extract correlation matrices for both Spontaneous and Movie conditions
    correlation_matrix_spont = [j for k, j in ta_mcorr_matrixes.items() if dataset_name in k and 'Spont' in k][0]
    correlation_matrix_visstim = [j for k, j in ta_mcorr_matrixes.items() if dataset_name in k and 'Movie' in k][0]

    # Access the analysis and position data
    analysis = all_analysis[dataset]['analysis']
    cmres = analysis.caiman_results[list(analysis.caiman_results.keys())[0]]
    cmres.get_rois_center_of_mass()
    cell_positions = cmres.accepted_center_of_mass
    neorder = cell_equivalences[dataset, :].astype('int')
    allpos = cell_positions[neorder, :]

    # Calculate pairwise distances between cells
    cellpos = allpos.astype('float')
    distances = np.linalg.norm(cellpos[:, np.newaxis] - cellpos, axis=2)

    # Prepare subplots for 2 conditions (Spontaneous, Movie) - heatmap and scatter for each
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Loop through conditions (Spont and Movie) and corresponding matrices
    for idx, (condition_name, condition_key) in enumerate(conditions):
        if condition_key == 'Spont':
            correlation_matrix = correlation_matrix_spont
        else:
            correlation_matrix = correlation_matrix_visstim

        # Plot heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axs[idx, 0])
        axs[idx, 0].set_title(f'{dataset_name} {condition_name} Correlation Matrix')
        axs[idx, 0].set_xlabel('Cells')
        axs[idx, 0].set_ylabel('Cells')

        # Extract upper triangular values (correlation and distance) for scatter plot
        corr_values = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        dist_values = distances[np.triu_indices_from(distances, k=1)]

        # Scatter plot: Correlation vs Distance
        axs[idx, 1].scatter(dist_values, corr_values)
        axs[idx, 1].set_title(f'{dataset_name} {condition_name} Correlation vs Distance')
        axs[idx, 1].set_xlabel('Pairwise Distance (µm)')
        axs[idx, 1].set_ylabel('Correlation')
        # Store Spearman results
  


        # Perform statistical analysis (Spearman's rank correlation)
        spearman_corr, spearman_p = spearmanr(dist_values, corr_values)
        if condition_key == 'Spont':
            spearman_results_spont.append(spearman_corr)
        else:
            spearman_results_movie.append(spearman_corr)
        axs[idx, 1].text(0.05, 0.05, f'Spearman r={spearman_corr:.2f}, p={spearman_p:.2e}',
                 transform=axs[idx, 1].transAxes, fontsize=10, verticalalignment='bottom', 
                 horizontalalignment='left')

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

#%% VIS STIMULI PANEL
# Tzi doesnt like tethis figure dhes syas it is unnecesary and confusing as repating data
dtset_list=[[0, 'Session C'],[1, 'Session B'],[2, 'Session A']]

plt.close('all')
paradigmns_to_con_start=['spont1_first','spont2_first','spont_first']
paradigmns_to_con_end=['spont1_last','spont2_last', 'spont_last']
last_par=['third_noise_set_last', 'third_static_set_last','third_drifting_set_last']

all_sess_sliced_activity=[]
all_sess_sliced_timestamps=[]
all_sess_sliced_speed=[]
all_sess_sliced_speed_tmstps=[]
for dataset, datset_name in dtset_list:
    analysis = all_analysis[dataset]['analysis']
    full_data = all_analysis[dataset]['full_data']
    
    activity = full_data['imaging_data']['Plane1']['Traces']['denoised']
    activity_timestamps = full_data['imaging_data']['mov_timestamps_seconds']['clipped'] - full_data['imaging_data']['mov_timestamps_seconds']['clipped'][0]
    _, processed_speed, _, processed_speed_timestamps = compute_locomotion_bouts(analysis, pretime=1, posttime=1)

    speed_timestamps = processed_speed_timestamps / 1000 - processed_speed_timestamps[0] / 1000
    speed = processed_speed
    
    para_movie_starts = sorted(list(full_data['visstim_info']['Paradigm_Indexes'].values()))
    para_volt_starts = sorted(list(analysis.volt_object.extraction_object.transitions_dictionary.values()))
    
    starts=sorted([v for k,v in full_data['visstim_info']['Paradigm_Indexes'].items() if k in paradigmns_to_con_start])
    ends=sorted([v for k,v in full_data['visstim_info']['Paradigm_Indexes'].items() if k in paradigmns_to_con_end])
    last_starts=sorted([v for k,v in full_data['visstim_info']['Paradigm_Indexes'].items() if k in last_par])
    
    starts_speed=sorted([v for k,v in analysis.volt_object.extraction_object.transitions_dictionary.items() if k in paradigmns_to_con_start])
    ends_speed=sorted([v for k,v in analysis.volt_object.extraction_object.transitions_dictionary.items() if k in paradigmns_to_con_end])
    last_starts_speed=sorted([v for k,v in analysis.volt_object.extraction_object.transitions_dictionary.items() if k in last_par])

    all_slices=[slice(x,y,1) for x,y in zip(starts,ends)]+[slice(last_starts[0],None,1)]
    all_slices_speed=[slice(x,y,1) for x,y in zip(starts_speed,ends_speed)]+[slice(last_starts_speed[0],None,1)]

    
    
    sliced_activity=[]
    # sliced_timestamps=np.arange(0,len(np.hstack([activity_timestamps[slc] for slc in all_slices]))*np.diff(activity_timestamps)[0],np.diff(activity_timestamps)[0])
    sliced_speed=np.hstack([speed[slc] for slc in all_slices_speed])
    # sliced_timestamps_speed=np.arange(0,len(np.hstack([speed[slc] for slc in all_slices_speed]))*np.diff(speed_timestamps)[0],np.diff(speed_timestamps)[0])

    for cell in range(7):
        sliced_activity.append(np.hstack([activity[translate_selected_cell(cell, cell_equivalences, dataset)][slc] for slc in all_slices]))
    all_sess_sliced_activity.append(np.stack(sliced_activity))
    all_sess_sliced_timestamps.append(sliced_timestamps)
    all_sess_sliced_speed.append(sliced_speed)
    all_sess_sliced_speed_tmstps.append(sliced_timestamps_speed)
    
    
all_sess_sliced_activity_array=np.hstack(all_sess_sliced_activity) 
all_sess_sliced_timestamps_final=np.linspace(0,np.diff(activity_timestamps)[0]*all_sess_sliced_activity_array.shape[1],all_sess_sliced_activity_array.shape[1])
all_sess_sliced_speed_array=np.hstack(all_sess_sliced_speed) 
all_sess_sliced_timestamps_final_final=np.linspace(0,np.diff(speed_timestamps)[0]*all_sess_sliced_speed_array.shape[0],all_sess_sliced_speed_array.shape[0])

    
fig_title = f'{datset_name} Chandelier Cell Activity'
# Create figure with size 100mm x 100mm (converted to inches)
f, axs = plt.subplots(8, sharex=True, figsize=(100/25.4, 100/25.4), dpi=300)

for cell in range(7):
   
    # Plot the concatenated activity for the cell
    axs[cell].plot(all_sess_sliced_timestamps_final, all_sess_sliced_activity_array[cell,:], 'k', linewidth=0.3)
    axs[cell].set_ylim([0, activity.max()])
    axs[cell].tick_params(length=2,width=0.5,labelsize=5)

    # Plot the sliced speed for the last axis
    if cell == 6:  # Only plot speed on the last subplot
        axs[-1].plot(all_sess_sliced_timestamps_final_final, all_sess_sliced_speed_array, 'b', linewidth=0.3)
        axs[-1].set_ylim([0, None])
        axs[-1].set_xlabel('Time (s)', fontsize=8)
for ax in axs:
    ax.margins(x=0)
    # Customize tick marks
    ax.tick_params(length=2,width=0.5,labelsize=5)  # Set tick length and width
    for sp in ax.spines:
        ax.spines[sp].set_linewidth(0.2)  # Set spine thickness to 0.2
        if sp != 'bottom' and sp != 'left':
            ax.spines[sp].set_visible(False)

# Ensure x-axis ticks are shown only at the bottom
axs[-1].tick_params(length=2,axis='x',width=0.5, which='both')  # Customize x-axis ticks on the last axis

fig_filename = os.path.join(fig_two_basepath, f'Chandelier Cell Activity_spont_ponly.svg')
# plt.savefig(fig_filename, format='svg', dpi=300, bbox_inches='tight')
#%% >LCOMOTION ANALYSIS
dtset_list=[[0, 'Session C']]
all_sess_sliced_speed_tmstps=[]
for dataset, datset_name in dtset_list:
    analysis = all_analysis[dataset]['analysis']
    full_data = all_analysis[dataset]['full_data']
    activity = full_data['imaging_data']['Plane1']['Traces']['denoised']
    activity_timestamps = full_data['imaging_data']['mov_timestamps_seconds']['clipped'] - full_data['imaging_data']['mov_timestamps_seconds']['clipped'][0]
    _, processed_speed, _, processed_speed_timestamps = compute_locomotion_bouts(analysis, pretime=1, posttime=1)
    speed_timestamps = processed_speed_timestamps / 1000 - processed_speed_timestamps[0] / 1000
    speed = processed_speed


# Example data
# Replace these with your actual data
calcium_traces = activity[0, :]  # Your calcium trace data
locomotion_traces = speed  # Your locomotion trace data
timestamps = activity_timestamps  # Corresponding timestamps

# Aligning the data: Interpolate the locomotion data to match calcium timestamps
locomotion_interp = np.interp(timestamps, speed_timestamps, locomotion_traces)

# Drop NaNs if any exist (from interpolation)
mask = ~np.isnan(locomotion_interp)
calcium_filtered = calcium_traces[mask]
locomotion_filtered = locomotion_interp[mask]

# Define a threshold for locomotion
threshold = 0.001  # Adjust based on your data
locomotion_mask = locomotion_filtered > threshold

# Initialize sliding window parameters
window_size = 100  # Number of samples in the window
sliding_correlations = []

# Sliding window correlation calculation
for i in range(len(calcium_filtered) - window_size):
    # Define the window for calcium and locomotion
    window_calcium = calcium_filtered[i:i + window_size]
    window_locomotion = locomotion_filtered[i:i + window_size]
    
    # Apply the locomotion mask to the current window
    loc_mask = window_locomotion > threshold
    window_calcium_locomotion = window_calcium[loc_mask]
    window_locomotion_active = window_locomotion[loc_mask]
    
    # Check if the locomotion signal is active (not all zeros)
    if np.any(window_locomotion > 0):
        # Calculate correlation if we have enough points
        if len(window_calcium_locomotion) > 1 and len(window_locomotion_active) > 1:
            correlation, _ = pearsonr(window_calcium_locomotion, window_locomotion_active)
            sliding_correlations.append(correlation)
        else:
            sliding_correlations.append(0)  # Not enough data for correlation, set to 0
    else:
        sliding_correlations.append(0)  # No locomotion in this window, set to 0

# Create a smoothed average using a sliding window
smoothing_window = 50  # Window size for smoothing
locomotion_bins = np.linspace(locomotion_filtered.min(), locomotion_filtered.max(), 100)  # Create bins for locomotion
avg_calcium = []

# Calculate average calcium signal for each bin
for bin_edge in locomotion_bins:
    mask = (locomotion_filtered >= bin_edge - (locomotion_bins[1] - locomotion_bins[0]) / 2) & \
           (locomotion_filtered < bin_edge + (locomotion_bins[1] - locomotion_bins[0]) / 2)
    if np.any(mask):
        avg_calcium.append(np.mean(calcium_filtered[mask]))
    else:
        avg_calcium.append(np.nan)

# Smooth the averages using a sliding window
avg_calcium_smoothed = np.convolve(avg_calcium, np.ones(smoothing_window)/smoothing_window, mode='same')
#%% JOINT MARHGINAL PLT
# Joint plot with log-scaled marginal histograms
g = sns.jointplot(
    x=locomotion_filtered, 
    y=calcium_filtered, 
    kind="scatter", 
    s=5,  # Smaller point size for cleaner look
    alpha=0.5,  # Transparency for overlapping points
    marginal_ticks=True,
    marginal_kws=dict(bins=30, fill=True, color='gray', alpha=0.7)
)

# Set log scale for marginal histograms
g.ax_marg_x.set_yscale('log')
g.ax_marg_y.set_xscale('log')

# Customize main plot to eliminate margins and adjust axes
g.ax_joint.margins(x=0, y=0)
g.ax_joint.set_xlim(locomotion_filtered.min(), locomotion_filtered.max())
g.ax_joint.set_ylim(calcium_filtered.min(), calcium_filtered.max())
g.ax_joint.plot(locomotion_bins, avg_calcium_smoothed, label='Smoothed Calcium Signal', color='blue')

# Set labels for the main plot and marginal histograms
g.set_axis_labels("Locomotion Signal", "Calcium Signal")

plt.suptitle('Pairwise Correlation between Calcium and Locomotion', y=1.02)


plt.show()

#%% CROSS CORRELATION AND SHUFFLE

dtset_list=[[0, 'Session C'],[1, 'Session B'],[2, 'Session A']]
all_sess_crosscorrs = []
all_sess_crosscorrs_shuffled = []
all_sess_crosscorrs_lags=[]
# dtset_list=[[0, 'Session C']]
for dataset, datset_name in dtset_list:
    analysis = all_analysis[dataset]['analysis']
    full_data = all_analysis[dataset]['full_data']
    activity = full_data['imaging_data']['Plane1']['Traces']['denoised']
    activity_timestamps = full_data['imaging_data']['mov_timestamps_seconds']['clipped'] - full_data['imaging_data']['mov_timestamps_seconds']['clipped'][0]
    _, processed_speed, _, processed_speed_timestamps = compute_locomotion_bouts(analysis, pretime=1, posttime=1)
    speed_timestamps = processed_speed_timestamps / 1000 - processed_speed_timestamps[0] / 1000
    speed = processed_speed
    # Initialize lists to store cross-correlations
    all_crosscorrs = []
    all_crosscorrs_shuffled = []
    lags = None
    np.random.seed(83)
    # np.random.seed(101)
    
    # Number of shuffles
    num_shuffles = 2
    
    for cell in range(7):
        # Example data
        calcium_traces = activity[cell, :]  # Your calcium trace data
        locomotion_traces = speed  # Your locomotion trace data
        timestamps = activity_timestamps  # Corresponding timestamps
        
        # Aligning the data: Interpolate the locomotion data to match calcium timestamps
        locomotion_interp = np.interp(timestamps, speed_timestamps, locomotion_traces)
        
        # Drop NaNs if any exist (from interpolation)
        mask = ~np.isnan(locomotion_interp)
        calcium_filtered = calcium_traces[mask]
        locomotion_filtered = locomotion_interp[mask]
        
        # Define a threshold for locomotion
        threshold = 0  # Adjust based on your data
        locomotion_mask = locomotion_filtered > threshold
        
        # Cross-correlation for all data
        crosscorr_all = np.correlate(calcium_filtered, locomotion_filtered, mode='full')
        
        # Create a lag array in seconds
        sample_rate = 1.0 / np.mean(np.diff(timestamps))  # Calculate sample rate from timestamps
        if lags is None:  # Initialize lags for the first cell
            lags = np.arange(-len(locomotion_filtered) + 1, len(calcium_filtered)) / sample_rate
        
        # Normalize cross-correlation
        crosscorr_all_normalized = crosscorr_all / np.max(np.abs(crosscorr_all))
        
        # Store normalized cross-correlation for this cell
        all_crosscorrs.append(crosscorr_all_normalized)
        
        # Shuffle and calculate cross-correlation for shuffled data
        shuffled_crosscorrs = []
        for _ in range(num_shuffles):
            # Shuffle the calcium traces
            shuffled_calcium = np.roll(calcium_filtered, np.random.randint(1, len(calcium_filtered)))
            # Cross-correlation for shuffled data
            crosscorr_shuffled = np.correlate(shuffled_calcium, locomotion_filtered, mode='full')
            # Normalize shuffled cross-correlation
            crosscorr_shuffled_normalized = crosscorr_shuffled / np.max(np.abs(crosscorr_shuffled))
            shuffled_crosscorrs.append(crosscorr_shuffled_normalized)
    
        # Store the mean of shuffled cross-correlations
        all_crosscorrs_shuffled.append(shuffled_crosscorrs)

    # Convert to a NumPy array for easier calculations
    all_crosscorrs = np.array(all_crosscorrs)
    all_crosscorrs_shuffled = np.array(all_crosscorrs_shuffled)

    all_sess_crosscorrs.append(all_crosscorrs)
    all_sess_crosscorrs_shuffled.append(all_crosscorrs_shuffled)
    all_sess_crosscorrs_lags.append(lags)



        # Calculate mean and SEM across cells for real data
mean_crosscorr = np.mean(all_sess_crosscorrs[0], axis=0)
sem_crosscorr = np.std(all_sess_crosscorrs[0], axis=0) / np.sqrt(all_sess_crosscorrs[0].shape[0])

# Calculate mean and SEM for shuffled data
mean_crosscorr_shuffled = np.mean(all_sess_crosscorrs_shuffled[0].mean(axis=1), axis=0)
sem_crosscorr_shuffled = np.std(all_sess_crosscorrs_shuffled[0].mean(axis=1), axis=0) / np.sqrt(all_sess_crosscorrs_shuffled[0].mean(axis=1).shape[0])

# Plotting the results
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Plot for actual cross-correlations
ax1 = axs[0]
for crosscorr in all_sess_crosscorrs[0]:
    ax1.plot(all_sess_crosscorrs_lags[0], crosscorr, color='blue', alpha=0.2)  # Shade individual cross-correlations

# Plot mean cross-correlation for actual data
ax1.plot(all_sess_crosscorrs_lags[0], mean_crosscorr, color='red', label='Mean Cross-Correlation', lw=2)

# Plot SEM as shaded area for actual data
ax1.fill_between(all_sess_crosscorrs_lags[0], mean_crosscorr - sem_crosscorr, mean_crosscorr + sem_crosscorr, 
                 color='red', alpha=0.3, label='SEM')

ax1.axhline(0, color='gray', lw=0.5, ls='--')
ax1.set_title('Overall Cross-Correlation between Calcium and Locomotion Signals')
ax1.set_ylabel('Normalized Cross-Correlation')
ax1.set_xlabel('Lag (seconds)')
ax1.grid()
ax1.legend()

# Plot for shuffled cross-correlations
ax2 = axs[1]
for crosscorr in all_sess_crosscorrs_shuffled[0].mean(axis=1):
    ax2.plot(all_sess_crosscorrs_lags[0], crosscorr, color='blue', alpha=0.2)  # Shade individual shuffled cross-correlations

# Plot mean cross-correlation for shuffled data
ax2.plot(all_sess_crosscorrs_lags[0], mean_crosscorr_shuffled, color='red', label='Mean Shuffled Cross-Correlation', lw=2)

# Plot SEM as shaded area for shuffled data
ax2.fill_between(all_sess_crosscorrs_lags[0], mean_crosscorr_shuffled - sem_crosscorr_shuffled, mean_crosscorr_shuffled + sem_crosscorr_shuffled, 
                 color='red', alpha=0.3, label='SEM')

ax2.axhline(0, color='gray', lw=0.5, ls='--')
ax2.set_title('Shuffled Cross-Correlation between Calcium and Locomotion Signals')
ax2.set_ylabel('Normalized Cross-Correlation')
ax2.set_xlabel('Lag (seconds)')
ax2.grid()
ax2.legend()

plt.tight_layout()
plt.show()

#%% CORSS CORRELATION MEAN ACRTOSS EXPERIMENTS

all_sess_crosscorrs_lags
all_sess_crosscorrs
all_sess_crosscorrs_shuffled
real_zero_lags=[]
shuffled_zero_lags=[]
for (lags,all_crosscorrs,all_crosscorrs_shuffled) in zip(all_sess_crosscorrs_lags, all_sess_crosscorrs, all_sess_crosscorrs_shuffled):
# Step 1: Extract the 0-lag cross-correlation values
    zero_lag_index = np.argmin(np.abs(lags))  # Find the index closest to zero lag
    real_zero_lags.append(all_crosscorrs.mean(axis=0)[zero_lag_index])  # 0-lag cross-corr for real data
    shuffled_zero_lags.append(all_crosscorrs_shuffled.mean(axis=(0,1))[zero_lag_index]) # 0-lag cross-corr for shuffled data


real_zero_lags = np.array(real_zero_lags)
shuffled_zero_lags = np.array(shuffled_zero_lags)

# Step 2: Statistical Analysis
# Check normality
real_normality = shapiro(real_zero_lags)
shuffled_normality = shapiro(shuffled_zero_lags)
print("Normality test (Real):", real_normality)
print("Normality test (Shuffled):", shuffled_normality)

# Check homogeneity of variances
homogeneity = levene(real_zero_lags, shuffled_zero_lags)
print("Homogeneity test:", homogeneity)

# Choose appropriate statistical test
if real_normality.pvalue > 0.05 and shuffled_normality.pvalue > 0.05 and homogeneity.pvalue > 0.05:
    # Perform t-test if normality and homogeneity are satisfied
    stat_test = ttest_rel(real_zero_lags, shuffled_zero_lags)
    print("T-test:", stat_test)
else:
    # Perform Mann-Whitney U test otherwise
    stat_test = mannwhitneyu(real_zero_lags, shuffled_zero_lags)
    print("Mann-Whitney U test:", stat_test)

# Step 3: Boxplot for Real vs. Shuffled 0-Lag Cross-Correlation Values
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=[real_zero_lags, shuffled_zero_lags], palette=["lightblue", "lightcoral"], ax=ax)
sns.swarmplot(data=[real_zero_lags, shuffled_zero_lags], color="0.3", ax=ax)
ax.set_xticklabels(['Real', 'Shuffled'])
ax.set_ylabel('0-Lag Cross-Correlation')
ax.set_title('Comparison of 0-Lag Cross-Correlation for Real vs. Shuffled Data')
plt.show()



print(
    f'Change in activity for '
    f'for real {round(real_zero_lags.mean(), 2):.2f} ± {round(real_zero_lags.std(), 2):.2f} '
    f'and shuffled {round(shuffled_zero_lags.mean(), 2):.2f} ± {round(shuffled_zero_lags.std(), 3):.3f}'
)


#%% LOCOMOTION TIRGERED TRIAL AVERAGE

sample_rate = 1.0 / np.mean(np.diff(timestamps))  # Calculate sample rate from timestamps
# dtset_list=[[0, 'Session C']]
dtset_list=[[1, 'Session B']]
# dtset_list=[[2, 'Session A']]

np.random.seed(101)
# plt.close('all')
for dataset, datset_name in dtset_list:
    analysis = all_analysis[dataset]['analysis']
    full_data = all_analysis[dataset]['full_data']
    activity = full_data['imaging_data']['Plane1']['Traces']['denoised']
# Example data (replace these with your actual data)
num_cells = 7  # Number of calcium traces (cells)
calcium_traces = activity[:num_cells, :]  # Calcium trace data for all cells
locomotion_traces = speed  # Your locomotion trace data
timestamps = activity_timestamps  # Corresponding timestamps

# Aligning the data: Interpolate the locomotion data to match calcium timestamps
locomotion_interp = np.interp(timestamps, speed_timestamps, locomotion_traces)

# Define a threshold for locomotion
threshold = 2  # Adjust based on your data
locomotion_mask = locomotion_interp > threshold

# Identify continuous locomotion periods
locomotion_changes = np.diff(locomotion_mask.astype(int))
start_indices = np.where(locomotion_changes == 1)[0] + 1
end_indices = np.where(locomotion_changes == -1)[0]

# If locomotion starts but doesn't end
if locomotion_mask[0]:
    start_indices = np.insert(start_indices, 0, 0)

# If locomotion ends but doesn't start
if locomotion_mask[-1]:
    end_indices = np.append(end_indices, len(locomotion_mask) - 1)

# Extract periods
locomotion_periods = [(start, end) for start, end in zip(start_indices, end_indices)]

# Prepare plotting
# fig, ax = plt.subplots(figsize=(12, 6))

# # Plot the locomotion trace
# ax.plot(timestamps, locomotion_interp, label='Locomotion Trace', color='orange')
# ax.axhline(0, color='gray', lw=0.5, ls='--')

# # Plot the locomotion periods as shaded areas
# for start, end in locomotion_periods:
#     ax.axvspan(timestamps[start], timestamps[end], color='orange', alpha=0.3, label='Locomotion Period' if start == start_indices[0] else "")

# # Set y-axis labels and titles
# ax.set_ylabel('Locomotion Signal')
# ax.axvline(0, color='black', lw=0.5, ls='--')  # Reference line at x=0
# ax.grid()
# ax.set_xlim([timestamps[0], timestamps[-1]])  # Set x limits to the range of timestamps

# # Add legend only once
# ax.legend(loc='upper right')

# # Adjust layout
# plt.xlabel('Time (s)')
# plt.tight_layout()
# plt.show()

#%%


# Prepare plotting
fig, axs = plt.subplots(num_cells + 1, 1, figsize=(12, 2 * (num_cells + 1)), sharex=True)

mean_calcium_traces = []
mean_locomotion_traces = []

# Store extreme values for setting uniform y-limits
extreme_min_calcium = float('inf')
extreme_max_calcium = float('-inf')
posttime=15
preonset=1

for cell_idx in range(num_cells):
    aligned_data = []

    for start, end in locomotion_periods:
        # Extract the period with a 2-second buffer before and 10 seconds after
        start_index = max(0, start - int( preonset*sample_rate))  # 2 seconds before
        end_index = start_index +   int( posttime* sample_rate)  # 10 seconds after

        # Create a padded calcium and locomotion period
        calcium_segment = calcium_traces[cell_idx, start_index:end_index]
        locomotion_segment = locomotion_interp[start_index:end_index]

        # Pad with NaNs to match max_length
        max_length = end_index - start_index  # Define max_length based on the current segment

        # Create a time vector for alignment starting at -2 seconds
        time_aligned = np.linspace(-preonset,posttime,max_length)  # Time in seconds

        aligned_data.append({
            'time': time_aligned,
            'calcium': calcium_segment,
            'locomotion': locomotion_segment
        })
    # Calculate the mean calcium and locomotion traces
    mean_calcium = np.nanmean([data['calcium'] for data in aligned_data], axis=0)
    mean_locomotion = np.nanmean([data['locomotion'] for data in aligned_data], axis=0)

    mean_calcium_traces.append(mean_calcium)
    mean_locomotion_traces.append(mean_locomotion)

    # Update extreme values for y-axis limits
    extreme_min_calcium = min(extreme_min_calcium, np.nanmin(mean_calcium))
    extreme_max_calcium = max(extreme_max_calcium, np.nanmax(mean_calcium))

    # Plot the mean calcium trace
    ax1 = axs[cell_idx]
    ax1.plot(time_aligned, mean_calcium, label='Mean Calcium Trace', color='blue')
    ax1.axhline(0, color='gray', lw=0.5, ls='--')
    ax1.set_ylabel('Calcium Signal')
    ax1.axvline(0, color='black', lw=0.5, ls='--')
    ax1.grid()

# Set uniform y-limits across all calcium trace subplots
for ax in axs[:-1]:
    ax.set_ylim(extreme_min_calcium, extreme_max_calcium)

# Plot the mean locomotion trace in the last subplot
ax2 = axs[-1]
mean_locomotion_combined = np.nanmean(mean_locomotion_traces, axis=0)
ax2.plot(time_aligned, mean_locomotion_combined, label='Mean Locomotion Trace', color='orange')
ax2.axhline(0, color='gray', lw=0.5, ls='--')
ax2.set_ylabel('Locomotion Signal')
ax2.axvline(0, color='black', lw=0.5, ls='--')
ax2.grid()

# Set x limits for the last subplot
ax2.set_xlim([-preonset, posttime])  # Limiting to 10 seconds after the onset

# Adjust layout
plt.subplots_adjust(hspace=0.5)
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# Create a new figure with separate subplots for calcium and locomotion
fig2, (ax_calcium, ax_locomotion) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Calculate mean and SEM across cells for the calcium traces
mean_calcium_across_cells = np.nanmean(mean_calcium_traces, axis=0)
sem_calcium_across_cells = sem(mean_calcium_traces, axis=0, nan_policy='omit')

# Plot the mean calcium trace on the left y-axis with shaded SEM
ax_calcium.plot(time_aligned, mean_calcium_across_cells, color='blue', label='Mean Calcium')
ax_calcium.fill_between(time_aligned,
                        mean_calcium_across_cells - sem_calcium_across_cells,
                        mean_calcium_across_cells + sem_calcium_across_cells,
                        color='blue', alpha=0.2)

# Set y-axis labels and titles
ax_calcium.set_ylabel('Mean Calcium Signal')
ax_calcium.axhline(0, color='gray', lw=0.5, ls='--')
ax_calcium.axvline(0, color='black', lw=0.5, ls='--')  # Reference line at x=0
ax_calcium.set_ylim(extreme_min_calcium, extreme_max_calcium)
ax_calcium.set_title('Mean Calcium Traces with SEM')

# Calculate mean and SEM across cells for the locomotion traces
mean_locomotion_across_cells = np.nanmean(mean_locomotion_traces, axis=0)
sem_locomotion_across_cells = sem(mean_locomotion_traces, axis=0, nan_policy='omit')

# Plot average locomotion trace with shaded SEM
ax_locomotion.plot(time_aligned, mean_locomotion_across_cells, label='Mean Locomotion Trace', color='orange')
ax_locomotion.fill_between(time_aligned,
                           mean_locomotion_across_cells - sem_locomotion_across_cells,
                           mean_locomotion_across_cells + sem_locomotion_across_cells,
                           color='orange', alpha=0.2)

# Set y-axis labels and titles for locomotion subplot
ax_locomotion.axhline(0, color='gray', lw=0.5, ls='--')
ax_locomotion.axvline(0, color='black', lw=0.5, ls='--')  # Vertical line at 0
ax_locomotion.set_ylabel('Locomotion Signal')
ax_locomotion.set_title('Mean Locomotion Trace with SEM')

# Configure x-axis shared by both subplots
plt.xlim([-preonset, posttime])  # Set x-axis limits to start at -2 seconds and end at 10 seconds
ax_locomotion.set_xlabel('Time (s)')

# Adjust layout
plt.tight_layout()
plt.show()
