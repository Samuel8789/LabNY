#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:57:03 2024

@author: sp3660
step by step trac eprocessing
"""
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
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
# import zetapy as pz
matplotlib.use('Agg')
import random
from sys import platform
if platform == "linux" or platform == "linux2":
    basepath=Path(r'/home/samuel/Dropbox/Projects/LabNY')
elif platform == "win32":
    basepath=Path(r'C:\Users\sp3660\Desktop')
    
    
temppath=basepath / 'TempPythonFigs'
tempprocessingpat= basepath / 'TempPythonObjects'


def plot_image_with_mean_and_std(data,  x_values=None, labels=None, line_index=None, use_std=True, 
                                 cmap='viridis', interpolation='nearest', aspect='auto', origin='upper', 
                                 vmin=None, vmax=None,ylim=None, alpha=None, fig_size=(8, 6),
                                 fig_title='Fig_',raster_label='Default',compare='opto',log=False,show=False, noraster=False):
    
    """
    Plot a 2D array as an image with a plot showing the mean along the y-axis with shaded standard deviation or standard error of the mean, and use a given vector for the x-axis.

    Parameters:
    -----------
    data : array_like
        The 2D array to be plotted.

    x_values : array_like, optional (default=None)
        Vector for the x-axis.

    labels : array_like, optional (default=None)
        Labels for the x-axis. If None, no labels will be shown.

    line_index : int or None, optional (default=None)
        Index at which to draw the vertical line. If None, no line will be drawn.

    use_std : bool, optional (default=True)
        If True, shaded standard deviation is plotted; if False, shaded standard error of the mean is plotted.

    cmap : str or Colormap, optional (default='viridis')
        The colormap to use.

    interpolation : str, optional (default='nearest')
        The interpolation method used. See Matplotlib imshow documentation for available options.

    aspect : {'equal', 'auto'} or float, optional (default='auto')
        Aspect ratio of the plot. 'equal' ensures equal aspect ratio, 'auto' adjusts aspect ratio based on data shape, float allows specifying specific aspect ratio.

    origin : {'upper', 'lower'}, optional (default='upper')
        Position of the [0, 0] index of the array in the plot.

    vmin, vmax : float, optional
        Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.

    alpha : float, optional
        The alpha blending value.

    fig_size : tuple, optional (default=(8, 6))
        Figure size in inches (width, height).



    Returns:
    --------
    ax : matplotlib Axes
        The axes on which the image was drawn.
    """
    temppath=Path(r'C:\Users\sp3660\Desktop\TempPythonFigs')

    def plot_trials_mean_1data(gs,axes_indexes, data,title, x_values=None, labels=None, line_index=None, use_std=True, cmap='viridis', interpolation='nearest', aspect='auto', origin='upper', vmin=None, vmax=None,ylim=None, alpha=None,noraster=noraster):
       

       if not noraster:
           ax_bottom = plt.subplot(gs[axes_indexes[1]])
           ax = plt.subplot(gs[axes_indexes[0]])


           img = ax.imshow(data, cmap=cmap, interpolation=interpolation, aspect=aspect, origin=origin, vmin=vmin, vmax=vmax, alpha=alpha)
           column_width = 1.0 / (data.shape[1] + 1)
   
           # Calculate the leftmost position for each column
           leftmost_positions = np.linspace(-0.5, 0.5, num=data.shape[1], endpoint=False)
       
       
           if x_values is not None:
               ax.set_xticks([])
           if labels is not None:
               num_labels = len(labels)
               num_data_points = data.shape[1]
               x_indices = np.linspace(0, num_data_points - 1, num_labels, dtype=int)
               ax.set_xticks(x_indices)
               ax.set_xticklabels(labels)
       
           # Remove y-axis ticks and hide top and right spines
           ax.set_yticks([])
           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
           ax.set_ylabel(raster_label)
    
       
           # Remove top plot title
           ax.set_title(title)
           ax_bottom.set_title('')

       else:
           ax_bottom = plt.subplot(gs[axes_indexes[0]])
           ax_bottom.set_title(title)
           

   
       # Plotting the mean along the y-axis with shaded standard deviation or standard error of the mean
       
       mean_data = np.nanmean(data, axis=0)
       if use_std:
           error_data = np.std(data, axis=0)
       else:
           error_data = np.std(data, axis=0) / np.sqrt(data.shape[0])
       img= ax_bottom.plot(x_values, mean_data, color='blue', label='Mean')
       ax_bottom.fill_between(x_values, mean_data - error_data, mean_data + error_data, color='lightblue', alpha=0.5, label='Standard Deviation' if use_std else 'Standard Error of the Mean')
       ax_bottom.margins(x=0)  # Remove margins on x-axis
       ax_bottom.set_ylim(ylim)
       ax_bottom.set_ylabel('ΔF/F (%)')
       ax_bottom.set_xlabel('Time (s)')
       if log:
           ax_bottom.set_yscale('log')
       for i in range(data.shape[0]):
           ax_bottom.plot(x_values, data[i], color='gray', alpha=0.5)
       
       x_left = 0  # Left x-value
       x_right = 1  # Right x-value
       y_bottom = ax_bottom.get_ylim()[0]  # Bottom y-value
       y_top =  ax_bottom.get_ylim()[1]  # Top y-value
        
       # Create the rectangle patch
       rect = patches.Rectangle((x_left, y_bottom),  # (x, y) of lower-left corner
                                 x_right - x_left,     # Width
                                 y_top - y_bottom,     # Height
                                 linewidth=1,          # Line width
                                 edgecolor='r',        # Edge color
                                 facecolor='r',        # Fill color
                                 alpha=0.3)            # Transparency
        
       # Add the rectangle patch to the plot
       ax_bottom.add_patch(rect)
           
       # Set x-axis limits for bottom plot to match top plot
       # ax_bottom.set_xlim(ax.get_xlim())
   
       # Add a line at x value of 0 in the bottom plot
       vl=ax_bottom.axvline(x=0, color='black', linestyle='-')
       if not noraster:
           x_fig, y_fig= ax.transData.transform((vl.get_data()[0][0], 0))
           x_fig, y_fig= ax.transData.transform((vl.get_data()[0][0], 0))
       
       
           # Adjust subplot layout to make top and bottom plots very close together
       
           # Make vertical line span the top plot also
           if line_index is not None:
               ax.axvline(x=line_index, color='r', linestyle='--')
           #     # ax_bottom.axvline(x=line_index, color='r', linestyle='--')
       
      
   
       return img

    if isinstance(data,list):
        subfigure_nr=len(data)
        height_ratios=[2,1,2,1]
    else:
        subfigure_nr=1
        height_ratios=[2,1]
        data=[data]
        
    seq1=[0 ,2 ,4 ,0 ,4]
    seq2=[0 ,1 ,1 ,0 ,2]
    nrows=seq1[subfigure_nr]
    ncolumns=seq2[subfigure_nr]
    timestr=time.strftime("%Y%m%d-%H%M%S")
    fig_title=fig_title+f'_{timestr}.pdf'
        
    
    if noraster:
        nrows=2
        ncolumns=2
        height_ratios=[1,1]

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(nrows, ncolumns, height_ratios=height_ratios)
    
    axes_indexes=[0 ,1]
    for i in range(len(data)):
        
        if i==0:
            if len(data)<=2:
                axes_indexes=[0 ,1]
            elif len(data)>2:
                axes_indexes=[0 ,2]
            if compare=='opto':
                title='Opto_Spontaneous'
            elif compare=='grating':
                title='0° Grating'
            elif compare=='visstim':
                title='Vis_Stim'
            if noraster:
                axes_indexes=[0]


        elif i==1:
            if len(data)==2:
                axes_indexes=[2, 3]
            elif len(data)==4:
                axes_indexes=[4 ,6]
            if compare=='opto':

                title='Control_Spontaneous'
            elif compare=='grating':
                title='45° Grating'
            elif compare=='visstim':
                title='Spontaneous'
            if noraster:
                axes_indexes=[2]

        elif i==2:
            axes_indexes=[1,3]
            
            if compare=='opto':

               title='Opto_Grating'
            elif compare=='grating':
                title='90° Grating'
            if noraster:
                axes_indexes=[1]
            
        elif i==3:
            axes_indexes=[5 ,7]
            
            if compare=='opto':

                title='Control_Grating'
            elif compare=='grating':
                title='135° Grating'
            if noraster:
                axes_indexes=[3]
    
        img=plot_trials_mean_1data(gs,axes_indexes, data[i],title, x_values, labels, line_index, use_std, cmap, interpolation, aspect, origin, vmin, vmax,ylim, alpha, noraster)
     
    if not noraster:    
        plt.subplots_adjust(hspace=0.1,wspace=0.3)
        # Add color bar
        # cax = fig.add_axes([0.995, 0.15, 0.02, 0.6])
        # plt.colorbar(img[0], cax=cax)
    plt.savefig(temppath /  Path(fig_title), dpi=300, bbox_inches='tight')
    if show:
        plt.show()


def process_dataset(experiment,aq_analysis,aq_all_info,scale=True):

    smoothing_window=10
    scale=scale
    nr_samples=1000
    
    
    sorted_df=aq_all_info['stim_table']
    res=aq_analysis.caiman_results[list(aq_analysis.caiman_results.keys())[0]]
    angles=np.linspace(0,360,9)[:-1].astype('int')
    opto_grating=int(list(set(sorted_df[(sorted_df['opto']==1) & (sorted_df['blank_sweep']==0)]['orientation'].values))[0])
    # angles=np.delete(angles,np.argwhere(angles==opto_grating)[0])
    
    trial_dict={'opto_blank':sorted_df[(sorted_df['opto']==1) & (sorted_df['blank_sweep']==1)],
                'opto_grating':sorted_df[(sorted_df['opto']==1) & (sorted_df['blank_sweep']==0)],
                'control_blank':sorted_df[(sorted_df['opto']==0) & (sorted_df['blank_sweep']==1)],
                'control_grating':sorted_df[(sorted_df['opto']==0) & (sorted_df['orientation']==opto_grating)],
                
                }
    
    for angle in angles:
        trial_dict[str(angle)]=sorted_df[(sorted_df['opto']==0) & (sorted_df['orientation']==angle)]
        
    chands=aq_all_info['optocellindex_dict']['chand']['opto']['All_planes']
    non_chands=aq_all_info['optocellindex_dict']['non_chand']['all']['All_planes']

    #% check traces integrity plane1
    # dfdt_raw_analysis=aq_analysis.full_data['imaging_data']['Plane1']['Traces']['dfdt_raw']
    # demixed_analysis=aq_analysis.full_data['imaging_data']['Plane1']['Traces']['demixed']
    # dfdt_raw_caiman_ob=res.data['proc']['deconv']['smooth_dfdt']['S'][res.accepted_indexes_sorter,:]
    # demixed_caiman_ob=(res.data['est']['C']+res.data['est']['YrA'])[res.accepted_indexes_sorter]
    # fr=aq_analysis.full_data['imaging_data']['Frame_rate']
    # fr_caiman=res.data['ops']['init_params_caiman']['data']['fr']
    
    # check_backgoundtrace=res.data['est']['f']
    
    # all(dfdt_raw_analysis==dfdt_raw_caiman_ob)
    # all(demixed_analysis==demixed_caiman_ob)
    # all(fr==fr_caiman)
    
    
    #% organize full data
    demixed=aq_analysis.full_data['imaging_data']['All_planes_rough']['Traces']['demixed']
    denoised=aq_analysis.full_data['imaging_data']['All_planes_rough']['Traces']['denoised']
    dfdt_raw=aq_analysis.full_data['imaging_data']['All_planes_rough']['Traces']['dfdt_raw']
    dfdt_smoothed=aq_analysis.full_data['imaging_data']['All_planes_rough']['Traces']['dfdt_smoothed']
    # dfdt raw corresponds to to recify smoothed dfdt(i have replicated in pyhton)
    # dfdt smoothed correspond to thresholded smoothd dfdt don in caiman results object with 2std
    # all_traces=[demixed[cell,:],denoised[cell,:],dfdt_raw[cell,:],dfdt_smoothed[cell,:]]
    def compute_locomotion_bouts(analysis):
        
        
        def convert_spped_to_units(analysis):
            raw_locomotion=analysis.full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['Locomotion'].values
            timestamps=analysis.full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['Time'].values
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
            width_ms = 1000  # 50 ms
            
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
            # ax[4].plot(loc_second_time,rolling_sums)
            # ax[4].plot(x,gaussian50ms)
        
        
            return (loc_second_time,np.array(rolling_sums))
        speed_timestamps,cm_s=convert_spped_to_units(analysis)
        
        
        upsampled_time_milliseconds_stamps=analysis.full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['Time'].values/1000
        upsampled_signal = np.interp(upsampled_time_milliseconds_stamps, speed_timestamps, cm_s)
        
        
        
        # f,ax=plt.subplots(1)
        # ax.plot(speed_timestamps, cm_s, label='Original (1 sample/second)')
        # ax.plot(upsampled_time_milliseconds, upsampled_signal, label='Upsampled (1000 samples/second)')
        # ax.set_xlabel('Time (seconds)')
        # ax.set_ylabel('Amplitude')
        # ax.legend()
        
        peaks=find_peaks(upsampled_signal)[0]
        firstpeaks=np.diff(peaks,prepend=-10000)>10000
        laststpeaks=np.roll(firstpeaks,-1)
        
        boutstarts=peaks[np.diff(peaks,prepend=-10000)>10000]-30
        boutsends=peaks[laststpeaks]+30
        
        clippedvoltagetime=analysis.full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['Time']
        ledclippedlocomotion=analysis.full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['Locomotion'].values
        # f,ax=plt.subplots(1)
        # # ax.plot(upsampled_signal)
        # # ax.plot(ledclippedlocomotion)
        
        # ax.plot(clippedvoltagetime,ledclippedlocomotion)
        # ax.plot(clippedvoltagetime,upsampled_signal)
        
        # ax.plot(clippedvoltagetime[boutstarts],upsampled_signal[boutstarts],'ro')
        # ax.plot(clippedvoltagetime[boutsends],upsampled_signal[boutsends],'bo')
        
        
        locomotion_on_offsets=[]
        for bout in range(len(boutstarts)):
            locomotion_on_offsets.append((np.abs( np.array(np.array(analysis.all_planes_timestamps['Plane1'])*1000 - boutstarts[bout])).argmin(),np.abs( np.array(np.array(analysis.all_planes_timestamps['Plane1'])*1000 - boutsends[bout])).argmin()))
        
        
        for j,i in enumerate(locomotion_on_offsets):
            locomotion_on_offsets[j]=(i[0],i[1],i[1]-i[0])
            
        return locomotion_on_offsets
        
    
    def compute_all_cells_info(full_array,trial_dict,sorted_df,pre_time,post_time,pre_frames,post_frames,smoothing_window=10,scale=True,nr_samples=10000,locomotion_on_offsets=None):
        
        
        full_trial_sliced_activity=[]
        full_trial_sliced_activity_dff=[]
        full_trials_statistics=[]
        sliced_time_vector=np.linspace(-pre_time/1000,post_time/1000,pre_frames+post_frames)
        full_all_trials=np.zeros(([full_array.shape[0]]+ [sorted_df.shape[0] ,sliced_time_vector.shape[0] ]))
        all_samples=np.zeros(([full_array.shape[0]]+[nr_samples,40]))
        full_all_peaks=[]
        full_thr=np.zeros(full_array.shape[0]) 
        
        for cell in range(full_array.shape[0]):
            trace_sel=full_array[cell,:]
    
            # % full trace
            scaled_trace = (trace_sel-np.min(trace_sel))/(np.max(trace_sel)-np.min(trace_sel)) # scale trace_sel to 01
            framenumber = len(trace_sel)
            frac = smoothing_window/framenumber
            filtered = lowess(trace_sel, np.arange(framenumber), frac=frac)[:,1]   # apply filter
            scaled_filtered=lowess(scaled_trace, np.arange(framenumber), frac=frac)[:,1]  
            
            #% trace selection
            if smoothing_window and scale:
                trace_use=scaled_filtered
            elif smoothing_window:
                trace_use=filtered
            elif scale:
                trace_use=scaled_trace
            else:
                trace_use=trace_sel
    
                
            trial_sliced_activity = {}
            trial_sliced_activity['sliced_time_vector']=  sliced_time_vector

            #% trial slicing
            for k,trials,in trial_dict.items():
                if k=='opto_blank' and trials.shape[0]==0:
                    sliced_activity=np.ones(( 40,pre_frames+post_frames))
            
                else:  
                    offsetjitter=[]
                    sliced_activity=np.zeros((trials.shape[0],pre_frames+post_frames))
                    for trial_nr in range(trials.shape[0]):
                        trial_onset=trials['start'].iloc[trial_nr]
                        tial_offset=trials['end'].iloc[trial_nr]
                        sliced_activity[trial_nr,:]=trace_use[trial_onset-pre_frames:trial_onset+post_frames]
            
                trial_sliced_activity[k]=sliced_activity      
                
            #%locomotion slicing
            locomotion_postframes=300 #frames at movie sampling rate
            locomotion_preframes=25 #ms
            locomotion_trials=np.zeros((len(locomotion_on_offsets),locomotion_preframes+locomotion_postframes))
            for j,bout in enumerate(locomotion_on_offsets):
                if bout[0]+locomotion_postframes<len(trace_use):
                    locomotion_trials[j,:]=trace_use[bout[0]-locomotion_preframes:bout[0]+locomotion_postframes]
                else:
                    locomotion_trials[j,:]=trace_use[bout[0]-locomotion_preframes:]
            
            trial_sliced_activity['locomotion']=locomotion_trials
            trial_sliced_activity['sliced_time_vector_locomotion']= np.linspace(-(locomotion_preframes*pre_time/pre_frames)/1000,(locomotion_postframes*pre_time/pre_frames)/1000,locomotion_preframes+locomotion_postframes)
    
       
    
    
            #% substracting the baseline
            half_stim_frames=int(np.ceil(post_frames/2))
            half_stim_frames=0 
    
            trial_sliced_activity_dff={}
            for k,treatment in trial_sliced_activity.items():
                
                
                if 'time' not in k:
                    trials_dff=np.zeros(treatment.shape)
                    for trial in range(treatment.shape[0]):
                        if 'locomotion' in k:
                            
                            pre_frames_use=locomotion_preframes
                        else:
                            pre_frames_use=pre_frames
                            
                    
                        ontrilatrace=treatment[trial,:]
                        preonset_trace=ontrilatrace[:pre_frames_use]
                        postonset_trace=ontrilatrace[pre_frames_use+half_stim_frames:]
                        #allen substraction
                        # dff=100 * ((ontrilatrace / np.nanmean(preonset_trace)) - 1)
                        #mine is the same basically
                        dff=100*(ontrilatrace- preonset_trace.mean())/preonset_trace.mean()
                        dff=ontrilatrace- preonset_trace.mean()

                        trials_dff[trial,:]=dff
                    trial_sliced_activity_dff[k]=trials_dff
            
            
            #% statistics ANOVA
            all_treatments_trials_statistics={}
            for k,v in trial_sliced_activity_dff.items():
            
                trials_dff=v
                trials_statistics={'peak':{'max':{'full':np.zeros(trials_dff.shape[0]),
                                                  'opto':np.zeros(trials_dff.shape[0])},
                                           'mean':{'full':np.zeros(trials_dff.shape[0]),
                                                   'opto':np.zeros(trials_dff.shape[0])}
                                           },
                                   'p_value':{'full':np.zeros(trials_dff.shape[0]),
                                              'opto':np.zeros(trials_dff.shape[0]),
                                              'post_opto':np.zeros(trials_dff.shape[0])},
                                   }
                
                for trial in range(trials_dff.shape[0]):
                    if 'locomotion' in k:
                        
                        pre_frames_use=locomotion_preframes
                        post_frames_use=locomotion_postframes

                    else:
                        pre_frames_use=pre_frames
                        post_frames_use=post_frames

                    
                    
                    dff= trials_dff[trial]
                    trials_statistics['peak']['mean']['full'][trial]=np.nanmean(dff[pre_frames_use:pre_frames_use+post_frames_use])
                    trials_statistics['peak']['mean']['opto'][trial]=np.nanmean(dff[pre_frames_use:pre_frames_use+int(np.floor(post_frames_use/2))])
                    trials_statistics['peak']['max']['full'][trial]=np.max(dff[pre_frames_use:pre_frames_use+post_frames_use])
                    trials_statistics['peak']['max']['opto'][trial]=np.max(dff[pre_frames_use:pre_frames_use+int(np.floor(post_frames_use/2))])
                    
                    (_,trials_statistics['p_value']['full'][trial])=st.f_oneway(dff[:pre_frames_use],dff[pre_frames_use: pre_frames_use + post_frames_use])
                    (_,trials_statistics['p_value']['opto'][trial])=st.f_oneway(dff[:pre_frames_use],dff[pre_frames_use: pre_frames_use + int(np.floor(post_frames_use/2))])
                    (_,trials_statistics['p_value']['post_opto'][trial])=st.f_oneway(dff[:pre_frames_use],dff[pre_frames_use+int(np.floor(post_frames_use/2)): pre_frames_use+post_frames_use])
                    # plt.plot(dff)
                    
                all_treatments_trials_statistics[k]=trials_statistics

                
            # print(f"number significan trials: {trials_statistics['p_value']['full'][trials_statistics['p_value']['full']<0.05].shape[0]}")
               
            # statistics bootstrpaing
            onlyact={k:v for k,v in list(trial_sliced_activity.items()) if k not in ['sliced_time_vector','control_grating','locomotion','sliced_time_vector_locomotion']}
            all_trials=np.concatenate(list(onlyact.values()),0)
            
            # vmax=1
            # f,axs=plt.subplots(1,12)
            # axs[0].imshow(all_trials,vmax=vmax)
            # for i,k in enumerate(onlyact.keys()):
            #     axs[i+1].imshow( onlyact[k],vmax=vmax)
                
            # f,axs=plt.subplots(12,sharex=True,sharey=True)
            # axs[0].plot(x_values,all_trials.mean(axis=0))
            # for i,k in enumerate(onlyact.keys()):
            #     axs[i+1].plot( x_values,onlyact[k].mean(axis=0))
                    
            
            
            # grating_peaks=[v.mean(axis=0).max() for k,v in trial_sliced_activity.items() if k in [str(i) for i in angles]]
            # f,axs=plt.subplots(1)
            # axs.plot(angles,peaks)
            
            # trialmean=trial_sliced_activity['opto_blank'].mean(axis=0)
            # trialsem=trial_sliced_activity['opto_blank'].std(axis=0)/np.sqrt(trial_sliced_activity['opto_blank'].shape[0]-1)
            
            
            total_trials=all_trials.shape[0]
            total_population = np.arange(total_trials)  # Example: numbers from 1 to 100
            
            # Define the number of elements to sample
            num_elements_to_sample = trial_sliced_activity['opto_blank'].shape[0] # Example: sample 10 elements
            
            # Randomly sample elements
            # f,ax=plt.subplots(1)
            samples=np.zeros((nr_samples,num_elements_to_sample),dtype='int')
            for i in range(nr_samples):
                samples[i,:] = np.sort(np.random.choice(total_population, size=num_elements_to_sample, replace=False))
                # ax.plot(all_trials[samples[i,:]].mean(axis=0))
            
            
            
            
            sampled_elements = np.random.choice(total_population, size=num_elements_to_sample * nr_samples)
            sampled_trials=np.reshape(all_trials[sampled_elements,:],[nr_samples,num_elements_to_sample,all_trials[sampled_elements,:].shape[1]]).mean(axis=1)
            
            
            
            all_peaks=[]
            for treatment in [trial_sliced_activity['opto_blank'],sampled_trials]:
                peak_vals=np.zeros([treatment.shape[0],2])
                for i in range(treatment.shape[0]):
                    num_t=40
                    peak_size=4
                    peak_pad_left = np.floor((peak_size - 1)/2)
                    peak_pad_right = np.ceil((peak_size - 1)/2)
                    peak_loc = np.argmax(treatment[i,:])
                    pk_l = peak_loc - peak_pad_left
                    pk_r = peak_loc+ peak_pad_right
                    
                    if pk_l < 1:
                           pk_l2 = pk_l + 1 - pk_l
                           pk_r2 = pk_r + 1 - pk_l
                    elif pk_r > num_t:
                        pk_l2 = pk_l + num_t - pk_r
                        pk_r2 = pk_r + num_t - pk_r
                    else:
                        pk_l2 = pk_l
                        pk_r2 = pk_r
                        
                    peak_val= np.nanmean(treatment[i,int(pk_l2):int(pk_r2)])
                    peak_vals[i,0]=peak_val
                    peak_vals[i,1]=peak_loc
                all_peaks.append(peak_vals)
                
                
            # f,ax=plt.subplots()
            # ax.hist(all_peaks[0][:,0],density=True)
            # ax.hist(all_peaks[1][:,0],density=True)
            
            p_val=5
            peak_thr=np.percentile(all_peaks[1][:,0],100-p_val)
            
            # hist, bins = np.histogram(all_peaks[1][:,0], bins=30, density=True)
            # ecdf = np.cumsum(hist * np.diff(bins))
            # f,ax=plt.subplots()
            # ax.hist(all_peaks[1][:,0], bins=bins, density=True, alpha=0.5, color='blue')
            # ax.plot(bins[:-1], ecdf, color='red', linestyle='-', marker='o')
            
            # given_value = all_peaks[0][:,0].mean()
            # idx = np.argmin(np.abs(bins[-1:] - given_value))
            # # Calculate the p-value for the given value
            # p_value = ecdf[idx]
          
            full_trial_sliced_activity.append(trial_sliced_activity)
            full_trial_sliced_activity_dff.append(trial_sliced_activity_dff)
            full_trials_statistics.append(all_treatments_trials_statistics)
            full_all_trials[cell,:,:]=all_trials
            full_all_peaks.append(all_peaks)
            full_thr[cell]=peak_thr
            all_samples[cell,:]=samples
            
            
            # f,ax=plt.subplots(1)
            # ax.plot(scaled_filtered)
            
            # trial_sliced_activity['opto_blank'].mean(axis=0)
            # labels=np.arange(-pre_time/1000,post_time/1000+0.5,0.5)
            # line_index=pre_frames-1
            # x_values=trial_sliced_activity['sliced_time_vector']
            
            # data=[trial_sliced_activity['opto_blank'],trial_sliced_activity['control_blank'],trial_sliced_activity['opto_grating'],trial_sliced_activity['control_grating']]
            # plot_image_with_mean_and_std(data, x_values, use_std=False, cmap='inferno', interpolation='nearest', aspect='auto', origin='upper', vmin=min([i.min() for i in data]),vmax=max([i.max() for i in data])*.8, ylim=[0.9*min([i.mean(axis=0).min() for i in data]),1.2*max([i.mean(axis=0).max() for i in data])], alpha=None, fig_size=(8, 6))         
            # data=[trial_sliced_activity_dff['opto_blank'],trial_sliced_activity_dff['control_blank'],trial_sliced_activity_dff['opto_grating'],trial_sliced_activity_dff['control_grating']]
            # plot_image_with_mean_and_std(data, x_values, use_std=False, cmap='inferno', interpolation='nearest', aspect='auto', origin='upper', vmin=min([i.min() for i in data]),vmax=max([i.max() for i in data])*.8, ylim=[0.9*min([i.mean(axis=0).min() for i in data]),1.2*max([i.mean(axis=0).max() for i in data])], alpha=None, fig_size=(8, 6))
            
    
        return   full_trial_sliced_activity, full_trial_sliced_activity_dff, full_trials_statistics, full_all_trials, full_all_peaks, full_thr
    
    locomotion_on_offsets=compute_locomotion_bouts(aq_analysis)
    full_info=compute_all_cells_info(demixed,trial_dict,aq_all_info['stim_table'],aq_all_info['pre_time_df'],aq_all_info['post_time_df'],aq_all_info['pre_frames_df'],aq_all_info['post_frames_df'],smoothing_window=smoothing_window,scale=scale,nr_samples=nr_samples,locomotion_on_offsets=locomotion_on_offsets)
    
    return experiment, aq_analysis, aq_all_info, opto_grating, chands, non_chands, demixed, trial_dict, smoothing_window, scale, nr_samples, full_info         

def save_temp_data(data_dict,datapath):
    if not os.path.isfile(datapath):
        with open(datapath, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
    return datapath

def load_temp_data(temp_data_list,dataindex):
    multiple_analysis={}
    if temp_data_list:
        selected_temp_data_path=temp_data_list[dataindex]
        with open( selected_temp_data_path, 'rb') as file:
            multiple_analysis= pickle.load(file)
    return multiple_analysis

def check_temp_data(tempprocessingpat,experimentalmousename) :
    temp_data_list=[]
    temp_data_list=glob.glob(str(tempprocessingpat)+os.sep+f'**{experimentalmousename}**', recursive=False)
    
    return temp_data_list


#%%OBJECTS FORM OTHER SCCRIPS

all_analysis # from to do latesoptoanalaysis it the analysis onbjects of all datasets


#%% load data form optoanalysis script


i=0
single_experiment=sorted(list(multiple_analysis.keys()))[i]
temp_data_list=check_temp_data(str(tempprocessingpat),single_experiment)
# dict_only_all_data=load_temp_data(temp_data_list,3)
dict_only_all_data=load_temp_data(temp_data_list,3)


if platform == "linux" or platform == "linux2":
    aq_analysis=None
elif platform == "win32":
    aq_analysis=[all_analysis[i]['analysis'] for i in range(len(all_analysis)) if all_analysis[i]['analysis'].acquisition_object.aquisition_name==single_experiment][0]
aq_all_info=multiple_analysis[single_experiment]


all_exp_data={}
if dict_only_all_data:
    for i,(k, experiment) in enumerate(multiple_analysis.items()):
        if platform == "linux" or platform == "linux2":
            aq_analysis=None
        elif platform == "win32":

            aq_analysis=[all_analysis[j]['analysis'] for j in range(len(all_analysis)) if all_analysis[j]['analysis'].acquisition_object.aquisition_name==k][0]
        aq_all_info=experiment
        all_exp_data[k]=[experiment,aq_analysis,aq_all_info]+list(dict_only_all_data[i])
    
if not all_exp_data:
    for i,(k, experiment) in enumerate(multiple_analysis.items()):
        
        aq_analysis=[all_analysis[j]['analysis'] for j in range(len(all_analysis)) if all_analysis[j]['analysis'].acquisition_object.aquisition_name==k][0]
        aq_all_info=experiment
        all_exp_data[k]=process_dataset(experiment,aq_analysis,aq_all_info,scale=True)
        
    
    dict_only_all_data=[v[2:] for v in  all_exp_data.values()]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_temp_data(dict_only_all_data,tempprocessingpat / Path(single_experiment+timestr))

#%% analyzing chandelier responses multiple datasets

all_datasets={}
all_mean_peaks={}
all_peak_thr={}
base_substrac=1
if base_substrac:
    dtsts=1
else:
    dtsts=0


for i,(l, experiment) in enumerate(all_exp_data.items()):
    all_treatments={}
    all_treatments_mean_peaks={}
    all_treatments_peak_thr={}

    #HERE SELECT THE   experiment[-1][1] i s baseline substrated
    #HERE SELECT THE   experiment[-1][0] i s just scaled 0 1

    
    for k,v in experiment[-1][1][dtsts].items():
        
        cell_number=len(experiment[-1][dtsts])
        fullarray=np.zeros(([cell_number]+list(v.shape)))
        fullarray_mean_peaks=np.zeros(([cell_number]))
        fullarray_peak_thr=np.zeros(([cell_number]))

        for cell in range(cell_number):
            fullarray[cell,:,:]=experiment[-1][dtsts][cell][k]          
            fullarray_mean_peaks[cell]=experiment[-1][2][cell][k]['peak']['mean']['opto'].mean()
            fullarray_peak_thr[cell]=experiment[-1][-1][cell]

        all_treatments_peak_thr[k]=fullarray_peak_thr
        all_treatments_mean_peaks[k]=fullarray_mean_peaks
        all_treatments[k]=fullarray
        
    all_peak_thr[l]=all_treatments_peak_thr
    all_mean_peaks[l]=all_treatments_mean_peaks
    all_datasets[l]=  all_treatments  
   
#%% SELECT CELL TYPE TO ANALYZE
cell_type=5

if cell_type==4:
    cell_type_name='chandelier'
elif cell_type==5:
    cell_type_name='non_chandelier'


cell_act={}
cell_act_full={}

cell_mean_peaks={}
cell_peak_thr={}


    
all_v=[]
for i in all_datasets.values():
    all_v=all_v+list(i.keys())
stim_set=sorted(set(all_v), key=all_v.index)

for l in stim_set:
    cell_act[l]=[]
    cell_act_full[l]=[]

    cell_mean_peaks[l]=[]
    cell_peak_thr[l]=[] 


    for k,v in all_datasets.items():
        cell_indexes=all_exp_data[k][cell_type]
        if l in v.keys():
            cell_act[l].append(v[l].mean(axis=1)[cell_indexes,:])
            cell_act_full[l].append(v[l][cell_indexes,:])
            cell_mean_peaks[l].append(all_mean_peaks[k][l][cell_indexes])
            cell_peak_thr[l].append(all_peak_thr[k][l][cell_indexes])

            
    cell_peak_thr[l]=np.concatenate(cell_peak_thr[l],axis=0)       
    cell_mean_peaks[l]=np.concatenate(cell_mean_peaks[l],axis=0)  
    if len( cell_act_full[l])>1:
        if cell_act_full[l][0].shape[1]!=cell_act_full[l][1].shape[1]:
           todo=np.argmin([cell_act_full[l][0].shape[1],cell_act_full[l][1].shape[1]])
           tododo=np.empty(cell_act_full[l][todo].shape)
           tododo[:]=np.nan
           cell_act_full[l][todo]=np.concatenate([cell_act_full[l][todo],tododo],axis=1)
    else:
           cell_act_full=cell_act_full    
    
    
    if l !='locomotion':
        cell_act_full[l]=np.concatenate(cell_act_full[l],axis=0)  
        cell_act[l]=np.concatenate(cell_act[l],axis=0)   
    else:
        pass
        # cell_act_full_loc[l]=np.concatenate(cell_act_full[l],axis=0)  
        # cell_act_loc[l]=np.concatenate(cell_act[l],axis=0)        
      
        



#%% sorting by orientation



# get the sorting orders
sorting_peaks=[np.flip(np.argsort(cell_mean_peaks['opto_blank'])),np.flip(np.argsort(cell_mean_peaks['control_blank'])),np.flip(np.argsort(cell_mean_peaks['opto_grating'])),np.flip(np.argsort(cell_mean_peaks['control_grating']))]
# gather the data
data=[cell_act['opto_blank'],cell_act['control_blank'],cell_act['opto_grating'],cell_act['control_grating']]
# split in top and bottom
top_chand_data=[cell_act_full['opto_blank'][sorting_peaks[0][0]],cell_act_full['control_blank'][sorting_peaks[0][0]],cell_act_full['opto_grating'][sorting_peaks[0][0]],cell_act_full['control_grating'][sorting_peaks[0][0]]]
bottom_chand_data=[cell_act_full['opto_blank'][sorting_peaks[0][-1]],cell_act_full['control_blank'][sorting_peaks[0][-1]],cell_act_full['opto_grating'][sorting_peaks[0][-1]],cell_act_full['control_grating'][sorting_peaks[0][-1]]]

#%% analyze based on orientationn responsivity
ori=[0,45,90,135]

mean_ori_grating=[np.nanmean(np.concatenate([cell_act_full[str(i)],cell_act_full[str(i+180)]],axis=1),axis=1) for i in ori ]
full_ori_grating=[np.concatenate([cell_act_full[str(i)],cell_act_full[str(i+180)]],axis=1) for i in ori ]
mean_ori_peaks=[np.nanmean([cell_mean_peaks[str(i)],cell_mean_peaks[str(i+180)]],axis=0) for i in ori ]

ori_top_chand_data=[j[sorting_peaks[0][0],:,:] for i,j in enumerate(full_ori_grating)]
ori_bottom_chand_data=[j[sorting_peaks[0][-1],:,:] for i,j in enumerate(full_ori_grating)]

# analyze based on global visual responsivity
all_vis_stim=np.stack(mean_ori_grating).mean(axis=0)
peaks=all_vis_stim[:,all_exp_data[k][2]['pre_frames_df']:].mean(axis=1)
sorting_visual_responsiveness=np.flip(np.argsort([peaks])[0])
#%% CROSs VALIDATION SORT
fulloptodata=[cell_act_full['opto_blank'],cell_act_full['control_blank'],cell_act_full['opto_grating'],cell_act_full['control_grating']]
sorting_trialls=0.5

n_cross=[ list(range(m.shape[1])) for m in fulloptodata]
[random.shuffle(l) for l in n_cross]
cross_trials=[j[:int(len(j)*sorting_trialls)] for j in n_cross]
data_trials=[j[int(len(j)*sorting_trialls):] for j in n_cross]


cross_data=[i[:,cross_trials[j],:] for j,i in enumerate(fulloptodata)]
trial_data=[i[:,data_trials[j],:] for j,i in enumerate(fulloptodata)]


cross_sorting_peaks_min=np.flip(np.argsort(np.min(np.nanmean(cross_data[0][:,:,13:],axis=1),axis=1)))
cross_sorting_peaks_max=np.flip(np.argsort(np.max(np.nanmean(cross_data[0][:,:,13:],axis=1),axis=1)))


opto_cross_sorted=[i[cross_sorting_peaks_max,:,:] for i in trial_data]
opto_cross_sorted_trial_avg=list(map(lambda x: np.nanmean(x,axis=(1)), opto_cross_sorted))


#%% find significance and sort
#select how to sort
opto_sorted=[i[sorting_peaks[0]] for i in data]
ori_data_sorted=[i[sorting_peaks[0]] for i in mean_ori_grating]
all_vis_stim_data_sorted=[all_vis_stim[sorting_peaks[0]], cell_act['control_blank'][sorting_peaks[0]]]
data_sorted_vis_res=[i[sorting_visual_responsiveness] for i in data]
all_vis_stim_data_sorted_vis_response=all_vis_stim[sorting_visual_responsiveness,:]

signifup=cell_mean_peaks['opto_grating'][np.argsort(cell_mean_peaks['opto_blank'])]>cell_peak_thr['opto_blank'][np.flip(np.argsort(cell_mean_peaks['opto_blank']))]
# data_sorted_thresholded=[i[sorting_peaks[0][signifup] ]for i in data]



#do the sorting

data_sorted=ori_data_sorted
extrem_threhold=np.zeros([len(data_sorted[0])]).astype('int')
extrem_threhold[:int(np.ceil(len(data_sorted[0])*0.15))]=1
extrem_threhold_index_up=extrem_threhold>0
sigup=np.flip(np.argsort(cell_mean_peaks['opto_blank']))[extrem_threhold_index_up]

extrem_threhold=np.zeros([len(data_sorted[0])]).astype('int')
extrem_threhold[-int(np.ceil(len(data_sorted[0])*0.05)):]=1
extrem_threhold_index_down=extrem_threhold>0
sigdown=np.argsort(cell_mean_peaks['opto_blank'])[extrem_threhold_index_down]

extrem_threhold_index=extrem_threhold>0
data_sorted_extremes=[i[sorting_peaks[0]][extrem_threhold_index] for i in data]
#%% zetapy test 
# trialdict=all_exp_data[k][7]
# vecTime=all_exp_data[k][1].full_data['imaging_data']['All_planes_rough']['Timestamps'][0]
# arrEventTimes=vecTime[trialdict['opto_blank']['start'].values]
# moddd=st.mode(np.ceil((trialdict['opto_blank']['end'].values-trialdict['opto_blank']['start'].values)/2).astype('int'))[0]

# arrEventTimesoff=vecTime[trialdict['opto_blank']['start'].values+moddd]
# allevventtimes=np.stack([arrEventTimes,arrEventTimesoff],axis=1)

# all_cells_p_value=[]
# for cell in range(all_exp_data[k][6].shape[0]):
#     vecValue=all_exp_data[k][6][cell,:]
#     (ppz,pzdict)=pz.zetatstest(vecTime,vecValue,allevventtimes)
#     all_cells_p_value.append(ppz)
    
    
    
# sig_cells=np.array(all_cells_p_value)<0.001

# baseline=all_exp_data[k][6][~sig_cells,trialdict['opto_blank']['start'].values[0]-13:trialdict['opto_blank']['start'].values[0]].mean(axis=1)
# trial1=all_exp_data[k][6][~sig_cells,trialdict['opto_blank']['start'].values[0]-13:trialdict['opto_blank']['start'].values[0]+25]
# trial1dff=(all_exp_data[k][6][~sig_cells,trialdict['opto_blank']['start'].values[0]-13:trialdict['opto_blank']['start'].values[0]+25]/baseline[:, np.newaxis])-1

# plt.imshow(trial1,aspect='auto')
# f,ax=plt.subplots()
# for i in range(trial1.shape[0]):
#     ax.plot(trial1[i,:])


#%% MAIN PLOTTING SORTED BY OPTORESPONSES
# trialaveraged opto treatments
xwindow=all_exp_data[k][-1][0][0]['sliced_time_vector']
labels=np.arange(-aq_all_info['pre_time_df']/1000,aq_all_info['post_time_df']/1000+0.5,0.5)
data_sorted

timestr = time.strftime("%Y%m%d-%H%M%S")
#%%
# trialaveraged opto

plot_image_with_mean_and_std(opto_sorted, x_values=xwindow, labels=labels, use_std=True,
                             cmap='viridis', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-15, vmax=150,ylim=[-15, 150], alpha=None, fig_size=(20, 20),
                             fig_title=f'All_{cell_type_name}_trial_averaged_opto',raster_label='Cells',log=False)

# crossvalidatedtrialaveraged opto

plot_image_with_mean_and_std(opto_cross_sorted_trial_avg, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-15, vmax=150,ylim=[-15, 150], alpha=None, fig_size=(20, 20),
                             fig_title=f'All_{cell_type_name}_trial_averaged_crossvalidated_opto_maxsort_fulll_stim',raster_label='Cells',log=False)
# trialaveraged orientations

plot_image_with_mean_and_std(ori_data_sorted, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=80,ylim=[-40, 80], alpha=None, fig_size=(20, 20),
                             fig_title= f'All_{cell_type_name}_trial_averaged_oris',raster_label='Cells',compare='grating',log=False)

# trialaveraged vissitm

plot_image_with_mean_and_std(all_vis_stim_data_sorted, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-10, vmax=50,ylim=[-10, 50], alpha=None, fig_size=(20, 20),
                             fig_title=f'All_{cell_type_name}_trial_averaged_visstim',raster_label='Cells',compare='visstim',log=False)

# srted by visstim responsivenes
plot_image_with_mean_and_std(data_sorted_vis_res, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-30, vmax=80,ylim=[-30, 80], alpha=None, fig_size=(20, 20),
                             fig_title=f'All_{cell_type_name}_trial_averaged_opto_stim_sorted',raster_label='Cells',compare='opto',log=False)

plot_image_with_mean_and_std(all_vis_stim_data_sorted_vis_response, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-20, vmax=60,ylim=[-20, 60], alpha=None, fig_size=(20, 20),
                             fig_title=f'All_{cell_type_name}_trial_averaged_visstim_sorted',raster_label='Cells',compare='grating',log=False)


# most extreme cells

plot_image_with_mean_and_std(top_chand_data, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-50, vmax=100,ylim=[-50, 100], alpha=None, fig_size=(20, 20),
                             fig_title=f'Top_{cell_type_name}_trial_act',raster_label='Trials',log=False)

plot_image_with_mean_and_std(bottom_chand_data, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-50, vmax=100,ylim=[-50, 100], alpha=None, fig_size=(20, 20),
                             fig_title=f'Bottom_{cell_type_name}_trial_act',raster_label='Trials',log=False)

plot_image_with_mean_and_std(ori_top_chand_data, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=80,ylim=[-40, 80], alpha=None, fig_size=(20, 20),
                             fig_title=f'Top_{cell_type_name}_trial_act_ori',raster_label='Trials',compare='grating',log=False)

plot_image_with_mean_and_std(ori_bottom_chand_data, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=80,ylim=[-40, 80], alpha=None, fig_size=(20, 20),
                             fig_title=f'Bottom_{cell_type_name}_trial_act_ori',raster_label='Trials',compare='grating',log=False)


#%% cell bases tuning analysis
orientations=[0,45,90,135]
mean_ori_grating=[np.nanmean(np.concatenate([cell_act_full[str(i)],cell_act_full[str(i+180)]],axis=1),axis=1) for i in orientations ]
full_ori_grating=[np.concatenate([cell_act_full[str(i)],cell_act_full[str(i+180)]],axis=1) for i in orientations ]
mean_ori_peaks=[np.nanmean([cell_mean_peaks[str(i)],cell_mean_peaks[str(i+180)]],axis=0) for i in orientations ]

selected_or=0

if selected_or<90:
    ortho_or=selected_or+90
else:
    ortho_or=selected_or-90
        
sel_ori=orientations.index(selected_or)
sel_ortho_ori=orientations.index(ortho_or)

threshold=1
f,axs=plt.subplots(1,figsize=(20,20))
for i in range(4):
    axs.plot(mean_ori_peaks[i])
    
all_strongly_responsive_cells=[]
for i in range(4):
    
    mean_baseline=np.mean(full_ori_grating[i][:,:,4:12],axis=(1,2))
    mean_onesecond=np.mean(full_ori_grating[i][:,:,13:25],axis=(1,2))
    mean_fullstimd=np.mean(full_ori_grating[i][:,:,13:],axis=(1,2))
    
    std_baseline=np.std(full_ori_grating[i][:,:,4:12],axis=(1,2))
    std_onesecond=np.std(full_ori_grating[i][:,:,13:25],axis=(1,2))
    std_fullstimd=np.std(full_ori_grating[i][:,:,13:],axis=(1,2))

    trialevoked=mean_onesecond-np.abs(mean_baseline)>threshold*std_baseline
    
    gratingvscontrol=np.abs(np.mean(fulloptodata[3][:,:,13:25],axis=(1,2))/np.mean(fulloptodata[1][:,:,13:25],axis=(1,2)))>5
    strongly_responsive_cells=np.where(np.logical_and(trialevoked,gratingvscontrol))[0]


    
    sorted_responsive_cells=strongly_responsive_cells[np.argsort(mean_ori_peaks[i][strongly_responsive_cells])][::-1]

    all_strongly_responsive_cells.append(sorted_responsive_cells)


result = set(all_strongly_responsive_cells[0])
for s in all_strongly_responsive_cells[1:]:
    result.intersection_update(s)
result=list(result)    
#%% compare effects of chand in the more tresponsive cells
pre_frames=12
most_visually_evoked_cells=sorted(list(result))
non_visually_evoked_cells = [x for x in list(range(data[0].shape[0])) if x not in most_visually_evoked_cells] 
    

reposinvecellsdata=[i[most_visually_evoked_cells,:,:] for i in fulloptodata]
peakresponse_responsive=list(map(lambda x: np.nanmean(x,axis=1),reposinvecellsdata))

unreposinvecellsdata=[i[non_visually_evoked_cells,:,:] for i in fulloptodata]
peakresponse_unresponsive=list(map(lambda x: np.nanmean(x,axis=1),unreposinvecellsdata))


for cell in most_visually_evoked_cells:
    f,ax=plt.subplots()
    for i in [1,3]:
        ax.plot(np.mean(fulloptodata[i][cell,:,:],axis=0))
        plt.show()
        
        

plot_image_with_mean_and_std(peakresponse_unresponsive, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=80,ylim=[-40, 80], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_trial_averaged_visula_responses_nonsigcells',raster_label='Cells',compare='opto',log=False)

plot_image_with_mean_and_std(peakresponse_responsive, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=80,ylim=[-40, 80], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_trial_averaged_visula_responses_sigcells',raster_label='Cells',compare='opto',log=False)


plot_image_with_mean_and_std(peakresponse_unresponsive, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=80,ylim=[-40, 80], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_trial_averaged_visula_responses_nonsigcells',raster_label='Cells',compare='opto',log=False,noraster=True)

plot_image_with_mean_and_std(peakresponse_responsive, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=80,ylim=[-40, 80], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_trial_averaged_visula_responses_sigcells',raster_label='Cells',compare='opto',log=False,noraster=True)


comparisons_labels=[['Control_Spontaneous','Opto_Spontaneous'],['Control_Grating','Opto_Grating'],['Opto_Spontaneous','Opto_Grating'],['Control_Spontaneous','Control_Grating']]
for i in range(4):
    select_comparison=i
    comparisons=[[1,0],[3,2],[0,2],[1,3]]
    selected_comparison=comparisons[select_comparison]	
    # use_data=data_significant
    use_data=peakresponse_unresponsive
    
    treatment1=use_data[selected_comparison[0]][:,pre_frames:].mean(axis=1)
    treatment2=use_data[selected_comparison[1]][:,pre_frames:].mean(axis=1)
    treatment3=treatment2/treatment1
    
    df = pd.DataFrame({
        comparisons_labels[select_comparison][0]: treatment1,
        comparisons_labels[select_comparison][1]: treatment2,\
    })
    
    
    	
    # Melt the DataFrame to long format
    df_melted = df.melt(value_name='Value', var_name='Treatment')
    
    # Plot using Seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    
    # Boxplot with only vertical lines (no box)
    # sns.boxplot(data=df_melted, x='Treatment', y='Value', color='gray', width=0.3, showcaps=True, boxprops={'linewidth': 0})
    
    # Scatter plot with dispersion
    sns.stripplot(data=df_melted, x='Treatment', y='Value', color='black', size=5, jitter=False)
    # Join points with lines
    for i in range(len(treatment1)):
        plt.plot([0, 1], [treatment1[i], treatment2[i]], color='gray', linestyle='-',alpha=0.5, linewidth=0.5)
    
    # Set y-axis limit to 0
    plt.ylim([-20, 60])
    
    
    # Add plot title and labels
    plt.title('')
    plt.xlabel('')
    plt.ylabel('ΔF/F (%)')
    fig_title='Mean_Stim_Activity_non_chand_visstim_{timestr}.pdf'
    plt.savefig(fig_title, dpi=300, bbox_inches='tight')
    # Show plot
    plt.show()

        #%%
        
      
iondextoremove=[[all_strongly_responsive_cells[i].tolist().index(j) for j in result] for i in range(4)]
for i in range(4):
    all_strongly_responsive_cells[i]=np.delete(all_strongly_responsive_cells[i],iondextoremove[i])


cell=0
tt=list(map(lambda x: np.nanmean(x[cell,:,13:20]), full_ori_grating))
tttt=list(map(lambda x: x[cell,:,:], full_ori_grating))
ttttt=list(map(lambda x: np.nanmean(x[:,:,:],axis=1), full_ori_grating))
tttttt=list(map(lambda x: np.nanmean(x[:,:,13:20],axis=(1,2)), full_ori_grating))

f,ax=plt.subplots()
ax.plot(orientations,tt)
plt.show()

f,ax=plt.subplots()
for i in range(tttttt[0].size):
   ax.plot(orientations,list(map(lambda x: x[i],tttttt)))
plt.show()


ttt=list(map(lambda x: np.nanmean(x[cell,:,:],axis=0), full_ori_grating))
f,ax=plt.subplots()
for i in ttt:
    ax.plot(xwindow,i)
plt.show()

plot_image_with_mean_and_std(ttttt, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=80,ylim=[-40, 80], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_trial_averaged_visula_responses',raster_label='Cells',compare='grating',log=False)
plot_image_with_mean_and_std(tttt, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=80,ylim=[-40, 80], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_visual_responses_{selected_or}',raster_label='Trials',compare='grating',log=False)


tuned_cells_trials=full_ori_grating[sel_ori][strongly_responsive_cells,:,:]
tuned_cells_mean=mean_ori_grating[sel_ori][strongly_responsive_cells,:]
tuned_cells_mean_orto=mean_ori_grating[sel_ortho_ori][strongly_responsive_cells,:]
plot_image_with_mean_and_std(tuned_cells_mean, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=80,ylim=[-40, 80], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_ttuned_cells_prefered_ori_{selected_or}',raster_label='Cells',compare='grating',log=False)
plot_image_with_mean_and_std(tuned_cells_mean_orto, x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=80,ylim=[-40, 80], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_ttuned_cells_orthoori_{ortho_or}',raster_label='Cells',compare='grating',log=False)


# get tuned cells to a GIVEN angle and plot the responses toopto treatments
all_tuned_cells_opto_reponses=[[data[j][all_strongly_responsive_cells[i]] for j in range(4)] for i in range(4)]

plot_image_with_mean_and_std(all_tuned_cells_opto_reponses[0], x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=100,ylim=[-40, 100], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_opto_responses_of_tuned_{orientations[0]}_cells',raster_label='Cells', compare='opto',log=False, show=False)
plot_image_with_mean_and_std(all_tuned_cells_opto_reponses[1], x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=100,ylim=[-40, 100], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_opto_responses_of_tuned_{orientations[1]}_cells',raster_label='Cells', compare='opto',log=False, show=False)
plot_image_with_mean_and_std(all_tuned_cells_opto_reponses[2], x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=100,ylim=[-40, 100], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_opto_responses_of_tuned_{orientations[2]}_cells',raster_label='Cells', compare='opto',log=False, show=False)
plot_image_with_mean_and_std(all_tuned_cells_opto_reponses[3], x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=100,ylim=[-40, 100], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_opto_responses_of_tuned_{orientations[3]}_cells',raster_label='Cells', compare='opto',log=False, show=False)
                

all_tuned_cells_grat_reponses=[[mean_ori_grating[j][all_strongly_responsive_cells[i]] for j in range(4)] for i in range(4)]

plot_image_with_mean_and_std(all_tuned_cells_grat_reponses[0], x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=100,ylim=[-40, 100], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_grat_responses_of_tuned_{orientations[0]}_cells',raster_label='Cells', compare='grating',log=False, show=False)
plot_image_with_mean_and_std(all_tuned_cells_grat_reponses[1], x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=100,ylim=[-40, 100], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_grat_responses_of_tuned_{orientations[1]}_cells',raster_label='Cells', compare='grating',log=False, show=False)
plot_image_with_mean_and_std(all_tuned_cells_grat_reponses[2], x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=100,ylim=[-40, 100], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_grat_responses_of_tuned_{orientations[2]}_cells',raster_label='Cells', compare='grating',log=False, show=False)
plot_image_with_mean_and_std(all_tuned_cells_grat_reponses[3], x_values=xwindow, labels=labels, use_std=True,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-40, vmax=100,ylim=[-40, 100], alpha=None, fig_size=(20, 20),
                             fig_title=f'{cell_type_name}_grat_responses_of_tuned_{orientations[3]}_cells',raster_label='Cells', compare='grating',log=False, show=False)
                


cell=0

f,axt=plt.subplots(1,figsize=(20,20))
for cell in strongly_responsive_cells:
    tune_peaks=[]
    for ir in range(4):
        tune_peaks.append(mean_ori_peaks[ir][cell])
    axt.plot(tune_peaks)
axt.set_ylim(-10,50)

cell=117
f,ax=plt.subplots(1,figsize=(20,20))
for ir in range(4):
    ax.plot(mean_ori_grating[ir][cell])
ax.set_ylim(-10,50)

#%% full prfile for single cell
    # tuning curve
    # trial rasers for all four orientations
cell=3
for cell in range(len(full_ori_grating[0])):
    single_cell_ful_act=[full_ori_grating[orient][cell,:,:] for orient in range(4) ]
    
    plot_image_with_mean_and_std(single_cell_ful_act, x_values=xwindow, labels=labels, use_std=True,
                                 cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                                 vmin=-40, vmax=100,ylim=[-40, 100], alpha=None, fig_size=(20, 20),
                                 fig_title=f'{cell_type_name}_cell_{cell}_all_tirial_orient',raster_label='Trial',compare='grating',log=False, show=False)
    
    
prestim=sum(xwindow<0)

significant=np.zeros([len(full_ori_grating[0]),4])
for orient in range(4):
    for cell in range(len(full_ori_grating[0])):
        stdthreshold=np.mean(mean_ori_grating[0][cell,:prestim])+ np.std(mean_ori_grating[0][cell,:prestim])
        significant[cell,orient]= np.mean(mean_ori_grating[orient][cell,prestim:prestim+12])>stdthreshold

#%% getting data to averrage
# pre_frames=all_exp_data[k][2]['pre_frames_df']

comparisons_labels=[['Control_Spontaneous','Opto_Spontaneous'],['Control_Grating','Opto_Grating'],['Opto_Spontaneous','Opto_Grating'],['Control_Spontaneous','Control_Grating']]
select_comparison=3
comparisons=[[1,0],[3,2],[0,2],[1,3]]
selected_comparison=comparisons[select_comparison]	
# use_data=data_significant
use_data=data_sorted_extremes

treatment1=use_data[selected_comparison[0]][:,pre_frames:].mean(axis=1)
treatment2=use_data[selected_comparison[1]][:,pre_frames:].mean(axis=1)
df = pd.DataFrame({
    comparisons_labels[select_comparison][0]: treatment1,
    comparisons_labels[select_comparison][1]: treatment2,
})

comparisons_labels[select_comparison]
	
# Melt the DataFrame to long format
df_melted = df.melt(value_name='Value', var_name='Treatment')

# Plot using Seaborn
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))

# Boxplot with only vertical lines (no box)
sns.boxplot(data=df_melted, x='Treatment', y='Value', color='gray', width=0.3, showcaps=True, boxprops={'linewidth': 0})

# Scatter plot with dispersion
sns.stripplot(data=df_melted, x='Treatment', y='Value', color='black', size=5, jitter=True)
# Join points with lines
for i in range(len(treatment1)):
    plt.plot([0, 1], [treatment1[i], treatment2[i]], color='gray', linestyle='-',alpha=0.5, linewidth=0.5)

# Set y-axis limit to 0
plt.ylim([-20, 60])


# Add plot title and labels
plt.title('')
plt.xlabel('')
plt.ylabel('ΔF/F (%)')
fig_title='Mean_Stim_Activity_non_chand_visstim_{timestr}.pdf'
plt.savefig(fig_title, dpi=300, bbox_inches='tight')
# Show plot
plt.show()
#%%


palette = sns.color_palette('pastel')
violin=sns.violinplot(data=df, y='Data', hue='Labels', palette=palette, split=True, alpha=0.5)
sns.stripplot(data=df, y='Data', hue='Labels', jitter=True, dodge=True, palette=palette, edgecolor=None, alpha=1)
handles, labels = violin.get_legend_handles_labels()
legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
                   for label, color in zip(labels[:len(labels)//2], palette)] # Take only the first half of labels
plt.legend(handles=legend_elements, labels=labels[:len(labels)//2], title='Labels')
plt.xlabel(None)
plt.ylabel('Data')
plt.title('Split Violin Plot with Overlapping Strip Plot')
plt.show()


#%% add tzi zi data

adapted_all_sliced_array
adapted_all_sliced_current
adapted_all_sliced_array['imaging_time_vector']
adapted_all_sliced_current['current_time_vector']
pre_frames=preframes
clustered_opto=[]
clustered_control=[]
if 'opto_grating' in adapted_all_sliced_array.keys() :
    for i in adapted_all_sliced_array.keys() :
        if ('opto' in i) and ('blank' not in i):
            clustered_opto.append(adapted_all_sliced_array[i])
            
        if ('control' in i) and ('blank' not in i):
            clustered_control.append(adapted_all_sliced_array[i])
            
    adapted_all_sliced_array['opto_all']=np.concatenate(clustered_opto,axis=1)
    adapted_all_sliced_array['control_all']=np.concatenate(clustered_control,axis=1)
    
    clustered_opto=[]
    clustered_control=[]
    
    for i in adapted_all_sliced_current.keys() :
        if ('opto' in i) and ('blank' not in i):
            clustered_opto.append(adapted_all_sliced_current[i])
            
        if ('control' in i) and ('blank' not in i):
            clustered_control.append(adapted_all_sliced_current[i])
            
    adapted_all_sliced_current['opto_all']=np.vstack(clustered_opto)
    adapted_all_sliced_current['control_all']=np.vstack(clustered_control)



treatment='opto_blank'
trialaverageca=np.nanmean(adapted_all_sliced_array[treatment],axis=1)
trialaveragecu=np.nanmean(adapted_all_sliced_current[treatment],axis=0)


fig_title=f'current_trace_{treatment}_{timestr}.pdf'
f,ax=plt.subplots(4,figsize=(20,20), height_ratios=[3,1,1,1])
im=ax[0].imshow(trialaverageca[1:,:], aspect='auto',vmax=20)
ax[1].plot(adapted_all_sliced_array['imaging_time_vector'],np.nanmean(trialaverageca[1:,:],axis=0))
ax[2].plot(adapted_all_sliced_array['imaging_time_vector'],trialaverageca[0,:])
ax[3].plot(adapted_all_sliced_current['current_time_vector'],trialaveragecu)
# ax[1].set_ylim([-3, 2])
# ax[2].set_ylim([-3, 2])


cax = f.add_axes([0.995, 0.15, 0.02, 0.6])
plt.colorbar(im, cax=cax)
plt.savefig(fig_title, dpi=300, bbox_inches='tight')
plt.show()






# data=[adapted_all_sliced_array['opto_blank'][1:,:,:].mean(axis=1),adapted_all_sliced_array['control_blank'][1:,:,:].mean(axis=1),np.nanmean(adapted_all_sliced_array['opto_all'][1:,:,:],axis=1),adapted_all_sliced_array['control_all'][1:,:,:].mean(axis=1)]
data=[adapted_all_sliced_array['opto_blank'][1:,:,:].mean(axis=1),adapted_all_sliced_array['control_blank'][1:,:,:].mean(axis=1)]

peaks=[i[:,preframes:].mean(axis=1) for i in data]

sorting_peaks=[np.flip(np.argsort(i)) for i in peaks]

data_sorted=[i[sorting_peaks[0]] for i in data]
peaks_sorted=[i[:,preframes:].mean(axis=1) for i in data_sorted]


sig_peaks=[np.where(i>i.std()*2)[0] for i in peaks_sorted]


data_significant=[i[sig_peaks[0]] for i in data_sorted]

xwindow=adapted_all_sliced_array['imaging_time_vector']

plot_image_with_mean_and_std(data_sorted, x_values=xwindow, labels=None, use_std=False,
                             cmap='inferno', interpolation='nearest', aspect='auto', origin='upper',
                             vmin=-5, vmax=20, ylim=[-5, 20], alpha=None, fig_size=(20, 20),
                             fig_title=f'Tzi_nmotscaled',raster_label='Cells',log=False)



# ori_data_sorted=[i[sorting_peaks[0]] for i in mean_ori_grating]
# all_vis_stim_data_sorted=[all_vis_stim[sorting_peaks[0]], cell_act['control_blank'][sorting_peaks[0]]]

#%% ststistically activate dcells

    # get a measure of trial aberaged trace on the control data


#%%
# # %% gatherinmg dasta to df for plotting
# # %matplotlib qt


# trace_type_dfs=[]
# for j,trace in enumerate(all_traces):
#     scaled_trace = (trace-np.min(trace))/(np.max(trace)-np.min(trace))
#     imgingtimestmaps_ms=np.arange(0,fr*len(scaled_trace),fr)
#     imgingtimestmaps_s=imgingtimestmaps_ms/1000

    
#     window = 10
#     framenumber = len(trace)
#     frac = window/framenumber
#     filtered = lowess(trace, np.arange(framenumber), frac=frac)[:,1]  
    
#     scaled_filtered=lowess(scaled_trace, np.arange(framenumber), frac=frac)[:,1]  

#     datasets=[trace, scaled_trace, filtered,scaled_filtered]
#     processing_labels=['raw','scaled','filtered','scaled_filtered']
    
#     concatenated_data = np.concatenate(datasets)
    
#     full_processing_labels=[]
#     for i in  processing_labels:
#         x = [i] * len(datasets[0]) 
#         full_processing_labels=full_processing_labels+x
        
    
        
    
#     df = pd.DataFrame({'data': concatenated_data,
#                        'timestamps': np.tile(imgingtimestmaps_s, len(datasets)),
#                        'processing': full_processing_labels})
    
#     # %% plotting all traces processing
    
#     sns.relplot(data=df,kind="line",
#                 x="timestamps",y="data",
#                 row="processing",
#                 facet_kws=dict(sharex=True,sharey=False))
#     qt_fig = plt.gcf().canvas.manager.window
#     qt_fig.setGeometry(0, 0, 800, 600)
#     plt.show()
#     trace_type_dfs.append(df)
    

#%% LOCALIZED SORTED CELLS IN PLANES
sorted_by_opto_reponse=sorting_peaks[0]
data_sorted=ori_data_sorted
extrem_threhold=np.zeros([len(data_sorted[0])]).astype('int')
extrem_threhold[:int(np.ceil(len(data_sorted[0])*0.15))]=1
extrem_threhold_index_up=extrem_threhold>0
sigup=np.argsort(cell_mean_peaks['opto_blank'])[extrem_threhold_index_up]

extrem_threhold=np.zeros([len(data_sorted[0])]).astype('int')
extrem_threhold[-int(np.ceil(len(data_sorted[0])*0.05)):]=1
extrem_threhold_index_down=extrem_threhold>0
sigdown=np.argsort(cell_mean_peaks['opto_blank'])[extrem_threhold_index_down]

pyrdatset1=all_exp_data['240118_SPSM_FOV1_2z_ShortDrift_AllCell_617optoFulp_25x_920_51020_60745_with-000'][6]
pyrdatset2=all_exp_data['231105_SPRZ_FOV1_2z_ShortDrift6171p_25x_920_51020_63075_with-000'][6]
multiple_analysis['240118_SPSM_FOV1_2z_ShortDrift_AllCell_617optoFulp_25x_920_51020_60745_with-000']['optocellindex_dict']['non_chand']['all']['All_planes']
multiple_analysis['240118_SPSM_FOV1_2z_ShortDrift_AllCell_617optoFulp_25x_920_51020_60745_with-000']['optocellindex_dict']['non_chand']['all']['Plane1']
multiple_analysis['240118_SPSM_FOV1_2z_ShortDrift_AllCell_617optoFulp_25x_920_51020_60745_with-000']['optocellindex_dict']['non_chand']['all']['Plane2']
multiple_analysis['231105_SPRZ_FOV1_2z_ShortDrift6171p_25x_920_51020_63075_with-000']['optocellindex_dict']['non_chand']['all']['All_planes']
multiple_analysis['231105_SPRZ_FOV1_2z_ShortDrift6171p_25x_920_51020_63075_with-000']['optocellindex_dict']['non_chand']['all']['Plane1']
multiple_analysis['231105_SPRZ_FOV1_2z_ShortDrift6171p_25x_920_51020_63075_with-000']['optocellindex_dict']['non_chand']['all']['Plane2']


correspondence=list(zip(list(range(len(pyrdatset1)+len(pyrdatset2))),np.concatenate((pyrdatset1,pyrdatset2)).tolist()))


plane_corr_up=[]
for sig_cell in sigup:
    if correspondence[sig_cell][1]>len(pyrdatset1):
        dataset='240118_SPSM_FOV1_2z_ShortDrift_AllCell_617optoFulp_25x_920_51020_60745_with-000'
    else:
        dataset='231105_SPRZ_FOV1_2z_ShortDrift6171p_25x_920_51020_63075_with-000'
    multiple_analysis[dataset]['optocellindex_dict']['non_chand']['all']['All_planes']
    if correspondence[sig_cell][1] in multiple_analysis[dataset]['optocellindex_dict']['non_chand']['all']['Plane1']:
        plane='Plane1'
    else:
        plane='Plane2'
    plane_corr_up.append((dataset,plane))



plane_corr_down=[]
for sig_cell in sigdown:
    if correspondence[sig_cell][1]>len(pyrdatset1):
        dataset='240118_SPSM_FOV1_2z_ShortDrift_AllCell_617optoFulp_25x_920_51020_60745_with-000'
    else:
        dataset='231105_SPRZ_FOV1_2z_ShortDrift6171p_25x_920_51020_63075_with-000'
    multiple_analysis[dataset]['optocellindex_dict']['non_chand']['all']['All_planes']
    if correspondence[sig_cell][1] in multiple_analysis[dataset]['optocellindex_dict']['non_chand']['all']['Plane1']:
        plane='Plane1'
    else:
        plane='Plane2'
    plane_corr_down.append((dataset,plane))


string_counts_down = {}
string_counts_up = {}

# Count repetitions of each string
for string in plane_corr_down:
    if string[1] in string_counts_down:
        string_counts_down[string[1]] += 1
    else:
        string_counts_down[string[1]] = 1
        
for string in plane_corr_up:
    if string[1] in string_counts_up:
        string_counts_up[string[1]] += 1
    else:
        string_counts_up[string[1]] = 1
        
#%% analyze locomotion effects

loc1=cell_act_full['locomotion'][0][:,:10,:]
loc2=cell_act_full['locomotion'][1]
mean_loc= np.concatenate([loc1.mean(axis=1),loc2.mean(axis=1)])


f,ax = plt.subplots()
ax.imshow(mean_loc, aspect='auto')
ax.axvline(x=25, color='black', linestyle='-')

plt.savefig(temppath /  'allcellslocomotion.pdf', dpi=300, bbox_inches='tight')


visuallyevokedcells=mean_loc[most_visually_evoked_cells,:]
optoievokedcells=mean_loc[sigup,:]

f,ax = plt.subplots()
ax.imshow(visuallyevokedcells, aspect='auto')
ax.axvline(x=25, color='black', linestyle='-')

plt.savefig(temppath /  'visuoevokedloc.pdf', dpi=300, bbox_inches='tight')


f,ax = plt.subplots()
ax.imshow(optoievokedcells, aspect='auto')
ax.axvline(x=25, color='black', linestyle='-')
plt.savefig(temppath /  'optoevokedloc.pdf', dpi=300, bbox_inches='tight')

data=mean_loc
f,ax_bottom=plt.subplots()
mean_data = np.nanmean(data, axis=0)
error_data = np.std(data, axis=0)
img= ax_bottom.plot(mean_data, color='blue', label='Mean')
# ax_bottom.fill_between(mean_data - error_data, mean_data + error_data, color='lightblue', alpha=0.5, label='Standard Deviation')
ax_bottom.margins(x=0)  # Remove margins on x-axis
ax_bottom.set_ylabel('ΔF/F (%)')
ax_bottom.set_xlabel('Time (s)')
for i in range(data.shape[0]):
    ax_bottom.plot( data[i], color='gray', alpha=0.5)
# Set x-axis limits for bottom plot to match top plot
# ax_bottom.set_xlim(ax.get_xlim())

# Add a line at x value of 0 in the bottom plot
vl=ax_bottom.axvline(x=25, color='black', linestyle='-')