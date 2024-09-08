# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 14:00:43 2024

@author: sp3660
"""

#scipt for a;llen analysis new for SPJZ with all thre

"""
allen C:
    Nat Moviwe 1
    Nat movie 2FPAth
    
    
Allen B
    Nat Mov 1
Allen A:
    Nat mov 1
    Drifting Gratings
    Nat mov 3
        
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
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
import matplotlib 
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 

"""

"""
cell_equivalences=np.zeros([3,7]) 

cell_equivalences[0,:]=[5,2,3,6,4,7,1]
cell_equivalences[1,:]=[6,3,1,5,4,7,2]
cell_equivalences[2,:]=[6,4,7,5,1,2,3]
cell_equivalences=cell_equivalences-1
temppath=Path(r'C:\Users\sp3660\Desktop\TempPythonFigs')

#%% functions
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
#%% LOAD OPTIC FLOW
from PIL import Image
import matplotlib.pyplot as plt
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import rgb_to_hsv
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
# Open the TIFF file
tiff_path = r'C:\Users\sp3660\Desktop\ChandPaper\Fig2\Movie1opticflow.tif'  # Replace with your TIFF file path

def rgb_to_angle(rgb):
    """ Convert RGB to an angle on the color wheel. """
    # Normalize RGB values to range [0, 1]
    rgb_normalized = np.array(rgb) / 255.0
    
    # Convert RGB to HSV
    hsv = rgb_to_hsv(rgb_normalized)
    
    # Extract hue (angle) from HSV
    hue = hsv[0] * 360  # Hue ranges from 0 to 360 degrees
    angle_rad = np.deg2rad(hue)
    
    # Map angle using cosine to achieve the desired scaling
    # Cosine function: cos(angle) will be 0 at 0 and 180 degrees, 
    # +1 at 90 degrees, and -1 at 270 degrees
    scaled_value = np.sin(angle_rad)
    
    return scaled_value

def load_multipage_image_as_numpy(file_path):
    """ Load a multi-page image into a NumPy array. """
    with Image.open(file_path) as img:
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frame = img.convert('RGB')
            frame_array = np.array(frame)
            frames.append(frame_array)
        frames_array = np.stack(frames, axis=0)
    return frames_array

def plot_rgb_images(images_array):
    """ Plot RGB images from a NumPy array. """
    num_images = images_array.shape[0]
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    if num_images == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one image

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images_array[i])
        ax.axis('off')
        ax.set_title(f'Page {i + 1}')
    
    plt.tight_layout()
    plt.show()



with Image.open(tiff_path) as img:
    frame_number = 0
    avg_rgb_values = []  # To store average RGB values for each frame
    
    while True:
        try:
            # Convert to RGB if not already in RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert image to numpy array
            img_array = np.array(img)
            
            # Calculate the average RGB values for the current frame
            avg_rgb = np.mean(img_array, axis=(0, 1))  # Mean over width and height
            
            # Append average RGB values to the list
            avg_rgb_values.append(avg_rgb)
            
            # Move to the next frame
            img.seek(img.tell() + 1)
            frame_number += 1
        except EOFError:
            # End of the multipage TIFF file
            break

# Convert list of average RGB values to a NumPy array for plotting
avg_rgb_values = np.array(avg_rgb_values)

# Plot the average RGB values over time
plt.figure(figsize=(12, 6))
plt.plot(avg_rgb_values[:, 0], label='Average Red', color='r')
plt.plot(avg_rgb_values[:, 1], label='Average Green', color='g')
plt.plot(avg_rgb_values[:, 2], label='Average Blue', color='b')
plt.xlabel('Frame Number')
plt.ylabel('Average RGB Value')
plt.title('Average RGB Values Over Time')
plt.legend()
plt.show()


#%%
angl=[rgb_to_angle(frame) for frame in avg_rgb_values]
filt=gaussian_filter(angl, sigma=2)
plt.plot(gaussian_filter(angl, sigma=1))
intensity=loadmat(r'C:\Users\sp3660\Desktop\ChandPaper\Fig2\VideoMeanIntensity.mat')['intensity'].flatten()
filt=np.insert(filt,0,0)
f,ax=plt.subplots(2,sharex=True)
ax[0].plot(filt)
ax[1].plot(np.abs(np.diff(np.abs(angl))))


#%% FREQUENCY COPMPOSITION
mo=load_multipage_image_as_numpy(r'C:\Users\sp3660\Desktop\ChandPaper\Fig2\Movie1tiff.tif')
import cv2

gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in mo]

#%%

import numpy as np

def compute_frequency_spectrum(frame):
    dft = np.fft.fft2(frame)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.abs(dft_shift)
    return magnitude_spectrum

frequency_spectra = [compute_frequency_spectrum(frame) for frame in gray_frames]

def calculate_power_spectrum(magnitude_spectrum):
    power_spectrum = magnitude_spectrum**2
    return np.mean(power_spectrum)

power_spectra = [calculate_power_spectrum(spectrum) for spectrum in frequency_spectra]

from scipy.stats import entropy

def calculate_entropy(magnitude_spectrum):
    flat_spectrum = magnitude_spectrum.flatten()
    norm_spectrum = flat_spectrum / np.sum(flat_spectrum)
    return entropy(norm_spectrum)

entropies = [calculate_entropy(spectrum) for spectrum in frequency_spectra]

average_entropy = np.mean(entropies)
average_power_spectrum = np.mean(power_spectra)

import matplotlib.pyplot as plt

# plt.plot(entropies)
plt.plot(power_spectra)

plt.xlabel('Frame Index')
plt.ylabel('Entropy')
plt.title('Entropy of Each Frame')
plt.show()



import matplotlib.pyplot as plt

# Compute power spectra
def compute_power_spectrum(magnitude_spectrum):
    return np.log(np.sum(magnitude_spectrum, axis=0) + 1)  # Adding 1 to avoid log(0)

# Calculate the power spectrum for each frame
power_spectra = [compute_power_spectrum(spectrum) for spectrum in frequency_spectra]

# Stack power spectra vertically to create the spectrogram
spectrogram = np.stack(power_spectra, axis=0)

# Normalize the spectrogram for better visualization
spectrogram -= np.min(spectrogram)
spectrogram /= np.max(spectrogram)


# Increase contrast using linear stretching
def enhance_contrast(image, lower_percentile=10, upper_percentile=80):
    """Enhance the contrast of an image using percentile stretching."""
    lower, upper = np.percentile(image, (lower_percentile, upper_percentile))
    return np.clip((image - lower) / (upper - lower), 0, 1)

# Apply contrast enhancement
contrast_spectrogram = enhance_contrast(spectrogram)
# Plot the spectrogram
plt.imshow(contrast_spectrogram.T, aspect='auto', cmap='inferno')
plt.colorbar(label='Power')
plt.xlabel('Frame')
plt.ylabel('Frequency Bin')
plt.title('Spectrogram of Video')
plt.show()

#%% LOAD FROMM TO DO SCRIPT

# this has 3 element, one for each of allen and thenm it has the resulytsanslys iboject and the full data dictionary
all_analysis=selected_analysis

plt.close('all')
#%% first is check foe each dataset if the signals are ok, how does the calcium signal look align to the stimuls etc

# dataset, datset_name=[0, 'Session C']
dataset, datset_name=[1, 'Session B']
# dataset, datset_name=[2, 'Session A']


dtset_list=[[0, 'Session C'],[1, 'Session B'],[2, 'Session A']]


single_dataset=all_analysis[dataset]
analysis=single_dataset['analysis']
full_data=single_dataset['full_data']# ==analysis.full_data but I extract for easierr acces
print(analysis.caiman_results[next(iter(analysis.caiman_results))].mat_results_paths[0])
cell=0

#% organize the cells betwen datasets
#check same number of cells
i=dataset
print(all_analysis[i]['analysis'].caiman_results[next(iter(all_analysis[i]['analysis'].caiman_results))].mat_results_paths[0])
print(all_analysis[i]['full_data']['imaging_data']['Plane1']['CellNumber'])
print(all_analysis[i]['analysis'].volt_object.extraction_object.transitions_dictionary)

  



#% Check datase mouse 
full_data.keys()
imaging_data=full_data['imaging_data']
voltage_traces=full_data['voltage_traces']
visstim_info=full_data['visstim_info']

#% REVIEW IMAGING DATA
imaging_data.keys()
# confirm frame rate this might be worng if gettign from database, check tiemstamos
# as i suspected the timestZAmp frame rate is 35.4 while the databse is 28.6 so I have to see where is the correction in the database calculations
imaging_data['Frame_rate']
imaging_data['Frame_number']
imaging_data['Interplane_period']

#review timestamps SPJZ dfataset doesnt have LED flash so timestamps are not changed but set some functions to review
#LED clipping comes from the bidishift object as it is doine the first thing before any firther processing
# if there is LED then there is shifts, if not tye nshifts are 0 and the timesamps should all be the same
# for this first dataset ALlen C tthe l,ast frame is missing in the clipped, there ius an endclip in the code that aI added because it was giving some error
# the important is that it starts at 0 and is the same length as the data

movie_timestamps=imaging_data['mov_timestamps_seconds']['clipped']
timestamps_based_rate=np.mean(np.diff(movie_timestamps))*1000

caiman_extraction=analysis.caiman_results[next(iter(analysis.caiman_results))]
print(caiman_extraction.mat_results_paths[0])

#data
# review demixed vs denoiused clacium in a plt
# always used denosied data
frames=slice(None)

data=imaging_data['Plane1']
data.keys()
print(data['CellNumber'])
# plt.close('all')
# f,ax=plt.subplots(2)
# ax[0].plot(movie_timestamps,data['Traces']['demixed'][cell,frames])
# ax[1].plot(movie_timestamps,data['Traces']['denoised'][translate_selected_cell(cell, cell_equivalences, dataset)  ,frames])

#% VOLTAGE DATA
# get teh full trace and  timstamps and plot with a clacium trace
# this is usels as this is the resmapled, I ghad rto fogo to the extraction objects diurectly
voltage_traces.keys()

voltage_traces['VisStim']
voltage_info=analysis.volt_object.extraction_object
signals=voltage_info.all_signals['Prairie']
movie1_starts=voltage_info.movie_one_trial_full_recording


raw_voltage_timestamps=signals['Time'].to_numpy().flatten()/1000
raw_voltage_visstim=signals['VisStim'].to_numpy().flatten()
raw_speed=np.abs(np.diff(signals['Locomotion'].to_numpy().flatten(),prepend=0))

# these are the reasmple indexe to the video frame rate, review teh align well
resampled_movie1_starts=visstim_info['Movie1']['Trial_Starts'].astype('int')
resampled_par_transitions=visstim_info['Paradigm_Indexes']

#% PLOTING SIGNAL AND VIDEO ALIGNMENTS

paradigmstplot=[movie1_starts]
resampled_paradigmstplot=[resampled_movie1_starts]



trace=data['Traces']['denoised'][translate_selected_cell(cell, cell_equivalences, dataset)  ,frames]
# plt.close('all')
f,axs=plt.subplots(3,sharex=True)
f.suptitle(f'Review Video Voltage Alignment dataset {dataset} cell {cell}')
axs[0].plot(movie_timestamps,trace)
axs[1].plot(raw_voltage_timestamps,raw_voltage_visstim)
axs[2].plot(raw_voltage_timestamps,raw_speed)

for ax in axs:
    ax. margins(x=0)

    for sp in ax.spines :
        if sp!='bottom':
            ax.spines[sp].set_visible(False)
            
            
# plot original paradigm indexes            
symbol_list=['x','o','<','^','v','s','>','+','d',]
color_list=['r', 'g']
n=2  
indexes=list(voltage_info.transitions_dictionary.values())
for i in range(0, len(indexes)-n+1, n):
    axs[0].vlines(indexes[i]/1000,min(trace),max(trace),color=color_list[i%2])
    axs[0].vlines(indexes[i+1]/1000,min(trace),max(trace), color=color_list[(i+1)%2])
    axs[1].plot(indexes[i]/1000, raw_voltage_visstim[indexes[i]],symbol_list[i-int(i/2)],  color=color_list[i%2],label='Start')
    axs[1].plot(indexes[i+1]/1000, raw_voltage_visstim[indexes[i+1]],symbol_list[i-int(i/2)],  color=color_list[(i+1)%2],label='End')
  
# # plot movies trial original indexes
# indexes=list(np.concatenate(paradigmstplot))
# for i in range(0, len(indexes)-n+1, n):
#     axs[0].vlines(indexes[i]/1000,min(trace),max(trace),color=color_list[i%2])
#     axs[0].vlines(indexes[i+1]/1000,min(trace),max(trace), color=color_list[(i+1)%2])
#     axs[1].plot(indexes[i]/1000, raw_voltage_visstim[indexes[i]],'rx')
#     axs[1].plot(indexes[i+1]/1000, raw_voltage_visstim[indexes[i+1]],'go')

# plot downsampled move transitions
indexes=list(np.concatenate(resampled_paradigmstplot))
for i in range(0, len(indexes)-n+1, n):
    axs[0].plot(movie_timestamps[indexes[i]], trace[indexes[i]],'rx')
    axs[0].plot(movie_timestamps[indexes[i+1]], trace[indexes[i+1]],'go')

    axs[1].vlines(movie_timestamps[indexes[i]],min(raw_voltage_visstim),max(raw_voltage_visstim),color=color_list[i%2])
    axs[1].vlines(movie_timestamps[indexes[i+1]],min(raw_voltage_visstim),max(raw_voltage_visstim), color=color_list[(i+1)%2])
    
    
box = axs[1].get_position()
axs[1].set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)




#%% PLOT ALL CELL ALL TRACE
plt.close('all')
for dataset,datset_name in dtset_list:

    analysis=all_analysis[dataset]['analysis']
    full_data=all_analysis[dataset]['full_data']
    
    activity=full_data['imaging_data']['Plane1']['Traces']['denoised']
    activity_timestamps=full_data['imaging_data']['mov_timestamps_seconds']['clipped']-full_data['imaging_data']['mov_timestamps_seconds']['clipped'][0]
    _, processed_speed, _,processed_speed_timestamps=compute_locomotion_bouts(analysis,pretime=1,posttime=1)

    speed_timestamps=processed_speed_timestamps/1000-processed_speed_timestamps[0]/1000
    speed=processed_speed
    
   
    para_movie_starts=list(full_data['visstim_info']['Paradigm_Indexes'].values())
    para_volt_starts=list(analysis.volt_object.extraction_object.transitions_dictionary.values())
   
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fuptit=f'{datset_name} Chandelier Cell Activity'

    f,axs=plt.subplots(8,sharex=True,figsize=(3.54,3.54), dpi=300)
    f.suptitle(fuptit)
    axs[-1].plot(speed_timestamps,speed,'b',linewidth=0.1)
    axs[-1].set_ylim([0,None])
    axs[-1].set_xlabel('Time (s)',fontsize=15)
    
    for cell in range(7):
        axs[cell].plot(activity_timestamps,activity[translate_selected_cell(cell, cell_equivalences, dataset) ,:],linewidth=0.1)
        axs[cell].set_ylim([0,activity.max()])

        for i,trial in enumerate(para_movie_starts):
            axs[cell].vlines(x=activity_timestamps[trial],ymin=0,ymax=activity.max(),linestyles ='dashed',color='r',linewidths=0.2)
        axs[-1].vlines(x=speed_timestamps[para_volt_starts[i]],ymin=0,ymax=speed.max(),color='r')
    for ax in axs:
        ax. margins(x=0)
        for sp in ax.spines :
            if sp!='bottom':
                ax.spines[sp].set_visible(False)
            else:
                ax.spines[sp].set_linewidth(0.2)
  
    plt.savefig(temppath /  Path(fuptit+f'_{timestr}.pdf'), dpi=300, bbox_inches='tight')
#%% PLOT ALL CELL ACTIVTY AROUND THE MOVIE1 SEGMENT THI

#ghet movie 1 paradigm
# plot trace
#plot start trial lines
# plot locomotion
plt.close('all')
for dataset,datset_name in dtset_list:

    analysis=all_analysis[dataset]['analysis']
    full_data=all_analysis[dataset]['full_data']
    for para in ['Natural Movie','Spontaneous']:
        if 'Spont' in para:
            if dataset==0:
                name='spont1_'
            else:    
                name='spont_'
        else:
                name='natural_movie_one_set_'
                
                
        padding_frames=20
        data_slice=slice(full_data['visstim_info']['Paradigm_Indexes'][f'{name}first']-padding_frames,full_data['visstim_info']['Paradigm_Indexes'][f'{name}last']+padding_frames)
        voltage_slice=slice(analysis.volt_object.extraction_object.transitions_dictionary[f'{name}first']-padding_frames,analysis.volt_object.extraction_object.transitions_dictionary[f'{name}last']+padding_frames)
        activity_timestamps=full_data['imaging_data']['mov_timestamps_seconds']['clipped'][data_slice]-full_data['imaging_data']['mov_timestamps_seconds']['clipped'][data_slice][0]
        _, processed_speed, _,processed_speed_timestamps=compute_locomotion_bouts(analysis,pretime=1,posttime=1)
        
        # speed_timestamps=analysis.volt_object.extraction_object.all_signals['Prairie']['Time'].to_numpy().flatten()[voltage_slice]/1000
        # speed=np.abs(np.diff(analysis.volt_object.extraction_object.all_signals['Prairie']['Locomotion'].to_numpy().flatten(),prepend=0))[voltage_slice]
        
        speed_timestamps=processed_speed_timestamps[voltage_slice]/1000-processed_speed_timestamps[voltage_slice][0]/1000
        speed=processed_speed[voltage_slice]
        
        movie1_movie_starts=full_data['visstim_info']['Movie1']['Trial_Starts'].astype('int')-data_slice.start
        movie1_volt_starts=analysis.volt_object.extraction_object.movie_one_trial_full_recording-voltage_slice.start
        
        
        all_cells_activity=full_data['imaging_data']['Plane1']['Traces']['denoised'][:  ,data_slice]
        corrected_cell_all_cell_activity=np.zeros_like(full_data['imaging_data']['Plane1']['Traces']['denoised'][:  ,data_slice])
    
        for cell in range(7):      
            corrected_cell_all_cell_activity[cell,:]=all_cells_activity[ translate_selected_cell(cell, cell_equivalences, dataset),:]
            

        data_slice=slice(full_data['visstim_info']['Paradigm_Indexes'][f'{name}first']-padding_frames,full_data['visstim_info']['Paradigm_Indexes'][f'{name}last']+padding_frames)
        voltage_slice=slice(analysis.volt_object.extraction_object.transitions_dictionary[f'{name}first']-padding_frames,analysis.volt_object.extraction_object.transitions_dictionary[f'{name}last']+padding_frames)
        activity_timestamps=full_data['imaging_data']['mov_timestamps_seconds']['clipped'][data_slice]-full_data['imaging_data']['mov_timestamps_seconds']['clipped'][data_slice][0]
        _, processed_speed, _,processed_speed_timestamps=compute_locomotion_bouts(analysis,pretime=1,posttime=1)
        

        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        fuptit=f'All Chandelier Cells for {datset_name}, {para}'
        f,axes = plt.subplots(8, 1, figsize=(3.54,3.54), dpi=300, constrained_layout=True)
        f.suptitle(fuptit)
        f.subplots_adjust(wspace=0, hspace=0)
        for cell in range(7):
            axes[cell].plot(activity_timestamps,corrected_cell_all_cell_activity[cell,:],linewidth=0.01)
            axes[cell].set_xticks([])
            axes[cell].set_ylim([0,corrected_cell_all_cell_activity.max()])
    
            for sp in axes[cell].spines :
               axes[cell].spines[sp].set_visible(False)
            if 'Spont' not in para:  
                for  i,trial in enumerate(movie1_movie_starts):
                   axes[cell].vlines(x=activity_timestamps[trial],ymin=0,ymax=corrected_cell_all_cell_activity.max(),linestyles ='dashed',color='r',linewidths=0.1)
               
        for sp in axes[-1].spines :
            if sp!='bottom':
                axes[-1].spines[sp].set_visible(False)
            else:
                ax.spines[sp].set_linewidth(0.2)
    
        axes[-1].plot(speed_timestamps,speed,'b',linewidth=0.01)
        axes[-1].set_ylim([0,None])
        # for  i,trial in enumerate(movie1_volt_starts):
        #     axes[-1].vlines(x=speed_timestamps[trial],ymin=0,ymax=max(speed),color='r')
    
        axes[-1].set_xlabel('Time (s)')
        plt.savefig(temppath /  Path(fuptit+f'_{timestr}.pdf'), dpi=300, bbox_inches='tight')
    
    
    
        for cell in range(7):        
            
            activity=full_data['imaging_data']['Plane1']['Traces']['denoised'][translate_selected_cell(cell, cell_equivalences, dataset)  ,data_slice]
             
            timestr = time.strftime("%Y%m%d-%H%M%S")
            fuptit=f'Chandelier Cell {cell+1} for {datset_name},  {para}'
            f,axs=plt.subplots(2,sharex=True,figsize=(3.54,3.54), dpi=300, constrained_layout=True)
            f.suptitle(fuptit)
            axs[0].plot(activity_timestamps,activity,linewidth=0.01)
            axs[-1].plot(speed_timestamps,speed,'b',linewidth=0.01)
            axs[-1].set_ylim([0,None])
            axs[-1].set_xlabel('Time (s)')
            if 'Spont' not in para:  
                for i,trial in enumerate(movie1_movie_starts):
                    axs[0].vlines(x=activity_timestamps[trial],ymin=0,ymax=max(activity),linestyles ='dashed',color='r',linewidths=0.1)
                    # axs[1].vlines(x=speed_timestamps[movie1_volt_starts[i]],ymin=0,ymax=max(speed),color='r')
            for ax in axs:
                ax. margins(x=0)
                for sp in ax.spines :
                    if sp!='bottom':
                        ax.spines[sp].set_visible(False)
                    else:
                        ax.spines[sp].set_linewidth(0.2)
                        
            plt.savefig(temppath /  Path(fuptit+f'_{timestr}.pdf'), dpi=300, bbox_inches='tight')
            
plt.close('all')
#%% PLOT ONE CELL ACTIVTY AROUND THE MOVIE1 SEGMENT THI

#ghet movie 1 paradigm
# plot trace
#plot start trial lines
# plot locomotion
plt.close('all')
for dataset,datset_name in dtset_list:

    analysis=all_analysis[dataset]['analysis']
    full_data=all_analysis[dataset]['full_data']
    for para in ['Natural Movie','Spontaneous']:
        if 'Spont' in para:
            if dataset==0:
                name='spont1_'
            else:    
                name='spont_'
        else:
                name='natural_movie_one_set_'
                
                
        padding_frames=20
        data_slice=slice(full_data['visstim_info']['Paradigm_Indexes'][f'{name}first']-padding_frames,full_data['visstim_info']['Paradigm_Indexes'][f'{name}last']+padding_frames)
        voltage_slice=slice(analysis.volt_object.extraction_object.transitions_dictionary[f'{name}first']-padding_frames,analysis.volt_object.extraction_object.transitions_dictionary[f'{name}last']+padding_frames)
        activity_timestamps=full_data['imaging_data']['mov_timestamps_seconds']['clipped'][data_slice]-full_data['imaging_data']['mov_timestamps_seconds']['clipped'][data_slice][0]
        _, processed_speed, _,processed_speed_timestamps=compute_locomotion_bouts(analysis,pretime=1,posttime=1)
        
        # speed_timestamps=analysis.volt_object.extraction_object.all_signals['Prairie']['Time'].to_numpy().flatten()[voltage_slice]/1000
        # speed=np.abs(np.diff(analysis.volt_object.extraction_object.all_signals['Prairie']['Locomotion'].to_numpy().flatten(),prepend=0))[voltage_slice]
        
        speed_timestamps=processed_speed_timestamps[voltage_slice]/1000-processed_speed_timestamps[voltage_slice][0]/1000
        speed=processed_speed[voltage_slice]
        
        movie1_movie_starts=full_data['visstim_info']['Movie1']['Trial_Starts'].astype('int')-data_slice.start
        movie1_volt_starts=analysis.volt_object.extraction_object.movie_one_trial_full_recording-voltage_slice.start
        
        
        all_cells_activity=full_data['imaging_data']['Plane1']['Traces']['denoised'][:  ,data_slice]
        corrected_cell_all_cell_activity=np.zeros_like(full_data['imaging_data']['Plane1']['Traces']['denoised'][:  ,data_slice])
    
        for cell in range(7):      
            corrected_cell_all_cell_activity[cell,:]=all_cells_activity[ translate_selected_cell(cell, cell_equivalences, dataset),:]
            

        data_slice=slice(full_data['visstim_info']['Paradigm_Indexes'][f'{name}first']-padding_frames,full_data['visstim_info']['Paradigm_Indexes'][f'{name}last']+padding_frames)
        voltage_slice=slice(analysis.volt_object.extraction_object.transitions_dictionary[f'{name}first']-padding_frames,analysis.volt_object.extraction_object.transitions_dictionary[f'{name}last']+padding_frames)
        activity_timestamps=full_data['imaging_data']['mov_timestamps_seconds']['clipped'][data_slice]-full_data['imaging_data']['mov_timestamps_seconds']['clipped'][data_slice][0]
        _, processed_speed, _,processed_speed_timestamps=compute_locomotion_bouts(analysis,pretime=1,posttime=1)
        

        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        fuptit=f'All Chandelier Cells for {datset_name}, {para}'
        f,axes = plt.subplots(8, 1, figsize=(3.54,3.54), dpi=300, constrained_layout=True)
        f.suptitle(fuptit)
        f.subplots_adjust(wspace=0, hspace=0)
        for cell in range(7):
            axes[cell].plot(activity_timestamps,corrected_cell_all_cell_activity[cell,:],linewidth=0.01)
            axes[cell].set_xticks([])
            axes[cell].set_ylim([0,corrected_cell_all_cell_activity.max()])
    
            for sp in axes[cell].spines :
               axes[cell].spines[sp].set_visible(False)
            if 'Spont' not in para:  
                for  i,trial in enumerate(movie1_movie_starts):
                   axes[cell].vlines(x=activity_timestamps[trial],ymin=0,ymax=corrected_cell_all_cell_activity.max(),linestyles ='dashed',color='r',linewidths=0.5)
               
        for sp in axes[-1].spines :
            if sp!='bottom':
                axes[-1].spines[sp].set_visible(False)
            else:
                axes[-1].spines[sp].set_linewidth(0.2)
    
        axes[-1].plot(speed_timestamps,speed,'b',linewidth=0.01)
        axes[-1].set_ylim([0,None])
        # for  i,trial in enumerate(movie1_volt_starts):
        #     axes[-1].vlines(x=speed_timestamps[trial],ymin=0,ymax=max(speed),color='r')
    
        axes[-1].set_xlabel('Time (s)')
        plt.savefig(temppath /  Path(fuptit+f'_{timestr}.pdf'), dpi=300, bbox_inches='tight')
    
    
    
        for cell in range(1):        
            
            activity=full_data['imaging_data']['Plane1']['Traces']['denoised'][translate_selected_cell(cell, cell_equivalences, dataset)  ,data_slice]
             
            timestr = time.strftime("%Y%m%d-%H%M%S")
            fuptit=f'Chandelier Cell {cell+1} for {datset_name},  {para}'
            f,axs=plt.subplots(2,sharex=True,figsize=(3.54,3.54), dpi=300, constrained_layout=True)
            f.suptitle(fuptit)
            axs[0].plot(activity_timestamps,activity,linewidth=0.01)
            axs[-1].plot(speed_timestamps,speed,'b',linewidth=0.01)
            axs[-1].set_ylim([0,None])
            axs[-1].set_xlabel('Time (s)')
            if 'Spont' not in para:  
                for i,trial in enumerate(movie1_movie_starts):
                    axs[0].vlines(x=activity_timestamps[trial],ymin=0,ymax=max(activity),linestyles ='dashed',color='r',linewidths=0.1)
                    # axs[1].vlines(x=speed_timestamps[movie1_volt_starts[i]],ymin=0,ymax=max(speed),color='r')
            for ax in axs:
                ax. margins(x=0)
                for sp in ax.spines :
                    if sp!='bottom':
                        ax.spines[sp].set_visible(False)
                    else:
                        ax.spines[sp].set_linewidth(0.2)
                        
            plt.savefig(temppath /  Path(fuptit+f'_{timestr}.pdf'), dpi=300, bbox_inches='tight')
            
plt.close('all')

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
        mask[np.triu_indices_from(mask,k=1)] = True
        mean_ta_similarity_across_cells[dataset,para]=np.mean(trial_averaged_similarity_across_cells[mask])  
        mean_con_similarity_across_cells[dataset,para]=np.mean(trial_concatenated_similarity_across_cells[mask])  
        
        
        #plot similarity across trial averaged cell responses
        # for dtst,typ in zip([trial_averaged_similarity_across_cells,trial_concatenated_similarity_across_cells],['Trial Averaged','Trial Concatenated']):
        #     timestr = time.strftime("%Y%m%d-%H%M%S")
        #     ffuptit=f'{typ} for {datset_name}, {para_name}, {similarity[select_similarity]} Similarity Across Cells'
        #     plt.figure(figsize=(20, 20))
        #     sns.heatmap(dtst,vmin=vmin,vmax=1,cmap='viridis', annot=True, fmt='.2f', cbar=True)
        #     plt.title(ffuptit)
            # plt.savefig(temppath /  Path(ffuptit+f'_{timestr}.pdf'), dpi=300, bbox_inches='tight')
            
            
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
            fig,axs=plt.subplots(3,sharex=True,figsize=(3.54,3.54), dpi=300, constrained_layout=True) 
            fig.suptitle(fuptit)
            all_tr_av_mean.append(mean.values)
            all_tr_av_sem.append(sem.values)
            axs[0].plot(corrected_timestamps[0,:], mean.values, color='blue',linewidth=0.5)
            axs[0].fill_between(corrected_timestamps[0,:], mean.values - sem.values, mean.values + sem.values, color='blue', alpha=0.3)
            sns.lineplot(data=df, x='Time (s)', y='Value', hue='Trial', palette='Greys', alpha=0.5,linewidth=0.5, ax=axs[0])
            for ax in axs:
                for spine in ax.spines.values():
                    if spine != ax.spines['bottom']:
                        spine.set_visible(False)
                        
            axs[0].get_legend().remove()
            axs[1].plot(np.linspace(0,30,899),np.abs(np.diff(np.abs(angl),prepend=0)),linewidth=0.01)
            axs[2].plot(np.linspace(0,30,900),intensity,linewidth=0.5)
            plt.savefig(temppath /  Path(fuptit+f'_{timestr}.pdf'), dpi=300, bbox_inches='tight')

        
            #PLot trial averaged activity and trials in separeted subplots

            # timestr = time.strftime("%Y%m%d-%H%M%S")
            # fffuptit=f'Chandelier Cell {cell+1} for {datset_name}, {para_name}, Trial Responses'
            # fig,axs=plt.subplots(11,sharex=True,figsize=(20,20), constrained_layout=True)
            # fig.suptitle(fffuptit)
            # for i in range(10):
            #     sns.lineplot(x=corrected_timestamps[0, :], y=activity[cell, i, :], ax=axs[i], color='gray', alpha=0.7,linewidth=1)
            #     for spine in axs[i].spines.values():
            #         if spine != axs[i].spines['bottom']:
            #             spine.set_visible(False)
            # axs[10].plot(corrected_timestamps[0, :], mean.values, color='blue', label='Average Trial',linewidth=2)
            # axs[10].fill_between(corrected_timestamps[0, :], mean.values - sem.values, mean.values + sem.values, color='blue', alpha=0.3)
            # for spine in axs[10].spines.values():
            #     if spine != axs[i].spines['bottom']:
            #         spine.set_visible(False)                   
            # axs[-1].set_xlabel('Time (s)')
            # plt.savefig(temppath /  Path(fffuptit+f'_{timestr}.png'), dpi=300, bbox_inches='tight')
            
                 
            # calculate similarity measure , plot heatmaps and stroa averaged similarity across trials per recoded cell
            trial_activity_no_loco=activity_without_loc[cell,:,:]
            if similarity[select_similarity]=='cosine':
                norms = np.linalg.norm(trial_activity_no_loco, axis=1, keepdims=True)
                normalized_X = trial_activity_no_loco / norms
                cell_similarity_across_trials= np.dot(normalized_X, normalized_X.T)

            elif similarity[select_similarity]=='pearson':
                cell_similarity_across_trials = np.corrcoef(trial_activity_no_loco)
                
            mask = np.zeros_like(cell_similarity_across_trials, dtype=bool)
            mask[np.triu_indices_from(mask,k=1)] = True
            mean_similarity_across_trials[dataset,para,cell]=np.mean(cell_similarity_across_trials[mask])
            
            # timestr = time.strftime("%Y%m%d-%H%M%S")
            # ffuptit=f'Chandelier Cell {cell+1} for {datset_name}, {para_name}, {similarity[select_similarity]} Similarity Across Trials'
            # plt.figure(figsize=(20, 20))
            # sns.heatmap(cell_similarity_across_trials,vmin=vmin,vmax=1,cmap='viridis', annot=True, fmt='.2f', cbar=True)
            # plt.title(ffuptit)
            # plt.savefig(temppath /  Path(ffuptit+f'_{timestr}.pdf'), dpi=300, bbox_inches='tight')


            
        all_para_tr_av_mean.append(all_tr_av_mean)
        all_para_tr_av_sem.append(all_tr_av_sem)
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # ffuptit=f'All Chandelier Cells for {datset_name}, {para_name}, Trial Averaged'
        # f,axes=plt.subplots(7,sharex=True,figsize=(20,20), constrained_layout=True)
        # f.suptitle(ffuptit)
        # for cell in range(data['CellNumber'])  :         
        #     axes[cell].plot(corrected_timestamps[0,:], all_tr_av_mean[cell], color='blue')
        #     axes[cell].fill_between(corrected_timestamps[0,:],all_tr_av_mean[cell] - all_tr_av_sem[cell], all_tr_av_mean[cell] + all_tr_av_sem[cell], color='blue', alpha=0.3)
        #     for spine in ax.spines.values():
        #         if spine != ax.spines['bottom']:
        #             spine.set_visible(False)
        # plt.savefig(temppath /  Path(ffuptit+f'_{timestr}.pdf'), dpi=300, bbox_inches='tight')
    all_datasets_tr_av_mean.append(all_para_tr_av_mean)
    all_datasets_tr_av_sem.append(all_para_tr_av_sem)
 
plt.close('all')
  
#%% CHEK GRAND AVERAGRE OF CELLS AND SESSIONS AGAINST MOTION
timestr = time.strftime("%Y%m%d-%H%M%S")
ffuptit=f'Grand Average Against Motion'
f,axes=plt.subplots(5,figsize=(20,20), constrained_layout=True)
f.suptitle(ffuptit)
tosave=[]
for sess in range(3):  
    axes[0].plot(np.vstack(all_datasets_tr_av_mean[sess][0]).mean(axis=0)[0:847])
    tosave.append(np.vstack(all_datasets_tr_av_mean[sess][0]).mean(axis=0)[0:847])
axes[1].plot(np.vstack(tosave).mean(axis=0))
axes[2].plot(np.linspace(0,30,899),angl)
axes[3].plot(np.linspace(0,30,899),np.abs(angl))
axes[4].plot(np.linspace(0,30,899),np.abs(np.diff(angl,prepend=0)))


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
            correlations = np.zeros((session_activity_no_loco.shape[0], session_activity_no_loco.shape[0]))  # Initialize correlation matrix
            for i in range(session_activity_no_loco.shape[0]):
                for j in range(i + 1, session_activity_no_loco.shape[0]):
                    corr, _ = pearsonr(session_activity_no_loco[i,:], session_activity_no_loco[j,:])
                    correlations[i, j] = corr
                    correlations[j, i] = corr  # Symmetric
                        
            trial_averaged_cell_similarity_across_sessions = np.corrcoef(session_activity_no_loco)
            vmin=-1

            
        mask = np.zeros_like(trial_averaged_cell_similarity_across_sessions, dtype=bool)
        mask[np.triu_indices_from(mask,k=1)] = True
        mean_cell_similarity_across_sessions[para,cell]=np.mean(trial_averaged_cell_similarity_across_sessions[mask])

        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        ffuptit=f'Chandelier Cell {cell+1} for {para_name}, {similarity[select_similarity]} Similarity Across Sessions'
        plt.figure(figsize=(20, 20))
        sns.heatmap(trial_averaged_cell_similarity_across_sessions,vmin=vmin,vmax=1,cmap='viridis', annot=True, fmt='.2f', cbar=True,annot_kws={"fontsize":16})
        plt.title(ffuptit)
        plt.savefig(temppath /  Path(ffuptit+f'_{timestr}.pdf'), dpi=300, bbox_inches='tight')


#%%
def fisher_z_transformation(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def fisher_z_to_z_stat(z, n):
    se_z = 1 / np.sqrt(n - 3)
    return z / se_z

n_measurements = session_activity_no_loco.shape[1]  # Number of measurements per trial
z_values = np.zeros((session_activity_no_loco.shape[0], session_activity_no_loco.shape[0]))
p_values = np.zeros((session_activity_no_loco.shape[0], session_activity_no_loco.shape[0]))

for i in range(session_activity_no_loco.shape[0]):
    for j in range(i + 1, session_activity_no_loco.shape[0]):
        r = correlations[i, j]
        z = fisher_z_transformation(r)
        z_stat = fisher_z_to_z_stat(z, n_measurements)
        p_value = 2 * (1 - st.norm.cdf(np.abs(z_stat)))
        z_values[i, j] = z_stat
        z_values[j, i] = z_stat  # Symmetric matrix
        p_values[i, j] = p_value
        p_values[j, i] = p_value  # Symmetric matrix

print("Z-values matrix:\n", z_values)
print("P-values matrix:\n", p_values)

from statsmodels.stats.multitest import multipletests

# Flatten p-values for Bonferroni correction
p_values_flat = p_values[np.triu_indices(session_activity_no_loco.shape[0], k=1)]  # Upper triangle, excluding diagonal

# Apply Bonferroni correction
corrected_p_values = multipletests(p_values_flat, method='bonferroni')[1]

# Update p-values matrix with corrected values
corrected_p_values_matrix = np.zeros_like(p_values)
triu_indices = np.triu_indices(session_activity_no_loco.shape[0], k=1)
corrected_p_values_matrix[triu_indices] = corrected_p_values
corrected_p_values_matrix += corrected_p_values_matrix.T  # Symmetric matrix

print("Corrected p-values matrix:\n", corrected_p_values_matrix)

#%% STATISTICAKLL ANALYSIS OF CORRELATIONS

# conditions = ['Movie', 'Spont']  # Two conditions
# trials = [f'Trial {i+1}' for i in range(mean_cell_similarity_across_sessions.shape[1])]  # Seven trials

# # Prepare data for boxplot
# df_list = []

# for i in range(mean_cell_similarity_across_sessions.shape[0]):  # Loop through conditions
#     condition_data = mean_cell_similarity_across_sessions[i, :]  # Extract data for each condition
#     df = pd.DataFrame({
#         'Value': condition_data,
#         'Condition': conditions[i]
#     })
#     df_list.append(df)

# df_all = pd.concat(df_list, ignore_index=True)

# # Plot the data
# plt.figure(figsize=(20, 20))
# sns.boxplot(data=df_all, x='Condition', y='Value',  palette='Set2')
# ffuptit='Mean Similarity Across Sessions'
# plt.title(ffuptit)
# plt.xlabel('Trial')
# plt.ylabel('Value')
# plt.legend(title='Condition')
# plt.tight_layout()
# plt.show()
# plt.savefig(temppath /  Path(ffuptit+f'_{timestr}.png'), dpi=300, bbox_inches='tight')


# Define labels
# conditions = ['Movie', 'Spontaneous']  # Conditions (second axis)
# sessions = ['Session A', 'Session B', 'Session C']  # Sessions (first axis)

# # Prepare data for boxplot
# df_list = []

# for i in range(mean_similarity_across_cells.shape[0]):  # Loop through sessions
#     session_data = mean_similarity_across_cells[i, :]  # Extract data for each session
#     df = pd.DataFrame({
#         'Condition': conditions,
#         'Value': session_data,
#         'Session': sessions[i]
#     })
#     df_list.append(df)

# df_all = pd.concat(df_list, ignore_index=True)

# # Plot the data
# plt.figure(figsize=(20, 20))
# sns.boxplot(data=df_all, x='Condition', y='Value', palette='Set2')
# ffuptit='Mean Cell Similarity Across Stimuli'
# plt.title(ffuptit)
# plt.xlabel('Condition')
# plt.ylabel('Value')
# plt.legend(title='Session')
# plt.tight_layout()
# plt.show()
# plt.savefig(temppath /  Path(ffuptit+f'_{timestr}.png'), dpi=300, bbox_inches='tight')

 
    

# # Define labels
# conditions = ['Movie', 'Spontaneous']  # Conditions (second dimension)
# sessions = ['Session A', 'Session B', 'Session C']  # Sessions (first dimension)

# # Prepare data for boxplot
# df_list = []

# for i in range(data.shape[0]):  # Loop through sessions
#     session_data = mean_similarity_across_trials[i, :, :].T  # Extract and transpose data for each session
#     df = pd.DataFrame(session_data, columns=conditions)  # Create DataFrame with conditions as columns
#     df['Session'] = sessions[i]  # Add session labels
#     df = df.melt(id_vars=['Session'], var_name='Condition', value_name='Mean Similarity')  # Melt DataFrame for seaborn
#     df_list.append(df)

# df_all = pd.concat(df_list, ignore_index=True)

# # Plot the data
# plt.figure(figsize=(20, 20))
# sns.boxplot(data=df_all, x='Session', y='Mean Similarity', hue='Condition', palette='Set2')
# ffuptit='Mean Trial Similarity Across Stimuli and Sessions'
# plt.title(ffuptit)
# plt.xlabel('Condition')
# plt.ylabel('Mean Similarity')
# plt.legend(title='Session')
# plt.tight_layout()
# plt.savefig(temppath /  Path(ffuptit+f'_{timestr}.png'), dpi=300, bbox_inches='tight')


# for para,para_name in [(0,'Natural Movie'),(1,'Spontaneous')]: 
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     ffuptit=f'All Chandelier Cells All Sessions, {para_name}, Trial Averaged'
#     f,axes=plt.subplots(7,sharex=True,figsize=(20,20), constrained_layout=True)
#     f.suptitle(ffuptit)    
#         for dataset,datset_name in dtset_list:


# for para,para_name in [(0,'Natural Movie'),(1,'Spontaneous')]: 
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     ffuptit=f'All Chandelier Cells All Sessions, {para_name}, Trial Averaged'
#     f,axes=plt.subplots(7,sharex=True,figsize=(20,20), constrained_layout=True)
#     f.suptitle(ffuptit)    
#     for cell in range(7):    
#         for dataset,datset_name in dtset_list:
            
#             if dataset==1:
#                 mean=all_datasets_tr_av_mean[dataset][para][cell][:-1]
#                 sem=all_datasets_tr_av_sem[dataset][para][cell][:-1]
#             else:
#                 mean=all_datasets_tr_av_mean[dataset][para][cell]
#                 sem=all_datasets_tr_av_sem[dataset][para][cell]
#             colors=['red','cyan','green']

#             axes[cell].plot(corrected_timestamps[0,:],   mean, color=colors[dataset])
#             axes[cell].fill_between(corrected_timestamps[0,:],mean -  sem, mean +   sem, color=colors[dataset], alpha=0.1,label=datset_name)
      
#         for spine in   axes[cell].spines.values():
#             if spine !=   axes[cell].spines['bottom']:
#                 spine.set_visible(False)
#     axes[0].legend()

#     plt.savefig(temppath /  Path(ffuptit+f'_{timestr}.png'), dpi=300, bbox_inches='tight')
   
# plt.close('all')


# for para,para_name in [(1,'Natural Movie'),(3,'Spontaneous')]: 
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     fffuptit=f'All Chandelier Cells All Sessions, {para_name}, Trial Responses'
#     fig,axs=plt.subplots(7,sharex=True,figsize=(20,20), constrained_layout=True)
#     fig.suptitle(fffuptit)
    
#     for cell in range(7):    
#         for dataset,datset_name in dtset_list:
#             colors=['red','cyan','green']

#             activity=all_res[dataset][para][cell]
#             timestamps=all_res[dataset][5][0]

            
#             corrected_timestamps=  timestamps - timestamps[ 0]

#             for trial in range(10):
#                 sns.lineplot(x=corrected_timestamps, y=activity[trial, :], ax=axs[cell], color=colors[dataset], alpha=0.7,linewidth=1)
#             for spine in axs[cell].spines.values():
#                 if spine != axs[cell].spines['bottom']:
#                     spine.set_visible(False)

#     plt.savefig(temppath /  Path(fffuptit+f'_{timestr}.png'), dpi=300, bbox_inches='tight')

    
    

# Rank the data
ranked_data = np.apply_along_axis(rankdata, 1, session_across_cells[treat,:,:])
spearman_corr_matrix = np.corrcoef(ranked_data)


pearson_corr_matrix = np.corrcoef(session_across_cells[treat,:,:])

        

        
#%% this is getting the corrwelations of sinmgle cells across trials



def analyzed_locomotion(dataset,all_analysis):
    
    analysis=all_analysis[dataset]['analysis']
    full_data=all_analysis[dataset]['full_data']
    imaging_data=full_data['imaging_data']
    
    
    locomotion_on_offsets, speed, raw_bouts,_=compute_locomotion_bouts(analysis,pretime=1,posttime=1)
    
    long_bouts=[off for off in locomotion_on_offsets if  off[2]>100]
    
    
    movie_timestamps=imaging_data['mov_timestamps_seconds']['clipped']
    data=imaging_data['Plane1']
    x_trace=imaging_data['Plane1']['Traces']['denoised']
    
    
    # get the slices for movie 1
    trial_slices=[]
    for trial, frame in enumerate(long_bouts):
            trial_slices.append(slice(frame[0],frame[1]))
            frames=min([  max((sl.stop - sl.start), 1) for sl in trial_slices])

        
    trial_slices=[]
    for trial, frame in enumerate(long_bouts):
        trial_slices.append(slice(frame[0],frame[0]+frames))
          
        
    # set the arrays
    trials_loco=np.zeros([data['CellNumber'],len(trial_slices),frames])
    trials_loco_timestamps=np.zeros([len(trial_slices),frames])
    
    
    
    # SLCIE DATA
    for cell in range(data['CellNumber'])  : 
        
        for trial, sl in enumerate(trial_slices):
            trials_loco[cell,trial,:]=data['Traces']['denoised'][translate_selected_cell(cell, cell_equivalences, dataset)  ,sl]
            trials_loco_timestamps[trial,:]=movie_timestamps[sl]

    return trials_loco, trials_loco_timestamps

#%% ANALYZE LOCOMOTION BOUT SIMILARITY
plt.close('all')
loc_res=[]
for dataset in range(3):
    loc_res.append(analyzed_locomotion(dataset,all_analysis))
    
    
    
#one cell all locomotion trials    
examp=loc_res[dataset][0][0,:,:]    
    
norms = np.linalg.norm(examp, axis=1, keepdims=True)
normalized_X = examp / norms

# Compute cosine similarity matrix
cosine_sim_matrix = np.dot(normalized_X, normalized_X.T)

# Plot the cosine similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim_matrix,vmin=0,vmax=1, cmap='viridis', annot=True, fmt='.2f', cbar=True)
plt.title('Cosine Similarity Matrix')
plt.xlabel('Row Index')
plt.ylabel('Row Index')
plt.show()


 #spont activity
 
examp_spont=res[dataset][3] 
#%% PLLOOT LOCOMOTIO BOUTS WITH WIDE MARGINS
plt.close('all')
preframes=10
locomotion_on_offsets, speed, raw_bouts,_=compute_locomotion_bouts(analysis,pretime=1, posttime=1)
long_bouts=locomotion_on_offsets
long_bouts_up=raw_bouts

    

cell=1

x_trace=imaging_data['Plane1']['Traces']['denoised']
x_locomotion=speed
x_volt_timestamps=signals['Time'].to_numpy().flatten()/1000
x_timestamps=imaging_data['mov_timestamps_seconds']['clipped']

f,axs=plt.subplots(2, sharex=True)
axs[0].plot(x_timestamps,x_trace[translate_selected_cell(cell, cell_equivalences, dataset),:])
axs[1].plot(x_volt_timestamps,x_locomotion)
for i in long_bouts_up:
    axs[1].plot(x_volt_timestamps[i[0]],x_locomotion[i[0]],'rx')
    axs[1].plot(x_volt_timestamps[i[1]],x_locomotion[i[1]],'co')



f,axs=plt.subplots(2*len(long_bouts), sharex=True)
for i in range(0,2*len(long_bouts_up),2):
    p=int(i/2)
    
    sliced_timestamps_volt= x_volt_timestamps[long_bouts_up[p][0]:long_bouts_up[p][1]] -x_volt_timestamps[long_bouts_up[p][0]:long_bouts_up[p][1]][0]
    sliced_timestamps= x_timestamps[long_bouts[p][0]:long_bouts[p][1]] -x_timestamps[long_bouts[p][0]:long_bouts[p][1]][0]


    sliced_trace=x_trace[translate_selected_cell(cell, cell_equivalences, dataset),long_bouts[p][0]:long_bouts[p][1]]
    sliced_locomotion=x_locomotion[long_bouts_up[p][0]:long_bouts_up[p][1]]
    axs.flatten()[i].plot(sliced_timestamps,sliced_trace)
    axs.flatten()[i+1].plot(sliced_timestamps_volt,sliced_locomotion,'r')


    # for spine in axs[i].spines.values():
    #     if spine != axs[i].spines['bottom'] or spine != axs[i].spines['left']:
    #        spine.set_visible(False)
    # for spine in axs[i+1].spines.values():
    #     if spine != axs[i].spines['bottom'] or spine != axs[i].spines['left']:
    #        spine.set_visible(False)
       




