# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:38:25 2024

@author: sp3660

Trying tio code ZETA for cell responsiveness to opto stimulus
"""

# firsat select single cell data oit uses soike times though
import numpy as np
from scipy.signal import gaussian, convolve,find_peaks
from scipy.interpolate import interp1d

cell=5
opto=np.vstack(trial_structure_sweep['opto_blank'][str(cell)])
control=np.vstack(trial_structure_sweep['control_blank'][str(cell)])

demixed=analysis.full_data['imaging_data']['All_planes_rough']['Traces']['demixed']
denoised=analysis.full_data['imaging_data']['All_planes_rough']['Traces']['denoised']
rawdfdt=analysis.full_data['imaging_data']['All_planes_rough']['Traces']['dfdt_raw']
smoothdfdt=analysis.full_data['imaging_data']['All_planes_rough']['Traces']['dfdt_smoothed']

f,ax=plt.subplots(4,sharex=True)
ax[0].plot(demixed[cell,:])
ax[1].plot(denoised[cell,:])
ax[2].plot(rawdfdt[cell,:])
ax[3].plot(smoothdfdt[cell,:])


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



f,ax=plt.subplots(1)
ax.plot(speed_timestamps, cm_s, label='Original (1 sample/second)')
ax.plot(upsampled_time_milliseconds_stamps, upsampled_signal, label='Upsampled (1000 samples/second)')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Amplitude')
ax.legend()

peaks=find_peaks(upsampled_signal)[0]
firstpeaks=np.diff(peaks,prepend=-10000)>10000
laststpeaks=np.roll(firstpeaks,-1)

boutstarts=peaks[np.diff(peaks,prepend=-10000)>10000]-30
boutsends=peaks[laststpeaks]+30

clippedvoltagetime=analysis.full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['Time']
ledclippedlocomotion=analysis.full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['Locomotion'].values
f,ax=plt.subplots(1)
# ax.plot(upsampled_signal)
# ax.plot(ledclippedlocomotion)

ax.plot(clippedvoltagetime,ledclippedlocomotion)
ax.plot(clippedvoltagetime,upsampled_signal)

ax.plot(clippedvoltagetime[boutstarts],upsampled_signal[boutstarts],'ro')
ax.plot(clippedvoltagetime[boutsends],upsampled_signal[boutsends],'bo')


locomotion_on_offsets=[]
for bout in range(len(boutstarts)):
    locomotion_on_offsets.append((np.abs( np.array(np.array(analysis.all_planes_timestamps['Plane1'])*1000 - boutstarts[bout])).argmin(),np.abs( np.array(np.array(analysis.all_planes_timestamps['Plane1'])*1000 - boutsends[bout])).argmin()))


for j,i in enumerate(locomotion_on_offsets):
    locomotion_on_offsets[j]=(i[0],i[1],i[1]-i[0])
    
    
    