#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:15:41 2024

@author: sp3660
"""

#replicating caiman smooth dfdt in python
import scipy.signal as sg
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

#%% load data form optoanalysis cript
i=0
experiment=list(multiple_analysis.keys())[i]
aq_analysis=all_analysis[i]['analysis']
aq_all_info=multiple_analysis[experiment]
res=aq_analysis.caiman_results[list(aq_analysis.caiman_results.keys())[0]]

#%%
normalize=False
rectify=True
do_smooth=True
timewindow=100 #ms


fr=aq_analysis.full_data['imaging_data']['Frame_rate']
do_smooth=res.data['proc']['deconv']['smooth_dfdt']['params']['convolve_gaus']
timewindow=res.data['proc']['deconv']['smooth_dfdt']['params']['gauss_kernel_simga']
normalize=res.data['proc']['deconv']['smooth_dfdt']['params']['normalize']
rectify=res.data['proc']['deconv']['smooth_dfdt']['params']['rectify']



dt=1000/fr
sigma_frames=timewindow/dt
cell=0
# these are the accepetd cell already
data=aq_analysis.full_data['imaging_data']['Plane1']['Traces']['demixed'][cell]
caiman_dfdt_smoothed=aq_analysis.full_data['imaging_data']['Plane1']['Traces']['dfdt_raw'][cell]
caiman_dfdt_smoothed_thresholded=aq_analysis.full_data['imaging_data']['Plane1']['Traces']['dfdt_smoothed'][cell]

#% make kernel]]
kernel_half_size = np.ceil(np.sqrt(-np.log(0.05)*2*sigma_frames**2));
gaus_win = np.arange(-kernel_half_size,kernel_half_size+1)
gaus_kernel = np.exp(-((gaus_win)**2)/(2*sigma_frames**2));
gaus_kernel = gaus_kernel/sum(gaus_kernel);


temp_data = data.astype('float64')

# !% derivative
temp_data = np.diff(temp_data,prepend=temp_data[0]).astype('float64')

# % convol!ve gaussian
if do_smooth:
    temp_data = sg.convolve(temp_data, gaus_kernel, mode='same');


# % no!rmalize
if normalize:
    temp_data = temp_data/max(temp_data)

# % rectify
if rectify:
    temp_data[temp_data < 0] = 0


smooth_dfdt_data= temp_data
f,ax=plt.subplots(2,sharex=True)
ax[0].plot(smooth_dfdt_data)
ax[0].plot(caiman_dfdt_smoothed)
ax[1].plot(caiman_dfdt_smoothed-smooth_dfdt_data)

#%%

dfdt_std_threshold=2
frames=res.data['proc']['deconv']['smooth_dfdt']['S'].shape[1]

dfdt_accepted_std= res.data['proc']['deconv']['smooth_dfdt']['S_std'][res.accepted_indexes_sorter][0]
thresholdline=np.array(dfdt_std_threshold* dfdt_accepted_std)
std_filter=np.tile( np.expand_dims( thresholdline, 0), [1, frames])


dfdt_thesholded_accepted_trace=deepcopy(smooth_dfdt_data)
dfdt_thesholded_accepted_trace[ dfdt_thesholded_accepted_trace< std_filter[0,:]]= std_filter[0,:][ dfdt_thesholded_accepted_trace< std_filter[0,:]]
dfdt_thesholded_accepted_trace=  dfdt_thesholded_accepted_trace- std_filter[0,:]
binarized_dfdt=np.where( dfdt_thesholded_accepted_trace > 0, 1, 0)


f,ax=plt.subplots(2,sharex=True)
ax[0].plot(dfdt_thesholded_accepted_trace)
ax[0].plot(caiman_dfdt_smoothed_thresholded)
ax[1].plot(dfdt_thesholded_accepted_trace-caiman_dfdt_smoothed_thresholded)
