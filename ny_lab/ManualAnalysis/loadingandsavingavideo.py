# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 08:45:44 2023

@author: sp3660
"""
import caiman as cm
import os
import glob
from pathlib import Path
from skimage.measure import block_reduce
import numpy as np
import subprocess 


acqdir=r'/media/sp3660/Data Slow/Projects/LabNY/Full_Mice_Pre_Processed_Data/Mice_Projects/Tigre_Controls/VGC/Ai162/SPSU/imaging/20240118/data aquisitions/FOV_1/240118_SPSU_FOV1_2z_ShortDrift_AllCell_617optoFulp_25x_920_51020_60745_with-000/planes/Plane2/Red/'
acqdir=r'/media/sp3660/Data Slow/Projects/LabNY/Full_Mice_Pre_Processed_Data/Mice_Projects/Tigre_Controls/VGC/Ai162/SPSU/imaging/20240118/data aquisitions/FOV_1/240118_SPSU_FOV1_2z_ShortDrift_AllCell_617optoFulp_25x_920_51020_60745_with-000/planes/Plane2/Green/'
mmapfilepath=Path(glob.glob(acqdir+os.sep+'**.mmap')[0])
tiffilename=mmapfilepath.stem + '_'+acqdir[acqdir.find('planes'+os.sep+'Plane')+14:-1]+'.tiff'


acqdir=os.path.join(os.path.expanduser('~'),r'Desktop/CaimanTemp')
mmapfilepath=Path(glob.glob(acqdir+os.sep+'**ACID**.mmap')[0])
tiffilename=mmapfilepath.stem + '_'+'green.tiff'


# desktop=r'C:\Users\sp3660\Desktop\MoviesToCheck'
desktop=os.path.join(os.path.expanduser('~'),r'Desktop/MoviesToCheck')


denoisedname='240118_SPSM_FOV1_2z_ShortDrift_AllCell_617optoFulp_25x_920_51020_60745_with-000_plane1_Shifted_Movie_MC_OnACID_d1_256_d2_256_d3_1_order_F_frames_20856_denoised.mmap'
mmapfilepath=str(Path(acqdir) / Path(denoisedname))

import subprocess 
subprocess.Popen(['xdg-open', desktop])

mov=cm.load(mmapfilepath)

mov2=mov.astype(np.uint16)

mov2.save(Path(desktop, tiffilename),compress=1, to32=False)

import scipy.signal as sg
def do_temporal_gaussian_smoothing(sigma,mov):
    
    frame=24

    sigma=100#ms

    def gaussian_smooth_kernel_convolution(signal, fr, sigma):
        dt = 1000/fr
        sigma_frames = sigma/dt
        # make kernel
        kernel_half_size = int(np.ceil(np.sqrt(-np.log(0.05)*2*sigma_frames**2)))
        gaus_win =list(range( -kernel_half_size,kernel_half_size+1))
        gaus_kernel = [np.exp(-(i**2)/(2*sigma_frames**2)) for i in gaus_win]
        gaus_kernel = gaus_kernel/sum(gaus_kernel)
        conv_trace = sg.convolve2d(np.expand_dims(signal,1), np.expand_dims(gaus_kernel,1), mode='same')
        return conv_trace.flatten()

    smoothed=np.zeros_like(mov)
    for x in np.arange(mov.shape[1]):
        for y in np.arange(mov.shape[2]):
            smoothed[:,x,y]=gaussian_smooth_kernel_convolution(mov[:,x,y],frame,sigma)
    smoothed_motion_corrected=cm.movie(smoothed)
    
    return smoothed_motion_corrected
    
filt=do_temporal_gaussian_smoothing(100,mov)
mmapilename=mmapfilepath.stem + '_'+'green.mmap'

mov.save(Path(desktop, mmapilename), to32=False)
