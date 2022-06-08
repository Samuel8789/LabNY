# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:11:20 2022

@author: sp3660
"""
import os
import glob
from ScanImageTiffReader import ScanImageTiffReader
import caiman as cm
import numpy as np
import scipy.signal as sg


mouseflder=r'G:\Projects\TemPrairireSSH\20220525 Hakim\Mice\SPKU'
ocord='0CoordinateAcquisiton'
surface='FOV_\SurfaceImage'
plane='FOV_'

ocrodfile=glob.glob(os.path.join(mouseflder,ocord)+'\**.tif')[0]
surfacefile=glob.glob(os.path.join(mouseflder,surface)+'\**.tif')[0]
planefile=glob.glob(os.path.join(mouseflder,plane)+'\**_Plane**.tif')[0]


output=os.path.join(mouseflder, r'ToTrack')

allfiles=[ocrodfile, surfacefile, planefile]
filenames=['0Coord.tiff', 'Surface.tiff', 'Plane.tiff' ]
for i, file in enumerate(allfiles):
    full_raw_acquisition=ScanImageTiffReader(file)
    vol=full_raw_acquisition
    
    fr = 60
    px = int(15360 / fr)
    px=512
    splits=1
    
    lmov = vol.__len__()
    
    data=[]
    ch=[]
    #%%
    for split in range(splits):
        
        subslice=int(lmov/(split+1))
        
        data1 = vol.data(split*subslice, subslice*(split+1) )
        data.append(data1.reshape([int(lmov/((split+1)*2)), 2, px, px]))
    
    #%%
    chs=[data[0][:, 0, :, : ] , data[0][:, 1, :, : ] ]    
    
    
    # ch2 = np.concatenate((data1[:, 1, :, : ], data2[:, 1, :, : ] ), axis = 0)
    
    
    movs=[cm.movie(i) for i in chs]
    
    
    
    frame=fr
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
    
    
    for j, mov in enumerate(movs):
        smoothed=np.zeros_like(mov)
        for x in np.arange(mov.shape[1]):
            for y in np.arange(mov.shape[2]):
                smoothed[:,x,y]=gaussian_smooth_kernel_convolution(mov[:,x,y],frame,sigma)
        smoothed_motion_corrected=cm.movie(smoothed)
        proje=smoothed_motion_corrected.mean(axis=0)
        proje.save(os.path.join(output,'Ch_{}_'.format(j+1)+ filenames[i]))