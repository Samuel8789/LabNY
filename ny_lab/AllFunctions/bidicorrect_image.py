# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:37:52 2021

@author: sp3660
"""
import numpy as np
import copy

def shiftBiDi(BiDiPhase, image):
    [Lx, Ly]=image.shape 
    shifted_image=copy.copy(image)
    yrange = np.arange(1,Ly,2)
    if BiDiPhase>0:
        shifted_image[yrange,BiDiPhase:Lx-1] = image[yrange, 0:(Lx-1-BiDiPhase)]
    else:
        shifted_image[yrange,0:Lx-1+BiDiPhase]  = image[yrange, -BiDiPhase:Lx-1]

    return shifted_image


def biDiPhaseOffsets(image):       
         
    [Lx, Ly]=image.shape  
    
    # lines scanned one direction
    yr1=np.arange(1,int(np.floor(Ly/2)*2),2)
    
    #lines scanned in other direction
    yr2=np.arange(0,int(np.floor(Ly/2)*2),2)
    
    eps0 = np.float32(1e-6)
    d1 = np.fft.fft(image[yr1,:], axis=1)
    d2 = np.conjugate(np.fft.fft(image[yr2,:],axis=1))
    d1 = d1/(abs(d1) + eps0)
    d2 = d2/(abs(d2) + eps0)
    
    cc = np.real_if_close(np.fft.ifft(d1*d2, axis=1))
    
    cc = np.fft.fftshift(cc, axes=1);
    cc = np.mean(cc,0)
    
    ix=np.argmax(cc[int(np.floor(Lx/2)) + np.arange(-5,6)])
    ix     = ix - (5);
    BiDiPhase = -1 * ix;

    return BiDiPhase