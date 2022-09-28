# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:37:52 2021

@author: sp3660
"""
import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy as sp
from numpy import fft


def biDiPhaseOffsets(image, maxphase=5, sigma=0.35):       
    [Ly, Lx]=image.shape  
    image=sp.ndimage.gaussian_filter(image,sigma)


    
    eps0 = np.float32(1e-6)
    d1 = np.fft.fft(image[1::2,:], axis=1)
    d1 = d1/(abs(d1) + eps0)

    d2 = np.conj(np.fft.fft(image[::2,:],axis=1))
    d2 = d2/(abs(d2) + eps0)
    
    cc = np.real(np.fft.ifft(d1*d2, axis=1))
    
    cc = np.fft.fftshift(cc, axes=1);
    cc_m = np.mean(cc,0)
    
    BiDiPhase = -(np.argmax(cc_m[-maxphase + Lx // 2 : maxphase+1 + Lx // 2]) - maxphase)

    return BiDiPhase, maxphase, sigma

def shiftBiDi(BiDiPhase, image, ixs=None):
    [Ly, Lx]=image.shape 
    shifted_image=copy.copy(image)

    if BiDiPhase>0:
        shifted_image[1::2, BiDiPhase:] = image[1::2, 0:-BiDiPhase]
    elif BiDiPhase<0:
        shifted_image[1::2, 0:BiDiPhase]  = image[1::2, -BiDiPhase:]
    else:
        pass
        
        
    # if bidiphase > 0:
    #     shifted_image[ 1::2, BiDiPhase:] = image[ 1::2, :-BiDiPhase]
    # else :
    #     shifted_image[ 1::2, :BiDiPhase] = image[ 1::2, -BiDiPhase:]
    # return frames    
        

    return shifted_image








