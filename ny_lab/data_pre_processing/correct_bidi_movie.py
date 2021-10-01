# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 19:57:36 2021

@author: sp3660
"""

import caiman as cm
import numpy as np
from .bidicorrect_image import shiftBiDi, biDiPhaseOffsets
import os 
import glob
def correct_bidi_movie(mmap_path, temporary_bidi_path, caiman_extra):
    rawmov=cm.load(mmap_path)
    shifted_images=np.zeros(rawmov.shape).astype('float32')
    bidiphases=[]
    for i in range(rawmov.shape[0]):
        BiDiPhase=biDiPhaseOffsets(rawmov[i,:,:])
        bidiphases.append(BiDiPhase)
        shifted_images[i,:,:]=shiftBiDi(BiDiPhase, rawmov[i,:,:])
    shifted_movie=cm.movie(shifted_images)
    del(shifted_images)
    temporary_bidi_path_save= temporary_bidi_path+'.mmap'       
    shifted_movie.save(temporary_bidi_path_save ,to32=False)  
    caiman_temporary_bidi=temporary_bidi_path+caiman_extra+'.mmap'  
    
    
    return caiman_temporary_bidi, bidiphases