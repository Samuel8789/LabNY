# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:44:15 2021

@author: sp3660
"""

import caiman as cm
import numpy as np

def save_imagej_hdf5(movie, file_name, extension):

    if len(movie.shape)==2:
       movie=np.expand_dims(movie, 0)
       movie.save(file_name + extension, order='F',imagej=False, to32=False, bigtiff=True)
    else:
      movie.save(file_name + extension,imagej=False, to32=False, bigtiff=True)
    
    
 
    
    