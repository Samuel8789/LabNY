# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:19:14 2021

@author: sp3660
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile
try:
    if __IPYTHON__:
        # this is used for debugging purposes only.
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
import time

class CaimanResults():
    
    def __init__(self, hdf5_file_path, movie_path):
        
        
        self.movie_path=movie_path
        self.hdf5_file_path=hdf5_file_path
        self.cnm = cnmf.online_cnmf.OnACID(path=self.hdf5_file_path)

        
        if '.tiff' in self.movie_path:
            with tifffile.TiffFile(self.movie_path) as tffl:
                 input_arr = tffl.asarray()
                 self.image_sequence=cm.movie(input_arr.astype(np.uint16))
            del(input_arr)   
        elif '.mmap' in self.movie_path:
            self.image_sequence=cm.load(self.movie_path)
            

            
if __name__ == "__main__":
    
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIK3planeallen\Plane1'
    hdf5path=temporary_path+ r'\211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000_Shifted_Movie_MC_OnACID_cnmf_results.hdf5'
    movie_path=temporary_path+ r'\211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000_Shifted_Movie_MC_kalman.tiff'
    movie_path2=temporary_path+ r'\211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000_Shifted_Movie_MC_OnACID_d1_256_d2_256_d3_1_order_F_frames_62499_.mmap'
    
    
    results=CaimanResults(hdf5path, movie_path2)
    cnm=results.cnm
   

