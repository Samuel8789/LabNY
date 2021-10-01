# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 12:45:27 2021

@author: sp3660
"""


import os
import caiman as cm
import numpy as np
import tifffile

from .kalman_stack_filter import kalman_stack_filter
from .save_imagej_hdf5_tiff import save_imagej_hdf5
from ..data_pre_processing.correct_bidi_movie import correct_bidi_movie
#%%'

def create_motcorrected_kalman(mmap_path, save_MC_mmap=False, correct=None):
    pass
    rawmovpath=mmap_path
    motion_corrected_fullpath=os.path.splitext(rawmovpath)[0]+'MCkalman.tiff'
    if correct or not os.path.isfile(motion_corrected_fullpath):

        filename=os.path.split(rawmovpath)[1]
        good_filename=filename[:filename.find('_d1_')]
        temporarypath=os.path.join('\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp', good_filename)
        caiman_extra=filename[filename.find('_d1_'):filename.find('_mmap')-4]
        
        temporary_bidi_path, bidiphases=correct_bidi_movie(rawmovpath, temporarypath, caiman_extra)  
        # temporary_bidi_path, bidiphases=correct_bidi_movie(rawmovpath)    
        
        rawmovMC=cm.motion_correction.MotionCorrect(temporary_bidi_path, max_deviation_rigid=10)

        # rawmovMC=cm.motion_correction.MotionCorrect(rawmovpath, max_deviation_rigid=10)
        print('Doing motion correction')
        rawmovMC.motion_correct(save_movie=False)
        corrected=rawmovMC.apply_shifts_movie(temporary_bidi_path)
        list_of_files = os.listdir('\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp')
        full_path = ['\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp\{0}'.format(x) for x in list_of_files]
        oldest_file = min(full_path, key=os.path.getctime)
        if len(list_of_files)>5 and oldest_file !=  temporary_bidi_path:
                os.remove(oldest_file)
        

        print('Doing kalman on '+ os.path.split(rawmovpath)[1]+'motioncorrected')
        dataset_kalman_array=kalman_stack_filter(corrected)
        dataset_kalman_array_changed_type=dataset_kalman_array.astype(np.uint16)
        dataset_kalman_caiman_movie=cm.movie(dataset_kalman_array_changed_type, fr=300,start_time=0,file_name=None, meta_data=None)
    
        
        save_imagej_hdf5(dataset_kalman_caiman_movie, os.path.splitext(rawmovpath)[0]+'MCkalman', '.tiff', )
        
        if save_MC_mmap:
            save_imagej_hdf5(dataset_kalman_caiman_movie, os.path.splitext(rawmovpath)[0]+'MC', '.mmap',)
        
      
    return motion_corrected_fullpath, bidiphases


if __name__ == '__main__':    
    mmap_path='\\\\?\\'+r'D:\Projects\LabNY\Full_Mice_Data\Mice_Projects\Tigre_Controls\VGC\Ai162\SPJP\imaging\20210522\data aquisitions\FOV_1\210522_SPJP_HabituationControlday2_940_WideFilter_mediumplane-000\planes\Plane1\Green\210522_SPJP_HabituationControlday2_940_WideFilter_mediumplane-000_d1_256_d2_256_d3_1_order_F_frames_42376_.mmap'
    create_motcorrected_kalman(mmap_path)
    # [rawmov, rawmovMCmov, dataset_kalman_caiman_movie_raw, dataset_kalman_caiman_movie, rawmovMC]=create_motcorrected_kalman(mmap_path)
   
    
   