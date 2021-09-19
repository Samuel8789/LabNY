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
 
#%%'

def create_motcorrected_kalman(mmap_path, save_MC_mmap=False):
    pass
    mmap_path=mmap_path
    rawmovpath=mmap_path
    motion_corrected_fullpath=os.path.splitext(rawmovpath)[0]+'MCkalman.tiff'
    if not os.path.isfile(motion_corrected_fullpath):
        rawmov=cm.load(rawmovpath)
        
        # CHECK IF ALREADY EXIST
        # if 
        
        
        
        # TO ADD AN OPTION TO CHECK LNENGTH OF VIDEO
    
        
        # if 'dview' in locals():
        #     cm.stop_server(dview=dview)
        #     dview.terminate()
        # c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None,
        #                                     single_thread=False)
          
        # rawmovMC=cm.motion_correction.MotionCorrect(rawmovpath, dview=dview,max_deviation_rigid=10)
        rawmovMC=cm.motion_correction.MotionCorrect(rawmovpath, max_deviation_rigid=10)
        print('Doing motion correction')
        rawmovMC.motion_correct(save_movie=False)
        corrected=rawmovMC.apply_shifts_movie(mmap_path)
        
        
        # rawmovMC.motion_correct_rigid(save_movie=True)
        # rawmovMC.motion_correct_pwrigid(save_movie=True)
    
        # cm.stop_server(dview=dview)
        # dview.terminate()
        
        # rawmovMCpath=os.path.splitext(rawmovpath)[0]  +  '._rig_'  +   os.path.split(mmap_path)[1][os.path.split(mmap_path)[1].find('_d1_'):]
        
        # rawmovMCmov=cm.load(rawmovpath)
        # rawmovMCpath.play()
        print('Doing kalman on '+ os.path.split(rawmovpath)[1]+'motioncorrected')
        dataset_kalman_array=kalman_stack_filter(corrected)
        dataset_kalman_array_changed_type=dataset_kalman_array.astype(np.uint16)
        dataset_kalman_caiman_movie=cm.movie(dataset_kalman_array_changed_type, fr=300,start_time=0,file_name=None, meta_data=None)
        
        
        # rawmov.play(fr=500,gain=0.2)
        # rawmovMCpath.play(fr=500,gain=0.2)
        # dataset_kalman_caiman_movie_raw.play(fr=500,gain=0.2)
        # dataset_kalman_caiman_movie.play(fr=500,gain=0.2)
        
        # return rawmov, rawmovMCmov, dataset_kalman_caiman_movie_raw, dataset_kalman_caiman_movie, rawmovMC
        
        
        save_imagej_hdf5(dataset_kalman_caiman_movie, os.path.splitext(rawmovpath)[0]+'MCkalman', '.tiff', )
        
        if save_MC_mmap:
            save_imagej_hdf5(dataset_kalman_caiman_movie, os.path.splitext(rawmovpath)[0]+'MC', '.mmap',)
        
        # os.remove(rawmovMCpath)
        # cm.stop_server(dview=dview)
        # if 'dview' in locals():
        #     cm.stop_server(dview=dview)
        #     dview.terminate()
    
    
        # rawmovMCpath=os.path.splitext(rawmovpath)[0]  +  '._rig_'  +   os.path.split(mmap_path)[1][os.path.split(mmap_path)[1].find('_d1_'):]
        # rawmovMC=cm.load(rawmovpath)
        # # rawmovMCpath.play()
        # dataset_kalman_array=kalman_stack_filter(rawmovMC)
        # dataset_kalman_array_changed_type=dataset_kalman_array.astype(np.uint16)
        # dataset_kalman_caiman_movie=cm.movie(dataset_kalman_array_changed_type, fr=300,start_time=0,file_name=None, meta_data=None) 
        # dataset_kalman_caiman_movie.play(fr=500,gain=0.2)
    
        # save_imagej_hdf5(dataset_kalman_caiman_movie, mmap_path+'kalmanraw', '.tiff',)
    return motion_corrected_fullpath


if __name__ == '__main__':    
    mmap_path='\\\\?\\'+r'D:\Projects\LabNY\Full_Mice_Data\Mice_Projects\Tigre_Controls\VGC\Ai162\SPJP\imaging\20210522\data aquisitions\FOV_1\210522_SPJP_HabituationControlday2_940_WideFilter_mediumplane-000\planes\Plane1\Green\210522_SPJP_HabituationControlday2_940_WideFilter_mediumplane-000_d1_256_d2_256_d3_1_order_F_frames_42376_.mmap'
    create_motcorrected_kalman(mmap_path)
    # [rawmov, rawmovMCmov, dataset_kalman_caiman_movie_raw, dataset_kalman_caiman_movie, rawmovMC]=create_motcorrected_kalman(mmap_path)
   
    
   