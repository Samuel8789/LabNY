# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 11:40:08 2021

@author: sp3660
"""
from caiman.source_extraction import cnmf as cnmf
import numpy as np
import caiman as cm
from caiman.motion_correction import MotionCorrect
import sys
# sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/AllFunctions')
# sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/ProcessingScripts')
# sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/MainClasses')
from lab_NY.AllFunctions.kalman_stack_filter import kalman_stack_filter

def caimanMotionCorrect(dataset_object):
    
    dataset_path=dataset_object.dataset_full_file_path
    dataset_metadata=dataset_object.associated_aquisiton.full_metadata 
    meta_dict=dataset_metadata.imaging_metadata
    aver=int(meta_dict[0]['RasterAveraging'])
    frameperiod=float(meta_dict[0]['framePeriod'])
    fr=1/(frameperiod*aver)
    
    fr = fr  # frame rate (Hz)
    mot_corr = True  # flag for online motion correction
    pw_rigid = False  # flag for pw-rigid motion correction (slower but potentially more accurate)
    max_shifts_online = np.ceil(10.).astype('int')  # maximum allowed shift during motion correction
    # set up some additional supporting parameters needed for the algorithm
    # (these are default values but can change depending on dataset properties)
    
    params_dict = {'fnames': dataset_path,
                   'fr': fr,
                   'motion_correct': mot_corr,
                   'init_method': 'bare',
                   'normalize': True,
                   'max_shifts_online': max_shifts_online,
                   'pw_rigid': pw_rigid,
                   'dist_shape_update': True,
                   'min_num_trial': 10,
                   }
    opts = cnmf.params.CNMFParams(params_dict=params_dict)
 #%%  
    try:
        cm.stop_server()  # stop it if it was running        
    except():
        pass
    
    if 'dview' in locals():
        cm.stop_server(dview=dview)
        
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None,
                                  single_thread=False)

    #%% Motion correction option  
    # mc = MotionCorrect(dataset_path, dview=dview, **opts.get_group('motion'))
    mc = MotionCorrect(dataset_path, **opts.get_group('motion'))

    # note that the file is not loaded in memory
    
    # %% Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)
    #%%
    try:
        cm.stop_server()  # stop it if it was running   
        dview.terminate()

    except():
        pass
    
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    
    #%%
  #%%  kalman and saving
    
    mot_correct_dataset_path=cm.load(mc.mmap_file)

    motcorrect_kalman_array=kalman_stack_filter(mot_correct_dataset_path)

    motcorrect_kalman_mov=cm.movie(motcorrect_kalman_array, fr=fr)
    
    motcorrect_kalman_mov.save(dataset_path[0:-5]+'motcor_KALMAN.h5')

    return motcorrect_kalman_mov