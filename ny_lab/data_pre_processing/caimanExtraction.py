# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:18:55 2021

@author: sp3660
"""

import numpy as np
import shutil
import os

from  ..AllFunctions.OnACID_YS import run_on_acid



class CaimanExtraction():
    
    def __init__(self, dataset_object):
        
        
        self.dataset_object=dataset_object
        self.dataset_mouse_path=self.dataset_object.dataset_full_file_path
        self.filename=os.path.split(self.dataset_mouse_path)[1]
        if os.path.isfile(self.dataset_mouse_path[0:-5] + 'cnmf_results.hdf5'):
            print('Already done')
            print(self.dataset_mouse_path)
            print(self.dataset_mouse_path[0:-5] + 'cnmf_results.hdf5')
        else:            
            self.temporarypath=os.path.join('\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp',self.filename)
            shutil.copyfile(self.dataset_mouse_path, self.temporarypath)
            list_of_files = os.listdir('\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp')
            full_path = ['\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp\{0}'.format(x) for x in list_of_files]
            oldest_file = min(full_path, key=os.path.getctime)
            if len(list_of_files)>5 and oldest_file !=  self.temporarypath:
                    os.remove(oldest_file)
            
            query="""
                SELECT *
                FROM Imaging_table
                WHERE ImagingFilename=?
            """
            params=(dataset_object.associated_aquisiton.aquisition_name,)
            
            datasetmeta=dataset_object.associated_aquisiton.mouse_object.Database_ref.arbitrary_query_to_df(query, params)
            if datasetmeta.Objective[0]=='MBL Olympus 20x':
                self.halfsize=2.5
            elif datasetmeta.Objective[0]=='Olympus fat 25xb':
               self.halfsize=5
    
    
    
       #%% 
            
            # self.framePeriod=float(datasetmeta.FramePeriod)
            # self.etl_frame_period=float(datasetmeta.InterFramePeriod)
            # self.rastersPerFrame=int(datasetmeta.FrameAveraging)
            # self.plane_period=float(self.framePeriod*self.rastersPerFrame)
            # self.number_planes=self.dataset_metadata[1]['Plane Number']
            # if self.number_planes=='Single':
            #     self.number_planes=1
          
            self.volume_period=datasetmeta.FinalVolumePeriod[0]
            
            
    #%%
            self.set_caiman_parameters()
            

    # %%   Set up some parameters
    def set_caiman_parameters(self):
  
        fr = 1/self.volume_period  # frame rate (Hz) 3pl + 4ms = 15.5455
        decay_time = 1# 2 for s 0.5 for f # approximate length of transient event in seconds
        gSig = (self.halfsize,self.halfsize)  # expected half size of neurons
        p = 2  # order of AR indicator dynamics
        min_SNR = 1.5   # minimum SNR for accepting new components
        ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
        gnb = 2  # number of background components
        gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int')) # recompute gSig if downsampling is involved
        mot_corr = True  # flag for online motion correction
        pw_rigid = True  # flag for pw-rigid motion correction (slower but potentially more accurate)
        max_shifts_online = 10  # maximum allowed shift during motion correction
        sniper_mode = True  # use a CNN to detect new neurons (o/w space correlation)
        rval_thr = 0.8  # soace correlation threshold for candidate components
        # set up some additional supporting parameters needed for the algorithm
        # (these are default values but can change depending on dataset properties)
        init_batch = 1000  # number of frames for initialization (presumably from the first file)
        K = 2  # initial number of components
        epochs = 3 # number of passes over the data
        show_movie = False # show the movie as the data gets processed
        merge_thr = 0.8
        use_cnn = True  # use the CNN classifier
        min_cnn_thr = 0.90  # if cnn classifier predicts below this value, reject
        cnn_lowest = 0.3  # neurons with cnn probability lowe

        
        
        self.dataset_caiman_parameters = {'fnames': self.temporarypath,
                                           'fr': fr,
                                           'decay_time': decay_time,
                                           'gSig': gSig,
                                           'p': p,
                                           'min_SNR': min_SNR,
                                           'rval_thr': rval_thr,
                                           'merge_thr': merge_thr,
                                           'ds_factor': ds_factor,
                                           'nb': gnb,
                                           'motion_correct': mot_corr,
                                           'init_batch': init_batch,
                                           'init_method': 'bare',
                                           'normalize': True,
                                           'sniper_mode': sniper_mode,
                                           'K': K,
                                           'max_shifts_online': max_shifts_online,
                                           'pw_rigid': pw_rigid,
                                           'dist_shape_update': True,
                                           'min_num_trial': 10,
                                           'show_movie': show_movie,
                                           'epochs':epochs,
                                           'use_cnn': use_cnn,
                                           'min_cnn_thr': min_cnn_thr,
                                           'cnn_lowest': cnn_lowest
                                            }
        
        
  
        
    def apply_caiman(self):
        if os.path.isfile(self.dataset_mouse_path[0:-5] + 'cnmf_results.hdf5'):
            return
        else:  
         run_on_acid(self,  self.dataset_caiman_parameters, self.dataset_object)
         
         # os.remove(self.temporarypath)
        
        






    
                
