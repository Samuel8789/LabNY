# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:21:42 2021

@author: sp3660
"""

import numpy as np
import os

from OnACID import run_on_acid
from metadata import Metadata
from motionCorrectedKalman import MotionCorrectedKalman
from summaryImages import SummaryImages

class CaimanExtraction():
    
    def __init__(self, bidishifted_movie_path, metdata_file_path, temporary_path=None):
        
        self.temporary_path=temporary_path
        self.bidishifted_movie_path=bidishifted_movie_path
        self.eliminate_caiman_extra_from_mmap()
        
        
        if os.path.isfile(os.path.splitext(self.bidishifted_movie_path)[0]+ 'cnmf_results.hdf5'):
            print('Already done')
            print(self.bidishifted_movie_path)
            print(os.path.splitext(self.bidishifted_movie_path)[0]+ 'cnmf_results.hdf5')
        else:            
            
            self.metadata=Metadata(aq_metadataPath=metadata_file_path)
            datasetmeta=self.metadata.imaging_metadata
            objective=self.metadata.imaging_metadata[0]['Objective']
            if objective=='MBL Olympus 20x':
                self.halfsize=2.5
            elif objective=='Olympus fat 25xb':
               self.halfsize=5
           
               if '20x' in self.bidishifted_movie_path:
                    self.halfsize=2.5
    
    
       #%% 
            
            self.framePeriod=float(datasetmeta[0]['framePeriod'])
            self.rastersPerFrame=int(datasetmeta[0]['RasterAveraging'])
            self.number_planes=datasetmeta[1]['PlaneNumber']
            if self.number_planes=='Single':
                self.number_planes=1
                self.volume_period=1/(self.framePeriod*self.rastersPerFrame)
            
            if self.number_planes==3:
                self.etl_frame_period=float(datasetmeta[2][0]['TopPlane']['framePeriod'])
                self.plane_period=float(self.framePeriod*self.rastersPerFrame)
                self.volume_period=1/(self.etl_frame_period*self.number_planes)
            
                        
           
            
            
    #%%
            self.set_caiman_parameters()
            self.apply_caiman()
            
            
            
    def eliminate_caiman_extra_from_mmap(self) :   

          self.mmap_directory, caiman_filename=os.path.split(self.bidishifted_movie_path) 
          if caiman_filename.find('_d1_')!=-1:
              self.good_filename=caiman_filename[:caiman_filename.find('_d1_')]   
              self.caiman_extra=caiman_filename[caiman_filename.find('_d1_'):caiman_filename.find('_mmap')-4]   
          else:
              self.good_filename=os.path.splitext(caiman_filename)[0]        

    # %%   Set up some parameters
    def set_caiman_parameters(self):
  
        fr = 1/self.volume_period  # frame rate (Hz) 3pl + 4ms = 15.5455
        decay_time = 0.5# 2 for s 0.5 for f # approximate length of transient event in seconds
        gSig = (self.halfsize,self.halfsize)  # expected half size of neurons
        p = 2  # order of AR indicator dynamics
        min_SNR = 1.5   # minimum SNR for accepting new components
        ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
        gnb = 2  # number of background components
        gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int')) # recompute gSig if downsampling is involved
        mot_corr = True  # flag for online motion correction
        pw_rigid = False  # flag for pw-rigid motion correction (slower but potentially more accurate)
        max_shifts_online = 10  # maximum allowed shift during motion correction
        sniper_mode = True  # use a CNN to detect new neurons (o/w space correlation)
        rval_thr = 0.8  # soace correlation threshold for candidate components
        # set up some additional supporting parameters needed for the algorithm
        # (these are default values but can change depending on dataset properties)
        init_batch = 1000  # number of frames for initialization (presumably from the first file)
        K = 2  # initial number of components
        epochs = 5 # number of passes over the data
        show_movie = False # show the movie as the data gets processed
        merge_thr = 0.8
        use_cnn = True  # use the CNN classifier
        min_cnn_thr = 0.90  # if cnn classifier predicts below this value, reject
        cnn_lowest = 0.3  # neurons with cnn probability lowe

        
        
        self.dataset_caiman_parameters = {'fnames': self.bidishifted_movie_path,
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
        if os.path.isfile(os.path.splitext(self.bidishifted_movie_path)[0] + 'cnmf_results.hdf5'):
            return
        else:  
         self.cnm_object=run_on_acid(self,  self.dataset_caiman_parameters)
         
        
        

if __name__ == "__main__":
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIK3planeallen\Plane3'
    dump='\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp'
    # image_sequence_path=os.path.join(temporary_path,'210930_SPKI_2mintestvideo_920_50024_narrow_without-000_Shifted_Movie_MC_kalman.tiff')
    metadata_file_path=os.path.join(temporary_path,'211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000.xml')
    dataset_full_file_mmap_path=os.path.join(temporary_path,'211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_62499_.mmap')
    
    CaimanExtr = CaimanExtraction(dataset_full_file_mmap_path, metadata_file_path, temporary_path=temporary_path)
    cnm=CaimanExtr.cnm_object




    
                
