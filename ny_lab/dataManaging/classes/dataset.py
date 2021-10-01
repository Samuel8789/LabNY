# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:46:42 2021

@author: sp3660
"""
import caiman as cm
import os 
import glob
import numpy as np
import tifffile
import matplotlib.pyplot as plt

from ...AllFunctions.create_motcorrected_kalman import create_motcorrected_kalman, save_imagej_hdf5
from ...data_pre_processing.bidicorrect_image import shiftBiDi, biDiPhaseOffsets


class ImageSequenceDataset:
    
    def __init__(self, aquisition_object, dataset_name=None, selected_dataset_raw_path=None, selected_dataset_mmap_path=None):
        
        self.associated_aquisiton=aquisition_object        
        self.selected_dataset_mmap_path=selected_dataset_mmap_path
        self.dataset_name=dataset_name
    
  
        if selected_dataset_raw_path:
            self.selected_dataset_raw_path=selected_dataset_raw_path
            self.only_read_dataset()
            
            if not os.path.isfile(self.dataset_full_file_path):
                if not os.path.isfile(os.path.join(self.selected_dataset_mmap_path, self.associated_aquisiton.aquisition_name) + '.mmap' ):
                    self.load_dataset_from_image_sequence()
                    self.bidi_shift=biDiPhaseOffsets()

                    # print(self.associated_aquisiton)
                    # print(self.dataset_name)
                    self.save_dataset_as_mmap()
                    
                
        else:
            # print('Reading Exiting Datasets')
            self.only_read_dataset()
            
            
            

        
        
    def load_dataset_from_image_sequence(self):
        image_sequence_files=os.listdir(self.selected_dataset_raw_path)
        image_sequence_paths= [os.path.join(self.selected_dataset_raw_path, image) for image in image_sequence_files]
        # print(self.dataset_name)
        self.image_sequence=cm.load(image_sequence_paths)
        
    def save_dataset_as_mmap(self):
        
        self.image_sequence.save(os.path.join(self.selected_dataset_mmap_path, self.associated_aquisiton.aquisition_name) + '.mmap' ,to32=False)       
        self.only_read_dataset()
        # print(self.selected_dataset_mmap_path)
        
    def load_dataset_from_mmap(self): 
        self.image_sequence=cm.load(self.dataset_full_file_path)
        return  self.image_sequence
    
    def only_read_dataset(self):
    
        self.image_sequence=None
        if glob.glob(self.selected_dataset_mmap_path+'\\**.mmap'):
            self.image_sequence_file_path=glob.glob(self.selected_dataset_mmap_path+'\\**.mmap')[0] 
        else:
            self.image_sequence_file_path=''
            
        self.dataset_full_file_path=self.image_sequence_file_path
        # create_motcorrected_kalman(self.dataset_full_file_path)
        # add the metadata for that dataset
        self.motion_corrected_fullpath=os.path.splitext(self.dataset_full_file_path)[0]+'MCkalman.tiff'
        self.read_projections()
        
        
    def create_mot_corrected_kalman_tiff(self, save_MC_mmap=False, correct=None):
        
        # print('moticorrected tiff '+self.dataset_name)
        self.motion_corrected_fullpath, self.bidiphases=create_motcorrected_kalman(self.dataset_full_file_path, save_MC_mmap, correct=correct)
            
    def read_dataset_metadat_from_database(self):
        print('in progress')
        
    def do_projections(self): 
        
        self.read_projections()
        
        all_projections= list(self.projection_paths_dic.values())[:-1]       
        
        if not all(os.path.isfile(x) for x in all_projections):
            
            if '.tiff' in self.path_to_project:
                with tifffile.TiffFile(self.path_to_project) as tffl:
                     input_arr = tffl.asarray()
                     rawmov=cm.movie(input_arr.astype(np.uint16))
             
                del(input_arr)    
            
            else:
                rawmov=cm.load(self.path_to_project)

            if not os.path.isfile(self.projection_paths_dic['average_projection_path']):
                self.average_projection=rawmov.mean(axis=0)
                save_imagej_hdf5(self.average_projection, os.path.splitext(self.path_to_project)[0]+'average_projection', '.tiff', )

            if not os.path.isfile(self.projection_paths_dic['max_projection_path']):
                self.max_projection=rawmov.max(axis=0)
                save_imagej_hdf5(self.max_projection, os.path.splitext(self.path_to_project)[0]+'max_projection', '.tiff', )

            if not os.path.isfile(self.projection_paths_dic['std_projection_path']):
                self.std_projection=rawmov.std(axis=0)
                save_imagej_hdf5(self.std_projection, os.path.splitext(self.path_to_project)[0]+'std_projection', '.tiff', )

            # if not os.path.isfile(self.projection_paths_dic['local_correlations_path']):  
            #     self.local_correlations=rawmov.local_correlations()
            #     array_sum = np.sum(self.local_correlations)
            #     array_has_nan = np.isnan(array_sum)
            #     if not array_has_nan:
            #         save_imagej_hdf5(self.local_correlations, os.path.splitext(self.path_to_project)[0]+'local_correlations', '.tiff', )

    def read_projections(self):
        self.path_to_project=self.dataset_full_file_path   
        if os.path.isfile(self.motion_corrected_fullpath):
            self.path_to_project= self.motion_corrected_fullpath
            
            
            
        self.projection_paths_dic={'average_projection_path':os.path.splitext(self.path_to_project)[0]+'average_projection.tiff',
                             'max_projection_path':os.path.splitext(self.path_to_project)[0]+'max_projection.tiff',
                             'std_projection_path':os.path.splitext(self.path_to_project)[0]+'std_projection.tiff',
                             # 'local_correlations_path':os.path.splitext(self.path_to_project)[0]+'local_correlations.tiff',
                             }
        # print('projecting' + self.path_to_project)

    def load_projections(self):    
        self.read_projections()
        self.average_projection=plt.imread(self.projection_paths_dic['average_projection_path'])
        self.max_projection=plt.imread(self.projection_paths_dic['max_projection_path'])
        self.std_projection=plt.imread(self.projection_paths_dic['std_projection_path'])
        # self.local_correlations=plt.imread(self.projection_paths_dic['local_correlations_path'])
            

        
        
 



        
       
        
        
        
        