# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 10:20:25 2021

@author: sp3660
"""

import caiman as cm
import os 
import glob
import numpy as np
import tifffile
import matplotlib.pyplot as plt

from create_motcorrected_kalman import create_motcorrected_kalman
from save_imagej_hdf5_tiff import save_imagej_hdf5
from bidicorrect_image import shiftBiDi, biDiPhaseOffsets

dataset_image_sequence_path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20210930\Mice\SPKI\TestAcquisitions\Aq_1\210930_SPKI_2mintestvideo_920_50024_narrow_without-000\Ch2Green\plane1'
dataset_mmap_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset'

class ImageSequenceDatasetStandAlone:
    
    def __init__(self, dataset_image_sequence_path=None, dataset_mmap_path=None):
        
        self.dataset_mmap_path=dataset_mmap_path
        self.dataset_name=os.path.split(os.path.split(os.path.split(dataset_image_sequence_path)[0])[0])[1]
        self.dataset_image_sequence_path=dataset_image_sequence_path
        self.only_read_dataset()
    
  
        if self.dataset_image_sequence_path:           
            if not os.path.isfile(self.dataset_full_file_path):
                if not os.path.isfile(os.path.join(self.dataset_mmap_path, self.dataset_name) + '.mmap' ):
                    self.load_dataset_from_image_sequence()
                    self.calculate_bidi_shifts()
                    
                    self.save_dataset_as_mmap()
                    self.save_bidi_sifts(self)


    def only_read_dataset(self):    
        self.image_sequence=None  
        if glob.glob(self.dataset_mmap_path+'\\**.mmap'):
            self.image_sequence_file_path=glob.glob(self.dataset_mmap_path+'\\**.mmap')[0] 
        else:
            self.image_sequence_file_path=''
            
        self.dataset_full_file_path=self.image_sequence_file_path
        self.motion_corrected_fullpath=os.path.splitext(self.dataset_full_file_path)[0]+'MCkalman.tiff'
        self.read_projections()        

    def read_projections(self):
        self.path_to_project=self.dataset_full_file_path   
        if os.path.isfile(self.motion_corrected_fullpath):
            self.path_to_project= self.motion_corrected_fullpath
            
 
        self.projection_paths_dic={'average_projection_path':os.path.splitext(self.path_to_project)[0]+'_average_projection.tiff',
                             'max_projection_path':os.path.splitext(self.path_to_project)[0]+'_max_projection.tiff',
                             'std_projection_path':os.path.splitext(self.path_to_project)[0]+'_std_projection.tiff',
                             # 'local_correlations_path':os.path.splitext(self.path_to_project)[0]+'local_correlations.tiff',
                             }
        # print('projecting' + self.path_to_project)


        
    def load_dataset_from_image_sequence(self):
        image_sequence_files=os.listdir(self.dataset_image_sequence_path)
        image_sequence_paths= [os.path.join(self.dataset_image_sequence_path, image) for image in image_sequence_files]
        # print(self.dataset_name)
        self.image_sequence=cm.load(image_sequence_paths)
        
    def save_dataset_as_mmap(self):
        
        self.image_sequence.save(os.path.join(self.dataset_mmap_path, self.associated_aquisiton.aquisition_name) + '.mmap' ,to32=False)       
        self.only_read_dataset()
        # print(self.dataset_mmap_path)
        
    def load_dataset_from_mmap(self): 
        self.image_sequence=cm.load(self.dataset_full_file_path)
        return  self.image_sequence
    

        
        
    def create_mot_corrected_kalman_tiff(self, save_MC_mmap=False, correct=None):
        
        self.motion_corrected_fullpath, self.bidiphases=create_motcorrected_kalman(self.dataset_full_file_path, save_MC_mmap, correct=correct)
            
    def read_dataset_metadat_from_database(self):
        print('in progress')
        
    def do_projections(self): 
        
        self.read_projections()
        
        all_projections= list(self.projection_paths_dic.values())      
        
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
                save_imagej_hdf5(self.average_projection, os.path.splitext(self.path_to_project)[0]+'_average_projection', '.tiff', )
            else:
                os.remove(self.projection_paths_dic['average_projection_path'])
                self.average_projection=rawmov.mean(axis=0)
                save_imagej_hdf5(self.average_projection, os.path.splitext(self.path_to_project)[0]+'_average_projection', '.tiff', )
                
    
            if not os.path.isfile(self.projection_paths_dic['max_projection_path']):
                self.max_projection=rawmov.max(axis=0)
                save_imagej_hdf5(self.max_projection, os.path.splitext(self.path_to_project)[0]+'_max_projection', '.tiff', )
            else:
                os.remove(self.projection_paths_dic['max_projection_path'])
                self.max_projection=rawmov.mean(axis=0)
                save_imagej_hdf5(self.max_projection, os.path.splitext(self.path_to_project)[0]+'_max_projection', '.tiff', )
    
            if not os.path.isfile(self.projection_paths_dic['std_projection_path']):
                self.std_projection=rawmov.std(axis=0)
                save_imagej_hdf5(self.std_projection, os.path.splitext(self.path_to_project)[0]+'_std_projection', '.tiff', )
            else:
                os.remove(self.projection_paths_dic['std_projection_path'])
                self.std_projection=rawmov.mean(axis=0)
                save_imagej_hdf5(self.std_projection, os.path.splitext(self.path_to_project)[0]+'_std_projection', '.tiff', )
    
            # if not os.path.isfile(self.projection_paths_dic['local_correlations_path']):  
            #     self.local_correlations=rawmov.local_correlations()
            #     array_sum = np.sum(self.local_correlations)
            #     array_has_nan = np.isnan(array_sum)
            #     if not array_has_nan:
            #         save_imagej_hdf5(self.local_correlations, os.path.splitext(self.path_to_project)[0]+'local_correlations', '.tiff', )

  
    def load_projections(self):    
        self.read_projections()
        self.average_projection=plt.imread(self.projection_paths_dic['average_projection_path'])
        self.max_projection=plt.imread(self.projection_paths_dic['max_projection_path'])
        self.std_projection=plt.imread(self.projection_paths_dic['std_projection_path'])
        # self.local_correlations=plt.imread(self.projection_paths_dic['local_correlations_path'])
            
        
        
        
if __name__ == "__main__":
    
    app = ImageSequenceDatasetStandAlone(dataset_image_sequence_path=dataset_image_sequence_path, dataset_mmap_path=dataset_mmap_path )

        
        
 



        
       
        
        
        
        