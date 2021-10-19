# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 22:14:17 2021

@author: sp3660
"""

import os
import caiman as cm
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import shutil

from save_imagej_hdf5_tiff import save_imagej_hdf5



class SummaryImages:
    
    def __init__(self, image_sequence_path=None, dataset_object=None):
        
        self.image_sequence_path=image_sequence_path
        self.eliminate_caiman_extra_from_mmap()          
        self.do_projections()
        self.save_projections()
        self.plotting()
        
    def eliminate_caiman_extra_from_mmap(self) :   

        self.mmap_directory, caiman_filename=os.path.split(self.image_sequence_path) 
        if caiman_filename.find('_d1_')!=-1:
            self.good_filename=caiman_filename[:caiman_filename.find('_d1_')]   
            self.caiman_extra=caiman_filename[caiman_filename.find('_d1_'):caiman_filename.find('_mmap')-4]   
        else:
            self.good_filename=os.path.splitext(caiman_filename)[0]
     
    def read_projections(self):
        self.projection_paths_dic={'average_projection_path':'.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'average_projection']),'tiff']),
                             'max_projection_path':'.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'max_projection']),'tiff']),
                             'std_projection_path':'.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'std_projection']),'tiff']),
                             'local_correlations_path':'.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'local_correlations']),'tiff']),
                             }  
        self.projection_dic={'average_projection_path':'',
                             'max_projection_path':'',
                             'std_projection_path':'',
                             'local_correlations_path':'',
                             }   
        
    def do_projections(self): 
        
        self.read_projections()
       
        if not all(os.path.isfile(x) for x in self.projection_paths_dic.values()):

            if '.tiff' in self.image_sequence_path:
                with tifffile.TiffFile(self.image_sequence_path) as tffl:
                     input_arr = tffl.asarray()
                     rawmov=cm.movie(input_arr.astype(np.uint16))
             
                del(input_arr)    
            
            else :
                rawmov=cm.load(self.image_sequence_path)
                
            for  k, image in self.projection_paths_dic.items():
    
                if not os.path.isfile(self.projection_paths_dic[k]):  
                    if 'average' in k:
                        self.projection_dic[k]=rawmov.mean(axis=0)
                    elif 'max' in k:
                        self.projection_dic[k]=rawmov.max(axis=0)
                    elif 'std' in k:
                        self.projection_dic[k]=rawmov.std(axis=0)
                    elif 'correlations' in k:
                        self.projection_dic[k]=cm.movie(rawmov.local_correlations())
                        
        else:
            self.load_projections()
        
    def save_projections(self):    
        for k,v in self.projection_dic.items():
            if not os.path.isfile(self.projection_paths_dic[k]):
                save_imagej_hdf5(v, os.path.splitext(self.projection_paths_dic[k])[0] , '.tiff')

  
    def load_projections(self):    
        self.read_projections()
        for key, x in  self.projection_paths_dic.items():
            with tifffile.TiffFile(x) as tffl:
                 self.projection_dic[key] = tffl.asarray()

        
    def plotting(self):

        for key, x in  self.projection_dic.items():
            plt.figure()
            plt.imshow(x)
            
            
    def copy_results_to_new_directory(self, new_directory):   

        for k, v in self.projection_dic.items():
            if os.path.isfile(self.projection_paths_dic[k]):
                shutil.copyfile(self.projection_paths_dic[k],   os.path.join(new_directory, os.path.split(self.projection_paths_dic[k])[1]))
         

     
if __name__ == "__main__":
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset'
    dump='\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp'
    kalman_path=os.path.join(temporary_path,'211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000_Shifted_Movie_MC_kalman.tiff')

    dataset_full_file_mmap_path=os.path.join(temporary_path,'211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000_Shifted_Movie_MC_movie_d1_256_d2_256_d3_1_order_F_frames_62499_.mmap')

    SumImages_kalman = SummaryImages(image_sequence_path=kalman_path)
    SumImages_raw = SummaryImages(image_sequence_path=dataset_full_file_mmap_path)

                       
    
    
