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
import gc
import sys
import glob
import logging 
import sys
module_logger = logging.getLogger(__name__)

# from save_imagej_hdf5_tiff import save_imagej_hdf5
from .save_imagej_hdf5_tiff import save_imagej_hdf5
from ...functions.transform_path import transform_path




class SummaryImages:
    
    def __init__(self, image_sequence_path=None, dataset_object=None):
        self.projection_dic={}
        self.image_sequence_path=image_sequence_path
        self.dataset_object=dataset_object
        
        if  self.dataset_object and self.image_sequence_path:
            self.eliminate_caiman_extra_from_mmap()   
            self.read_projections()
            self.check_dir_for_projections()
            if not self.projections:
                self.do_projections()
                self.save_projections()
                self.unload_summary_images()

        else:
            self.check_dir_for_projections()
        
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
                             # 'local_correlations_path':'.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'local_correlations']),'tiff']),
                             }  
        self.projection_dic={'average_projection_path':np.array([False]),
                             'max_projection_path':np.array([False]),
                             'std_projection_path':np.array([False]),
                             # 'local_correlations_path':'',
                             }   
        
    def do_projections(self): 
        
        self.read_projections()
       
        if not any(os.path.isfile(x) for x in self.projection_paths_dic.values()):

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
                    # elif 'correlations' in k:
                    #     self.projection_dic[k]=cm.movie(rawmov.local_correlations())
                        
        else:
            self.load_projections()
        
    def save_projections(self):    
        for k,v in self.projection_dic.items():
            if not os.path.isfile(self.projection_paths_dic[k]):
                save_imagej_hdf5(v, os.path.splitext(self.projection_paths_dic[k])[0] , '.tiff')

  
    def load_projections(self):   
        if self.projection_paths_dic:
            for key, x in  self.projection_paths_dic.items():
                with tifffile.TiffFile(x) as tffl:
                     self.projection_dic[key] = tffl.asarray()
                 
    def unload_summary_images(self):
        try:
            if self.projection_dic:
                del self.projection_dic
                gc.collect()
                # sys.stdout.flush()
                self.projection_dic={}
        except:
            module_logger.exception('Problem unloading summary images' +   self.dataset_object.selected_dataset_mmap_path)
 
        
    def plotting(self):

        for key, x in  self.projection_dic.items():
            plt.figure()
            plt.imshow(x)
    
            
    def check_dir_for_projections(self):
        
        self.projections=glob.glob( self.dataset_object.selected_dataset_mmap_path+'\\**Movie**projection.tiff')
        self.custom_projections=glob.glob( self.dataset_object.selected_dataset_mmap_path+'\\**custom**projection.tiff')
        if self.custom_projections:
            self.projections=self.custom_projections
            


        projection_names={'average_projection','max_projection','std_projection'}
  
        self.projection_paths_dic={}
        for element in projection_names:
            for element2 in  self.projections:
                if element in element2:
                    self.projection_paths_dic[element+'_path']=element2
                    
                    
        self.projection_dic={'average_projection_path':'',
                             'max_projection_path':'',
                             'std_projection_path':'',
                             # 'local_correlations_path':'',
                             }   

  
            
    def copy_results_to_new_directory(self, new_directory):   

        for k, v in self.projection_dic.items():
            if os.path.isfile(self.projection_paths_dic[k]):
                shutil.copyfile(self.projection_paths_dic[k],   os.path.join(new_directory, os.path.split(self.projection_paths_dic[k])[1]))
         

     
if __name__ == "__main__":
    # filename='211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000'
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset'
    # kalman_path=os.path.join(temporary_path,'211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000_Shifted_Movie_MC_kalman.tiff')
    # dataset_full_file_mmap_path=os.path.join(temporary_path,filename+ '_Shifted_Movie_MC_movie_d1_256_d2_256_d3_1_order_F_frames_62499_.mmap')

    filename=r'211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\Plane3'
    kalman_path=os.path.join(temporary_path,r'211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_MC_kalman.tiff')
    dataset_full_file_mmap_path=os.path.join(temporary_path, filename+'_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_64416_.mmap')


    dump='\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp'
    SumImages_kalman = SummaryImages(image_sequence_path=kalman_path)
    SumImages_raw = SummaryImages(image_sequence_path=dataset_full_file_mmap_path)

                       
    
    
