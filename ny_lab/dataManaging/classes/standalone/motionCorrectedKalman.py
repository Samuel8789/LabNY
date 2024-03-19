# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 18:43:19 2021

@author: sp3660
"""
import gc
import sys
import os
import caiman as cm
import numpy as np
import pickle
import glob
import scipy.signal as sg
from PIL import Image
import matplotlib.pyplot as plt
import tifffile
import logging 
module_logger = logging.getLogger(__name__)

from caiman.motion_correction import apply_shift_online
# from kalman_stack_filter import kalman_stack_filter
# from save_imagej_hdf5_tiff import save_imagej_hdf5
# from bidiShiftManager import BidiShiftManager
try :
    from .kalman_stack_filter import kalman_stack_filter
    from .save_imagej_hdf5_tiff import save_imagej_hdf5
    from .bidiShiftManager import BidiShiftManager
    
except:
    from kalman_stack_filter import kalman_stack_filter
    from save_imagej_hdf5_tiff import save_imagej_hdf5
    from bidiShiftManager import BidiShiftManager


class MotionCorrectedKalman:
    
    def __init__(self,
                 dataset_image_sequence_path=None,  
                 shifted_mmap_path=None,
                 temporary_path=None,
                 dataset_object=None):
        
        self.dataset_image_sequence_path=dataset_image_sequence_path
        self.shifted_mmap_path=shifted_mmap_path
        self.temporary_path=temporary_path
        self.dataset_object=dataset_object
        
        
        self.MC_corrected=np.array([False])
        self.dataset_kalman_caiman_movie=np.array([False])

        # self.get_proper_filenames()
    
    
    
        if  self.dataset_object:
            self.temporary_path=self.dataset_object.selected_dataset_mmap_path
            self.get_proper_filenames()

            if self.shifted_mmap_path:
                self.build_output_filenames_if_not_there()
                
                if os.path.isfile(self.kalman_path):
                        #this is redoink kamlan if i have aded a new start end flah
                        if self.dataset_object.bidishift_object.start_end_flag:                        
                                
                            if self.kalman_path==self.kalman_full_path:
                                module_logger.info('doing new custom kalman')
                                self.MC_corrected=cm.load(self.shifted_mmap_path) 
                                self.do_kalman_filter()
                                self.do_temporal_gaussian_smoothing(100)
                                module_logger.info('saving kalman')
                    
                                self.save_MC_kalman_movie()
                                self.save_gaussian_smoothed_movie()
                                module_logger.info('rechecking files')
                    
                                self.check_MC_info_mmap_directory()
                                module_logger.info('removing unlcipeed')
                    
                                self.remove_unclipped_issue_shifted_kalmans()
                                self.unload_kalman_movie()
                        else:
                            module_logger.info('kalman already done')
        
                #this is the main do kalmna
                else:
                    self.MC_corrected=cm.load(self.shifted_mmap_path) 
                    module_logger.info('doing first kalman')
                    self.do_kalman_filter()
                    module_logger.info('doing gaussian100')

                    self.do_temporal_gaussian_smoothing(100)
                    module_logger.info('saving kalman')
                    self.save_MC_kalman_movie()
                    self.save_gaussian_smoothed_movie()
                    module_logger.info('rechecking files')
                    self.check_MC_info_mmap_directory()
                    module_logger.info('removing unlcipeed')
        
                    self.remove_unclipped_issue_shifted_kalmans()
                    self.unload_kalman_movie()
                          
            else:
                if self.kalman_path:
                    module_logger.info('kalman initialized not loaded')
                else:
                    self.check_MC_info_mmap_directory()
                    module_logger.info('no kalman in directory')


                
    
        
        
   #%% checking in folder     
    def check_MC_info_mmap_directory(self):
 
         self.kalman_custom_paths=[]
         self.kalman_full_paths=[]
         self.kalman_custom_path=None
         self.kalman_full_path=None
         self.kalman_path=None

         
         self.kalman_full_paths=glob.glob(self.temporary_path + os.sep+'**Movie_MC_OnACID_**MC_kalman.tiff')
         self.kalman_custom_paths=glob.glob(self.temporary_path + os.sep+'**end_MC_OnACID_**MC_kalman.tiff')

         if self.kalman_full_paths:
             self.kalman_full_path= self.kalman_full_paths[0]
         if self.kalman_custom_paths:
             self.kalman_custom_path= self.kalman_custom_paths[0]
         
         if self.kalman_custom_path:  
             self.kalman_path=self.kalman_custom_path
         elif self.kalman_full_path:
             self.kalman_path=self.kalman_full_path
             
             
         self.gauss_custom_paths=[]
         self.gauss_full_paths=[]
         self.gauss_custom_path=None
         self.gauss_full_path=None
         self.gauss_path=None

        
         self.gauss_full_paths=glob.glob(self.temporary_path + os.sep+'**Movie_MC_OnACID_**MC_smoothed_100ms.tiff')
         self.gauss_custom_paths=glob.glob(self.temporary_path + os.sep+'**end_MC_OnACID_**MC_smoothed_100ms.tiff')

         if self.gauss_full_paths:
            self.gauss_full_path= self.gauss_full_paths[0]
         if self.gauss_custom_paths:
            self.gauss_custom_path= self.gauss_custom_paths[0]
        
         if self.gauss_custom_path:  
            self.gauss_path=self.gauss_custom_path
         elif self.gauss_full_path:
            self.gauss_path=self.gauss_full_path

         
         
        
#%% names and info          
            
    def get_proper_filenames(self):         
        
        if self.dataset_object:
            self.eliminate_caiman_extra_from_mmap()
            self.check_MC_info_mmap_directory()

                
    def eliminate_caiman_extra_from_mmap(self) :  
        
        if self.dataset_object and self.shifted_mmap_path:
            self.mmap_directory, caiman_filename=os.path.split(self.shifted_mmap_path) 
            self.good_filename=caiman_filename[:caiman_filename.find('_d1_')]   

    def build_output_filenames_if_not_there(self):
 
        if not self.kalman_path:
            if self.temporary_path and not self.dataset_object.bidishift_object.start_end_flag:
                   self.kalman_path='_'.join([os.path.join(self.temporary_path,self.good_filename),'MC_kalman.tiff'])
        
            elif self.mmap_directory:
                   self.kalman_path='_'.join([os.path.join(self.mmap_directory,self.good_filename),'MC_kalman.tiff'])   
                   
        if not self.gauss_path:
            if self.temporary_path and not self.dataset_object.bidishift_object.start_end_flag:
                   self.gauss_path='_'.join([os.path.join(self.temporary_path,self.good_filename),'MC_smoothed_100ms.tiff'])
        
            elif self.mmap_directory:
                   self.gauss_path='_'.join([os.path.join(self.mmap_directory,self.good_filename),'MC_smoothed_100ms.tiff'])   
   
        
#%% loading from folder        

    # def load_dataset_from_image_sequence(self):
    #     image_sequence_files=os.listdir(self.dataset_image_sequence_path)
    #     image_sequence_paths= [os.path.join(self.dataset_image_sequence_path, image) for image in image_sequence_files]
    #     self.image_sequence=cm.load(image_sequence_paths)
        
    # def load_dataset_from_mmap(self): 
    #     if self.dataset_full_file_mmap_path:
    #         self.image_sequence=cm.load(self.dataset_full_file_mmap_path)
    #     elif self.shifted_mmap_path:
    #         self.image_sequence=cm.load(self.shifted_mmap_path)


    def load_mc_kalman_tiff(self):
        if self.kalman_path:
            if os.path.isfile(self.kalman_path):
    
                module_logger.info('loading MC_kalman')
    
                with tifffile.TiffFile(self.kalman_path) as tffl:
                     input_arr = tffl.asarray()
                     self.dataset_kalman_caiman_movie=cm.movie(input_arr.astype(np.uint16))
                     # self.MC_corrected=self.dataset_kalman_caiman_movie
                del(input_arr)    
    
            module_logger.info('MC_kalman loaded')       
        else:
            module_logger.info('no kalman loaded')    
            
    def load_mc_gauss_tiff(self):
        if self.gauss_path:
            if os.path.isfile(self.gauss_path):
    
                module_logger.info('loading MC_gayss')
    
                with tifffile.TiffFile(self.gauss_path) as tffl:
                     input_arr = tffl.asarray()
                     self.smoothed_motion_corrected=cm.movie(input_arr.astype(np.uint16))
                     # self.MC_corrected=self.dataset_kalman_caiman_movie
                del(input_arr)    
    
            module_logger.info('MC_gauss loaded')       
        else:
            module_logger.info('no kalman loaded')   

 #%% processing   

    def do_kalman_filter(self):
        if not self.dataset_kalman_caiman_movie.any() and not os.path.isfile(self.kalman_path):
            module_logger.info('doing kalman')
            dataset_kalman_array=kalman_stack_filter(self.MC_corrected)
            dataset_kalman_array_changed_type=dataset_kalman_array.astype(np.uint16)
            self.dataset_kalman_caiman_movie=cm.movie(dataset_kalman_array_changed_type, fr=300,start_time=0,file_name=None, meta_data=None)
        else:
            module_logger.info('kalman already done')

    def do_temporal_gaussian_smoothing(self, sigma):
        
        if self.dataset_object.metadata.translated_imaging_metadata:
            frame=self.dataset_object.metadata.translated_imaging_metadata['FinalFrequency']
        else:
            frame=10
        sigma=100#ms

        def gaussian_smooth_kernel_convolution(signal, fr, sigma):
            dt = 1000/fr
            sigma_frames = sigma/dt
            # make kernel
            kernel_half_size = int(np.ceil(np.sqrt(-np.log(0.05)*2*sigma_frames**2)))
            gaus_win =list(range( -kernel_half_size,kernel_half_size+1))
            gaus_kernel = [np.exp(-(i**2)/(2*sigma_frames**2)) for i in gaus_win]
            gaus_kernel = gaus_kernel/sum(gaus_kernel)
            conv_trace = sg.convolve2d(np.expand_dims(signal,1), np.expand_dims(gaus_kernel,1), mode='same')
            return conv_trace.flatten()

        smoothed=np.zeros_like(self.MC_corrected)
        for x in np.arange(self.MC_corrected.shape[1]):
            for y in np.arange(self.MC_corrected.shape[2]):
                smoothed[:,x,y]=gaussian_smooth_kernel_convolution(self.MC_corrected[:,x,y],frame,sigma)
        self.smoothed_motion_corrected=cm.movie(smoothed)

    
        
        
        
        
        

#%% saving 
    def save_MC_kalman_movie(self):
     
        if self.kalman_path and self.dataset_kalman_caiman_movie.any():
            if not os.path.isfile( self.kalman_path):
                save_imagej_hdf5(self.dataset_kalman_caiman_movie, os.path.splitext(self.kalman_path)[0], '.tiff', )
            # if not os.path.isfile( os.path.join(os.path.splitext(self.kalman_path)[0],'.mmap')):
            #     self.dataset_kalman_caiman_movie.save(os.path.splitext(self.kalman_path)[0]+'.mmap' ,to32=False)  
            
    def save_gaussian_smoothed_movie(self):
     
        if self.gauss_path and self.smoothed_motion_corrected.any():
            if not os.path.isfile( self.gauss_path):
                save_imagej_hdf5(self.smoothed_motion_corrected, os.path.splitext(self.gauss_path)[0], '.tiff', )
            # if not os.path.isfile( os.path.join(os.path.splitext(self.kalman_path)[0],'.mmap')):
            #     self.dataset_kalman_caiman_movie.save(os.path.splitext(self.kalman_path)[0]+'.mmap' ,to32=False)


    def unload_kalman_movie(self): 
        module_logger.info('unloading kalman movies')
           
        if self.dataset_kalman_caiman_movie.any():
            del self.dataset_kalman_caiman_movie
        if self.MC_corrected.any():
            del self.MC_corrected
            
        gc.collect()
        sys.stdout.flush()
        self.dataset_kalman_caiman_movie=np.array([False])
        self.MC_corrected=np.array([False])

#%% ploting and others            

 
    def remove_some_files_form_temp(self):
        list_of_files = glob.glob(self.dump_temp_path+os.sep+'**')
        if list_of_files:
           oldest_file = min(list_of_files, key=os.path.getctime)
        if len(list_of_files)>5 and oldest_file !=  self.MC_file_path:
                os.remove(oldest_file)
   
    def remove_unclipped_issue_shifted_kalmans(self):
        module_logger.info('removing unclipped ')

        if self.kalman_custom_path and  self.kalman_full_path :
            if os.path.isfile(self.kalman_full_path):
                os.remove(self.kalman_full_path)

        

        

if __name__ == "__main__":
    
    # filename=r'211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000'
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIK3planeallen\Plane3'
    # dataset_full_file_mmap_path=os.path.join(temporary_path,filename+'_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_62499_.mmap')
    
    filename=r'211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'
    filename=r'211022_SPKS_FOV1_AllenA_20x_920_50024_narrow_with-000'

    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\Plane2'
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211022_SPKS_FOV1_AllenA_20x_920_50024_narrow_with-000\Plane1'
    

    dataset_full_file_mmap_path=os.path.join(temporary_path, filename+'_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_64416_.mmap')
    dataset_full_file_mmap_path=os.path.join(temporary_path, filename+'_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_226002_.mmap')

    
    
    dump='\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp'
    MotCorr = MotionCorrectedKalman(shifted_mmap_path=dataset_full_file_mmap_path, dump_temp_path=dump, keep_registered=True )
    


   