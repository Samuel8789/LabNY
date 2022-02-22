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
from PIL import Image
import matplotlib.pyplot as plt
import tifffile
import logging 
module_logger = logging.getLogger(__name__)

from caiman.motion_correction import apply_shift_online
# from kalman_stack_filter import kalman_stack_filter
# from save_imagej_hdf5_tiff import save_imagej_hdf5
# from bidiShiftManager import BidiShiftManager

from .kalman_stack_filter import kalman_stack_filter
from .save_imagej_hdf5_tiff import save_imagej_hdf5
from .bidiShiftManager import BidiShiftManager
from ...functions.transform_path import transform_path


class MotionCorrectedKalman:
    
    def __init__(self, caiman_movie=None, dataset_full_file_mmap_path=None, dataset_image_sequence_path=None, 
                 motionshifts_file_path=None, register_template_full_path=None, kalman_motion_corrected_path=None, registered_movie_path=None,
                 shifted_mmap_path=None, shited_caiman_movie=None, bidiphase_file_path=None,
                 dump_temp_path=None,temporary_path=None, keep_registered=False, kalman_only=False,
                 dataset_object=None, bidimanager_object=None, load=False):
        
        self.caiman_movie=caiman_movie
        self.dataset_full_file_mmap_path=dataset_full_file_mmap_path
        self.dataset_image_sequence_path=dataset_image_sequence_path
        
        self.motionshifts_file_path=motionshifts_file_path
        self.register_template_full_path=register_template_full_path
        self.kalman_path=kalman_motion_corrected_path
        self.registered_movie_path=registered_movie_path
        
        self.temporary_path=temporary_path
        self.dump_temp_path=dump_temp_path
        self.keep_registered=keep_registered
        self.kalman_only=kalman_only
        
        self.dataset_object=dataset_object
        
        self.bidimanager_object=bidimanager_object
        self.shifted_mmap_path=shifted_mmap_path
        self.shited_caiman_movie=shited_caiman_movie
        
        self.MC_corrected=np.array([False])
        self.dataset_kalman_caiman_movie=np.array([False])
        self.MC_template=np.array([False])
        self.MC_shifts=None

        # self.get_proper_filenames()
    
    
    
        if  self.dataset_object:
            self.temporary_path=self.dataset_object.selected_dataset_mmap_path
            self.check_MC_info_mmap_directory()
          
            if self.shifted_mmap_path:
                if self.kalman_path:           
                    if os.path.isfile(self.kalman_path):
        
                        if self.dataset_object.bidishift_object.start_end_flag:
                            
                            if self.kalman_path==self.kalman_custom_path:
                                module_logger.info('Kalman already there, not loaded')
                                # self.remove_unclipped_issue_shifted_kalmans()
                                
                            elif self.kalman_path==self.kalman_full_path:
                                module_logger.info('doing new custom kalman')
                                self.MC_corrected=cm.load(self.shifted_mmap_path) 
                                self.do_kalman_filter()
                                module_logger.info('saving kalman')
                    
                                self.save_MC_kalman_movie()
                                module_logger.info('rechecking files')
                    
                                self.check_MC_info_mmap_directory()
                                module_logger.info('removing unlcipeed')
                    
                                self.remove_unclipped_issue_shifted_kalmans()
                                self.unload_kalman_movie()
        
                        else:
                            module_logger.info('kalman already done')
        
        
                else:
                    self.get_proper_filenames()
                    self.build_output_filenames_if_not_there()
                    self.MC_corrected=cm.load(self.shifted_mmap_path) 
                    self.do_kalman_filter()
                    module_logger.info('doing first kalman')
                    module_logger.info('saving kalman')
                    self.save_MC_kalman_movie()
                    module_logger.info('rechecking files')
                    self.check_MC_info_mmap_directory()
                    module_logger.info('removing unlcipeed')
        
                    self.remove_unclipped_issue_shifted_kalmans()
                    self.unload_kalman_movie()
            else:
                if self.kalman_path:
                    module_logger.info('kalman initialized not loaded')
                else:
                    module_logger.info('no kalman in directory')


                
    
        # else:
            # if load:
                
            #   self.load_mc_kalman_tiff()
                
              
            # elif self.registered_movie_path and not self.kalman_path:
            #     self.MC_corrected=cm.load(self.registered_movie_path)           
            #     module_logger.info('Loading registered movie')
                
            # elif not self.kalman_path:
            
            #     if self.caiman_movie:            
            #         self.BidiManager=BidiShiftManager(caiman_movie=self.caiman_movie, temporary_path=self.temporary_path )
            #         self.image_sequence=self.BidiManager.shifted_movie     
                    
            #     elif self.dataset_full_file_mmap_path:
            #         self.BidiManager=BidiShiftManager(dataset_full_file_mmap_path=self.dataset_full_file_mmap_path )
            #         self.image_sequence=self.BidiManager.shifted_movie   
            #         self.file_path_to_correct=self.BidiManager.shifted_movie_path
         
            #     elif self.dataset_image_sequence_path:
            #         self.BidiManager=BidiShiftManager(dataset_image_sequence_path=self.dataset_image_sequence_path, temporary_path=self.temporary_path )
            #         self.image_sequence=self.BidiManager.shifted_movie  
            #         self.file_path_to_correct=self.BidiManager.shifted_movie_path
                    
            #     elif self.shifted_mmap_path:
            #         self.BidiManager=BidiShiftManager(shifted_movie_path=self.shifted_mmap_path, temporary_path=self.temporary_path )
            #         self.file_path_to_correct=self.BidiManager.shifted_movie_path
            #         self.load_dataset_from_mmap()
                            
            #     elif self.shited_caiman_movie:
            #         self.image_sequence=self.shited_caiman_movie 
                    
            # if self.motionshifts_file_path and self.register_template_full_path:
            #     self.load_MC_shifts_from_file()
            #     self.load_MC_template_from_file()
            #     module_logger.info('loading template')
    
    
            # self.do_motion_correction()
            # self.save_MC_shifts()
            # self.save_MC_template()
            # self.plotting()
            # self.do_kalman_filter()
            # self.save_MC_kalman_movie()
            # self.save_MC_movie()
            # self.remove_some_files_form_temp()
        
        
   #%% checking in folder     
    def check_MC_info_mmap_directory(self):
         # text_files=glob.glob(self.mmap_directory+'\\**.txt')
         # pkl_files=glob.glob(self.mmap_directory+'\\**.pkl')
         # mmap_files=glob.glob(self.mmap_directory+'\\**.mmap')
         # tif_files=glob.glob(self.mmap_directory+'\\**.tif')
         # tiff_files=glob.glob(self.mmap_directory+'\\**.tiff')

         # self.motionshifts_file_path=[text for text in text_files if 'MC_shifts' in text]
         # self.register_template_full_path=[tif for tif in tif_files if 'MC_template' in tif]
         # self.registered_movie_path=[mmap for mmap in mmap_files if 'MC_movie' in mmap]
         # self.kalman_paths=[tiff for tiff in tiff_files if 'MC_kalman.tiff' in tiff]
         # self.registered_movie_OnAcid_paths=[mmap for mmap in mmap_files if 'MC_OnACID_d1' in mmap]
         
         # if self.motionshifts_file_paths:
         #     self.motionshifts_file_path=self.motionshifts_file_path[0]
         # if self.register_template_full_paths:
         #     self.register_template_full_path=self.register_template_full_path[0]
         # if self.registered_movie_paths:
         #     self.registered_movie_path=self.registered_movie_path[0]
         # if self.kalman_paths:
         #     self.kalman_path=self.kalman_path[0]
         # if self.registered_movie_OnAcid_path:
         #    self.registered_movie_OnAcid_path=self.registered_movie_OnAcid_path[0]
         #    self.registered_movie_path=self.registered_movie_OnAcid_path    

         self.kalman_custom_paths=[]
         self.kalman_full_paths=[]
         self.kalman_custom_path=None
         self.kalman_full_path=None
         self.kalman_path=None
         
         self.kalman_full_paths=glob.glob(self.temporary_path + '\\**Movie_MC_OnACID_**MC_kalman.tiff')
         self.kalman_custom_paths=glob.glob(self.temporary_path + '\\**end_MC_OnACID_**MC_kalman.tiff')

         if self.kalman_full_paths:
             self.kalman_full_path= self.kalman_full_paths[0]
         if self.kalman_custom_paths:
             self.kalman_custom_path= self.kalman_custom_paths[0]
         
         if self.kalman_custom_path:  
             self.kalman_path=self.kalman_custom_path
         elif self.kalman_full_path:
             self.kalman_path=self.kalman_full_path

         
         
        
#%% names and info          
            
    def get_proper_filenames(self):         
        
        if self.dataset_object:
            self.eliminate_caiman_extra_from_mmap()
            self.check_MC_info_mmap_directory()

        # elif self.dataset_image_sequence_path:
        #     first_frame_filename=os.path.split(glob.glob(self.dataset_image_sequence_path+'\\**.tif')[0])[1]
        #     self.good_filename=first_frame_filename[0:first_frame_filename.find('_Cycle')]
        # elif self.dataset_full_file_mmap_path or self.shifted_mmap_path or self.kalman_path:            
        #     self.eliminate_caiman_extra_from_mmap()
        #     self.check_MC_info_mmap_directory()
                
    def eliminate_caiman_extra_from_mmap(self) :  
        
        if self.dataset_object and self.shifted_mmap_path:
            self.mmap_directory, caiman_filename=os.path.split(self.shifted_mmap_path) 
            self.good_filename=caiman_filename[:caiman_filename.find('_d1_')]   

        # elif self.dataset_full_file_mmap_path:
        #     self.mmap_directory, caiman_filename=os.path.split(self.dataset_full_file_mmap_path)
        # elif self.shifted_mmap_path:
        #     self.mmap_directory, caiman_filename=os.path.split(self.shifted_mmap_path)  
        # elif self.kalman_path:
        #     self.mmap_directory, caiman_filename=os.path.split(self.kalman_path)  
        #     self.good_filename=caiman_filename[:caiman_filename.find('_d1_')]   
        # self.caiman_extra=caiman_filename[caiman_filename.find('_d1_'):caiman_filename.find('_mmap')-4]
    
    def build_output_filenames_if_not_there(self):
 
        if not self.kalman_path:
            if self.temporary_path and not self.dataset_object.bidishift_object.start_end_flag:
                   self.kalman_path='_'.join([os.path.join(self.temporary_path,self.good_filename),'MC_kalman.tiff'])
        
            elif self.mmap_directory:
                   self.kalman_path='_'.join([os.path.join(self.mmap_directory,self.good_filename),'MC_kalman.tiff'])    
   
        
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
            
    def load_MC_shifts_from_file(self):   
        if os.path.isfile(self.motionshifts_file_path):
            with open(self.motionshifts_file_path, "rb") as fp:   # Unpickling
              self.MC_shifts = pickle.load(fp)  
              
    def load_MC_template_from_file(self):   
        if os.path.isfile(self.register_template_full_path):
            self.MC_template= cm.load(self.register_template_full_path)        

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

 #%% processing   
    # def do_motion_correction(self):
    #     if not self.MC_corrected.any() and not self.kalman_only and not self.kalman_path:
    #         if self.temporary_path:
    #             self.MC_file_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'MC']),'txt'])
    #         else:
    #             self.MC_file_path='.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'MC']),'txt'])
                
    #         self.image_sequence_motion_correction=cm.motion_correction.MotionCorrect(self.file_path_to_correct, max_deviation_rigid=10)
    #         module_logger.info('doing motion correction')
    #         if self.MC_template.any() and not self.MC_shifts:
    #             module_logger.info('using template')
    #             self.image_sequence_motion_correction.motion_correct(template=self.MC_template, save_movie=False)
    #             self.MC_shifts=self.image_sequence_motion_correction.shifts_rig
    #             self.MC_corrected=self.image_sequence_motion_correction.apply_shifts_movie(self.file_path_to_correct)
    #             # this has to be tested
    #         elif self.MC_shifts:
    #             module_logger.info('using shifts')
    #             self.MC_corrected=apply_shift_online(self.image_sequence, self.MC_shifts)
                
    #             # self.image_sequence_motion_correction=cm.motion_correction.MotionCorrect(self.file_path_to_correct, max_deviation_rigid=10)
    #             # self.MC_corrected=self.image_sequence_motion_correction.apply_shifts_movie(self.file_path_to_correct)
    #         else:
    #             module_logger.info('from scratch')
    #             self.image_sequence_motion_correction.motion_correct(save_movie=False)
    #             self.MC_template=self.image_sequence_motion_correction.total_template_rig
    #             self.MC_shifts=self.image_sequence_motion_correction.shifts_rig
    #             self.MC_corrected=self.image_sequence_motion_correction.apply_shifts_movie(self.file_path_to_correct)
        

    def do_kalman_filter(self):
        if not self.dataset_kalman_caiman_movie.any() and not os.path.isfile(self.kalman_path):
            module_logger.info('doing kalman')
            dataset_kalman_array=kalman_stack_filter(self.MC_corrected)
            dataset_kalman_array_changed_type=dataset_kalman_array.astype(np.uint16)
            self.dataset_kalman_caiman_movie=cm.movie(dataset_kalman_array_changed_type, fr=300,start_time=0,file_name=None, meta_data=None)
        else:
            module_logger.info('kalman already done')



#%% saving 
    def save_MC_kalman_movie(self):
     
        if self.kalman_path and self.dataset_kalman_caiman_movie.any():
            if not os.path.isfile( self.kalman_path):
                save_imagej_hdf5(self.dataset_kalman_caiman_movie, os.path.splitext(self.kalman_path)[0], '.tiff', )
            # if not os.path.isfile( os.path.join(os.path.splitext(self.kalman_path)[0],'.mmap')):
            #     self.dataset_kalman_caiman_movie.save(os.path.splitext(self.kalman_path)[0]+'.mmap' ,to32=False)  
                

          
    # def save_MC_shifts(self):
    #     if self.MC_shifts:
    #         if self.temporary_path:
    #             self.MC_shifts_file_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'MC_shifts']),'txt'])
    #         elif self.dataset_full_file_mmap_path or self.shifted_mmap_path: 
    #             self.MC_shifts_file_path='.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'MC_shifts']),'txt']) 
    #         else:
    #             module_logger.info('MC_shifts not saved')
    #             return False
    
    #         with open(self.MC_shifts_file_path, "wb") as fp:   #Pickling
    #             pickle.dump(self.MC_shifts, fp)
            
    # def save_MC_template(self):
    #     if self.MC_template:
             
    #         if self.temporary_path:
    #             self.MC_template_file_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'MC_template']),'tif'])
    #         elif self.dataset_full_file_mmap_path or self.shifted_mmap_path: 
    #             self.MC_template_file_path='.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'MC_template']),'tif']) 
    #         else:
    #             module_logger.info('MC_template not saved')
    #             return False
     
            # im = Image.fromarray(self.MC_template)
            # im.save(self.MC_template_file_path)
        
    # def save_MC_movie(self):
    #     if self.keep_registered:    
    #         if self.MC_corrected:
    #             if self.temporary_path:
    #                 self.MC_movie_file_path='_'.join([os.path.join(self.temporary_path,self.good_filename),'MC_movie'])
    #             elif self.dataset_full_file_mmap_path or self.shifted_mmap_path: 
    #                 self.MC_movie_file_path='_'.join([os.path.join(self.mmap_directory,self.good_filename),'MC_movie'])
                    
    #             elif self.dump_temp_path:
    #                 self.MC_movie_file_path='_'.join([os.path.join(self.dump_temp_path,self.good_filename),'MC_movie'])
    
    #             save_imagej_hdf5(self.MC_corrected, self.MC_movie_file_path, '.mmap',)            


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
    def plotting(self):
        if self.MC_template.any() and self.MC_shifts:
            plt.figure()
            plt.imshow(self.MC_template)
            plt.figure()
            plt.plot(list(zip(*self.MC_shifts))[0])
            plt.figure()
            plt.plot(list(zip(*self.MC_shifts))[1])
            plt.figure()
            plt.hist(list(zip(*self.MC_shifts))[0])
            plt.figure()
            plt.hist(list(zip(*self.MC_shifts))[1])
        else:
            module_logger.info('No motion correction')
  

 
    def remove_some_files_form_temp(self):
        list_of_files = glob.glob(self.dump_temp_path+'\\**')
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
    


   