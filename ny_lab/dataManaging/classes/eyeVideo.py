# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:25:44 2021

@author: sp3660
"""
import tifffile
import caiman as cm
import numpy as np
import os
import gc
import sys
import json
import logging 
module_logger = logging.getLogger(__name__)

class EyeVideo():
    
    def __init__(self, aquisition_object=None, selected_eyevideo_raw_path=None, eyecameraslowstoragepath=None, eyecameraslowstoragepathmeta=None):
        self.aquisition_object=aquisition_object
        self.raw_face_camera_metadata=None
        self.full_eye_camera=np.array([False])
        self.raw_metadata_path2=None
        self.raw_image_sequence_paths2=None
        if self.aquisition_object:
            self.associated_aquisiton=aquisition_object        
            self.working_camera_full_path=os.path.join(self.associated_aquisiton.slow_storage_all_paths['eye camera'] ,self.associated_aquisiton.aquisition_name+'_full_face_camera.tiff')
            self.working_camera_full_metadata_path=os.path.join(self.associated_aquisiton.slow_storage_all_paths['eye camera'] ,self.associated_aquisiton.aquisition_name+'_full_face_camera_metadata.json')

    
        if selected_eyevideo_raw_path:

            if not os.path.isfile(self.working_camera_full_path):
                self.selected_eyevideo_raw_path=selected_eyevideo_raw_path
                image_sequence_files=os.listdir(self.selected_eyevideo_raw_path)
                
                if os.path.isdir(os.path.join(self.selected_eyevideo_raw_path,'Default')):
                    image_sequence_files2=os.listdir(os.path.join(self.selected_eyevideo_raw_path,'Default'))
                    self.raw_metadata_path2= [os.path.join(os.path.join(self.selected_eyevideo_raw_path,'Default'), image) for image in image_sequence_files2 if 'metadata.txt' in image]
                    self.raw_image_sequence_paths2= [os.path.join(os.path.join(self.selected_eyevideo_raw_path,'Default'), image) for image in image_sequence_files2 if '.tif' in image]



                self.raw_image_sequence_paths= [os.path.join(self.selected_eyevideo_raw_path, image) for image in image_sequence_files if '.tif' in image]
                self.raw_metadata_path1= [os.path.join(self.selected_eyevideo_raw_path, image) for image in image_sequence_files if 'metadata.txt' in image]
                if   self.raw_metadata_path1:
                    self.raw_metadata_path= self.raw_metadata_path1
                elif   self.raw_metadata_path2:
                    self.raw_metadata_path= self.raw_metadata_path2

                if not os.path.isfile(self.working_camera_full_path):
                    self.load_and_save_raw_camera_movies()
                if not os.path.isfile(self.working_camera_full_metadata_path):
                    self.load_and_save_raw_camera_metadata()
                self.unload_camera_file()
                self.unload_camera_metadata()
            else:
                module_logger.info('Camera already processed')
                    
                
            
#%% methods        
    def load_and_save_raw_camera_movies(self):  
        module_logger.info('processing eye video')

        if self.raw_image_sequence_paths:
            for i, file_name in enumerate(self.raw_image_sequence_paths):
                module_logger.info('processing '+str(i)+ ' video')
    
                with tifffile.TiffFile(file_name) as tffl:
                    input_arr = tffl.asarray()
                    self.full_eye_camera=cm.movie(input_arr.astype(np.uint8),
                                      fr= 30,
                                      start_time=0,                   
                                      file_name=os.path.split(file_name)[-1],
                                      meta_data=None,
                                      )
             
                del(input_arr)
                gc.collect()
                
        elif self.raw_image_sequence_paths2:    
             self.full_eye_camera=cm.load(self.raw_image_sequence_paths2, outtype=np.uint8)
            

        # self.associated_aquisiton.aquisition_name
        module_logger.info('Saving Face Camera')
        self.full_eye_camera.save(self.working_camera_full_path, to32=False,  imagej=False, bigtiff=True)
        # self.full_eye_camera.save(os.path.splitext(self.working_camera_full_path)[0]+'.mmap' ,to32=False)  

        module_logger.info('Saved Face Camera')
        
    def load_preprocessed_video(self):

        if self.aquisition_object:
            if os.path.isfile(self.working_camera_full_path):
                module_logger.info('Loading joined Face Camera')
                with tifffile.TiffFile(self.working_camera_full_path) as tffl:
                    input_arr = tffl.asarray()
                    self.full_eye_camera=cm.movie(input_arr.astype(np.uint8),
                                      fr= 30,
                                      start_time=0,                   
                                      file_name=os.path.split(self.working_camera_full_path)[-1],
                                      meta_data=None,
                                      )              
                del(input_arr)
                gc.collect()
                
   
                # self.full_eye_camera=cm.load(self.working_camera_full_path)
            else:
                module_logger.info('No file processed camera')
            
    def load_preprocessed_metadata(self):
        
        if self.aquisition_object:
            if os.path.isfile(self.working_camera_full_metadata_path):                       
                with open(self.working_camera_full_metadata_path,) as file :    
                    self.face_camera_metadata=json.load(file)
            else:
                module_logger.info('No file processed camera')   
        
        
        
    def load_and_save_raw_camera_metadata(self):      
        #%%
        if self.raw_metadata_path:
            self.raw_metadata_path=self.raw_metadata_path[0]
            
            with open(self.raw_metadata_path,) as file :    
                self.raw_face_camera_metadata=json.load(file)
            
            with open(self.working_camera_full_metadata_path, "w") as out_file:
                json.dump(self.raw_face_camera_metadata,  out_file)
                
                
                

    def unload_camera_file(self):
        module_logger.info('Unloading eye camera')

        if self.full_eye_camera.any():
            del self.full_eye_camera
            gc.collect()
            sys.stdout.flush()
            self.full_eye_camera=np.array([False])
 
    def check_if_camera_file_loaded(self):
        pass
            
    def unload_camera_metadata(self):     
        if self.raw_face_camera_metadata:
            del self.raw_face_camera_metadata
            gc.collect()
            self.raw_face_camera_metadata=[]
    

    def load_all(self):
        self.load_preprocessed_video()
        self.load_preprocessed_metadata()
            
