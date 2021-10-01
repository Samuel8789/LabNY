# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:25:44 2021

@author: sp3660
"""
import tifffile
import caiman as cm
import numpy as np
import os
import json
import logging 
logger = logging.getLogger(__name__)

class EyeVideo():
    
    def __init__(self, aquisition_object, selected_eyevideo_raw_path=None):
        
        self.associated_aquisiton=aquisition_object        
        self.working_camera_full_path=os.path.join(self.associated_aquisiton.working_facecamera_path ,self.associated_aquisiton.aquisition_name+'_full_face_camera.tiff')
        self.working_camera_full_metadata_path=os.path.join(self.associated_aquisiton.working_facecamera_path ,self.associated_aquisiton.aquisition_name+'_full_face_camera_metadata.json')

    
        if selected_eyevideo_raw_path:
            if not os.path.isfile(self.working_camera_full_path):
                self.selected_eyevideo_raw_path=selected_eyevideo_raw_path
                image_sequence_files=os.listdir(self.selected_eyevideo_raw_path)
                
                self.raw_metadata_path= [os.path.join(self.selected_eyevideo_raw_path, image) for image in image_sequence_files if 'metadata.txt' in image]
                self.raw_image_sequence_paths= [os.path.join(self.selected_eyevideo_raw_path, image) for image in image_sequence_files if '.tif' in image]
                self.load_and_save_raw_camera_movies()
                self.load_and_save_raw_camera_metadata()
            else:
                print('Camera already processed')

        
    def load_and_save_raw_camera_movies(self):  
        # print('Loading Face Camera')
        for file_name in self.raw_image_sequence_paths:
             with tifffile.TiffFile(file_name) as tffl:
                 input_arr = tffl.asarray()
                 self.full_eye_camera=cm.movie(input_arr.astype(np.uint8),
                                   fr= 30,
                                   start_time=0,                   
                                   file_name=os.path.split(file_name)[-1],
                                   meta_data=None,
                                   )
             
        del(input_arr)
        self.associated_aquisiton.aquisition_name
        # print('Saving Face Camera')
        self.full_eye_camera.save(self.working_camera_full_path, to32=False,  imagej=False, bigtiff=True)
        # print('Saved Face Camera')

        
    def load_and_save_raw_camera_metadata(self):      
        #%%
        if self.raw_metadata_path:
            self.raw_metadata_path=self.raw_metadata_path[0]
            
            with open(self.raw_metadata_path,) as file :    
                self.raw_face_camera_metadata=json.load(file)
            
            with open(self.working_camera_full_metadata_path, "w") as out_file:
                json.dump(self.raw_face_camera_metadata,  out_file)

    