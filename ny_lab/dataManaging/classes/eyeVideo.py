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
import glob
import logging 
module_logger = logging.getLogger(__name__)

class EyeVideo():
    
    def __init__(self, aquisition_object=None, selected_eyevideo_raw_path=None, eyecameraslowstoragepath=None, eyecameraslowstoragepathmeta=None):
        
        self.associated_aquisiton=aquisition_object
        self.selected_eyevideo_raw_path=selected_eyevideo_raw_path
        self.temporary_path=r'C:\Users\sp3660\Desktop\TemporaryProcessing'
        
        self.raw_metadata_path=None


        self.define_paths()
        
        if self.selected_eyevideo_raw_path:
            
            self.get_raw_files_paths()

            if not os.path.isfile(self.working_camera_full_path_hdf5):
                self.load_raw_camera_movies_update()
                self.save_combined_camera()
                self.load_raw_camera_metadata()
                self.save_camera_metadata()

                
                self.unload_camera_file()
                self.unload_camera_metadata()
                
            else:
                module_logger.info('Camera already processed')

#%% methods        
    def define_paths(self):
        self.raw_face_camera_metadata=None
        self.full_eye_camera=np.array([False])
        self.raw_metadata_path2=None
        self.raw_image_sequence_paths2=None
        
        if self.associated_aquisiton:
            self.working_camera_full_path=os.path.join(self.associated_aquisiton.slow_storage_all_paths['eye camera'] ,self.associated_aquisiton.aquisition_name+'_full_face_camera.tiff')
            self.working_camera_full_path_mmap=os.path.join(self.associated_aquisiton.slow_storage_all_paths['eye camera'] ,self.associated_aquisiton.aquisition_name+'_full_face_camera.mmap')
            self.working_camera_full_path_hdf5=os.path.join(self.associated_aquisiton.slow_storage_all_paths['eye camera'] ,self.associated_aquisiton.aquisition_name+'_full_face_camera.hdf5')

            self.working_camera_full_metadata_path=os.path.join(self.associated_aquisiton.slow_storage_all_paths['eye camera'] ,self.associated_aquisiton.aquisition_name+'_full_face_camera_metadata.json')
        else:
            self.camera_name=glob.glob(self.selected_eyevideo_raw_path+os.sep+'**metadata.txt')[0][:-13]
            self.working_camera_full_path=os.path.join(self.temporary_path, os.path.split(self.camera_name)[1]+'_full_face_camera.tiff')
            self.working_camera_full_metadata_path=os.path.join(self.temporary_path ,os.path.split(self.camera_name)[1]+'_full_face_camera_metadata.json')
            self.working_camera_full_path_hdf5=os.path.join(self.temporary_path, os.path.split(self.camera_name)[1]+'_full_face_camera.hdf5')


    def get_raw_files_paths(self):
        #2 cases, one is big tiff and othe ris singkle files
        
        if self.selected_eyevideo_raw_path:

            if os.path.isdir(os.path.join(self.selected_eyevideo_raw_path,'Default')):
                
                self.flag='single_file'
                self.raw_image_sequence_paths=glob.glob(os.path.join(self.selected_eyevideo_raw_path,'Default')+os.sep+'**.tif')
                
                if glob.glob(os.path.join(self.selected_eyevideo_raw_path,'Default')+os.sep+'**metadata.txt'):
                    self.raw_metadata_path= glob.glob(os.path.join(self.selected_eyevideo_raw_path,'Default')+os.sep+'**metadata.txt')[0]  
                
            else:
                self.flag='big_tiffs'

                self.raw_image_sequence_paths=glob.glob(self.selected_eyevideo_raw_path+os.sep+'**.tif')
                
                if glob.glob(os.path.join(self.selected_eyevideo_raw_path)+os.sep+'**_metadata.txt'):
                    self.raw_metadata_path= glob.glob(os.path.join(self.selected_eyevideo_raw_path)+os.sep+'**_metadata.txt')[0]

    
    def load_raw_camera_movies(self):  
        module_logger.info('processing eye video')

        if self.raw_image_sequence_paths:
            if self.flag=='big_tiffs':
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
                    sys.stdout.flush()

                
            elif  self.flag=='single_file' : 
                self.full_eye_camera=cm.load(self.raw_image_sequence_paths, outtype=np.uint8)
                
                
    def load_raw_camera_movies_update(self):     

        module_logger.info('processing eye video')
      
        if self.raw_image_sequence_paths:
            if self.flag=='big_tiffs':
 
                self.full_eye_camera=cm.load(self.raw_image_sequence_paths[0],outtype=np.uint8)# outtype is not used when coming from  a tif file
     
                gc.collect()
                sys.stdout.flush()
                
            elif  self.flag=='single_file' : 
                self.full_eye_camera=cm.load(self.raw_image_sequence_paths, outtype=np.uint8)       

                      
                  
            
    def save_combined_camera(self):
        
        if self.full_eye_camera.any() and not os.path.isfile(self.working_camera_full_path_hdf5):
            module_logger.info('Saving Face Camera')
            self.full_eye_camera.save(self.working_camera_full_path_hdf5, to32=False,  imagej=False, bigtiff=True)

            # self.full_eye_camera.save(os.path.splitext(self.working_camera_full_path)[0]+'.mmap' ,to32=False)  
            module_logger.info('Saved Face Camera')
        
    # def load_preprocessed_video(self):

    #     if self.associated_aquisiton:
    #         if os.path.isfile(self.working_camera_full_path):
    #             module_logger.info('Loading joined Face Camera')
    #             with tifffile.TiffFile(self.working_camera_full_path) as tffl:
    #                 input_arr = tffl.asarray()
    #                 self.full_eye_camera=cm.movie(input_arr.astype(np.uint8),
    #                                   fr= 30,
    #                                   start_time=0,                   
    #                                   file_name=os.path.split(self.working_camera_full_path)[-1],
    #                                   meta_data=None,
    #                                   )              
    #             del(input_arr)
    #             gc.collect()
                
                
   
    #             # self.full_eye_camera=cm.load(self.working_camera_full_path)
    #         else:
    #             module_logger.info('No file processed camera')
                
    def load_preprocessed_video(self):

        if self.associated_aquisiton:
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
            elif os.path.isfile(self.working_camera_full_path_hdf5):
                    self.full_eye_camera=cm.load(self.working_camera_full_path_hdf5, outtype=np.uint8)       

                
            else:
                module_logger.info('No file processed camera')
            
    def load_preprocessed_metadata(self):
        
        if self.associated_aquisiton:
            if os.path.isfile(self.working_camera_full_metadata_path):                       
                with open(self.working_camera_full_metadata_path,) as file :    
                    self.face_camera_metadata=json.load(file)
            else:
                module_logger.info('No file processed camera')   
        
        
        
    def load_raw_camera_metadata(self):      
        #%%
        if self.raw_metadata_path:
            
            with open(self.raw_metadata_path,) as file :    
                self.raw_face_camera_metadata=json.load(file)
            
    
                
    def save_camera_metadata(self):      
        #%%
        if self.raw_face_camera_metadata and not os.path.isfile(self.working_camera_full_metadata_path) :
 
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
            
if __name__ == "__main__":
    
    dataset_image_sequence_path=r'F:\Projects\LabNY\Imaging\2022\20220428\Mice\SPMT\FOV_1\Aq_1\FaceCamera'

    eyevideo = EyeVideo(selected_eyevideo_raw_path=dataset_image_sequence_path )

    # dataset_full_file_mmap_path=os.path.join(temporary_path,'210930_SPKI_2mintestvideo_920_50024_narrow_without-000_shifted_movie_d1_256_d2_256_d3_1_order_F_frames_3391_.mmap')
    # bidihits = BidiShiftManager(dataset_full_file_mmap_path=dataset_full_file_mmap_path )

