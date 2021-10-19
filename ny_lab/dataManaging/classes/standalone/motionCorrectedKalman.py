# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 18:43:19 2021

@author: sp3660
"""

import os
import caiman as cm
import numpy as np
import pickle
import glob
from PIL import Image
import matplotlib.pyplot as plt
import tifffile

from kalman_stack_filter import kalman_stack_filter
from save_imagej_hdf5_tiff import save_imagej_hdf5
from bidiShiftManager import BidiShiftManager



class MotionCorrectedKalman:
    
    def __init__(self, caiman_movie=None, dataset_full_file_mmap_path=None, dataset_image_sequence_path=None, 
                 motionshifts_file_path=None, register_template_full_path=None, kalman_motion_corrected_path=None, registered_movie_path=None,
                 shifted_mmap_path=None, shited_caiman_movie=None, bidiphase_file_path=None,
                 dump_temp_path=None,temporary_path=None, keep_registered=False, kalman_only=False,
                 dataset_object=None, bidimanager_object=None):
        
        self.caiman_movie=caiman_movie
        self.dataset_full_file_mmap_path=dataset_full_file_mmap_path
        self.dataset_image_sequence_path=dataset_image_sequence_path
        
        self.motionshifts_file_path=motionshifts_file_path
        self.register_template_full_path=register_template_full_path
        self.kalman_motion_corrected_path=kalman_motion_corrected_path
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

        self.get_proper_filenames()
    
        if self.kalman_motion_corrected_path:
            self.MC_corrected=True
            if '.tiff' in self.kalman_motion_corrected_path:
                with tifffile.TiffFile(self.kalman_motion_corrected_path) as tffl:
                     input_arr = tffl.asarray()
                     self.dataset_kalman_caiman_movie=cm.movie(input_arr.astype(np.uint16))
                del(input_arr)    

            print('MC_klaman loaded')
            
        elif self.registered_movie_path:
            self.MC_corrected=cm.load(self.registered_movie_path)           
            print('Loading registered movie')
            
        else:
        
            if self.caiman_movie:            
                self.BidiManager=BidiShiftManager(caiman_movie=self.caiman_movie, temporary_path=self.temporary_path )
                self.image_sequence=self.BidiManager.shifted_movie     
                
            elif self.dataset_full_file_mmap_path:
                self.BidiManager=BidiShiftManager(dataset_full_file_mmap_path=self.dataset_full_file_mmap_path )
                self.image_sequence=self.BidiManager.shifted_movie   
                self.file_path_to_correct=self.BidiManager.shifted_movie_path
     
            elif self.dataset_image_sequence_path:
                self.BidiManager=BidiShiftManager(dataset_image_sequence_path=self.dataset_image_sequence_path, temporary_path=self.temporary_path )
                self.image_sequence=self.BidiManager.shifted_movie  
                self.file_path_to_correct=self.BidiManager.shifted_movie_path
                
            elif self.shifted_mmap_path:
                self.BidiManager=BidiShiftManager(shifted_movie_path=self.shifted_mmap_path, temporary_path=self.temporary_path )
                self.file_path_to_correct=self.BidiManager.shifted_movie_path
                self.load_dataset_from_mmap()
                        
            elif self.shited_caiman_movie:
                self.image_sequence=self.shited_caiman_movie 
                
        if self.motionshifts_file_path and self.register_template_full_path:
            self.load_MC_shifts_from_file()
            self.load_MC_template_from_file()
            print('loading template')


        self.do_motion_correction()
        self.save_MC_shifts()
        self.save_MC_template()
        self.plotting()
        self.do_kalman_filter()
        self.save_MC_kalman_movie()
        self.save_MC_movie()
        self.remove_some_files_form_temp()

    def load_dataset_from_image_sequence(self):
        image_sequence_files=os.listdir(self.dataset_image_sequence_path)
        image_sequence_paths= [os.path.join(self.dataset_image_sequence_path, image) for image in image_sequence_files]
        self.image_sequence=cm.load(image_sequence_paths)
        
    def load_dataset_from_mmap(self): 
        if self.dataset_full_file_mmap_path:
            self.image_sequence=cm.load(self.dataset_full_file_mmap_path)
        elif self.shifted_mmap_path:
            self.image_sequence=cm.load(self.shifted_mmap_path)
            
    def load_MC_shifts_from_file(self):   
        if os.path.isfile(self.motionshifts_file_path):
            with open(self.motionshifts_file_path, "rb") as fp:   # Unpickling
              self.MC_shifts = pickle.load(fp)  
              
    def load_MC_template_from_file(self):   
        if os.path.isfile(self.register_template_full_path):
            self.MC_template= cm.load(self.register_template_full_path)        
              
    
    def do_motion_correction(self):
        if not self.MC_corrected.any() and not self.kalman_only:
            if self.temporary_path:
                self.MC_file_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'MC']),'txt'])
            else:
                self.MC_file_path='.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'MC']),'txt'])
                
            self.image_sequence_motion_correction=cm.motion_correction.MotionCorrect(self.file_path_to_correct, max_deviation_rigid=10)
            print('doing motion correction')
            if self.MC_template.any():
                self.image_sequence_motion_correction.motion_correct(template=self.MC_template, save_movie=False)
            else:
                self.image_sequence_motion_correction.motion_correct(save_movie=False)
                
            self.MC_template=self.image_sequence_motion_correction.total_template_rig
            self.MC_shifts=self.image_sequence_motion_correction.shifts_rig
            self.MC_corrected=self.image_sequence_motion_correction.apply_shifts_movie(self.file_path_to_correct)
        
            
            
    
    def remove_some_files_form_temp(self):
        list_of_files = glob.glob(self.dump_temp_path+'\\**')
        if list_of_files:
           oldest_file = min(list_of_files, key=os.path.getctime)
        if len(list_of_files)>5 and oldest_file !=  self.MC_file_path:
                os.remove(oldest_file)

    def do_kalman_filter(self):
        if not self.dataset_kalman_caiman_movie.any():
            print('doing kalman')
            dataset_kalman_array=kalman_stack_filter(self.MC_corrected)
            dataset_kalman_array_changed_type=dataset_kalman_array.astype(np.uint16)
            self.dataset_kalman_caiman_movie=cm.movie(dataset_kalman_array_changed_type, fr=300,start_time=0,file_name=None, meta_data=None)
        

    def save_MC_kalman_movie(self):
            if self.temporary_path:
                self.MC_kalman_file_path='_'.join([os.path.join(self.temporary_path,self.good_filename),'MC_kalman'])
            elif self.dataset_full_file_mmap_path or self.shifted_mmap_path: 
                self.MC_kalman_file_path='_'.join([os.path.join(self.mmap_directory,self.good_filename),'MC_kalman'])    
            else:
                print('MC_kalman not saved')
                return False
         
            if not os.path.isfile(self.MC_kalman_file_path):
                save_imagej_hdf5(self.dataset_kalman_caiman_movie,  self.MC_kalman_file_path, '.tiff', )
            
    def eliminate_caiman_extra_from_mmap(self) :   
        if self.dataset_full_file_mmap_path:
            self.mmap_directory, caiman_filename=os.path.split(self.dataset_full_file_mmap_path)
        elif self.shifted_mmap_path:
            self.mmap_directory, caiman_filename=os.path.split(self.shifted_mmap_path)  
        elif self.kalman_motion_corrected_path:
            self.mmap_directory, caiman_filename=os.path.split(self.kalman_motion_corrected_path)  
        self.good_filename=caiman_filename[:caiman_filename.find('_d1_')]   
        self.caiman_extra=caiman_filename[caiman_filename.find('_d1_'):caiman_filename.find('_mmap')-4]
    
    def get_proper_filenames(self):           
        if self.dataset_image_sequence_path:
            first_frame_filename=os.path.split(glob.glob(self.dataset_image_sequence_path+'\\**.tif')[0])[1]
            self.good_filename=first_frame_filename[0:first_frame_filename.find('_Cycle')]
        elif self.dataset_full_file_mmap_path or self.shifted_mmap_path or self.kalman_motion_corrected_path:            
            self.eliminate_caiman_extra_from_mmap()
            self.check_MC_info_mmap_directory()
            
            
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
            print('No motion correction')
  
    def save_MC_shifts(self):
        
        if self.temporary_path:
            self.MC_shifts_file_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'MC_shifts']),'txt'])
        elif self.dataset_full_file_mmap_path or self.shifted_mmap_path: 
            self.MC_shifts_file_path='.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'MC_shifts']),'txt']) 
        else:
            print('MC_shifts not saved')
            return False

        with open(self.MC_shifts_file_path, "wb") as fp:   #Pickling
            pickle.dump(self.MC_shifts, fp)
            
    def save_MC_template(self):
             
        if self.temporary_path:
            self.MC_template_file_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'MC_template']),'tif'])
        elif self.dataset_full_file_mmap_path or self.shifted_mmap_path: 
            self.MC_template_file_path='.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'MC_template']),'tif']) 
        else:
            print('MC_template not saved')
            return False
 
        im = Image.fromarray(self.MC_template)
        im.save(self.MC_template_file_path)
        
    def save_MC_movie(self):
        if self.keep_registered:     
            if self.temporary_path:
                self.MC_movie_file_path='_'.join([os.path.join(self.temporary_path,self.good_filename),'MC_movie'])
            elif self.dataset_full_file_mmap_path or self.shifted_mmap_path: 
                self.MC_movie_file_path='_'.join([os.path.join(self.mmap_directory,self.good_filename),'MC_movie'])
                
            elif self.dump_temp_path:
                self.MC_movie_file_path='_'.join([os.path.join(self.dump_temp_path,self.good_filename),'MC_movie'])

            save_imagej_hdf5(self.MC_corrected, self.MC_movie_file_path, '.mmap',)

    def check_MC_info_mmap_directory(self):
         text_files=glob.glob(self.mmap_directory+'\\**.txt')
         mmap_files=glob.glob(self.mmap_directory+'\\**.mmap')
         tif_files=glob.glob(self.mmap_directory+'\\**.tif')
         tiff_files=glob.glob(self.mmap_directory+'\\**.tiff')

         self.motionshifts_file_path=[text for text in text_files if 'MC_shifts' in text]
         self.register_template_full_path=[tif for tif in tif_files if 'MC_template' in tif]
         self.registered_movie_path=[mmap for mmap in mmap_files if 'MC_movie' in mmap]
         self.kalman_motion_corrected_path=[tiff for tiff in tiff_files if 'MC_kalman' in tiff]
         self.registered_movie_OnAcid_path=[mmap for mmap in mmap_files if 'MC_OnACID' in mmap]
         
         if self.motionshifts_file_path:
             self.motionshifts_file_path=self.motionshifts_file_path[0]
         if self.register_template_full_path:
             self.register_template_full_path=self.register_template_full_path[0]
         if self.registered_movie_path:
             self.registered_movie_path=self.registered_movie_path[0]
         if self.kalman_motion_corrected_path:
             self.kalman_motion_corrected_path=self.kalman_motion_corrected_path[0]
         if self.registered_movie_OnAcid_path:
            self.registered_movie_OnAcid_path=self.registered_movie_OnAcid_path[0]
            self.registered_movie_path=self.registered_movie_OnAcid_path


if __name__ == "__main__":
    
    # filename=r'211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000'
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIK3planeallen\Plane3'
    # dataset_full_file_mmap_path=os.path.join(temporary_path,filename+'_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_62499_.mmap')
    
    filename=r'211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\Plane2'
    dataset_full_file_mmap_path=os.path.join(temporary_path, filename+'_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_64416_.mmap')
    
    
    dump='\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp'
    MotCorr = MotionCorrectedKalman(shifted_mmap_path=dataset_full_file_mmap_path, dump_temp_path=dump, keep_registered=True )
    

   