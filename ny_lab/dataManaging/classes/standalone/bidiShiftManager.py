# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 19:57:36 2021

@author: sp3660
"""

import caiman as cm
import numpy as np
import os 
import pickle
import glob
from bidicorrect_image import shiftBiDi, biDiPhaseOffsets
import matplotlib.pyplot as plt
import shutil 

class BidiShiftManager:
    
    def __init__(self, caiman_movie=None, expanded_dataset_name=None, dataset_full_file_mmap_path=None, dataset_image_sequence_path=None, bidiphase_file_path=None, shifted_movie_path=None, temporary_path=None, dataset_object=None):
        
        self.caiman_movie=caiman_movie
        self.dataset_full_file_mmap_path=dataset_full_file_mmap_path
        self.dataset_image_sequence_path=dataset_image_sequence_path
        
        self.bidiphase_file_path=bidiphase_file_path
        self.shifted_movie_path=shifted_movie_path
        
        self.temporary_path=temporary_path
        
        self.shifted_movie=np.array([False])
        self.bidiphases=[]
        self.expanded_dataset_name=expanded_dataset_name
        
        if self.expanded_dataset_name:
            self.good_filename=self.expanded_dataset_name            
        else:
            self.get_proper_filenames()
        
        
        if self.shifted_movie_path:
            self.load_shifted_movie_from_mmap()
            self.load_bidiphases_from_file()
            self.path_to_save_shifted_movie=self.shifted_movie_path
            print('bidishifted movie loaded')
      
        else:
        
            if self.caiman_movie:
                self.image_sequence=self.caiman_movie
                
            elif self.dataset_full_file_mmap_path:
                self.load_dataset_from_mmap()
     
            elif self.dataset_image_sequence_path:
                 self.load_dataset_from_image_sequence()   

            if self.bidiphase_file_path:
                self.load_bidiphases_from_file()
              
            self.correct_bidi_movie()
           
    
        self.save_shifts()
        self.plot_bidiphases()
        self.save_shifted_movie()
            

    def load_dataset_from_image_sequence(self):
        image_sequence_files=os.listdir(self.dataset_image_sequence_path)
        image_sequence_paths= [os.path.join(self.dataset_image_sequence_path, image) for image in image_sequence_files]
        self.image_sequence=cm.load(image_sequence_paths)

    def load_dataset_from_mmap(self): 
        self.image_sequence=cm.load(self.dataset_full_file_mmap_path)
        
    def load_shifted_movie_from_mmap(self): 
        self.shifted_movie=cm.load(self.shifted_movie_path)

    def correct_bidi_movie(self): 
        if not self.bidiphases:
            print('calculating bidiphases')
            self.calculate_bidi_shifts()  
        else:
            print('bidiphases loaded')
        if not self.shifted_movie.any():
            print('calculating bidishifted movie')
            self.shifted_movie=cm.movie(self.shift_images())
        else:
            print('bidishifted movie loaded')
        
    
    def calculate_bidi_shifts(self):
        self.bidiphases=[]
        for i in range(self.image_sequence.shape[0]):
            BiDiPhase=biDiPhaseOffsets(self.image_sequence[i,:,:])
            self.bidiphases.append(BiDiPhase)
    
    def shift_images(self):
        shifted_images=np.zeros(self.image_sequence.shape).astype('float32')
        for i in range(self.image_sequence.shape[0]):
            shifted_images[i,:,:]=shiftBiDi(self.bidiphases[i], self.image_sequence[i,:,:])
        return shifted_images
    
    def save_shifts(self):
        if not self.bidiphase_file_path:
            if self.temporary_path:
                self.bidiphase_file_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Bidiphases']),'txt'])
            elif self.dataset_full_file_mmap_path: 
                self.bidiphase_file_path='.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'Bidiphases']),'txt']) 
            else:
                print('BidiShifts not saved')
                return False
    
            with open(self.bidiphase_file_path, "wb") as fp:   #Pickling
                pickle.dump(self.bidiphases, fp)
               
    def save_shifted_movie(self):
        if self.temporary_path:
            self.path_to_save_shifted_movie='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Shifted_Movie']),'mmap'])
        else:
            self.path_to_save_shifted_movie='.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'Shifted_Movie']),'mmap'])
            
        self.shifted_movie.save(self.path_to_save_shifted_movie ,to32=False)  
    
    def load_bidiphases_from_file(self):   
        if os.path.isfile(self.bidiphase_file_path):
            with open(self.bidiphase_file_path, "rb") as fp:   # Unpickling
              self.bidiphases = pickle.load(fp)
              
    def eliminate_caiman_extra_from_mmap(self):
        if self.dataset_full_file_mmap_path:
            self.mmap_directory, caiman_filename=os.path.split(self.dataset_full_file_mmap_path)
        elif self.shifted_movie_path:
            self.mmap_directory, caiman_filename=os.path.split(self.shifted_movie_path)

        self.good_filename=caiman_filename[:caiman_filename.find('_d1_')]   
        self.caiman_extra=caiman_filename[caiman_filename.find('_d1_'):caiman_filename.find('_mmap')-4]
    
    def get_proper_filenames(self):
        
        if self.dataset_image_sequence_path:
            first_frame_filename=os.path.split(glob.glob(self.dataset_image_sequence_path+'\\**.tif')[0])[1]
            self.good_filename=first_frame_filename[0:first_frame_filename.find('_Cycle')]
        elif self.dataset_full_file_mmap_path or self.shifted_movie_path:            
            self.eliminate_caiman_extra_from_mmap()
            self.check_bidiphases_mmap_directory()
            
    def check_bidiphases_mmap_directory(self):
        text_files=glob.glob(self.mmap_directory+'\\**.txt')
        mmap_files=glob.glob(self.mmap_directory+'\\**.mmap')
        self.bidiphase_file_path=[text for text in text_files if 'Bidiphases' in text]
        if self.bidiphase_file_path:
            self.bidiphase_file_path=self.bidiphase_file_path[0]
        self.shifted_movie_path=[mmap for mmap in mmap_files if 'Shifted_Movie_d1' in mmap]
        if self.shifted_movie_path:
            self.shifted_movie_path=self.shifted_movie_path[0]
    
    def plot_bidiphases(self):
        plt.figure()
        plt.plot(self.bidiphases)
        plt.figure()
        plt.hist(self.bidiphases, 50, density=True, alpha=0.75)
        
    def copy_results_to_new_directory(self, new_directory):        
        self.path_to_save_shifted_movie
        self.bidiphase_file_path
        
        shutil.copyfile(self.path_to_save_shifted_movie,   os.path.join(new_directory, os.path.split(self.path_to_save_shifted_movie)[1]))
        shutil.copyfile(self.bidiphase_file_path,          os.path.join(new_directory, os.path.split(self.bidiphase_file_path)[1]))

         
        
        
                
if __name__ == "__main__":
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIK3planeallen\Plane3'
    # dataset_image_sequence_path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20211007\Mice\SPIK\FOV_1\Aq_1\211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000\Ch2Green\plane3'
    
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\Plane3'
    dataset_image_sequence_path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20211015\Mice\SPKG\FOV_1\Aq_1\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\Ch2Green\plane3'

    bidihits = BidiShiftManager(dataset_image_sequence_path=dataset_image_sequence_path, temporary_path=temporary_path )

    # dataset_full_file_mmap_path=os.path.join(temporary_path,'210930_SPKI_2mintestvideo_920_50024_narrow_without-000_shifted_movie_d1_256_d2_256_d3_1_order_F_frames_3391_.mmap')
    # bidihits = BidiShiftManager(dataset_full_file_mmap_path=dataset_full_file_mmap_path )




    