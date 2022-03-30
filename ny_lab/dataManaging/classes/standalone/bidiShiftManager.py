# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 19:57:36 2021

@author: sp3660
"""
import sys
import caiman as cm
import numpy as np
import os 
import pickle
import glob
import gc

# from bidicorrect_image import shiftBiDi, biDiPhaseOffsets
from .bidicorrect_image import shiftBiDi, biDiPhaseOffsets

import matplotlib.pyplot as plt
import shutil 
from ...functions.transform_path import transform_path

import logging 
module_logger = logging.getLogger(__name__)

class BidiShiftManager:
    
    def __init__(self, caiman_movie=None, expanded_dataset_name=None, dataset_full_file_mmap_path=None, 
                 dataset_image_sequence_path=None, bidiphase_file_path=None, shifted_movie_path=None,
                 temporary_path=None, dataset_object=None, fullscripttemp=False, custom_start_end=False, force=False):
        
        self.start_end_flag=False
        self.caiman_movie=caiman_movie
        self.dataset_full_file_mmap_path=dataset_full_file_mmap_path
        self.dataset_image_sequence_path=dataset_image_sequence_path
        self.fullscripttemp=fullscripttemp
        self.bidiphase_file_path=bidiphase_file_path
        self.shifted_movie_path=shifted_movie_path
        self.frame_start=0
        self.frame_end=-1
        self.custom_start_end=custom_start_end
        self.temporary_path=temporary_path
        self.shifted_movie=np.array([False])
        self.bidiphases=[]
        self.expanded_dataset_name=expanded_dataset_name
        self.dataset_object=dataset_object
        
        
        
        if self.dataset_object:
            self.temporary_path=self.dataset_object.selected_dataset_mmap_path
            self.custom_start_end=True
         
            if self.temporary_path:

                self.check_shifted_movie_path()
                self.check_bidiphases_in_directory()
                self.get_proper_filenames() 
                     
                if self.custom_start_end:
                    module_logger.info('checking_custom_start_end')
                    self.read_custom_start_end()
                self.create_output_names_if_dont_exist()

                    # check if there is already a shifted file
                if self.shifted_movie_full_caiman_path and not force:
                    
                    if self.start_end_flag:
    
                        if self.shifted_movie_full_caiman_path==self.shifted_movie_custom_files_path:
                            module_logger.info('bidishifted movie there, not loaded')
                            # self.load_shifted_movie_from_mmap()              
                            if self.bidiphase_file_path:
                                module_logger.info('bidishifs there, not loaded')
                                # self.load_bidiphases_from_file()
                            # self.remove_unclipped_issue_shifted_movies()
    
                        elif self.shifted_movie_full_caiman_path==self.shifted_movie_files_path:
                            module_logger.info('doing new custom bidishifting')
                            self.dataset_full_file_mmap_path=self.shifted_movie_full_caiman_path
                            self.load_dataset_from_mmap()
                            self.correct_bidi_movie() 
                            module_logger.info('saving files')
                            self.save_shifts()
                            self.save_shifted_movie()
                            module_logger.info('saving files')
                            self.save_shifts()
                            self.save_shifted_movie()
                            module_logger.info('rechecking output files')
                            self.check_shifted_movie_path()
                            self.remove_unclipped_issue_shifted_movies()
                            self.unload_shifted_movie()
                            self.unload_bidishifts()
                            


                    else:        
                        module_logger.info('bidishifted movie there, not loaded')
                        try:
                            module_logger.info('bidishifted movie path: '+ self.shifted_movie_full_caiman_path)
                        except:
                            module_logger.exception('bidishifted movie path doesn exist')
                        try:
                            module_logger.info('bidishifs path: '+  self.bidiphase_file_path)
                        except:
                            module_logger.exception('bidishifs path doesn exist')
                
                elif self.dataset_image_sequence_path:  
                    # self.dataset_image_sequence_path=self.selected_dataset_raw_path

                    if os.path.isfile(self.bidiphase_file_path):
                        module_logger.info('loading bidishifts')
                        self.load_bidiphases_from_file()
                        
                    module_logger.info('loading raw image sequence')
                    # self.load_dataset_from_image_sequence()
                    self.load_dataset_from_image_sequence()
                    
                    if not self.start_end_flag:
                       

                        self.frame_end= len(self.image_sequence)+1
                    self.correct_bidi_movie() 
              
                        
                    module_logger.info('saving files')
                    self.save_shifts()
                    self.save_shifted_movie()
                    module_logger.info('rechecking output files')
                    self.check_shifted_movie_path()
                    self.remove_unclipped_issue_shifted_movies()
                    self.unload_shifted_movie()
                    self.unload_bidishifts()
                
        else:
            
            if self.shifted_movie_path:
                # self.load_shifted_movie_from_mmap()
                self.load_bidiphases_from_file()
                self.shifted_movie_path=self.shifted_movie_path
                module_logger.info('bidishifted movie loaded')
                
            elif self.custom_start_end:
                if self.bidiphase_file_path:
                    self.load_bidiphases_from_file()
                self.load_dataset_from_image_sequence()
               
                self.correct_bidi_movie()
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
            # self.plot_bidiphases()
            self.save_shifted_movie()
        
#%% check what ther is in the directory        
    def read_custom_start_end(self):
        start_end_file=os.path.join(os.path.split(os.path.split(self.temporary_path)[0])[0],'Start_End.txt')
        if os.path.isfile(start_end_file):
            self.start_end_flag=True
            with open(start_end_file) as f:
                lines = f.readlines()
            module_logger.info('changing custom start end')    
            start_end=[int(x) for x in lines]
            self.frame_start= start_end[0]
            self.frame_end=start_end[1]-1    

    def check_bidiphases_in_directory(self):
        self.bidiphase_custom_file_paths=[]
        self.bidiphase_full_file_paths=[]
        self.bidiphase_custom_file_path=None
        self.bidiphase_full_file_path=None
        self.bidiphase_file_path=None

        self.all_bidipahse_files=glob.glob(self.temporary_path+'\\**Bidiphases**.pkl')
        # mmap_files=glob.glob(self.mmap_directory+'\\**.mmap')
        self.bidiphase_custom_file_paths=[i for i in self.all_bidipahse_files if 'Bidiphases_custom' in i ]
        self.bidiphase_full_file_paths=[i for i in self.all_bidipahse_files if 'Bidiphases.' in i]
        
        if self.bidiphase_custom_file_paths:
            self.bidiphase_custom_file_path= self.bidiphase_custom_file_paths[0]
        if self.bidiphase_full_file_paths:
            self.bidiphase_full_file_path= self.bidiphase_full_file_paths[0]
        
        
        if self.bidiphase_custom_file_path:
            self.bidiphase_file_path= self.bidiphase_custom_file_path    
        elif self.bidiphase_full_file_path:
            self.bidiphase_file_path= self.bidiphase_full_file_path  
        
    def check_shifted_movie_path(self): 
        
        self.shifted_movie_custom_files_paths=[]
        self.shifted_movie_files_paths=[]
        self.shifted_movie_custom_files_path=None
        self.shifted_movie_files_path=None
        self.shifted_movie_full_caiman_path=None
        self.all_mmap_files=glob.glob(self.temporary_path+'\\**.mmap')
        self.shifted_movie_files_paths=[i for i in self.all_mmap_files if 'Shifted_Movie_d1' in i ]
        self.shifted_movie_custom_files_paths=[i for i in self.all_mmap_files if 'Shifted_Movie_custom_start_end_d1' in i ]


        if self.shifted_movie_custom_files_paths:
            self.shifted_movie_custom_files_path= self.shifted_movie_custom_files_paths[0]
        if self.shifted_movie_files_paths:
            self.shifted_movie_files_path= self.shifted_movie_files_paths[0]


        if self.shifted_movie_custom_files_path:
            self.shifted_movie_full_caiman_path= self.shifted_movie_custom_files_path    
        elif self.shifted_movie_files_path:
            self.shifted_movie_full_caiman_path= self.shifted_movie_files_path    
 
            
#%% get proper variables 
    
    def get_proper_filenames(self):
        
        if self.dataset_full_file_mmap_path or self.shifted_movie_full_caiman_path  :       
            self.eliminate_caiman_extra_from_mmap()
            # self.check_bidiphases_in_directory()
        elif self.dataset_full_file_mmap_path:
            first_frame_filename=os.path.split(glob.glob(self.dataset_image_sequence_path+'\\**.tif')[0])[1]
            self.good_filename=first_frame_filename[0:first_frame_filename.find('_Cycle')]
            # self.check_bidiphases_in_directory()
        elif self.dataset_image_sequence_path:
            first_frame_filename=os.path.split(glob.glob(self.dataset_image_sequence_path+'\\**.tif')[0])[1]
            self.good_filename=first_frame_filename[0:first_frame_filename.find('_Cycle')]
                        
    def eliminate_caiman_extra_from_mmap(self):
        if self.dataset_full_file_mmap_path:
            self.mmap_directory, caiman_filename=os.path.split(self.dataset_full_file_mmap_path)
            
        elif self.shifted_movie_full_caiman_path:
            self.mmap_directory, caiman_filename=os.path.split(self.shifted_movie_full_caiman_path)

        self.good_filename=caiman_filename[:caiman_filename.find('_d1_')]   
        # self.caiman_extra=caiman_filename[caiman_filename.find('_d1_'):caiman_filename.find('_mmap')-4]       
            
    def create_output_names_if_dont_exist(self):
        
        if self.temporary_path and not self.start_end_flag:
            self.bidiphase_file_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Bidiphases']),'pkl'])
            self.shifted_movie_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Shifted_Movie']),'mmap'])

        elif self.start_end_flag:
            self.bidiphase_file_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Bidiphases_custom_start_end']),'pkl'])  
            self.shifted_movie_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Shifted_Movie_custom_start_end']),'mmap'])

        elif self.mmap_directory:
            self.bidiphase_file_path='.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'Bidiphases']),'pkl']) 
            self.shifted_movie_path='.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'Shifted_Movie']),'mmap'])

        
#%% load thing if existing files
    

    def load_bidiphases_from_file(self):   
        if os.path.isfile(self.bidiphase_file_path):
            module_logger.info('loading bidishifts')
            with open(self.bidiphase_file_path, "rb") as fp:   # Unpickling
              self.bidiphases = pickle.load(fp)

    def load_dataset_from_image_sequence(self):
        image_sequence_files=os.listdir(self.dataset_image_sequence_path)
        image_sequence_paths= [os.path.join(self.dataset_image_sequence_path, image) for image in image_sequence_files if os.path.getsize(os.path.join(self.dataset_image_sequence_path, image))!=0 ]
        module_logger.info('loading files')
        self.image_sequence=cm.load(image_sequence_paths)
        if len(self.image_sequence)>1:
            self.image_sequence= self.image_sequence[self.frame_start:self.frame_end,:,:]
        else:
            pass

    def load_dataset_from_mmap(self): 
        self.image_sequence
        self.image_sequence= self.image_sequence[self.frame_start:self.frame_end,:,:]

        
    def load_shifted_movie_from_mmap(self): 
        if self.shifted_movie_full_caiman_path:
            if os.path.isfile(self.shifted_movie_full_caiman_path):
                self.shifted_movie=cm.load(self.shifted_movie_full_caiman_path)



#%% do the processing

    def correct_bidi_movie(self): 
        if not self.bidiphases:
            module_logger.info('calculating bidiphases')
            self.calculate_bidi_shifts()  
        else:
            module_logger.info('bidiphases loaded')
        if not self.shifted_movie.any():
            module_logger.info('calculating bidishifted movie')
            self.shifted_movie=cm.movie(self.shift_images())
        else:
            module_logger.info('bidishifted movie loaded')

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
    

    
#%% save the stuff
   
    def save_shifts(self):
        if self.bidiphases and self.bidiphase_file_path:
            if not os.path.isfile(self.bidiphase_file_path):
                with open(self.bidiphase_file_path, "wb") as fp:   #Pickling
                    pickle.dump(self.bidiphases, fp)
               
    def save_shifted_movie(self):
   
        if self.shifted_movie.any() and self.shifted_movie_path:
            if not os.path.isfile(self.shifted_movie_path):
                self.shifted_movie.save(self.shifted_movie_path ,to32=False)  

    def unload_shifted_movie(self):     
        module_logger.info('unloading bidishifted movies')

        if self.shifted_movie.any():
            del self.shifted_movie
            gc.collect()
            sys.stdout.flush()
            self.shifted_movie=np.array([False])
            
    def unload_bidishifts(self):
        if self.bidiphases:
            del self.bidiphases
            gc.collect()
            self.bidiphases=[]
        
       

#%% plotting and others           

    def plot_bidiphases(self):
        plt.figure()
        plt.plot(self.bidiphases)
        plt.figure()
        plt.hist(self.bidiphases, 50, density=True, alpha=0.75)
        
    def copy_results_to_new_directory(self, new_directory):        
        self.shifted_movie_path
        self.bidiphase_file_path
        
        shutil.copyfile(self.path_to_save_shifted_movie,   os.path.join(new_directory, os.path.split(self.path_to_save_shifted_movie)[1]))
        shutil.copyfile(self.bidiphase_file_path,          os.path.join(new_directory, os.path.split(self.bidiphase_file_path)[1]))

    def remove_unclipped_issue_shifted_movies(self):
        module_logger.info('removing unclipped ')

        if self.bidiphase_custom_file_path and  self.bidiphase_full_file_path :
            if os.path.isfile(self.bidiphase_full_file_path):
                os.remove(self.bidiphase_full_file_path)


        if self.shifted_movie_custom_files_path and  self.shifted_movie_files_path :
            if os.path.isfile(self.shifted_movie_files_path):
                os.remove(self.shifted_movie_files_path)
        
                
if __name__ == "__main__":
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIK3planeallen\Plane3'
    # dataset_image_sequence_path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20211007\Mice\SPIK\FOV_1\Aq_1\211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000\Ch2Green\plane3'
    
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\Plane3'
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211022_SPKS_FOV1_AllenA_20x_920_50024_narrow_with-000\Plane1'
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211113_SPKQ_FOV1_2planeAllenA_20x_920_50024_narrow_without-000\Plane1'
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211113_SPKQ_FOV1_2planeAllenA_20x_920_50024_narrow_without-000\Plane2'



    # dataset_image_sequence_path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20211015\Mice\SPKG\FOV_1\Aq_1\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\Ch2Green\plane3'
    # dataset_image_sequence_path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20211022\Mice\SPKS\FOV_1\Aq_1\211022_SPKS_FOV1_AllenA_20x_920_50024_narrow_with-000\Ch2Green\plane1'
    # dataset_image_sequence_path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20211113\Mice\SPKQ\FOV_1\Aq_1\211113_SPKQ_FOV1_2planeAllenA_20x_920_50024_narrow_without-000\Ch2Green\plane1'
    dataset_image_sequence_path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20211113\Mice\SPKQ\FOV_1\Aq_1\211113_SPKQ_FOV1_2planeAllenA_20x_920_50024_narrow_without-000\Ch2Green\plane2'




    bidihits = BidiShiftManager(dataset_image_sequence_path=dataset_image_sequence_path, temporary_path=temporary_path )

    # dataset_full_file_mmap_path=os.path.join(temporary_path,'210930_SPKI_2mintestvideo_920_50024_narrow_without-000_shifted_movie_d1_256_d2_256_d3_1_order_F_frames_3391_.mmap')
    # bidihits = BidiShiftManager(dataset_full_file_mmap_path=dataset_full_file_mmap_path )




    