# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:30:39 2021

@author: sp3660
"""

import caiman as cm
import os 
import glob
import numpy as np
import tifffile
import matplotlib.pyplot as plt

from metadata import Metadata
from motionCorrectedKalman import MotionCorrectedKalman
from summaryImages import SummaryImages
from bidiShiftManager import BidiShiftManager
from caimanExtraction import CaimanExtraction

class ImageSequenceDataset:
    
    def __init__(self, raw_dataset_path=None, dataset_name=None, slow_storage_directory=None, bidishifted_dataset_path=None, temporary_directory=None, acquisition_object=None, testing=False, batch=False, dump_path=None):
        
        # raw_dataset_path is the plane* folder
        # bidishifted_dataset_path is the slow path sotrage for mmap
        # temporary folde pat is the temp to play with it
        # temporary shifted path extracted from bidishift
        # 
        
        self.acquisition_object=acquisition_object    
        self.slow_storage_directory=slow_storage_directory
        self.raw_dataset_path=raw_dataset_path
        self.bidishifted_dataset_path=bidishifted_dataset_path
        self.dataset_name=dataset_name
        self.temporary_directory=temporary_directory
        self.testing=testing
        self.batch=batch
        self.dump_path=dump_path
        self.caiman_extractions={}
        if self.bidishifted_dataset_path:
            self.dataset_name=os.path.split(self.bidishifted_dataset_path)[1]
        elif self.raw_dataset_path:
            self.dataset_name=os.path.split(os.path.split(os.path.split(self.raw_dataset_path)[0])[0])[1]

        
        self.add_plane_and_channel_to_dataset_name()
        
  
  
        """
        1st. Do bidishift and save shifted mmap to dump
        2nd save shits to dump also
        3rd copy shifted mempa and shifts to slow
        4rd do motion correct and kalman on dumtemp
        5th save kalman file MC template, mc shifts and MC mmamp to dumtemp
        6th copy kalman file MC template, mc shifts to slow
        7th do summary images of 
            bidishifted_MC_kalman
            bidishifted_MC
            bidihifted
        8th copy summary images to slow
        9th copy metadata files to slow and dumptemp
        10th copvy voltage files to slow and dumptemp
        11th read_metdata_from_file in dumptemp
        (11b) read metdata from database
        12 do caiman on dumptem
        13 save caiman resulys on dumptem
        14 copy caiman results to slow
        
        MANUAL
        15 evaluate with caiman sorter dumptemp
        16 save caiman sorter results to dumptemp and 
        
        
        """
        if self.raw_dataset_path:
            image_sequence_directory_full_path=os.path.split(os.path.split(self.raw_dataset_path)[0])[0]
            xmlfiles=glob.glob(image_sequence_directory_full_path+'\\**.xml')
            aq_metadataPath=xmlfiles[0]
            voltagerec_metadataPath=xmlfiles[1]
        
            if self.testing:
                # create bidishifte movie and shift file to temporary fast file
                self.bidishifts = BidiShiftManager(dataset_image_sequence_path=self.raw_dataset_path, temporary_path=self.temporary_directory, dataset_object=self)
                self.summary_images_bidishifted = SummaryImages(image_sequence_path=self.bidishifts.bidiphase_file_path, dataset_object=self)
                self.bidishifted_dataset_path=self.bidishifts.path_to_save_shifted_movie
                self.metadata= Metadata(aq_metadataPath=aq_metadataPath,voltagerec_metadataPath=voltagerec_metadataPath, temporary_path=self.temporary_directory)

            # get slow dataset patjh from acquisition object
                if self.acquisition_object:
                    self.bidishifts.copy_results_to_new_directory(self.slow_storage_directory)
                    self.summary_images_bidishifted.copy_results_to_new_directory(self.slow_storage_directory)
                    # copy temp bidishifts, bidishift movie and all   summary to slow
                
            # if batch doi it directly to slow if testing do it to temp and copy to slow
            if self.batch and self.acquisition_object:
                self.temporary_path=self.slow_storage_directory               
                self.bidishifts = BidiShiftManager(dataset_image_sequence_path=self.raw_dataset_path, temporary_path=self.temporary_directory, dataset_object=self)
                self.summary_images_bidishifted = SummaryImages(image_sequence_path=self.bidishifts.bidiphase_file_path, dataset_object=self)
                self.bidishifted_dataset_path=self.bidishifts.path_to_save_shifted_movie    
                self.metadata =Metadata(aq_metadataPath=aq_metadataPath,voltagerec_metadataPath=voltagerec_metadataPath, temporary_path=self.temporary_directory)
            
        if self.bidishifted_dataset_path:
                # if batch copy to slow if testing do it on slow directly
            # check metadata in folder
            # check 
            self.motion_correction_kalman = MotionCorrectedKalman(shifted_mmap_path=self.bidishifted_dataset_path, dump_temp_path=self.dump_path, keep_registered=True )
            # copy motion correction paramteres and kalman movie to to slow       
            self.MC_kalman_movie_path=self.motion_correction_kalman.MC_kalman_file_path
            self.MC_movie_path=self.motion_correction_kalman.MC_movie_file_path
            self.summary_images_kalman = SummaryImages(image_sequence_path=self.MC_kalman_movie_path)
            self.summary_images_MC = SummaryImages(image_sequence_path= self.MC_movie_path)

            self.caiman_extraction = CaimanExtraction(self.bidishifted_dataset_path, self.metadata.transfered_metadata_paths[0], temporary_path=self.temporary_directory)
            # add to keep track of differnt caiman extraciton, reading exiting hdf5
            # self.caiman_extractions[self.caiman_extraction]= 
    
  
    def add_plane_and_channel_to_dataset_name(self):
        
        if self.raw_dataset_path:
            dataset_plane=os.path.split(self.raw_dataset_path)[1]
            dataset_channel=os.path.split(os.path.split(self.raw_dataset_path)[0])[1]
            self.expanded_dataset_name='_'.join([self.dataset_name, dataset_plane, dataset_channel])
    
        
   
    def read_dataset_metadat_from_database(self):
        print('in progress')

        
             
if __name__ == "__main__":

    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIK3planeallen'
    dump='\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp'
    extra='\\\\?\\'
    raw_dataset_path=r'F:\Projects\LabNY\Imaging\2021\20211007\Mice\SPIK\FOV_1\Aq_1\211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000\Ch2Green\plane2'
    mmap_director=r'K:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Imaging\G2C\Ai14\SPIK\imaging'
    bidishited_path=r'211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_62499_.mmap'

    # ImageSequenceDataset(raw_dataset_path=raw_dataset_path,temporary_directory=temporary_path,testing=True)   
    ImageSequenceDataset(bidishifted_dataset_path=os.path.join(temporary_path, bidishited_path),temporary_directory=temporary_path,testing=True)               
            
 



        
       
        
        
        
        