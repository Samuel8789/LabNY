# -*- coding: utf-8 -*-
"""
Created on Sun May  1 13:10:24 2022

@author: sp3660
"""

import caiman as cm
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import glob
import gc
import shutil 

import logging 
from ScanImageTiffReader import ScanImageTiffReader



module_logger = logging.getLogger(__name__)

class RawAcquisitionManager:
    
    TEMP_DIR=r'C:\Users\sp3660\Desktop\TemporaryProcessing'

    def __init__(self, dataset_object=None, prairie_single_dataset_directory=None, scanimage_dataset_directory=None):
        
        self.dataset_object=dataset_object
        self.prairie_single_dataset_directory=prairie_single_dataset_directory
        self.scanimage_dataset_directory=scanimage_dataset_directory
        
        #set defaulst
        self.is_mmap_input=False
        self.is_hdf5_input=False
        self.is_raw_input=False
        self.prairire_raw=False
        self.scanimage_raw=False
        self.raw_caiman_movie=np.array([False])

        # run pipeline
        self.define_input_paths()
        self.define_dataset_info()
        self.define_output_paths()
        self.real_saved_mmap_file_path()
        
        self.check_input_files()
        self.check_output_files()
        
        # self.load_output_dataset()
        # self.load_input_dataset()
        
    def define_input_paths(self):  
        
        if self.dataset_object:
            self.raw_dataset_directory_path=self.dataset_object.raw_directory_path
            if 'Prairie' in self.dataset_object.microscope:
                self.prairire_raw=True
            elif 'Hakim' in self.dataset_object.microscope:
                self.scanimage_raw=self.dataset_object.microscope

        elif self.prairie_single_dataset_directory:
            self.raw_dataset_directory_path=self.prairie_single_dataset_directory
            self.prairire_raw=True
        elif self.scanimage_dataset_directory:  
            self.raw_dataset_directory_path=self.scanimage_dataset_directory
            self.scanimage_raw=True
            
        self.raw_dataset_fullfile_paths=glob.glob(self.raw_dataset_directory_path+os.sep+'**.tif')

    def define_dataset_info(self):
        
        if self.dataset_object:
            self.raw_dataset_name=self.dataset_object.name
            self.raw_dataset_plane=self.dataset_object.plane
            self.raw_dataset_channel=self.dataset_object.channel
            self.raw_dataset_frame_number=self.dataset_object.frame_number

        elif self.prairie_single_dataset_directory:
            self.raw_dataset_name=os.path.split(os.path.split(os.path.split( self.raw_dataset_directory_path)[0])[0])[1]
            self.raw_dataset_plane=os.path.split( self.raw_dataset_directory_path)[1]
            self.raw_dataset_channel=os.path.split(os.path.split( self.raw_dataset_directory_path)[0])[1]
            self.raw_dataset_frame_number=len(self.raw_dataset_fullfile_paths)
           
        elif self.scanimage_dataset_directory:  
            self.raw_dataset_name=os.path.split( self.raw_dataset_directory_path)[1]
            self.raw_dataset_plane=''
            self.raw_dataset_channel=''
            self.raw_dataset_frame_number=''

    def define_output_paths(self):  
        
        if self.dataset_object:
            self.raw_mmap_file_path=os.path.join(self.dataset_object.processed_dataset_directory, '_'.join([self.raw_dataset_name, self.raw_dataset_channel, self.raw_dataset_plane])+'.mmap')
            self.raw_hdf5_file_path=os.path.join(self.dataset_object.processed_dataset_directory, '_'.join([self.raw_dataset_name, self.raw_dataset_channel, self.raw_dataset_plane])+'.hdf5')
            
        elif self.prairie_single_dataset_directory or self.scanimage_dataset_directory:
            self.raw_mmap_file_path=os.path.join(self.TEMP_DIR, '_'.join([self.raw_dataset_name, self.raw_dataset_channel, self.raw_dataset_plane])+'.mmap')
            self.raw_hdf5_file_path=os.path.join(self.TEMP_DIR, '_'.join([self.raw_dataset_name, self.raw_dataset_channel, self.raw_dataset_plane])+'.hdf5')
            
    def real_saved_mmap_file_path(self):
        self.saved_raw_mmap_file_path=''
        
        mmap_files=glob.glob(os.path.split(self.raw_mmap_file_path)[0]+os.sep+'**.mmap')
        saved_raw_mmap_file_paths=[fil  for fil in mmap_files if '_'.join([self.raw_dataset_name, self.raw_dataset_channel, self.raw_dataset_plane]) in fil]
        if saved_raw_mmap_file_paths:
            self.saved_raw_mmap_file_path=saved_raw_mmap_file_paths[0]
        
    def check_input_files(self):
        
        self.is_raw_input=all([os.path.isfile(i) for i in self.raw_dataset_fullfile_paths])
        
    def check_output_files(self):
        
        self.is_mmap_input=os.path.isfile( self.saved_raw_mmap_file_path)
        self.is_hdf5_input=os.path.isfile( self.raw_hdf5_file_path)


    def load_input_dataset(self):
        
        if self.is_raw_input and not self.raw_caiman_movie.any():
            
            if self.prairire_raw:
                self.raw_caiman_movie=cm.load( self.raw_dataset_fullfile_paths)

            elif  self.scanimage_raw:
                self.full_raw_acquisition=ScanImageTiffReader(self.raw_dataset_fullfile_paths[0])
                vol=self.full_raw_acquisition
                
                fr = 60
                px = int(15360 / fr)

                lmov = vol.__len__()
                
                data1 = vol.data(0, int(lmov/2) )
                data2 = vol.data(int(lmov/2), lmov)
                
                data1 = data1.reshape([int(lmov/4), 2, px, px])
                data2 = data2.reshape([int(lmov/4), 2, px, px])
                
                ch1 = np.concatenate((data1[200:, 0, :, : ], data2[:-100, 0, :, : ] ), axis = 0)
                ch2 = np.concatenate((data1[200:, 1, :, : ], data2[:-100, 1, :, : ] ), axis = 0)
                

    def load_output_dataset(self):
        if self.is_mmap_input:
            self.raw_caiman_movie=cm.load(self.saved_raw_mmap_file_path)
            
        elif self.is_hdf5_input:
            self.raw_caiman_movie=cm.load(self.raw_hdf5_file_path)

    def save_output_dataset(self, output_format='.mmap' ): 
        
        if 'mmap' in output_format:
            output_path=self.raw_mmap_file_path
            
        elif 'hdf5' in output_format:
            output_path=self.raw_hdf5_file_path
        
        if self.raw_caiman_movie.any() and output_path:
            if not os.path.isfile(output_path):
                self.raw_caiman_movie.save(output_path ,to32=False)  
  
    def unload_dataset(self):
        if self.raw_caiman_movie.any():
            del self.raw_caiman_movie
            gc.collect()
            sys.stdout.flush()
            self.raw_caiman_movie=np.array([False])

   

if __name__ == "__main__":
    
    # dataset_image_sequence_path=r'F:\Projects\LabNY\Imaging\2022\20220428\Mice\SPMT\FOV_1\Aq_1\220428_SPMT_FOV2_AllenA_25x_920_52570_570620_without-000\Ch2Green\plane1'
    # dataset_image_sequence_path=r'F:\Projects\LabNY\Imaging\2022\20220428\Mice\SPMT\FOV_1\SurfaceImage\Aq_1\220428_SPMT_FOV2_Surface_25x_920_52570_570620_without-000\Ch2Green\plane1'
    dataset_image_sequence_path=r'F:\Projects\LabNY\Imaging\2022\20220428\Mice\SPMT\0CoordinateAcquisiton\Aq_1\220428_SPMT_0Coordinate_25x_940_52570_570620_wit-000\Ch2Green\plane1'
    raw_dataset = RawAcquisitionManager(prairie_single_dataset_directory=dataset_image_sequence_path)

    # scanimagedatasetdir=r'F:\Projects\LabNY\Imaging\2022\20220408 Hakim\Mice\SPKU\FOV_\220408_SPKU_FOV1_10minspont_25x_940_1064hakim'
    # raw_dataset = RawAcquisitionManager(scanimage_dataset_directory=scanimagedatasetdir)
    # raw_dataset.raw_caiman_movie.play()
    raw_dataset.load_input_dataset()
    # self.load_input_dataset()
    raw_dataset.save_output_dataset()
    raw_dataset.save_output_dataset('.hdf5')
   


