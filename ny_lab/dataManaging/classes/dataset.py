# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:46:42 2021

@author: sp3660
"""
import caiman as cm
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import glob
import tifffile
import logging 
from ScanImageTiffReader import ScanImageTiffReader

module_logger = logging.getLogger(__name__)

try:
    # from ...AllFunctions.save_imagej_hdf5_tiff  import  save_imagej_hdf5
    from ...data_pre_processing.bidicorrect_image import shiftBiDi, biDiPhaseOffsets
    from ..functions.functionsDataOrganization import open_directory,check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories
    # from functionsDataOrganization import check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories
    from .standalone.motionCorrectedKalman import MotionCorrectedKalman
    from .standalone.summaryImages import SummaryImages
    from .standalone.bidiShiftManager import BidiShiftManager
    from .standalone.caimanExtraction import CaimanExtraction
    from .rawAcquisitionManager import RawAcquisitionManager
    
except:

    from standalone.motionCorrectedKalman import MotionCorrectedKalman
    from standalone.summaryImages import SummaryImages
    from standalone.bidiShiftManager import BidiShiftManager
    from standalone.caimanExtraction import CaimanExtraction
    from rawAcquisitionManager import RawAcquisitionManager


class ImageSequenceDataset:
    
    TEMP_DIR=r'C:\Users\sp3660\Desktop\TemporaryProcessing'

    def __init__(self, aquisition_object=None, dataset_name=None, selected_dataset_raw_path=None, selected_dataset_mmap_path=None):
        
        self.associated_aquisiton=aquisition_object    
        self.selected_dataset_raw_path=selected_dataset_raw_path
        self.selected_dataset_mmap_path=selected_dataset_mmap_path
        self.dataset_name=dataset_name

        self.microscope='Prairie'
        self.bidishift_object=None
        self.most_updated_caiman=None
        self.kalman_object=None
        self.summary_images_object=None
        self.initial_caiman=None
        
        self.define_dataset_info()

        if  self.associated_aquisiton:
            self.metadata = self.associated_aquisiton.metadata_object


            if selected_dataset_raw_path:
                # self.raw_dataset_object=RawAcquisitionManager(dataset_object=self)

    
                self.channel=os.path.split(os.path.split(self.selected_dataset_raw_path)[0])[1]
                self.plane=os.path.split(self.selected_dataset_raw_path)[1]
                self.find_associated_channel_dataset()
                self.kalman_movie_path=None
                self.shifted_movie_path=None
            
                module_logger.info('Processing raw data ' + self.selected_dataset_raw_path)
    
                self.dataset_frame_number=len(os.listdir(self.selected_dataset_raw_path))
                # self.process_raw_dataset()
             
    
            else:
                self.read_all_paths()
                module_logger.info('Reading Exiting Datasets')
                
        else:           
            self.raw_dataset_object=RawAcquisitionManager(dataset_object=self)
            self.bidishift_object=BidiShiftManager( raw_dataset_object= self.raw_dataset_object, custom_start_end=True)
            
    def process_raw_dataset(self):
        try :
            open_directory(self.selected_dataset_mmap_path)
            if self.associated_aquisiton.subaq_object=='TestAquisition':
                self.do_bidishift()
                self.do_summary_images(self.shifted_movie_path)


            elif self.associated_aquisiton.subaq_object=='Coordinate0Aquisition':
                self.do_bidishift()

                self.do_summary_images(self.shifted_movie_path)

            elif self.associated_aquisiton.subaq_object=='NonimagingAquisition':
                # self.do_bidishift()

                module_logger.info('Non imaging ')

                
            elif self.associated_aquisiton.Atlas_object:  
                pass
                
            elif self.associated_aquisiton.FOV_object:   
                
                if  self.associated_aquisiton.subaq_object==None or  self.associated_aquisiton.subaq_object=='OtherAcqAquisition':
                    self.do_bidishift()
                    self.do_summary_images(self.shifted_movie_path)
                    module_logger.info('SHort file doing summary images directly')


                elif self.associated_aquisiton.subaq_object=='TomatoHighResStack1050Acquisition' or self.associated_aquisiton.subaq_object=='HighResStackGreenAcquisition':
                    module_logger.info('High Res Stack')
                    self.do_bidishift()
                    self.do_summary_images(self.shifted_movie_path)
            
                else:
                    self.do_bidishift()
                    self.do_summary_images(self.shifted_movie_path)
                    
            # self.unload_dataset()        
            module_logger.info('Finished processing dataset ' + self.selected_dataset_raw_path 
                               + '_____' + self.selected_dataset_mmap_path)
            
            self.unload_dataset()

        except:
            module_logger.exception('Something wrong with raw processing ' +   self.selected_dataset_raw_path)
        
        
        
    

    
    def full_process_raw_dataset(self):
        try :
            if self.associated_aquisiton.subaq_object=='TestAquisition':
                self.do_bidishift()

                if  self.dataset_frame_number>100:
                    if 'Red' in  self.selected_dataset_raw_path and self.associated_channel_dataset_object:
                        self.apply_green_MC_to_red_channel()
                    else:

                        self.do_initial_caiman_extraction()
                    self.do_initial_kalman(self.mc_onacid_path)
                    self.do_summary_images(self.kalman_movie_path)
                    self.do_summary_images(self.gauss_path)

                    
                elif self.dataset_frame_number>1:   
                    self.do_summary_images(self.shifted_movie_path)
                    module_logger.info('SHort file doing summary images directly')

                else:
                    module_logger.info('File Problem')

            elif self.associated_aquisiton.subaq_object=='Coordinate0Aquisition':
                self.do_bidishift()

                self.do_summary_images(self.shifted_movie_path)

            elif self.associated_aquisiton.subaq_object=='NonimagingAquisition':
                # self.do_bidishift()

                module_logger.info('Non imaging ')

                
            elif self.associated_aquisiton.Atlas_object:  
                pass
                
            elif self.associated_aquisiton.FOV_object:   
                
                if  self.associated_aquisiton.subaq_object==None or  self.associated_aquisiton.subaq_object=='OtherAcqAquisition':
                    self.do_bidishift()

                    if  self.dataset_frame_number>100:
                        if 'Red' in  self.selected_dataset_raw_path and self.associated_channel_dataset_object:
                            self.apply_green_MC_to_red_channel()
                        else:
                            self.do_initial_caiman_extraction()
                        self.do_initial_kalman(self.mc_onacid_path)
                        self.do_summary_images(self.kalman_movie_path)
                        
                        self.do_summary_images(self.gauss_path)
                        
                    elif self.dataset_frame_number>1:   
                        self.do_summary_images(self.shifted_movie_path)
                        module_logger.info('SHort file doing summary images directly')

                    else:
                        module_logger.info('File Problem')


                elif self.associated_aquisiton.subaq_object=='TomatoHighResStack1050Acquisition' or self.associated_aquisiton.subaq_object=='HighResStackGreenAcquisition':
                    module_logger.info('High Res Stack')
                    self.do_bidishift()
                    self.do_summary_images(self.shifted_movie_path)
            
                else:
                    self.do_bidishift()

                    self.do_summary_images(self.shifted_movie_path)
                    
            # self.unload_dataset()        
            module_logger.info('Finished processing dataset ' + self.selected_dataset_raw_path 
                               + '_____' + self.selected_dataset_mmap_path)

        except:
            module_logger.exception('Something wrong with raw processing ' +   self.selected_dataset_raw_path)
        
        
        pass
        
        
    def load_motion_corrected_movie(self):
        self.mc_movie=cm.load(self.mc_onacid_path)
        
        
        
        
    def define_dataset_info(self):
        
        
        if self.associated_aquisiton and self.selected_dataset_raw_path:
            
            self.raw_directory_path=self.selected_dataset_raw_path
            self.name=os.path.split(os.path.split(os.path.split( self.selected_dataset_raw_path)[0])[0])[1]
            self.plane=os.path.split(os.path.split(self.selected_dataset_raw_path)[0])[1]
            self.channel=os.path.split(self.selected_dataset_raw_path)[1]
            self.frame_number=len(glob.glob(self.selected_dataset_raw_path+'\\**.tif'))
            self.processed_dataset_directory=self.selected_dataset_mmap_path
            
            
        elif self.associated_aquisiton and not self.selected_dataset_raw_path:
            
            self.raw_directory_path=self.selected_dataset_raw_path
            self.name=os.path.split(os.path.split(os.path.split(os.path.split( self.selected_dataset_mmap_path)[0])[0])[0])[1]
            plane=os.path.split(os.path.split(self.selected_dataset_mmap_path)[0])[1]
            self.plane=plane.lower()
            self.channel=os.path.split(self.selected_dataset_mmap_path)[1]
            self.processed_dataset_directory=self.selected_dataset_mmap_path
        
        elif self.selected_dataset_raw_path:
            
            self.raw_directory_path=self.selected_dataset_raw_path
            self.name=os.path.split(os.path.split(os.path.split( self.selected_dataset_raw_path)[0])[0])[1]
            self.plane=os.path.split( self.selected_dataset_raw_path)[1]
            self.channel=os.path.split(os.path.split( self.selected_dataset_raw_path)[0])[1]
            self.frame_number=len(glob.glob(self.selected_dataset_raw_path+'\\**.tif'))
            self.processed_dataset_directory=self.TEMP_DIR
            

        
        #processing
        # bidishift
        
        # motioncorrection
        #     caiman(on acid)
        #     suite2p
            
        # visualizations
        #     smoothedgaussain bidishifted_MC
        #     kalman bidishifted_MC
            
        # projections 
        #     projections smoothed
        #     projection kalman
        #     projections bidishifted_MC
            
        # caiman_processing
        #     data_dir=data/caiman
        #     initial
        #     deeep_onacod
        #     standard_cnmf
            
        # suite2p_processing
        #     data_dir=data/caiman
        #     initial
        #     deeep_onacod
        #     standard_cnmf
            
 
    def read_all_paths(self):
 
        self.bidishift_object=BidiShiftManager(temporary_path=self.selected_dataset_mmap_path , dataset_object=self, custom_start_end=True)
        self.shifted_movie_path=self.bidishift_object.shifted_movie_full_caiman_path        
        self.most_updated_caiman=CaimanExtraction(temporary_path=self.selected_dataset_mmap_path, dataset_object=self)
        self.mc_onacid_path=self.most_updated_caiman.mc_onacid_path
        self.kalman_object=MotionCorrectedKalman(dataset_object=self)                                           
        self.summary_images_object=SummaryImages(dataset_object=self)

    def load_dataset(self, kalman=True):
        # if self.bidishift_object:
        #     self.bidishift_object.load_shifted_movie_from_mmap()
        #     self.bidishift_object.load_bidiphases_from_file()
            
        if self.most_updated_caiman:   
             self.most_updated_caiman.load_cnmf_object()
             
        if kalman and self.kalman_object:
            self.kalman_object.load_mc_kalman_tiff()
            
        if self.summary_images_object:
            self.summary_images_object.load_projections()
    
    def unload_dataset(self):
        if self.bidishift_object:
            self.bidishift_object.unload_shifted_movie()    
            self.bidishift_object.unload_bidishifts()
            
        if self.most_updated_caiman:   
            self.most_updated_caiman.unload_cnmf_object()
            
        # if self.initial_caiman:   
        #     self.initial_caiman.unload_cnmf_object()
            
        if self.kalman_object:
            self.kalman_object.unload_kalman_movie()
            
        if self.summary_images_object:
            self.summary_images_object.unload_summary_images()
    

    
    def read_issues_file(self):
        
        
        pass     

    def do_bidishift(self, force=False):  
        module_logger.info('Bidishifting ' + self.selected_dataset_raw_path)

        self.bidishift_object=BidiShiftManager(dataset_image_sequence_path=self.selected_dataset_raw_path, temporary_path=self.selected_dataset_mmap_path , dataset_object=self, custom_start_end=True, force=force)
        self.shifted_movie_path=self.bidishift_object.shifted_movie_full_caiman_path        

    def do_initial_caiman_extraction(self):
        self.initial_caiman_full=None
        self.initial_caiman_custom=None
        self.initial_caiman=None
        try:
            module_logger.info('Running Initial Caiman ' + self.dataset_name)
            
            if self.bidishift_object.start_end_flag and self.shifted_movie_path and self.bidishift_object.shifted_movie_full_caiman_path==self.bidishift_object.shifted_movie_custom_files_path:
                module_logger.info('Running Caiman custom length ' + self.selected_dataset_raw_path)
                self.initial_caiman_custom=CaimanExtraction(self.shifted_movie_path, temporary_path= self.selected_dataset_mmap_path, first_pass_mot_correct=True, force_run=True, dataset_object=self)
                self.initial_caiman=  self.initial_caiman_custom
            elif not self.bidishift_object.start_end_flag and self.shifted_movie_path and self.bidishift_object.shifted_movie_full_caiman_path==self.bidishift_object.shifted_movie_files_path:
                module_logger.info('Running Unaltered Caiman ' + self.dataset_name)
                self.initial_caiman_full=CaimanExtraction(self.shifted_movie_path, temporary_path=self.selected_dataset_mmap_path, first_pass_mot_correct=True, dataset_object=self)
                self.initial_caiman= self.initial_caiman_full
            if self.initial_caiman:
                self.mc_onacid_path=self.initial_caiman.mc_onacid_path
                self.initial_caiman.load_cnmf_object()
        except:
            self.bidishift_object.load_shifted_movie_from_mmap()
            self.shifted_movie_path=self.bidishift_object.shifted_movie_full_caiman_path
            self.initial_caiman_full=CaimanExtraction(self.shifted_movie_path, temporary_path=self.selected_dataset_mmap_path, first_pass_mot_correct=True, dataset_object=self)
            self.initial_caiman= self.initial_caiman_full

            module_logger.exception('Something wrong with On Acid '  +  self.dataset_name)

    def do_initial_kalman(self,shifted_mmap_path): 
        try:
            module_logger.info('Running Kalman Filter ' + self.dataset_name)
            self.kalman_object=MotionCorrectedKalman(shifted_mmap_path=shifted_mmap_path, dataset_object=self)                                           
            self.kalman_movie_path =self.kalman_object.kalman_path
            self.gauss_path =self.kalman_object.gauss_path                
        except:
            module_logger.exception('Something wrong with kalman ' +   self.dataset_name)

    def do_summary_images(self, source_movie_path): 
        try:
            
            module_logger.info('Getting  summary images ' + self.dataset_name)

            if source_movie_path==self.kalman_movie_path:
                module_logger.info('Getting kalman summary images ' + self.dataset_name)
                
            elif source_movie_path==self.shifted_movie_path:
                module_logger.info('Getting non kalman summary images ' + self.dataset_name)

            if   source_movie_path: 
                self.summary_images_object=SummaryImages(image_sequence_path= source_movie_path, dataset_object=self)
            else:
                module_logger.warning('FIle for summary images worng ' + self.dataset_name)
        except:
            module_logger.exception('Something wrong with non kalman summary images ' +   self.dataset_name)        
                     
    def apply_green_bidishifts_to_red_channel(self):
        print('doing')
        
    def check_for_mc_with_green_files(self):
        
        self.mc_with_green_file=None                     
        self.mc_with_green_file_full=None
        self.mc_with_green_file_custom=None

        corrected_files_full=glob.glob(self.selected_dataset_mmap_path+'\\**Movie_MC_OnACID_green_shifts_d1**.mmap')
        corrected_files_custom=glob.glob(self.selected_dataset_mmap_path+'\\**custom_start**MC_OnACID_green_shifts_d1**.mmap')
  
        if corrected_files_full:
            self.mc_with_green_file_full=corrected_files_full[0]
            
        if corrected_files_custom:
            self.mc_with_green_file_custom=corrected_files_custom[0]

        if   self.mc_with_green_file_custom:
            self.mc_with_green_file=self.mc_with_green_file_custom
        elif  self.mc_with_green_file_full:
            self.mc_with_green_file=self.mc_with_green_file_full
        
        
    def apply_green_MC_to_red_channel(self):
        try:
   
            if self.associated_channel_dataset_object:
                # check equovalent green dataset

                shifts=self.associated_channel_dataset_object.initial_caiman.cnm_object.estimates.shifts[-self.associated_channel_dataset_object.initial_caiman.cnm_object.estimates.C.shape[-1]:]   
                mmap_directory, caiman_filename=os.path.split(self.shifted_movie_path)
                good_filename=caiman_filename[:caiman_filename.find('_d1_')]  

                MC_onacid_file_path='_'.join([os.path.join(self.selected_dataset_mmap_path, good_filename),'MC_OnACID_green_shifts'])
               
                self.check_for_mc_with_green_files()
      
                if not self.mc_with_green_file:
                    images = cm.load( self.shifted_movie_path)
                    module_logger.info('motion correcting red with green ' +   self.selected_dataset_raw_path)
                    self.mc_onacid_path= cm.motion_correction.apply_shift_online(images, shifts,
                                                                        save_base_name=MC_onacid_file_path)
                    self.mc_with_green_file=self.mc_onacid_path
                    
                else:
                    module_logger.info('file already corrected ' +   self.selected_dataset_raw_path)

                self.remove_uncliped_mc_with_gren_file()
                self.check_for_mc_with_green_files()
                self.mc_onacid_path=self.mc_with_green_file

        except:
            module_logger.exception('Something wrong with correcting red with green ' +   self.selected_dataset_raw_path)
            
    def remove_uncliped_mc_with_gren_file(self):    
        
        if self.mc_with_green_file_custom and self.mc_with_green_file_full :
                if os.path.isfile(self.mc_with_green_file_full ):
                    os.remove(self.mc_with_green_file_full )

    def find_associated_channel_dataset(self):
        if self.selected_dataset_raw_path:
            try:
                self.associated_channel_dataset_object=None
                module_logger.info('getting associated channel dataset ' +   self.selected_dataset_raw_path)
                self.assocated_channel_raw_paths=[key for key in self.associated_aquisiton.dataset_path_equivalences.keys() if (self.plane in key) and   (self.channel not in key)]
                if  self.assocated_channel_raw_paths:
                    self.assocated_channel_raw_path= self.assocated_channel_raw_paths[0]
                    self.assocated_channel_working_path=self.associated_aquisiton.dataset_path_equivalences[ self.assocated_channel_raw_path][2]
    
                    self.associated_dataset_name=   self.dataset_name[:self.dataset_name.find('_Plane')]+'_'+\
                                    self.associated_aquisiton.dataset_path_equivalences[ self.assocated_channel_raw_path][1]+'_'+\
                                    self.associated_aquisiton.dataset_path_equivalences[ self.assocated_channel_raw_path][0]
                                
                    if self.associated_aquisiton.all_raw_datasets:            
                        if  self.associated_dataset_name in self.associated_aquisiton.all_raw_datasets.keys():
                            self.associated_channel_dataset_object=self.associated_aquisiton.all_raw_datasets[ self.associated_dataset_name]
                            module_logger.info('gotten associated channel dataset ' +  self.dataset_name+' '+  self.associated_dataset_name)
                        else:    
                            module_logger.info('no green ' + self.selected_dataset_raw_path)
                    else:
                        module_logger.info('no green ' + self.selected_dataset_raw_path)
                else:
                    module_logger.info('no green associated ' + self.selected_dataset_raw_path)
            except:
                module_logger.exception('Something wrong with getting associated dataset ' +   self.selected_dataset_raw_path)
        else:      
            self.associated_channel_datasets={key:dataset for key, dataset in self.associated_aquisiton.all_datasets.items() if self.plane in key and 'Red' in key}
            self.associated_channel_dataset_object=list(self.associated_channel_datasets.values())[0]
  
    def find_associated_fov_tomato_dataset(self):   
        
        self.associated_aquisiton.get_associated_tomato_fov_datasets()        
        self.plane=os.path.split(os.path.split(self.selected_dataset_mmap_path)[0])[1]
        self.associated_fov_tomato_datasets={key:dataset for key, dataset in self.associated_aquisiton.associated_fov_tomato_acquisition.all_datasets.items() if self.plane in key}
        
        for dataset_name in list(   self.associated_fov_tomato_datasets.keys()):
            if  'Green' in dataset_name:
                self.associated_fov_tomato_dataset_green_object=self.associated_aquisiton.associated_fov_tomato_acquisition.all_datasets[dataset_name]
            elif 'Red'  in dataset_name:
                self.associated_fov_tomato_dataset_red_object=self.associated_aquisiton.associated_fov_tomato_acquisition.all_datasets[dataset_name]
         
    def  redo_initial_caiman(self):
        # this i dont know why is here
        self.initial_caiman=CaimanExtraction(self.most_updated_caiman.mc_onacid_path, temporary_path=self.selected_dataset_mmap_path, metadata_object=self.metadata, force_run=True)       


    def do_deep_caiman(self, new_parameters: dict=None):
        
        # self.deep_caiman=CaimanExtraction(self.most_updated_caiman.mc_onacid_path, temporary_path=self.selected_dataset_mmap_path, metadata_object=self.metadata, force_run=True, deep=True, new_parameters=new_parameters)
        self.deep_caiman=CaimanExtraction(self.mc_onacid_path, temporary_path=self.selected_dataset_mmap_path, metadata_object=self.metadata, force_run=True, deep=True, new_parameters=new_parameters)
        # self.read_all_paths()

        
        
            
    def galois_caiman(self, new_parameters: dict=None):
        
        self.deep_caiman=CaimanExtraction(self.most_updated_caiman.mc_onacid_path, temporary_path=self.selected_dataset_mmap_path, metadata_object=self.metadata, new_parameters=new_parameters, galois=True)
        
    def open_dataset_directory(self):
        os.startfile(self.selected_dataset_mmap_path)

    def open_raw_video_on_imagej(self):
        pass
    
if __name__ == "__main__":
    
    # dataset_image_sequence_path=r'F:\Projects\LabNY\Imaging\2022\20220428\Mice\SPMT\FOV_1\Aq_1\220428_SPMT_FOV2_AllenA_25x_920_52570_570620_without-000\Ch2Green\plane1'
    dataset_image_sequence_path=r'F:\Projects\LabNY\Imaging\2022\20220428\Mice\SPMT\FOV_1\SurfaceImage\Aq_1\220428_SPMT_FOV2_Surface_25x_920_52570_570620_without-000\Ch2Green\plane1'
    # dataset_image_sequence_path=r'F:\Projects\LabNY\Imaging\2022\20220428\Mice\SPMT\0CoordinateAcquisiton\Aq_1\220428_SPMT_0Coordinate_25x_940_52570_570620_wit-000\Ch2Green\plane1'
    temp=r'C:\Users\sp3660\Desktop\TemporaryProcessing\220428_SPMT_0Coordinate_25x_940_52570_570620_wit-000_Ch2Green_plane1_d1_512_d2_512_d3_1_order_F_frames_19_.mmap'

    dataset = ImageSequenceDataset(  selected_dataset_raw_path=dataset_image_sequence_path, selected_dataset_mmap_path=None)



