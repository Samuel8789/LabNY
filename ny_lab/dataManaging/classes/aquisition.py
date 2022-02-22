# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 10:24:33 2021

@author: sp3660
"""
import os
import glob
import shutil
import numpy as np
import pandas as pd
import gc
import sys

from ...AllFunctions.create_dir_structure import create_dir_structure
from .dataset import ImageSequenceDataset
from .voltageSignals import VoltageSignals

from .eyeVideo import EyeVideo
from ..functions.functionsDataOrganization import check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories
# from functionsDataOrganization import check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories
from .standalone.initialProcessing import InitialProcessing
import logging 
module_logger = logging.getLogger(__name__)
from .standalone.metadata import Metadata


class Aquisition:   
    def __init__(self, aqu_name, raw_input_path=None, FOV_object=None, mouse_imaging_session_object=None, atlas_object=None, subaq_object=None, non_imaging=False,):
        self.subaq_object=subaq_object
        self.mouse_imaging_session_object=mouse_imaging_session_object
        self.FOV_object=FOV_object
        self.Atlas_object=atlas_object
        self.non_imaging=non_imaging
        self.raw_input_path=raw_input_path
        self.Atlas=[]
        self.metadata_object=None
        module_logger.info('loading ' + aqu_name)
        self.acquisition_type=None
        self.aquisition_dir_raw_structure=[]    
        self.all_datasets={}
        self.plane_number=0   
        # self.voltage_signals_dictionary={}

        

              
        self.aquisition_structure={'planes',
                             'locomotion',
                             'eye camera',
                             'visual stim',
                             'photodiode',
                             'behaviour',
                             'whiskers camera',
                             'photostim',
                             'metadata',
                             'ref images',
                             'raw_volatge_csv'
                             }    
        
        self.plane_structure={'Green',
                         'Red',
                         'PhotoStimMask',
                         }
        
        if raw_input_path and not non_imaging:
            self.Prairireaqpath=os.path.split(glob.glob(raw_input_path +'\\**\\**.env', recursive=True)[0])[0]      
            self.aquisition_path=os.path.split(self.Prairireaqpath)[0]
            self.aquisition_name= os.path.split(self.Prairireaqpath)[1]
            
        else:
            self.aquisition_name=aqu_name


        self.get_mouse_acquisition_path()
        self.read_acqu_and_plane_structure()

#%%  processing raw datasets 
        if raw_input_path and not non_imaging:

            self.aquisition_dir_raw_structure=self.file_cleanup_prairie_dataset()     
           
            if any(self.aquisition_dir_raw_structure):
                self.read_raw_aquisition_structure()
                self.create_aquisition_structure_in_mouse_file()
                self.read_acqu_and_plane_structure()

                self.raw_main_path_equivalence()
                self.solve_all_datasets()
                # self.unload_voltage_signals()
                
            # dont remember what this is
            else:
                self.read_raw_aquisition_structure()
                self.create_aquisition_structure_in_mouse_file()  
                self.read_acqu_and_plane_structure()
                self.aquisition_path=raw_input_path
                self.solve_all_datasets()
                # self.unload_voltage_signals()

       
#%% non imaging   dataset TO REVIEW          
        elif raw_input_path and non_imaging:
            # module_logger.info('processing raw aquisition '+ self.aquisition_name)  
            self.read_raw_aquisition_structure()
            self.create_aquisition_structure_in_mouse_file()  
            self.read_acqu_and_plane_structure()
            self.aquisition_path=raw_input_path
            self.solve_all_datasets()
            # self.unload_voltage_signals()

            
 #%% reading exiting datasets           
        elif not non_imaging: 
            self.read_acqu_and_plane_structure()
            if not self.mouse_imaging_session_object.yet_to_add:       
                self.get_all_database_info()
                self.load_metadata_from_database()
            self.read_existing_datasets()
            
 #%% nopn imaging no pariaire files, only face camera
        elif non_imaging:                
              module_logger.info('i have to give tit a name to do facecamera')
#%% methods   
#%% dealing with directory structures and paths
#%% slow   
    def get_mouse_acquisition_path(self):
        
           if self.mouse_imaging_session_object:
               
               if self.subaq_object=='TestAquisition':
                   self.mouse_aquisition_path=os.path.join(self.mouse_imaging_session_object.mouse_session_path, 
                                                       'test aquisitions',
                                                        self.aquisition_name)
               if self.subaq_object=='Coordinate0Aquisition':
                  self.mouse_aquisition_path=os.path.join(self.mouse_imaging_session_object.mouse_session_path, 
                                                       '0Coordinate acquisition',
                                                        self.aquisition_name)
               
               if self.subaq_object=='NonimagingAquisition':       
                   self.mouse_aquisition_path=os.path.join(self.mouse_imaging_session_object.mouse_session_path, 
                                                           'nonimaging acquisitions',
                                                            self.aquisition_name)
    
               self.mouse_name = self.mouse_imaging_session_object.mouse_object.mouse_name
               # module_logger.info('associating with mouse object ' + self.mouse_name)           
               if self.mouse_name:
                   self.mouse_object=self.mouse_imaging_session_object.mouse_object
                   
           elif self.Atlas_object:              
               self.Atlas=self.Atlas_object.atlas_name
       
               if self.subaq_object=='AtlasOverview':   
                   self.mouse_aquisition_path=os.path.join(self.Atlas_object.mouse_session_atlas_path, 
                                                           'Overview',
                                                            self.aquisition_name)
               if self.subaq_object=='AtlasPreview':       
                   self.mouse_aquisition_path=os.path.join(self.Atlas_object.mouse_session_atlas_path, 
                                                           'Preview',
                                                            self.aquisition_name)
                  
               if self.subaq_object=='AtlasVolume':       
                   self.mouse_aquisition_path=os.path.join(self.Atlas_object.mouse_session_atlas_path, 
                                                           'Volumes',
                                                            self.aquisition_name)
                   
                   
               self.mouse_name = self.Atlas_object.mouse_imaging_session_object.mouse_object.mouse_name
               # module_logger.info('associating with mouse object ' + self.mouse_name)           
               if self.mouse_name:
                   self.mouse_object=self.Atlas_object.mouse_imaging_session_object.mouse_object    
                                  
           elif self.FOV_object:        
               self.FOV=self.FOV_object.FOV_name
    
               if self.subaq_object==None :
                   self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path,                                                
                                                             self.aquisition_name)
               elif self.subaq_object:
       
                   if self.subaq_object=='Tomato1050Acquisition':
                       self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path, '1050_Tomato',                                               
                                                             self.aquisition_name)
                   if self.subaq_object=='Tomato3Plane1050Acquisition':
                       self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path, '1050_3PlaneTomato',                                               
                                                             self.aquisition_name)
                   if self.subaq_object=='TomatoHighResStack1050Acquisition':
                       self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path, '1050_HighResStackTomato',                                                
                                                             self.aquisition_name)
                   if self.subaq_object =='HighResStackGreenAcquisition':
                       self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path, 'HighResStackGreen',                                               
                                                             self.aquisition_name)
                   if self.subaq_object=='SurfaceImageAquisition':
                       self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path, 'SurfaceImage',                                               
                                                             self.aquisition_name)
                   if self.subaq_object=='OtherAcqAquisition':   
                       self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path, 'OtherAcq',                                               
                                                             self.aquisition_name)
               
       
               self.mouse_name = self.FOV_object.mouse_imaging_session_object.mouse_object.mouse_name
               # module_logger.info('associating with mouse object ' + self.mouse_name)           
               if self.mouse_name:
                   self.mouse_object=self.FOV_object.mouse_imaging_session_object.mouse_object
               self.mouse_imaging_session_object = self.FOV_object.mouse_imaging_session_object
              
    def read_acqu_and_plane_structure(self):
        self.slow_storage_all_paths={}
        for info in self.aquisition_structure:
            self.slow_storage_all_paths[info]=os.path.join(self.mouse_aquisition_path, info)  


        if os.path.isdir(self.slow_storage_all_paths['planes']):
            if len(os.listdir(self.slow_storage_all_paths['planes']))>0:
                   self.plane_number=len(next(os.walk(self.slow_storage_all_paths['planes']))[1])
                   
        self.plane_channel_paths={}
        if self.plane_number>0:
            for plane in range(self.plane_number) :
                self.plane_channel_paths['Plane' + str(plane+1)]={}
                for info in   self.plane_structure:
                    if 'Photo' not in info:
                        self.plane_channel_paths['Plane' + str(plane+1)][info]=os.path.join(self.slow_storage_all_paths['planes'], 'Plane' + str(plane+1), info )
                        
#%% raw  
    def create_aquisition_structure_in_mouse_file(self):
        
       if not os.path.isdir(self.mouse_aquisition_path):
            os.makedirs(self.mouse_aquisition_path)   
      
       create_dir_structure(self.mouse_aquisition_path, self.aquisition_structure)

       self.plane_paths={}       
       if any(self.aquisition_dir_raw_structure):
           for plane in range(self.plane_number) :
                self.plane_paths['Plane' + str(plane+1)]=os.path.join(self.slow_storage_all_paths['planes'], 'Plane' + str(plane+1) )
                if not os.path.isdir(self.plane_paths['Plane' + str(plane+1)]):
                    os.mkdir(self.plane_paths['Plane' + str(plane+1)])
                    create_dir_structure(self.plane_paths['Plane' + str(plane+1)], self.plane_structure)

    def read_raw_aquisition_structure(self):
        
        if self.aquisition_dir_raw_structure:
            if not self.non_imaging :
                self.plane_number=self.aquisition_dir_raw_structure[1]
                self.imaged_channels=[chan for chan in self.aquisition_dir_raw_structure[0] if 'Ch' in chan]
                self.all_raw_image_sequence_paths=self.aquisition_dir_raw_structure[2]
            else:
                pass

        if self.non_imaging :
            aquisition_to_process=self.raw_input_path
        else:
           aquisition_to_process=os.path.join(self.aquisition_path, self.aquisition_name)
           
        temp_path=os.path.split(os.path.split(aquisition_to_process)[0])[0]
        
        aquisition_date=aquisition_to_process[temp_path.find('\SP')-13:temp_path.find('\SP')-5]
        self.formated_aquisition_date=aquisition_date     
        
        self.session_path=os.path.join(self.mouse_object.mouse_slow_subproject_path, 'imaging', self.formated_aquisition_date) 
             
    def raw_main_path_equivalence(self):
         self.dataset_path_equivalences={}
         for dataset_raw_path in sorted(self.all_raw_image_sequence_paths, reverse=True):
             

             channel_info=os.path.split(os.path.split(dataset_raw_path)[0])[1]
             channel=channel_info[channel_info.find('Ch')+3:]
             plane_number=os.path.split(dataset_raw_path)[1][os.path.split(dataset_raw_path)[1].find('plane')+5:] 
             self.dataset_path_equivalences[dataset_raw_path]=[channel, 
                                                               'Plane'+ str(plane_number), 
                                                               os.path.join(self.mouse_aquisition_path,
                                                                            'planes',
                                                                            'Plane'+ str(plane_number),
                                                                             channel)
                                                               ]
       
    def file_cleanup_prairie_dataset(self):
                 
    # file_list = os.listdir(self.Prairireaqpath)
      
      # check channle and plane structure     and current folders
          directory_red=os.path.join(self.Prairireaqpath,'Ch1Red')
          directory_green=os.path.join(self.Prairireaqpath, 'Ch2Green')
          
          correction=False  
          ChannelPaths=[directory_red, directory_green]  
          ChannelRedExists=False
          ChannelGreenExists=False
          PlaneNumber=False
          
          if os.path.exists(directory_red) or os.path.exists(directory_green):            
                aq_info=check_channels_and_planes(self.Prairireaqpath, correction)
                correction=True  
                if os.path.exists(directory_red):
                    ChannelRedExists=True            
                    folder_selected_list_red = os.listdir(directory_red)
                    if any('plane' in file_name  for file_name in folder_selected_list_red if os.path.isdir(os.path.join(directory_red , file_name))):
                        
                        PlaneNumber=len(folder_selected_list_red) + aq_info[9]
                        Multiplane=False
                        if PlaneNumber>1:
                            Multiplane=True
                    else:
                        aq_info =check_channels_and_planes(directory_red, correction)
                    
                if os.path.exists(directory_green):
                    ChannelGreenExists=True           
                    folder_selected_list_green = os.listdir(directory_green)  
      
                    if any('plane' in file_name  for file_name in folder_selected_list_green if os.path.isdir(directory_green + os.sep + file_name)):
                        PlaneNumber=len(folder_selected_list_green) + aq_info[10]
                        Multiplane=False
                        if PlaneNumber>1:
                            Multiplane=True
                    else:
                       aq_info=check_channels_and_planes(directory_green, correction)
      
          else:
              aq_info = check_channels_and_planes(self.Prairireaqpath, correction)
              if aq_info[0]:
                  ChannelRedExists=1
                  PlaneNumber=aq_info[9]
              if aq_info[1]:
                  ChannelGreenExists=1
                  PlaneNumber=aq_info[10]
      
          ImagedChannels=['No','No']
          if ChannelRedExists:
              ImagedChannels[0]='Ch1Red'
          if ChannelGreenExists:
              ImagedChannels[1]='Ch2Green'
      
           # create necessary folders     
          if ChannelRedExists or ChannelGreenExists:     
              all_image_sequence_paths=[]
              PlanePaths=[os.sep +'plane'+str(i+1) for i in range(PlaneNumber)]       
              for ch in ImagedChannels:
                  for i, channel_path in enumerate(ChannelPaths):
                      if ch in channel_path :
                          for n in range(PlaneNumber):
                              all_image_sequence_paths.append(ChannelPaths[i]+PlanePaths[n])
                              if not os.path.exists(ChannelPaths[i]+PlanePaths[n]):
                                  os.makedirs(ChannelPaths[i]+PlanePaths[n])
              # move files  
              # module_logger.info('Moving Files')     
              Multiplane=aq_info[8]
              if correction:
                  if glob.glob(self.Prairireaqpath+'\\**.tif', recursive=False):             
                      move_files(self.Prairireaqpath,ChannelPaths,PlanePaths, Multiplane,aq_info[-1] ) 
                  for channel_folder in ChannelPaths:
                      if os.path.isdir(channel_folder):
                          file_list_channel = os.listdir(channel_folder)
                          
                          if len (file_list_channel)>3:
                               move_files(channel_folder,ChannelPaths,PlanePaths, Multiplane, aq_info[-1]) 
                          elif len(file_list_channel)<3:  
                               for plane_folder in file_list_channel:
                                   if os.path.isdir(plane_folder):
                                       # file_list_plane=os.listdir(plane_folder)
                                       move_files(plane_folder,ChannelPaths,PlanePaths, Multiplane,aq_info[-1] ) 
                    
              else:       
                  move_files(self.Prairireaqpath,ChannelPaths,PlanePaths, Multiplane,aq_info[-1]) 
      
              return  [ImagedChannels, PlaneNumber, all_image_sequence_paths]      
          else:        
              return  [False, False, False]  
#%%main processors     
#%%  main raw loader
    def solve_all_datasets(self):
        try:

            if self.subaq_object!='NonimagingAquisition': 
                
                
                module_logger.info('loading metadata for raw processing')
                self.load_metadata_raw()

                module_logger.info('trasnfering vis stim')               
                self.transfer_vis_stim_info()
                
                module_logger.info('processing voltage recording')           
                self.load_voltage_signals_object()

                # module_logger.info('transfering csv')
                # self.transfer_raw_csv()
                
                module_logger.info('transfering references')
                self.transfer_ref_images()
                
                module_logger.info('processing face camera')
                self.process_face_camera()
                
                module_logger.info('processing raw datasets')

                self.all_raw_datasets={}

                for key, value in self.dataset_path_equivalences.items():
                    if self.FOV_object:
                        dataset_name=self.mouse_name +'_'+ self.formated_aquisition_date +'_'+self.FOV +'_'+self.aquisition_name+'_'+value[1]+'_'+value[0]
                    elif self.mouse_imaging_session_object:  
                        dataset_name=self.mouse_name +'_'+ self.formated_aquisition_date +'_'+self.aquisition_name+'_'+value[1]+'_'+value[0]
                        
                    try:    
                        self.all_raw_datasets[dataset_name]=ImageSequenceDataset(self,
                                                                             dataset_name,
                                                                             selected_dataset_raw_path=key, 
                                                                         selected_dataset_mmap_path=value[2])
                    except:
                        module_logger.exception('dataset not loaded')
                        
            else:
                
                 module_logger.info('to do NON IMGING ACQUISITION ' +  self.raw_input_path +' '+   self.aquisition_name)

        except:
            module_logger.exception('could not solve all acquisition datasets')
            
#%% main reader            
    def read_existing_datasets(self):
        
        # self.load_metadata_slow_working_directories()
        self.load_voltage_signals_object()         
        self.read_working_vistim_info()
        self.read_face_camera()
        
        if self.subaq_object!='NonimagingAquisition': 
            
            for channels in self.plane_channel_paths.values():
                for dataset_path in channels.values():
                    if os.path.isdir(dataset_path):
                        if len(os.listdir(dataset_path))>0:
                            dataset_name= os.path.split(self.mouse_aquisition_path)[1] +'_'+ os.path.split(os.path.split(dataset_path)[0])[1] +'_'+ os.path.split(dataset_path)[1]
                            self.all_datasets[dataset_name]=ImageSequenceDataset(self,
                                                                      dataset_name,                                                               
                                                                      selected_dataset_mmap_path=dataset_path) 
          
            
    def load_all_datasets(self): 
        if self.all_datasets:
            for dataset in self.all_datasets.values():
                dataset.load_dataset()
        
    def unload_all_datasets(self): 
        if self.all_datasets:
            for dataset in self.all_datasets.values():
                dataset.unload_dataset()

#%%other object functions        
#%%  dailing with metadata 


    def get_all_database_info(self):
        self.aq_ID=self.mouse_imaging_session_object.database_acquisitions.loc[self.mouse_imaging_session_object.database_acquisitions['SlowDiskPath'] ==     self.mouse_aquisition_path]['ID'].iloc[0]  
        if self.aq_ID:
            self.full_database_dictionary=self.mouse_imaging_session_object.mouse_object.Database_ref.ImagingDatabase_class.get_single_acquisition_database_info(self.aq_ID)
            self.acquisition_database_info= self.full_database_dictionary['Acq']
            self.imaging_database_info= self.full_database_dictionary['Imaging']
            

       
    def transfer_ref_images(self):
        self.references_raw_files_full_path=[file for file in glob.glob( os.path.join(self.aquisition_path,self.aquisition_name)+'\\References\\**', recursive=False) if '.tif' in file  ]
        self.reference_images_working_fullpaths=[]
        for file in self.references_raw_files_full_path:
            if not os.path.isfile(os.path.join(self.slow_storage_all_paths['ref images'], os.path.split(file)[1])):
            
                self.reference_images_working_fullpaths.append(shutil.copy(file, self.slow_storage_all_paths['ref images']))             
        self.read_existing_datasets()     
        
        
    def load_metadata_raw(self):
        if self.Prairireaqpath:
            self.metadata_object=Metadata(acquisition_directory_raw=self.Prairireaqpath,temporary_path=self.slow_storage_all_paths['metadata'], aquisition_object=self)
               
    def load_metadata_slow_working_directories(self):
        if self.slow_storage_all_paths['metadata']:
            self.metadata_object=Metadata(acquisition_directory_raw=self.slow_storage_all_paths['metadata'],  aquisition_object=self)
            
    def load_metadata_from_database(self):
        self.metadata_object=Metadata(aquisition_object=self, from_database=True)
        pass

#%% dealing with voltage signals 

    def load_voltage_signals_object(self):
        
        self.voltage_signal_object=VoltageSignals(acquisition_object=self)
        
    def load_voltage_signals(self):
       if self.voltage_signal_object:
           module_logger.info('loading voltage signals')

           self.voltage_signal_object.load_slow_storage_voltage_signals()
       else:
           module_logger.info('no voltage signals')

       
    def unload_voltage_signals(self):
        
       self.voltage_signal_object.unload_voltage_signals()
       
       

    # def transfer_raw_csv(self):
    #     self.voltage_recording_raw_file_transfered_path=self.slow_storage_all_paths['raw_volatge_csv']    
    #     self.voltage_recording_raw_file_full_path=[file for file in glob.glob( os.path.join(self.aquisition_path,self.aquisition_name)+'\\**', recursive=False) if 'VoltageRecording' in file and '.csv' in file   ]
    #     if self.voltage_recording_raw_file_full_path and not os.path.isfile(self.voltage_recording_raw_file_transfered_path):
    #         self.voltage_recording_raw_file_full_path=self.voltage_recording_raw_file_full_path[0]
    #         final_file_path=os.path.join(self.voltage_recording_raw_file_transfered_path,os.path.split(self.voltage_recording_raw_file_full_path)[1])
    #         if not os.path.isfile(final_file_path):
    #             shutil.copy(self.voltage_recording_raw_file_full_path, self.voltage_recording_raw_file_transfered_path)
            
    # def process_raw_voltage_recording(self):
    #     self.read_voltage_signals()
       
    #     self.voltage_recording_raw_file_full_path=[file for file in glob.glob( os.path.join(self.aquisition_path,self.aquisition_name)+'\\**', recursive=False) if 'VoltageRecording' in file and '.csv' in file   ]
    #     if self.voltage_recording_raw_file_full_path:
    #         self.voltage_recording_raw_file_full_path=self.voltage_recording_raw_file_full_path[0]
    #         try:
    #             self.voltage_signals = pd.read_csv(self.voltage_recording_raw_file_full_path)
    #             self.voltage_signals_dictionary={signal:self.voltage_signals[signal].to_frame() for signal in self.voltage_signals.columns.tolist()[1:]}
    #         except:
    #             self.voltage_signals_dictionary={'Locomotion': pd.DataFrame({'Locomotion' : []}),
    #                                              'VisStim':pd.DataFrame({'VisStim' : []}),
    #                                              'LED':pd.DataFrame({'LED' : []}),
    #                                              'PhotoDiode':pd.DataFrame({'PhotoDiode' : []})}
    #             module_logger.exception('cretaing dataframes from scratch')

        
    #     for signal, dataf in self.voltage_signals_dictionary.items():
    #         if 'Locomotion' in signal:     
    #             self.voltage_signals_dictionary[signal].to_feather(self.transfered_locomotion_fullpath)
                
    #         elif 'VisStim' in signal:

    #             self.voltage_signals_dictionary[signal].to_feather(self.transfered_visstim_fullpath)           
    #         elif 'LED' in signal:
               
    #             self.voltage_signals_dictionary[signal].to_feather(self.transfered_led_fullpath)
    #         elif 'PhotoDiode' in signal:
                
    #             self.voltage_signals_dictionary[signal].to_feather(self.transfered_photodiode_fullpath)
                
                
    #     self.read_voltage_signals() 
        
    # def read_voltage_signals(self): 
        
    #     self.transfered_locomotion_fullpath=os.path.join(self.slow_storage_all_paths['locomotion'], 'locomotion.ftr')
    #     self.transfered_visstim_fullpath=os.path.join(self.slow_storage_all_paths['visual stim'], 'vis_stim_voltage.ftr')
    #     self.transfered_led_fullpath=os.path.join(self.slow_storage_all_paths['eye camera'], 'led.ftr')
    #     self.transfered_photodiode_fullpath=os.path.join(self.slow_storage_all_paths['photodiode'], 'screen_photodiode.ftr')
        
    # def load_voltage_signals(self):   
    
    #     self.voltage_signals_dictionary={}
    #     if os.path.isfile(self.transfered_locomotion_fullpath) and os.path.getsize(self.transfered_locomotion_fullpath)>0:  
    #         self.voltage_signals_dictionary['Locomotion']=pd.read_feather(self.transfered_locomotion_fullpath)
    #     if os.path.isfile(self.transfered_visstim_fullpath) and os.path.getsize(self.transfered_visstim_fullpath)>0:  
    #         self.voltage_signals_dictionary['VisStim']=pd.read_feather(self.transfered_visstim_fullpath)  
    #     if os.path.isfile(self.transfered_led_fullpath) and os.path.getsize(self.transfered_led_fullpath)>0:  
    #         self.voltage_signals_dictionary['LED']=pd.read_feather(self.transfered_led_fullpath)
    #     if os.path.isfile(self.transfered_photodiode_fullpath) and os.path.getsize(self.transfered_photodiode_fullpath)>0:  
    #         self.voltage_signals_dictionary['PhotoDiode']=pd.read_feather(self.transfered_photodiode_fullpath)  
            
            
    # def unload_voltage_signals(self):
    #     module_logger.info('Unloading voltage signals')

    #     if self.voltage_signals_dictionary:
    #         del self.voltage_signals_dictionary
    #         gc.collect()
    #         # sys.stdout.flush()
                 
            
#%% dealing with face camera 
    def process_face_camera(self):
        try:
            # module_logger.info(self.aquisition_path)
            # module_logger.info(self.aquisition_name)
            self.face_camera_raw_path=os.path.join(self.aquisition_path, 'FaceCamera')
            if os.path.isdir(self.face_camera_raw_path):
                module_logger.info('eye video acquisition')
    
                self.face_camera=EyeVideo(self, selected_eyevideo_raw_path=self.face_camera_raw_path)
            else:
                # pass
                module_logger.info('No Raw Face Camera')
        except:
            module_logger.exception('Somthing wrong with eye video')
        
    def read_face_camera(self):
        self.face_camera=None
        if glob.glob(self.slow_storage_all_paths['eye camera']+'\\**', recursive=False):
            self.face_camera=EyeVideo(self)
        else:
            pass
            # module_logger.info('No Processed Face Camera')
            
    def load_face_camera(self):
        self.read_face_camera()
        if self.face_camera:
            module_logger.info('loading face camera')

            self.face_camera.load_all()
        else:
            module_logger.info('No face camera')

        
    def unload_camera(self):
        self.face_camera.unload_camera_file()
        
#%% dealing with visstim  

    def transfer_vis_stim_info(self):  
        vistimpath=os.path.join(self.aquisition_path, 'VisStim')
        self.raw_vis_stim_info=[]
        if os.path.isdir(vistimpath):
            module_logger.info('ransfering vis stim info')

            for file in os.listdir(vistimpath):
                self.raw_vis_stim_info.append(os.path.join(vistimpath,file))
                if not os.path.isfile(os.path.join(self.slow_storage_all_paths['visual stim'],file)):

                    shutil.copy(os.path.join(vistimpath,file), self.slow_storage_all_paths['visual stim'])
        else:
            module_logger.info('No visstim')

            
        self.read_working_vistim_info()     
        
    def read_working_vistim_info(self): 
        self.all_vis_stim_mat_files=glob.glob(self.slow_storage_all_paths['visual stim']+'\\**.mat', recursive=False)
        
    def load_vis_stim_info(self):
        pass
      
#%% loading unloading all
    def load_all(self):
        self.load_vis_stim_info()
        self.load_face_camera()
        # self.load_voltage_signals()
        self.load_voltage_signals()
        self.load_all_datasets()
        self.load_metadata_slow_working_directories()

        # self.mouse_imaging_session_object.

    def unload_all(self):
        self.unload_camera()
        # self.unload_voltage_signals()
        self.unload_voltage_signals
        self.unload_all_datasets()


#%% raw paths
# aq.Prairireaqpath
# aq.aquisition_path
# aq.all_raw_image_sequence_paths
# aq.face_camera_raw_path
# aq.raw_vis_stim_info


# #%% raw file paths
# aq.voltage_recording_raw_file_full_path
# aq.metadata_raw_files_full_path
# aq.references_raw_files_full_path





# #%% slo working paths
# aq.session_path
# aq.mouse_aquisition_path
# aq.plane_paths

# aq.working_metadata_path
# aq.working_ref_images_path
# aq.working_vistim_path
# aq.working_facecamera_path  
# aq.working_locomotion_path 
# aq.working_photodiode_path
# aq.working_behavioursignal_path
# aq.working_ledsignal_path
# aq.working_metadata_path
# aq.working_ref_images_path
# aq.working_photostim_path
# aq.working_planes_path
# aq.voltage_recording_raw_file_transfered_path


# #%% slo working fullfile paths
# aq.metadata_preprocessed_files_full_path
# aq.transfered_locomotion_fullpath
# aq.transfered_visstim_fullpath
# aq.transfered_led_fullpath
# aq.transfered_photodiode_fullpath
# aq.all_vis_stim_mat_files
# aq.reference_images_working_fullpaths



# #%% higher object references
# aq.mouse_object
# aq.subaq_object
# aq.mouse_imaging_session_object
# aq.FOV_object
# aq.Atlas_object

# #lower object references
# aq.face_camera
# aq.all_raw_datasets




# #%% tags and other
# aq.non_imaging


# #%% names and info
# aq.aquisition_name
# aq.mouse_name
# aq.FOV
# aq.plane_number
# aq.imaged_channels
# aq.formated_aquisition_date
# aq.Atlas

# #%% complex info
# aq.aquisition_dir_raw_structure
# test=aq.aquisition_dir_raw_structure

# aq.dataset_path_equivalences
# test=aq.dataset_path_equivalences


# #%% data
# aq.voltage_signals_dictionary
# test1=aq.voltage_signals_dictionary

# aq.voltage_signals
# test2=aq.voltage_signals