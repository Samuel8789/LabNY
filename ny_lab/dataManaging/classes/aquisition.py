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
import tifffile
import time
import itertools
import scipy.io as spio
from pathlib import Path
# import matlab.engine
import keyboard

from ...AllFunctions.create_dir_structure import create_dir_structure
from .dataset import ImageSequenceDataset
from .voltageSignals import VoltageSignals
from ...data_analysis.resultsAnalysis import ResultsAnalysis


from .eyeVideo import EyeVideo
from ..functions.functionsDataOrganization import check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories
# from functionsDataOrganization import check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories
# from .standalone.initialProcessing import InitialProcessing
import logging 
module_logger = logging.getLogger(__name__)
from .standalone.metadata import Metadata

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


class Aquisition:   
    def __init__(self, aqu_name, raw_input_path=None, FOV_object=None, mouse_imaging_session_object=None, atlas_object=None, subaq_object=None, non_imaging=False,scanimage_raw_path=False):
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
        self.full_database_dictionary={}
        self.scanimage_raw_path=scanimage_raw_path
        # self.voltage_signals_dictionary={}
        self.all_raw_datasets={}
        self.Prairireaqpath=None
        

              
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
                             'raw_volatge_csv',
                             'voltage_signals_daq'
                             }    
        
        self.plane_structure={'Green',
                         'Red',
                         'PhotoStimMask',
                         }
        
        if raw_input_path and not non_imaging:
            self.Prairireaqpath=os.path.split(glob.glob(raw_input_path +os.sep+'**'+os.sep+'**.env', recursive=True)[0])[0]      
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
              
              
              
        if self.scanimage_raw_path:
            
            self.scanimage_raw_path=r'F:\Projects\LabNY\Imaging\2022\20220525Hakim\Mice\SPKU\FOV_1\Aq_1'
            
            self.scanimagepath=os.path.split(glob.glob(scanimage_raw_path +os.sep+'**'+os.sep+'**.csv', recursive=True)[0])[0]      
            self.aquisition_path=os.path.split(self.scanimagepath)[0]
            self.aquisition_name= os.path.split(self.scanimagepath)[1]
            
            
            
            
            
            pass
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
                   
        if '1050_HighResStackTomato' in self.mouse_aquisition_path or 'HighResStackGreen' in self.mouse_aquisition_path:
            acq_type_name='Volume'
        else:
            acq_type_name='Plane'           
                   
                   
        self.plane_channel_paths={}
        if self.plane_number>0:
            for plane in range(self.plane_number) :
                self.plane_channel_paths[acq_type_name + str(plane+1)]={}
                for info in   self.plane_structure:
                    if 'Photo' not in info:
                        self.plane_channel_paths[acq_type_name + str(plane+1)][info]=os.path.join(self.slow_storage_all_paths['planes'], acq_type_name + str(plane+1), info )
                        
#%% raw  

    def create_aquisition_structure_in_mouse_file(self):
        
       if not os.path.isdir(self.mouse_aquisition_path):
            os.makedirs(self.mouse_aquisition_path)   
      
       create_dir_structure(self.mouse_aquisition_path, self.aquisition_structure)

       self.plane_paths={}       
       if any(self.aquisition_dir_raw_structure):
           for plane in range(self.plane_number) :
                if '1050_HighResStackTomato' in self.mouse_aquisition_path or 'HighResStackGreen' in self.mouse_aquisition_path:
                    acq_type_name='Volume'
                else:
                    acq_type_name='Plane'

                self.plane_paths[acq_type_name + str(plane+1)]=os.path.join(self.slow_storage_all_paths['planes'], acq_type_name + str(plane+1) )

                if not os.path.isdir(self.plane_paths[acq_type_name + str(plane+1)]):
                    os.mkdir(self.plane_paths[acq_type_name + str(plane+1)])
                    create_dir_structure(self.plane_paths[acq_type_name + str(plane+1)], self.plane_structure)

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
        
        aquisition_date=aquisition_to_process[temp_path.find(os.sep+'SP')-13:temp_path.find(os.sep+'SP')-5]
        self.formated_aquisition_date=aquisition_date     
        
        self.session_path=os.path.join(self.mouse_object.mouse_slow_subproject_path, 'imaging', self.formated_aquisition_date) 
             
    def raw_main_path_equivalence(self):
         self.dataset_path_equivalences={}
         if '1050_HighResStackTomato' in self.mouse_aquisition_path or 'HighResStackGreen' in self.mouse_aquisition_path:
             acq_type_name='Volume'
             acq_type_name2=acq_type_name
         else:
             acq_type_name='plane'
             acq_type_name2='Plane'


         for dataset_raw_path in sorted(self.all_raw_image_sequence_paths, reverse=True):
             channel_info=os.path.split(os.path.split(dataset_raw_path)[0])[1]
             channel=channel_info[channel_info.find('Ch')+3:]
             
             
             plane_number=os.path.split(dataset_raw_path)[1][os.path.split(dataset_raw_path)[1].find(acq_type_name)+len(acq_type_name):] 
             self.dataset_path_equivalences[dataset_raw_path]=[channel, 
                                                               acq_type_name2+ str(plane_number), 
                                                               os.path.join(self.mouse_aquisition_path,
                                                                            'planes',
                                                                            acq_type_name2+ str(plane_number),
                                                                             channel)
                                                               ]
       
    def file_cleanup_prairie_dataset(self):
                 
      
      # check channle and plane structure     and current folders
          directory_red=os.path.join(self.Prairireaqpath,'Ch1Red')
          directory_green=os.path.join(self.Prairireaqpath, 'Ch2Green')
          
          correction=False  
          ChannelPaths=[directory_red, directory_green]  
          ChannelRedExists=False
          ChannelGreenExists=False
          PlaneNumber=False
          
          # IF Already have done some processing it can be a numbe rof things
          
          if os.path.exists(directory_red) or os.path.exists(directory_green):   
                correction=True  
                #here is soem what probmelatic
                aq_info=check_channels_and_planes(self.Prairireaqpath, correction)
                
                if os.path.exists(directory_red):
                    ChannelRedExists=True            
                    folder_selected_list_red = os.listdir(directory_red)
                    if any('plane' in file_name  for file_name in folder_selected_list_red if os.path.isdir(os.path.join(directory_red , file_name))):
                        
                        if aq_info[9]:
                            PlaneNumber= aq_info[9]
                        else:
                            PlaneNumber=len(folder_selected_list_red) + aq_info[9]

                    elif any('Volume' in file_name  for file_name in folder_selected_list_red if os.path.isdir(os.path.join(directory_red , file_name))):
                        last_cycle=len(folder_selected_list_red) + aq_info[9]
                        PlaneNumber=len(glob.glob(os.path.join(directory_red,folder_selected_list_red[0])+os.sep+'**'))     
                        Multiplane=False
        
                    else:
                        aq_info =check_channels_and_planes(directory_red, correction)
                    
                if os.path.exists(directory_green):
                    ChannelGreenExists=True           
                    folder_selected_list_green = os.listdir(directory_green)  
      
                    if any('plane' in file_name  for file_name in folder_selected_list_green if os.path.isdir(directory_green + os.sep + file_name)):
                        # PlaneNumber=len(folder_selected_list_green) + aq_info[10]
                        
                        if aq_info[10]:
                            PlaneNumber= aq_info[10]
                        else:
                            PlaneNumber=len(folder_selected_list_green) + aq_info[9]
                
                            
                    elif any('Volume' in file_name  for file_name in folder_selected_list_green if os.path.isdir(os.path.join(directory_green , file_name))):
                        last_cycle=len(folder_selected_list_green) + aq_info[10]
                        PlaneNumber=len(glob.glob(os.path.join(directory_green,folder_selected_list_green[0])+os.sep+'**'))     
                    
                    else:
                       aq_info=check_channels_and_planes(directory_green, correction)
               
      
          else:
              aq_info = check_channels_and_planes(self.Prairireaqpath, correction)
              
          if aq_info[0]:
                ChannelRedExists=1
                PlaneNumber=aq_info[9]     
                first_cycle=  int(aq_info[3][5:] ) 
                last_cycle=  int(aq_info[4][5:]   )
    
          if aq_info[1]:
                ChannelGreenExists=1
                PlaneNumber=aq_info[10]
                first_cycle=  int(aq_info[6][5:])
                last_cycle=  int(aq_info[7] [5:] )  
                
          Multiplane=False
          if PlaneNumber>1:
              Multiplane=True     

          ImagedChannels=['No','No']
          if ChannelRedExists:
              ImagedChannels[0]='Ch1Red'
          if ChannelGreenExists:
              ImagedChannels[1]='Ch2Green'
      
           # create necessary folders     
          if (ChannelRedExists or ChannelGreenExists) and PlaneNumber<6:     
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
              time.sleep(10)
              if correction:
                  if glob.glob(self.Prairireaqpath+os.sep+'**.tif', recursive=False):             
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
          
          elif (ChannelRedExists or ChannelGreenExists) and PlaneNumber>6:    
                all_image_sequence_paths=[]

                CyclesPaths=[os.sep +'Volume'+str(i+1) for i in range(last_cycle)]       
                for ch in ImagedChannels:
                    for i, channel_path in enumerate(ChannelPaths):
                          if ch in channel_path :
                              for n,j in enumerate(CyclesPaths):
                                  all_image_sequence_paths.append(ChannelPaths[i]+CyclesPaths[n])
                                  if not os.path.exists(ChannelPaths[i]+CyclesPaths[n]):
                                      os.makedirs(ChannelPaths[i]+CyclesPaths[n])
                             
                Multiplane=aq_info[8]
                if correction:
                    if glob.glob(self.Prairireaqpath+os.sep+'**.tif', recursive=False):             
                        move_files(self.Prairireaqpath,ChannelPaths,CyclesPaths, Multiplane,aq_info[-1] ) 
                    for channel_folder in ChannelPaths:
                        if os.path.isdir(channel_folder):
                            file_list_channel = os.listdir(channel_folder)
                            
                            if len (file_list_channel)>last_cycle:
                                  move_files(channel_folder,ChannelPaths,CyclesPaths, Multiplane, aq_info[-1]) 
                            elif len(file_list_channel)<3:  
                                  for volume_folder in file_list_channel:
                                      if os.path.isdir(os.path.join(channel_folder, volume_folder)):
                                          # file_list_plane=os.listdir(plane_folder)
                                          pass
                                          # move_files(os.path.join(channel_folder, volume_folder),ChannelPaths,CyclesPaths, Multiplane,aq_info[-1] ) 
                      
                else:       
                    move_files(self.Prairireaqpath, ChannelPaths, CyclesPaths, Multiplane, aq_info[-1],  is_highstack=True) 
                
                VolumeNumber=last_cycle
        
                return  [ImagedChannels, VolumeNumber, all_image_sequence_paths]     
            
          else:        
              return  [[], False, []]  
          
            
          
    def break_up_clean_up(self):
        pass
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
        self.read_reference_images()
        
        if self.subaq_object!='NonimagingAquisition': 
            
            for channels in self.plane_channel_paths.values():
                for dataset_path in channels.values():
                    if os.path.isdir(dataset_path):
                        if len(os.listdir(dataset_path))>1:
                            dataset_name= os.path.split(self.mouse_aquisition_path)[1] +'_'+ os.path.split(os.path.split(dataset_path)[0])[1] +'_'+ os.path.split(dataset_path)[1]
                            self.all_datasets[dataset_name]=ImageSequenceDataset(self,
                                                                      dataset_name,                                                               
                                                                      selected_dataset_mmap_path=dataset_path) 
          
            
    def load_all_datasets(self, kalman=True): 
        if self.all_datasets:
            for dataset in self.all_datasets.values():
                dataset.load_dataset(kalman=kalman)
        
    def unload_all_datasets(self): 
        if self.all_datasets:
            for dataset in self.all_datasets.values():
                dataset.unload_dataset()

#%%other object functions        
#%%  dailing with metadata 


    def get_all_database_info(self):
        
        
      
        df=pd.DataFrame([self.mouse_imaging_session_object.mouse_object.data_managing_object.os_transform_databasepath(i) for i in self.mouse_imaging_session_object.database_acquisitions['SlowDiskPath'].values])
        try:
            self.aq_ID=self.mouse_imaging_session_object.database_acquisitions.loc[df.iloc[:,0]==self.mouse_aquisition_path]['ID'].iloc[0]
        except:
            print('error loading Acquisitoin info:'+self.mouse_aquisition_path)
        
        # self.aq_ID=self.mouse_imaging_session_object.database_acquisitions.loc[self.mouse_imaging_session_object.database_acquisitions['SlowDiskPath']==self.mouse_aquisition_path]['ID'].iloc[0]
        
        # print(self.aq_ID)

        # if self.aq_ID==561:
        #     print(self.aq_ID)
        if self.aq_ID:
            self.full_database_dictionary=self.mouse_imaging_session_object.mouse_object.Database_ref.ImagingDatabase_class.get_single_acquisition_database_info(self.aq_ID)
            self.acquisition_database_info= self.full_database_dictionary['Acq']
            self.imaging_database_info= self.full_database_dictionary['Imaging']
            # self.database_acq_raw_path=Path(self.acquisition_database_info.loc[0, 'AcquisitonRawPath']).resolve()
            self.database_acq_raw_path=Path(self.mouse_imaging_session_object.mouse_object.data_managing_object.os_transform_databasepath(self.acquisition_database_info.loc[0, 'AcquisitonRawPath'])).resolve()

            
            if glob.glob(str(self.database_acq_raw_path)+os.sep+'**'):
                self.database_acq_raw_path= Path(glob.glob(str(self.database_acq_raw_path)+os.sep+'**')[0])
            else:
                self.database_acq_raw_path=None
               
        
    def load_reference_images(self):
        self.reference_image_dic={title:'' for title in self.reference_images_working_fullpaths}
        for x in  self.reference_images_working_fullpaths:
            with tifffile.TiffFile(x) as tffl:
                 self.reference_image_dic[x] = tffl.asarray()
                 
    def unload_reference_images(self):
        
        if self.reference_image_dic:
            del self.reference_image_dic
            gc.collect()
            # sys.stdout.flush()
            self.reference_image_dic={}

    def read_reference_images(self):
        
        self.reference_images_working_fullpaths=glob.glob(self.slow_storage_all_paths['ref images']+os.sep+'**', recursive=False)
        
    
    def transfer_ref_images(self):
        self.references_raw_files_full_path=[file for file in glob.glob( os.path.join(self.aquisition_path,self.aquisition_name)+os.sep+'References'+os.sep+'**', recursive=False) if '.tif' in file  ]
        self.reference_images_working_fullpaths=[]
        for file in self.references_raw_files_full_path:
            if not os.path.isfile(os.path.join(self.slow_storage_all_paths['ref images'], os.path.split(file)[1])):
            
                self.reference_images_working_fullpaths.append(shutil.copy(file, self.slow_storage_all_paths['ref images']))             
        # self.read_existing_datasets()     
        
        
    def load_metadata_raw(self):
        if self.Prairireaqpath:
            self.metadata_object=Metadata(acquisition_directory_raw=self.Prairireaqpath,temporary_path=self.slow_storage_all_paths['metadata'], aquisition_object=self)
        elif self.database_acq_raw_path:
            self.metadata_object=Metadata(acquisition_directory_raw=self.database_acq_raw_path,temporary_path=self.slow_storage_all_paths['metadata'], aquisition_object=self)

               
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
           self.voltage_signal_object.load_slow_storage_voltage_signals_daq()
           
       else:
           try:
               self.load_voltage_signals_object()
               self.voltage_signal_object.load_slow_storage_voltage_signals()
               self.voltage_signal_object.load_slow_storage_voltage_signals_daq()

           except:
               module_logger.info('no voltage signals to load')

       
    def unload_voltage_signals(self):
        
       self.voltage_signal_object.unload_voltage_signals()
       
       

    # def transfer_raw_csv(self):
    #     self.voltage_recording_raw_file_transfered_path=self.slow_storage_all_paths['raw_volatge_csv']    
    #     self.voltage_recording_raw_file_full_path=[file for file in glob.glob( os.path.join(self.aquisition_path,self.aquisition_name)+os.sep+'**', recursive=False) if 'VoltageRecording' in file and '.csv' in file   ]
    #     if self.voltage_recording_raw_file_full_path and not os.path.isfile(self.voltage_recording_raw_file_transfered_path):
    #         self.voltage_recording_raw_file_full_path=self.voltage_recording_raw_file_full_path[0]
    #         final_file_path=os.path.join(self.voltage_recording_raw_file_transfered_path,os.path.split(self.voltage_recording_raw_file_full_path)[1])
    #         if not os.path.isfile(final_file_path):
    #             shutil.copy(self.voltage_recording_raw_file_full_path, self.voltage_recording_raw_file_transfered_path)
            
    # def process_raw_voltage_recording(self):
    #     self.read_voltage_signals()
       
    #     self.voltage_recording_raw_file_full_path=[file for file in glob.glob( os.path.join(self.aquisition_path,self.aquisition_name)+os.sep+'**', recursive=False) if 'VoltageRecording' in file and '.csv' in file   ]
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
        if glob.glob(self.slow_storage_all_paths['eye camera']+os.sep+'**', recursive=False):
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
        if self.face_camera:
            module_logger.info('unloading face camera')

            self.face_camera.unload_camera_file()
        else:
            module_logger.info('No face camera')
        
        
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
        self.all_vis_stim_mat_files=glob.glob(self.slow_storage_all_paths['visual stim']+os.sep+'**.mat', recursive=False)
        
    def load_vis_stim_info(self):
        
        
        try:
            if self.all_vis_stim_mat_files and self.all_vis_stim_mat_files[0]:
                # if 'matlab.engine' in sys.modules:        
                #     eng = matlab.engine.start_matlab()
                #     eng.addpath(r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\dataManaging\functions',nargout=0)
                # #testing new versions
                # #         eng.addpath(r'C:\Users\sp3660\Downloads\caiman_sorter-master',nargout=0)

                #     eng.resave_vis_stim_file(self.all_vis_stim_mat_files[0],nargout=0)
                   
                # else:
                #     print('Cant connect to matlab')
                    
                    
                self.mat = loadmat(self.all_vis_stim_mat_files[0])
                if 'full_info' in self.mat.keys():
                
                    outarray=self.mat['full_info']
                    ops=self.mat['ops']
                    durationseconds=np.array([outarray[1:,2][i][0] for i in range(outarray[1:,2].shape[0])])-outarray[1:,1]
                    durationminutes=durationseconds/60
                    outarray[1:,3]=durationseconds
                    stimparadigms=outarray[:,0].tolist()
                    if 'SessionA' in self.all_vis_stim_mat_files[0]:
                        pass
                    elif 'SessionB' in self.all_vis_stim_mat_files[0]:
                        staticparadigms=np.where(['Static' in paradigm for paradigm in stimparadigms])[0]
                        imagesparadigms=np.where(['Images' in paradigm for paradigm in stimparadigms])[0]
                        movieparadigms=np.where(['Movie' in paradigm for paradigm in stimparadigms])[0]
                
                        statictrialinfo=outarray[staticparadigms,:]
                        imagestrialinfo=outarray[imagesparadigms,:]
                        movietrialinfo=outarray[movieparadigms,:]
                
                        firststaticgratings=statictrialinfo[0][4][1:-1,4].astype(int)
                        secondstaticgratings=statictrialinfo[1][4][1:-1,4].astype(int)
                        thirdstaticgratings=statictrialinfo[2][4][1:-1,4].astype(int)
                
                        self.allstaticindexes=np.hstack([firststaticgratings,secondstaticgratings,thirdstaticgratings])
               
                        firstnaturalimages=imagestrialinfo[0][4][1:-1,4].astype(int)
                        secondnaturalimages=imagestrialinfo[1][4][1:-1,4].astype(int)
                        thirdnaturalimages=imagestrialinfo[2][4][1:-1,4].astype(int)
                
                        self.allnatiuralindexes=np.hstack([firstnaturalimages,secondnaturalimages,thirdnaturalimages])
                    elif 'SessionC' in self.all_vis_stim_mat_files[0]:
                        pass
                else:
                    
                    ops=self.mat['ops']
                    if 'SessionA' in self.all_vis_stim_mat_files[0]:
                        pass
                    pass
                
                visstim=self.mat
                #these 3 conditions are for old code that didnet save this info
                if 'is135' in visstim.keys():
                    is135=visstim['is135']
                else:
                    is135='Not Saved'
                    
                if 'isi_color' in visstim.keys():
                    isi_color=visstim['isi_color']
                else:
                    isi_color='Not Saved'
    
                if 'isi_color_texture' in visstim.keys():
                    isi_color_texture=visstim['isi_color_texture']
                else:
                    isi_color_texture='Not Saved'
                
                if 'opto' in visstim.keys():
                    opto=visstim['opto']
                elif 'optoop' in visstim.keys():
                    opto=visstim['optoop']
                else:
                    opto='Not Saved'
                
                
                self.visstimdict={'is135':is135,
                             'isi_color':isi_color,
                             'isi_color_texture':isi_color_texture,
                             'ops':visstim['ops'],
                             'opto':opto,
                             'full_info':'',  
                    }
    

                if 'full_info' in visstim.keys():
                    self.visstiminfofull=[]
    
                    for l,paradigm in enumerate(visstim['full_info'][1:,0].tolist()):
                        self.visstiminfofull.append({paradigm: {k:'' for k in visstim['full_info'][0,1:].tolist()}})
                        for k,columns1 in enumerate(self.visstiminfofull[l][paradigm].keys()):
                            if columns1!='Trials':
                                temp=visstim['full_info'][l+1,k+1] 
                                self.visstiminfofull[l][paradigm][columns1]=np.array(temp)
                            else:
                                self.visstiminfofull[l][paradigm][columns1]=[]
                                trials=visstim['full_info'][l+1,k+1]
                                
                                for trial in trials[1:,0].tolist():
                                   trisldiact={o:'' for o in trials[0,1:].tolist()}
                                   
                                   for k,columns2 in enumerate(trials[0,1:].tolist()):
                                       if columns2 not in ['Frames','Phases']:
                                           temp=trials[trial,k+1]
                                           trisldiact[columns2]=temp
                               
                                       else:
                                           
                                           trisldiact[columns2]=[]
                                           extra_struct=trials[trial,k+1]
    
                                           for extra in extra_struct[1:,0].tolist():
                                              if isinstance(extra,int):
                                                  extradiact={o:'' for o in extra_struct[0,1:].tolist()}
                                                  
                                                  for k,columns3 in enumerate(extra_struct[0,1:].tolist()):
                                                      temp=extra_struct[extra,k+1]
                                                      extradiact[columns3]=temp
                                                  if columns2=='Frames' :   
                                                      extradiact['FrameEnd']=extradiact['FrameTime']
                                                  elif columns2=='Frames' :   
                                                      extradiact['PhaseEnd']=extradiact['PhaseTime']
                                                      
                                               
                                                  trisldiact[columns2].append(extradiact)
                                              
                                   self.visstiminfofull[l][paradigm][columns1].append(trisldiact)
                   
                        self.visstiminfofull[l][paradigm]['ParadigmDuration']= self.visstiminfofull[l][paradigm]['EndParadigmTime'][0]- self.visstiminfofull[l][paradigm]['StartParadigmTime']
          
            
                    for p,k in enumerate(self.visstiminfofull):
                        for n, trial in enumerate(k[list(k.keys())[0]]['Trials']):
                            if n!=len(k[list(k.keys())[0]]['Trials'])-1:
                                trial['TrialEnd']= k[list(k.keys())[0]]['Trials'][n+1]['TrialStart']
                            else:
                                trial['TrialEnd']=   self.visstiminfofull[p][list(self.visstiminfofull[p].keys())[0]]['EndParadigmTime']
                            trial['TrialTime']=trial['TrialEnd'][0]-trial['TrialStart'][0]
                            
                            # if list(trial.keys())[-1] in ['Phases'] and trial['DrifitingIndex']!=0:# from second drifting grating pradadigm the blak trial has pahases but not thr previous one
                            if list(trial.keys())[-1] in ['Phases'] and   trial[list(trial.keys())[-1]][1]['PahseStart'].any():
                                for phn, ph in enumerate(trial[list(trial.keys())[-1]]):
                                    if phn==0:
                                        ph['PahseStart']=trial['StimStart']
                                        # ph['PhaseValue']=
                                        ph['PhaseEnd']= trial[list(trial.keys())[-1]][phn+1]['PahseStart']
        
                                    elif phn!=len(trial[list(trial.keys())[-1]])-1:
                                        # ph['PhaseValue']=
                                        ph['PhaseEnd']= trial[list(trial.keys())[-1]][phn+1]['PahseStart']
        
                                    else:
                                        ph['PhaseEnd']= trial['TrialEnd']
    
                                        
                                    
                                ph['PhaseTime']=ph['PhaseEnd'][0]-ph['PahseStart'][0]
    
                            elif list(trial.keys())[-1] in ['Frames']:
                                
                                for frn, fr in enumerate(trial[list(trial.keys())[-1]]):
                                
                                    if frn==0:
                                        fr['FrameStart']=trial['TrialStart']
                                        fr['FrameEnd']= trial[list(trial.keys())[-1]][frn+1]['FrameStart']
                                   
                                    elif frn!=len(trial[list(trial.keys())[-1]])-1:
                                        fr['FrameEnd']= trial[list(trial.keys())[-1]][frn+1]['FrameStart']
                                   
                                    else:
                                        fr['FrameEnd']= trial['TrialEnd']
        
                                    fr['FrameTime']=fr['FrameEnd'][0]-fr['FrameStart'][0]
        
               
                    self.visstimdict['full_info']=self.visstiminfofull
                
                
               
                else:
                    print("Old version no info available")
                    self.visstiminfofull='Old version no info available'
                    self.visstimdict['full_info']=self.visstiminfofull
                    # sequence=visstim['ops']['paradigm_sequence'].tolist()
    
                    # titles0=['Paradigms', 'StartParadigmTime','EndParadigmTime','ParadigmDuration','Trials']
                    # trilacolumns=['Trial','TrialStart','StimStart','TrialEnd','TrialTime']
                    # stimspecialcolumns=['GratingIndex','SceneIndex','Frames','DriftingIndex','Phases']
                    
                    # self.visstiminfofull={}
                    # for i,column in enumerate(titles0):
                    #     if i==0:
                    #         self.visstiminfofull[column]=np.array()
                    #     elif i==1:
                    #         self.visstiminfofull[column]=np.zeros([len(sequence),0])
                    #         # self.visstiminfofull[titles0[2]][0]- self.visstiminfofull[titles0[1]]
                    #     elif i==4:
                            
                            
                    #         trisldiact={}
                    #             for k,trial_column in enumerate(trilacolumns):
                    #                 trisldiact[trial_column]=np.zeros([len(sequence),0])
                    #                 temp1
    
    
                    #         trilacolumns
                    #         stimspecialcolumns
                    #         stim_start=[i for i in visstim['stim_times']    ]
                    #         trial_end=[i for i in visstim['end_stim_times']    ]
                    #         drif_index=[i for i in visstim['stim_ang']    ]
                    #         temp2=[i for i in visstim['frameendtimes']    ]
                    #         temp3=[i for i in visstim['framestarttimes']    ]
        except:
            print('Something wrong with the mat visstim file')
                              
                
#%% loading unloading all
    def load_all(self, camera=True, kalman=True):
        self.load_vis_stim_info()
        if camera:
            self.load_face_camera()
        # self.load_voltage_signals()
        self.load_voltage_signals()
        self.load_all_datasets(kalman=kalman)
        self.load_metadata_slow_working_directories()
        self.load_reference_images()

        # self.mouse_imaging_session_object.

    def unload_all(self):
        self.unload_camera()
        # self.unload_voltage_signals()
        self.unload_voltage_signals
        self.unload_all_datasets()
        self.unload_reference_images()
        
    def load_results_analysis(self,nondatabase=None, new_full_data=False):
        
        self.analysis_object=ResultsAnalysis(acquisition_object=self, nondatabase=nondatabase, new_full_data=new_full_data)
        
    def load_results_analysis_cloud(self, nondatabase=None, new_full_data=False):
        
        self.analysis_object=ResultsAnalysis(acquisition_object=self, nondatabase=nondatabase, new_full_data=new_full_data)
        
 
    def get_associated_tomato_fov_datasets(self):

        self.associated_fov_tomato_acquisition= self.FOV_object.all_existing_10503planetomato[list(self.FOV_object.all_existing_10503planetomato.keys())[0]]
        self.associated_fov_tomato_acquisition.load_all_datasets()
        
        
        
        # dataset_object.summary_images_object.projection_dic['std_projection_path']
        
        
        
    def get_all_raw_aq_tiffiles(raw_aq_dir):
        unprocessed=glob.glob(raw_aq_dir+'\**.tif')
        processed=[]
        for ch in ['Ch1Red','Ch2Green']:
            if os.path.isdir(os.path.join(raw_aq_dir,ch)):
                for i in os.listdir( os.path.join(raw_aq_dir,ch)):
                    processed.append(glob.glob(os.path.join(raw_aq_dir,ch,i,'**.tif')))
            
        processed.append(unprocessed)
        
        allfiles=list(itertools.chain.from_iterable(processed))
        
        return allfiles
        
    
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