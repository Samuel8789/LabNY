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

from ...AllFunctions.create_dir_structure import create_dir_structure
from .dataset import ImageSequenceDataset
from .eyeVideo import EyeVideo
from ...data_pre_processing.voltageSignalsExtractions import VoltageSignalsExtractions




class Aquisition:   
    def __init__(self, aqu_name, raw_input_path=None, FOV_object=None, mouse_imaging_session_object=None, atlas_object=None, subaq_object=None, non_imaging=False,):
        self.subaq_object=subaq_object
        self.FOV_object=FOV_object
        self.Atlas_object=atlas_object
        self.non_imaging=non_imaging
        # print(aqu_name)

        if raw_input_path and not non_imaging:
            Prairireaqpath=os.path.split(glob.glob( raw_input_path +'\\**\\**.env', recursive=True)[0])[0]      
            self.aquisition_path=os.path.split(Prairireaqpath)[0]
            self.aquisition_name= os.path.split(Prairireaqpath)[1]
        else:
            self.aquisition_name=aqu_name

        if mouse_imaging_session_object:
            self.mouse_imaging_session_object=mouse_imaging_session_object
            
            if subaq_object=='TestAquisition':
                self.mouse_aquisition_path=os.path.join(self.mouse_imaging_session_object.mouse_session_path, 
                                                    'test aquisitions',
                                                     self.aquisition_name)
            if subaq_object=='Coordinate0Aquisition':
               self.mouse_aquisition_path=os.path.join(self.mouse_imaging_session_object.mouse_session_path, 
                                                    '0Coordinate acquisition',
                                                     self.aquisition_name)
            
            if subaq_object=='NonimagingAquisition':       
                self.mouse_aquisition_path=os.path.join(self.mouse_imaging_session_object.mouse_session_path, 
                                                        'nonimaging acquisitions',
                                                         self.aquisition_name)

            self.mouse_name = self.mouse_imaging_session_object.mouse_object.mouse_name
            # print('associating with mouse object ' + self.mouse_name)           
            if self.mouse_name:
                self.mouse_object=self.mouse_imaging_session_object.mouse_object
        
        
        elif atlas_object:              
            self.Atlas=self.Atlas_object.atlas_name
    
            if subaq_object=='AtlasOverview':   
                self.mouse_aquisition_path=os.path.join(self.Atlas_object.mouse_session_atlas_path, 
                                                        'Overview',
                                                         self.aquisition_name)
            if subaq_object=='AtlasPreview':       
                self.mouse_aquisition_path=os.path.join(self.Atlas_object.mouse_session_atlas_path, 
                                                        'Preview',
                                                         self.aquisition_name)
               
            if subaq_object=='AtlasVolume':       
                self.mouse_aquisition_path=os.path.join(self.Atlas_object.mouse_session_atlas_path, 
                                                        'Volumes',
                                                         self.aquisition_name)
                
                
            self.mouse_name = self.Atlas_object.mouse_imaging_session_object.mouse_object.mouse_name
            # print('associating with mouse object ' + self.mouse_name)           
            if self.mouse_name:
                self.mouse_object=self.Atlas_object.mouse_imaging_session_object.mouse_object    
                
                
        elif FOV_object:        
            self.FOV=self.FOV_object.FOV_name

            if subaq_object==None :
                self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path,                                                
                                                          self.aquisition_name)
            elif subaq_object:
    
                if subaq_object=='Tomato1050Acquisition':
                    self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path, '1050_Tomato',                                               
                                                          self.aquisition_name)
                if subaq_object=='Tomato3Plane1050Acquisition':
                    self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path, '1050_3PlaneTomato',                                               
                                                          self.aquisition_name)
                if subaq_object=='TomatoHighResStack1050Acquisition':
                    self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path, '1050_HighResStackTomato',                                                
                                                          self.aquisition_name)
                if subaq_object =='HighResStackGreenAcquisition':
                    self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path, 'HighResStackGreen',                                               
                                                          self.aquisition_name)
                if subaq_object=='SurfaceImageAquisition':
                    self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path, 'SurfaceImage',                                               
                                                          self.aquisition_name)
                if subaq_object=='OtherAcqAquisition':   
                    self.mouse_aquisition_path=os.path.join(self.FOV_object.mouse_session_FOV_path, 'OtherAcq',                                               
                                                          self.aquisition_name)
            
    
            self.mouse_name = self.FOV_object.mouse_imaging_session_object.mouse_object.mouse_name
            # print('associating with mouse object ' + self.mouse_name)           
            if self.mouse_name:
                self.mouse_object=self.FOV_object.mouse_imaging_session_object.mouse_object
    
    
    
#%% standarrd    
        if raw_input_path and not non_imaging:
            # Prairireaqpath=os.path.split(glob.glob( raw_input_path +'\\**\\**.env', recursive=True)[0])[0]      
            # self.aquisition_path=Prairireaqpath
            # self.aquisition_name= os.path.split(Prairireaqpath)[1]
            
                
 #%% prairie clenauo           
            # print('processing raw aquisition '+ self.aquisition_name)  
            if self.FOV_object:
                self.aquisition_dir_strucutre=self.FOV_object.mouse_imaging_session_object.mouse_object.data_managing_object.file_cleanup_prairie_new(Prairireaqpath) 
            elif self.FOV_object:
                self.aquisition_dir_strucutre=self.Atlas_object.mouse_imaging_session_object.mouse_object.data_managing_object.file_cleanup_prairie_new(Prairireaqpath) 
            else:
                self.aquisition_dir_strucutre=self.mouse_imaging_session_object.mouse_object.data_managing_object.file_cleanup_prairie_new(Prairireaqpath) 
                
                
#%%  readin files              
            # print('checking dir structure') 
            if any(self.aquisition_dir_strucutre):
                self.read_aquisition_structure()
                self.create_aquisition_structure_in_mouse_file()
                # print('now convert to mmap and transfer to folder')     
                self.raw_main_path_equivalence()
                self.solve_all_datasets()
            else:
                aquisition_to_process=os.path.join(self.aquisition_path,self.aquisition_name)
                temp_path=os.path.split(os.path.split(aquisition_to_process)[0])[0]
                aquisition_date=aquisition_to_process[temp_path.find('\SP')-13:temp_path.find('\SP')-5]
                self.formated_aquisition_date=aquisition_date            
                self.session_path=os.path.join(self.mouse_object.mouse_slow_subproject_path, 'imaging', self.formated_aquisition_date)
                self.create_aquisition_structure_in_mouse_file()  
                self.aquisition_path=raw_input_path
                self.solve_all_datasets()
#%% not raw           
        elif not non_imaging:    

            self.read_existing_datasets()
#%% non imaging            
        elif raw_input_path and non_imaging:
            # print('processing raw aquisition '+ self.aquisition_name)  
            self.aquisition_dir_strucutre=[]            
            aquisition_to_process=raw_input_path
            temp_path=os.path.split(os.path.split(aquisition_to_process)[0])[0]
            aquisition_date=aquisition_to_process[temp_path.find('\SP')-13:temp_path.find('\SP')-5]
            self.formated_aquisition_date=aquisition_date            
            self.session_path=os.path.join(self.mouse_object.mouse_slow_subproject_path, 'imaging', self.formated_aquisition_date)
            self.create_aquisition_structure_in_mouse_file()  
            self.aquisition_path=raw_input_path
            self.solve_all_datasets()
            
 #%%
        elif non_imaging:                
              print('i have to give tit a name to do facecamera')
    
#%% methiods         
    def read_aquisition_structure(self):

          self.plane_number=self.aquisition_dir_strucutre[1]
          self.imaged_channels=[chan for chan in self.aquisition_dir_strucutre[0] if 'Ch' in chan]
          self.all_raw_image_sequence_paths=self.aquisition_dir_strucutre[2]
          aquisition_to_process=os.path.join(self.aquisition_path,self.aquisition_name)
          temp_path=os.path.split(os.path.split(aquisition_to_process)[0])[0]
          aquisition_date=aquisition_to_process[temp_path.find('\SP')-13:temp_path.find('\SP')-5]
          self.formated_aquisition_date=aquisition_date            
          self.session_path=os.path.join(self.mouse_object.mouse_slow_subproject_path, 'imaging', self.formated_aquisition_date) 
             
    def create_aquisition_structure_in_mouse_file(self):

       if not os.path.isdir(self.mouse_aquisition_path):
            os.makedirs(self.mouse_aquisition_path)   
            
            aquisition_structure={'planes',
                                'locomotion',
                                'eye camera',
                                'visual stim',
                                'photodiode',
                                'behaviour',
                                'whiskers camera',
                                'photostim',
                                'metadata',
                                'ref images'
                                }    
            create_dir_structure(self.mouse_aquisition_path, aquisition_structure)
            
       if not os.path.isdir(os.path.join(self.mouse_aquisition_path, 'metadata')):
          os.makedirs(os.path.join(self.mouse_aquisition_path,'metadata')  )
                              
       if not os.path.isdir(os.path.join(self.mouse_aquisition_path,'ref images')):
          os.makedirs(os.path.join(self.mouse_aquisition_path,'ref images') )  
       
        
       self.plane_paths={}       
       if any(self.aquisition_dir_strucutre):
           for plane in range(self.plane_number) :
                self.plane_paths['Plane' + str(plane+1)]=os.path.join(self.mouse_aquisition_path, 'planes', 'Plane' + str(plane+1) )
                if not os.path.isdir(self.plane_paths['Plane' + str(plane+1)]):
                    os.mkdir(self.plane_paths['Plane' + str(plane+1)])
                    
                    plane_structure={'Green',
                                     'Red',
                                     'PhotoStimMask',
                                     }
                    create_dir_structure(self.plane_paths['Plane' + str(plane+1)], plane_structure)
             
    def raw_main_path_equivalence(self):
         self.dataset_path_equivalences={}
         for dataset_raw_path in self.all_raw_image_sequence_paths:

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
        
    def solve_all_datasets(self):
        
        if self.subaq_object!='NonimagingAquisition': 

            self.all_raw_datasets={}
            
            for key, value in self.dataset_path_equivalences.items():
                if self.FOV_object:
                    dataset_name=self.mouse_name +'_'+ self.formated_aquisition_date +'_'+self.FOV +'_'+self.aquisition_name+'_'+value[1]+'_'+value[0]
                elif self.mouse_imaging_session_object:  
                    dataset_name=self.mouse_name +'_'+ self.formated_aquisition_date +'_'+self.aquisition_name+'_'+value[1]+'_'+value[0]
                    
                    
                self.all_raw_datasets[dataset_name]=ImageSequenceDataset(self,
                                                                     dataset_name,
                                                                     selected_dataset_raw_path=key, 
                                                                 selected_dataset_mmap_path=value[2])
            # print(dataset_name)   
        self.read_existing_datasets()   
        self.transfer_vis_stim_info()
        self.process_raw_voltage_recording()
        self.transfer_metadata()
        self.transfer_ref_images()
        self.process_face_camera()
        
    def process_raw_voltage_recording(self):
        self.voltage_signals_dictionary={}
        self.voltage_recording_raw_file_full_path=[file for file in glob.glob( os.path.join(self.aquisition_path,self.aquisition_name)+'\\**', recursive=False) if 'VoltageRecording' in file and '.csv' in file   ]
        if self.voltage_recording_raw_file_full_path:
            self.voltage_recording_raw_file_full_path=self.voltage_recording_raw_file_full_path[0]
            try:
                self.voltage_signals = pd.read_csv(self.voltage_recording_raw_file_full_path)
                self.voltage_signals_dictionary={signal:self.voltage_signals[signal].to_frame() for signal in self.voltage_signals.columns.tolist()[1:]}
            except:
                self.voltage_signals_dictionary={'Locomotion': pd.DataFrame({'Locomotion' : []}),
                                                 'VisStim':pd.DataFrame({'VisStim' : []}),
                                                 'LED':pd.DataFrame({'LED' : []}),
                                                 'PhotoDiode':pd.DataFrame({'PhotoDiode' : []})}
        
        for signal, dataf in self.voltage_signals_dictionary.items():
            if 'Locomotion' in signal:     
                self.voltage_signals_dictionary[signal].to_feather(self.transfered_locomotion_fullpath)
                
            elif 'VisStim' in signal:

                self.voltage_signals_dictionary[signal].to_feather(self.transfered_visstim_fullpath)           
            elif 'LED' in signal:
               
                self.voltage_signals_dictionary[signal].to_feather(self.transfered_led_fullpath)
            elif 'PhotoDiode' in signal:
                
                self.voltage_signals_dictionary[signal].to_feather(self.transfered_photodiode_fullpath)
                
                
        self.read_voltage_signals()

    def transfer_metadata(self):    
        self.metadata_raw_files_full_path=[file for file in glob.glob( os.path.join(self.aquisition_path,self.aquisition_name)+'\\**', recursive=False) if '.xml' in file  ]
        
        for file in self.metadata_raw_files_full_path:
            shutil.copy(file, self.working_metadata_path)
        self.read_existing_datasets() 
            
    def transfer_ref_images(self):
        self.references_raw_files_full_path=[file for file in glob.glob( os.path.join(self.aquisition_path,self.aquisition_name)+'\\References\\**', recursive=False) if '.tif' in file  ]
        
        for file in self.references_raw_files_full_path:
            shutil.copy(file, self.working_ref_images_path)             
        self.read_existing_datasets()     
        
    def read_existing_datasets(self):
        self.working_vistim_path=os.path.join(self.mouse_aquisition_path, 'visual stim')     
        self.working_facecamera_path=os.path.join(self.mouse_aquisition_path, 'eye camera')     
        self.working_locomotion_path=os.path.join(self.mouse_aquisition_path, 'locomotion')     
        self.working_photodiode_path=os.path.join(self.mouse_aquisition_path, 'photodiode')
        self.working_behavioursignal_path=os.path.join(self.mouse_aquisition_path, 'behaviour')
        self.working_ledsignal_path=os.path.join(self.mouse_aquisition_path, 'eye camera')
        self.working_metadata_path=os.path.join(self.mouse_aquisition_path, 'metadata')
        self.working_ref_images_path=os.path.join(self.mouse_aquisition_path, 'ref images')
        
        self.metadata_preprocessed_files_full_path = glob.glob( self.working_metadata_path+'\\**', recursive=False)     

        self.all_datasets={}
        for plane in os.listdir(os.path.join(self.mouse_aquisition_path, 'planes')):
            for channel in os.listdir(os.path.join(self.mouse_aquisition_path, 'planes', plane )):
                for dataset in os.listdir(os.path.join(self.mouse_aquisition_path, 'planes', plane, channel )):
                    if dataset.endswith('mmap'):
                       self.all_datasets[plane+'_'+channel+'_'+dataset]=ImageSequenceDataset(self,
                                                                 dataset,                                                               
                                                                 selected_dataset_mmap_path=os.path.join(self.mouse_aquisition_path,
                                                                                                         'planes',
                                                                                                         plane, 
                                                                                                         channel,
                                                                                                         )) 
                   
        self.read_voltage_signals()         
        self.read_working_vistim_info()
        self.read_face_camera()
        
    def read_voltage_signals(self): 
        
        self.transfered_locomotion_fullpath=os.path.join(self.working_locomotion_path, 'locomotion.ftr')
        self.transfered_visstim_fullpath=os.path.join(self.working_vistim_path, 'vis_stim_voltage.ftr')
        self.transfered_led_fullpath=os.path.join(self.working_ledsignal_path, 'led.ftr')
        self.transfered_photodiode_fullpath=os.path.join(self.working_photodiode_path, 'screen_photodiode.ftr')
        
        self.voltage_signals_dictionary={}
        if os.path.isfile(self.transfered_locomotion_fullpath):  
            self.voltage_signals_dictionary['Locomotion']=pd.read_feather(self.transfered_locomotion_fullpath)
        if os.path.isfile(self.transfered_visstim_fullpath):  
            self.voltage_signals_dictionary['VisStim']=pd.read_feather(self.transfered_visstim_fullpath)  
        if os.path.isfile(self.transfered_led_fullpath):  
            self.voltage_signals_dictionary['LED']=pd.read_feather(self.transfered_led_fullpath)
        if os.path.isfile(self.transfered_photodiode_fullpath):  
            self.voltage_signals_dictionary['PhotoDiode']=pd.read_feather(self.transfered_photodiode_fullpath)  
        
    def transfer_vis_stim_info(self):  
        vistimpath=os.path.join(self.aquisition_path, 'VisStim')
        self.raw_vis_stim_info=[]
        if os.path.isdir(vistimpath):
            for file in os.listdir(vistimpath):
                self.raw_vis_stim_info.append(shutil.copy(os.path.join(vistimpath,file), self.working_vistim_path))
        self.read_working_vistim_info()     
        
    def read_working_vistim_info(self): 
        self.all_vis_stim_mat_files=glob.glob(self.working_vistim_path+'\\**', recursive=False)
        
    def process_face_camera(self):
        # print(self.aquisition_path)
        # print(self.aquisition_name)
        self.face_camera_raw_path=os.path.join(self.aquisition_path, 'FaceCamera')
        if os.path.isdir(self.face_camera_raw_path):
            self.face_camera=EyeVideo(self, selected_eyevideo_raw_path=self.face_camera_raw_path)
        else:
            # pass
            print('No Raw Face Camera')
     
    def read_face_camera(self):
        if glob.glob(self.working_facecamera_path+'\\**', recursive=False):
            self.face_camera=EyeVideo(self)
        else:
            pass
            # print('No Processed Face Camera')
            
    def run_voltage_processing(self):
        self.voltage_analysis=VoltageSignalsExtractions(self)
        
            
            