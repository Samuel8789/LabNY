# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:58:42 2021

@author: sp3660
"""
import os
import glob

import logging 
module_logger = logging.getLogger(__name__)

from .aquisition import Aquisition
# from .acquisitonVariants import Tomato1050Acquisition, Tomato3Plane1050Acquisition, TomatoHighResStack1050Acquisition, HighResStackGreenAcquisition, SurfaceImageAquisition, OtherAcqAquisition
from .acquisitonVariants import *

from ...data_pre_processing.dataExplorations import DataExplorations

class FOV():
    
    def __init__(self, FOV_name, raw_FOV_path=None, mouse_imaging_session_object=None):
        self.all_aquisitions={}
        self.all_existing_1050tomato={}
        self.all_existing_10503planetomato  = {}
        self.all_existing_1050HighResStackTomato={}
        self.all_existing_HighResStackGreen={}
        self.all_existing_OtherAcq={}
        self.all_existing_SurfaceImage={}
        
        #  module_logger.info('loading mouse FOV:' + FOV_name)
        self.FOV_name=FOV_name       
        self.mouse_imaging_session_object=mouse_imaging_session_object
        self.mouse_session_FOV_path=os.path.join(self.mouse_imaging_session_object.mouse_session_path,
                                                 'data aquisitions',
                                                 self.FOV_name)
 
        if not os.path.isdir(self.mouse_session_FOV_path):
          os.mkdir(self.mouse_session_FOV_path)
          
          
        if raw_FOV_path:
            self.FOV_path=raw_FOV_path
            module_logger.info('adding raw fovaquisitions')
            self.load_raw_aquisitions()
            module_logger.info('adding raw 1050tom')
            self.load_raw_1050tomato()
            module_logger.info('adding 10503planetom')
            self.load_raw_10503planetomato()
            module_logger.info('adding 1050higres')
            self.load_raw_1050HighResStackTomato()
            module_logger.info('adding greenhighres')
            self.load_raw_HighResStackGreen()
            module_logger.info('adding otheracq')
            self.load_raw_OtherAcq()
            module_logger.info('adding surfaceimage')
            self.load_raw_SurfaceImage()
        else:
            self.load_existing_aquisitions() 
            self.load_existing_1050tomato()
            self.load_existing_10503planetomato()
            self.load_existing_1050HighResStackTomato()
            self.load_existing_HighResStackGreen()
            self.load_existing_SurfaceImage()
            self.load_existing_OtherAcq()
        
        self.all_datasets=[self.all_aquisitions, self.all_existing_1050tomato, self.all_existing_10503planetomato, 
                           self.all_existing_1050HighResStackTomato, self.all_existing_HighResStackGreen, self.all_existing_OtherAcq,
                           self.all_existing_SurfaceImage ]    
        self.data_explorations={}  
     
        
     
        
#%% raw     
    def load_raw_aquisitions(self):
        if os.path.isdir(self.FOV_path): 
            self.all_raw_aquisitions={aqu:Aquisition(aqu, 
                                                 os.path.join( self.FOV_path, aqu),
                                                 self) 
                                  for aqu in os.listdir(self.FOV_path)
                                  if 'Aq_' in aqu}
            self.load_existing_aquisitions() 
        else:
             pass
         
    def load_raw_1050tomato(self):
        if os.path.isdir(os.path.join(self.FOV_path,'1050_Tomato')):
            self.all_raw_1050tomato={aqu:Tomato1050Acquisition(glob.glob( os.path.join(self.FOV_path, '1050_Tomato')+'\\**',recursive=False)[0],                                                          
                                                self,
                                                raw_input_path=os.path.join( self.FOV_path,'1050_Tomato', aqu))
                                     
                                  for aqu in os.listdir(os.path.join(self.FOV_path,'1050_Tomato'))
                                  if 'Aq_' in aqu}
            self.load_existing_1050tomato() 
        else:
             pass        
         
    def load_raw_10503planetomato(self):
        if os.path.isdir(os.path.join(self.FOV_path,'1050_3PlaneTomato')):
            self.all_raw_10503planetomato={aqu:Tomato3Plane1050Acquisition(glob.glob( os.path.join(self.FOV_path, '1050_3PlaneTomato')+'\\**',recursive=False)[0],                                                          
                                                self,
                                                raw_input_path=os.path.join( self.FOV_path, '1050_3PlaneTomato',aqu))
                                  for aqu in os.listdir(os.path.join(self.FOV_path,'1050_3PlaneTomato'))
                                  if 'Aq_' in aqu }
            self.load_existing_10503planetomato() 
        else:
             pass            
         
    def load_raw_1050HighResStackTomato(self):
        if os.path.isdir(os.path.join(self.FOV_path,'1050_HighResStackTomato')): 
            self.all_raw_1050HighResStackTomato={aqu:TomatoHighResStack1050Acquisition(glob.glob( os.path.join(self.FOV_path, '1050_HighResStackTomato')+'\\**',recursive=False)[0],                                                          
                                                self,
                                                raw_input_path=os.path.join( self.FOV_path,'1050_HighResStackTomato', aqu))
                                  for aqu in os.listdir(os.path.join(self.FOV_path,'1050_HighResStackTomato'))
                                  if 'Aq_' in aqu}
            self.load_existing_1050HighResStackTomato() 
        else:
             pass            
    
    def load_raw_HighResStackGreen(self):
        if os.path.isdir(os.path.join(self.FOV_path,'HighResStackGreen')):
            self.all_raw_HighResStackGreen={aqu:HighResStackGreenAcquisition(glob.glob( os.path.join(self.FOV_path, 'HighResStackGreen')+'\\**',recursive=False)[0],                                                          
                                                self,
                                                raw_input_path=os.path.join( self.FOV_path,'HighResStackGreen', aqu))
                                  for aqu in os.listdir(os.path.join(self.FOV_path,'HighResStackGreen'))
                                  if 'Aq_' in aqu}
            self.load_existing_HighResStackGreen() 
        else:
             pass    
         
    def load_raw_OtherAcq(self):
        if os.path.isdir(os.path.join(self.FOV_path,'OtherAcq')):
            self.all_raw_OtherAcq={aqu:OtherAcqAquisition(glob.glob( os.path.join(self.FOV_path, 'OtherAcq')+'\\**',recursive=False)[0],                                                          
                                                self,
                                                raw_input_path=os.path.join( self.FOV_path,'OtherAcq', aqu))
                                  for aqu in os.listdir(os.path.join(self.FOV_path,'OtherAcq'))
                                  if 'Aq_' in aqu}
                                                          
            self.load_existing_OtherAcq() 
        else:
             pass   

            
    def load_raw_SurfaceImage(self):
        if os.path.isdir(os.path.join(self.FOV_path,'SurfaceImage')):
            self.all_raw_SurfaceImage={aqu:SurfaceImageAquisition(glob.glob( os.path.join(self.FOV_path, 'SurfaceImage')+'\\**',recursive=False)[0],                                                          
                                                self,
                                                raw_input_path=os.path.join( self.FOV_path,'SurfaceImage', aqu))
                                  for aqu in os.listdir(os.path.join(self.FOV_path,'SurfaceImage'))
                                  if 'Aq_' in aqu}
            self.load_existing_SurfaceImage() 
        else:
             pass
#%% existing            
    def load_existing_aquisitions(self):
     if os.path.isdir(self.mouse_session_FOV_path):
        self.all_aquisitions={aqu:Aquisition(aqu, 
                                             FOV_object=self) 
                              for aqu in os.listdir(self.mouse_session_FOV_path)                          
                              if glob.glob(os.path.join(self.mouse_session_FOV_path,aqu)+'\\planes', recursive=False)}
        
    
    def load_existing_1050tomato(self):
        if os.path.isdir(os.path.join(self.mouse_session_FOV_path, '1050_Tomato')):

            self.all_existing_1050tomato={aqu:Tomato1050Acquisition(aqu,self) 
                                  for aqu in os.listdir(os.path.join(self.mouse_session_FOV_path, '1050_Tomato'))
                                  }   
        else:
             pass

    def load_existing_10503planetomato(self):
        
        if os.path.isdir(os.path.join(self.mouse_session_FOV_path, '1050_3PlaneTomato')):

            self.all_existing_10503planetomato={aqu:Tomato3Plane1050Acquisition(aqu, 
                                                 self) 
                                  for aqu in os.listdir(os.path.join(self.mouse_session_FOV_path, '1050_3PlaneTomato'))
                                  }       
        else:
             pass

    def load_existing_1050HighResStackTomato(self):
        if os.path.isdir(os.path.join(self.mouse_session_FOV_path, '1050_HighResStackTomato')):

            self.all_existing_1050HighResStackTomato={aqu:TomatoHighResStack1050Acquisition(aqu, 
                                                 self) 
                                  for aqu in os.listdir(os.path.join(self.mouse_session_FOV_path, '1050_HighResStackTomato'))
                                  
                                  }       
        else:
             pass

    def load_existing_HighResStackGreen(self):
        if os.path.isdir(os.path.join(self.mouse_session_FOV_path, 'HighResStackGreen')):

            self.all_existing_HighResStackGreen={aqu:HighResStackGreenAcquisition(aqu, 
                                                 self) 
                                  for aqu in os.listdir(os.path.join(self.mouse_session_FOV_path, 'HighResStackGreen'))
                                  
                                  }       
        else:
             pass   

    def load_existing_OtherAcq(self):
        if os.path.isdir(os.path.join(self.mouse_session_FOV_path, 'OtherAcq')):

            self.all_existing_OtherAcq={aqu:OtherAcqAquisition(aqu, 
                                                 self) 
                                  for aqu in os.listdir(os.path.join(self.mouse_session_FOV_path, 'OtherAcq'))
                                  
                                  }       
        else:
             pass
         

    def load_existing_SurfaceImage(self):
        if os.path.isdir(os.path.join(self.mouse_session_FOV_path, 'SurfaceImage')):
            self.all_existing_SurfaceImage={aqu:SurfaceImageAquisition(aqu, self) 
                                  for aqu in os.listdir(os.path.join(self.mouse_session_FOV_path, 'SurfaceImage'))}       
        else:
             pass
        
        
#%% other        
    def explore_data(self, description):
        
        self.data_explorations[description]  = DataExplorations(self) 
        
        return DataExplorations(self)    
     
     