# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 08:52:07 2021

@author: sp3660
"""

import os
import glob

from .aquisition import Aquisition
from .acquisitonVariants import AtlasOverview, AtlasPreview, AtlasVolume


class Atlas():
    def __init__(self, atlas_name, raw_atlas_path=None, mouse_imaging_session_object=None):
        
        self.atlas_name =atlas_name
        self.atlas_raw_path=raw_atlas_path
        self.mouse_imaging_session_object=mouse_imaging_session_object
        self.overview_raw_path=os.path.join(self.atlas_raw_path,'Overview')
        self.coordinates_raw_path=os.path.join(self.atlas_raw_path,'Coordinates')
        self.volumes_raw_path=os.path.join(self.atlas_raw_path,'Volumes')
        self.preview_raw_path=os.path.join(self.atlas_raw_path,'Preview')
        self.preview_raw_coordinates=os.path.join(self.atlas_raw_path,'Coordinates')
        self.mouse_session_atlas_path=os.path.join(self.mouse_imaging_session_object.mouse_session_path, 'atlases', self.atlas_name)
        if not os.path.isdir(self.mouse_session_atlas_path):
            os.mkdir(self.mouse_session_atlas_path)
        
 
    def load_raw_Overviews(self):
        if os.path.isdir(self.overview_raw_path):
            self.all_raw_Overviews={aqu:AtlasOverview(aqu,                                                          
                                                self,
                                                raw_input_path=glob.glob( self.overview_raw_path+os.sep+'**',recursive=False)[0])
                                     
                                  for aqu in os.listdir(self.overview_raw_path)
                                  if 'Aq_' in aqu}
            self.load_existing_Overviews() 
        else:
             pass
         
    def load_raw_Previews(self):
        if os.path.isdir(self.preview_raw_path):
            self.all_raw_Previews={aqu:AtlasPreview(aqu,                                                          
                                                self,
                                                raw_input_path=glob.glob( self.preview_raw_path+os.sep+'**',recursive=False)[0])
                                     
                                  for aqu in os.listdir(self.preview_raw_path)
                                  if 'Aq_' in aqu}
            self.load_existing_Previews() 
        else:
             pass 
         
    def load_raw_Volumes(self):
        if os.path.isdir(self.volumes_raw_path):
            self.all_raw_Volumes={aqu:AtlasVolume(aqu,                                                          
                                                self,
                                                raw_input_path=glob.glob( self.volumes_raw_path+os.sep+'**',recursive=False)[0])
                                     
                                  for aqu in os.listdir(self.volumes_raw_path)
                                  if 'Aq_' in aqu}
            self.load_existing_Volumes() 
        else:
             pass       
            
         
    def load_existing_Overviews(self):
        if os.path.isdir(os.path.join(self.mouse_session_atlas_path, 'Overview')):
            self.all_existing_Overviews={aqu:AtlasOverview(aqu,self) 
                                  for aqu in os.listdir(os.path.join(self.mouse_session_atlas_path, 'Overview'))
                                  }   
        else:
              pass        
         
    def load_existing_Previews(self):
        if os.path.isdir(os.path.join(self.mouse_session_atlas_path, 'Preview')):
            self.all_existing_Previews={aqu:AtlasPreview(aqu,self) 
                                  for aqu in os.listdir(os.path.join(self.mouse_session_atlas_path, 'Preview'))
                                  }   
        else:
              pass        
                     
    def load_existing_Volumes(self):
        if os.path.isdir(os.path.join(self.mouse_session_atlas_path, 'Volume')):
            self.all_existing_Volumes={aqu:AtlasVolume(aqu,self) 
                                  for aqu in os.listdir(os.path.join(self.mouse_session_atlas_path, 'Volume'))
                                  }   
        else:
              pass        
                  
            