# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:18:06 2021

@author: sp3660
"""
import os
import shutil
import matplotlib.pyplot as plt
import logging 
module_logger = logging.getLogger(__name__)


class WideFieldImage:
    
    def __init__(self,file_name, mouse_imaging_session_object=None, raw_input_path=None, slostragepath=None):
        
        self.mouse_imaging_session_object=mouse_imaging_session_object

         
        if raw_input_path and mouse_imaging_session_object:
            self.WidefieldImagePath_raw=raw_input_path
            module_logger.info('Processing raw widefield ' + self.WidefieldImagePath_raw)

            
            self.roi_zip_path_raw=os.path.join(os.path.split(self.WidefieldImagePath_raw)[0],'Rois', self.mouse_imaging_session_object.mouse_object.mouse_name+'.zip')
            self.WidefieldImagePath_stored=os.path.join(self.mouse_imaging_session_object.mouse_session_path,'widefield image',file_name)
            self.roi_zip_path_stored=os.path.join(self.mouse_imaging_session_object.mouse_session_path,'widefield image',self.mouse_imaging_session_object.mouse_object.mouse_name+'.zip')
            shutil.copyfile(self.WidefieldImagePath_raw, self.WidefieldImagePath_stored)
            if os.path.isfile(self.roi_zip_path_raw):
                shutil.copyfile(self.roi_zip_path_raw, self.roi_zip_path_stored)
            
        else:
            self.WidefieldImagePath_stored=os.path.join(self.mouse_imaging_session_object.mouse_session_path,'widefield image',file_name)


    def plot_image(self):
        self.load_image()
        self.imgplot = plt.imshow(self.widefield_image,cmap='inferno')
    def load_image(self):
        self.widefield_image = plt.imread(self.WidefieldImagePath_stored)

