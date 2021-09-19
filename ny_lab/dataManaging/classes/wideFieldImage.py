# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:18:06 2021

@author: sp3660
"""
import os
import shutil
import matplotlib.pyplot as plt


class WideFieldImage:
    
    def __init__(self, mouse_imaging_session_object, file_name, raw_input_path=None):
        
        self.mouse_imaging_session_object=mouse_imaging_session_object

        if raw_input_path:
            self.WidefieldImagePath_raw=raw_input_path
            self.WidefieldImagePath_stored=os.path.join(self.mouse_imaging_session_object.mouse_session_path,'widefield image',file_name)
            shutil.copyfile(self.WidefieldImagePath_raw, self.WidefieldImagePath_stored)
        else:
            self.WidefieldImagePath_stored=os.path.join(self.mouse_imaging_session_object.mouse_session_path,'widefield image',file_name)


            




    def plot_image(self):
        I = plt.imread(self.WidefieldImagePath_stored)
        self.imgplot = plt.imshow(I,cmap='inferno')
