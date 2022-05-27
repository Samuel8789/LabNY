# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:18:06 2021

@author: sp3660
"""
import shutil
import logging 
module_logger = logging.getLogger(__name__)
from roifile import ImagejRoi
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import glob
import scipy.signal as sg

from zipfile import ZipFile

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
            self.widefield_data_path=os.path.join(self.mouse_imaging_session_object.mouse_session_path,'widefield image')
            self.WidefieldImagePath_stored=os.path.join(self.widefield_data_path,file_name)


    def create_figure_and_plot(self):
        self.load_all()
        self.fig, self.ax=plt.subplots(1)
        self.plot_image(self.ax)
        self.plot_rois(self.ax)

    def load_all(self):
        self.load_image()
        self.load_rois()
        self.create_roi_patches()

    def plot_image(self,ax=None):
        self.imgplot = ax.imshow(self.widefield_image,cmap='inferno')
        
    def load_image(self):
        self.widefield_image = plt.imread(self.WidefieldImagePath_stored)
        
    def load_rois(self):
        zipfiles=glob.glob( self.widefield_data_path+'\*.zip')
        roifiles=glob.glob( self.widefield_data_path+'\*.roi')
        
        if zipfiles and not roifiles:
            zipobject= ZipFile(zipfiles[0]) 
            zipobject.extractall(self.widefield_data_path)
            roifiles=[os.path.join(self.widefield_data_path,x.filename) for x in zipobject.infolist()]
            
        self.rois_info=[]    
        for roi_file_path in roifiles:
            roi = ImagejRoi.fromfile(roi_file_path)
            x=roi.coordinates()[0][0]
            y=roi.coordinates()[0][1]
            width=roi.coordinates()[2][0]-roi.coordinates()[0][0]
            height=roi.coordinates()[1][1]-roi.coordinates()[0][1]
            
            self.rois_info.append((roi,x,y,width,height,os.path.split(roi_file_path)[1]))
    
    def create_roi_patches(self):
        self.roi_patches=[(roi[5],mpl.patches.Rectangle((roi[1], roi[2]), roi[3], roi[4], linewidth=1, edgecolor='r', facecolor='none')) for roi in self.rois_info]
        

    def plot_rois(self, ax=None) :
        for rect in self.roi_patches:
            ax.add_patch(rect[1])
            

