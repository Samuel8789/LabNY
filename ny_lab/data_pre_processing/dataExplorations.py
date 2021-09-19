# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:18:02 2021

@author: sp3660
"""

import caiman as cm
import os
import shutil


from ..AllFunctions.kalman_stack_filter import kalman_stack_filter
from ..AllFunctions.save_imagej_hdf5_tiff import save_imagej_hdf5
from  ..AllFunctions import  bidiphase as bd


class DataExplorations():
       
    
    def __init__(self, FOV_object):
        
        self.FOV=FOV_object
        self.all_aquisitions=self.FOV.all_aquisitions
        self.full_aquisitions=self.FOV.all_datasets
        self.all_datasets={}
        for element in self.full_aquisitions:
            for aq, aq_object in element.items():
                for dataset, dataset_object in aq_object.all_datasets.items():
                    self.all_datasets[aq + '_' + dataset]=dataset_object
                    
        self.all_raw_kalman=[self.apply_kalman(dataset) for k, dataset in self.all_datasets.items()]  
         
        self.all_tiffs_for_imagej={}   
        
    # def apply_IPCA_to_dataset(self, dataset_object):
    #     dataset_IPCA_denoised=dataset.IPCA_denoise()   


    def save_for_imagej(self, caiman_movie):
        
          save_imagej_hdf5(caiman_movie, self.dataset_kalman_caiman_movie_name, '.tiff')
        
    def send_video_to_dektop(self, file):
        
        desktop=r'C:\Users\sp3660\Desktop'
        shutil.copy(file, desktop)

        print('in progress')

    # def apply_bidi_shifts(self):
        
    #     bd.compute(self.)
    

        
        