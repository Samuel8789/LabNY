# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:19:19 2021

@author: sp3660
"""

# manually process a movie

from bidiShiftManager import BidiShiftManager
from motionCorrectedKalman import MotionCorrectedKalman
from summaryImages import SummaryImages
import os

filename=r'211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'
dataset_image_sequence_path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20211015\Mice\SPKG\FOV_1\Aq_1\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\Ch1Red\plane1'
dump='\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp'
temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\Plane1\Red'

dataset_full_file_mmap_path=os.path.join(temporary_path, filename+'_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_64416_.mmap')
motionshifts_file_path=os.path.join(temporary_path,'211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_MC_shifts.txt')
register_template_full_path=os.path.join(temporary_path,'211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_MC_template.tif')
bidiphase_file_path=os.path.join(temporary_path, '211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Bidiphases.txt')
kalman_path=os.path.join(temporary_path,r'211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_MC_kalman.tiff')

#%%
bidihits = BidiShiftManager(dataset_image_sequence_path=dataset_image_sequence_path, temporary_path=temporary_path, bidiphase_file_path=bidiphase_file_path )
#%%
MotCorr = MotionCorrectedKalman(shifted_mmap_path=dataset_full_file_mmap_path, dump_temp_path=dump, motionshifts_file_path=motionshifts_file_path, register_template_full_path=register_template_full_path, keep_registered=True )
#%%
SumImages_kalman = SummaryImages(image_sequence_path=kalman_path)
SumImages_raw = SummaryImages(image_sequence_path=dataset_full_file_mmap_path)