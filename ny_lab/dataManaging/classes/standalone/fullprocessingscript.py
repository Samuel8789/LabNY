# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:44:05 2021

@author: sp3660
"""
from metadata import Metadata
from motionCorrectedKalman import MotionCorrectedKalman
from summaryImages import SummaryImages
from bidiShiftManager import BidiShiftManager
from caimanExtraction import CaimanExtraction
from voltageSignalsExtractions import VoltageSignalsExtractions
import glob
import os


temporary_path_base='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset'
acquisition_path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20211113\Mice\SPKQ\FOV_1\Aq_1\211113_SPKQ_FOV1_2planeAllenA_20x_920_50024_narrow_without-000'
acquisition_name=os.path.split(acquisition_path)[1]
rawchannle='Ch2Green'
klamanext='_Shifted_Movie_MC_kalman.tiff'

#%% correct bidishifts
firstplane=2
lastplane=2

kalman_path=[]
temporary_path=[]
dataset_image_sequence_path=[]
bidishits=[]
meta=[]
voltagesignals=[]
shiftedmmap=[]
metadata_file_path=[]
CaimanExtr=[]
MCKalman=[]
SumImages_kalman=[]

for i, plane in enumerate(range(firstplane, lastplane+1)):
    rawplane='plane'+str(plane)
    tempplane='Plane'+str(plane)
    kalman_path.append(os.path.join(temporary_path_base, acquisition_name, tempplane,acquisition_name+klamanext))
    temporary_path.append(os.path.join(temporary_path_base, acquisition_name,tempplane ))
    dataset_image_sequence_path.append(os.path.join(acquisition_path, rawchannle, rawplane ))
    # bidishits.append( BidiShiftManager(dataset_image_sequence_path=dataset_image_sequence_path[i], temporary_path=temporary_path[i] ))
    # meta.append(Metadata(acquisition_directory_raw=acquisition_path, temporary_path=temporary_path[i]))
    # voltagesignals.append(VoltageSignalsExtractions(acquisition_directory_raw=acquisition_path, temporary_path=temporary_path[i], just_copy=True))
    shiftedmmap.append([i for i in glob.glob(temporary_path[i]+'\\**.mmap') if 'Shifted_Movie_d1' in i ][0])
    metadata_file_path.append([i for i in glob.glob(temporary_path[i]+'\\**.xml') if 'Cycle' not in i][0])
    CaimanExtr.append(CaimanExtraction(shiftedmmap[i], metadata_file_path[i], temporary_path=temporary_path[i], first_pass_mot_correct=True))
    MCKalman.append(MotionCorrectedKalman(shifted_mmap_path=shiftedmmap[i]))
    SumImages_kalman.append(SummaryImages(image_sequence_path=kalman_path[i]))
    
    
    
    
    


