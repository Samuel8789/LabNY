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

aqlist=[ #r'F:\Projects\LabNY\Imaging\2021\20211111\Mice\SPJM\TestAcquisitions\Aq_5\211111_SPJM_CellX_opto_25x_920_50024_narrow_with_10ms_14um_16sp_10x_1pw-000',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211111\Mice\SPJM\FOV_1\Aq_1\211111_SPJM_FOV1_2planeAllenA_25x_920_50024_narrow_with-000', jumbloed up after around80000frames
        # r'F:\Projects\LabNY\Imaging\2021\20211113\Mice\SPKF\FOV_1\Aq_1\211113_SPKF_FOV1_3planteAllenA_25x_920_50024_narrow_without-000',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211113\Mice\SPKG\FOV_1\Aq_1\211113_SPKG_FOV1_3PlaneAllenA_25x_920_50024_narrow_without-000',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211113\Mice\SPKQ\FOV_1\Aq_1\211113_SPKQ_FOV1_2planeAllenA_20x_920_50024_narrow_without-000', already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_1\OtherAcq\Aq_1\211117_SPJY_FOV1_Opto_25x_920_50024_narrow_without_210_0.6-000',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_2\OtherAcq\Aq_1\211117_SPJY_FOV2_Opto_25x_920_50024_narrow_without_210_0.6-000',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_3\OtherAcq\Aq_1\211117_SPJY_FOV3_Opto_25x_920_50024_narrow_without_210_0.8-000',already processed
        # # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_3\OtherAcq\Aq_2\211117_SPJY_FOV3_Opto_25x_920_50024_narrow_without_210_0.8-001',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_3\OtherAcq\Aq_3\211117_SPJY_FOV3_Opto_25x_920_50024_narrow_without_210_0.8-002',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_3\OtherAcq\Aq_4\211117_SPJY_FOV3_Opto_25x_920_50024_narrow_without_210_0.8-003',# less tan 1000
        # # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_3\OtherAcq\Aq_5\211117_SPJY_FOV3_Opto_25x_920_50024_narrow_without_210_0.8-004',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_3\OtherAcq\Aq_6\211117_SPJY_FOV3_Opto_25x_920_50024_narrow_without_210_0.8-005',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_3\OtherAcq\Aq_7\211117_SPJY_FOV3_Opto_25x_920_50024_narrow_without_210_0.8-006',# less tan 1000
        # # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_3\OtherAcq\Aq_8\211117_SPJY_FOV3_Opto_25x_920_50024_narrow_without_210_0.8-007',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_3\OtherAcq\Aq_9\211117_SPJY_FOV3_Opto_25x_920_50024_narrow_without_210_0.8-008',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_3\OtherAcq\Aq_10\211117_SPJY_FOV3_Opto_25x_920_50024_narrow_without_210_0.8-009',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_3\OtherAcq\Aq_11\211117_SPJY_FOV3_Opto_25x_920_50024_narrow_without_210_0.8-010',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_3\OtherAcq\Aq_12\211117_SPJY_FOV3_Opto_25x_920_50024_narrow_without_210_0.8-011',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_3\OtherAcq\Aq_13\211117_SPJY_FOV3_Opto_25x_920_50024_narrow_without_210_0.8-012',# less tan 1000
        # # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPJY\FOV_4\Aq_1\211117_SPJY_FOVX_10Minspont_25x_920_50024_narrow_without-000',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPKA\FOV_1\OtherAcq\Aq_1\211117_SPKA_Cell1Opto_25x_920_50024_narrow_with_10x_10_ms_16sp_210la_0.6di-000',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPKA\FOV_1\OtherAcq\Aq_2\211117_SPKA_Cell1Opto_25x_920_50024_narrow_with_10x_10_ms_16sp_210la_0.6di-001',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPKA\FOV_1\OtherAcq\Aq_3\211117_SPKA_Cell1Opto_25x_920_50024_narrow_with_10x_10_ms_16sp_210la_0.6di-002',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPKA\FOV_1\OtherAcq\Aq_4\211117_SPKA_Cell1Opto_25x_920_50024_narrow_with_10x_10_ms_16sp_210la_0.6di-003',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPKA\FOV_1\OtherAcq\Aq_5\211117_SPKA_Cell1Opto_25x_920_50024_narrow_with_10x_10_ms_16sp_21la_0.6di-000',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPKA\FOV_2\OtherAcq\Aq_1\211117_SPKA_Cell2Opto_25x_920_50024_narrow_with_10x_10_ms_16sp_210la_0.6di-000',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPKA\FOV_3\OtherAcq\Aq_1\211117_SPKA_Cell3Opto_25x_920_50024_narrow_with_10x_10_ms_16sp_210la_0.6di-000',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211117\Mice\SPKA\FOV_4\OtherAcq\Aq_1\211117_SPKA_CellXOpto10min_25x_920_50024_narrow_with_10x_10_ms_16sp_210la_0.6di-000',#already processed shutil.copy(self.voltage_excel_path, self.temporary_path)
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKA\FOV_1\OtherAcq\Aq_1\211118_SPKA_FOV1_opto_25x_920_50024_narrow_with_cell1_210_0.6-000',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKA\FOV_2\OtherAcq\Aq_1\211118_SPKA_FOV2_opto_25x_920_50024_narrow_with_cell1_210_0.6-001',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKA\FOV_2\OtherAcq\Aq_2\211118_SPKA_FOV2_opto_25x_920_50024_narrow_with_cell1_210_1-002',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKA\FOV_3\OtherAcq\Aq_1\211118_SPKA_FOV3_opto_25x_920_50024_narrow_with_cell1_210_1-003',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKA\FOV_4\OtherAcq\Aq_1\211118_SPKA_FOVGood_optotest2cells_25x_920_50024_narrow_with_cell1_2_210_1-000',already processed
        r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKA\FOV_4\OtherAcq\Aq_2\211118_SPKA_FOVGood_Surface_25x_920_50024_narrow_with-000',# 10 frame
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKM\FOV_1\OtherAcq\Aq_1\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_cell1_210_1-000',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKM\FOV_2\OtherAcq\Aq_1\211118_SPKN_FOV2_opto_25x_920_50024_narrow_with_cell1_210_1-000',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKM\FOV_2\OtherAcq\Aq_2\211118_SPKN_FOV2_opto_25x_920_50024_narrow_with_cell1_210_1.5-001',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKM\FOV_2\OtherAcq\Aq_3\211118_SPKN_FOV2_opto_25x_920_50024_narrow_with_cell2_210_1.2-000',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKM\FOV_2\OtherAcq\Aq_4\211118_SPKN_FOV2_opto_25x_920_50024_narrow_with_cell2_210_1.2-001',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKM\FOV_3\OtherAcq\Aq_1\211118_SPKN_FOV3_opto_25x_920_50024_narrow_with_cell1_210_1.2-000',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_1\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_0.5_Cell1-000',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_2\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_0.5_Cell2-000',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_3\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_0.5_Cell3-000',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_4\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1.5_Cell1-001',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_5\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1.5_Cell10-009',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_6\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1.5_Cell11-13-010',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_7\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1.5_Cell3-002',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_8\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1.5_Cell9-008',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_9\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1_Cell2-004',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_10\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1_Cell3-001',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_11\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1_Cell4-000',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_12\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1_Cell4-001',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_13\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1_Cell4-003',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_14\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1_Cell5-000',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_15\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1_Cell5-001',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_16\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1_Cell6-005',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_17\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1_Cell7-006',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_18\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_1_Cell8-007',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_1\OtherAcq\Aq_19\211118_SPKN_FOV1_opto_25x_920_50024_narrow_with_210_2_Cell5-000',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_2\OtherAcq\Aq_1\211118_SPKN_FOV2_opto_25x_920_50024_narrow_with_210_1.5_Cell1-013',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_2\OtherAcq\Aq_2\211118_SPKN_FOV2_opto_25x_920_50024_narrow_with_210_1.5_Cell1-4-011',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_2\OtherAcq\Aq_3\211118_SPKN_FOV2_opto_25x_920_50024_narrow_with_210_1_Cell1-012',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_3\OtherAcq\Aq_1\211118_SPKN_FOV3_opto_25x_920_50024_narrow_with_210_1.3_Cell1-14-014',already processed
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_3\OtherAcq\Aq_2\211118_SPKN_FOV3_opto_25x_920_50024_narrow_with_210_1.3_Cell1-14-015', not converted from raw deleted
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_3\OtherAcq\Aq_3\211118_SPKN_FOV3_opto_25x_920_50024_narrow_with_210_1_14-015',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_4\OtherAcq\Aq_1\211118_SPKN_FOV4_opto_25x_920_50024_narrow_with_210_1.5_Cell1-017',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_4\OtherAcq\Aq_2\211118_SPKN_FOV4_opto_25x_920_50024_narrow_with_210_1_Cell1-016',# less tan 1000
        # r'F:\Projects\LabNY\Imaging\2021\20211118\Mice\SPKN\FOV_4\OtherAcq\Aq_3\211118_SPKN_FOV4_opto_25x_920_50024_narrow_with_210_2_Cell1-018']# less tan 1000
        r'F:\Projects\LabNY\Imaging\2021\20211105\Mice\SPJM\FOV_1\Aq_1\211105_SPJM_FOV1_10MinSpont_25x_920_50024_narrow_with_-000',
        r'F:\Projects\LabNY\Imaging\2021\20211105\Mice\SPJM\TestAcquisitions\Aq_1\211105_SPJM_Test_25x_920_50024_narrow_with_opto_13um_16sp_12ms_20x_10i_1pw-002',
        r'F:\Projects\LabNY\Imaging\2021\20211105\Mice\SPJM\TestAcquisitions\Aq_2\211105_SPJM_Test_25x_920_50024_narrow_with_opto_13um_16sp_12ms_20x_10i_1pw-003',
        r'F:\Projects\LabNY\Imaging\2021\20211105\Mice\SPJM\TestAcquisitions\Aq_3\211105_SPJM_Test_25x_920_50024_narrow_with_opto_13um_16sp_12ms_20x_10i_25pw-000',
        r'F:\Projects\LabNY\Imaging\2021\20211105\Mice\SPJM\TestAcquisitions\Aq_4\211105_SPJM_Test_25x_920_50024_narrow_with_opto_13um_16sp_12ms_20x_10i_25pw-001',
        r'F:\Projects\LabNY\Imaging\2021\20211105\Mice\SPJM\TestAcquisitions\Aq_5\211105_SPJM_Test_25x_920_50024_narrow_with_opto_13um_16sp_12ms_20x_10i_50pw-004',
        r'F:\Projects\LabNY\Imaging\2021\20211105\Mice\SPJM\TestAcquisitions\Aq_6\211105_SPJM_Test_25x_920_50024_narrow_with_opto_13um_16sp_12ms_20x_10i_50pw-005',
        ]
plane_numbers=[]
undone=[] 
for acq in aqlist:
    plane_number=len(os.listdir(os.path.join(acq,'Ch2Green')))
    plane_numbers.append(plane_number)
    temporary_path_base='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset'
    acquisition_path='\\\\?\\'+acq
    acquisition_name=os.path.split(acquisition_path)[1]
    rawchannle='Ch2Green'
    tomchan='Ch1Red'

    klamanext='_Shifted_Movie_MC_kalman.tiff'
    
    
    #%% correct bidishifts
    firstplane=1
   
    lastplane=plane_number
    
    kalman_path=[]
    temporary_path=[]
    dataset_image_sequence_path=[]
    dataset_image_sequence_path_red=[]

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
        if not os.path.exists(temporary_path[i]):
            os.makedirs(temporary_path[i])
        dataset_image_sequence_path.append(os.path.join(acquisition_path, rawchannle, rawplane ))
        dataset_image_sequence_path_red.append(os.path.join(acquisition_path, tomchan, rawplane ))

        
        if len(os.listdir(dataset_image_sequence_path[i]))<100:
            undone.append(dataset_image_sequence_path[i])
            break
        print(acq)
        bidishits.append( BidiShiftManager(dataset_image_sequence_path=dataset_image_sequence_path[i], temporary_path=temporary_path[i] , fullscripttemp=True))
        print(acq)
        meta.append(Metadata(acquisition_directory_raw=acquisition_path, temporary_path=temporary_path[i]))
        print(acq)
        voltagesignals.append(VoltageSignalsExtractions(acquisition_directory_raw=acquisition_path, temporary_path=temporary_path[i], just_copy=True))
        print(acq)
        shiftedmmap.append([i for i in glob.glob(temporary_path[i]+'\\**.mmap') if 'Shifted_Movie_d1' in i ][0])
        print(acq)
        metadata_file_path.append([i for i in glob.glob(temporary_path[i]+'\\**.xml') if 'Cycle' not in i][0])
        print(acq)
        CaimanExtr.append(CaimanExtraction(shiftedmmap[i], metadata_file_path[i], temporary_path=temporary_path[i], first_pass_mot_correct=True))
        print(acq)
        MCKalman.append(MotionCorrectedKalman(shifted_mmap_path=shiftedmmap[i]))
        print(acq)
        SumImages_kalman.append(SummaryImages(image_sequence_path=kalman_path[i]))
        print(acq)
        
        
        
        
        
    
    
