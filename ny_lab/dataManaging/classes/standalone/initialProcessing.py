# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 18:05:23 2022

@author: sp3660
"""


from .metadata import Metadata
from .motionCorrectedKalman import MotionCorrectedKalman
from .summaryImages import SummaryImages
from .bidiShiftManager import BidiShiftManager
from .caimanExtraction import CaimanExtraction
from .voltageSignalsExtractions import VoltageSignalsExtractions
from ...functions.get_slow_working_paths import get_slow_working_paths
from ..eyeVideo import EyeVideo
from ..wideFieldImage import WideFieldImage

import logging 
module_logger = logging.getLogger(__name__)

import glob
import os

class InitialProcessing():
    def __init__(self, acquisition_object=None):
        module_logger.info('Instantiating ' +__name__)

        self.acquisition_object=acquisition_object
        
        self.acquisition_object.mouse_aquisition_path
        
        if self.acquisition_object.mouse_imaging_session_object:
            dat_ref=self.acquisition_object.mouse_imaging_session_object.mouse_object.Database_ref
            mouse_code=self.acquisition_object.mouse_imaging_session_object.mouse_code
            
        elif self.acquisition_object.FOV_object:
            dat_ref=self.acquisition_object.FOV_object.mouse_imaging_session_object.mouse_object.Database_ref
            mouse_code=self.acquisition_object.FOV_object.mouse_imaging_session_object.mouse_code
            
        elif self.acquisition_object.Atlas_object:
            dat_ref=self.acquisition_object.atlas_object.mouse_imaging_session_object.mouse_object.Database_ref
            mouse_code=self.acquisition_object.atlas_object.mouse_imaging_session_object.mouse_code


        self.databse_ref=dat_ref
        self.mouse_code=mouse_code
        acq=self.acquisition_object.Prairireaqpath
        
         
        query_get_exp_id= """
            SELECT SlowStoragePath
            FROM ExperimentalAnimals_table
            WHERE Code=?
            """
        params=(self.mouse_code,)
        slow_mouse_path=self.databse_ref.arbitrary_query_to_df(query_get_exp_id, params).values.tolist()[0][0]
        slowstoragepaths=get_slow_working_paths(acq,slow_mouse_path)

    
        slowstorpath='\\\\?\\'+slowstoragepaths['Aq']
        if os.path.isdir(os.path.join(acq,'Ch2Green'))     :
            plane_number=len(os.listdir(os.path.join(acq,'Ch2Green')))
        else:
            plane_number=len(os.listdir(os.path.join(acq,'Ch1Red')))

        acquisition_path=acq
        acquisition_name=os.path.split(acquisition_path)[1]
        rawchannle='Ch2Green'
        tomchan='Ch1Red'
        saved_channel1='Green'
        saved_channel2='Red'

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
        mcmmappath=[]
        
        # selected_eyevideo_raw_path=os.path.join(os.path.split(acq)[0],'FaceCamera')
        # file_name=''
        # raw_input_path=glob.glob(os.path.join(acq[:acq.find('Mice')+9], 'Widefield')+'\\**.tif')[0]
        
        # if os.path.isdir(selected_eyevideo_raw_path):
        #     module_logger.info('eye video initial processin')
        #     EyeVideo(aquisition_object=self.acquisition_object, selected_eyevideo_raw_path=selected_eyevideo_raw_path, eyecameraslowstoragepath=slowstoragepaths['EyeCamera'], eyecameraslowstoragepathmeta=slowstoragepaths['EyeCameraMeta'])
            
        # WideFieldImage(file_name, raw_input_path=raw_input_path, slostragepath=slowstoragepaths['WideField'])
        
        try:                      
                              
        
            for i, plane in enumerate(range(firstplane, lastplane+1)):
                rawplane='plane'+str(plane)
                tempplane='Plane'+str(plane)
                temporary_path.append(os.path.join(slowstorpath, 'planes',tempplane,saved_channel1))
    
    
                kalman_path.append(os.path.join(temporary_path[i], acquisition_name+klamanext))
                if not os.path.exists(temporary_path[i]):
                    os.makedirs(temporary_path[i])
                dataset_image_sequence_path.append(os.path.join(acquisition_path, rawchannle, rawplane ))
                dataset_image_sequence_path_red.append(os.path.join(acquisition_path, tomchan, rawplane ))
              
                channles={'Green':os.path.join(acq,'Ch2Green'), 'Red':os.path.join(acq,'Ch1Red')}
                imaged_channels={key: val for key, val in channles.items() if os.path.isdir(val)}
            
    
                for chan, channel_path in imaged_channels.items():
                    
                    if chan=='Green':
                        temporary_path[i]=os.path.join(slowstorpath, 'planes',tempplane,saved_channel1)
                        dataset_image_sequence_path[i]=dataset_image_sequence_path[i]
                    elif chan=='Red':
                        temporary_path[i]=os.path.join(slowstorpath, 'planes',tempplane,saved_channel2)
                        dataset_image_sequence_path[i]=dataset_image_sequence_path_red[i]
        
                    
                    if len(os.listdir(dataset_image_sequence_path[i]))<20 and len(os.listdir(dataset_image_sequence_path[i]))>2 :
                        module_logger.info('Bidishifting short aq' + acq + ' '+rawplane)
                        bidishits.append( BidiShiftManager(dataset_image_sequence_path=dataset_image_sequence_path[i], temporary_path=temporary_path[i] , fullscripttemp=True, custom_start_end=True))
                        module_logger.info('Metadating  short aq' + acq+ ' '+rawplane)
                        meta.append(Metadata(acquisition_directory_raw=acquisition_path, temporary_path=temporary_path[i]))
                        module_logger.info('Voltagextracting  short aq' + acq+ ' '+rawplane)
                        voltagesignals.append(VoltageSignalsExtractions(acquisition_directory_raw=acquisition_path, temporary_path=temporary_path[i], just_copy=True))
                        shiftedmmap.append([i for i in glob.glob(temporary_path[i]+'\\**.mmap') if 'Shifted_Movie_d1' in i ][0])
                        if [i for i in glob.glob(temporary_path[i]+'\\**.mmap') if 'Shifted_Movie_custom' in i ]:
                            shiftedmmap[i]=[i for i in glob.glob(temporary_path[i]+'\\**.mmap') if 'Shifted_Movie_custom' in i ][0]
                        metadata_file_path.append([i for i in glob.glob(temporary_path[i]+'\\**.xml') if 'Cycle' not in i][0])
                                                   
                        module_logger.info(acq)
                        try:
                            module_logger.info('Getting summary images short aq' + self.mouse_code + ' '+ acq + rawplane)
                            SumImages_kalman.append(SummaryImages(image_sequence_path=shiftedmmap[i]))
                        except:
                                module_logger.exception('Something wrong with summary images ' + self.mouse_code + ' '+ acq  + rawplane)
                        module_logger.info(acq)
             
                    
                    elif len(os.listdir(dataset_image_sequence_path[i]))<100:
                         break
                     
                    elif len(os.listdir(dataset_image_sequence_path[i]))>100:    
                        module_logger.info('Bidishifting ' + acq + ' '+rawplane)
                        bidishits.append( BidiShiftManager(dataset_image_sequence_path=dataset_image_sequence_path[i], temporary_path=temporary_path[i] , fullscripttemp=True, custom_start_end=True))
                        module_logger.info('Metadating ' + acq+ ' '+rawplane)
                        meta.append(Metadata(acquisition_directory_raw=acquisition_path, temporary_path=temporary_path[i]))
                        module_logger.info('Voltagextracting ' + acq+ ' '+rawplane)
                        voltagesignals.append(VoltageSignalsExtractions(acquisition_directory_raw=acquisition_path, temporary_path=temporary_path[i], just_copy=True))
                        shiftedmmap.append([i for i in glob.glob(temporary_path[i]+'\\**.mmap') if 'Shifted_Movie_d1' in i ][0])
                        if [i for i in glob.glob(temporary_path[i]+'\\**.mmap') if 'Shifted_Movie_custom' in i ]:
                            shiftedmmap[i]=[i for i in glob.glob(temporary_path[i]+'\\**.mmap') if 'Shifted_Movie_custom' in i ][0]
                        metadata_file_path.append([i for i in glob.glob(temporary_path[i]+'\\**.xml') if 'Cycle' not in i][0])
                        try:
                            module_logger.info('Running Caiman ' + self.mouse_code + ' '+ acq  + rawplane)
                            CaimanExtr.append(CaimanExtraction(shiftedmmap[i], metadata_file_path[i], temporary_path=temporary_path[i], first_pass_mot_correct=True, metdata_object=meta[i]))
                            if bidishits[i].start_end_flag:
                                CaimanExtr.append(CaimanExtraction(shiftedmmap[i], metadata_file_path[i], temporary_path=temporary_path[i], first_pass_mot_correct=True, metdata_object=meta[i], force_run=True))
                        except:
                            module_logger.exception('Something wrong with On Acid ' + self.mouse_code + ' '+ acq  + rawplane)
                        module_logger.info(acq)
                        try:
                            mcmmappath.append([i for i in glob.glob(temporary_path[i]+'\\**.mmap') if 'MC_OnACID_d1' in i ][0])
                            module_logger.info('Running Kalman Filter ' + self.mouse_code + ' '+ acq  + rawplane)
                            MCKalman.append(MotionCorrectedKalman(shifted_mmap_path=mcmmappath[i]))
                            kalman_path[i]=[i for i in glob.glob(temporary_path[i]+'\\**.tiff') if 'MC_kalman' in i ][0]
                        except:
                            module_logger.exception('Something wrong with kalman ' + self.mouse_code + ' '+ acq  + rawplane)
                        module_logger.info(acq)
                        try:
                            module_logger.info('Getting summary images ' + self.mouse_code + ' '+ acq + rawplane)
                            SumImages_kalman.append(SummaryImages(image_sequence_path=kalman_path[i]))
                        except:
                                module_logger.exception('Something wrong with summary images ' + self.mouse_code + ' '+ acq  + rawplane)
                        module_logger.info(acq)
        
        except:
            module_logger.exception('initial processing failed ' + self.mouse_code + ' '+ acq  + rawplane)
