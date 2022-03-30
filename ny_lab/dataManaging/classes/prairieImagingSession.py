# -*- coding: utf-8 -*-
"""
Created on Sun May 23 08:26:29 2021

@author: sp3660
"""

import logging 
module_logger = logging.getLogger(__name__)
# from .mouse import Mouse
import os
import glob
from ..functions.functionsDataOrganization import check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories
# from functionsDataOrganization import check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories

# import shutil
# from ..functions.select_face_camera import select_face_camera


class PrairieImagingSession():

    """
    Clas Representing a single praririe imaging seesion. It had to include every parameter important for the imaging
    Associate with the database
        If come form database load
        If doesn come from database create new database entry
            first chechk database
        
    """
    def __init__(self, ImagingDatabaseObject=None, session_raw_path=False, session_ID=False, data_managing_object=False ):
        # module_logger.info('Instantiating ' +__name__)

        
        self.ImagingDatabase=ImagingDatabaseObject
        self.datamanagingobject=data_managing_object
        
        
        # Case procesing raw prairire session
        if session_raw_path: 
            
            self.imaging_session_raw_path=session_raw_path
            self.imaging_session_name= os.path.basename(session_raw_path)
            module_logger.info('Loading Raw Session ' +self.imaging_session_name)

            self.imaging_session_mice_path=os.path.join(self.imaging_session_raw_path,'Mice')
            self.imaging_session_calibration_path=os.path.join(self.imaging_session_raw_path, 'Calibrations')
            self.session_imaged_mice_paths=glob.glob(self.imaging_session_mice_path+'\\*\\')
            self.cleaning_up_empty_mouse()
            # self.cleaning_up_calibrations()
            self.session_imaged_mice_codes=[mouse_path[-5:-1] for mouse_path in self.session_imaged_mice_paths if '_' not in mouse_path]
            

        # Case building prairie session from database
        elif session_ID:
            self.FullDatabase=self.ImagingDatabase.databse_ref
            
            self.imaging_session_ID= session_ID
            self.imaging_session_full_info=ImagingDatabaseObject.all_imaging_sessions.iloc[self.imaging_session_ID-1]
            self.imaging_session_raw_path=self.imaging_session_full_info.ImagingSessionRawPath
            self.imaging_session_name= os.path.basename(self.imaging_session_raw_path)
            module_logger.info('Loading Database Session ' +self.imaging_session_name)

            self.imaging_session_mice_path=self.imaging_session_full_info.MiceRawPath
            self.imaging_session_calibration_path=self.imaging_session_full_info.CalibrationsRawPath
            get_mice_info_for_session_query="""
                                        SELECT a.*, b.Code, b.Project
                                        FROM ImagedMice_table a
                                        LEFT JOIN ExperimentalAnimals_table b ON b.ID=a.ExpID
                                        LEFT JOIN ImagingSessions_table c ON c.ID=a.SessionID
                                        LEFT JOIN MICE_table d ON d.ID=b.Mouse_ID
                                        WHERE a.SessionID=?
                                        """
            params=  (self.imaging_session_ID,)     
            self.session_imaged_mice_info=self.FullDatabase.arbitrary_query_to_df(get_mice_info_for_session_query, params) 
            self.session_imaged_mice_paths=self.session_imaged_mice_info.MouseRawPath
            self.session_imaged_mice_codes=self.session_imaged_mice_info.Code
            self.session_imaged_mice_codes=self.session_imaged_mice_codes.values.tolist()
     
#%%            
    def load_all_yet_to_database_mice(self):
        module_logger.info('Loading mice not in database ' +self.imaging_session_name)
        for mouse_code in self.session_imaged_mice_codes:
             self.datamanagingobject.all_experimetal_mice_objects[mouse_code].load_processed_imaging_session_not_in_database(self.imaging_session_name)
       
    def read_all_yet_to_database_mice(self):
        module_logger.info('Loading mice not in database ' +self.imaging_session_name)
        for mouse_code in self.session_imaged_mice_codes:
             self.datamanagingobject.all_experimetal_mice_objects[mouse_code].read_processed_imaging_session_not_in_database(self.imaging_session_name)   
            
    def process_all_imaged_mice(self):
        module_logger.info('Preprocessing New Mice ' +self.imaging_session_name)
        for mouse_code in self.session_imaged_mice_codes:
            if mouse_code!='SPHQ':
             self.datamanagingobject.all_experimetal_mice_objects[mouse_code].add_prairie_session(self.imaging_session_raw_path, self.imaging_session_name)
           
             #%%
    def cleaning_up_empty_mouse(self):
       removed_mice_codes=[recursively_eliminate_empty_folders(mouse_path) for mouse_path in self.session_imaged_mice_paths if '_'  in mouse_path]
            
        
          
    # def deal_with_calibrations(self)
    def cleaning_up_raw_mouse_acquisitions(self):
             
        mice_paths=glob.glob(self.imaging_session_mice_path+'\\SP**', recursive=False)
       
  
          
    def  cleaning_up_calibrations(self, session_path):  
          
          calibrations_directory_path=os.path.join(session_path,'Calibrations')
          recursively_eliminate_empty_folders(calibrations_directory_path)       