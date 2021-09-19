# -*- coding: utf-8 -*-
"""
Created on Sun May 23 08:26:29 2021

@author: sp3660
"""

import logging 
logger = logging.getLogger(__name__)
from .mouse import Mouse
import os
import glob

class PrairieImagingSession():
    """
    Clas Representing a single praririe imaging seesion. It had to include every parameter important for the imaging
    Associate with the database
        If come form database load
        If doesn come from database create new database entry
            first chechk database
        
    """
    def __init__(self, ImagingDatabaseObject, session_raw_path=False, session_ID=False, data_managing_object=False ):
        
        self.ImagingDatabase=ImagingDatabaseObject
        self.FullDatabase=self.ImagingDatabase.databse_ref
        self.datamanagingobject=data_managing_object
        if session_raw_path: 
            
            self.imaging_session_raw_path=session_raw_path
            self.imaging_session_name= os.path.basename(session_raw_path)
            self.imaging_session_mice_path=os.path.join(self.imaging_session_raw_path,'Mice')
            self.imaging_session_calibration_path=os.path.join(self.imaging_session_raw_path, 'Calibrations')
            self.session_imaged_mice_paths=glob.glob(self.imaging_session_mice_path+'\\*\\')
            self.session_imaged_mice_codes=[mouse_path[-4:] for mouse_path in self.session_imaged_mice_paths]

           
        elif session_ID:
            
            self.imaging_session_ID= session_ID
            self.imaging_session_full_info=ImagingDatabaseObject.all_imaging_sessions.iloc[self.imaging_session_ID-1]
            self.imaging_session_raw_path=self.imaging_session_full_info.ImagingSessionRawPath
            self.imaging_session_name= os.path.basename(self.imaging_session_raw_path)
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

    def load_all_imaged_mice(self):
        print('loading new mice mmaps')
        for mouse_code in self.session_imaged_mice_codes.values.tolist():
            if mouse_code!='SPHQ':
             self.datamanagingobject.all_experimetal_mice_objects[mouse_code].add_prairie_session(self.imaging_session_raw_path, self.imaging_session_name)
              
              
           

        
    
  
  
          
    # def deal_with_calibrations(self)
     
          
      
      