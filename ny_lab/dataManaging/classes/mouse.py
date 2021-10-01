# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:18:52 2021

@author: sp3660
"""
import logging 
logger = logging.getLogger(__name__)
import os
from ...AllFunctions.create_dir_structure import create_dir_structure
from .mouseImagingSession import MouseImagingSession

class Mouse:
    
    level0_structure={'surgeries',
                'treatments',
                'training',
                'imaging',
                'euthanasia',   
                'histology',
                'ex vivo',
                'data'
                }
    
    surgeries_structure={'injection',
                    'cranial window',
                    }
      
    def __init__(self, Mouse_Name, LabNY_object, data_managing_object=False, mouse_info=None):
        
        self.mouse_name=Mouse_Name
        self.LabNY_object=LabNY_object
        self.data_managing_object=data_managing_object
        # self.primary_data_directory=self.data_managing_object.primary_data_directory
        # self.secondary_data_directory=self.data_managing_object.secondary_data_path
        self.Database_ref=self.data_managing_object.Database_ref
        self.all_mouse_inf=mouse_info
        #%% HERES IS WHEN CHANGING DISCS   
        self.mouse_slow_subproject_path =self.all_mouse_inf.iloc[0]['SlowStoragePath']
        self.mouse_slow_subproject_path=self.mouse_slow_subproject_path.replace('\\?\D:\Projects', '\\?\F:\Projects')
        
        self.mouse_working_subproject_path=self.all_mouse_inf.iloc[0]['WorkingStoragePath']

        create_dir_structure( self.mouse_slow_subproject_path , Mouse.level0_structure)
        create_dir_structure(os.path.join( self.mouse_slow_subproject_path ,'surgeries'), Mouse.surgeries_structure)
          
        
        quey_imaging_session="""SELECT b.ImagingDate AS SessionDate, a.*, b.* ,c.*, d.*
                            FROM ImagedMice_table a 
                            LEFT JOIN ImagingSessions_table b ON b.ID=a.SessionID 
                            LEFT JOIN ExperimentalAnimals_table c ON c.ID=a.ExpID 
                            LEFT JOIN MICE_table d ON d.ID=c.Mouse_ID 
                            WHERE c.Code=?
                            """
        params=(self.mouse_name,)
        self.imaging_sessions_database=self.Database_ref.arbitrary_query_to_df(quey_imaging_session, params)        
        self.load_all_imaging_sessions()
        
 
    def load_all_imaging_sessions(self):
        
        self.imaging_sessions_objects={session[0].replace('-', ''):MouseImagingSession(session[0].replace('-', ''), imaging_session_ID=session[1], mouse_object=self)  for idx, session in self.imaging_sessions_database.iterrows()}       
    
    def add_prairie_session(self, raw_imaging_session_path, session_name):
        print('Processing '+self.mouse_name)
        # print('Adding prairie sessions')
        self.raw_imaging_sessions_objects={}
        self.raw_imaging_sessions_objects[session_name]=MouseImagingSession(session_name, raw_imaging_session_path=raw_imaging_session_path, mouse_object=self)
        # self.load_all_imaging_sessions()
       
    def get_all_mouse_FOVdata_datasets(self):
        
        self.all_mouse_FOVdata_datasets={}
        
        for name1, imaging_session in self.imaging_sessions_objects.items():
            for name2, FOV in imaging_session.all_FOVs.items():
                for name3, aquisition in FOV.all_aquisitions.items():
                    for name4, dataset in aquisition.all_datasets.items():
                        self.all_mouse_FOVdata_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset
            
       
    def get_all_mouse_acquisitions_datasets(self):  
       self.all_mouse_acquisitions_datasets={}
       for name1, imaging_session in self.imaging_sessions_objects.items():
                  
           for name2, aquisition in imaging_session.all_nonimaging_Aquisitions.items():
               for name3, dataset in aquisition.all_datasets.items():
                       self.all_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3]=dataset
           for name2, aquisition in imaging_session.all_Test_Aquisitions.items():
               for name3, dataset in aquisition.all_datasets.items():
                       self.all_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3]=dataset           
           for name2, aquisition in imaging_session.all_0coordinate_Aquisitions.items():
               for name3, dataset in aquisition.all_datasets.items():
                       self.all_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3]=dataset       
           for name2, FOV in imaging_session.all_FOVs.items():
               for name3, aquisition in FOV.all_existing_1050tomato.items():
                   for name4, dataset in aquisition.all_datasets.items():
                       self.all_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset               
               for name3, aquisition in FOV.all_existing_10503planetomato.items():
                   for name4, dataset in aquisition.all_datasets.items():
                       self.all_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset               
               for name3, aquisition in FOV.all_existing_1050HighResStackTomato.items():
                   for name4, dataset in aquisition.all_datasets.items():
                       self.all_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset               
               for name3, aquisition in FOV.all_existing_HighResStackGreen.items():
                   for name4, dataset in aquisition.all_datasets.items():
                       self.all_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset               
               for name3, aquisition in FOV.all_existing_OtherAcq.items():
                   for name4, dataset in aquisition.all_datasets.items():
                       self.all_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset               
               for name3, aquisition in FOV.all_existing_SurfaceImage.items():
                   for name4, dataset in aquisition.all_datasets.items():
                       self.all_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset
           

