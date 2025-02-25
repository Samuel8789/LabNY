# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:18:52 2021
@author: sp3660
"""
import logging 
module_logger = logging.getLogger(__name__)
import os
import gc
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
      
    def __init__(self, Mouse_Name, LabNY_object, data_managing_object=False, mouse_info=None, cloud=False):
        # module_logger.info('Instantiating ' +__name__)

        self.raw_imaging_sessions_objects={}

        self.mouse_name=Mouse_Name
        if Mouse_Name=='SPSM':
            print('stop')
            
            
        module_logger.info('Loading Mouse ' +self.mouse_name)


        self.LabNY_object=LabNY_object
        self.data_managing_object=data_managing_object
        # self.primary_data_directory=self.data_managing_object.primary_data_directory
        # self.secondary_data_directory=self.data_managing_object.secondary_data_path
        self.Database_ref=self.data_managing_object.Database_ref
        self.all_mouse_inf=mouse_info
        #%% HERES IS WHEN CHANGING DISCS 
        if not cloud:
            self.mouse_slow_subproject_path_db =self.all_mouse_inf.iloc[0]['SlowStoragePath']
            # # self.mouse_slow_subproject_path=self.mouse_slow_subproject_path.replace('\\?\D:\Projects', '\\?\F:\Projects')
            self.mouse_working_subproject_path_db=self.all_mouse_inf.iloc[0]['WorkingStoragePath']
            
            self.mouse_slow_subproject_path =self.data_managing_object.os_transform_databasepath( self.mouse_slow_subproject_path_db)
            self.mouse_working_subproject_path=self.data_managing_object.os_transform_databasepath( self.mouse_working_subproject_path_db)

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
            self.load_all_imaging_sessions_from_database()
            self.get_all_mouse_acquisitions(self.imaging_sessions_objects)
            self.get_all_mouse_acquisitions_datasets(self.imaging_sessions_objects)
            self.get_all_mouse_FOVdata_datasets()
 
    
#%% mouse imaging sessions
    
    
    def read_processed_imaging_session_not_in_database(self, session_name):
         module_logger.info('Reading '+ self.mouse_name)
         self.imaging_sessions_not_yet_database_objects={}
         self.imaging_sessions_not_yet_database_objects[session_name]=MouseImagingSession(session_name, mouse_object=self, yet_to_add=True)
         self.get_all_mouse_acquisitions(self.imaging_sessions_not_yet_database_objects)

    def load_all_imaging_sessions_from_database(self):
        
        self.imaging_sessions_objects={session[0].replace('-', ''):MouseImagingSession(session[0].replace('-', ''), imaging_session_ID=session[1], mouse_object=self)  for idx, session in self.imaging_sessions_database.iterrows()}  
    
    def add_prairie_session(self, raw_imaging_session_path, session_name):
        module_logger.info('Processing '+ self.mouse_name)
        # module_logger.info('Adding prairie sessions')
        self.raw_imaging_sessions_objects[session_name]=MouseImagingSession(session_name, raw_imaging_session_path=raw_imaging_session_path, mouse_object=self, yet_to_add=True)
        # self.unload_full_imaging_session(session_name)
        # self.load_all_imaging_sessions()
       
        
       
#%% reading datasets
            
    def get_all_mouse_FOVdata_datasets(self):
        
        self.all_mouse_FOVdata_datasets={}
        
        for name1, imaging_session in self.imaging_sessions_objects.items():
            for name2, FOV in imaging_session.all_FOVs.items():
                for name3, aquisition in FOV.all_aquisitions.items():
                    for name4, dataset in aquisition.all_datasets.items():
                        self.all_mouse_FOVdata_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset
                        


    def get_all_mouse_acquisitions(self, session_list):  
       self.all_mouse_acquisitions={}
       for name1, imaging_session in session_list.items():
                  
           for name2, aquisition in imaging_session.all_nonimaging_Aquisitions.items():
                       self.all_mouse_acquisitions[name1+'_'+name2]=aquisition
                       
           for name2, aquisition in imaging_session.all_Test_Aquisitions.items():
                       self.all_mouse_acquisitions[name1+'_'+name2]=aquisition 
                       
           for name2, aquisition in imaging_session.all_0coordinate_Aquisitions.items():
                       self.all_mouse_acquisitions[name1+'_'+name2]=aquisition  
                       
           for name2, FOV in imaging_session.all_FOVs.items():
               
               for name3, aquisition in FOV.all_aquisitions.items():
                       self.all_mouse_acquisitions[name1+'_'+name2+'_'+name3]=aquisition
        
               for name3, aquisition in FOV.all_existing_1050tomato.items():
                       self.all_mouse_acquisitions[name1+'_'+name2+'_'+name3]=aquisition    
                       
               for name3, aquisition in FOV.all_existing_10503planetomato.items():
                       self.all_mouse_acquisitions[name1+'_'+name2+'_'+name3]=aquisition 
                       
               for name3, aquisition in FOV.all_existing_1050HighResStackTomato.items():
                       self.all_mouse_acquisitions[name1+'_'+name2+'_'+name3]=aquisition  
                       
               for name3, aquisition in FOV.all_existing_HighResStackGreen.items():
                       self.all_mouse_acquisitions[name1+'_'+name2+'_'+name3]=aquisition 
                       
               for name3, aquisition in FOV.all_existing_OtherAcq.items():
                       self.all_mouse_acquisitions[name1+'_'+name2+'_'+name3]=aquisition 
                       
               for name3, aquisition in FOV.all_existing_SurfaceImage.items():
                       self.all_mouse_acquisitions[name1+'_'+name2+'_'+name3]=aquisition
           
    def get_all_mouse_acquisitions_datasets(self, session_list):  
       self.all_mouse_acquisitions_datasets={}
       for name1, imaging_session in session_list.items():
                  
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
               
               for name3, aquisition in FOV.all_aquisitions.items():
                   for name4, dataset in aquisition.all_datasets.items():
                       self.all_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset 
               
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
                       
    def get_all_mouse_raw_acquisitions_datasets(self, session_list):  
       self.all_raw_mouse_acquisitions_datasets={}
       for name1, imaging_session in session_list.items():
                  
           for name2, aquisition in imaging_session.all_raw_nonimaging_Aquisitions.items():
               for name3, dataset in aquisition.all_raw_datasets.items():
                       self.all_raw_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3]=dataset
                       
           for name2, aquisition in imaging_session.all_raw_Test_Aquisitions.items():
               for name3, dataset in aquisition.all_raw_datasets.items():
                       self.all_raw_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3]=dataset  
                       
           for name2, aquisition in imaging_session.all_raw_0coordinate_Aquisitions.items():
               for name3, dataset in aquisition.all_raw_datasets.items():
                       self.all_raw_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3]=dataset  
                       
           for name2, FOV in imaging_session.all_raw_FOVs.items():
               
               for name3, aquisition in FOV.all_raw_aquisitions.items():
                   for name4, dataset in aquisition.all_raw_datasets.items():
                       self.all_raw_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset 
               
               for name3, aquisition in FOV.all_existing_1050tomato.items():
                   for name4, dataset in aquisition.all_raw_datasets.items():
                       self.all_raw_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset   
                       
               for name3, aquisition in FOV.all_existing_10503planetomato.items():
                   for name4, dataset in aquisition.all_raw_datasets.items():
                       self.all_raw_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset 
                       
               for name3, aquisition in FOV.all_existing_1050HighResStackTomato.items():
                   for name4, dataset in aquisition.all_raw_datasets.items():
                       self.all_raw_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset  
                       
               for name3, aquisition in FOV.all_existing_HighResStackGreen.items():
                   for name4, dataset in aquisition.all_raw_datasets.items():
                       self.all_raw_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset  
                       
               for name3, aquisition in FOV.all_existing_OtherAcq.items():
                   for name4, dataset in aquisition.all_raw_datasets.items():
                       self.all_raw_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset  
                       
               for name3, aquisition in FOV.all_existing_SurfaceImage.items():
                   for name4, dataset in aquisition.all_raw_datasets.items():
                       self.all_raw_mouse_acquisitions_datasets[name1+'_'+name2+'_'+name3+'_'+name4]=dataset
#%% laoding datasets
                       
                       
    def load_processed_imaging_session_not_in_database(self, session_name):
      
        for acq in  self.all_mouse_acquisitions.values():
            acq.load_all()
            
    def unload_full_imaging_session(self, session_name):
        module_logger.info('Unloading  '+ session_name)

        if self.raw_imaging_sessions_objects:
            del self.raw_imaging_sessions_objects
            gc.collect()
