# -*- coding: utf-8 -*-
"""
Created on Sun May 23 15:45:09 2021

@author: sp3660
"""

import pandas as pd 
import os
import glob
import shutil
import tkinter as Tkinter
from distutils.dir_util import copy_tree
import logging 
from pathlib import Path


module_logger  = logging.getLogger(__name__)

from .functions.select_face_camera import select_face_camera
from .functions.functionsDataOrganization import check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories
from .functions.transform_path import transform_path
from .classes.mouse import Mouse
from .classes.prairieImagingSession  import PrairieImagingSession
import datetime

class DataManaging():

    
    def __init__(self, project_object, full=True):
        module_logger.info('Instantiating ' +__name__)

        
        self.LabProjectObject=project_object
        

        self.Database_ref=self.LabProjectObject.database
        # self.update_all_imaging_data_paths()
        if full:
            self.data_paths=['Imaging', r'Full_Mice_Pre_Processed_Data'+os.sep+'Mice_Projects', r'Working_Mice_Data_1'+os.sep+'Mice_Projects', r'Working_Mice_Data_2'+os.sep+'Mice_Projects', r'Full_Mice_Pre_Processed_Data'+os.sep+'Mice_Projects', 'Imaging']
            self.data_paths_data={name:os.path.join(self.LabProjectObject.data_paths_project[name], self.data_paths[i])  for i , name in enumerate(self.LabProjectObject.data_paths_names)}     
    
            
            # self.raw_data= self.data_paths_data['Raw']
            # self.analysis_fast_1_data_path=self.data_paths_data['Analysis_Fast_1']
            # self.analysis_fast_2_data_path=self.data_paths_data['Analysis_Fast_2']
            # self.pre_processed_data_path=self.data_paths_data['Pre_proccessed_slow']
            
    
            # self.primary_data_directory=self.analysis_fast_1_data_path
            # self.secondary_data_path=self.pre_processed_data_path
                 
            """
            1read primary data
            2read fullsecondary data
            3read imaging foldes
            4build mouse objects
            5get mouse objects in primary data
            6select mouse data to trasnfer
            7trasnfer mouse data to primary
            8find new imagingsession
            9 process imaging sessin to ful data
            """
    
            module_logger.info('Building Prairire Imaging Sessions')
    
            self.all_existing_sessions={}
            self.read_all_imaging_sessions_from_directories() 
            # this checks all sessions in the F drive
            self.all_existing_sessions_database=[]
            self.read_all_imaging_sessions_from_database()
            # this checks all sessions in the the database dont build prairire imagibg sessions
            self.all_existing_sessions_database_objects={}
            self.build_all_paririe_session_from_database()
            self.all_existing_unprocessed_sessions=[]
            self.all_existing_unprocessed_session2={}

            self.read_all_imaging_sessions_not_in_database()
            self.all_existing_sessions_not_database_objects={}
            self.build_all_prairie_sessions_not_in_database()
            
            # this builds all prairie imaging sessions based on the database list
            module_logger.info('Building Mouse Objects')
            self.all_non_imaged_mice_objects={}
            self.all_imaged_mice_objects={}
            self.build_all_mice_objects_from_database()
            self.find_imaged_mouse_codes_not_in_database()
            self.build_all_unimaged_mice_objects_from_database()
            self.all_experimetal_mice_objects={**self.all_imaged_mice_objects, **self.all_non_imaged_mice_objects}
            # self.get_all_deep_caiman_objects()
            
            
            module_logger.info('Reading directory structure')
            self.read_all_data_path_structures()
            self.update_pre_process_slow_data_structure() # this adds new mouse folders to K(altern F) after new experimental mice are added
            
                    
            module_logger.info('Data managing done')
            print('Data managing done')
        else:
           self.read_dropbox_data_dir()

           pass

        
        
#%% building mouse objects and prairire imaging sessions        
    def build_all_mice_objects_from_database(self):
    
        query_all_imaged_mice="""
            SELECT  c.Code,         
                    Sex_types AS Sex,
                    Line_short,
                    g.Projects,
                    
                    b.ImagingDate,
                    b.Objectives,
                    b.EndOfSessionSummary,  
                    
                    date(f.DOB) AS DOB,
                    date(c.Injection1Date) AS InjectionDate,
                    date(c.WindowDate) AS WindowDate,
                                     
                    a.EyesComments,
                    a.BehaviourComments,
                    a.FurComments,
                    a.LessionComments,
                    d.DamagedAreas,
                    d.Notes AS WindowNotes,
                    
                    e.Notes AS InjectionNotes,
                    Combination,
                    e.InjectionSite1volume,                                      
                    k.Sensors AS Sensors1,
                    e.DilutionSensor1,
                    l.Optos AS Optos1,
                    m.Promoters AS Promoters1,
                    n.Recombinases AS Recombinases1,
                    o.Sensors AS Sensors2,
                    e.DilutionSensor2,
                    p.Promoters AS Promoters2,
                    q.Recombinases AS Recombinases2,
                    r.Optos AS Optos3,
                    e.DilutionOpto,
                    s.Promoters AS Promoters3,
                    t.Recombinases AS Recombinases3,
                    
                    G2C_table.Genotypes_types AS G2C,
                    Ai14_table.Genotypes_types AS Ai14,
                    Ai75_table.Genotypes_types AS Ai75,
                    VRC_table.Genotypes_types AS VRC,
                    SLF_table.Genotypes_types AS SLF,
                    PVF_table.Genotypes_types AS PVF,
                    Ai65_table.Genotypes_types AS Ai65,
                    Ai80_table.Genotypes_types AS Ai80,
                    VGC_table.Genotypes_types AS VGC,
                    Ai162_table.Genotypes_types AS Ai162,
                    Ai148_table.Genotypes_types AS Ai148 ,    
                    c.SlowStoragePath, 
                    c.WorkingStoragePath
                    
            
            FROM ImagedMice_table  a
            LEFT JOIN ExperimentalAnimals_table  c ON  c.ID=a.ExpID
            LEFT JOIN ImagingSessions_table b ON a.SessionID=b.ID
            LEFT JOIN Windows_table d ON d.ID=c.WindowID
            LEFT JOIN Injections_table e ON e.ID=c.Injection1ID
            LEFT JOIN MICE_table f ON f.ID=c.Mouse_ID
            LEFT JOIN Projects_table g ON g.ID=c.Project 
            LEFT JOIN VirusCombinations_table  ON VirusCombinations_table.ID=e.VirusCombination         
            LEFT JOIN Virus_table h ON h.ID=VirusCombinations_table.Virus1
            LEFT JOIN Virus_table i ON i.ID=VirusCombinations_table.Virus2
            LEFT JOIN Virus_table j ON j.ID=VirusCombinations_table.Virus3
            LEFT JOIN Sensors_table k ON k.ID=h.Sensor
            LEFT JOIN Optos_table l ON l.ID=h.Opto
            LEFT JOIN Promoter_table m ON m.ID=h.Promoter
            LEFT JOIN Recombinase_table n ON n.ID=h.Recombinase
            LEFT JOIN Sensors_table o ON o.ID=i.Sensor
            LEFT JOIN Promoter_table p ON p.ID=i.Promoter
            LEFT JOIN Recombinase_table q ON q.ID=i.Recombinase
            LEFT JOIN Optos_table r ON r.ID=j.Opto
            LEFT JOIN Promoter_table s ON s.ID=j.Promoter
            LEFT JOIN Recombinase_table t ON t.ID=j.Recombinase
            LEFT JOIN Genotypes_table AS G2C_table   ON f.G2C   = G2C_table.ID
            LEFT JOIN Genotypes_table AS Ai14_table   ON f.Ai14   = Ai14_table.ID
            LEFT JOIN Genotypes_table AS Ai75_table   ON f.Ai75   = Ai75_table.ID
            LEFT JOIN Genotypes_table AS VRC_table   ON f.VRC   = VRC_table.ID
            LEFT JOIN Genotypes_table AS SLF_table   ON f.SLF   = SLF_table.ID
            LEFT JOIN Genotypes_table AS PVF_table   ON f.PVF   = PVF_table.ID
            LEFT JOIN Genotypes_table AS Ai65_table   ON f.Ai65   = Ai65_table.ID
            LEFT JOIN Genotypes_table AS Ai80_table   ON f.Ai80   = Ai80_table.ID
            LEFT JOIN Genotypes_table AS VGC_table   ON f.VGC  = VGC_table.ID
            LEFT JOIN Genotypes_table AS Ai162_table   ON f.Ai162   = Ai162_table.ID
            LEFT JOIN Genotypes_table AS Ai148_table   ON f.Ai148  = Ai148_table.ID          
            LEFT JOIN Sex_table ON Sex_table.ID=f.Sex
            LEFT JOIN Lines_table v ON v.ID=f.Line 
            LEFT JOIN WideField_table z ON z.ID=a.WideFIeldID 
            
            """
        params=()    
        self.all_imaged_mice=self.Database_ref.arbitrary_query_to_df(query_all_imaged_mice, params)        
        age=pd.to_datetime(self.all_imaged_mice.ImagingDate) - pd.to_datetime(self.all_imaged_mice.DOB)
        wai=pd.to_datetime(self.all_imaged_mice.ImagingDate) - pd.to_datetime(self.all_imaged_mice.InjectionDate)
        waw=pd.to_datetime(self.all_imaged_mice.ImagingDate) - pd.to_datetime(self.all_imaged_mice.WindowDate)
        age=age.dt.days/7
        wai=wai.dt.days/7
        waw=waw.dt.days/7
        self.all_imaged_mice.insert(8, 'Age',  age.round(1))
        self.all_imaged_mice.insert(10, 'WAI',   wai.round(1))
        self.all_imaged_mice.insert(12, 'WAW',   waw.round(1))       
        self.all_imaged_mice.Combination.fillna(value='NoInjection', inplace=True)
        self.all_imaged_mice.Sensors1.fillna(value='NoInjection', inplace=True)
        self.all_imaged_mice.Optos1.fillna(value='NoInjection', inplace=True)
        self.all_imaged_mice.Sensors2.fillna(value='NoInjection', inplace=True)
        self.all_imaged_mice.Recombinases1.fillna(value='NoInjection', inplace=True)
        self.all_imaged_mice.Recombinases2.fillna(value='NoInjection', inplace=True)
        self.all_imaged_mice.Recombinases3.fillna(value='NoInjection', inplace=True)
        self.all_imaged_mice.Optos3.fillna(value='NoInjection', inplace=True) 
        self.all_imaged_mouse_dbinfo=self.all_imaged_mice.copy(0) 
        self.all_imaged_mouse_dbinfo.drop_duplicates(subset ="Code",keep='first', inplace=True)  
        
        for idx, mouse in self.all_imaged_mouse_dbinfo.iterrows():
            self.all_imaged_mice_objects[mouse['Code']]=Mouse(mouse['Code'], 
                                                                   self.LabProjectObject, 
                                                                   data_managing_object=self,
                                                                   mouse_info=self.all_imaged_mice[self.all_imaged_mice['Code']==mouse['Code']])
                      
    def build_all_unimaged_mice_objects_from_database(self):
         query_all_experimental_mice="""
             SELECT  c.Code,         
                     Sex_types AS Sex,
                     Line_short,
                     g.Projects,
           
                     date(f.DOB) AS DOB,
                     date(c.Injection1Date) AS InjectionDate,
                     date(c.WindowDate) AS WindowDate,
                                      
                     d.DamagedAreas,
                     d.Notes AS WindowNotes,
                     
                     e.Notes AS InjectionNotes,
                     Combination,
                     e.InjectionSite1volume,                                      
                     k.Sensors AS Sensors1,
                     e.DilutionSensor1,
                     l.Optos AS Optos1,
                     m.Promoters AS Promoters1,
                     n.Recombinases AS Recombinases1,
                     o.Sensors AS Sensors2,
                     e.DilutionSensor2,
                     p.Promoters AS Promoters2,
                     q.Recombinases AS Recombinases2,
                     r.Optos AS Optos3,
                     e.DilutionOpto,
                     s.Promoters AS Promoters3,
                     t.Recombinases AS Recombinases3,
                     
                     G2C_table.Genotypes_types AS G2C,
                     Ai14_table.Genotypes_types AS Ai14,
                     Ai75_table.Genotypes_types AS Ai75,
                     VRC_table.Genotypes_types AS VRC,
                     SLF_table.Genotypes_types AS SLF,
                     PVF_table.Genotypes_types AS PVF,
                     Ai65_table.Genotypes_types AS Ai65,
                     Ai80_table.Genotypes_types AS Ai80,
                     VGC_table.Genotypes_types AS VGC,
                     Ai162_table.Genotypes_types AS Ai162,
                     Ai148_table.Genotypes_types AS Ai148 ,    
                     c.SlowStoragePath, 
                     c.WorkingStoragePath
                     
             
             FROM ExperimentalAnimals_table  c
             LEFT JOIN Windows_table d ON d.ID=c.WindowID
             LEFT JOIN Injections_table e ON e.ID=c.Injection1ID
             LEFT JOIN MICE_table f ON f.ID=c.Mouse_ID
             LEFT JOIN Projects_table g ON g.ID=c.Project 
             LEFT JOIN VirusCombinations_table  ON VirusCombinations_table.ID=e.VirusCombination         
             LEFT JOIN Virus_table h ON h.ID=VirusCombinations_table.Virus1
             LEFT JOIN Virus_table i ON i.ID=VirusCombinations_table.Virus2
             LEFT JOIN Virus_table j ON j.ID=VirusCombinations_table.Virus3
             LEFT JOIN Sensors_table k ON k.ID=h.Sensor
             LEFT JOIN Optos_table l ON l.ID=h.Opto
             LEFT JOIN Promoter_table m ON m.ID=h.Promoter
             LEFT JOIN Recombinase_table n ON n.ID=h.Recombinase
             LEFT JOIN Sensors_table o ON o.ID=i.Sensor
             LEFT JOIN Promoter_table p ON p.ID=i.Promoter
             LEFT JOIN Recombinase_table q ON q.ID=i.Recombinase
             LEFT JOIN Optos_table r ON r.ID=j.Opto
             LEFT JOIN Promoter_table s ON s.ID=j.Promoter
             LEFT JOIN Recombinase_table t ON t.ID=j.Recombinase
             LEFT JOIN Genotypes_table AS G2C_table   ON f.G2C   = G2C_table.ID
             LEFT JOIN Genotypes_table AS Ai14_table   ON f.Ai14   = Ai14_table.ID
             LEFT JOIN Genotypes_table AS Ai75_table   ON f.Ai75   = Ai75_table.ID
             LEFT JOIN Genotypes_table AS VRC_table   ON f.VRC   = VRC_table.ID
             LEFT JOIN Genotypes_table AS SLF_table   ON f.SLF   = SLF_table.ID
             LEFT JOIN Genotypes_table AS PVF_table   ON f.PVF   = PVF_table.ID
             LEFT JOIN Genotypes_table AS Ai65_table   ON f.Ai65   = Ai65_table.ID
             LEFT JOIN Genotypes_table AS Ai80_table   ON f.Ai80   = Ai80_table.ID
             LEFT JOIN Genotypes_table AS VGC_table   ON f.VGC  = VGC_table.ID
             LEFT JOIN Genotypes_table AS Ai162_table   ON f.Ai162   = Ai162_table.ID
             LEFT JOIN Genotypes_table AS Ai148_table   ON f.Ai148  = Ai148_table.ID          
             LEFT JOIN Sex_table ON Sex_table.ID=f.Sex
             LEFT JOIN Lines_table v ON v.ID=f.Line 
             
             
             
             """
         params=()    
         self.all_experimental_mice=self.Database_ref.arbitrary_query_to_df(query_all_experimental_mice, params)        
         age=datetime.datetime.today() - pd.to_datetime(self.all_experimental_mice.DOB)
         wai=datetime.datetime.today() - pd.to_datetime(self.all_experimental_mice.InjectionDate)
         waw=datetime.datetime.today() - pd.to_datetime(self.all_experimental_mice.WindowDate)
         age=age.dt.days/7
         wai=wai.dt.days/7
         waw=waw.dt.days/7
         self.all_experimental_mice.insert(8, 'Age',  age.round(1))
         self.all_experimental_mice.insert(10, 'WAI',   wai.round(1))
         self.all_experimental_mice.insert(12, 'WAW',   waw.round(1))       
         self.all_experimental_mice.Combination.fillna(value='NoInjection', inplace=True)
         self.all_experimental_mice.Sensors1.fillna(value='NoInjection', inplace=True)
         self.all_experimental_mice.Optos1.fillna(value='NoInjection', inplace=True)
         self.all_experimental_mice.Sensors2.fillna(value='NoInjection', inplace=True)
         self.all_experimental_mice.Recombinases1.fillna(value='NoInjection', inplace=True)
         self.all_experimental_mice.Recombinases2.fillna(value='NoInjection', inplace=True)
         self.all_experimental_mice.Recombinases3.fillna(value='NoInjection', inplace=True)
         self.all_experimental_mice.Optos3.fillna(value='NoInjection', inplace=True) 
         self.all_exp_mouse_dbinfo=self.all_experimental_mice.copy(0) 
         self.all_exp_mouse_dbinfo.drop_duplicates(subset ="Code",keep='first', inplace=True)  
         
          
         for idx, mouse in self.all_exp_mouse_dbinfo.iterrows():
             if mouse['Code'] not in self.all_imaged_mice['Code'] and mouse['Code'] in self.new_mouse_codes:
                 self.all_non_imaged_mice_objects[mouse['Code']]=Mouse(mouse['Code'], 
                                                                      self.LabProjectObject, 
                                                                      data_managing_object=self,
                                                                      mouse_info=self.all_experimental_mice[self.all_experimental_mice['Code']==mouse['Code']])
                 
    def find_imaged_mouse_codes_not_in_database(self):
        
        self.non_database_imaged_mice=[]
        for value in  self.all_existing_sessions_not_database_objects.values():
            self.non_database_imaged_mice.append(value.session_imaged_mice_codes)    
            
        self.non_database_imaged_mice=list(set([y for x in self.non_database_imaged_mice for y in x ]))
        
        mouse_codes_inDb=list(self.all_imaged_mice_objects.keys())
        
        self.new_mouse_codes=[x for x in self.non_database_imaged_mice if x not in mouse_codes_inDb]
 
#%% prairrie session prcessing   
 #%% reading sessions                                      
    def read_all_imaging_sessions_from_directories(self):
        multidirwild=os.sep+'**'+os.sep+'**'
        longpathstr=os.sep+os.sep+os.sep+'?'+os.sep+os.sep
        
        
        self.all_existing_sessions={session[len(session)-8:]:session for session in glob.glob( self.data_paths_data['Raw']+multidirwild, recursive=False)}
        self.all_existing_sessions2={session[len(session)-8:]:session for session in glob.glob( self.data_paths_data['Raw2']+multidirwild, recursive=False)}
        self.all_existing_sessions.update(self.all_existing_sessions2)
 
    def read_all_imaging_sessions_from_database(self): 
        query_sessions="SELECT ID, ImagingDate, ImagingSessionRawPath FROM ImagingSessions_table"
        self.all_existing_sessions_database=self.Database_ref.arbitrary_query_to_df(query_sessions).values.tolist()
        
    def correct_linux_crazy_drives(self, path)   :
        
        if 'mnt' not in path:
            level=3
        else:
            level=2

        temp_path=self.nested_split(path,level)
        print(temp_path)
        corrected_path=[k for k in self.LabProjectObject.all_paths_for_this_system.keys() if temp_path in k]
        if not corrected_path:
           corrected_path=path
        else:
          corrected_path=corrected_path[0]+path[len(temp_path):]
        print(corrected_path)

 
        return corrected_path
     
    
    def nested_split(self,path, level):
        if path.count('/')==level:
            split_path=path
            return split_path
        else:
            split_path=os.path.split(path)[0]
            return  self.nested_split(split_path,level)

        
    
    def transform_databasepath_tolinux(self,windows_path):    
        longpathstr=os.sep+os.sep+os.sep+'?'+os.sep+os.sep
        if not windows_path:
            print('problem')
        if ':' in windows_path:

            if r'F:' in windows_path:
                drive=r'F:' 
                newstem=self.LabProjectObject.data_paths_roots['Raw']
                
            elif r'J:' in windows_path:
                drive=r'j:' 
                newstem=self.LabProjectObject.data_paths_roots['Raw']
    
            elif r'I:' in windows_path:
                drive=r'I:' 
                newstem=self.LabProjectObject.data_paths_roots['Raw2']
                  
            elif r'K:' in windows_path:
                drive=r'K:' 
                newstem=self.LabProjectObject.data_paths_roots['Pre_proccessed_slow_chandelier_tigres']
                
            elif r'C:' in windows_path:
                drive=r'C:' 
                newstem=self.LabProjectObject.data_paths_roots['Analysis_Fast_1']
            
            elif r'G:' in windows_path:
                drive=r'G:' 
                newstem=self.LabProjectObject.data_paths_roots['Analysis_Fast_2']
            
            elif r'D:' in windows_path:
                drive=r'D:' 
                newstem=self.LabProjectObject.data_paths_roots['Pre_proccessed_slow_interneurons_others']
                    
      
                
            linux_path=newstem+windows_path[windows_path.find(drive)+2:].replace('\\','/')
            
        else:
            linux_path=self.correct_linux_crazy_drives(windows_path)
            
        return linux_path

        
    def read_all_imaging_sessions_not_in_database(self):     
        
        
        test=[self.transform_databasepath_tolinux(session[2] ) for session in self.all_existing_sessions_database]

        self.all_existing_unprocessed_sessions=[session for session in  self.all_existing_sessions.values() if session not in test]
        # this ignores jesus and hakim sessions any folder with name sin it
        thresholdforsession=datetime.datetime.strptime('20220630','%Y%m%d') #arbitrary dynamic

        # thresholdforsession=datetime.datetime.strptime( os.path.split(test[-1])[1],'%Y%m%d')# latest session ind atabase
        test2=[value for key, value in self.all_existing_sessions.items() if key.isdigit() and datetime.datetime.strptime(key,'%Y%m%d')>thresholdforsession]
        self.all_new_unprocessed_session={session_name:session for session_name, session in  self.all_existing_sessions.items() if (session in test2) and (session not in test)}
        self.all_database_session={session_name:session for session_name, session in  self.all_existing_sessions.items() if session in test}



#%% building sessions
    def build_all_paririe_session_from_database(self):
        for i in  self.all_existing_sessions_database:    
            self.build_paririe_session_from_database(i[0])
    
    def build_all_prairie_sessions_not_in_database(self):
          for  session_name, session in self.all_new_unprocessed_session.items():
              self.build_prairie_session_not_in_database(session_name, session) 
    
    
    def build_paririe_session_from_database(self, ID):
          query_session_name="SELECT ImagingDate FROM ImagingSessions_table WHERE ID=?"
          params=(ID,)
          session_name=self.Database_ref.arbitrary_query_to_df(query_session_name, params).values.tolist()[0][0]    
          session_name=session_name.replace('-', '')
          self.all_existing_sessions_database_objects[session_name]=PrairieImagingSession(self.Database_ref.ImagingDatabase_class, session_ID=ID, data_managing_object=self)

    def build_prairie_session_not_in_database(self, session_name, session_path):
          self.all_existing_sessions_not_database_objects[session_name]=PrairieImagingSession(self.Database_ref.ImagingDatabase_class, session_raw_path=session_path, data_managing_object=self)    
        

#%% reading data_path_structures

    def read_all_data_path_structures(self):
        self.mouse_data_structure_paths={name: [file for file in  glob.glob(self.data_paths_data[name]+os.sep+'**'+os.sep+'SP**', recursive=True) if len(file)<120] 
                                         for i , name in enumerate(self.LabProjectObject.data_paths_names) if 'Raw' not in name}
        
        self.mouse_data_structure_paths_mouse_codes={name: [i[-4:] for i in self.mouse_data_structure_paths[name]] 
                                                     for i , name in enumerate(self.mouse_data_structure_paths.keys()) }
        
        
        self.mouse_data_structure_projects_mouse_codes={name: {  self.mouse_data_structure_paths_mouse_codes[name][i]   :j[j.find('Mice_Projects'+os.sep)+15:j.find('\\Mice_Projects'+os.sep)+41][:j[j.find('\\Mice_Projects'+os.sep)+15:j.find('\\Mice_Projects'+os.sep)+41].find(''+os.sep)] for i, j in enumerate(self.mouse_data_structure_paths[name])} for  name in self.mouse_data_structure_paths.keys()}
        
        self.mouse_data_structure_paths={name:{i[-4:]: i for i in self.mouse_data_structure_paths[name] }
            for  name in self.mouse_data_structure_paths.keys()}
        
        # self.delete_pre_procesed_strucutre_mouse_without_data()
        
    def delete_pre_procesed_strucutre_mouse_without_data(self):
        for key in self.mouse_data_structure_paths.keys():
            if ('Pre_proccessed_slow_chandelier_tigres' in key) or ('Pre_proccessed_slow_interneurons_others' in key) :
                
                mouse_to_delete=[ mouse for mouse in self.mouse_data_structure_paths[key].keys() if mouse not in  list(self.all_imaged_mice_objects.keys())]
                for mouse in mouse_to_delete:
                    if os.path.isdir(self.mouse_data_structure_paths[key][mouse]):
                       recursively_delete_back_directories(self.mouse_data_structure_paths[key][mouse])
#%% creating data_path_structures        
        
        
    def update_pre_process_slow_data_structure(self, update=False):
        # module_logger.info('updating experimnetal folders')
        query_new_codes="""
                SELECT a.Code
                FROM ExperimentalAnimals_table a   
                WHERE Project!=6 AND SlowStoragePath IS NULL
                """
                 
        mice_codes=self.LabProjectObject.database.arbitrary_query_to_df(query_new_codes).values.tolist()
        mice_codes=[mouse[0] for mouse in mice_codes]
        self.new_working_directories=self.create_pre_process_slow_data_structure(mice_codes)
        
        if update:
            query_all_codes="""
                    SELECT a.Code, a.SlowStoragePath, a.WorkingStoragePath
                    FROM ExperimentalAnimals_table a   
                    WHERE Project!=6 
                    """
            params=()
            all_mouse_info=self.LabProjectObject.database.arbitrary_query_to_df(query_all_codes,params).values.tolist()
            self.all_working_directories={mouse[0]:[mouse[2], mouse[1]] for mouse in all_mouse_info}
            mice_codes=list(self.all_working_directories.keys())
            self.new_working_directories=self.create_pre_process_slow_data_structure( mice_codes, update=update)

        self.delete_pre_procesed_strucutre_mouse_without_data()      
        
    def create_pre_process_slow_data_structure(self, mice_codes, update=None):
        # module_logger.info('creating new experimnetal folders')
        query_projects="""
               SELECT a.Project, a.Code, b.Projects, c.Line, d.Line_Short
               FROM ExperimentalAnimals_table a
               LEFT JOIN Projects_table b ON b.ID=a.Project
               LEFT JOIN MICE_table c ON c.ID=a.Mouse_ID
               LEFT JOIN Lines_table d ON D.ID=c.Line
               WHERE a.Code IN (%s)""" % ','.join('?' for i in mice_codes) 
        params=tuple(mice_codes)
        if update:
            params=()
            query_projects="""
               SELECT a.Project, a.Code, b.Projects, c.Line, d.Line_Short
               FROM ExperimentalAnimals_table a
               LEFT JOIN Projects_table b ON b.ID=a.Project
               LEFT JOIN MICE_table c ON c.ID=a.Mouse_ID
               LEFT JOIN Lines_table d ON D.ID=c.Line
               WHERE Project!=6
               """ 
  
        mice_codes_and_projects=self.LabProjectObject.database.arbitrary_query_to_df(query_projects,params)
        mice_codes_and_projects=mice_codes_and_projects.dropna()
        all_mice_paths={}
        query_mice_paths_update="""
                UPDATE ExperimentalAnimals_table
                SET SlowStoragePath=?, WorkingStoragePath=?
                WHERE Code=?
                """  
        for i, code in enumerate(mice_codes_and_projects.Code.values.tolist()):
            mouse_info=mice_codes_and_projects[mice_codes_and_projects['Code']==code]
            project=mouse_info.Projects.iloc[0]
            line=mouse_info.Line_Short.iloc[0]
            
            if 'Interneuron' in project:
                fast_disk='Analysis_Fast_2'
                slowdisk='Pre_proccessed_slow_interneurons_others'
            if 'Chandelier' in project:
                fast_disk='Analysis_Fast_1'
                slowdisk='Pre_proccessed_slow_chandelier_tigres'

            if 'Tigre' in project   : 
                fast_disk='Analysis_Fast_2'
                slowdisk='Pre_proccessed_slow_chandelier_tigres'

            if 'TEST' in project   : 
                fast_disk='Analysis_Fast_2'
                slowdisk='Pre_proccessed_slow_interneurons_others'

            if 'Collaborations' in project   : 
                fast_disk='Analysis_Fast_1'   
                slowdisk='Pre_proccessed_slow_interneurons_others'

            
            # os.makedirs(os.path.join(self.secondary_data_path,projects[i],line[i],code))
            if '::' in line:
                indexes = [j for j in range(len(line)) if line.startswith('::', j)]
                if len(indexes)==1:
                    firstline=line[:indexes[0]]
                    secondline=line[(indexes[0]+2):]   
                    workingdir=os.path.join(self.data_paths_data[fast_disk] ,project,firstline,secondline,code)
                    slowdir=os.path.join(self.data_paths_data[slowdisk] ,project,firstline,secondline,code)
                    all_mice_paths[code]=[workingdir, slowdir]

                    if not os.path.isdir(slowdir):
                        os.makedirs(slowdir)
                elif len(indexes)==2:
                    firstline=line[:indexes[0]]
                    secondline=line[(indexes[0]+2):indexes[1]]
                    thirdline=line[(indexes[1]+2):]
                    workingdir=os.path.join(self.data_paths_data[fast_disk] ,project,firstline,secondline,thirdline,code)
                    slowdir=os.path.join(self.data_paths_data[slowdisk] ,project,firstline,secondline,thirdline,code)
                    all_mice_paths[code]=[workingdir, slowdir]
                    
                    if not os.path.isdir(slowdir):
                        os.makedirs(slowdir)
            else:
                workingdir=os.path.join(self.data_paths_data[fast_disk] ,project,line,code)
                slowdir=os.path.join(self.data_paths_data[slowdisk] ,project,line,code)
                all_mice_paths[code]=[workingdir, slowdir]
                
                if not os.path.isdir(slowdir):
                        os.makedirs(slowdir)
  
            params=(all_mice_paths[code][1],all_mice_paths[code][0], code)   
            self.LabProjectObject.database.arbitrary_updating_record(query_mice_paths_update, params, commit=True)         
              
        return all_mice_paths
    #%% others
    def copy_all_mouse_with_data_to_working_path(self, mice_codes):
               
        query_all_codes="""
            SELECT a.Code,a.ID, a.SlowStoragePath, a.WorkingStoragePath
            FROM ExperimentalAnimals_table a   
            WHERE Project!=6  AND a.Code IN (%s)""" % ','.join('?' for i in mice_codes) 
            
        params=(tuple(mice_codes))
        all_mouse_info=self.LabProjectObject.database.arbitrary_query_to_df(query_all_codes,params).values.tolist()

        for mouse in all_mouse_info:

            slow_path='\\\\?'+os.sep+ mouse[2]
            fast_path='\\\\?'+os.sep+ mouse[3]
            
            recursively_copy_changed_files_and_directories_from_slow_to_fast(slow_path, fast_path)
            recursively_eliminate_empty_folders(fast_path)
            self.read_all_data_path_structures()    
      
    def remove_mouse_info_from_fast_disk(self,mouse):
        pass
 
    
#%% reading and updating all database imaging stroage paths  


    def update_mouse_slow_storages(self):
        
        query_all_codes="""
                SELECT a.Code,a.ID, a.SlowStoragePath, a.WorkingStoragePath
                FROM ExperimentalAnimals_table a   
                WHERE Project!=6 
                """
        params=()
        all_mouse_info=self.LabProjectObject.database.arbitrary_query_to_df(query_all_codes,params).values.tolist()
        newdrive='D'
        
        for mouse in all_mouse_info:
            mouse_expid=mouse[1]
            mouse_SlowStoragePath=mouse[2]
            mouse_WorkingStoragePath=mouse[3]

            if ('Interneuron_' in mouse_SlowStoragePath) or ('Collaborations_' in mouse_SlowStoragePath):
                newslowstorage=newdrive+mouse_SlowStoragePath[1:]
                query_mouse_slow_storage_update="""
                      UPDATE ExperimentalAnimals_table
                      SET SlowStoragePath=?
                      WHERE ID=?
                      """  
                params=(newslowstorage, mouse_expid)
                self.LabProjectObject.database.arbitrary_updating_record(query_mouse_slow_storage_update, params, commit=True)    
                
        self.update_all_imaging_data_paths()
        
        
    def create_linux_mouse_path(self, windows_path):
        
        if ':' in windows_path:

            if r'F:' in windows_path:
                drive=r'F:' 
                newstem=self.LabProjectObject.data_paths_roots['Raw']
                
            elif r'J:' in windows_path:
                drive=r'j:' 
                newstem=self.LabProjectObject.data_paths_roots['Raw']
    
            elif r'I:' in windows_path:
                drive=r'I:' 
                newstem=self.LabProjectObject.data_paths_roots['Raw2']
                  
            elif r'K:' in windows_path:
                drive=r'K:' 
                newstem=self.LabProjectObject.data_paths_roots['Pre_proccessed_slow_chandelier_tigres']
                
            elif r'C:' in windows_path:
                drive=r'C:' 
                newstem=self.LabProjectObject.data_paths_roots['Analysis_Fast_1']
            
            elif r'G:' in windows_path:
                drive=r'G:' 
                newstem=self.LabProjectObject.data_paths_roots['Analysis_Fast_2']
            
            elif r'D:' in windows_path:
                drive=r'D:' 
                newstem=self.LabProjectObject.data_paths_roots['Pre_proccessed_slow_interneurons_others']
                
            linux_path=newstem+windows_path[windows_path.find(drive)+2:].replace('\\','/')
        else:
            linux_path=windows_path
            
            return linux_path

    def update_all_imaging_data_paths(self) :
        print('Correcting database paths for projects')
        query_all_codes="""
                SELECT a.Code,a.ID, a.SlowStoragePath, a.WorkingStoragePath
                FROM ExperimentalAnimals_table a   
                WHERE Project!=6 
                """
        params=()
        all_mouse_info=self.LabProjectObject.database.arbitrary_query_to_df(query_all_codes,params).values.tolist()
        
        for mouse in all_mouse_info:

            if mouse[0]=='SPQZ':
                print("updating mouse paths" + mouse[0])
                print(mouse[0])
                mouse_expid=mouse[1]
                mouse_SlowStoragePath=mouse[2]
                mouse_WorkingStoragePath=mouse[3]
    #%%
                query_all_imaged_sessions="""
                        SELECT a.ID, a.SlowStoragePath, a.WorkingStoragePath, a.IsSlowStorage, a.IsWorkingStorage, a.MouseRawPath
                        FROM ImagedMice_table a   
                        WHERE ExpID=?
                        """
                params=(mouse_expid,)
                imaged_mice_info=self.LabProjectObject.database.arbitrary_query_to_df(query_all_imaged_sessions,params).values.tolist()
                
                if imaged_mice_info:
                    for imaged_mouse in imaged_mice_info:
                        sessiondate=imaged_mouse[-1][imaged_mouse[-1].find('Mice')-9:imaged_mouse[-1].find('Mice')-1]
                        imaged_mouse_relative_path='imaging'+os.sep+sessiondate
                        new_slow_imaged_mice_path= os.path.join(mouse_SlowStoragePath , imaged_mouse_relative_path)
                        new_fast_imaged_mice_path= os.path.join(mouse_WorkingStoragePath , imaged_mouse_relative_path) 
                        isSlowStorage=os.path.isdir(new_slow_imaged_mice_path)
                        isWorkingStorage=os.path.isdir(new_fast_imaged_mice_path)
              
                        query_imaged_mice_paths_update="""
                              UPDATE ImagedMice_table
                              SET SlowStoragePath=?, WorkingStoragePath=?, IsSlowStorage=?, IsWorkingStorage=?
                              WHERE ID=?
                              """  
                        params=(new_slow_imaged_mice_path,new_fast_imaged_mice_path, isSlowStorage, isWorkingStorage, imaged_mouse[0])
                        self.LabProjectObject.database.arbitrary_updating_record(query_imaged_mice_paths_update, params, commit=True)         
        #%%
                        query_all_widefields="""
                            SELECT a.ID, a.SlowStoragePath, a.WorkingStoragePath, a.IsSlowPath, a.IsWorkingStorage, a.WideFieldFileName
                            FROM WideField_table a   
                            WHERE ImagedMouseID=?
                            """
                        params=(imaged_mouse[0],)
                        all_widefields_info=self.LabProjectObject.database.arbitrary_query_to_df(query_all_widefields,params).values.tolist()
                        if all_widefields_info:
                            for widefield in all_widefields_info:                                
                                widefield_relative_path='widefield image'+os.sep + widefield[-1]                               
                                new_slow_widefield_path= os.path.join(new_slow_imaged_mice_path , widefield_relative_path)
                                new_fast_widefield_path= os.path.join(new_fast_imaged_mice_path , widefield_relative_path)
                                isSlowStorage=os.path.isfile(new_slow_widefield_path)
                                isWorkingStorage=os.path.isfile(new_fast_widefield_path)  
                          
                                query_widefield_paths_update="""
                                      UPDATE WideField_table
                                        SET SlowStoragePath=?, WorkingStoragePath=?, IsSlowPath=?, IsWorkingStorage=?
                                        WHERE ID=?
                                    """  
                                          
                                params=(new_slow_widefield_path,new_fast_widefield_path, isSlowStorage, isWorkingStorage, widefield[0])
                                self.LabProjectObject.database.arbitrary_updating_record(query_widefield_paths_update, params, commit=True)
          
         #%%   

                        query_all_acquistions="""
                                SELECT a.ID, a.SlowDiskPath, a.WorkingDiskPath, a.IsSlowDisk, a.IsWorkingDisk, a.AcquisitonRawPath, a.IsTestAcquisition, a.IsNonImagingAcquisition, a.Is0CoordinateAcquisiton, a.IsFOVAcquisition,a.IsSurfaceImage, a.IsImaging
                                FROM Acquisitions_table a   
                                WHERE ImagedMouseID=?
                                """
                        params=(imaged_mouse[0],)
                        all_acquisitons_info=self.LabProjectObject.database.arbitrary_query_to_df(query_all_acquistions,params).values.tolist()
      
                        if all_acquisitons_info:
                            for aqc in all_acquisitons_info:
                                query_imaging_name="""
                                SELECT a.ID, a.ImagingFilename, a.RedFilter
                                FROM Imaging_table a   
                                WHERE AcquisitionID=?
                                """
                                params=(aqc[0],)
                                imaging_name=self.LabProjectObject.database.arbitrary_query_to_df(query_imaging_name,params).values.tolist()
                                if imaging_name:
                                    imaging_name=imaging_name[0]
                                
                                
                                
                                    if not aqc[7]:
                                        if aqc[8]:
                                            acquisiton_relative_path= '0Coordinate acquisition'+os.sep + imaging_name[1]
                                        elif aqc[6]:
                                            acquisiton_relative_path= 'test aquisitions'+os.sep + imaging_name[1]
                                        elif aqc[9]:
                                            fov_relative_path= 'data aquisitions\\{}'+os.sep.format(aqc[5][aqc[5].find('FOV_'):aqc[5].find('FOV_')+5])
                                            
                                            if aqc[10]:
                                                acquisiton_relative_path= fov_relative_path+ 'SurfaceImage'+os.sep + imaging_name[1]
                                            elif '1050_Tomato' in aqc[5]:
                                                acquisiton_relative_path= fov_relative_path+'1050_Tomato'+os.sep + imaging_name[1]
                                            elif '1050_3PlaneTomato' in aqc[5]:
                                                acquisiton_relative_path= fov_relative_path+'1050_3PlaneTomato'+os.sep + imaging_name[1]
                                            elif '1050_HighResStackTomato' in aqc[5]:
                                                acquisiton_relative_path=fov_relative_path+ '1050_HighResStackTomato'+os.sep + imaging_name[1]
                                            elif 'HighResStackGreen' in aqc[5]:
                                                acquisiton_relative_path=fov_relative_path+ 'HighResStackGreen'+os.sep + imaging_name[1]
                                            elif 'OtherAcq' in aqc[5]:
                                                acquisiton_relative_path=fov_relative_path+ 'OtherAcq'+os.sep + imaging_name[1]
                                            else:
                                                acquisiton_relative_path= fov_relative_path + imaging_name[1]
    
                                    elif imaging_name[1]:
                                        acquisiton_relative_path='nonimaging acquisitions'+os.sep+ imaging_name[1]

                                elif not imaging_name :
                                    acquisiton_relative_path='nonimaging acquisitions\\Aq_1_NonImaging'    
                                    
                                new_slow_acquistion_path= os.path.join(new_slow_imaged_mice_path , acquisiton_relative_path)
                                new_fast_acquisition_path= os.path.join(new_fast_imaged_mice_path , acquisiton_relative_path)
                                isSlowStorage=os.path.isdir(new_slow_acquistion_path)
                                isWorkingStorage=os.path.isdir(new_fast_acquisition_path) 
                                
                                query_acquistions_paths_update="""
                                  UPDATE Acquisitions_table
                                  SET SlowDiskPath=?, WorkingDiskPath=?, IsSlowDisk=?, IsWorkingDisk=?
                                  WHERE ID=?
                                  """  
                                params=(new_slow_acquistion_path,new_fast_acquisition_path, isSlowStorage, isWorkingStorage, aqc[0])
                                self.LabProjectObject.database.arbitrary_updating_record(query_acquistions_paths_update, params, commit=True)
             #%% 
                                query_all_imaging="""
                                    SELECT a.ID, a.SlowStoragePath, a.WorkingStoragePath, a.IsSlowStorage, a.IsWorkingStorage
                                    FROM Imaging_table a   
                                    WHERE AcquisitionID=?
                                    """
                                params=(aqc[0],)
                                all_imagings_info=self.LabProjectObject.database.arbitrary_query_to_df(query_all_imaging,params).values.tolist()
                                
                                
                        
                                if all_imagings_info:
                                    for imaging in all_imagings_info:
                                            
                                            new_slow_imaging_path= os.path.join(new_slow_acquistion_path, 'planes\Plane1')
                                            new_fast_imaging_path=  os.path.join(new_fast_acquisition_path, 'planes\Plane1')
                                            isSlowStorage=os.path.isdir(new_slow_imaging_path)
                                            isWorkingStorage=os.path.isdir(new_fast_imaging_path)  
                                  
                                            query_imaging_paths_update="""
                                                    UPDATE Imaging_table
                                                    SET SlowStoragePath=?, WorkingStoragePath=?, IsSlowStorage=?, IsWorkingStorage=?
                                                    WHERE ID=?
                                                    """  
                                            params=(new_slow_imaging_path,new_fast_imaging_path, isSlowStorage, isWorkingStorage, imaging[0])
                                            self.LabProjectObject.database.arbitrary_updating_record(query_imaging_paths_update, params, commit=True)            
                #%%
                                query_all_facecamera="""
                                    SELECT a.ID, a.SlowStoragePath, a.WorkingStoragePath, a.IsSlowStorage, a.IsWorkingStorage,a.VideoPath, a.EyeCameraFilename
                                    FROM FaceCamera_table a   
                                    WHERE AcquisitionID=?
                                    """
                                params=(aqc[0],)
                                all_facecameras_info=self.LabProjectObject.database.arbitrary_query_to_df(query_all_facecamera,params).values.tolist()
                                if all_facecameras_info:
                                    for face_camera in all_facecameras_info:
                                        
                                        if face_camera[1]:
                                            facecamera_relative_path=os.path.join('eye camera' , os.path.split(new_slow_acquistion_path)[1]+ '_full_face_camera.tiff')
     
                                        elif face_camera[-1]=='_1':
                                            facecamera_relative_path='eye camera\\Aq_1_NonImaging_full_face_camera.tiff'
                                        else :
                                            facecamera_relative_path='eye camera'+os.sep+ face_camera[-1]  +'_full_face_camera.tiff'
    
                                        new_slow_facecamera_path= os.path.join(new_slow_acquistion_path , facecamera_relative_path)
                                        new_fast_facecamera_path= os.path.join(new_fast_acquisition_path , facecamera_relative_path)
                                        isSlowStorage=os.path.isfile('\\\\?'+os.sep+new_slow_facecamera_path)
                                        isWorkingStorage=os.path.isfile('\\\\?'+os.sep+new_fast_facecamera_path) 
                                  
                                        query_facecamera_paths_update="""
                                             UPDATE FaceCamera_table
                                            SET SlowStoragePath=?, WorkingStoragePath=?, IsSlowStorage=?, IsWorkingStorage=?
                                            WHERE ID=?
                                            """  
                                                  
                                        params=(new_slow_facecamera_path,new_fast_facecamera_path, isSlowStorage, isWorkingStorage, face_camera[0])
                                        self.LabProjectObject.database.arbitrary_updating_record(query_facecamera_paths_update, params, commit=True)            
                          
                
             #%%   
                                query_all_visualstimulations="""
                                    SELECT a.ID, a.SlowStoragePath, a.WorkingStoragePath, a.IsSlowStorage, a.IsWorkingStorage, a.VisStimLogName
                                    FROM VisualStimulations_table a   
                                    WHERE AcquisitionID=?
                                    """
                                params=(aqc[0],)
                                all_visualstimulations_info=self.LabProjectObject.database.arbitrary_query_to_df(query_all_visualstimulations,params).values.tolist()
                                if all_visualstimulations_info:
                                    for visual_stim in all_visualstimulations_info:
                
                                        visual_stim_relative_path='visual stim'+os.sep + visual_stim[-1]                                       
                                        new_slow_visstim_path= os.path.join(new_slow_acquistion_path , visual_stim_relative_path)
                                        new_fast_visstim_path= os.path.join(new_fast_acquisition_path , visual_stim_relative_path)
                                        isSlowStorage=os.path.isfile('\\\\?'+os.sep+new_slow_visstim_path)
                                        isWorkingStorage=os.path.isfile('\\\\?'+os.sep+new_fast_visstim_path) 
                                  
                                        query_visualstimulations_paths_update="""
                                              UPDATE VisualStimulations_table
                                                SET SlowStoragePath=?, WorkingStoragePath=?, IsSlowStorage=?, IsWorkingStorage=?
                                                WHERE ID=?
                                                """   
                                                  
                                        params=(new_slow_visstim_path,new_fast_visstim_path, isSlowStorage, isWorkingStorage, visual_stim[0])
                                        self.LabProjectObject.database.arbitrary_updating_record(query_visualstimulations_paths_update, params, commit=True)          
                

#%% processing paririe imaging sessin raw folders 
    def load_raw_session(self, session_path):
        imaging_session=PrairieImagingSession(session_raw_path=session_path, data_managing_object=self)
        return imaging_session
#%% dep caiman
    def get_all_deep_caiman_objects(self):    
         self.all_deep_caiman_objects={}      
         for mouse_code in self.all_imaged_mice_objects.keys():
             mouse_object=self.all_experimetal_mice_objects[mouse_code]
             mouse_object.get_all_mouse_FOVdata_datasets()       
             greendataset={key:v for key,v in mouse_object.all_mouse_FOVdata_datasets.items() if 'Green' in key and 'Red' not in key}   
             tododeepgreensatasets={key:dataset for key,dataset in greendataset.items() if dataset.associated_aquisiton.metadata_object.imaging_metadata_database[0]['ToDoDeepCaiman']==1 and dataset.associated_aquisiton.metadata_object.imaging_metadata_database[0]['Is10MinRec']!=1}
             for name, dat in tododeepgreensatasets.items():
                 if len(dat.most_updated_caiman.all_caiman_full_paths)>1:
                    self.all_deep_caiman_objects[name]=dat.most_updated_caiman

    def do_deep_caiman_of_mice_datasets(self, mice_codes:list, fovonly=False, nonfov=False):
        for mouse_code in mice_codes:
            mouse_object=self.all_experimetal_mice_objects[mouse_code]
            mouse_object.get_all_mouse_FOVdata_datasets()       
            mouse_object.all_mouse_acquisitions_datasets
            greendataset={key:v for key,v in mouse_object.all_mouse_FOVdata_datasets.items() if 'Green' in key}   
            greennonfovdatasets={key:v for key,v in mouse_object.all_mouse_acquisitions_datasets.items() if 'Green' in key}   
            
            
            tododeepgreensatasets={key:dataset for key,dataset in greendataset.items() if (dataset.associated_aquisiton.metadata_object.imaging_metadata_database[0]['ToDoDeepCaiman']==1 or dataset.associated_aquisiton.metadata_object.imaging_metadata_database[0]['Is10MinRec']==1)}
            todononfovdeepgreensatasets={key:dataset for key,dataset in greennonfovdatasets.items() if (dataset.associated_aquisiton.metadata_object.imaging_metadata_database[0]['ToDoDeepCaiman']==1 or dataset.associated_aquisiton.metadata_object.imaging_metadata_database[0]['Is10MinRec']==1)}
            
            if fovonly:
                for dataset in tododeepgreensatasets.values():
                    if dataset.metadata.imaging_metadata_database[0]['ToDoDeepCaiman']:
                     dataset.do_deep_caiman()
            elif nonfov:
                for dataset in todononfovdeepgreensatasets.values():
                    if dataset.metadata.imaging_metadata_database[0]['ToDoDeepCaiman']:
                     dataset.do_deep_caiman()
                         
        self.get_all_deep_caiman_objects()
        
#%% cloud transfers        
    def read_dropbox_data_dir(self):
        pass
        
    def copy_full_data_to_dropbox(self, mouse_code):
        pass
    
    def copy_data_dir_to_dropbox(self, mouse_code):
        mouse_object=self.all_imaged_mice_objects[mouse_code]
        aqc_list=glob.glob(os.path.join(mouse_object.mouse_slow_subproject_path, 'data'+os.sep+'*'))
        for acq in aqc_list:
            dirpath=os.path.join(self.LabProjectObject.data_paths_project['ResultsContainers'],''+os.sep.join(acq.split(''+os.sep)[-6:]) )
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath)
            filestocopy=[i for i in glob.glob(acq+os.sep+'*') if os.path.isfile(i)]
            [shutil.copy(f,dirpath) for f in filestocopy]
            
        
    def copy_jesus_runs_to_dropbox(self, mouse_code):
        pass
    def copy_pca_to_dropbox(self, mouse_code):
        pass
    def copy_crf_to_dropbox(self, mouse_code):
        pass
    def copy_allen_to_dropbox(self, mouse_code):
        pass
        
        
        
        
    
        
        