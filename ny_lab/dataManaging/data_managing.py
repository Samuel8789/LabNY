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
logger = logging.getLogger(__name__)

from .functions.select_face_camera import select_face_camera
from .functions.functionsDataOrganization import check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories
from .classes.mouse import Mouse
from .classes.prairieImagingSession  import PrairieImagingSession


class DataManaging():
    
    def __init__(self, project_object):
        
        self.LabProjectObject=project_object
        self.Database_ref=self.LabProjectObject.database
        # self.update_all_imaging_data_paths()
        
        self.data_paths=['Imaging', r'Full_Mice_Pre_Processed_Data\Mice_Projects', r'Working_Mice_Data_1\Mice_Projects', r'Working_Mice_Data_2\Mice_Projects']
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

        print('Building Prairire Imaging Sessions')
        self.all_existing_sessions={}
        self.read_all_imaging_sessions_from_directories() 
        # this checks all sessions in the F drive
        self.all_existing_sessions_database=[]
        self.read_all_imaging_sessions_from_database()
        # this checks all sessions in the the database dont build prairire imagibg sessions
        self.all_existing_sessions_database_objects={}
        self.build_all_paririe_session_from_database()
        self.all_existing_unprocessed_sessions=[]
        self.read_all_immaging_session_not_in_database()
        # this builds all prairie imaging sessions based on the database list
        print('Building Mouse Objects')
        self.all_experimetal_mice_objects={}
        self.build_all_mice_objects_from_database()
        
        
        
        print('Reading directory structure')
        self.read_all_data_path_structures()
        
                
        self.update_pre_process_slow_data_structure() # this adds new mouse folders to K(altern F) after new experimental mice are added
        print('Damanagaing done')
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
        self.all_mouse_info=self.all_imaged_mice.copy(0) 
        self.all_mouse_info.drop_duplicates(subset ="Code",keep='first', inplace=True)  
        
        for idx, mouse in self.all_mouse_info.iterrows():
            self.all_experimetal_mice_objects[mouse['Code']]=Mouse(mouse['Code'], 
                                                                   self.LabProjectObject, 
                                                                   data_managing_object=self,
                                                                   mouse_info=self.all_imaged_mice[self.all_imaged_mice['Code']==mouse['Code']])
        
    def build_all_paririe_session_from_database(self):
        
        for i in  self.all_existing_sessions_database:    
            self.build_paririe_session_from_database(i[0])
    
    def build_paririe_session_from_database(self, ID):
          query_session_name="SELECT ImagingDate FROM ImagingSessions_table WHERE ID=?"
          params=(ID,)
          session_name=self.Database_ref.arbitrary_query_to_df(query_session_name, params).values.tolist()[0][0]    
          session_name=session_name.replace('-', '')
          self.all_existing_sessions_database_objects[session_name]=PrairieImagingSession(self.Database_ref.ImagingDatabase_class, session_ID=ID, data_managing_object=self)
        
    def read_all_imaging_sessions_from_directories(self):
        self.all_existing_sessions={session[len(session)-8:]:'\\\?\\'+session for session in glob.glob( self.data_paths_data['Raw']+'\\**\\**', recursive=False)}
        
    def read_all_immaging_session_not_in_database(self):     
        test=[session[2] for session in self.all_existing_sessions_database]
        self.all_existing_unprocessed_sessions=[session for session in  self.all_existing_sessions.values() if session not in  test]
    
        
    def read_all_imaging_sessions_from_database(self):
          
        query_sessions="SELECT ID, ImagingDate, ImagingSessionRawPath FROM ImagingSessions_table"
        self.all_existing_sessions_database=self.Database_ref.arbitrary_query_to_df(query_sessions).values.tolist()
#%% reading data_path _structures

    
    def read_all_data_path_structures(self):
        self.mouse_data_structure_paths={name: [file for file in  glob.glob(self.data_paths_data[name]+'\\**\\SP**', recursive=True) if len(file)<120] 
                                         for i , name in enumerate(self.LabProjectObject.data_paths_names) if 'Raw' not in name}
        
        self.mouse_data_structure_paths_mouse_codes={name: [i[-4:] for i in self.mouse_data_structure_paths[name]] 
                                                     for i , name in enumerate(self.mouse_data_structure_paths.keys()) }
        
        
        self.mouse_data_structure_projects_mouse_codes={name: {  self.mouse_data_structure_paths_mouse_codes[name][i]   :j[j.find('\\Mice_Projects\\')+15:j.find('\\Mice_Projects\\')+41][:j[j.find('\\Mice_Projects\\')+15:j.find('\\Mice_Projects\\')+41].find('\\')] for i, j in enumerate(self.mouse_data_structure_paths[name])} for  name in self.mouse_data_structure_paths.keys()}
        
        self.mouse_data_structure_paths={name:{i[-4:]: i for i in self.mouse_data_structure_paths[name] }
            for  name in self.mouse_data_structure_paths.keys()}
        
        # self.delete_pre_procesed_strucutre_mouse_without_data()
        
    def delete_pre_procesed_strucutre_mouse_without_data(self):
        mouse_to_delete=[ mouse for mouse in self.mouse_data_structure_paths['Pre_proccessed_slow'].keys() if mouse not in  list(self.all_experimetal_mice_objects.keys())]
        for mouse in mouse_to_delete:
            if os.path.isdir(self.mouse_data_structure_paths['Pre_proccessed_slow'][mouse]):
               recursively_delete_back_directories(self.mouse_data_structure_paths['Pre_proccessed_slow'][mouse])
#%% creating data_path_structures        
        
        
    def update_pre_process_slow_data_structure(self, update=False):
        # print('updating experimnetal folders')
        query_new_codes="""
                SELECT a.Code
                FROM ExperimentalAnimals_table a   
                WHERE Project!=6 AND SlowStoragePath IS NULL
                """
                 
        mice_codes=self.LabProjectObject.database.arbitrary_query_to_df(query_new_codes).values.tolist()
        mice_codes=[mouse[0] for mouse in mice_codes]
        self.new_working_directories=self.create_pre_process_slow_data_structure(self.data_paths_data['Pre_proccessed_slow'], mice_codes)
        
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
        # print('creating new experimnetal folders')
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
            if 'Chandelier' in project:
                fast_disk='Analysis_Fast_1'
            if 'Tigre' in project   : 
                fast_disk='Analysis_Fast_2'
            if 'TEST' in project   : 
                fast_disk='Analysis_Fast_2'
            if 'Collaborations' in project   : 
                fast_disk='Analysis_Fast_1'   
            
            # os.makedirs(os.path.join(self.secondary_data_path,projects[i],line[i],code))
            if '::' in line:
                indexes = [j for j in range(len(line)) if line.startswith('::', j)]
                if len(indexes)==1:
                    firstline=line[:indexes[0]]
                    secondline=line[(indexes[0]+2):]   
                    workingdir=os.path.join(self.data_paths_data[fast_disk] ,project,firstline,secondline,code)
                    slowdir=os.path.join(self.data_paths_data['Pre_proccessed_slow'] ,project,firstline,secondline,code)
                    all_mice_paths[code]=[workingdir, slowdir]

                    if not os.path.isdir(slowdir):
                        os.makedirs(slowdir)
                elif len(indexes)==2:
                    firstline=line[:indexes[0]]
                    secondline=line[(indexes[0]+2):indexes[1]]
                    thirdline=line[(indexes[1]+2):]
                    workingdir=os.path.join(self.data_paths_data[fast_disk] ,project,firstline,secondline,thirdline,code)
                    slowdir=os.path.join(self.data_paths_data['Pre_proccessed_slow'] ,project,firstline,secondline,thirdline,code)
                    all_mice_paths[code]=[workingdir, slowdir]
                    
                    if not os.path.isdir(slowdir):
                        os.makedirs(slowdir)
            else:
                workingdir=os.path.join(self.data_paths_data[fast_disk] ,project,line,code)
                slowdir=os.path.join(self.data_paths_data['Pre_proccessed_slow'] ,project,line,code)
                all_mice_paths[code]=[workingdir, slowdir]
                
                if not os.path.isdir(slowdir):
                        os.makedirs(slowdir)
  
            params=(all_mice_paths[code][1],all_mice_paths[code][0], code)   
            self.LabProjectObject.database.arbitrary_updating_record(query_mice_paths_update, params, commit=True)         
              
        return all_mice_paths
    
    def copy_all_mouse_with_data_to_working_path(self, mice_codes):
               
        query_all_codes="""
            SELECT a.Code,a.ID, a.SlowStoragePath, a.WorkingStoragePath
            FROM ExperimentalAnimals_table a   
            WHERE Project!=6  AND a.Code IN (%s)""" % ','.join('?' for i in mice_codes) 
            
        params=(tuple(mice_codes))
        all_mouse_info=self.LabProjectObject.database.arbitrary_query_to_df(query_all_codes,params).values.tolist()

        for mouse in all_mouse_info:

            slow_path='\\\\?\\'+ mouse[2]
            fast_path='\\\\?\\'+ mouse[3]
            
            recursively_copy_changed_files_and_directories_from_slow_to_fast(slow_path, fast_path)
            recursively_eliminate_empty_folders(fast_path)
            self.read_all_data_path_structures()    
      

 
    
#%% reading and updating all database imaging stroage paths  

    def update_all_imaging_data_paths(self) :
        query_all_codes="""
                SELECT a.Code,a.ID, a.SlowStoragePath, a.WorkingStoragePath
                FROM ExperimentalAnimals_table a   
                WHERE Project!=6 
                """
        params=()
        all_mouse_info=self.LabProjectObject.database.arbitrary_query_to_df(query_all_codes,params).values.tolist()
        
        for mouse in all_mouse_info:
            # if mouse[0]=='SPIH':
                # print("updating mouse paths" + mouse[0])
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
                        imaged_mouse_relative_path='imaging\\'+sessiondate
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
                                widefield_relative_path='widefield image\\' + widefield[-1]                               
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
                                            acquisiton_relative_path= '0Coordinate acquisition\\' + imaging_name[1]
                                        elif aqc[6]:
                                            acquisiton_relative_path= 'test aquisitions\\' + imaging_name[1]
                                        elif aqc[9]:
                                            fov_relative_path= 'data aquisitions\\{}\\'.format(aqc[5][aqc[5].find('FOV_'):aqc[5].find('FOV_')+5])
                                            
                                            if aqc[10]:
                                                acquisiton_relative_path= fov_relative_path+ 'SurfaceImage\\' + imaging_name[1]
                                            elif '1050_Tomato' in aqc[5]:
                                                acquisiton_relative_path= fov_relative_path+'1050_Tomato\\' + imaging_name[1]
                                            elif '1050_3PlaneTomato' in aqc[5]:
                                                acquisiton_relative_path= fov_relative_path+'1050_3PlaneTomato\\' + imaging_name[1]
                                            elif '1050_HighResStackTomato' in aqc[5]:
                                                acquisiton_relative_path=fov_relative_path+ '1050_HighResStackTomato\\' + imaging_name[1]
                                            elif 'HighResStackGreen' in aqc[5]:
                                                acquisiton_relative_path=fov_relative_path+ 'HighResStackGreen\\' + imaging_name[1]
                                            elif 'OtherAcq' in aqc[5]:
                                                acquisiton_relative_path=fov_relative_path+ 'OtherAcq\\' + imaging_name[1]
                                            else:
                                                acquisiton_relative_path= fov_relative_path + imaging_name[1]
    
                                    elif imaging_name[1]:
                                        acquisiton_relative_path='nonimaging acquisitions\\'+ imaging_name[1]

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
                                            facecamera_relative_path='eye camera\\'+ face_camera[-1]  +'_full_face_camera.tiff'
    
                                        new_slow_facecamera_path= os.path.join(new_slow_acquistion_path , facecamera_relative_path)
                                        new_fast_facecamera_path= os.path.join(new_fast_acquisition_path , facecamera_relative_path)
                                        isSlowStorage=os.path.isfile('\\\\?\\'+new_slow_facecamera_path)
                                        isWorkingStorage=os.path.isfile('\\\\?\\'+new_fast_facecamera_path) 
                                  
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
                
                                        visual_stim_relative_path='visual stim\\' + visual_stim[-1]                                       
                                        new_slow_visstim_path= os.path.join(new_slow_acquistion_path , visual_stim_relative_path)
                                        new_fast_visstim_path= os.path.join(new_fast_acquisition_path , visual_stim_relative_path)
                                        isSlowStorage=os.path.isfile('\\\\?\\'+new_slow_visstim_path)
                                        isWorkingStorage=os.path.isfile('\\\\?\\'+new_fast_visstim_path) 
                                  
                                        query_visualstimulations_paths_update="""
                                              UPDATE VisualStimulations_table
                                                SET SlowStoragePath=?, WorkingStoragePath=?, IsSlowStorage=?, IsWorkingStorage=?
                                                WHERE ID=?
                                                """   
                                                  
                                        params=(new_slow_visstim_path,new_fast_visstim_path, isSlowStorage, isWorkingStorage, visual_stim[0])
                                        self.LabProjectObject.database.arbitrary_updating_record(query_visualstimulations_paths_update, params, commit=True)          
                

#%% processing paririe imaging sessin raw folders 
    def cleaning_up_raw_acquisitions(self, session_path):
            
        mice_directory_path=os.path.join(session_path,'Mice')
        mice_paths=glob.glob( mice_directory_path+'\\SP**', recursive=False)
        # mice_codes=[os.path.split(mouse_path)[1] for mouse_path in mice_paths]
        
        
        for mouse_path in mice_paths:
            Coordinate0path=os.path.join(mouse_path,'0CoordinateAcquisiton')   
            nonimagingacquisitionspath=os.path.join(mouse_path,'NonImagingAcquisitions')  
            Testacquisitionspath=os.path.join(mouse_path,'TestAcquisitions')  
            
            allAq_folders=[Coordinate0path, nonimagingacquisitionspath, Testacquisitionspath]
            
            self.process_aquisition_folder(allAq_folders, mouse_path)
            emptyatlases=glob.glob( mouse_path+'\\Atlas_', recursive=False)
            emptyfovs=glob.glob( mouse_path+'\\FOV_', recursive=False)
            for  emptyfov in emptyfovs:
                noimagin=0
                nofacecamera=0
                novisstim=0       
                if not glob.glob( emptyfov+'\\**\\**.env', recursive=True) :
                    noimagin=1                  
        
                if not glob.glob( emptyfov+'\\**\\DisplaySettings.json', recursive=True) :
                    nofacecamera=1
        
                if not glob.glob( emptyfov+'\\**\\**.mat', recursive=True) :
                    novisstim=1
                    
                if noimagin and  nofacecamera and novisstim:
                    recursively_eliminate_empty_folders(emptyfov)  
                    
                else:    
                    if emptyfov.endswith('FOV_'):
                        all_fovs=glob.glob(mouse_path +'\\FOV_**', recursive=False) 
                        empty_fovs=glob.glob(mouse_path +'\\FOV_', recursive=False) 
                        current_fov=[i for i in all_fovs if i not in empty_fovs]
                        curent_good_fovs=len(current_fov)
                        fov_number=[os.path.split(i)[1] for i in all_fovs if i not in empty_fovs]
                        current=[int(i[i.find('_')+1:]) for i in fov_number]
                        wanted=[i+1 for i in range(curent_good_fovs)]
                        to_change=[i for i in current if i not in wanted ]                  
                        to_change_to=[i for i in wanted if i not in current ]
                        
                        if curent_good_fovs==0:
                            to_change_to=[1]                         
                        os.rename(emptyfov, emptyfov+str(to_change_to[0]))  
                        
            for  emptyatlas in emptyatlases:
                if not glob.glob( emptyatlas+'\\**\\**.env', recursive=True) :
                    recursively_eliminate_empty_folders(emptyatlas) 
                else:    
                     if emptyatlas.endswith('Atlas_'):
                         all_atlases=glob.glob(mouse_path +'\\Atlas_**', recursive=False) 
                         empty_atlases=glob.glob(mouse_path +'\\Atlas_', recursive=False) 
                         current_atlas=[i for i in all_atlases if i not in empty_atlases]
                         curent_good_atlases=len(current_atlas)
                         atlas_number=[os.path.split(i)[1] for i in all_atlases if i not in empty_atlases]
                         current_atlas=[int(i[i.find('_')+1:]) for i in atlas_number]
                         wanted_atlas=[i+1 for i in range(curent_good_atlases)]
                         to_change_atlases=[i for i in current_atlas if i not in wanted_atlas ]                  
                         to_change_to_atlases=[i for i in wanted_atlas if i not in current_atlas ]
                         
                         if curent_good_atlases==0:
                             to_change_to_atlases=[1]                         
                         os.rename(emptyatlas, emptyatlas+str(to_change_to_atlases[0]))     
                    
                
            all_atlases=glob.glob(mouse_path +'\\Atlas_**', recursive=False) 
            all_fovs=glob.glob(mouse_path +'\\FOV_**', recursive=False) 
            
            empty_fovs=glob.glob(mouse_path +'\\FOV_', recursive=False) 
            empty_atlases=glob.glob(mouse_path +'\\Atlas_**', recursive=False) 

            current_fov=[i for i in all_fovs if i not in empty_fovs]
            current_atlas=[i for i in all_atlases if i not in empty_atlases]

            curent_good_fovs=len(current_fov)
            curent_good_atlases=len(current_atlas)

            fov_number=[os.path.split(i)[1] for i in all_fovs if i not in empty_fovs]
            atlas_number=[os.path.split(i)[1] for i in all_atlases if i not in empty_atlases]

            current=[int(i[i.find('_')+1:]) for i in fov_number]
            current_atlas=[int(i[i.find('_')+1:]) for i in atlas_number]
            
            wanted=[i+1 for i in range(curent_good_fovs)]
            wanted_atlas=[i+1 for i in range(curent_good_atlases)]

            to_change=[i for i in current_atlas if i not in wanted_atlas ]
            to_change_atlases=[i for i in current_atlas if i not in wanted_atlas ]
            
            
            for atlas in current_atlas:
                 if to_change_atlases:
                    if atlas.find('Atlas_' + str(to_change[0]))!=-1: 
                        os.rename(atlas, atlas[:atlas.find('Atlas_')+5]+str(to_change_to_atlases[0]))   
                        to_change_atlases.remove(to_change_atlases[0])
                             
            all_atlases=glob.glob(mouse_path +'\\Atlas_**', recursive=False) 
            
            for fov in current_fov:
                if to_change:
                   if fov.find('FOV_' + str(to_change[0]))!=-1: 
                       os.rename(fov, fov[:fov.find('FOV_')+3]+str(to_change_to[0]))   
                       to_change.remove(to_change[0])
                            
            all_fovs=glob.glob(mouse_path +'\\FOV_**', recursive=False) 
            
            
            for FOV in all_fovs:
        
                Plane3Tomato1050=os.path.join(FOV, '1050_3PlaneTomato')
                HighResStackTomato1050=os.path.join(FOV, '1050_HighResStackTomato' )  
                Tomato1050=os.path.join(FOV, '1050_Tomato')
                HighResStackGreen=os.path.join(FOV, 'HighResStackGreen')
                SurfaceImage =os.path.join(FOV, 'SurfaceImage' )
                OtherAcq=os.path.join(FOV, 'OtherAcq')
                
                all_FOVs_Aq_folders=[Plane3Tomato1050, HighResStackTomato1050, Tomato1050, HighResStackGreen, SurfaceImage, OtherAcq, FOV]       
        
                self.process_aquisition_folder(all_FOVs_Aq_folders, mouse_path)
                              
            for atlas in all_atlases:
          
                  Overview=os.path.join(atlas, 'Overview')
                  Preview=os.path.join(atlas, 'Preview' )  
                  Volumes=os.path.join(atlas, 'Volumes')
                  Coordinates=os.path.join(atlas, 'Coordinates')
                  if os.path.isdir(Coordinates):
                    if  len(os.listdir(Coordinates))==0:
                        recursively_eliminate_empty_folders(Coordinates)  
                  
                  all_atlases_Aq_folders=[Overview, Preview, Volumes]       
          
                  self.process_aquisition_folder(all_atlases_Aq_folders, mouse_path)
                            
    def process_aquisition_folder(self, aq_folder_list, mouse_path):
    
        UnprocessedFaceCameras=os.path.join(mouse_path,'UnprocessedFaceCameras')   
        UnprocessedVisStim=os.path.join(mouse_path,'UnprocessedVisStim')   
        UnprocessedFaceCameraspaths=glob.glob(UnprocessedFaceCameras +'\\**\\**Default.ome.tif', recursive=False)
        UnprocessedFaceCamerasnames=[os.path.split(UnprocessedFaceCameraspath)[1] for UnprocessedFaceCameraspath in UnprocessedFaceCameraspaths]
        UnprocessedVisStimpaths=glob.glob(UnprocessedVisStim +'\\**.mat', recursive=False)
        UnprocessedVisStimnames=[os.path.split(UnprocessedVisStimpaths)[1] for UnprocessedVisStimpaths in UnprocessedVisStimpaths]
    
        
        if glob.glob(UnprocessedVisStim +'\\**.mat.mat', recursive=False) :
            for mat in glob.glob(UnprocessedVisStim +'\\**.mat.mat', recursive=False):      
                os.rename(mat,mat[:-4])
    
        
        for generic_aq_folder in aq_folder_list:
            # print(generic_aq_folder)
            if 'FOV' in os.path.split(generic_aq_folder)[1]:
                generic_aq_folder_prairieaq=glob.glob(generic_aq_folder +'\\**\\**.env', recursive=False)  
            elif 'Atlas' in os.path.split(generic_aq_folder)[1] :
                generic_aq_folder_prairieaq=glob.glob(generic_aq_folder +'\\**\\**.env', recursive=False)  
            else:
                generic_aq_folder_prairieaq=glob.glob(generic_aq_folder +'\\**\\**.env', recursive=False) 
                           
            allaq=glob.glob(generic_aq_folder +'\\Aq_**', recursive=False) 
            emptyaq=glob.glob(generic_aq_folder +'\\Aq_', recursive=False) 
            Aq_number=[os.path.split(i)[1] for i in allaq if i not in emptyaq]
            currentaq=[i for i in allaq if i not in emptyaq]
            curent_good_aq=len(Aq_number)
    
            if generic_aq_folder_prairieaq:
                for i, aq_path in enumerate(generic_aq_folder_prairieaq):  
                    destination=os.path.join(generic_aq_folder,'Aq_'+str(1+i+curent_good_aq))
                    os.mkdir(destination) 
                    shutil.move(os.path.split(aq_path)[0],destination)
                    
            allaq=glob.glob(generic_aq_folder +'\\Aq_**', recursive=False)    
    
    
            for aq in allaq:
                # print(aq)
                
                if glob.glob(aq +'\\EyeCamera', recursive=False):
                    os.rename(aq +'\\EyeCamera',aq +'\\FaceCamera')
                    
                noimagin=0
                nofacecamera=0
                novisstim=0
                
                if not glob.glob(aq +'\\**\\**.env', recursive=False):
                    noimagin=1                  
                if not glob.glob(aq +'\\FaceCamera\\**\\DisplaySettings.json', recursive=True):          
                    nofacecamera=1
                    if os.path.isdir(aq +'\\FaceCamera'):
                         recursively_eliminate_empty_folders(aq +'\\FaceCamera')  
                else:
                    if not glob.glob(aq +'\\FaceCamera\\DisplaySettings.json', recursive=False):
                        falsefacecameradir=os.path.split(glob.glob(aq +'\\FaceCamera\\**\\DisplaySettings.json', recursive=False)[0])[0]
                        
                        files = os.listdir(falsefacecameradir)
                        for f in files:
                            shutil.move(os.path.join(falsefacecameradir, f), aq +'\\FaceCamera') 
                        if not glob.glob(falsefacecameradir +'\\DisplaySettings.json', recursive=False):
                            recursively_eliminate_empty_folders(falsefacecameradir)      
                    
                    
                if not glob.glob(aq +'\\VisStim\\**.mat', recursive=False):
                    novisstim=1
                    if os.path.isdir(aq +'\\VisStim'):
                        recursively_eliminate_empty_folders(aq +'\\VisStim') 
                        
                elif glob.glob(aq +'\\VisStim'+'\\**.mat.mat', recursive=False) :                   
                    for mat in glob.glob(aq +'\\VisStim'+'\\**.mat.mat', recursive=False) :  
                        os.rename(mat,mat[:-3])
    
                if noimagin and  nofacecamera and novisstim:
                    recursively_eliminate_empty_folders(aq)  
                    
                elif noimagin and novisstim:
                    print('Face camera only')
                    
                    
                else:    
                    prairie_imaging_path=os.path.split(glob.glob(aq +'\\**\\**.env', recursive=False)[0])[0]
                    if aq.endswith('Aq_'):
                       allaq=glob.glob(generic_aq_folder +'\\Aq_**', recursive=False) 
                       emptyaq=glob.glob(generic_aq_folder +'\\Aq_', recursive=False) 
                       currentaq=[i for i in allaq if i not in emptyaq]
                       curent_good_aq=len(currentaq)
                       Aq_number=[os.path.split(i)[1] for i in allaq if i not in emptyaq]
                       current=[int(i[i.find('_')+1:]) for i in Aq_number]
                       wanted=[i+1 for i in range(curent_good_aq)]
                       to_change=[i for i in current if i not in wanted ]                  
                       to_change_to=[i for i in wanted if i not in current ]                   
                       os.rename(aq, aq+str(to_change_to[0]))  
                       
                    if os.path.isdir(prairie_imaging_path):
                        self.file_cleanup_prairie_new(prairie_imaging_path)   
                    
                    if nofacecamera and novisstim:  
                    
                        if UnprocessedFaceCamerasnames or UnprocessedVisStimnames:   
                            if not UnprocessedFaceCamerasnames:
                                UnprocessedFaceCamerasnames=['None']
                            if not UnprocessedVisStimnames:
                                UnprocessedVisStimnames=['None']
                            
                            # root = Tkinter.Tk()
                            # app = select_face_camera(root, os.path.split(glob.glob(aq +'\\**', recursive=False)[0])[1], UnprocessedFaceCamerasnames, UnprocessedVisStimnames)
                            # root.mainloop()
                            # get_values=app.values

                            
                            self.select_face_camera_window=select_face_camera(self.LabProjectObject.gui, os.path.split(glob.glob(aq +'\\**', recursive=False)[0])[1], UnprocessedFaceCamerasnames, UnprocessedVisStimnames)
                            self.select_face_camera_window.wait_window()
                            get_values= self.select_face_camera_window.values
                            
                            
                            
                            if get_values[1][1]:
                                facecameradir=os.path.join(aq, 'FaceCamera')   
                                # unprocessedfacecameraname= os.path.split(os.path.split([name for name in UnprocessedFaceCameraspaths if get_values[1][1] in name][0])[0])[1]
                                unprocessedfacecamerafullpath=os.path.split([name for name in UnprocessedFaceCameraspaths if get_values[1][1] in name][0])[0]

                                if not os.path.isdir(facecameradir):
                                    os.mkdir(facecameradir)
                                
                                files = glob.glob(unprocessedfacecamerafullpath+'\\**' )
                                for f in files:
                                      shutil.move(f, facecameradir)               
                                
                            if get_values[2][1]:
                                visstimdir=os.path.join(aq, 'VisStim')   
                                unprocessedvisstim= os.path.split([name for name in UnprocessedVisStimpaths if get_values[2][1] in name][0])[0]
                                if not os.path.isdir(visstimdir):
                                    os.mkdir(visstimdir)
                                
                                files = glob.glob(unprocessedvisstim+'\\**' )
                                for f in files:
                                        shutil.move(f, visstimdir)   
                           
                           
   
            UnprocessedFaceCameraspaths=glob.glob(UnprocessedFaceCameras +'\\**\\**Default.ome.tif', recursive=False)
            UnprocessedFaceCamerasnames=[os.path.split(UnprocessedFaceCameraspath)[1] for UnprocessedFaceCameraspath in UnprocessedFaceCameraspaths]
            UnprocessedVisStimpaths=glob.glob(UnprocessedVisStim +'\\**.mat', recursive=False)
            UnprocessedVisStimnames=[os.path.split(UnprocessedVisStimpaths)[1] for UnprocessedVisStimpaths in UnprocessedVisStimpaths]   
            if not UnprocessedFaceCameraspaths:
                if os.path.isdir(UnprocessedFaceCameras):
                    recursively_eliminate_empty_folders(UnprocessedFaceCameras)
            if not UnprocessedVisStimpaths:
                if os.path.isdir(UnprocessedVisStim):
                    recursively_eliminate_empty_folders(UnprocessedVisStim)    
                        
    
            allaq=glob.glob(generic_aq_folder +'\\Aq_**', recursive=False) 
            emptyaq=glob.glob(generic_aq_folder +'\\Aq_', recursive=False) 
            currentaq=[i for i in allaq if i not in emptyaq]
            curent_good_aq=len(currentaq)
            Aq_number=[os.path.split(i)[1] for i in allaq if i not in emptyaq]
            current=[int(i[i.find('_')+1:]) for i in Aq_number]
            wanted=[i+1 for i in range(curent_good_aq)]
            to_change=[i for i in current if i not in wanted ]                  
            to_change_to=[i for i in wanted if i not in current ] 
            
            for aq in currentaq:
                if to_change:
                   if aq.find('Aq_' + str(to_change[0]))!=-1: 
                       os.rename(aq, aq[:aq.find('Aq_')+3]+str(to_change_to[0]))   
                       to_change.remove(to_change[0])
            # print('finalremoval')           
            recursively_eliminate_empty_folders(generic_aq_folder)
            # print('finalremovaldone')                   
  
    def file_cleanup_prairie_new(self, prairie_imaging_path):
               
        # file_list = os.listdir(prairie_imaging_path)
    
    # check channle and plane structure     and current folders
        directory_red=os.path.join(prairie_imaging_path,'Ch1Red')
        directory_green=os.path.join(prairie_imaging_path, 'Ch2Green')
        
        correction=False  
        ChannelPaths=[directory_red, directory_green]  
        ChannelRedExists=False
        ChannelGreenExists=False
        PlaneNumber=False
        
        if os.path.exists(directory_red) or os.path.exists(directory_green):            
              aq_info=check_channels_and_planes(prairie_imaging_path, correction)
              correction=True  
              if os.path.exists(directory_red):
                  ChannelRedExists=True            
                  folder_selected_list_red = os.listdir(directory_red)
                  if any('plane' in file_name  for file_name in folder_selected_list_red if os.path.isdir(os.path.join(directory_red , file_name))):
                      
                      PlaneNumber=len(folder_selected_list_red) + aq_info[9]
                      Multiplane=False
                      if PlaneNumber>1:
                          Multiplane=True
                  else:
                      aq_info =check_channels_and_planes(directory_red, correction)
                  
              if os.path.exists(directory_green):
                  ChannelGreenExists=True           
                  folder_selected_list_green = os.listdir(directory_green)  
    
                  if any('plane' in file_name  for file_name in folder_selected_list_green if os.path.isdir(directory_green + os.sep + file_name)):
                      PlaneNumber=len(folder_selected_list_green) + aq_info[10]
                      Multiplane=False
                      if PlaneNumber>1:
                          Multiplane=True
                  else:
                     aq_info=check_channels_and_planes(directory_green, correction)
    
        else:
            aq_info = check_channels_and_planes(prairie_imaging_path, correction)
            if aq_info[0]:
                ChannelRedExists=1
                PlaneNumber=aq_info[9]
            if aq_info[1]:
                ChannelGreenExists=1
                PlaneNumber=aq_info[10]
    
        ImagedChannels=['lolo','lolo']
        if ChannelRedExists:
            ImagedChannels[0]='Ch1Red'
        if ChannelGreenExists:
            ImagedChannels[1]='Ch2Green'
    
         # create necessary folders     
        if ChannelRedExists or ChannelGreenExists:     
            all_image_sequence_paths=[]
            PlanePaths=[os.sep +'plane'+str(i+1) for i in range(PlaneNumber)]       
            for ch in ImagedChannels:
                for i, channel_path in enumerate(ChannelPaths):
                    if ch in channel_path :
                        for n in range(PlaneNumber):
                            all_image_sequence_paths.append(ChannelPaths[i]+PlanePaths[n])
                            if not os.path.exists(ChannelPaths[i]+PlanePaths[n]):
                                os.makedirs(ChannelPaths[i]+PlanePaths[n])
    
           
                            
            # move files  
            # print('Moving Files')     
            Multiplane=aq_info[8]
            if correction:
                if glob.glob(prairie_imaging_path+'\\**.tif', recursive=False):             
                    move_files(prairie_imaging_path,ChannelPaths,PlanePaths, Multiplane,aq_info[-1] ) 
                for channel_folder in ChannelPaths:
                    if os.path.isdir(channel_folder):
                        file_list_channel = os.listdir(channel_folder)
                        
                        if len (file_list_channel)>3:
                             move_files(channel_folder,ChannelPaths,PlanePaths, Multiplane, aq_info[-1]) 
                        elif len(file_list_channel)<3:  
                             for plane_folder in file_list_channel:
                                 if os.path.isdir(plane_folder):
                                     # file_list_plane=os.listdir(plane_folder)
                                     move_files(plane_folder,ChannelPaths,PlanePaths, Multiplane,aq_info[-1] ) 
                  
            else:       
                move_files(prairie_imaging_path,ChannelPaths,PlanePaths, Multiplane,aq_info[-1]) 
    
            return  [ImagedChannels, PlaneNumber, all_image_sequence_paths]      
        else:        
            return  [False, False, False]  
        
    def  cleaning_up_calibrations(self, session_path):  
        
        calibrations_directory_path=os.path.join(session_path,'Calibrations')
        recursively_eliminate_empty_folders(calibrations_directory_path)          
        

