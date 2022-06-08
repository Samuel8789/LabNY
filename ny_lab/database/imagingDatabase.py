
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 10:17:25 2021

@author: sp3660
"""
import logging 
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/AllFunctions')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/MainClasses')
# import tkinter as Tkinter
import pandas as pd
import numpy as np
import datetime
import os
import glob

from .fun.guiFunctions.addSessionInfo import AddSessionInfo
from .fun.guiFunctions.addImagedmouseInfo import AddImagedmouseInfo
from .fun.guiFunctions.addWidefieldInfo import AddWidefieldInfo
from .fun.guiFunctions.addAcquisitionInfo import AddAcquisitionInfo
from .fun.guiFunctions.addImagingInfo import AddImagingInfo
from .fun.guiFunctions.addFacecameraInfo import AddFacecameraInfo


from .fun.databaseCodesTransformations import transform_filterinfo_to_codes

from ..dataManaging.classes.standalone.metadata import Metadata


#%%
class ImagingDatabase():
    
    
    def __init__(self, database_object):
        self.databse_ref=database_object
        self.database_connection = self.databse_ref.database_connection
        self.update_variables()
        
    def update_variables(self):
        
        self.table_all_imaging_session()
        self.table_all_imaged_mice()
        # print('Getting chandim Specific Tables')
        self.get_chandelier_imaging_imaged_mice()
        # print('Getting chandopt Specific Tables')
        self.get_chandelier_opto_imaged_mice()
        # print('Getting inim Specific Tables')
        self.get_interneuron_imaging_imaged_mice()
        # print('Getting inop Specific Tables')
        self.get_interneuron_opto_imaged_mice()  
        # print('Getting tig Specific Tables')
        self.get_tigres_imaged_mice()
        # print('Getting Last IDS')
        self.get_last_imagedmice_id()
        self.get_last_acquisition_id()
        self.get_last_imagingsession_id()
        self.get_last_imaging_id()        
        self.get_last_vistim_id()
        self.get_last_facecamera_id()
        self.get_last_widefield_id()
        # print('Getting Full Tables')
        self.get_all_acquisitions()
        self.full_info_table()

        
#%% imging database queries
    def table_all_imaging_session(self):
        
        query_all_imaging_session="""
        SELECT *
        FROM ImagingSessions_table a
           
        """
        self.all_imaging_sessions=self.databse_ref.arbitrary_query_to_df(query_all_imaging_session)
        
    def table_all_imaged_mice(self):  

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
                    Ai148_table.Genotypes_types AS Ai148                  

            FROM ImagedMice_table  a   
            LEFT JOIN ImagingSessions_table b ON a.SessionID=b.ID
            LEFT JOIN ExperimentalAnimals_table c ON c.ID=a.ExpID
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
            
        self.all_imaged_mice=self.databse_ref.arbitrary_query_to_df(query_all_imaged_mice)        
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
        
    def get_chandelier_imaging_imaged_mice(self):
        
        chandelier_imaging_imaged_mice=self.all_imaged_mice[self.all_imaged_mice['Projects']=='Chandelier_Imaging']
        chandelier_imaging_imaged_mice=chandelier_imaging_imaged_mice.sort_values(by=['ImagingDate'],ascending=False)
        
        chandelier_imaging_imaged_mice_grouped=chandelier_imaging_imaged_mice.groupby(['Code','Projects','Combination','Recombinases1','Sensors1','Recombinases2','Sensors2'])['ImagingDate'].apply(list)        
        chandelier_imaging_imaged_mice_frame=chandelier_imaging_imaged_mice_grouped.to_frame()
        self.chandelier_imaging_imaged_mice=pd.DataFrame(chandelier_imaging_imaged_mice_frame.ImagingDate.values.tolist(), chandelier_imaging_imaged_mice_frame.index).add_prefix('Session_')
        self.chandelier_imaging_imaged_mice.reset_index(inplace=True)
    def get_chandelier_opto_imaged_mice(self):
        
        chandelier_opto_imaged_mice=self.all_imaged_mice[self.all_imaged_mice['Projects']=='Chandelier_Optogenetics']
        chandelier_opto_imaged_mice=chandelier_opto_imaged_mice.sort_values(by=['ImagingDate'],ascending=False)
        
        
        chandelier_opto_imaged_mice_grouped=chandelier_opto_imaged_mice.groupby(['Code','Projects',
                                                                                 'Combination','Recombinases1',
                                                                                 'Sensors1','Optos1','Recombinases2',
                                                                                 'Sensors2','Recombinases3','Optos3'])['ImagingDate'].apply(list)        
        chandelier_opto_imaged_mice_frame=chandelier_opto_imaged_mice_grouped.to_frame()
        self.chandelier_opto_imaged_mice=pd.DataFrame(chandelier_opto_imaged_mice_frame.ImagingDate.values.tolist(),
                                                      chandelier_opto_imaged_mice_frame.index).add_prefix('Session_')
        self.chandelier_opto_imaged_mice.reset_index(inplace=True)
    def get_interneuron_imaging_imaged_mice(self):
        
        interneuron_imaging_imaged_mice=self.all_imaged_mice[self.all_imaged_mice['Projects']=='Interneuron_Imaging']
        interneuron_imaging_imaged_mice=interneuron_imaging_imaged_mice.sort_values(by=['ImagingDate'],ascending=False)
        
        interneuron_imaging_imaged_mice_grouped=interneuron_imaging_imaged_mice.groupby(['Code','Projects','Combination','Recombinases1','Sensors1','Recombinases2','Sensors2'])['ImagingDate'].apply(list)        
        interneuron_imaging_imaged_mice_frame=interneuron_imaging_imaged_mice_grouped.to_frame()
        self.interneuron_imaging_imaged_mice=pd.DataFrame(interneuron_imaging_imaged_mice_frame.ImagingDate.values.tolist(), interneuron_imaging_imaged_mice_frame.index).add_prefix('Session_')
        self.interneuron_imaging_imaged_mice.reset_index(inplace=True)
    def get_interneuron_opto_imaged_mice(self):    
        
        interneuron_opto_imaged_mice =self.all_imaged_mice[self.all_imaged_mice['Projects']=='Interneuron_Optogenetics']
        interneuron_opto_imaged_mice=interneuron_opto_imaged_mice.sort_values(by=['ImagingDate'],ascending=False)
      
        interneuron_opto_imaged_mice_grouped=interneuron_opto_imaged_mice.groupby(['Code','Projects','Combination','Recombinases1','Sensors1','Optos1','Recombinases2','Sensors2','Recombinases3','Optos3'])['ImagingDate'].apply(list)        
        interneuron_opto_imaged_mice_frame=interneuron_opto_imaged_mice_grouped.to_frame()
        self.interneuron_opto_imaged_mice=pd.DataFrame(interneuron_opto_imaged_mice_frame.ImagingDate.values.tolist(), interneuron_opto_imaged_mice_frame.index).add_prefix('Session_')
        self.interneuron_opto_imaged_mice.reset_index(inplace=True)
    def get_tigres_imaged_mice(self):
        
        tigres_imaged_mice=self.all_imaged_mice[self.all_imaged_mice['Projects']=='Tigre_Controls']
        tigres_imaged_mice=tigres_imaged_mice.sort_values(by=['ImagingDate'],ascending=False)
    
        tigres_imaged_mice_grouped=tigres_imaged_mice.groupby(['Code','Projects','Combination','Recombinases3','Optos3'])['ImagingDate'].apply(list)        
        tigres_imaged_mice_grouped_frame=tigres_imaged_mice_grouped.to_frame()
        self.tigres_imaged_mice=pd.DataFrame(tigres_imaged_mice_grouped_frame.ImagingDate.values.tolist(), tigres_imaged_mice_grouped_frame.index).add_prefix('Session_')
        self.tigres_imaged_mice.reset_index(inplace=True)
    def get_all_acquisitions(self):
        
         query_all_acquistions="""
            SELECT f.Code,
                    a.*,
                    b.*,
                    c.*,
                    d.*   
            
            FROM Acquisitions_table  a   
            LEFT JOIN Imaging_table b             ON b.ID=a.ImagingID
            LEFT JOIN FaceCamera_table c          ON c.ID=a.FaceCameraID
            LEFT JOIN VisualStimulations_table d  ON d.ID=a.VisStimulationID
            LEFT JOIN ImagedMice_table e          ON e.ID=a.ImagedMouseID
            LEFT JOIN ExperimentalAnimals_table f ON f.ID=e.ExpID
            """
         self.all_acquisitions=self.databse_ref.arbitrary_query_to_df(query_all_acquistions)        


    def get_filtered_df(self,df, columns_values_tuple):
        df_filtered=df
        for db_filter in columns_values_tuple:
            df_filtered=df_filtered[df_filtered[db_filter[0]]==db_filter[1]]     
            
        return df_filtered
        
    def full_info_table(self)   :
        query_all_acquistion_full_info="""
            SELECT
                    c.Code,         
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
                    
                    Acquisitions_table.*,
                    z.*,
                    aa.*,
                    bb.*,
                    cc.*,
                    
                    
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
                    Ai148_table.Genotypes_types AS Ai148      
            
            
            
            
            FROM Acquisitions_table   
            LEFT JOIN ImagedMice_table  a   ON a.ID=Acquisitions_table.ImagedMouseID
            LEFT JOIN ImagingSessions_table b ON a.SessionID=b.ID
            LEFT JOIN ExperimentalAnimals_table c ON c.ID=a.ExpID
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
            LEFT JOIN Imaging_table aa            ON aa.ID=Acquisitions_table.ImagingID
            LEFT JOIN FaceCamera_table bb        ON bb.ID=Acquisitions_table.FaceCameraID
            LEFT JOIN VisualStimulations_table cc  ON cc.ID=Acquisitions_table.VisStimulationID
            

            """
        self.all_acquisitons_full_info=self.databse_ref.arbitrary_query_to_df(query_all_acquistion_full_info)       
        # print('Making dates')
        age=pd.to_datetime(self.all_acquisitons_full_info.ImagingDate) - pd.to_datetime(self.all_acquisitons_full_info.DOB)
        wai=pd.to_datetime(self.all_acquisitons_full_info.ImagingDate) - pd.to_datetime(self.all_acquisitons_full_info.InjectionDate)
        waw=pd.to_datetime(self.all_acquisitons_full_info.ImagingDate) - pd.to_datetime(self.all_acquisitons_full_info.WindowDate)
        age=age.dt.days/7
        wai=wai.dt.days/7
        waw=waw.dt.days/7
        
        
        self.all_acquisitons_full_info.insert(8, 'Age',  age.round(1))
        self.all_acquisitons_full_info.insert(10, 'WAI',   wai.round(1))
        self.all_acquisitons_full_info.insert(12, 'WAW',   waw.round(1))
        # print('Filloin NAs')



    



        self.all_acquisitons_full_info.Combination=self.all_acquisitons_full_info.Combination.fillna(value='NoInjection')
        self.all_acquisitons_full_info.Sensors1=self.all_acquisitons_full_info.Sensors1.fillna(value='NoInjection')
        self.all_acquisitons_full_info.Optos1=self.all_acquisitons_full_info.Optos1.fillna(value='NoInjection')
        self.all_acquisitons_full_info.Sensors2=self.all_acquisitons_full_info.Sensors2.fillna(value='NoInjection')
        self.all_acquisitons_full_info.Recombinases1=self.all_acquisitons_full_info.Recombinases1.fillna(value='NoInjection')
        self.all_acquisitons_full_info.Recombinases2=self.all_acquisitons_full_info.Recombinases2.fillna(value='NoInjection')
        self.all_acquisitons_full_info.Recombinases3=self.all_acquisitons_full_info.Recombinases3.fillna(value='NoInjection')
        self.all_acquisitons_full_info.Optos3=self.all_acquisitons_full_info.Optos3.fillna(value='NoInjection')
#          query_all_info="""
#              SELECT
#                     c.Code,         
#                     Sex_types AS Sex,
#                     Labels_types,
#                     Line_short,
#                     g.Projects,
#                     c.Notes AS ExpMouseNotes,
                    
#                     b.ImagingDate,
#                     b.Objectives,
#                     b.EndOfSessionSummary,  
                    
#                     date(f.DOB) AS DOB,
#                     date(c.Injection1Date) AS InjectionDate,
#                     date(c.WindowDate) AS WindowDate,
                                     
#                     a.EyesComments,
#                     a.BehaviourComments,
#                     a.FurComments,
#                     a.LessionComments,
                    
                    
#                     Acquisitions_table.*,
#                     z.*,
#                     aa.*,
#                     bb.*,
#                     cc.*,
                    

#                     e.Notes AS InjectionNotes,
#                     Combination,
#                     e.InjectionSite1volume,                                      
#                     k.Sensors AS Sensors1,
#                     e.DilutionSensor1,
#                     l.Optos AS Optos1,
#                     m.Promoters AS Promoters1,
#                     n.Recombinases AS Recombinases1,
#                     o.Sensors AS Sensors2,
#                     e.DilutionSensor2,
#                     p.Promoters AS Promoters2,
#                     q.Recombinases AS Recombinases2,
#                     r.Optos AS Optos3,
#                     e.DilutionOpto,
#                     s.Promoters AS Promoters3,
#                     t.Recombinases AS Recombinases3,
                    
#                     G2C_table.Genotypes_types AS G2C,
#                     Ai14_table.Genotypes_types AS Ai14,
#                     Ai75_table.Genotypes_types AS Ai75,
#                     VRC_table.Genotypes_types AS VRC,
#                     SLF_table.Genotypes_types AS SLF,
#                     PVF_table.Genotypes_types AS PVF,
#                     Ai65_table.Genotypes_types AS Ai65,
#                     Ai80_table.Genotypes_types AS Ai80,
#                     VGC_table.Genotypes_types AS VGC,
#                     Ai162_table.Genotypes_types AS Ai162,
#                     Ai148_table.Genotypes_types AS Ai148 ,
#                     Room_types,
#                     f.Alive,
#                     Genotypings_types,
#                     f.Notes AS MouseNotes,
#                     Experimental_types,
#                     ExperimentalStatus_table.Status,
#                     GreenFilters_table.*,
#                     RedFilters_table.*,
#                     DichroicBeamSplitters_table.DichroicBeamSplitter,
#                     InjectionArea.*,
#                     Inj1Coord.*,
#                     Inj2Coord.*,
#                     WindowArea.*,
#                     CoverslipSize,
#                     WindowTypes_table.CraniectomyType,
#                     CraniectomySize,
#                     Behaviours_table.Behaviour
            
            
            
#             FROM Acquisitions_table   
#             LEFT JOIN ImagedMice_table  a   ON a.ID=Acquisitions_table.ImagedMouseID
#             LEFT JOIN ImagingSessions_table b ON a.SessionID=b.ID
#             LEFT JOIN ExperimentalAnimals_table c ON c.ID=a.ExpID
#             LEFT JOIN Windows_table d ON d.ID=c.WindowID
#             LEFT JOIN Injections_table e ON e.ID=c.Injection1ID
#             LEFT JOIN MICE_table f ON f.ID=c.Mouse_ID
#             LEFT JOIN Projects_table g ON g.ID=c.Project 
#             LEFT JOIN VirusCombinations_table  ON VirusCombinations_table.ID=e.VirusCombination         
#             LEFT JOIN Virus_table h ON h.ID=VirusCombinations_table.Virus1
#             LEFT JOIN Virus_table i ON i.ID=VirusCombinations_table.Virus2
#             LEFT JOIN Virus_table j ON j.ID=VirusCombinations_table.Virus3
#             LEFT JOIN Sensors_table k ON k.ID=h.Sensor
#             LEFT JOIN Optos_table l ON l.ID=h.Opto
#             LEFT JOIN Promoter_table m ON m.ID=h.Promoter
#             LEFT JOIN Recombinase_table n ON n.ID=h.Recombinase
#             LEFT JOIN Sensors_table o ON o.ID=i.Sensor
#             LEFT JOIN Promoter_table p ON p.ID=i.Promoter
#             LEFT JOIN Recombinase_table q ON q.ID=i.Recombinase
#             LEFT JOIN Optos_table r ON r.ID=j.Opto
#             LEFT JOIN Promoter_table s ON s.ID=j.Promoter
#             LEFT JOIN Recombinase_table t ON t.ID=j.Recombinase
#             LEFT JOIN Genotypes_table AS G2C_table   ON f.G2C   = G2C_table.ID
#             LEFT JOIN Genotypes_table AS Ai14_table   ON f.Ai14   = Ai14_table.ID
#             LEFT JOIN Genotypes_table AS Ai75_table   ON f.Ai75   = Ai75_table.ID
#             LEFT JOIN Genotypes_table AS VRC_table   ON f.VRC   = VRC_table.ID
#             LEFT JOIN Genotypes_table AS SLF_table   ON f.SLF   = SLF_table.ID
#             LEFT JOIN Genotypes_table AS PVF_table   ON f.PVF   = PVF_table.ID
#             LEFT JOIN Genotypes_table AS Ai65_table   ON f.Ai65   = Ai65_table.ID
#             LEFT JOIN Genotypes_table AS Ai80_table   ON f.Ai80   = Ai80_table.ID
#             LEFT JOIN Genotypes_table AS VGC_table   ON f.VGC  = VGC_table.ID
#             LEFT JOIN Genotypes_table AS Ai162_table   ON f.Ai162   = Ai162_table.ID
#             LEFT JOIN Genotypes_table AS Ai148_table   ON f.Ai148  = Ai148_table.ID          
#             LEFT JOIN Sex_table ON Sex_table.ID=f.Sex
#             LEFT JOIN Lines_table v ON v.ID=f.Line 
#             LEFT JOIN WideField_table z ON z.ID=a.WideFIeldID 
#             LEFT JOIN Imaging_table aa            ON aa.ID=Acquisitions_table.ImagingID
#             LEFT JOIN FaceCamera_table bb        ON bb.ID=Acquisitions_table.FaceCameraID
#             LEFT JOIN VisualStimulations_table cc  ON cc.ID=Acquisitions_table.VisStimulationID     
#             LEFT JOIN Experimental_table ON Experimental_table.ID=c.Experiment           
#             LEFT JOIN ExperimentalStatus_table ON ExperimentalStatus_table.ID=c.Experimental_status   
#             LEFT JOIN Labels_table ON Labels_table.ID=f.Label
#             LEFT JOIN Rooms_table ON Rooms_table.ID=f.Room    
#             LEFT JOIN Genotypings_table ON Genotypings_table.ID=f.Genotyping_Status  
            
#             LEFT JOIN Behaviours_table ON Behaviours_table.ID=a.BehaviourProtocol
            
#             LEFT JOIN DichroicBeamSplitters_table ON DichroicBeamSplitters_table.ID=aa.DichroicBeamsplitter
#             LEFT JOIN GreenFilters_table ON GreenFilters_table.ID=aa.GreenFilter
#             LEFT JOIN RedFilters_table ON RedFilters_table.ID=aa.RedFilter           
#             LEFT JOIN Microscopes_table ON Microscopes_table.ID=b.Microscope
#             LEFT JOIN Sterocoordinates_table ON Sterocoordinates_table.ID=d.HeadPlateCoordinates
#             LEFT JOIN WindowTypes_table ON WindowTypes_table.ID=d.WindowType
#             LEFT JOIN CoverSize_table ON CoverSize_table.ID=d.CoverSize
#             LEFT JOIN Covertype_table ON Covertype_table.ID=d.CoverType
#             LEFT JOIN Craniosize_table ON Craniosize_table.ID=d.CranioSize
#             LEFT JOIN Brain_Areas_table AS WindowArea ON WindowArea.ID=d.CorticalArea
#             LEFT JOIN Brain_Areas_table AS InjectionArea ON InjectionArea.ID=e.CorticalArea
#             LEFT JOIN Sterocoordinates_table AS Inj1Coord ON Inj1Coord.ID=e.InjectionSite1Coordinates
#             LEFT JOIN Sterocoordinates_table AS Inj2Coord ON Inj2Coord.ID=e.InjectionSite2Coordinates

        
#         """
# all_info=MouseDat.arbitrary_query_to_df(query_all_info)     
    def get_all_cage_info_for_imaging(self, cage_list):
        
        query_all_animals_recovery="""
           SELECT 
                Cage, 
                Labels_types,
                Code,
                Lab_number, 
                round(round(julianday('now') - julianday(Injection1Date))) AS DaysFromInjection,
                round(round(julianday('now') - julianday(WindowDate))) AS DaysFromWindow,
                d.Projects,
                Line_short,
                Code as Code2,
                e.InjectionSite1goodvolume,
                e.InjectionSite1bleeding,
                e.InjectionSite2goodvolume,
                e.InjectionSite2bleeding,          
                e.Notes AS InjectionNotes,    
                Code as Code3,
                f.DamagedAreas,
                f.Notes AS WindowNotes,
                Code as Code4,
                Combination,
                k.Sensors AS Sensors1,
                l.Optos AS Optos1,          
                o.Sensors AS Sensors2,           
                r.Optos AS Optos3, 
                Code as Code5,
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
                Ai148_table.Genotypes_types AS Ai148           
            FROM ExperimentalAnimals_table a
            LEFT JOIN MICE_table b ON a.Mouse_ID  = b.ID        
            LEFT JOIN Lines_table c ON c.ID=b.Line
            LEFT JOIN Projects_table d ON d.ID=a.Project
            LEFT JOIN Injections_table e ON e.ExpID = a.ID
            LEFT JOIN Windows_table f ON f.ExpID = a.ID
            LEFT JOIN VirusCombinations_table g ON g.ID=e.VirusCombination
            LEFT JOIN Virus_table h ON h.ID=g.Virus1
            LEFT JOIN Virus_table i ON i.ID=g.Virus2
            LEFT JOIN Virus_table j ON j.ID=g.Virus3
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
            LEFT JOIN Genotypes_table AS G2C_table   ON b.G2C   = G2C_table.ID
            LEFT JOIN Genotypes_table AS Ai14_table   ON b.Ai14   = Ai14_table.ID
            LEFT JOIN Genotypes_table AS Ai75_table   ON b.Ai75   = Ai75_table.ID
            LEFT JOIN Genotypes_table AS VRC_table   ON b.VRC   = VRC_table.ID
            LEFT JOIN Genotypes_table AS SLF_table   ON b.SLF   = SLF_table.ID
            LEFT JOIN Genotypes_table AS PVF_table   ON b.PVF   = PVF_table.ID
            LEFT JOIN Genotypes_table AS Ai65_table   ON b.Ai65   = Ai65_table.ID
            LEFT JOIN Genotypes_table AS Ai80_table   ON b.Ai80   = Ai80_table.ID
            LEFT JOIN Genotypes_table AS VGC_table   ON b.VGC  = VGC_table.ID
            LEFT JOIN Genotypes_table AS Ai162_table   ON b.Ai162   = Ai162_table.ID
            LEFT JOIN Genotypes_table AS Ai148_table   ON b.Ai148  = Ai148_table.ID
            LEFT JOIN Labels_table z on z.ID=a.EarMark 
            WHERE Experiment=4 AND a.Experimental_status=2 AND b.Cage IN(%s)""" % ','.join('?' for i in cage_list) 

        params=tuple(cage_list)
        micebrought=self.databse_ref.remove_unnecesary_genes_from_df((self.databse_ref.remove_unnecesary_virusinfo_from_df(self.databse_ref.arbitrary_query_to_df(query_all_animals_recovery, params))))
        all_dfs=[micebrought.loc[:,'Cage':'Code2'],micebrought.loc[:,'Code2':'Code3'],micebrought.loc[:,'Code3':'Code4'],micebrought.loc[:,'Code4':'Code5'],micebrought.loc[:,'Code5':]]
        
        len(all_dfs[0])+3
        
        with pd.ExcelWriter(os.path.join(r'C:\Users\sp3660\Desktop\Temp_Excel_Files' ,'MouseInfo_{0}_cages({1}).xlsx'.format(datetime.date.today().strftime("%Y%m%d"), str(cage_list)  )),engine='xlsxwriter') as writer:
            for i, df in enumerate(all_dfs):
                df.to_excel(writer,sheet_name='Imaging',startrow=i*(len(all_dfs[0])+3) , startcol=0)
                
    def get_single_acquisition_database_info(self, acquisition_ID):
        
        query_single_acquisition_info="""
            SELECT *        
            FROM Acquisitions_table s
            WHERE ID IN(?)""" 
        query_single_acquisition_imaging_info="""
            SELECT *        
            FROM Imaging_table 
            WHERE AcquisitionID IN(?)""" 
        
        query_single_acquisition_facecam_info="""
            SELECT *        
            FROM FaceCamera_table 
            WHERE AcquisitionID IN(?)""" 
            
        query_single_acquisition_visstim_info="""
            SELECT a.*, b.ID as ProtocolID, b.VisStimProtocol_name        
            FROM VisualStimulations_table a 
            LEFT JOIN VisualStimProtocols_table b ON b.ID=a.VisualStimulationProtocolID
            WHERE AcquisitionID IN(?)""" 
        
        params=(int(acquisition_ID),)
        acq_info=self.databse_ref.arbitrary_query_to_df(query_single_acquisition_info, params)
        imaging_info=self.databse_ref.arbitrary_query_to_df(query_single_acquisition_imaging_info, params)
        visstim_info=self.databse_ref.arbitrary_query_to_df(query_single_acquisition_visstim_info, params)
        facecam_info=self.databse_ref.arbitrary_query_to_df(query_single_acquisition_facecam_info, params)

        acquisition_database_info_dict={'Acq':acq_info,'Imaging':imaging_info,'FaceCam':facecam_info,'VisStim':visstim_info}
        
        return acquisition_database_info_dict
        

#%% ADDING FUNCTIOSN        
    def add_new_session_to_database(self, gui, session_path):
        
        
        self.update_variables() #
        session_ID=self.max_imagingsession_id+1
        UnFormatttedSessionDate=os.path.split(session_path)[1]
        ImagingDate=datetime.datetime.strptime(UnFormatttedSessionDate,'%Y%m%d')
        
        SessionMiceRawPath=os.path.join(session_path,'Mice')
        imagedmicepaths=glob.glob( SessionMiceRawPath+'\\SP**', recursive=False)
        
        # add all mouse here
        for mouse_path in imagedmicepaths:         
            self.add_new_imaged_mice(gui, session_ID, mouse_path, UnFormatttedSessionDate)
        
        ImagingSessionRawPath=session_path
        CalibrationsRawPath=os.path.join(session_path,'Calibrations')
        PowerCalPath=os.path.join(CalibrationsRawPath,'Power')
        MechanicalZStackPath=os.path.join(CalibrationsRawPath,'MechanicalZ')
        ETLCalibrationsPath=os.path.join(CalibrationsRawPath,'ETL')
        AlignmentCalibrationsPath=os.path.join(CalibrationsRawPath,'Alignment')
        MiceRawPath=os.path.join(session_path,'Mice')
        
        # add session gui here
        self.add_session_info_window=AddSessionInfo(gui, ImagingDate)
        self.add_session_info_window.wait_window()
        get_values= self.add_session_info_window.values
        
        query_microscopes="""SELECT* FROM Microscopes_table WHERE Microscope=?"""
        params=(get_values[1][1],)
        Microscope=int(self.databse_ref.arbitrary_query_to_df(query_microscopes, params).ID.iloc[0])
        if len(get_values[2][1])>2:
            StartTime=format(datetime.datetime.strptime(get_values[2][1], "%H:%M\n"),"%H:%M")
        else:
            StartTime=''
        Objectives=get_values[3][1]
        EndOfSessionSummary=get_values[4][1]
        IssuesDuringImaging=get_values[5][1]
        
        
        
        query_add_session="""
                INSERT INTO ImagingSessions_table(
                    ID,
                    ImagingDate,
                    StartTime,
                    Microscope,
                    ImagingSessionRawPath,
                    CalibrationsRawPath,
                    PowerCalPath,
                    MechanicalZStackPath,
                    ETLCalibrationsPath,
                    AlignmentCalibrationsPath,
                    MiceRawPath,
                    Objectives,
                    EndOfSessionSummary,
                    IssuesDuringImaging
                    )
                VALUES(?, date(?), time(?),?,?,?,?,?,?,?,?,?,?,?)
                """
                                      
        params=(session_ID,
                ImagingDate,
                StartTime,
                Microscope,
                ImagingSessionRawPath,
                CalibrationsRawPath,
                PowerCalPath,
                MechanicalZStackPath,
                ETLCalibrationsPath,
                AlignmentCalibrationsPath,
                MiceRawPath,
                Objectives,
                EndOfSessionSummary,
                IssuesDuringImaging)
        
          
        self.databse_ref.arbitrary_inserting_record(query_add_session, params, commit=True)  
        
    def add_new_imaged_mice(self, gui, session_ID, mice_code_path,UnFormatttedSessionDate ):
        self.update_variables()
        imaged_mice_id=self.max_imagedmice_id+1
        
        SessionDate = UnFormatttedSessionDate
        
        Mouse_Code=os.path.split(mice_code_path)[1] 
         
        query_get_exp_id= """
            SELECT ID, SlowStoragePath, WorkingStoragePath
            FROM ExperimentalAnimals_table
            WHERE Code=?
            """
        params=(Mouse_Code,)
        ExpID=int(self.databse_ref.arbitrary_query_to_df(query_get_exp_id, params).iloc[0]['ID'])
        # ExpID=MouseDat.arbitrary_query_to_df(query_get_exp_id, params).iloc[0]['ID']
      
        MouseRawPath=mice_code_path
        
        WideFieldFolderPath=os.path.join(MouseRawPath, 'Widefield')
        WideFieldPath=glob.glob( WideFieldFolderPath+'\\**.tif', recursive=False)[0]
        WideFieldID=np.nan
    

        all_aq_folder=glob.glob( mice_code_path +'\\**\\**Aq_**', recursive=True)
        true_aq_folders=[aq for aq in all_aq_folder if aq[-1]!='_']

        slowstoragepath= os.path.join(self.databse_ref.arbitrary_query_to_df(query_get_exp_id, params).iloc[0]['SlowStoragePath'],'imaging', SessionDate)
        workingstoragepath= os.path.join(self.databse_ref.arbitrary_query_to_df(query_get_exp_id, params).iloc[0]['WorkingStoragePath'],'imaging', SessionDate)
        
        
        for aq in true_aq_folders:
            self.add_new_acquisition(gui,aq, slowstoragepath, workingstoragepath,imaged_mice_id )
         
        if WideFieldPath:
            IsWideFIeld=1
            WideFieldID=self.add_new_widefield(gui,imaged_mice_id, WideFieldPath,slowstoragepath, workingstoragepath)

        self.add_imaged_mouse_info_window=AddImagedmouseInfo(gui, SessionDate, Mouse_Code)
        self.add_imaged_mouse_info_window.wait_window()
        get_values= self.add_imaged_mouse_info_window.values

        TimeSetOnWheel=get_values[1][1]
        EyesComments=get_values[2][1]
        BehaviourComments=get_values[3][1]
        FurComments=get_values[4][1]
        LessionComments=get_values[5][1]
        BehaviourProtocolComments=get_values[8][1]
        OptogeneticsProtocolComments=get_values[10][1]
        Objectives=get_values[11][1]
        EndOfSessionSummary=get_values[12][1]
        IssuesDuringImaging=get_values[13][1]
        
        if get_values[6][1]=='No':
            DexaInjection=0
        else:
            DexaInjection=1
            
        OptogeneticsProtocol=0
        if get_values[9][1]=='None':
            OptogeneticsProtocol=0
        
        query_BehaviourProtocol="""SELECT* FROM Behaviours_table WHERE Behaviour=?"""
        params=(get_values[7][1],)
        BehaviourProtocol=int(self.databse_ref.arbitrary_query_to_df(query_BehaviourProtocol, params).ID.iloc[0])
    
        

        query_add_imagedmouse="""
            INSERT INTO ImagedMice_table(
                    ID,
                    SessionID,
                    ExpID,
                    TimeSetOnWheel,
                    MouseRawPath,
                    EyesComments,
                    BehaviourComments,
                    FurComments,
                    LessionComments,
                    DexaInjection,
                    BehaviourProtocol,
                    BehaviourProtocolComments,
                    OptogeneticsProtocol,
                    OptogeneticsProtocolComments,
                    Objectives,
                    EndOfSessionSummary,
                    IssuesDuringImaging,
                    IsWideFIeld,
                    WideFieldID,
                    SlowStoragePath,
                    WorkingStoragePath,
                    IsSlowStorage,
                    IsWorkingStorage
                    
                    
                )
            VALUES(?,?,?,time(?),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """    
        params=(imaged_mice_id,session_ID, ExpID, TimeSetOnWheel, MouseRawPath, EyesComments, BehaviourComments,
                  FurComments, LessionComments, DexaInjection, BehaviourProtocol,
                  BehaviourProtocolComments, OptogeneticsProtocol,
                  OptogeneticsProtocolComments, Objectives, EndOfSessionSummary,
                  IssuesDuringImaging, IsWideFIeld, WideFieldID,
                  slowstoragepath,workingstoragepath,0,0)    
        
        
        
        
        self.databse_ref.arbitrary_inserting_record(query_add_imagedmouse, params,)    
    
    def add_new_widefield(self,gui, imaged_mice_ID, WideFieldFolderPath, mousesessionslowstoragepath, mousesessionworkingstoragepath):
         self.update_variables()
         widefield_id=self.max_widefield_id+1
         ImagedMouseID=imaged_mice_ID   
         WideFieldImagePath=os.path.split(WideFieldFolderPath)[0]
         WideFieldFileName=os.path.split(WideFieldFolderPath)[1]
         WideFieldDate=WideFieldFileName[:WideFieldFileName.find('SP')-1]
         WideFieldCode=WideFieldFileName[WideFieldFileName.find('SP'):-4]
         
         
         
         self.add_widefield_info_window=AddWidefieldInfo(gui, WideFieldDate, WideFieldCode)
         self.add_widefield_info_window.wait_window()
         get_values= self.add_widefield_info_window.values

         # root = Tkinter.Tk()
         # app = add_widefield_info(root, WideFieldDate, WideFieldCode)
         # root.mainloop()
         # get_values=app.values

         WideFieldComments=get_values[1][1]
         
         slowstoragepath=os.path.join(mousesessionslowstoragepath,'widefield image',WideFieldFileName)   
         workingstoragepath=os.path.join(mousesessionworkingstoragepath,'widefield image',WideFieldFileName)  
         
         
         query_add_widefield="""
                    INSERT INTO WideField_table(
                        ID,
                        ImagedMouseID,
                        WideFieldImagePath,
                        WideFieldFileName,
                        WideFieldComments,
                            IsSlowPath,
                            SlowStoragePath,
                            IsWorkingStorage,
                            WorkingStoragePath                           
                        )
                    VALUES(?,?,?,?,?,?,?,?,?)
                    """    
         params=(widefield_id,
                 ImagedMouseID,
                WideFieldImagePath,
                WideFieldFileName,
                WideFieldComments,
                    0,
                    slowstoragepath,
                    0,
                    workingstoragepath
                    )
                
         self.databse_ref.arbitrary_inserting_record(query_add_widefield, params, )       
         return widefield_id
         
    def add_new_acquisition(self,gui, acquisition_path, mousesessionslowstoragepath, mousesessionworkingstoragepath, imaged_mice_ID=False ):
        self.update_variables()
        acquisition_id=self.max_acquisition_id+1
        IsMouse=0
        
        if imaged_mice_ID:
            IsMouse=1
            ImagedMouseID=imaged_mice_ID   
            
        AcquisitonRawPath=acquisition_path
        AcquisitionNumber=int(os.path.split(AcquisitonRawPath)[1][os.path.split(AcquisitonRawPath)[1].find('_')+1:])
        IsCalibration=0
        IsTestAcquisition=0
        IsNonImagingAcquisition=0   
        Is0CoordinateAcquisiton=0
        IsFOVAcquisition=0
        IsImaging=0
        IsFaceCamera=0
        IsSurfaceImage=0
        IsLocomotion=0
        IsPhotodiode=0
        IsVisStimSignal=0
        IsBehaviour=0
        IsVisualStimulation=0
        IsOptogenetic=0
        IsAtlas=0
        IsAtlasPreview=0
        IsAtlasOverview=0
        IsAtlasVolume=0
        IsGoodName=1
        IsTomatoRef=0   
        IsMultiplanetomatoref=0   
        Ishighrestomatostack=0  
        Ishighresgreenstack=0      
        Isotherfovaq=0   
        if glob.glob( acquisition_path+'\\**', recursive=False) :
            acq_name=glob.glob(acquisition_path+'\\**', recursive=False)[0]
        
      
        self.add_acquisition_info_window=AddAcquisitionInfo(gui, acq_name)
        self.add_acquisition_info_window.wait_window()
        get_values= self.add_acquisition_info_window.values
        Comments=get_values[1][1]
        IsGoodName=get_values[2][1]


        imaging_metadat_file=False
        
        if glob.glob( acquisition_path+'\\**\\**.env', recursive=False) :
            Prairieimagingpath=os.path.split(glob.glob( acquisition_path+'\\**\\**.env', recursive=False)[0])[0]
            Prairieimagingname=os.path.split(Prairieimagingpath)[1]
            IsImaging=1        
            metadata_files=glob.glob( acquisition_path+'\\**\\**.xml', recursive=False)         
            imaging_metadat_file=[file for file in metadata_files if  'VoltageOutput'  not in file and 'VoltageRecording' not in file][0]
            imaging_path=Prairieimagingpath
          
           
        voltage_metadata_file=False
        voltage_files=glob.glob( acquisition_path+'\\**\\**VoltageRecording**', recursive=False)       
        if voltage_files:
            voltage_metadata_file= [voltage_file for voltage_file in voltage_files if '.xml' in voltage_file][0]
        else:
            voltage_metadata_file=False

   
        # loading_metadata
        if imaging_path:
            metadata=Metadata(aq_metadataPath=imaging_metadat_file, voltagerec_metadataPath=voltage_metadata_file)
            if metadata.imaging_metadata:
                imaging_aq_time=metadata.imaging_metadata[0]['ImagingTime']

        if voltage_files :       
            voltage_aq_time=metadata.voltage_aq_time
            recorded_signals=[metadata.recorded_signals, metadata.recorded_signals_csv]


            if 'Locomotion' in list(set(recorded_signals[0]) & set(recorded_signals[1])):
                IsLocomotion=1
            if 'PhotoDiode' or 'Photo Diode' in list(set(recorded_signals[0]) & set(recorded_signals[1])):
                IsPhotodiode=1
            if 'VisStim' in list(set(recorded_signals[0]) & set(recorded_signals[1])):
                IsVisStimSignal=1   
                
  #%% defining which kind of acquisition is              
        if 'NonImagingAcquisitions' in acquisition_path:
            IsNonImagingAcquisition=1  
            IsImaging=0
            imaging_aq_time=False  
            if imaging_path:
                slowstoragepath=os.path.join(mousesessionslowstoragepath,'nonimaging acquisitions',Prairieimagingname)   
                workingstoragepath=os.path.join(mousesessionworkingstoragepath,'nonimaging acquisitions',Prairieimagingname)  
            
            else:
                os.path.split(os.path.split(os.path.split(os.path.split(os.path.split(AcquisitonRawPath)[0])[0])[0])[0])[1]
                
                slowstoragepath=os.path.join(mousesessionslowstoragepath,'nonimaging acquisitions', 'Aq_1_NonImaging')   
                workingstoragepath=os.path.join(mousesessionworkingstoragepath,'nonimaging acquisitions','Aq_1_NonImaging')  

       
        if 'Atlas' in acquisition_path:
            IsAtlas=1    
            slowstoragepath=os.path.join(mousesessionslowstoragepath,'atlases',Prairieimagingname)   
            workingstoragepath=os.path.join(mousesessionworkingstoragepath,'atlases',Prairieimagingname)   
            
            if 'Overview' in acquisition_path:
                IsAtlasOverview=1   
                slowstoragepath=os.path.join(os.path.split(slowstoragepath)[0],'Overview',Prairieimagingname)   
                workingstoragepath=os.path.join(os.path.split(workingstoragepath)[0],'Overview',Prairieimagingname)    
            elif 'Preview' in acquisition_path:
                IsAtlasPreview=1   
                slowstoragepath=os.path.join(os.path.split(slowstoragepath)[0],'Preview',Prairieimagingname)   
                workingstoragepath=os.path.join(os.path.split(workingstoragepath)[0],'Preview',Prairieimagingname)     
            elif 'Volume' in acquisition_path:
                IsAtlasVolume=1   
                slowstoragepath=os.path.join(os.path.split(slowstoragepath)[0],'AtlasVolume',Prairieimagingname)   
                workingstoragepath=os.path.join(os.path.split(workingstoragepath)[0],'AtlasVolume',Prairieimagingname)     

        # OptogeneticsID=''
        if 'Calibrations' in acquisition_path:
            IsCalibration=1
            # to add geting sample ID
            
        if 'TestAcquisitions' in acquisition_path:
            IsTestAcquisition=1    
            slowstoragepath=os.path.join(mousesessionslowstoragepath,'test aquisitions',Prairieimagingname)   
            workingstoragepath=os.path.join(mousesessionworkingstoragepath,'test aquisitions',Prairieimagingname)   

        if '0CoordinateAcquisiton' in acquisition_path:
            Is0CoordinateAcquisiton=1  
            slowstoragepath=os.path.join(mousesessionslowstoragepath,'0Coordinate acquisition',Prairieimagingname)   
            workingstoragepath=os.path.join(mousesessionworkingstoragepath,'0Coordinate acquisition',Prairieimagingname)                
            
        if 'FOV_' in acquisition_path:
            acquisition_path[acquisition_path.find('FOV_'):acquisition_path.find('FOV_')+5]
            IsFOVAcquisition=1 
            slowstoragepath=os.path.join(mousesessionslowstoragepath,'data aquisitions',acquisition_path[acquisition_path.find('FOV_'):acquisition_path.find('FOV_')+5], Prairieimagingname)   
            workingstoragepath=os.path.join(mousesessionworkingstoragepath,'data aquisitions',acquisition_path[acquisition_path.find('FOV_'):acquisition_path.find('FOV_')+5], Prairieimagingname) 
            
            if 'SurfaceImage' in acquisition_path:
                IsSurfaceImage=1   
                slowstoragepath=os.path.join(os.path.split(slowstoragepath)[0],'SurfaceImage',Prairieimagingname)   
                workingstoragepath=os.path.join(os.path.split(workingstoragepath)[0],'SurfaceImage',Prairieimagingname)                       
            if '1050_Tomato'in acquisition_path:    
                IsTomatoRef=1   
                slowstoragepath=os.path.join(os.path.split(slowstoragepath)[0],'1050_Tomato',Prairieimagingname)   
                workingstoragepath=os.path.join(os.path.split(workingstoragepath)[0],'1050_Tomato',Prairieimagingname)                            
            if '1050_3PlaneTomato'in acquisition_path:  
                IsMultiplanetomatoref=1   
                slowstoragepath=os.path.join(os.path.split(slowstoragepath)[0],'1050_3PlaneTomato',Prairieimagingname)   
                workingstoragepath=os.path.join(os.path.split(workingstoragepath)[0],'1050_3PlaneTomato',Prairieimagingname)                                  
            if '1050_HighResStackTomato'in acquisition_path:  
                Ishighrestomatostack=1   
                slowstoragepath=os.path.join(os.path.split(slowstoragepath)[0],'1050_HighResStackTomato',Prairieimagingname)   
                workingstoragepath=os.path.join(os.path.split(workingstoragepath)[0],'1050_HighResStackTomato',Prairieimagingname)                                                  
            if 'HighResStackGreen' in acquisition_path:  
                Ishighresgreenstack=1   
                slowstoragepath=os.path.join(os.path.split(slowstoragepath)[0],'HighResStackGreen',Prairieimagingname)   
                workingstoragepath=os.path.join(os.path.split(workingstoragepath)[0],'HighResStackGreen',Prairieimagingname)                                                                               
            if 'OtherAcq' in acquisition_path:                
                Isotherfovaq=1   
                slowstoragepath=os.path.join(os.path.split(slowstoragepath)[0],'OtherAcq',Prairieimagingname)   
                workingstoragepath=os.path.join(os.path.split(workingstoragepath)[0],'OtherAcq',Prairieimagingname)         
 #%%              
        FaceCameraID=np.nan 
        if glob.glob( acquisition_path+'\\**\\DisplaySettings.json', recursive=False) :
           IsFaceCamera=1
           face_camera_path=os.path.join(acquisition_path,'FaceCamera')
           if face_camera_path:    
               FaceCameraID=self.add_face_camera(gui,acquisition_id, face_camera_path, slowstoragepath, workingstoragepath)    
               
               
        VisStimulationID=np.nan       
        if glob.glob( acquisition_path+'\\VisStim\\**', recursive=False) :
           
            VisStim_path=os.path.join(acquisition_path,'VisStim')
            IsBehaviour=1        
            IsVisualStimulation=1
            VisStimLog=glob.glob( VisStim_path+ '\\**.mat', recursive=False)[0]
            if VisStimLog:
                VisStimulationID=self.add_new_visual_stimulation(acquisition_id, VisStim_path, VisStimLog,slowstoragepath, workingstoragepath)
        
        if imaging_aq_time:
            AqTime=imaging_aq_time
        elif voltage_aq_time:
            AqTime=voltage_aq_time
        else:
            AqTime=''    
            
        ImagingID=np.nan    
        if imaging_path:
            if IsImaging:
              ImagingID=self.add_new_imaging(gui,acquisition_id, imaging_path, metadata, slowstoragepath, workingstoragepath )    
            else:
              ImagingID=self.add_new_imaging(gui,acquisition_id, imaging_path, metadata, slowstoragepath, workingstoragepath, locomotion_only=True )    
   
            
            
        query_add_acquisition="""
                    INSERT INTO Acquisitions_table(
                        ID,
                        IsMouse,
                        ImagedMouseID,
                        AcquisitonRawPath,
                        IsCalibration,
                        IsTestAcquisition,
                        IsNonImagingAcquisition,
                        Is0CoordinateAcquisiton,
                        IsFOVAcquisition,
                        IsSurfaceImage,
                        IsTomatoRef,
                        IsMultiplanetomatoref,
                        Ishighrestomatostack,
                        Ishighresgreenstack,
                        Isotherfovaq,
                        IsAtlas,
                        IsAtlasPreview,
                        IsAtlasOverview,
                        IsAtlasVolume,                      
                        AcquisitonNumber,
                        AqTime,
                        IsImaging,
                        IsFaceCamera,
                        IsLocomotion,
                        IsPhotodiode,
                        IsVisStimSignal,
                        IsBehaviour,
                        IsVisualStimulation,
                        IsOptogenetic,
                        ImagingID,
                        VisStimulationID,
                        FaceCameraID, 
                        Comments,
                        IsSlowDisk,                       
                        IsWorkingDisk,
                        SlowDiskPath,
                        WorkingDiskPath,
                        IsGoodName
                        
                        )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """    
        params=(acquisition_id,
                IsMouse,
                ImagedMouseID,
                AcquisitonRawPath,
                IsCalibration,
                IsTestAcquisition,
                IsNonImagingAcquisition,
                Is0CoordinateAcquisiton,
                IsFOVAcquisition,
                IsSurfaceImage,
                IsTomatoRef,
                IsMultiplanetomatoref,
                Ishighrestomatostack,
                Ishighresgreenstack,
                Isotherfovaq,
                IsAtlas,
                IsAtlasPreview,
                IsAtlasOverview,
                IsAtlasVolume,
                AcquisitionNumber,
                AqTime,
                IsImaging,
                IsFaceCamera,
                IsLocomotion,
                IsPhotodiode,
                IsVisStimSignal,
                IsBehaviour,
                IsVisualStimulation,
                IsOptogenetic,
                ImagingID,
                VisStimulationID,
                FaceCameraID,
                Comments,
                0,
                0,
                slowstoragepath,
                workingstoragepath,
                IsGoodName
                )


        self.databse_ref.arbitrary_inserting_record(query_add_acquisition, params ) 
        
    def add_new_imaging(self,gui, acquisition_ID, imaging_path, metadata_object, aqslowstoragepath, aqworkingstoragepath, locomotion_only=False):
        self.update_variables()
        ImagingID=self.max_imaging_id+1
        metadata=metadata_object
        AcquisitionID=acquisition_ID  
        ImagingFullFilePath=imaging_path
        ImagingFilename=os.path.split(imaging_path)[1]
                
        PowerSetting=np.nan
        Objective=np.nan
        PMT1GainRed=np.nan
        PMT2GainGreen=np.nan
        FrameAveraging=np.nan
        ObjectivePositions=np.nan
        ETLPositions=np.nan
        PlaneNumber=np.nan
        TotalVolumes=np.nan
        IsETLStack=np.nan
        IsObjectiveStack=np.nan
        InterFramePeriod=np.nan
        FinalVolumePeriod=np.nan
        FinalFrequency=np.nan
        TotalFrames=np.nan
        FOVNumber=np.nan
        ExcitationWavelength=np.nan
        CoherentPower=np.nan
        CalculatedPower=np.nan
        Comments=np.nan
        IsChannel1Red=np.nan
        IsChannel2Green=np.nan
        IsGalvo=np.nan
        IsResonant=np.nan   
        Resolution=np.nan
        DwellTime=np.nan
        Multisampling=np.nan
        BitDepth=np.nan
        LinePeriod=np.nan
        FramePeriod=np.nan
        FullAcquisitionTime=np.nan    
        RedFilter=np.nan
        GreenFilter=np.nan
        DichroicBeamsplitter=np.nan
        IsBlockingDichroic=np.nan
        OverlapPercentage=np.nan
        AtlasOverlap=np.nan
        OverlapPercentageMetadata=np.nan
        AtlasDirection=np.nan
        AtlasZStructure=np.nan
        AtlasGridSize=np.nan
        IsGoodObjective=1
        correctedObjectivePositions=np.nan
        correctedETLPositions=np.nan
        Is10MinRec=np.nan
        
        
      
       
        if not locomotion_only:
            PowerSetting=metadata.imaging_metadata[1]['Planepowers']    
            Objective=metadata.imaging_metadata[0]['Objective']       
            PMT1GainRed=metadata.imaging_metadata[1]['pmtGains_Red']
            PMT2GainGreen=metadata.imaging_metadata[1]['pmtGains_Green']
            FrameAveraging=metadata.imaging_metadata[0]['RasterAveraging']   
            ObjectivePositions=metadata.imaging_metadata[1]['PlanePositionsOBJ']
            ETLPositions=metadata.imaging_metadata[1]['PlanePositionsETL'] 
            Xpositions=metadata.imaging_metadata[1]['XPositions'] 
            Ypositions=metadata.imaging_metadata[1]['YPositions'] 
            ImagingTime=metadata.imaging_metadata[0]['ImagingTime'] 
            MicronsPerPixelX=metadata.imaging_metadata[0]['MicronsPerPixelX'] 
            MicronsPerPixelY=metadata.imaging_metadata[0]['MicronsPerPixelY'] 
            Zoom=metadata.imaging_metadata[0]['OpticalZoom'] 
            correctedObjectivePositions=metadata.imaging_metadata[1]['PlanePositionsOBJ']
            correctedETLPositions=metadata.imaging_metadata[1]['PlanePositionsETL'] 


            if metadata.imaging_metadata[1]['PlaneNumber']=='Single':
                    IsETLStack=0
                    IsObjectiveStack=0
                    PlaneNumber=1
                    TotalFrames=metadata.imaging_metadata[1]['FrameNumber']
                    InterFramePeriod=metadata.imaging_metadata[0]['framePeriod']*FrameAveraging
                    FinalVolumePeriod=InterFramePeriod
                    FinalFrequency=1/InterFramePeriod
                    TotalVolumes=TotalFrames
            else:
                TotalVolumes=metadata.imaging_metadata[1]['VolumeNumber']
                IsETLStack=0
                IsObjectiveStack=0
                PlaneNumber=metadata.imaging_metadata[1]['PlaneNumber']
               
                InterFramePeriod=metadata.imaging_metadata[0]['framePeriod']
                if not isinstance(metadata.imaging_metadata[2][0][list(metadata.imaging_metadata[2][0].keys())[0]]['framePeriod'], str):
                    FinalVolumePeriod=metadata.imaging_metadata[2][0][list(metadata.imaging_metadata[2][0].keys())[0]]['framePeriod']*PlaneNumber
                else:
                    FinalVolumePeriod= metadata.imaging_metadata[0]['framePeriod']*PlaneNumber
                    
                FinalFrequency=1/FinalVolumePeriod
                TotalFrames=TotalVolumes*PlaneNumber
                PowerSetting=str(PowerSetting)
                correctedObjectivePositions=[float(i[8:]) if isinstance(i, str) else i for i in ObjectivePositions]
                correctedETLPositions=[float(i[8:]) if isinstance(i, str) else i for i in ETLPositions]
                if not all(element == correctedObjectivePositions[0] for element in correctedObjectivePositions):
                    IsObjectiveStack=1
                if not all(element == correctedETLPositions[0] for element in correctedETLPositions):
                    IsETLStack=1

            FOVNumber=np.nan
            if 'FOV_' in imaging_path:            
                 FOVNumber=imaging_path[ imaging_path.index('FOV_')+4]
                 
                    
                 
            self.add_imaging_info_window=AddImagingInfo(gui, ImagingFilename)
            self.add_imaging_info_window.wait_window()
            get_values= self.add_imaging_info_window.values
            # root = Tkinter.Tk()
            # app = add_imaging_info(root, ImagingFilename)
            # root.mainloop()
            # get_values=app.values
            RedFilter=get_values[2][1]
            GreenFilter=get_values[1][1]
            DichroicBeamsplitter=get_values[3][1]
            filtervalues=[RedFilter,GreenFilter,DichroicBeamsplitter]
            filtercodes=transform_filterinfo_to_codes(filtervalues, self.databse_ref)
            
            

            IsBlockingDichroic=0   
            if 'Yes' in get_values[4][1]:
                IsBlockingDichroic=1
                
            if 'No' in get_values[5][1]:
                IsGoodObjective=1

            ExcitationWavelength=get_values[6][1]
            CoherentPower=get_values[7][1]
            CalculatedPower=np.nan
            Comments=get_values[8][1]
            
            ToDoDeepCaiman=0
            if 'Yes' in get_values[12][1]:
                ToDoDeepCaiman=1

            
            Xpositions=metadata.imaging_metadata[1]['XPositions']
            Ypositions=metadata.imaging_metadata[1]['YPositions']
            ImagingTime=metadata.imaging_metadata[0]['ImagingTime']
            MicronsPerPixelX=metadata.imaging_metadata[0]['MicronsPerPixelX']
            MicronsPerPixelY=metadata.imaging_metadata[0]['MicronsPerPixelY']
            Zoom=metadata.imaging_metadata[0]['OpticalZoom']
            
            
            if 'Atlas' in  metadata.imaging_metadata[1]['MultiplanePrompt']:
                OverlapPercentage=get_values[9][1]
                AtlasZStructure=get_values[10][1]
                AtlasDirection=get_values[11][1]
                AtlasOverlap=str((metadata.imaging_metadata[1]['StageGridXOverlap'],  metadata.imaging_metadata[1]['StageGridYOverlap'] ))
                OverlapPercentageMetadata= metadata.imaging_metadata[1]['StageGridOverlapPercentage']
                AtlasGridSize=str((metadata.imaging_metadata[1]['StageGridNumXPositions'],  metadata.imaging_metadata[1]['StageGridNumYPositions']))
                Xpositions=str(tuple(Xpositions))
                Ypositions=str(tuple(Ypositions))
               

            IsChannel1Red=0
            IsChannel2Green=0
    
            if not metadata.imaging_metadata[1]['RedChannelName']=='No Channel':
                IsChannel1Red=1
            if not metadata.imaging_metadata[1]['GreenChannelName']=='No Channel':
                IsChannel2Green=1
            IsGalvo=1
            IsResonant=0
            if 'Resonant' in  metadata.imaging_metadata[0]['ScanMode']:
                 IsResonant=1
                 IsGalvo=0
                 Multisampling=metadata.imaging_metadata[0]['ResonantSampling']
            
            Resolution=str(metadata.imaging_metadata[0]['LinesPerFrame'])+'x'+ str(metadata.imaging_metadata[0]['PixelsPerLine'])
            DwellTime=metadata.imaging_metadata[0]['dwellTime']
            
            BitDepth=metadata.imaging_metadata[0]['BitDepth']
                
            LinePeriod=metadata.imaging_metadata[0]['ScanLinePeriod']
            FramePeriod=metadata.imaging_metadata[0]['framePeriod']
            FullAcquisitionTime=metadata.imaging_metadata[1]['FullAcquisitionTime']

   
        IsVoltageRecording=0
        VoltageRecordingChannels=np.nan
        VoltageRecordingFrequency=np.nan
        if hasattr(metadata, 'full_voltage_recording_metadata'):
            IsVoltageRecording=1
            VoltageRecordingChannels=str((metadata.recorded_signals and metadata_object.recorded_signals_csv))
            VoltageRecordingFrequency=metadata.translated_imaging_metadata['VoltageRecordingFrequency']

        CaimanComments=None
    
        query_add_imaging="""
                INSERT INTO Imaging_table(
                    ID,
                    AcquisitionID,
                    ImagingFullFilePath,
                    ImagingFilename,
                    RedFilter,
                    GreenFilter,
                    DichroicBeamsplitter,
                    IsBlockingDichroic,
                    FOVNumber,
                    IsETLStack,
                    Is10MinRec,
                    ToDoDeepCaiman,
                    IsObjectiveStack,
                    PlaneNumber,
                    Objective,
                    ObjectivePositions,
                    ETLPositions,
                    PMT1GainRed,
                    PMT2GainGreen,
                    IsChannel1Red,
                    IsChannel2Green,
                    ExcitationWavelength,
                    CoherentPower,
                    PowerSetting,
                    CalculatedPower,
                    IsGalvo,
                    IsResonant,
                    Resolution,
                    DwellTime,
                    Multisampling,
                    BitDepth,
                    FrameAveraging,
                    LinePeriod,
                    FramePeriod,
                    InterFramePeriod,
                    FinalVolumePeriod,
                    FinalFrequency,
                    FullAcquisitionTime,
                    TotalFrames,
                    TotalVolumes,
                    IsVoltageRecording,
                    VoltageRecordingChannels,
                    VoltageRecordingFrequency,
                    Comments,
                    SlowStoragePath,
                    WorkingStoragePath,
                    IsWorkingStorage,
                    IsSlowStorage,
                    OverlapPercentage,
                    AtlasOverlap,
                    AtlasDirection,
                    AtlasZStructure,
                    AtlasGridSize,
                    OverlapPercentageMetadata,
                    Xpositions,
                    Ypositions,
                    ImagingTime,
                    MicronsPerPixelX,
                    MicronsPerPixelY,
                    Zoom,
                    IsGoodObjective,
                    CaimanComments
                    )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """    
        params=(ImagingID,
                AcquisitionID,
                ImagingFullFilePath,
                ImagingFilename,
                int(filtercodes[0]),
                int(filtercodes[1]),
                int(filtercodes[2]),
                IsBlockingDichroic,
                FOVNumber,
                IsETLStack,
                Is10MinRec,
                ToDoDeepCaiman,
                IsObjectiveStack,
                PlaneNumber,
                Objective,
                str(correctedObjectivePositions),
                str(correctedETLPositions),
                PMT1GainRed,
                PMT2GainGreen,
                IsChannel1Red,
                IsChannel2Green,
                ExcitationWavelength,
                CoherentPower,
                PowerSetting,
                CalculatedPower,
                IsGalvo,
                IsResonant,
                Resolution,
                DwellTime,
                Multisampling,
                BitDepth,
                FrameAveraging,
                LinePeriod,
                FramePeriod,
                InterFramePeriod,
                FinalVolumePeriod,
                FinalFrequency,
                FullAcquisitionTime,
                TotalFrames,
                TotalVolumes,
                IsVoltageRecording,
                VoltageRecordingChannels,
                VoltageRecordingFrequency,
                Comments,
                aqslowstoragepath,
                aqworkingstoragepath,
                0,
                1,
                OverlapPercentage,
                AtlasOverlap,
                AtlasDirection,
                AtlasZStructure,
                AtlasGridSize,
                OverlapPercentageMetadata,
                Xpositions,
                Ypositions,
                ImagingTime,
                MicronsPerPixelX,
                MicronsPerPixelY,
                Zoom,
                IsGoodObjective,
                CaimanComments
                )
        self.databse_ref.arbitrary_inserting_record(query_add_imaging, params )   
        return ImagingID
    
    def add_face_camera(self,gui, acquisition_ID, face_camera_path, aqslowstoragepath, aqworkingstoragepath, ):
        self.update_variables()
        FaceCameraID= self.max_facecamera_id+1
        AcquisitionID=acquisition_ID
        VideoPath=face_camera_path
        acq_name=os.path.split(aqslowstoragepath)[1]
        EyeCameraFilename_processed=acq_name+'_full_face_camera.tiff'          
        # EyeCameraFilename= os.path.split(glob.glob( VideoPath+'\\**.tif', recursive=False)[0])[1]
        slowstoragepath=os.path.join(aqslowstoragepath,'eye camera', EyeCameraFilename_processed)   
        workingstoragepath=os.path.join(aqworkingstoragepath,'eye camera', EyeCameraFilename_processed)  

        Exposure=33.3
        Frequency=1/33.3
        Resolution='640x480'
        BitDepth=8
        VideoFormat='.tif'
              
        self.add_face_camera_info_window=AddFacecameraInfo(gui, face_camera_path)
        self.add_face_camera_info_window.wait_window()
        get_values= self.add_face_camera_info_window.values
        # root = Tkinter.Tk()
        # app = add_facecamera_info(root, face_camera_path)
        # root.mainloop()
        # get_values=app.values
        

        
        IsIRlight=get_values[1][1]
        IRLightPosition=get_values[2][1]
        CameraPosition=get_values[3][1]
        SideImaged= get_values[4][1]   
        SynchronizeMethods=get_values[5][1]
        Comments=get_values[6][1]
        
        query_add_facecamera="""
                    INSERT INTO FaceCamera_table(
                            AcquisitionID,
                            VideoPath,
                            EyeCameraFilename,
                            Exposure,
                            Frequency,
                            Resolution,
                            BitDepth,
                            IsIRlight,
                            IRLightPosition,
                            CameraPosition,
                            SideImaged,
                            VideoFormat,
                            SynchronizeMethods,
                            Comments,
                            IsSlowStorage,
                            SlowStoragePath,
                            IsWorkingStorage,
                            WorkingStoragePath
                            
                        )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """    
        params=(AcquisitionID,
                VideoPath,
                EyeCameraFilename_processed,
                Exposure,
                Frequency,
                Resolution,
                BitDepth,
                IsIRlight,
                IRLightPosition,
                CameraPosition,
                SideImaged,
                VideoFormat,
                SynchronizeMethods,
                Comments,
                    0,
                   slowstoragepath ,
                    0,
                    workingstoragepath
                )
                
        self.databse_ref.arbitrary_inserting_record(query_add_facecamera, params, )       
        
        return FaceCameraID
            
    def add_new_visual_stimulation(self, acquisition_ID, VIsStimpath, VisStimLog, aqslowstoragepath, aqworkingstoragepath):
        self.update_variables()
        VisStimulationID=self.max_visstim_id+1
        AcquisitionID=acquisition_ID
        # MatLog=scp.io.loadmat(VisStimLog)
        IsMATLAB=1
        IsPython=0
        
       
        
        VisStimLogPath=os.path.split(VisStimLog)[0]
        VisStimLogName=os.path.split(VisStimLog)[1]
        slowstoragepath=os.path.join(aqslowstoragepath,'visual stim', VisStimLogName)   
        workingstoragepath=os.path.join(aqworkingstoragepath,'visual stim', VisStimLogName)  
        IsWithSignal=1
        SynchronizeMethods='Signal and photodiode'
        if 'Habituation' in  VisStimLog :
             Behaviour=3
             Treatment='Habituation'
             if ('Day1' or 'Day7') in VisStimLog:
                 VisStimSequence='10minspont+(30sISI+100sSTIM SINGLE GRATING)*5trials+80*(5IS+5STIM)'
             else:
                 VisStimSequence='10minspont+(30sISI+100sSTIM SINGLE GRATING)'
                 
        elif 'Control' in VisStimLog:
             Behaviour=3
             Treatment='Control'
             if ('Day1' or 'Day7') in VisStimLog:
                 VisStimSequence='10minspont+(30sISI+100sSTIM SINGLE GRATING)*5trials+80*(5IS+5STIM)'
             else:
                 VisStimSequence='10minspont+(30sISI+100sSTIM RANDOM GRATING)'
        
        elif 'Allen' in VisStimLogName:
            Behaviour=5
            Treatment='SessionA'
            VisStimSequence='10mingratings+5xmovie3+10xmovie1+10mingrating+5minspont+5xmovie3+10mingrating'
        else:
            Behaviour=4
            Treatment='Mistmatch'
            VisStimSequence=''
        
        query_add_visstim="""
                    INSERT INTO VisualStimulations_table(
                                AcquisitionID,
                                Behaviour,
                                Treatment,
                                VisStimSequence,
                                IsMATLAB,
                                IsPython,
                                VisStimLogPath,
                                VisStimLogName,
                                IsWithSignal,
                                SynchronizeMethods,
                            IsSlowStorage,
                            SlowStoragePath,
                            IsWorkingStorage,
                            WorkingStoragePath
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """    
        
        params=(AcquisitionID,
                    Behaviour,
                    Treatment,
                    VisStimSequence,
                    IsMATLAB,
                    IsPython,
                    VisStimLogPath,
                    VisStimLogName,
                    IsWithSignal,
                    SynchronizeMethods,
                    0,
                    slowstoragepath,
                    0,
                    workingstoragepath
                    )
                
        self.databse_ref.arbitrary_inserting_record(query_add_visstim, params, )       
        return VisStimulationID
        
        
        
        
 #%% Getting Last IDs
        
        
    def get_last_imagedmice_id(self):
        c=self.databse_ref.database_connection.cursor()
        query_max_code="SELECT MAX(ID) FROM ImagedMice_table"
        c.execute(query_max_code)
        max_current_code=c.fetchall()
        self.max_imagedmice_id=max_current_code[0][0]
        
    def get_last_widefield_id(self):
        c=self.databse_ref.database_connection.cursor()
        query_max_code="SELECT MAX(ID) FROM WideField_table"
        c.execute(query_max_code)
        max_current_code=c.fetchall()
        self.max_widefield_id=max_current_code[0][0]   

    def get_last_imagingsession_id(self):
        c=self.databse_ref.database_connection.cursor()
        query_max_code="SELECT MAX(ID) FROM ImagingSessions_table"
        c.execute(query_max_code)
        max_current_code=c.fetchall()
        self.max_imagingsession_id=max_current_code[0][0]     
        
    def get_last_acquisition_id(self):
        c=self.databse_ref.database_connection.cursor()
        query_max_code="SELECT MAX(ID) FROM Acquisitions_table"
        c.execute(query_max_code)
        max_current_code=c.fetchall()
        self.max_acquisition_id=max_current_code[0][0]        

    def get_last_imaging_id(self):
        c=self.databse_ref.database_connection.cursor()
        query_max_code="SELECT MAX(ID) FROM Imaging_table"
        c.execute(query_max_code)
        max_current_code=c.fetchall()
        self.max_imaging_id=max_current_code[0][0]      
        
    def get_last_vistim_id(self):
        c=self.databse_ref.database_connection.cursor()
        query_max_code="SELECT MAX(ID) FROM VisualStimulations_table"
        c.execute(query_max_code)
        max_current_code=c.fetchall()
        self.max_visstim_id=max_current_code[0][0]                  
                   
    def get_last_facecamera_id(self):
        c=self.databse_ref.database_connection.cursor()
        query_max_code="SELECT MAX(ID) FROM FaceCamera_table"
        c.execute(query_max_code)
        max_current_code=c.fetchall()
        self.max_facecamera_id=max_current_code[0][0]                  
       

#%% update all storages


