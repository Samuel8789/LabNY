# -*- coding: utf-8 -*-
"""
Created on Tue May 11 08:43:30 2021

@author: sp3660

Database full calss
"""

import tkinter as Tkinter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sqlite3
import datetime
import os
from shutil import copyfile
import glob
from pathlib import Path
import logging 
module_logger = logging.getLogger(__name__)
from ..AllFunctions.select_values_gui import select_values_gui
from .fun.guiFunctions.updateLitterInput import UpdateLitterInput
from .fun.guiFunctions.input_genotypes import input_genotypes
from ..gui.tabs.mouseVisit.new_window_multilitter import new_window_multilitter

from .experimentalDatabase import ExperimentalDatabase
from .imagingDatabase import ImagingDatabase

#%%
class MouseDatabase():
    def __init__(self,database_file_path, project_object=False):
        self.LabProjectObject=project_object
        self.database_file_path=database_file_path
        self.database_connection=sqlite3.connect(self.database_file_path)
        self.database_backup()
        print('Loading Experimental Database')
        self.Experimental_class=ExperimentalDatabase(self)
        print('Loading Imaging Database')
        self.ImagingDatabase_class=ImagingDatabase(self)
        print('Updating Databases')
        self.update_variables()
        
#%%        general and tables
    def update_variables(self):
        self.table_all_alive_non_exp()
        self.get_all_tables()
        self.table_all_alive_nonexp_byline()
        self.get_actions()
        self.table_stock_cages()
        self.table_to_genotype()
        self.table_breeders()
        self.get_max_cage()
        self.get_max_mouse_code()    
        self.all_litters=self.get_litters(True)
        self.current_litters=self.get_litters(False)
        self.Experimental_class.update_variables()
        self.ImagingDatabase_class.update_variables()
        self.full_tables()
        self.get_cage_number()    

    def close_database(self):
        self.database_connection.close()
    def reconnect_database(self):
        self.database_connection=sqlite3.connect(self.database_file_path)
        self.database_backup()
        print('Loading Experimental Database')
        self.Experimental_class=ExperimentalDatabase(self)
        print('Loading Imaging Database')
        self.ImagingDatabase_class=ImagingDatabase(self)
        print('Updating Databases')
        self.update_variables()
        
        
    def mouse_previsit(self, new_visit=False):       
        today_date=datetime.date.today().strftime("%Y%m%d")
        

        
        file_path=Path(self.LabProjectObject.all_paths_for_this_system['Documents'] ,'LabNY' ,'4. Mouse Managing' ,'MouseVisits', today_date)

        if not os.path.isdir(file_path):
            os.makedirs(file_path)
          
        file_path2= file_path / 'PreVisit_'
        self.max_codes = pd.DataFrame([[self.max_current_cage,self.max_current_code]], columns = ['MaxCage', 'MaxCode'])
        pth=file_path / 'PreVisit_MouseVisit.pdf'
        if new_visit:
            file_path2=file_path / 'PreVisit2_'
            pth=file_path / 'PreVisit2_MouseVisit.pdf'
            # pth=os.path.join(file_path ,'PreVisit2_MouseVisit.html')
            # pth2=os.path.join(file_path ,'PreVisit2_MouseVisit.pdf')

        all_dfs=[ self.breedings, self.stock_mice, self.mice_to_genotype, self.max_codes, self.current_litters]
        # with pd.ExcelWriter(pth,engine='xlsxwriter') as writer:
        #     for i, df in enumerate(all_dfs):
        #             df.to_excel(writer,sheet_name='MouseVisit',startrow=i*60+1 , startcol=0)
        # self.all_colony_mice.to_excel(os.path.join(file_path2 + 'AllMice.xlsx'))
        # self.current_litters.to_excel(os.path.join(file_path2 + 'Litters.xlsx'))
        
        rows_per_page=30
        pages=np.ceil(self.all_colony_mice.shape[0]/rows_per_page)
        
        fragments=[]
        for i in range(int(pages)):
            start=(i*rows_per_page)+1
            end=start+rows_per_page
            if i==0:
                start=0
                end=start+rows_per_page+1
            if i==int(pages)-1:
                end=self.all_colony_mice.shape[0]+1
            fragment=self.all_colony_mice.iloc[start:end,:]
            fragment2=fragment.fillna(False)
            fig, ax =plt.subplots(figsize=(12,4))
            ax.axis('tight')
            ax.axis('off')
            
            
            if not fragment2.empty:
                the_table = ax.table(cellText=fragment2.values, colLabels=fragment2.columns,loc='center')

            else:
                fragment2=fragment2.reindex(list(range(0, 1))).reset_index(drop=True)
                the_table = ax.table(cellText=fragment2.values, colLabels=fragment2.columns,loc='center')


            
            cells = the_table.properties()["celld"]
            lastecell=sorted(list(cells.keys()))[-1]

            for k in range (0,lastecell[0]+1):
                for j in range(0, lastecell[1]+1):
                    cells[k, j].set_text_props(ha="center")
                    
            fragments.append(fig)
            
            pp = PdfPages(str(file_path2) + 'AllMice.pdf')
            
            for fig in fragments: ## will open an empty extra figure :(
                pp.savefig( fig , bbox_inches='tight')
            pp.close()

        figs=[]
        for i, df in enumerate(all_dfs):
            if i in [0,4]:
                df.fillna(0, inplace=True)
                df['Notes'] = pd.Series(dtype='int')
                df.fillna(0, inplace=True)
            else:
                df.fillna(0, inplace=True)
                
            fig, ax =plt.subplots(figsize=(12,4))
            ax.axis('tight')
            ax.axis('off')
            if not df.empty:
                the_table = ax.table(cellText=df.values, colLabels=df.columns,loc='center')
            else:
                df=df.reindex(list(range(0, 1))).reset_index(drop=True)
                the_table = ax.table(cellText=df.values, colLabels=df.columns,loc='center')

            cells = the_table.properties()["celld"]
            lastecell=sorted(list(cells.keys()))[-1]

            for k in range (0,lastecell[0]+1):
                for j in range(0, lastecell[1]+1):
                    cells[k, j].set_text_props(ha="center")
            if i in [0,4]:
                the_table.scale(1, 3)
                for k in range (0,lastecell[0]+1):
                    cells[k,lastecell[1]].set_width(0.8)
                
                
            figs.append(fig)   
        
        pp2 = PdfPages(pth)
        
        for fig in figs: ## will open an empty extra figure :(
            pp2.savefig( fig , bbox_inches='tight')
        pp2.close()
        

    def mouse_postvisit(self,date_performed=False,new_visit=False):   
        
        if not date_performed:
            today_date=datetime.date.today().strftime("%Y%m%d")
        else:
            today_date=date_performed
            
        file_path=Path(self.LabProjectObject.all_paths_for_this_system['Documents'] ,'LabNY' ,'4. Mouse Managing' ,'MouseVisits', today_date)
        file_path2=file_path / 'PostVisit_'       
        self.visit_actions=self.actions[self.actions['Date'].str.contains(datetime.date.today().strftime("%Y-%m-%d"))]
        all_dfs=[ self.breedings,self.current_litters,self.stock_mice,self.mice_to_genotype,self.visit_actions]
        pth=os.path.join(file_path ,'PostVisit_MouseVisit.xlsx')
        if new_visit:
            file_path2=file_path / 'PostVisit2_'    
            pth=os.path.join(file_path ,'PostVisit2_MouseVisit.xlsx')
        
        with pd.ExcelWriter(pth,engine='xlsxwriter') as writer:
            for i, df in enumerate(all_dfs):
                if i==0:
                    df.to_excel(writer,sheet_name='MouseVisit',startrow=0 , startcol=0)
                    lastrow=len(df)
                if i>0:
                    df.to_excel(writer,sheet_name='MouseVisit',startrow=lastrow+3 , startcol=0)
                    lastrow=lastrow+3+len(df)    
        self.all_colony_mice.to_excel(str(file_path2) +'AllMice.xlsx')    
        
    def database_backup(self):
        
        backuppath=Path(self.LabProjectObject.all_paths_for_this_system['Documents'] ,'LabNY' ,'4. Mouse Managing' ,'DatabaseBackups')
        backuppath_dropbox=Path(self.LabProjectObject.all_paths_for_this_system['Dropbox'],'LabNY' ,'DatabaseBackups')
        backuppath_F=''
        if hasattr(self.LabProjectObject, 'data_paths_project'):
            backuppath_F=Path(self.LabProjectObject.data_paths_project['Raw'],'DatabaseBackups')
        
        
        
        # if self.LabProjectObject.platform=='win32':
        #     list_of_backups = glob.glob(backuppath_dropbox+os.sep+'*') # * means all if need specific format then *.csv
        # elif self.LabProjectObject.platform=='linux':
        list_of_backups = glob.glob(str(backuppath_dropbox / '**')) # * means all if need specific format then *.csv

        latest_file = max(list_of_backups, key=os.path.getctime)
        
        
        self.last_backup_date=datetime.datetime.strptime(latest_file[-11:-3], '%Y%m%d')

        today_date=datetime.date.today().strftime("%Y%m%d")
        now  = datetime.datetime.now()                         
        duration = now - self.last_backup_date                    
        if duration.days >=7:
            dst=os.path.join(backuppath, 'MouseDatabase_Backup_{date}'.format(date=today_date)+'.db')
            dst_dropbox=os.path.join(backuppath_dropbox, 'MouseDatabase_Backup_{date}'.format(date=today_date)+'.db')
            dst_F=os.path.join(backuppath_F, 'MouseDatabase_Backup_{date}'.format(date=today_date)+'.db')
            if os.path.isdir(backuppath_F):
                copyfile(self.database_file_path, dst)
                copyfile(self.database_file_path, dst_F)
                assert os.path.isfile(dst)
                assert os.path.isfile(dst_F)

            copyfile(self.database_file_path, dst_dropbox)
            assert os.path.isfile(dst_dropbox)
            self.last_backup_date=datetime.datetime.now() 
            print('Backup done')
        else:
            print('RecentBackup')
            
    def full_export_toexcell(self):
        print('in progress')
               
    def get_all_tables (self) :
        c=self.database_connection.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.all_tables=c.fetchall()

    def get_actions(self):
        action_check_query="""
            SELECT 
                a.ID,
                a.Date,
                g.Action_types,  
                a.Cage_start,
                a.Cage_end,
                b.Lab_Number,
                c.Lab_Number,
                d.Lab_Number,
                e.Lab_Number,
                f.Lab_Number
            FROM Actions_table a 
            LEFT JOIN MICE_table b ON b.ID=a.Mouse_1 
            LEFT JOIN MICE_table c ON c.ID=a.Mouse_2 
            LEFT JOIN MICE_table d ON d.ID=a.Mouse_3 
            LEFT JOIN MICE_table e ON e.ID=a.Mouse_4 
            LEFT JOIN MICE_table f ON f.ID=a.Mouse_5 
            LEFT JOIN Action_types_table g ON g.ID=a.Action_type           
            """
        action_check=self.arbitrary_query_to_df(action_check_query)
        self.actions=action_check.sort_values(by=['ID'],ascending=False)   

    def get_litters(self, include_weaned=False):       
        if include_weaned:
            query_litters="""
                SELECT 
                    a.*,
                    date(julianday(Date_Seen)-Age)  AS DOB, 
                    round(julianday('now') - (julianday(Date_Seen)-Age)) AS DaysOld
                FROM Litters_table  a                 
                LEFT JOIN Breedings_table b ON b.ID=a.Breeding_Parents               
                WHERE b.Cage IS NOT NULL
            """
        else:    
            query_litters="""
                SELECT  
                    a.ID,
                    a.Cage,
                    a.NumberAlive,
                    a.NumberDead,                    
                    round(julianday('now') - (julianday(Date_Seen)-Age)) AS DaysOld
                FROM Litters_table a
                LEFT JOIN Breedings_table b ON b.ID=a.Breeding_Parents
                WHERE b.Cage IS NOT NULL AND Date_Weaned IS NULL
            """
        selected_litters=self.arbitrary_query_to_df(query_litters)
        return selected_litters.sort_values(by=['Cage'],ascending=True)       
    
    def table_stock_cages(self):
        query_stock="""
        SELECT 
            Lab_number,
            Sex_types AS Sex, 
            Cage, 
            round(julianday('now') - julianday(DOB)) AS DaysOld,
            round(round(julianday('now') - julianday(DOB))/7) AS WeeksOld,
            Breeders_types,
            Labels_types, 
            Line_Short, 
            Genotypings_types       
        FROM MICE_table 
        LEFT JOIN Breeders_table ON Breeders_table.ID=MICE_table.Breeding_status
        LEFT JOIN Labels_table ON Labels_table.ID=MICE_table.Label
        LEFT JOIN Sex_table ON Sex_table.ID=MICE_table.Sex
        LEFT JOIN Experimental_table ON Experimental_table.ID=MICE_table.Experimental_Status
        LEFT JOIN Genotypings_table ON Genotypings_table.ID=MICE_table.Genotyping_Status
        LEFT JOIN Lines_table ON Lines_table.ID=MICE_table.Line
        LEFT JOIN Rooms_table ON Rooms_table.ID=MICE_table.Room     
        WHERE Alive=1 AND (Experimental_Status=1 OR Experimental_Status=3)  AND Breeding_status=3   AND Room=2     
        """
        Alive_non_exp_stock =self.arbitrary_query_to_df(query_stock)
        Alive_non_exp_stock_sorted= Alive_non_exp_stock.sort_values(by=['Sex','Lab_Number'],ascending=True)       
        Alive_non_exp_stock_grouped = Alive_non_exp_stock_sorted.groupby(['Breeders_types', 'Cage', 'WeeksOld','Line_Short', 'Sex' ])['Lab_Number'].apply(list)
        Alive_non_exp_stock_grouped_frame=Alive_non_exp_stock_grouped.to_frame()
        self.stock_mice=pd.DataFrame(Alive_non_exp_stock_grouped_frame.Lab_Number.values.tolist(), Alive_non_exp_stock_grouped_frame.index).add_prefix('Mouse_').astype('Int64')
        self.stock_mice.reset_index(inplace=True)
        self.stock_mice.drop(['Breeders_types'],axis=1, inplace=True)
        
    def table_to_genotype(self):
        query_to_genotype="""
        SELECT 
            Lab_number,
            Sex_types, 
            Cage, 
            round(julianday('now') - julianday(DOB)) AS DaysOld,
            round(round(julianday('now') - julianday(DOB))/7) AS WeeksOld,
            Breeders_types,
            Labels_types, 
            Line_Short, 
            Genotypings_types        
        FROM MICE_table 
        LEFT JOIN Breeders_table ON Breeders_table.ID=MICE_table.Breeding_status
        LEFT JOIN Labels_table ON Labels_table.ID=MICE_table.Label
        LEFT JOIN Sex_table ON Sex_table.ID=MICE_table.Sex
        LEFT JOIN Experimental_table ON Experimental_table.ID=MICE_table.Experimental_Status
        LEFT JOIN Genotypings_table ON Genotypings_table.ID=MICE_table.Genotyping_Status
        LEFT JOIN Lines_table ON Lines_table.ID=MICE_table.Line
        LEFT JOIN Rooms_table ON Rooms_table.ID=MICE_table.Room
        WHERE Alive=1 AND (Experimental_Status=1 OR Experimental_Status=3)  AND Breeding_status=3 AND Genotyping_Status IN (1,2)      
        """
        Alive_non_exp_stock_to_genotype =self.arbitrary_query_to_df(query_to_genotype)
        Alive_non_exp_stock_to_genotype.sort_values(by=['Sex_types'],ascending=False)
        Alive_non_exp_stock_to_genotype_sorted= Alive_non_exp_stock_to_genotype.sort_values(by=['Sex_types'],ascending=False)      
        Alive_non_exp_stock_to_genotype_grouped = Alive_non_exp_stock_to_genotype_sorted.groupby(['Breeders_types', 'Cage', 'Line_Short' ])['Lab_Number'].apply(list)
        Alive_non_exp_stock_to_genotype_grouped_frame=Alive_non_exp_stock_to_genotype_grouped.to_frame()
        self.mice_to_genotype=pd.DataFrame(Alive_non_exp_stock_to_genotype_grouped_frame.Lab_Number.values.tolist(), Alive_non_exp_stock_to_genotype_grouped_frame.index).add_prefix('Mouse_')
        self.mice_to_genotype.reset_index(inplace=True)
       
        
    def table_breeders(self):
        query_breeders="""
        SELECT 
            Breedings_table.ID,
            Breedings_table.Cage, 
            Breeders_types,
            Line_Short AS Breeding_Line,
            a.Lab_number AS Male,
            b.Lab_number AS Female1,
            c.Lab_number AS Female2,
            Male_Cage
        FROM Breedings_table 
        LEFT JOIN Lines_table ON Lines_table.ID=Breedings_table.Line
        LEFT JOIN Breeders_table ON Breeders_table.ID=Breedings_table.Breeding_status
        LEFT JOIN MICE_table a ON a.ID=Breedings_table.Male
        LEFT JOIN MICE_table b ON b.ID=Breedings_table.Female1
        LEFT JOIN MICE_table c ON c.ID=Breedings_table.Female2
        WHERE Breedings_table.Breeding_status=2 OR Breedings_table.Breeding_status=5
        """
        Alive_non_exp_breeders = self.arbitrary_query_to_df(query_breeders)  
        Aatable_ALL_NONEXP_BREEDERS= Alive_non_exp_breeders.sort_values(by=['Cage'],ascending=True)
        Aatable_ALL_NONEXP_BREEDERS['Female2'] = Aatable_ALL_NONEXP_BREEDERS['Female2'].astype('Int64')
        Aatable_ALL_NONEXP_BREEDERS['Male_Cage'] = Aatable_ALL_NONEXP_BREEDERS['Male_Cage'].astype('Int64')
        self.breedings=Aatable_ALL_NONEXP_BREEDERS
        self.breedings['Female1'].astype('Int64')
        self.breedings.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
    
    def table_all_alive_non_exp(self):
        query_all_alive="""
        SELECT 
            MICE_table.ID,
            Lab_number,
            Sex_types, 
            Cage, 
            round(julianday('now') - julianday(DOB)) AS DaysOld,
            round(round(julianday('now') - julianday(DOB))/7) AS WeeksOld,
            Breeders_types,
            Line_Short, 
            Genotypings_types
        FROM MICE_table 
        LEFT JOIN Breeders_table ON Breeders_table.ID=MICE_table.Breeding_status
        LEFT JOIN Sex_table ON Sex_table.ID=MICE_table.Sex
        LEFT JOIN Experimental_table ON Experimental_table.ID=MICE_table.Experimental_Status
        LEFT JOIN Genotypings_table ON Genotypings_table.ID=MICE_table.Genotyping_Status
        LEFT JOIN Lines_table ON Lines_table.ID=MICE_table.Line
        LEFT JOIN Rooms_table ON Rooms_table.ID=MICE_table.Room
        WHERE Alive=1 AND Room=2 AND (Experimental_Status=1 OR Experimental_Status=3)
        """
        Alive_non_exp = self.arbitrary_query_to_df(query_all_alive)     
        self.all_colony_mice= Alive_non_exp.sort_values(by=['Lab_Number'],ascending=True)
    
    def table_all_alive_nonexp_byline(self):   
        query_all_alive_line="""
        SELECT 
            MICE_table.ID,
            Lab_number,
            Sex_types, 
            Cage, 
            round(julianday('now') - julianday(DOB)) AS DaysOld,
            round(round(julianday('now') - julianday(DOB))/7) AS WeeksOld,
            Breeders_types,
            Line_Short, 
            Genotypings_types,
            Experimental_Status,
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
            Ai148_table.Genotypes_types AS Ai148,
            MICE_table.Experimental_Code
        FROM MICE_table         
        LEFT JOIN Breeders_table ON Breeders_table.ID=MICE_table.Breeding_status
        LEFT JOIN Genotypings_table ON Genotypings_table.ID=MICE_table.Genotyping_Status
        LEFT JOIN Sex_table ON Sex_table.ID=MICE_table.Sex
        LEFT JOIN Lines_table ON Lines_table.ID=MICE_table.Line       
        LEFT JOIN Genotypes_table AS G2C_table   ON MICE_table.G2C   = G2C_table.ID
        LEFT JOIN Genotypes_table AS Ai14_table   ON MICE_table.Ai14   = Ai14_table.ID
        LEFT JOIN Genotypes_table AS Ai75_table   ON MICE_table.Ai75   = Ai75_table.ID
        LEFT JOIN Genotypes_table AS VRC_table   ON MICE_table.VRC   = VRC_table.ID
        LEFT JOIN Genotypes_table AS SLF_table   ON MICE_table.SLF   = SLF_table.ID
        LEFT JOIN Genotypes_table AS PVF_table   ON MICE_table.PVF   = PVF_table.ID
        LEFT JOIN Genotypes_table AS Ai65_table   ON MICE_table.Ai65   = Ai65_table.ID
        LEFT JOIN Genotypes_table AS Ai80_table   ON MICE_table.Ai80   = Ai80_table.ID
        LEFT JOIN Genotypes_table AS VGC_table   ON MICE_table.VGC  = VGC_table.ID
        LEFT JOIN Genotypes_table AS Ai162_table   ON MICE_table.Ai162   = Ai162_table.ID
        LEFT JOIN Genotypes_table AS Ai148_table   ON MICE_table.Ai148  = Ai148_table.ID
        WHERE Alive=1 AND Room=2 AND(Experimental_Status=1 OR Experimental_Status=3) AND (Line=? OR Line=? OR Line=? OR Line=? OR Line=?)
        """
        params=(5,12,13,14,15,)
        Alive_non_exp_gad = self.arbitrary_query_to_df(query_all_alive_line, params)
        params=(1,1,1,1,1,)
        Alive_non_exp_ai14 = self.arbitrary_query_to_df(query_all_alive_line, params)
        params=(3,3,3,3,3,)
        Alive_non_exp_ai75 = self.arbitrary_query_to_df(query_all_alive_line, params)
        params=(2,2,2,2,2,)
        Alive_non_exp_ai65 = self.arbitrary_query_to_df(query_all_alive_line, params)
        params=(4,4,4,4,4,)
        Alive_non_exp_ai80 = self.arbitrary_query_to_df(query_all_alive_line, params)         
        params=(6,21,22,23,11,)        
        Alive_non_exp_PVF = self.arbitrary_query_to_df(query_all_alive_line, params)   
        params=(7,18,19,20,11,)
        Alive_non_exp_SLF = self.arbitrary_query_to_df(query_all_alive_line, params)
        params=(8,9,10,16,17,)
        Alive_non_exp_TIGRES = self.arbitrary_query_to_df(query_all_alive_line, params)
       
        
        self.Gad=self.remove_unnecesary_genes_from_df(Alive_non_exp_gad)
        self.Ai14=self.remove_unnecesary_genes_from_df(Alive_non_exp_ai14)
        self.Ai75=self.remove_unnecesary_genes_from_df(Alive_non_exp_ai75)
        self.Ai65=self.remove_unnecesary_genes_from_df(Alive_non_exp_ai65)
        self.Ai80=self.remove_unnecesary_genes_from_df(Alive_non_exp_ai80)
        self.PVF=self.remove_unnecesary_genes_from_df(Alive_non_exp_PVF)
        self.SLF=self.remove_unnecesary_genes_from_df(Alive_non_exp_SLF)
        self.TIGRES=self.remove_unnecesary_genes_from_df(Alive_non_exp_TIGRES)
        
    def full_tables(self):
        params=()
        query_MICE="""
        SELECT*
        FROM MICE_table
        """
        query_Experimental="""
        SELECT*
        FROM ExperimentalAnimals_table
        """
        query_Injections="""
        SELECT*
        FROM Injections_table
        """
        query_Windows="""
        SELECT*
        FROM Windows_table
        """
        query_Actions="""
        SELECT*
        FROM Actions_table       
        """
        query_imagingsessions="""
        SELECT*
        FROM ImagingSessions_table
        """
        query_imagedmice="""
        SELECT*
        FROM ImagedMice_table        
        """
        query_acquistions="""
        SELECT*
        FROM Acquisitions_table        
        """
        query_imaging="""
        SELECT*
        FROM Imaging_table        
        """
        query_widefield="""
        SELECT*
        FROM WideField_table        
        """
        query_facecamera="""
        SELECT*
        FROM FaceCamera_table        
        """    
        query_visualstim="""
        SELECT*
        FROM VisualStimulations_table       
        """
        virusstockinfo="""
        SELECT *
        FROM Virus_table
        WHERE CurrentAliquots!=0
        """
        projects="""
        SELECT *
        FROM Projects_table
        """
        self.allMICE=self.arbitrary_query_to_df(query_MICE, params)
        self.allMICE=self.allMICE.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
        self.allEXPERIMENTAL=self.arbitrary_query_to_df(query_Experimental,params)
        self.allEXPERIMENTAL=self.allEXPERIMENTAL.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
        self.allINJECTIONS=self.arbitrary_query_to_df(query_Injections,params)
        self.allINJECTIONS=self.allINJECTIONS.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
        self.allWINDOWS=self.arbitrary_query_to_df(query_Windows,params)
        self.allWINDOWS=self.allWINDOWS.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
        self.allACTIONS=self.arbitrary_query_to_df(query_Actions,params)
        self.allACTIONS=self.allACTIONS.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
        self.allIMAGINGSESSIONS=self.arbitrary_query_to_df(query_imagingsessions,params)
        self.allIMAGINGSESSIONS=self.allIMAGINGSESSIONS.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
        self.allIMAGEDMICE=self.arbitrary_query_to_df(query_imagedmice,params)
        self.allIMAGEDMICE=self.allIMAGEDMICE.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
        self.allACQUISITIONS=self.arbitrary_query_to_df(query_acquistions,params)
        self.allACQUISITIONS=self.allACQUISITIONS.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
        self.allIMAGING=self.arbitrary_query_to_df(query_imaging,params)
        self.allIMAGING=self.allIMAGING.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
        self.allWIDEFIELD=self.arbitrary_query_to_df(query_widefield,params)
        self.allWIDEFIELD=self.allWIDEFIELD.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
        self.allFACECAMERA=self.arbitrary_query_to_df(query_facecamera,params)
        self.allFACECAMERA=self.allFACECAMERA.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
        self.allVISSTIMS=self.arbitrary_query_to_df(query_visualstim,params)
        self.allVISSTIMS=self.allVISSTIMS.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
        self.allVirusstock=self.arbitrary_query_to_df(virusstockinfo, params)
        self.allVirusstock=self.allVirusstock.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)
        self.allprojects=self.arbitrary_query_to_df(projects, params)
        self.allprojects=self.allprojects.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)


#%% get max vavalues
    def remove_unnecesary_genes_from_df(self, df):
        
        genes=['G2C', 'Ai14', 'Ai75', 'VRC', 'SLF', 'PVF', 'Ai65', 'Ai80', 'VGC', 'Ai162', 'Ai148']
        
        genes_in_index=[gene for gene in genes if gene in df.columns.tolist()]
        wt_filter=df[genes_in_index].eq('WT').all()
        genes_to_filter_out=wt_filter.where(lambda x: x).dropna().index
        new_df=df.drop(genes_to_filter_out.tolist(), axis=1)
        return new_df
    
    def remove_unnecesary_virusinfo_from_df(self, df):
        new_df=df.dropna(how='all', axis=1)
        return new_df

    def get_max_cage(self):
        c=self.database_connection.cursor()
        query_max_cage_code="SELECT MAX(Cage) FROM MICE_table WHERE Alive=1 AND (Experimental_Status=1 OR Experimental_Status=3)"
        c.execute(query_max_cage_code)
        max_current_cage=c.fetchall()
        self.max_current_cage=max_current_cage[0][0]
    
    def get_max_mouse_code(self):
        c=self.database_connection.cursor()
        query_max_code="SELECT MAX(Lab_Number) FROM MICE_table"
        c.execute(query_max_code)
        max_current_code=c.fetchall()
        self.max_current_code=max_current_code[0][0]
        
    def get_cage_number(self):
        
        self.colony_cages=self.all_colony_mice['Cage'].nunique()
  
        
        
        
        
#%% Action methods
    def add_new_breeding(self, cage, male, females , date_performed=False):
        if date_performed:
             date_performed=datetime.datetime.strptime(date_performed, '%Y%m%d')
        else:     
             date_performed=datetime.datetime.now()
        
        
        query_get_lines= 'SELECT ID, Lab_Number, Line, Cage FROM MICE_table WHERE Lab_Number IN (?,%s)' % ','.join('?' for i in range(len(females)))
        params=(male,)+tuple(females)
        linesdf=self.arbitrary_query_to_df(query_get_lines,params)
        lines=linesdf['Line'].values.tolist()
        if lines.count(lines[0]) == len(lines):
            line=lines[0]
        elif 5 in lines and 1 in lines:
               line=12
        elif 5 in lines and 3 in lines:
               line=13       
        elif 18 in lines and ((2 in lines) or (19 in lines)):
               line=19      
        elif 18 in lines and 4 in lines:
               line=20    
        elif 21 in lines and 2 in lines:
               line=22       
        elif 21 in lines and 4 in lines:
               line=23
        elif 5 in lines and 2 in lines:
               line=5
        elif 2 in lines:
               line=2
        elif 12 in lines :
               line=12
           
                 
               
             
        maleID  = int(linesdf[ linesdf['Lab_Number']== male]['ID'].iloc[0])
        female1ID=int(linesdf[ linesdf['Lab_Number']== females[0]]['ID'].iloc[0])
        female1=int(females[0])
        female2=None
        female2ID='NULL'
        MaleCage =int(linesdf[ linesdf['Lab_Number']== male]['Cage'].iloc[0])
        Female1Cage=int(linesdf[ linesdf['Lab_Number']== females[0]]['Cage'].iloc[0])
        if len(females)==2:
            female2=int(females[1])
            Female2Cage=int(linesdf[ linesdf['Lab_Number']== females[1]]['Cage'].iloc[0])
            female2ID=int(linesdf[ linesdf['Lab_Number']== females[1]]['ID'].iloc[0])
        requires_genotyping=0
        if line in (2,18,19,20):
            requires_genotyping=1
        

        query_insert_new_breeding= ''' INSERT INTO Breedings_table(Cage, Male, Female1, Female2, Line,StartDate,Breeding_Status, Requires_Genotyping)
              VALUES(?,?,?,?,?,date(?),?,?) 
          '''
         
        params=(cage,maleID,female1ID,female2ID,line,date_performed,2,requires_genotyping)
        self.arbitrary_inserting_record(query_insert_new_breeding, params)
        
        Action_Type_BreedingStart=2
        Action_Type_Transfer=13

        actions_dictionary={MaleCage:{Action_Type_Transfer:((male,),(cage),(date_performed))
                                      },
                            Female1Cage:{Action_Type_Transfer:((female1,),(cage),(date_performed))
                                      },
                            }
        if len(females)==2: 
            if Female2Cage==Female1Cage:
                actions_dictionary[Female1Cage]={Action_Type_Transfer:((female1,female2),(cage),(date_performed))}                
            else:
                actions_dictionary[Female2Cage]={Action_Type_Transfer:((female2,),(cage),(date_performed))}          
        
        self.add_multiple_actions(actions_dictionary)
        
        query_update_mice='UPDATE MICE_table SET  Cage=?, Breeding_status=2  WHERE Lab_Number IN (?,%s)' % ','.join('?' for i in range(len(females)))            
        params2=(cage, male,)+tuple(females)
        self.arbitrary_updating_record(query_update_mice, params2, commit=True)
        
        
        
        
        actions_dictionary2={cage:{Action_Type_BreedingStart:((male, female1,),(),(date_performed))   
                                          }
                            }
        if len(females)==2: 
            actions_dictionary2[cage]={Action_Type_BreedingStart:((male, female1, female2),(),(date_performed)) }
            
        self.add_multiple_actions(actions_dictionary2)
   

        self.independent_commit() 
        
        
    def Separate_male(self,BreedingCage, NewCage, commit=False):
                
        breeding_info=self.get_breeding_info(BreedingCage)

        #update male
       
         # add actions
        Action_Type_Separation=6
        Action_Type_Transfer=13
        Action_Type_Temp_stop=9
             
        actions_dictionary={BreedingCage:{Action_Type_Separation:((breeding_info[0],),(NewCage),()), 
                                          Action_Type_Temp_stop:((breeding_info[1]),(),()),
                                          Action_Type_Transfer:((breeding_info[0],),(NewCage),())
                                          }
                            }
        self.add_multiple_actions(actions_dictionary)
        
        query_update_male="""
        UPDATE MICE_table
        SET  Cage=?, Breeding_status=4
        WHERE ID=?
        """
        params=(NewCage, breeding_info[2])
        self.arbitrary_updating_record(query_update_male, params)
               
        query_update_females='UPDATE MICE_table SET  Breeding_status=5 WHERE ID IN (%s)' % ','.join('?' for i in range(len(breeding_info[3])))
        params=breeding_info[3]
        self.arbitrary_updating_record(query_update_females, params)
        
        
        # update breding info
            
        query_update_breeding="""
        UPDATE Breedings_table
        SET  Male_Cage=?, Breeding_Status=5
        WHERE ID=?
        """
        params=(NewCage,breeding_info[4])
        self.arbitrary_updating_record(query_update_breeding, params, commit=commit)

    def get_breeding_info(self, BreedingCage):
        query_all_mouse_id_for_breeding="""
            SELECT  
                Male, 
                b.Lab_Number AS Male_number,
                Female1, 
                c.Lab_Number AS Female1_number,
                Female2, 
                d.Lab_Number AS Female2_number,
                a.ID 
            FROM Breedings_table a
            LEFT JOIN MICE_table b ON b.ID= a.Male
            LEFT JOIN MICE_table c ON c.ID= a.Female1
            LEFT JOIN MICE_table d ON d.ID= a.Female2
            WHERE a.Cage=? """  
            
        params=(BreedingCage,)
        ParentCodes=self.arbitrary_query_to_df(query_all_mouse_id_for_breeding,params)


        maleLabNumber= ParentCodes[['Male_number']].values.tolist()[0][0]
        femalesLabnumbers=tuple([i for i in ParentCodes[['Female1_number','Female2_number']].values.tolist()[0] if i!=None])
        maleID=ParentCodes[['Male']].values.tolist()[0][0]
        femalesID=tuple([x for x in ParentCodes[['Female1','Female2']].values.tolist()[0] if x is not None])
        breeding_id= ParentCodes[['ID']].values.tolist()[0][0]
        return [maleLabNumber, femalesLabnumbers, maleID, femalesID, breeding_id]


    def readd_male(self, BreedingCage, MaleCage):
        breeding_info=self.get_breeding_info(BreedingCage)

        # add actions
        Action_Type_ReaddingMale=7
        Action_Type_Transfer=13
        Action_Type_BreedingRestart=16

        actions_dictionary={MaleCage:{Action_Type_ReaddingMale:((breeding_info[0],),(BreedingCage),()), 
                                      Action_Type_Transfer:((breeding_info[0],),(BreedingCage),())
                                      },
                            BreedingCage:{Action_Type_BreedingRestart:((breeding_info[1]),(),())   
                                          }
                            }
        self.add_multiple_actions(actions_dictionary)
        #update male
        query_update_male=""" UPDATE MICE_table 
                              SET  Cage=?, Breeding_status=2 
                              WHERE ID=?"""           
        params=(BreedingCage, breeding_info[2])
        self.arbitrary_updating_record(query_update_male, params)
        
        #update females add tTO DO OPTION FOR SINGLE FEMALE BREEDING
        query_update_females='UPDATE MICE_table SET  Breeding_status=2 WHERE ID IN (%s)' % ','.join('?' for i in range(len(breeding_info[3])))
        params=breeding_info[3]
        self.arbitrary_updating_record(query_update_females, params)
        
        
        # update breding info
        query_update_breeding="""
        UPDATE Breedings_table
        SET  Male_Cage=NULL, Breeding_Status=2
        WHERE ID=?
        """
        params=(breeding_info[4],)
        self.arbitrary_updating_record(query_update_breeding, params, commit=True)
        
    def Add_Weaning(self, gui, LitterCode, MaleNumber=0, FemaleNumber=0, MaleCage=None, FemaleCage=None, MiceSaced=0 ):      

        MiceAlive=MaleNumber+FemaleNumber+MiceSaced
            
        Action_Type=1
        Breeding_status=3
        Alive=1
        Experimental_Status=1
        Room=2
        query_all_genes="SELECT  Genes_types FROM Genes_table"    
        Genes=self.arbitrary_query_to_df(query_all_genes).Genes_types.values.tolist()   

        genes_switcher = {gene:3 for gene in Genes}
        Genotyping_Status=4
        Label=1
        
        get_litter_row="""
            SELECT  
                ID,
                Cage,
                Breeding_Parents, 
                date((julianday(Date_Seen)-Age)) AS DOB,
                round(julianday('now') - (julianday(Date_Seen)-Age)) AS DaysOld, 
                Line 
            FROM Litters_table 
            WHERE Date_Weaned IS NULL AND  ID=? """
            
        params=(LitterCode,)
        litter_info=self.arbitrary_query_to_df(get_litter_row,params)
        BreedingCage=int(litter_info.loc[0,'Cage'])
        Line=int(litter_info.loc[0,'Line'])
        Parent_Breeding=int(litter_info.loc[0,'Breeding_Parents'])
        query_all_mouse_id_for_breeding="SELECT  Male, Female1, Female2, Requires_Genotyping  FROM Breedings_table    WHERE Breedings_table.ID=?"      
        params=(Parent_Breeding,)
        ParentCodes=self.arbitrary_query_to_df(query_all_mouse_id_for_breeding, params).values.tolist()[0]
        
        all_breeding_litters_query="SELECT  ID, date((julianday(Date_Seen)-Age)) AS DOB, Breeding_Parents, round(julianday('now') - (julianday(Date_Seen)-Age)) AS DaysOld, Line FROM Litters_table WHERE Date_Weaned IS NULL AND  Cage=? "
        params=(BreedingCage,)
        all_breeding_litters_info =self.arbitrary_query_to_df(all_breeding_litters_query, params)
        
  
        if len(all_breeding_litters_info)>1:
            MultipleLitter=True
        else:
            MultipleLitter=False

        if not MultipleLitter:
            DOB=litter_info.loc[0,'DOB']
            query_update_litter="""
            UPDATE Litters_table
            SET  Date_Weaned=date( 'now'), NumberAlive= ?
            WHERE ID=?
            """
            params=(MiceAlive, LitterCode)
            self.arbitrary_updating_record(query_update_litter, params) 
                
        elif MultipleLitter:
            # self.Combining_window=select_values_gui(["Combined","NotCombined"], 'CombineLitter')
            self.Combining_window=new_window_multilitter(gui, ["Combined","NotCombined"], 'CombineLitter') 
            self.Combining_window.wait_window()
            Combining_window=self.Combining_window.values

            
            if  Combining_window=="Combined":
                DOB=max(all_breeding_litters_info.loc[0,'DOB'], all_breeding_litters_info.loc[1,'DOB'])
                Idx=(int(all_breeding_litters_info.loc[0,'ID']), int(all_breeding_litters_info.loc[1,'ID']))
                
                query_update_litter_multple="""
                UPDATE Litters_table
                SET  Date_Weaned=date( 'now'), NumberAlive= ?
                WHERE ID=? OR ID=?
                """
                params=(MiceAlive, Idx[0],Idx[1])
                self.arbitrary_updating_record(query_update_litter_multple, params) 
                
            elif  Combining_window=="NotCombined":
                DOB=litter_info.loc[0,'DOB']
                query_update_litter="""
                UPDATE Litters_table
                SET  Date_Weaned=date( 'now'), NumberAlive= ?
                WHERE ID=?
                """
                params=(MiceAlive, LitterCode)
                
                self.arbitrary_updating_record(query_update_litter, params) 
            
        if Line==1:
                genes_switcher['Ai14']=5
        if Line==3:
                genes_switcher['Ai75']=5
        if Line==4:
                genes_switcher['Ai80']=5
        if Line==5:
                genes_switcher['G2C']=5
        if Line==12:
                genes_switcher['Ai14']=6
                genes_switcher['G2C']=6
        if Line==13:
                genes_switcher['Ai14']=6
                genes_switcher['G2C']=6
        if Line==17:
                genes_switcher['VGC']=6
                genes_switcher['Ai162']=6
        if Line==21:
                genes_switcher['VRC']=5
                genes_switcher['PVF']=5
               
    
        if Line==2:
            query_3mouse_breeding="""
                                    SELECT MICE_table.ID, Lab_Number, Sex, Cage,Breeding_status,Parent_Breeding, Ai65, Genotypes_types  
                                    FROM MICE_table 
                                    LEFT JOIN Genotypes_table ON Genotypes_table.ID=MICE_table.Ai65  
                                    WHERE MICE_table.ID=? OR MICE_table.ID=? OR MICE_table.ID=?"""
            params=(ParentCodes[0],ParentCodes[1],ParentCodes[2])
            mouse_from_breeding = self.arbitrary_query_to_df(query_3mouse_breeding, params)
            
            if 2  in mouse_from_breeding.Ai65.values :
                genes_switcher['Ai65']=4
                Genotyping_Status=1
                Label=7
                
        if Line==18:
            query_3mouse_breeding="""
                                    SELECT MICE_table.ID, Lab_Number, Sex, Cage,Breeding_status,Parent_Breeding, SLF,VRC, Genotypes_types  
                                    FROM MICE_table 
                                    LEFT JOIN Genotypes_table ON Genotypes_table.ID=MICE_table.SLF  
                                    WHERE MICE_table.ID=? OR MICE_table.ID=? OR MICE_table.ID=?"""         
            params=(ParentCodes[0],ParentCodes[1],ParentCodes[2])
            mouse_from_breeding = self.arbitrary_query_to_df(query_3mouse_breeding, params)
             
            if 2  in mouse_from_breeding.SLF.values :
                genes_switcher['SLF']=4
                genes_switcher['VRC']=5
                Genotyping_Status=1
                Label=7
                     
        if Line==19:
            query_3mouse_breeding="""
                                    SELECT MICE_table.ID, Lab_Number, Sex, Cage,Breeding_status,Parent_Breeding, SLF,VRC,Ai65, Genotypes_types 
                                    FROM MICE_table 
                                    LEFT JOIN Genotypes_table ON Genotypes_table.ID=MICE_table.SLF  
                                    WHERE MICE_table.ID=? OR MICE_table.ID=? OR MICE_table.ID=?"""
            params=(ParentCodes[0],ParentCodes[1],ParentCodes[2])
            mouse_from_breeding = self.arbitrary_query_to_df(query_3mouse_breeding, params)
            
            for gene in ['SLF', 'VRC', 'Ai65']:
                if not all([True if i in [1,5] else False for i in mouse_from_breeding[gene].values ]):
                    genes_switcher[gene]=4
                    Genotyping_Status=1
                    Label=7
                
        if Line==22:
              query_3mouse_breeding="""
                                      SELECT MICE_table.ID, Lab_Number, Sex, Cage,Breeding_status,Parent_Breeding, Ai65, PVF, VRC, Genotypes_types  
                                      FROM MICE_table 
                                      LEFT JOIN Genotypes_table ON Genotypes_table.ID=MICE_table.SLF 
                                      WHERE MICE_table.ID=? OR MICE_table.ID=? OR MICE_table.ID=?"""
              params=(ParentCodes[0],ParentCodes[1],ParentCodes[2])
              mouse_from_breeding = self.arbitrary_query_to_df(query_3mouse_breeding, params)
              for gene in ['PVF', 'VRC', 'Ai65']:
                if not all([True if i in [1,5] else False for i in mouse_from_breeding[gene].values ]):
                      genes_switcher[gene]=4
                      Genotyping_Status=1
                      Label=7

        if  ParentCodes[-1]:
            Genotyping_Status=1
            Label=7

                
        notes=''     
        common_values= (Breeding_status,DOB,Label,Parent_Breeding,
                    genes_switcher['Ai14'],
                    genes_switcher['Ai65'],
                    genes_switcher['Ai75'],
                    genes_switcher['Ai80'],
                    genes_switcher['Ai148'],
                    genes_switcher['Ai162'],
                    genes_switcher['G2C'],
                    genes_switcher['VGC'],
                    genes_switcher['VRC'],
                    genes_switcher['PVF'],
                    genes_switcher['SLF'],
                    Line,Room,Experimental_Status,Genotyping_Status,Alive,notes)
        
        OnlyMales=False
        OnlyFemales=False
        if MaleNumber==0:
            OnlyFemales=True
        if FemaleNumber==0:
            OnlyMales=True

        if OnlyFemales:
            Females_Lab_Numbers=[ int(self.max_current_code+i+1 )   for i in range(FemaleNumber)]   
            self.add_new_cage(Lab_Numbers=Females_Lab_Numbers, Sex=2, Cage=FemaleCage, common_values=common_values, commit=True)          
           
            actions_dictionary_females={BreedingCage:{Action_Type:((Females_Lab_Numbers),(FemaleCage),()) }        
                                        }    
            self.add_multiple_actions(actions_dictionary_females, commit=True)    
            
        elif OnlyMales:
            Males_Lab_Numbers=[ int(self.max_current_code+i+1 )   for i in range(MaleNumber)]
            self.add_new_cage(Lab_Numbers=Males_Lab_Numbers, Sex=1, Cage=MaleCage, common_values=common_values, commit=True)
            
            actions_dictionary_males={BreedingCage:{Action_Type:((Males_Lab_Numbers),(MaleCage),()) }         
                                      }      
            self.add_multiple_actions(actions_dictionary_males, commit=True)        

        else:
            Males_Lab_Numbers=[ int(self.max_current_code+i+1 )   for i in range(MaleNumber)]
            Females_Lab_Numbers=[  int(Males_Lab_Numbers[-1]+i+1)    for i in range(FemaleNumber)]
            
            self.add_new_cage(Lab_Numbers=Males_Lab_Numbers, Sex=1,Cage= MaleCage, common_values=common_values, commit=True)
            self.add_new_cage(Lab_Numbers=Females_Lab_Numbers, Sex=2,Cage= FemaleCage, common_values=common_values, commit=True)
            
            actions_dictionary_males={BreedingCage:{Action_Type:(tuple(Males_Lab_Numbers),(MaleCage),()) }         
                                      }   
            actions_dictionary_females={BreedingCage:{Action_Type:(tuple(Females_Lab_Numbers),(FemaleCage),()) }        
                                        }   
            self.add_multiple_actions(actions_dictionary_females, commit=False)     
            self.add_multiple_actions(actions_dictionary_males, commit=True)   
        print('Weanings added')
   

    def update_old_litters(self, values):
        c=self.database_connection.cursor()
        updated_litters=values
  
        query_update_litter="""
        UPDATE Litters_table
        SET   NumberAlive=?, NumberDead=?
        WHERE ID=?
            """
        for lit in updated_litters[1:]:
               Alive=lit[2]
               Dead=lit[3]
               ID=lit[1]
               c.execute(query_update_litter,( int(Alive),int(Dead), int(ID),))
               
               if int(Alive)==0:
                   x='Dead Litter'
                   query_update_litter2="""
                        UPDATE Litters_table
                        SET  Date_Weaned=date(date('now')), Notes=?
                        WHERE ID=?
                        """
                   params=( x, int(ID))
                   self.arbitrary_updating_record(query_update_litter2, params)
                                   
        self.independent_commit()
        print('Litter Updated')
    def add_new_litters(self,gui, new_litter_number):
        
        self.update_litter_window = UpdateLitterInput(gui, self, new_litter_number)  
        self.update_litter_window.wait_window()
        new_litters=self.update_litter_window.values
   
        new_litter= ''' INSERT INTO Litters_table(Cage,Breeding_Parents,Date_Seen, Age,Line,NumberAlive,NumberDead)
              VALUES(?,?,date('now'),?,?,?,?) 
              '''
        for lit in new_litters[1:]: 
            Cage=lit[0]
            Alive=lit[2]
            Dead=lit[3]
            Age=lit[4]

            query_breeding_from_litters="""
                SELECT ID, Line
                FROM Breedings_table
                WHERE Cage=?
            """
            params=(Cage,)
            breeding_from_litters=self.arbitrary_query_to_df(query_breeding_from_litters, params)
            
            Breeding_Parents=int(breeding_from_litters['ID'][0])
            Line=int(breeding_from_litters['Line'][0])
            
            
            params=(Cage,Breeding_Parents, Age,Line,Alive,Dead)
            self.arbitrary_inserting_record(new_litter, params)
            
        self.independent_commit()

        
    def labelling(self, cages_labelled):
        
        Action_Type_Labelling=14
        self.cage_pending_genotypes=[]
        
        query_update_labellings="""
                UPDATE MICE_table
                SET  Label=?, Genotyping_Status=2
                WHERE ID=?
                """    
        query_mice_labelled="SELECT ID, Lab_number  FROM MICE_table  WHERE Cage=?"
        query_labels="SELECT * FROM Labels_table "

        for cage in cages_labelled:
            
            self.cage_pending_genotypes.append(cage)
            
            params=(cage,)
            mice_labelled_info =self.arbitrary_query_to_df(query_mice_labelled, params).values.tolist()
           
            labels =self.arbitrary_query_to_df(query_labels).values.tolist()
              
            for i, mouse in enumerate(mice_labelled_info):
                
                params=(labels[i+1][0], mouse[0])
                self.arbitrary_updating_record(query_update_labellings, params)
            
         # add actions

            mice_lablelled_numbers=[i[1] for i in mice_labelled_info]   
            
            
            actions_dictionary={cage:{Action_Type_Labelling:(tuple(mice_lablelled_numbers),(),()), 
                                      },
                            }
            self.add_multiple_actions(actions_dictionary)
            
        self.independent_commit()

        
        
    def genotyping(self,gui, cage_list):
        
        Action_Type_Genotyping=10

        for cage in cage_list:
            
            query_mice_labelled="SELECT * FROM MICE_table  WHERE Cage=?"
            params=(cage,)
            mice_genotypes =self.arbitrary_query_to_df(query_mice_labelled,params)
            mice_codes= mice_genotypes['Lab_Number'].values.tolist()
            genes_to_genotype=[]
            for column in mice_genotypes.iloc[:, 8:19]:
                if 4 in  mice_genotypes[column].to_list():
                    genes_to_genotype.append(column)
            app_rows=len(mice_genotypes[column].to_list())+1

            self.input_genotypes_window= input_genotypes(gui,mice_codes, app_rows, genes_to_genotype)
            self.input_genotypes_window.wait_window()
            updated_genotypes=self.input_genotypes_window.values
            only_genotypes=updated_genotypes[1:]
            
            
            
            genotype_codes=[]
            for genotype in only_genotypes:
                if '+/-' in genotype:
                    genotype_codes.append(2)
                if '+/+' in genotype:
                    genotype_codes.append(1)            
                if '-/-' in genotype:
                    genotype_codes.append(3)            
            
            
            query_mice_genotyped="SELECT ID, Line FROM MICE_table  WHERE Cage=?"
            params=(cage,)
            mice_genotyped_ID =self.arbitrary_query_to_df(query_mice_genotyped,params).values.tolist()
            

            line=mice_genotyped_ID[0][1]
            
            if line==2:
                query_update_genotypes="""
                UPDATE MICE_table
                SET  Ai65=?, Genotyping_Status=3
                WHERE ID=?
                """
                
            elif line==18 or line==19:
                query_update_genotypes="""
                UPDATE MICE_table
                SET  SLF=?, Genotyping_Status=3
                WHERE ID=?
                """

            for i, mouse in enumerate(mice_genotyped_ID):
                params=(genotype_codes[i], mouse[0])
                self.arbitrary_updating_record(query_update_genotypes, params)
                
         # add actions
            actions_dictionary={cage:{Action_Type_Genotyping:(tuple(mice_codes),(),()), 
                                      },
                            }
            self.add_multiple_actions(actions_dictionary)
        
        self.independent_commit()


    def cage_sacing(self, cages_saced):  
        
        for cage in cages_saced:

            query_mice_saced_info="SELECT ID, Lab_number FROM MICE_table  WHERE Cage=?"
            params=(cage,)
            mice_saced=self.arbitrary_query_to_df(query_mice_saced_info,params)
            
            mice_saced_labnumbers=mice_saced.Lab_Number.values.tolist()

            Action_Type_cage_sacing=5    
            actions_dictionary={cage:{Action_Type_cage_sacing:(tuple(mice_saced_labnumbers),(),()), 
                                      },
                            }
            self.add_multiple_actions(actions_dictionary)  
                        
            for i, mouse in enumerate(mice_saced_labnumbers):
                self.mouse_sacing(mouse)
                
        self.independent_commit()
                               
    def mouse_sacing(self, mouse_saced_lab_number, commit=False):  
               
        query_mouse_saced_info="SELECT ID, Cage FROM MICE_table  WHERE Lab_number=?"
        params=(mouse_saced_lab_number,)
        mouse_saced_info=self.arbitrary_query_to_df(query_mouse_saced_info,params).values.tolist()[0]
        
        Action_Type_mouse_sacing=4                
        actions_dictionary={mouse_saced_info[1]:{Action_Type_mouse_sacing:((mouse_saced_lab_number,),(),()), 
                                  },
                        }
        self.add_multiple_actions(actions_dictionary)
        
        
        query_mouse_saced="""
                UPDATE MICE_table
                SET  Cage=NULL, Alive=0
                WHERE ID=?
                """
        params=(mouse_saced_info[0],)
        self.arbitrary_updating_record(query_mouse_saced, params, commit=commit)
            
            
    def mouse_dead(self, mouse_dead_lab_number, comment, commit=False):
        
        query_mouse_dead_info="SELECT ID, Cage FROM MICE_table  WHERE Lab_number=?"
        params=(mouse_dead_lab_number,)
        mouse_saced_info=self.arbitrary_query_to_df(query_mouse_dead_info, params).values.tolist()[0]
        
        Action_Type_mouse_sacing=4                
        actions_dictionary={mouse_saced_info[1]:{Action_Type_mouse_sacing:((mouse_dead_lab_number,),(),()), 
                                  },
                        }
        self.add_multiple_actions(actions_dictionary)
        
        
        query_mouse_saced="""
                UPDATE MICE_table
                SET  Cage=NULL, Alive=0, Notes=?
                WHERE ID=?
                """
        params=(comment,mouse_saced_info[0])
        self.arbitrary_updating_record(query_mouse_saced, params, commit=commit)
        
        
    def dead_mother(self, mouse_dead_lab_number,comment,commit=False): 
        query_mouse_dead_info="SELECT ID, Cage, Breeding_status FROM MICE_table  WHERE Lab_number=?"
        params_dead=(mouse_dead_lab_number,)
        mouse_dead_info=self.arbitrary_query_to_df(query_mouse_dead_info, params_dead).values.tolist()[0]
        
        query_mouse_check_breeding="SELECT ID,Cage, Female1, Female2 FROM Breedings_table WHERE Cage=? "
        params_breeding=(mouse_dead_info[1],)
        mouse_dead_breeding_info=self.arbitrary_query_to_df(query_mouse_check_breeding, params_breeding).values.tolist()[0]
        
        if mouse_dead_info[0]==mouse_dead_breeding_info[2]:
            query_mouse_saced="""
                    UPDATE Breedings_table
                    SET  Female1=NULL,  Notes=?
                    WHERE ID=?
                    """
            params=(comment,mouse_dead_breeding_info[0])
            self.arbitrary_updating_record(query_mouse_saced, params, commit=commit)
        elif mouse_dead_info[0]==mouse_dead_breeding_info[3]:
            query_mouse_saced="""
                    UPDATE Breedings_table
                    SET  Female2=NULL,  Notes=?
                    WHERE ID=?
                    """
            params=(comment, mouse_dead_breeding_info[0])
            self.arbitrary_updating_record(query_mouse_saced, params, commit=commit)   

        self.mouse_dead(mouse_dead_lab_number, comment, commit=commit)
                          
    def breeding_stop(self, breeding_cages):   
        
        for BreedingCage in breeding_cages:
            query_all_mouse_id_for_stop_breeding=""" SELECT  
                                                        Male, 
                                                        b.Lab_Number AS Male_number,
                                                        Female1, 
                                                        c.Lab_Number AS Female1_number,
                                                        Female2, 
                                                        d.Lab_Number AS Female2_number,
                                                        a.ID 
                                                    FROM Breedings_table a
                                                    LEFT JOIN MICE_table b ON b.ID= a.Male
                                                    LEFT JOIN MICE_table c ON c.ID= a.Female1
                                                    LEFT JOIN MICE_table d ON d.ID= a.Female2
                                                    WHERE a.Cage=? """ 
                                                    
            params=(BreedingCage,)
            ParentCodes=self.arbitrary_query_to_df(query_all_mouse_id_for_stop_breeding,params)
            maleLabNumber= ParentCodes[['Male_number']].values.tolist()[0][0]
            femalesnumber=ParentCodes[['Female1_number','Female2_number']].values.tolist()[0]
            femlesnumbers=tuple([i for i in femalesnumber if i!=None])
            maleID=ParentCodes[['Male']].values.tolist()[0][0]
            femalesID=tuple(ParentCodes[['Female1','Female2']].values.tolist()[0])
            parenstIDs=((maleID,)+femalesID)
            #update male
            
            for parentcode in parenstIDs:
            
                query_update_mouse=""" UPDATE MICE_table
                                       SET  Breeding_status=6
                                       WHERE ID=?"""
                params=(parentcode,)
                self.arbitrary_updating_record(query_update_mouse, params)
            
            # update breding info
                
            breeding_id= ParentCodes.ID.values.tolist()[0]
            
            query_update_breeding=""" UPDATE Breedings_table
                                      SET   EndDate=date('now') ,Breeding_Status=6
                                      WHERE ID=?"""
            params=(breeding_id,)
            self.arbitrary_updating_record(query_update_breeding, params)
            
            # add actions
            
            Action_Type_Breeding_Stop=8
            actions_dictionary={BreedingCage:{Action_Type_Breeding_Stop:(((maleLabNumber,)+femlesnumbers),(),()), 
                                      },
                            }
            self.add_multiple_actions(actions_dictionary, commit=True)
                   
    
    
    def add_multiple_actions(self, actions_dictionary, commit=False):  
              
                 
        for cage_to_process in actions_dictionary.keys():
            cage_start=cage_to_process         
            for action_type in actions_dictionary[cage_start].keys():
                mice=actions_dictionary[cage_start][action_type][0] 
                cage_end=actions_dictionary[cage_start][action_type][1] 
                action_date=actions_dictionary[cage_start][action_type][2] 
                
                self.add_single_action(action_type, cage_start, cage_end, mice, action_date, commit=commit)
         
    def add_single_action(self, action_type, cage_start, cage_end, mice, action_date, commit=False):

        if not action_date:
            action_date= datetime.date.today()
         
        query_mice_info="SELECT ID, Lab_number FROM MICE_table WHERE Cage=?"  

        params=(int(cage_start),)
        mice_info_df=self.arbitrary_query_to_df(query_mice_info, params)
        mice_IDs=mice_info_df['ID'].tolist()
        mice_Lab_numbers=mice_info_df['Lab_Number'].tolist()
        
        if action_type==1:
            params=(cage_end,)
            mice_info_df=self.arbitrary_query_to_df(query_mice_info,params)
            mice_IDs=mice_info_df['ID'].tolist()
            mice_Lab_numbers=mice_info_df['Lab_Number'].tolist()
               
        if not cage_end:
            cage_end=cage_start
            
        if mice:
            mice_number=len(mice)
            mice_IDs=[mice_IDs[ind] for ind in mice_info_df.index.values.tolist() if  mice_Lab_numbers[ind] in mice]
            
        else:
            mice_number=len(mice_IDs)
            
        dif=5-mice_number   
        for i in range(dif):
            mice_IDs.append(np.nan)    
            
        mice_tuple_for_action=tuple(mice_IDs)    
     
        query_add_action=""" INSERT INTO Actions_table( Date, Action_Type, Cage_start ,Cage_end, Mouse_1, Mouse_2, Mouse_3, Mouse_4, Mouse_5)
                             VALUES(date(date(?)),?,?,?,?,?,?,?,?) """   
                         
        params=((action_date, action_type, cage_start, cage_end,) + mice_tuple_for_action) 
        self.arbitrary_inserting_record(query_add_action, params, commit=commit)




    def add_new_cage(self, Lab_Numbers=None, Sex=None, Cage=None, common_values=None, commit=False, number_of_animals=None, outside_mice=None):
        if not outside_mice:
            for new_mouse in Lab_Numbers:  
                Lab_Number=new_mouse
                self.add_single_mouse(Lab_Number, Sex, Cage, common_values, commit=commit)
        else:
            Cage=self.max_current_cage + 1
            max_lab_number=self.max_current_code  
            next_number=max_lab_number + 1
            for new_mouse in range(number_of_animals):
                next_number=next_number + new_mouse 
                Lab_Number=next_number
                self.add_single_mouse(Lab_Number, Sex, Cage, common_values, commit=commit)
                
            
                
    def add_single_mouse(self, Lab_Number, Sex, Cage, common_values, commit=False, external=False):    
        new_mice= ''' INSERT INTO MICE_table( Lab_Number,Sex,Cage,Breeding_status,DOB,Label,Parent_Breeding,
                                            Ai14,Ai65,Ai75,Ai80,Ai148,Ai162,G2C,VGC,VRC,PVF,SLF,
                                            Line,Room,Experimental_Status,Genotyping_Status,Alive,Notes)
                      VALUES(?,?,?,?,date(?),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
       
        individual_values=[Lab_Number, Sex,Cage]     
        params= tuple(individual_values + list(common_values))

        self.arbitrary_inserting_record(new_mice, params, commit=commit)
       

       
        
           
#%% sqlite queries              
    def arbitrary_query_to_df(self, query, params=False):   
        if not params:
            params=()        
        selected_table=pd.read_sql_query(query, self.database_connection,params=params)
        return selected_table
        
    def arbitrary_inserting_record(self, query, params, commit=False):   
        c=self.database_connection.cursor()
        c.execute(query, (params))
        if commit:
            self.database_connection.commit()  
        self.update_variables() 

        
    def arbitrary_updating_record(self, query, params, commit=False):   
        c=self.database_connection.cursor()
        c.execute(query, (params))
        if commit:
            self.database_connection.commit()  
        self.update_variables()    

    def arbitrary_remove_record(self, query, params, commit=False):   
        c=self.database_connection.cursor()
        c.execute(query, (params))
        if commit:
            self.database_connection.commit()  
        self.update_variables()  

    def arbitrary_add_column(self, query, params, commit=False):   
        c=self.database_connection.cursor()
        c.execute(query, (params))
        if commit:
            self.database_connection.commit()  
        self.update_variables()  
        
    def independent_commit(self) :
        self.database_connection.commit()  
        self.update_variables()  
        

    
            
            
            
        
        
        
        
        
        