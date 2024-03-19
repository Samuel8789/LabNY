# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 08:33:15 2021

@author: sp3660
"""


import os 
import sys

from project_manager.ProjectsCLass import Project
#
from .database import MouseDatabase
from .dataManaging.data_managing import DataManaging
from .gui.gui import Gui
from .AllFunctions.select_values_gui import select_values_gui
import logging 
module_logger = logging.getLogger(__name__)


class RunNYLab(Project):    
    def __init__(self,  githubtoken_path, gui=False ):
        Project.__init__(self,'LabNY', githubtoken_path, Project.computer, Project.platform)
        #loadmicedatabase     
        
        self.databasefile=os.path.join(Project.all_paths_for_this_system['Dropbox'],'LabNY', 'MouseDatabase.db')
        module_logger.info('Loading Mouse Database')
        self.database=MouseDatabase(self.databasefile, self)   
        #load all mice info       
        self.all_mouse_info=self.database.allEXPERIMENTAL
        if Project.computer=='DESKTOP-OKLQSQS':
            self.data_paths_names=['Raw','Pre_proccessed_slow_chandelier_tigres', 'Analysis_Fast_1', 'Analysis_Fast_2', 'Pre_proccessed_slow_interneurons_others', 'Raw2']    
                
            self.data_paths_roots={}
            self.load_datbaseSaved_paths_roots()
            try:

                self.data_paths_project={name: os.path.join(self.all_paths_for_this_system[self.data_paths_roots[name]], 'LabNY') for name in self.data_paths_names}
                self.data_paths_project['ResultsContainers']= os.path.join(self.all_paths_for_this_system['Dropbox'], 'LabNY','ResultsContainers')    

    
            except KeyError:
                self.change_and_save_selected_paths_roots()
                self.data_paths_project={name: os.path.join(self.all_paths_for_this_system[self.data_paths_roots], 'LabNY') for name in self.data_paths_names}  
                
        elif Project.computer=='sp3660-YusteLab':
           
                self.data_paths_names=['Raw','Pre_proccessed_slow_chandelier_tigres', 'Analysis_Fast_1', 'Analysis_Fast_2', 'Pre_proccessed_slow_interneurons_others', 'Raw2']    
                    
                self.data_paths_roots={}
                self.load_datbaseSaved_paths_roots()
                try:

                    self.data_paths_project={name: os.path.join(self.all_paths_for_this_system[self.data_paths_roots[name]], 'LabNY') for name in self.data_paths_names}
                    self.data_paths_project['ResultsContainers']= os.path.join(self.all_paths_for_this_system['Dropbox'], 'LabNY','ResultsContainers')    

        
                except KeyError:
                    self.change_and_save_selected_paths_roots()
                    self.data_paths_project={name: os.path.join(self.all_paths_for_this_system[self.data_paths_roots], 'LabNY') for name in self.data_paths_names}  
            
            
            
            
        else:
            self.data_paths_names=['ResultsContainers']
            self.data_paths_project={name: os.path.join(self.all_paths_for_this_system['Dropbox'], 'LabNY','ResultsContainers') for name in self.data_paths_names}     


     
            
            
        self.database.database_backup()
 
        module_logger.info('Starting Data Managing')
        self.datamanaging=None
        self.gui=[]
        if gui:
            self.gui = Gui(self)
            self.gui.mainloop()
    
    def do_datamanaging(self, full=True):
        self.datamanaging=DataManaging(self, full=full)
    
    def change_and_save_selected_paths_roots(self): 
                        
        self.define_path_roots()           
        query_update_paths="""
                 UPDATE SavedPaths_table
                 SET Path=?
                 WHERE DataFolder=?
                 """
        
        for path_name, path in self.data_paths_roots.items():
            params=(path,path_name)
            self.database.arbitrary_updating_record(query_update_paths, params, commit=True)
        self.load_datbaseSaved_paths_roots()
            
    def load_datbaseSaved_paths_roots(self):
        
        
        
        if Project.computer=='DESKTOP-OKLQSQS':
            query_get_paths="""
                        SELECT *
                        FROM SavedPaths_table
                        """                
        elif Project.computer=='sp3660-YusteLab':
            query_get_paths="""
                        SELECT *
                        FROM SavedPathsLinux_table
                        """                
            
            
        dataPaths=self.database.arbitrary_query_to_df(query_get_paths)
        self.data_paths_roots={name:dataPaths.Path.values.tolist()[i] for i, name  in enumerate(self.data_paths_names)}


    def define_path_roots(self):
        
        self.data_paths_roots={name:select_values_gui(list(self.all_paths_for_this_system.keys()), name) for name  in self.data_paths_names}
        

        

if __name__ == "__main__":
    # execute only if run as a script
    lab=RunNYLab()        
         