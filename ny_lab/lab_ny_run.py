# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 08:33:15 2021

@author: sp3660
"""
import logging 
logger = logging.getLogger(__name__)
import os 
import sys

from project_manager.ProjectsCLass import Project
#
from .database import MouseDatabase
from .dataManaging.data_managing import DataManaging
from .gui.gui import Gui
from ny_lab.AllFunctions.select_values_gui import select_values_gui



class RunNYLab(Project):    
    def __init__(self,  githubtoken_path, gui=False ):
        Project.__init__(self,'LabNY', githubtoken_path)
        #loadmicedatabase     
        
        self.databasefile=os.path.join(Project.all_paths_for_this_system['Dropbox'],'LabNY', 'MouseDatabase.db')
        self.data_paths_names=['Raw','Pre_proccessed_slow', 'Analysis_Fast_1', 'Analysis_Fast_2']    

        print('Loading Mouse Database')
        self.database=MouseDatabase(self.databasefile, self)   
        #load all mice info       
        self.all_mouse_info=self.database.allEXPERIMENTAL
        self.data_paths_roots={}
        self.load_datbaseSaved_paths_roots()
        try:
            # self.raw_data_path='\\\?\\' + os.path.join(self.all_paths_for_this_system[self.project_raw_data_path],'LabNY')
            # self.main_directory= '\\\?\\' +os.path.join(self.all_paths_for_this_system[self.project_primary_data_path],'LabNY')
            # self.secondary_disk='\\\?\\' +os.path.join(self.all_paths_for_this_system[ self.project_secondary_data_path],'LabNY')
 
            
            self.data_paths_project={name: os.path.join(self.all_paths_for_this_system[self.data_paths_roots[name]], 'LabNY') for name in self.data_paths_names}

        except KeyError:
            self.change_and_save_selected_paths_roots()
            self.data_paths_project={name: os.paht.join(self.all_paths_for_this_system[self.data_paths_roots], 'LabNY') for name in self.data_paths_names}     
            
            
            
        print('Starting Data Managing')
        self.datamanaging=DataManaging(self)
        
    
        if gui:
            self.gui = Gui(self)
            self.gui.mainloop()
        
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
        query_get_paths="""
                    SELECT *
                    FROM SavedPaths_table
                    """                        
        dataPaths=self.database.arbitrary_query_to_df(query_get_paths)
        self.data_paths_roots={name:dataPaths.Path.values.tolist()[i] for i, name  in enumerate(self.data_paths_names)}


    def define_path_roots(self):
        
        self.data_paths_roots={name:select_values_gui(list(self.all_paths_for_this_system.keys()), name) for name  in self.data_paths_names}
        
        

if __name__ == "__main__":
    # execute only if run as a script
    lab=RunNYLab()        
         