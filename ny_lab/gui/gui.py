# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 11:21:28 2021

@author: sp3660
"""



import datetime
import tkinter as tk
from tkinter import ttk


from .utils import button_update_database


from .tabs.main.mainGuiTab import MainGuiTab
from .tabs.mouseDatabase.mouseDatabaseTab import MouseDatabaseTab
from .tabs.mouseVisit.mouseVisitTab import MouseVisitTab
from .tabs.mouseExperimental.mouseExperimentalTab import MouseExperimentalTab
from .tabs.mouseImaging.mouseImagingTab import MouseImagingTab
from .tabs.dataProcessing.dataProcessingTab import DataProcessingTab
from .tabs.dataAnalysis.dataAnalysisTab import DataAnalysisTab
from .tabs.dataExploration.dataExplorationsTab import DataExplorationsTab



#%% GENERAL GUI CONFIGURATION
class Gui(tk.Tk):
    def __init__(self, lab_object):
        super().__init__()

        # configure the root window
        self.title('LAB App')
        self.geometry('1200x1200')
        self.todays_date=datetime.date.today().strftime("%Y%m%d")
        self.lab=lab_object
        self.datamanaging=self.lab.datamanaging
        self.MouseDat=self.lab.database

#%% TAB DEFINITIONS
        self.main_tabs={}
        self.main_tabControl = ttk.Notebook(self)
        self.main_tabs_names=['General Navigation','Mouse Database','Mouse Visit',
                              'Mouse Experimental','Mouse Imaging','Data Processing',
                              'Data Explorations', 'Data Analysis']
      
        self.main_tabs['General Navigation']=MainGuiTab(self, self.main_tabControl)
        self.main_tabControl.add(self.main_tabs['General Navigation'], text='General Navigation')

        self.main_tabs['Mouse Database']=MouseDatabaseTab(self, self.main_tabControl)
        self.main_tabControl.add(self.main_tabs['Mouse Database'], text='Mouse Database')

        self.main_tabs['Mouse Visit']=MouseVisitTab(self, self.main_tabControl)
        self.main_tabControl.add(self.main_tabs['Mouse Visit'], text='Mouse Visit')

        self.main_tabs['Mouse Experimental']=MouseExperimentalTab(self, self.main_tabControl)
        self.main_tabControl.add(self.main_tabs['Mouse Experimental'], text='Mouse Experimental')
        
        self.main_tabs['Mouse Imaging']=MouseImagingTab(self,self.main_tabControl)
        self.main_tabControl.add(self.main_tabs['Mouse Imaging'], text='Mouse Imaging')
        
        self.main_tabs['Data Processing']=DataProcessingTab(self, self.main_tabControl)
        self.main_tabControl.add(self.main_tabs['Data Processing'], text='Data Processing')
        
        self.main_tabs['Data Explorations']=DataExplorationsTab(self, self.main_tabControl)
        self.main_tabControl.add(self.main_tabs['Data Explorations'], text='Data Explorations')
        
        self.main_tabs['Data Analysis']=DataAnalysisTab(self, self.main_tabControl)
        self.main_tabControl.add(self.main_tabs['Data Analysis'], text='Data Analysis')
        self.main_tabControl.pack(expand=1, fill="both")  
   
        button_update_database(self)

   # self.message_window
    
if __name__ == "__main__":
    
    app = Gui()
    app.mainloop()