# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:20:16 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk

from pandastable import Table

from ...utils import open_multiple_df_in_new_tkinter_window

class MouseDatabaseTab(tk.Frame):
    def __init__(self, gui_object, gui_tab_control):
        super().__init__(gui_tab_control)
        
        #%%TAB 1 'Mouse Database' 
        self.gui_ref=gui_object

        self.frames_names=['Raw Table Buttons', 
                           'Stock Table' ]
        self.frames={}
        for i in range(len(self.frames_names)):
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4)
    
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe") 
        self.frames[self.frames_names[1]].grid(row=0, column=1, sticky="nswe")

#%%TAB 1 'Mouse Database' 'Raw Table Buttons' 
        self.frame1= self.frames[self.frames_names[0]]
      
        self.frame1.buttons={}
        self.frame1.buttons_names=['Raw Database Tables', 'Raw Experimental Tables', 
                                            'Raw Imaging Tables', 'Breedings Tables',
                                            'Basic Tables', 'Stock Tables']
        self.frame1.buttons_commands=[self.button_open_database_raw_tables, self.button_open_experimental_raw_tables, 
                                                self.button_open_imaging_raw_tables, self.button_open_breedings_tables,
                                                self.button_open_basic_database_tables, self.button_open_stock_tables]
        for i in range(len(  self.frame1.buttons_names)):
              self.frame1.buttons[self.frame1.buttons_names[i]]= ttk.Button(self.frame1 , text=self.frame1.buttons_names[i], command=self.frame1.buttons_commands[i])
       

        self.frame1.buttons[self.frame1.buttons_names[0]].grid(column=0, row=0)      
        self.frame1.buttons[self.frame1.buttons_names[1]].grid(column=0, row=1)      
        self.frame1.buttons[self.frame1.buttons_names[2]].grid(column=0, row=2)      
        self.frame1.buttons[self.frame1.buttons_names[3]].grid(column=0, row=3)                            
        self.frame1.buttons[self.frame1.buttons_names[4]].grid(column=0, row=4)                          
        self.frame1.buttons[self.frame1.buttons_names[5]].grid(column=0, row=5)        
       
#%%TAB 1 'Mouse Database' 'Stock Table'  
        self.frame2=self.frames[self.frames_names[1]]
        self.update_table()
  
        
    def update_table(self): 
        self.frame2_table = Table(self.frame2, dataframe=self.gui_ref.MouseDat.stock_mice, showtoolbar=True, showstatusbar=True)
        self.frame2_table.sortTable(columnIndex=1, ascending=1, index=False)
        self.frame2_table.show()
        
    def button_open_database_raw_tables(self):
        window_title="Raw Database Tables"       
        df_dictionary={'All Mice':self.gui_ref.MouseDat.allMICE,
                       'All Actions':self.gui_ref.MouseDat.allACTIONS}     
        open_multiple_df_in_new_tkinter_window(self, window_title, df_dictionary)
 
    def button_open_experimental_raw_tables(self):
        window_title="Raw Experimental Tables"       
        df_dictionary={'All Experimental':self.gui_ref.MouseDat.allEXPERIMENTAL,
                       'All Injections':self.gui_ref.MouseDat.allINJECTIONS, 
                       'All Windows':self.gui_ref.MouseDat.allWINDOWS,
                       'Visrus Stocks': self.gui_ref.MouseDat.allVirusstock
                       
                       }     
        open_multiple_df_in_new_tkinter_window(self, window_title, df_dictionary)
            
    def button_open_imaging_raw_tables(self):
        window_title="Raw Imaging Tables"       
        df_dictionary={'All Imaging Sessions':self.gui_ref.MouseDat.allIMAGINGSESSIONS,
                       'All Imaged Mice':self.gui_ref.MouseDat.allIMAGEDMICE, 
                       'All Acquisitions':self.gui_ref.MouseDat.allACQUISITIONS, 
                       'All Imaging Aqcuisitions':self.gui_ref.MouseDat.allIMAGING, 
                       'All Widefiled':self.gui_ref.MouseDat.allWIDEFIELD, 
                       'All FaceCameras':self.gui_ref.MouseDat.allFACECAMERA, 
                       'All Visual Stimulations':self.gui_ref.MouseDat.allVISSTIMS}     
        open_multiple_df_in_new_tkinter_window(self, window_title, df_dictionary)
                
                 
    def button_open_breedings_tables(self):
        window_title="Breedings Tables"       
        df_dictionary={'Breedings':self.gui_ref.MouseDat.breedings,
                       'All Litters':self.gui_ref.MouseDat.all_litters, 
                       'Current Litters':self.gui_ref.MouseDat.get_litters()}     
        open_multiple_df_in_new_tkinter_window(self, window_title, df_dictionary)
        
    def button_open_basic_database_tables(self):
          
        window_title="Global Tables"       
        df_dictionary={'Colony Mice':self.gui_ref.MouseDat.all_colony_mice,
                       'Actions':self.gui_ref.MouseDat.actions, 
                       'Database Attributes':self.Database_attributes, 
                       'Database Structure':self.database_structure,
                       'ExpDatabase Attributes':self.ExpDatabase_attributes,
                       'ImDatabase Attributes':self.ImagingDatabase_attributes,
                       }     
        open_multiple_df_in_new_tkinter_window(self, window_title, df_dictionary)        
        
    def button_open_stock_tables(self):
        self.slfstock=self.gui_ref.MouseDat.SLF[(self.gui_ref.MouseDat.SLF['Breeders_types']=='Stock') & (self.gui_ref.MouseDat.SLF['Experimental_Status']==1)]
        self.pvfstock=self.gui_ref.MouseDat.PVF[(self.gui_ref.MouseDat.PVF['Breeders_types']=='Stock') & (self.gui_ref.MouseDat.PVF['Experimental_Status']==1)]
        window_title="Global Tables"       
        df_dictionary={'Stock Mice':self.gui_ref.MouseDat.stock_mice,
                       'Mice To Genotype':self.gui_ref.MouseDat.mice_to_genotype, 
                       'Gad Mice':self.gui_ref.MouseDat.Gad,
                       'Gad Stock':self.gui_ref.MouseDat.Gad[(self.gui_ref.MouseDat.Gad['Breeders_types']=='Stock') & (self.gui_ref.MouseDat.Gad['Experimental_Status']==1)],
                       'Ai14 Mice':self.gui_ref.MouseDat.Ai14,
                       'Ai65 Mice':self.gui_ref.MouseDat.Ai65, 
                       'Ai75 Mice':self.gui_ref.MouseDat.Ai75, 
                       'Ai80 Mice':self.gui_ref.MouseDat.Ai80,
                       'SLF Mice':self.gui_ref.MouseDat.SLF, 
                       'SLF Stock':self.slfstock,
                       'SLF Stock Doubles':self.slfstock[self.slfstock['Line_Short']=='VRC::SLF'],
                       'SLF Stock Triples':self.slfstock[self.slfstock['Line_Short']=='VRC::SLF::Ai65'],
                       'PVF Mice':self.gui_ref.MouseDat.PVF,
                       'PVF Stock':self.pvfstock,
                       'PVF Stock Doubles':self.pvfstock[self.pvfstock['Line_Short']=='VRC::PVF'],
                       'PVF Stock Triples':self.pvfstock[self.pvfstock['Line_Short']=='VRC::PVF::Ai65'] ,                     
                       'TIGRE Mice':self.gui_ref.MouseDat.TIGRES,
                       'TIGRE Stock':self.gui_ref.MouseDat.TIGRES[(self.gui_ref.MouseDat.TIGRES['Breeders_types']=='Stock') & (self.gui_ref.MouseDat.TIGRES['Experimental_Status']==1)]}   
        open_multiple_df_in_new_tkinter_window(self, window_title, df_dictionary)   