# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:48:45 2021

@author: sp3660
"""

from pandastable import Table, TableModel
import tkinter as tk
from tkinter import ttk
import pandas as pd

def open_df_as_table_new_tkinter_frame(gui, frame, table, table_title, update=False):
    pt = Table(frame, dataframe=table, showtoolbar=True, showstatusbar=True)
    if not update:
        pt.show()
    elif update:  
        pt.updateModel(TableModel(table))
        pt.redraw()
    

def open_multiple_df_in_tkinter_frame(app, frame, df_dictionary, update=False):

  if not update  :
      win=frame
      tabControl = ttk.Notebook(win)
      frame.tabs={}  
      df_dictionary.items
      for table_title, table_to_plot in df_dictionary.items():
        frame.tabs[table_title]=ttk.Frame(tabControl)
        tabControl.add( frame.tabs[table_title], text=table_title)
        open_df_as_table_new_tkinter_frame(app, frame.tabs[table_title], table_to_plot, table_title,  update=update)
        tabControl.pack(expand=1, fill="both")
  elif update:  
     for table_title, table_to_plot in df_dictionary.items():       
          open_df_as_table_new_tkinter_frame(app,  frame.tabs[table_title], table_to_plot, table_title,  update=update)
      
      
      

def open_multiple_df_in_new_tkinter_window(app, window_title, df_dictionary):

  tableWindow = tk.Toplevel(app)
  win=tableWindow
  tableWindow.title(window_title)
  tabControl = ttk.Notebook(win)
  tabs=[]  
  df_dictionary.items
  count=0
  for table_title, table_to_plot in df_dictionary.items():
    tabs.append(ttk.Frame(tabControl))
    tabControl.add(tabs[count], text=table_title)
    open_df_as_table_new_tkinter_frame(app, tabs[count], table_to_plot, table_title)
    count=count+1
    tabControl.pack(expand=1, fill="both")    
    
    
    
def button_update_database(gui):
        gui.MouseDat.update_variables()    
        gui.Database_attributes=pd.DataFrame(list(vars( gui.MouseDat).keys()))      
        gui.ExpDatabase_attributes=pd.DataFrame(list(vars( gui.MouseDat.Experimental_class).keys()))
        gui.ImagingDatabase_attributes=pd.DataFrame(list(vars( gui.MouseDat.ImagingDatabase_class).keys()))
       
        gui.database_structure={table[0]:[] for table in  gui.MouseDat.all_tables if table[0]!='sqlite_sequence'}
        for key in  gui.database_structure.keys():
            query_build='select * from ' + key
            cursor =  gui.MouseDat.database_connection.execute(query_build)
            names = [description[0] for description in cursor.description]
            gui.database_structure[key]=names   
        gui.database_structure = pd.DataFrame(list( gui.database_structure.values()), index= gui.database_structure.keys())       
        gui.main_tabs['Mouse Database'].update_table()
        gui.main_tabs['Mouse Experimental'].update_dataframes()
        print('Database Updated')
        