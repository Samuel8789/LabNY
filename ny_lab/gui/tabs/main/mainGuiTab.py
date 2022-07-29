# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:06:28 2021

@author: sp3660
"""
import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import time
from ....database import MouseDatabase
import os


class MainGuiTab(tk.Frame):
    def __init__(self, gui_object, gui_tab_control):
        super().__init__(gui_tab_control)
        
        self.gui_ref=gui_object
        

        self.frames_names=['Database Buttons', 
                           'Mouse Map Image' ]
        self.frames={}
        for i in range(len(self.frames_names)):
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4)
    
        self.grid_columnconfigure(0, weight=1)
        # self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe") 
        self.frames[self.frames_names[1]].grid(row=1, column=0, sticky="nswe")

 #%%
         
        self.frame1= self.frames[self.frames_names[0]]
        self.frame1.buttons={}
        self.frame1.buttons_names=['Update Database',
                                   'Close Database',
                                   'Restart Database',
                                   'Do Datamanaging'
                                   
                                   ]
        self.frame1.buttons_commands=[self.button_update_database, 
                                      self.button_close_database, 
                                      self.button_restart_database,
                                      self.do_datamanaging_button
                                      ]
        for i in range(len(self.frame1.buttons_names)):
            self.frame1.buttons[self.frame1.buttons_names[i]]= ttk.Button(self.frame1, text=self.frame1.buttons_names[i], command=self.frame1.buttons_commands[i])
       
        self.frame1.buttons[self.frame1.buttons_names[0]].grid(column=0, row=0)
        self.frame1.buttons[self.frame1.buttons_names[1]].grid(column=0, row=1)
        self.frame1.buttons[self.frame1.buttons_names[2]].grid(column=0, row=2)
        self.frame1.buttons[self.frame1.buttons_names[3]].grid(column=0, row=3)

#%%   
        self.frame2= self.frames[self.frames_names[1]]
        
        file_path=os.path.join(self.gui_ref.lab.dropbox_path, 'Projects','LabNY','mouse_maps.jpg')
        self.img = mpimg.imread(file_path)

        self.frame2.fig = Figure(figsize=(5, 5), dpi=100)
        self.frame2.ax=self.frame2.fig.add_axes([0.1,0.1,0.8,0.8])
        self.frame2.canvas = FigureCanvasTkAgg(self.frame2.fig, master=self.frame2)  # A tk.DrawingArea.
        self.frame2.canvas.draw()
        self.frame2.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)        
        self.frame2.ax.clear()
        self.frame2.ax.imshow( self.img)
        self.frame2.canvas.draw()







#%%
    def do_datamanaging_button(self):
        print('Doing Datamanging')
        t = time.time()
        self.gui_ref.lab.do_datamanaging()
        self.gui_ref.datamanaging= self.gui_ref.lab.datamanaging
        elapsed = time.time() - t

        print('Finsihed Datamanging'+str(elapsed))

       
        

    def button_update_database(self):
            self.gui_ref.MouseDat.update_variables()    
            self.gui_ref.Database_attributes=pd.DataFrame(list(vars( self.gui_ref.MouseDat).keys()))      
            self.gui_ref.ExpDatabase_attributes=pd.DataFrame(list(vars( self.gui_ref.MouseDat.Experimental_class).keys()))
            self.gui_ref.ImagingDatabase_attributes=pd.DataFrame(list(vars( self.gui_ref.MouseDat.ImagingDatabase_class).keys()))
           
            self.gui_ref.database_structure={table[0]:[] for table in  self.gui_ref.MouseDat.all_tables if table[0]!='sqlite_sequence'}
            for key in  self.gui_ref.database_structure.keys():
                query_build='select * from ' + key
                cursor =  self.gui_ref.MouseDat.database_connection.execute(query_build)
                names = [description[0] for description in cursor.description]
                self.gui_ref.database_structure[key]=names   
            self.gui_ref.database_structure = pd.DataFrame(list( self.gui_ref.database_structure.values()), index= self.gui_ref.database_structure.keys())
            self.update_static_stock_table()
            self.gui_ref.main_tabs['Mouse Experimental'].update_dataframes()
            print('Database Updated')
    def button_close_database(self):     
         self.gui_ref.MouseDat.close_database()
         print('Database Closed')   
    def button_restart_database(self):   
         self.gui_ref.MouseDat.close_database()
         self.gui_ref.MouseDat=MouseDatabase( self.gui_ref.lab.databasefile,  self.gui_ref.lab)
         self.button_update_database()
         print('Database Restarted')
    def update_static_stock_table(self) :
         self.gui_ref.main_tabs['Mouse Database'].update_table()
