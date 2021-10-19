# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:23:01 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from .widgetSelectMouseSession import WidgetSelectMouseSession
import  matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

class DataExplorationsTab(tk.Frame):
    def __init__(self, gui_object, gui_tab):
        super().__init__(gui_tab)
        
        self.gui_ref=gui_object
        #%%TAB 6  'Data Analysis'
   
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)

        self.frames_names=['Widefield 1',
                           'Widefield 2'
                           ]
        self.frames={}
        for i in range(len(self.frames_names)):
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4,relief='groove')
              
        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe") 
        self.frames[self.frames_names[1]].grid(row=1, column=0, sticky="nswe") 
        
        #%%
        self.mouse_session_select_1=self.frames[self.frames_names[0]]
        self.mouse_session_select_1.session_select_widget=WidgetSelectMouseSession(self.gui_ref, self.mouse_session_select_1)
        self.mouse_session_select_1.session_select_widget.grid(row=0, column=0, sticky="nswe") 
        self.mouse_session_select_1.selected_session=tk.StringVar()
        self.mouse_session_select_1.session_select_widget.selected_session.trace_add('write', self.get_info_1)
        self.mouse_session_select_1_widefield_frame=ttk.Frame(self.mouse_session_select_1, borderwidth = 4,relief='groove')
        self.mouse_session_select_1_widefield_frame.grid(row=1, column=0, sticky="nswe") 
        self.mouse_session_select_1_widefield_frame.fig = Figure(figsize=(5, 5), dpi=100)
        self.mouse_session_select_1_widefield_frame.ax=self.mouse_session_select_1_widefield_frame.fig.add_axes([0.1,0.1,0.8,0.8])
        self.mouse_session_select_1_widefield_frame.canvas = FigureCanvasTkAgg(self.mouse_session_select_1_widefield_frame.fig, master=self.mouse_session_select_1_widefield_frame)  # A tk.DrawingArea.
        self.mouse_session_select_1_widefield_frame.canvas.draw()
        self.mouse_session_select_1_widefield_frame.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)        
        self.mouse_session_select_1_widefield_frame.toolbar = NavigationToolbar2Tk(self.mouse_session_select_1_widefield_frame.canvas, self.mouse_session_select_1_widefield_frame)
        self.mouse_session_select_1_widefield_frame.toolbar.update()
        self.mouse_session_select_1_widefield_frame.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        

  #%%     
        self.mouse_session_select_2=self.frames[self.frames_names[1]]
        self.mouse_session_select_2.session_select_widget=WidgetSelectMouseSession(self.gui_ref, self.mouse_session_select_2)
        self.mouse_session_select_2.session_select_widget.grid(row=0, column=0, sticky="nswe") 
        self.mouse_session_select_2.selected_session=tk.StringVar()
        self.mouse_session_select_2.session_select_widget.selected_session.trace_add('write', self.get_info_2)
        self.mouse_session_select_2_widefield_frame=ttk.Frame(self.mouse_session_select_2, borderwidth = 4,relief='groove')
        self.mouse_session_select_2_widefield_frame.grid(row=1, column=0, sticky="nswe") 
        self.mouse_session_select_2_widefield_frame.fig = Figure(figsize=(5, 5), dpi=100)
        self.mouse_session_select_2_widefield_frame.ax=self.mouse_session_select_2_widefield_frame.fig.add_axes([0.1,0.1,0.8,0.8])
        self.mouse_session_select_2_widefield_frame.canvas = FigureCanvasTkAgg(self.mouse_session_select_2_widefield_frame.fig, master=self.mouse_session_select_2_widefield_frame)  # A tk.DrawingArea.
        self.mouse_session_select_2_widefield_frame.canvas.draw()
        self.mouse_session_select_2_widefield_frame.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)   

        self.mouse_session_select_2_widefield_frame.toolbar = NavigationToolbar2Tk(self.mouse_session_select_2_widefield_frame.canvas, self.mouse_session_select_2_widefield_frame)
        self.mouse_session_select_2_widefield_frame.toolbar.update()
        
    def get_info_1(self, *a):
        self.mouse_session_select_1.selected_mouse=self.mouse_session_select_1.session_select_widget.selected_mouse.get()
        self.mouse_session_select_1.selected_session=self.mouse_session_select_1.session_select_widget.selected_session.get()
        self.gui_ref.datamanaging.all_experimetal_mice_objects[self.mouse_session_select_1.selected_mouse
                                                               ].imaging_sessions_objects[self.mouse_session_select_1.selected_session
                                                                                          ].widefield_image[self.mouse_session_select_1.selected_mouse+self.mouse_session_select_1.selected_session+'_Widefield'
                                                                                                            ].load_image()
        self.gui_ref.datamanaging.all_experimetal_mice_objects[self.mouse_session_select_1.selected_mouse
                                                         ].imaging_sessions_objects[self.mouse_session_select_1.selected_session
                                                                                    ].all_FOVs
                                                                
        self.gui_ref.datamanaging.all_experimetal_mice_objects[self.mouse_session_select_1.selected_mouse
                                                         ].imaging_sessions_objects[self.mouse_session_select_1.selected_session
                                                                                    ].all_0coordinate_Aquisitions
                                                                                                            
                                                                                                            
        self.mouse_session_select_1_widefield_frame_widefield=self.gui_ref.datamanaging.all_experimetal_mice_objects[self.mouse_session_select_1.selected_mouse].imaging_sessions_objects[self.mouse_session_select_1.selected_session].widefield_image[self.mouse_session_select_1.selected_mouse+self.mouse_session_select_1.selected_session+'_Widefield'].widefield_image
        self.mouse_session_select_1_widefield_frame.ax.clear()
        self.mouse_session_select_1_widefield_frame.ax.imshow(self.mouse_session_select_1_widefield_frame_widefield, cmap='inferno')
        self.mouse_session_select_1_widefield_frame.canvas.draw()
           
    def get_info_2(self, *a):
        self.mouse_session_select_2.selected_mouse=self.mouse_session_select_2.session_select_widget.selected_mouse.get()
        self.mouse_session_select_2.selected_session=self.mouse_session_select_2.session_select_widget.selected_session.get()
        self.gui_ref.datamanaging.all_experimetal_mice_objects[self.mouse_session_select_2.selected_mouse].imaging_sessions_objects[self.mouse_session_select_2.selected_session].widefield_image[self.mouse_session_select_2.selected_mouse+self.mouse_session_select_2.selected_session+'_Widefield'].load_image()
        self.mouse_session_select_2_widefield_frame_widefield=self.gui_ref.datamanaging.all_experimetal_mice_objects[self.mouse_session_select_2.selected_mouse].imaging_sessions_objects[self.mouse_session_select_2.selected_session].widefield_image[self.mouse_session_select_2.selected_mouse+self.mouse_session_select_2.selected_session+'_Widefield'].widefield_image
        self.mouse_session_select_2_widefield_frame.ax.clear()
        self.mouse_session_select_2_widefield_frame.ax.imshow(self.mouse_session_select_2_widefield_frame_widefield, cmap='inferno')
        self.mouse_session_select_2_widefield_frame.canvas.draw()
              
       
       
       
       
       
       
       
       
       
       