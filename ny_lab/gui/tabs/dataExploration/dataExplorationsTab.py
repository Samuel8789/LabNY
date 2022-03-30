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
              
       
        
''' 
TO REVIEW
#%% data exploration
# mouse_codes=datamanaging.all_existing_sessions_not_database_objects[session_name].session_imaged_mice_codes
# imaging_session=mouse_object.imaging_sessions_not_yet_database_objects[session_name]
# os.startfile(imaging_session.mouse_session_path)
 
# acqs=[datamanaging.all_experimetal_mice_objects[mouse_code].all_mouse_acquisitions  for mouse_code in mouse_codes]
# imaging_session=[datamanaging.all_experimetal_mice_objects[mouse_code]  for mouse_code in mouse_codes]
# fullalen=acqs[1][list(acqs[1].keys())[-2]]
# fullalen.face_camera.full_eye_camera.play()
# test=pd.DataFrame(fullalen.voltage_signals_dictionary['Locomotion'])
# test.plot()
# fullalen.metadata_object
# fullalen.all_datasets
# surface=list(fullalen.FOV_object.all_datasets[-1].values())[0]
# surface_green=list(surface.all_datasets.values())[0]
# surface_red=list(surface.all_datasets.values())[0]
# surface_green.summary_images_object.plotting()
# surface_red.summary_images_object.plotting()
# fullalgrenplane1=fullalen.all_datasets[list(fullalen.all_datasets.keys())[0]]
# fullalgrenplane1.kalman_object.dataset_kalman_caiman_movie.play(fr=1000)
# fullalgrenplane1.summary_images_object.plotting()
# # %matplotlib qt
# fullalgrenplane1.most_updated_caiman.cnm_object.estimates.view_components()
# fullalgrenplane1.selected_dataset_mmap_path
# os.startfile(fullalgrenplane1.selected_dataset_mmap_path)
# coord0=list(fullalen.FOV_object.mouse_imaging_session_object.all_0coordinate_Aquisitions.values())[0]
# widef=fullalen.FOV_object.mouse_imaging_session_object.widefield_image[list(fullalen.FOV_object.mouse_imaging_session_object.widefield_image.keys())[0]]
# widef.plot_image()

    #%%
# mice_codes=['SPJA', 'SPJC']
# datamanaging.copy_all_mouse_with_data_to_working_path(mice_codes)

# all_prairie_sessions=datamanaging.all_existing_sessions
# all_database_sessions_objects=datamanaging.all_existing_sessions_database_objects
# all_database_sessions=datamanaging.all_existing_sessions_database

 

# selected_mouse='SPJA'
# selected_mouse_info={'Project':secondary_data_mice_projects[selected_mouse], 
#                  'Path': secondary_data_mice_paths[selected_mouse], 
#                  'Code':selected_mouse,
#                  'Mouse_object': all_experimetal_mice_objects[selected_mouse],
#                  'imaging_sessions':all_experimetal_mice_objects[selected_mouse].imaging_sessions_objects
#                      }
# selected_mouse_info['Mouse_object'].get_all_mouse_FOVdata_datasets()
# mooom=selected_mouse_info['Mouse_object'].all_mouse_FOVdata_datasets
# session=selected_mouse_info['imaging_sessions']['20210624']  
# fov=session.all_FOVs['FOV_1']                   
# fov.all_existing_1050tomato
# fov.all_existing_1050tomato
# dataset=fov.all_aquisitions[list(fov.all_aquisitions.keys())[0]]


       '''
       
       
       
       
       
       
       
       
       