# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:22:24 2021

@author: sp3660
"""
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg


from ...utils import button_update_database  
from .widgetSelectAcquisition import WidgetSelectAcquisition
from ....data_pre_processing.caimanExtraction import CaimanExtraction

class DataProcessingTab(tk.Frame):
    def __init__(self, gui_object, gui_tab_control):
        super().__init__(gui_tab_control)
        self.gui_ref=gui_object        
        #%%TAB 5 'Data Processing',                        
   
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        
        self.frames_names=['First Data Cleaning',
                           'Caiman'
                           ]
        self.frames={}
        for i in range(len(self.frames_names)):
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4)
              
        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe") 
        self.frames[self.frames_names[1]].grid(row=1, column=0, sticky="nswe") 
        # self.frames[self.frames_names[2]].grid(row=0, column=0, sticky="nswe") 

        
#%%'Imaging Data Cleaning Up'                        
        
        self.frame1= self.frames[self.frames_names[0]]  
        self.frame1.buttons={}
        self.frame1.buttons_names=['Clean Up Session Folders',
                                   'Add Imaging Session To Database',
                                   'Convert All To Mmap',
                                   'Do all motioncorrect/kalman(only fovs, not red)',
                                   'Do all projections'   
                                     ]
        self.frame1.buttons_commands=[self.clean_up_raw_imaging_folders_button, 
                                      self.add_new_imaging_session_button,
                                      self.convert_all_to_mmap_button,
                                      self.motion_correct_kalman_button,
                                      self.do_all_projections_button
                                           ]
        for i in range(len( self.frame1.buttons_names)):
              self.frame1.buttons[ self.frame1.buttons_names[i]]= ttk.Button( self.frame1 , text= self.frame1.buttons_names[i], command= self.frame1.buttons_commands[i])

        self.frame1.entries={}
        self.frame1.entries_names=['Session Path']
        for i in range(len( self.frame1.entries_names)):
            self.frame1.entries[ self.frame1.entries_names[i]]=ttk.Entry( self.frame1 , text='', width=45)

        self.frame1.labels={}
        self.frame1.labels_names=['Session Path']
        for i in range(len( self.frame1.labels_names)):
            self.frame1.labels[ self.frame1.labels_names[i]]=ttk.Label( self.frame1, text= self.frame1.labels_names[i], width=20)


        self.frame1.labels[ self.frame1.labels_names[0]].grid(column=0, row=0)
        self.frame1.entries[ self.frame1.entries_names[0]].grid(column=1, row=0)
        self.frame1.entries[ self.frame1.entries_names[0]].insert(0, r'F:\Projects\LabNY\Imaging\2021\2021MMDD')
                
        self.frame1.buttons[ self.frame1.buttons_names[0]].grid(column=2, row=0)
        self.frame1.buttons[ self.frame1.buttons_names[1]].grid(column=3, row=0)
        self.frame1.buttons[ self.frame1.buttons_names[2]].grid(column=0, row=1)

        self.frame1.buttons[ self.frame1.buttons_names[3]].grid(column=0, row=2)
        self.frame1.buttons[ self.frame1.buttons_names[4]].grid(column=0, row=3)


        """

             process acqusition red channels mmaps
                 do motion correction(no kalman)
                 save motion corrections
                 do average projection
            
        """

        #%% caiman frame
        self.frame2= self.frames[self.frames_names[1]]  
        
        self.frame2.selected_aquisition_frame=ttk.Frame(self.frame2, borderwidth = 4)
        self.frame2.selected_aquisition_frame_widget=WidgetSelectAcquisition(self.gui_ref,self.frame2.selected_aquisition_frame)
        self.frame2.selected_aquisition_frame.grid(column=0, row=0)
        
        
        
        self.frame2.buttons={}
        self.frame2.buttons_names=['Processed Selected Acqusition',
                                   'Process All Acqusisitons',
                                        ]
        self.frame2.buttons_commands=[self.process_selected_acquisition, 
                                      self.process_all_acquisition_from_mouse,
                                        ]
        for i in range(len( self.frame2.buttons_names)):
              self.frame2.buttons[ self.frame2.buttons_names[i]]= ttk.Button( self.frame2 , text= self.frame2.buttons_names[i], command= self.frame2.buttons_commands[i])
        
        
        self.frame2.buttons[ self.frame2.buttons_names[0]].grid(column=0, row=1)
        self.frame2.buttons[ self.frame2.buttons_names[1]].grid(column=0, row=2)
        
       
        
#%% voltage processing

  
#         self.frame1= self.frames[self.frames_names[0]]  
#         self.frame1.buttons={}
#         self.frame1.buttons_names=['Clean Up Session Folders',
#                                         'Add Imaging Session To Database',
#                                         'Convert All To Mmap'
#                                         ]
#         self.frame1.buttons_commands=[self.clean_up_raw_imaging_folders_button, 
#                                            self.add_new_imaging_session_button,
#                                            self.convert_all_to_mmap_button,
#                                            ]
#         for i in range(len( self.frame1.buttons_names)):
#               self.frame1.buttons[ self.frame1.buttons_names[i]]= ttk.Button( self.frame1 , text= self.frame1.buttons_names[i], command= self.frame1.buttons_commands[i])





        
        
# #%% frame 2 image plotting




#         self.frame3= self.frames[self.frames_names[1]]
        
#         file_path=r'C:\Users\sp3660\Desktop\mouse_maps.jpg'
#         self.img = mpimg.imread(file_path)
#         self.fig, self.ax = plt.subplots()
#         self.ax.imshow(self.img)

#         self.chart_type = FigureCanvasTkAgg(self.fig, master=self.frame3)
#         self.chart_type._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)


#%% buttons
    def add_new_imaging_session_button(self):
        sessions=['\\\?\\' +self.frame1.entries[self.frame1.entries_names[0]].get()]
        self.gui_ref.MouseDat.ImagingDatabase_class.add_new_session_to_database(sessions)
        button_update_database(self.gui_ref)


         
    def clean_up_raw_imaging_folders_button(self):
        session=['\\\?\\' + self.frame1.entries[self.frame1.entries_names[0]].get()]
        self.gui_ref.lab.datamanaging.cleaning_up_raw_acquisitions(session)
        self.gui_ref.lab.datamanaging.cleaning_up_calibrations(session)
        button_update_database(self.gui_ref)

        
    def convert_all_to_mmap_button(self):     
        for k,v in self.gui_ref.lab.datamanaging.all_existing_sessions_database_objects.items():
            self.gui_ref.lab.datamanaging.all_existing_sessions_database_objects[k].load_all_imaged_mice()
            
    def do_all_projections_button(self):     
        for key, mouse in self.gui_ref.lab.datamanaging.all_experimetal_mice_objects.items():
            mouse.get_all_mouse_acquisitions_datasets()
            mouse.get_all_mouse_FOVdata_datasets()
            for dataset, data_object in mouse.all_mouse_acquisitions_datasets.items():
                data_object.do_projections()
            for dataset, data_object in mouse.all_mouse_FOVdata_datasets.items():
                data_object.do_projections()
                
                
    def motion_correct_kalman_button(self):
    
        for key, mouse in self.gui_ref.lab.datamanaging.all_experimetal_mice_objects.items():
            mouse.get_all_mouse_acquisitions_datasets()
            mouse.get_all_mouse_FOVdata_datasets()
            to_ignore=['SurfaceImage','0Coordinate', 'nonimaging','etl','MaxResMech','Tomato','1050', '\Red']               
            for dataset, data_object in   mouse.all_mouse_FOVdata_datasets.items():                     
                    data_object.create_mot_corrected_kalman_tiff()                   
            for dataset, data_object in mouse.all_mouse_acquisitions_datasets.items():
                if not any(x in data_object.dataset_full_file_path for x in to_ignore):                   
                    data_object.create_mot_corrected_kalman_tiff()
                
    def process_selected_acquisition(self):
        dataset=self.frame2.selected_aquisition_frame_widget.selected_dataset.get()
        data_object=self.frame2.selected_aquisition_frame_widget.mouse_all_fovs_datasets_dic[dataset]
        caim=CaimanExtraction(data_object)
        caim.apply_caiman()

    def process_all_acquisition_from_mouse(self): 
        mouse_datasets=self.frame2.selected_aquisition_frame_widget.mouse_all_fovs_datasets_dic                         
        for dataset, data_object in mouse_datasets.items():
                caim=CaimanExtraction(data_object)
                caim.apply_caiman()