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
import datetime
import os

from ...utils import button_update_database  
from .widgetSelectAcquisition import WidgetSelectAcquisition
# from ....data_pre_processing.caimanExtraction import CaimanExtraction

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
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4,relief='groove')
              
        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe") 
        self.frames[self.frames_names[1]].grid(row=1, column=0, sticky="nswe") 
        # self.frames[self.frames_names[2]].grid(row=0, column=0, sticky="nswe") 

        
#%%'Imaging Data Cleaning Up'                        
        
        self.frame1= self.frames[self.frames_names[0]]  
        self.frame1.buttons={}
        self.frame1.buttons_names=['Clean Up Session Folders',
                                   'Add Imaging Session To Database',

                                     ]
        self.frame1.buttons_commands=[self.clean_up_raw_imaging_folders_button, 
                                      self.add_new_imaging_session_button,
            
                                           ]
        for i in range(len( self.frame1.buttons_names)):
              self.frame1.buttons[ self.frame1.buttons_names[i]]= ttk.Button( self.frame1 , text= self.frame1.buttons_names[i], command= self.frame1.buttons_commands[i])

        # self.frame1.entries={}
        # self.frame1.entries_names=['Session Path']
        # for i in range(len( self.frame1.entries_names)):
        #     self.frame1.entries[ self.frame1.entries_names[i]]=ttk.Entry( self.frame1 , text='', width=45)

        self.frame1.labels={}
        self.frame1.labels_names=['Unprocessed Session Path']
        for i in range(len( self.frame1.labels_names)):
            self.frame1.labels[ self.frame1.labels_names[i]]=ttk.Label( self.frame1, text= self.frame1.labels_names[i], width=20)


        self.frame1.labels[ self.frame1.labels_names[0]].grid(column=0, row=0)
        # self.frame1.entries[ self.frame1.entries_names[0]].grid(column=1, row=0)
        # self.frame1.entries[ self.frame1.entries_names[0]].insert(0, r'F:\Projects\LabNY\Imaging\2021\2021MMDD')
                
        self.frame1.buttons[ self.frame1.buttons_names[0]].grid(column=2, row=0)
        self.frame1.buttons[ self.frame1.buttons_names[1]].grid(column=3, row=0)
    



        self.session_to_process=tk.StringVar()
        self.prairie_session_combobox=ttk.Combobox( self.frame1, values= self.gui_ref.datamanaging.all_existing_unprocessed_sessions, textvariable=self.session_to_process, width=60)   
        self.prairie_session_combobox.grid(column=1, row=0)
        
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
        # self.frame2.buttons_commands=[self.process_selected_acquisition, 
        #                               self.process_all_acquisition_from_mouse,
        #                                 ]
        # for i in range(len( self.frame2.buttons_names)):
        #       self.frame2.buttons[ self.frame2.buttons_names[i]]= ttk.Button( self.frame2 , text= self.frame2.buttons_names[i], command= self.frame2.buttons_commands[i])
        
        
        # self.frame2.buttons[ self.frame2.buttons_names[0]].grid(column=0, row=1)
        # self.frame2.buttons[ self.frame2.buttons_names[1]].grid(column=0, row=2)
        
      
        
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
        session=self.session_to_process.get()
        self.gui_ref.MouseDat.ImagingDatabase_class.add_new_session_to_database( self.gui_ref, session)
        button_update_database(self.gui_ref)
        self.gui_ref.datamanaging.all_existing_sessions_not_database_objects[session].load_all_yet_to_database_mice()
        # mouse_codes=datamanaging.all_existing_sessions_not_database_objects[session_name].session_imaged_mice_codes
        # widef=fullalen.FOV_object.mouse_imaging_session_object.widefield_image[list(fullalen.FOV_object.mouse_imaging_session_object.widefield_image.keys())[0]]
        # widef.plot_image()
        
        

    def clean_up_raw_imaging_folders_button(self):
        session=self.session_to_process.get()
        sess=self.gui_ref.lab.datamanaging.load_raw_session(session)
        sess.cleaning_up_raw_mouse_acquisitions()
        # sess.cleaning_up_calibrations(session)
        button_update_database(self.gui_ref)

    # this gos over all prairire imaging session and read all datasets an creates mmap in slow directory folder 
    # Make a gui to select session
    

           
    # def do_all_projections_button(self):     
    #     print('Doing projections')
    #     for key, mouse in self.gui_ref.lab.datamanaging.all_experimetal_mice_objects.items():
    #         mouse.get_all_mouse_acquisitions_datasets()
    #         mouse.get_all_mouse_FOVdata_datasets()
    #         for dataset, data_object in mouse.all_mouse_acquisitions_datasets.items():
    #             data_object.do_projections()
    #         for dataset, data_object in mouse.all_mouse_FOVdata_datasets.items():
    #             data_object.do_projections()
    #     print('pronjections done')
                
    # this process al datasets selected to do kalman and motion corretciuoin
    # Make a gui to select mouse to process and datasets             
                
    # def motion_correct_kalman_button(self):
    #     for key, mouse in self.gui_ref.lab.datamanaging.all_experimetal_mice_objects.items():
    #         mouse.get_all_mouse_acquisitions_datasets()
    #         mouse.get_all_mouse_FOVdata_datasets()
    #         to_ignore=['SurfaceImage','0Coordinate', 'nonimaging','etl','MaxResMech','Tomato','1050', '\Red']               
    #         for dataset, data_object in   mouse.all_mouse_FOVdata_datasets.items():                     
    #                 data_object.create_mot_corrected_kalman_tiff()                   
    #         for dataset, data_object in mouse.all_mouse_acquisitions_datasets.items():
    #             if not any(x in data_object.dataset_full_file_path for x in to_ignore):                   
    #                 data_object.create_mot_corrected_kalman_tiff()
                    
    # def motion_correct_kalman_button_correct(self):
 
    #      for key, mouse in self.gui_ref.lab.datamanaging.all_experimetal_mice_objects.items():
    #          # to_ignore_mouse=['SPJO','SPJP','SPGV','SPGW','SPGX', 'SPIB','SPIL']
    #          # if key not in to_ignore_mouse:
    #              mouse.get_all_mouse_acquisitions_datasets()
    #              mouse.get_all_mouse_FOVdata_datasets()
    #              to_ignore=['SurfaceImage','0Coordinate', 'nonimaging','etl','MaxResMech','Tomato','1050', '\Red']       
    #              for dataset, data_object in   mouse.all_mouse_FOVdata_datasets.items():  
    #                  # if dataset not in ['20210525_FOV_1_210525_SPIN_920_50024_SinglePLane10min-000_Plane1_Green_210525_SPIN_920_50024_SinglePLane10min-000_d1_256_d2_256_d3_1_order_F_frames_10015_.mmap','20210618_FOV_1_210618_SPIN_FOV1_10minspont_920_50024_narrow_without-000_Plane1_Green_210618_SPIN_FOV1_10minspont_920_50024_narrow_without-000_d1_256_d2_256_d3_1_order_F_frames_16951_.mmap', '20210618_FOV_1_210618_SPIN_FOV1_10minspont_920_50024_narrow_without-000_Plane1_Red_210618_SPIN_FOV1_10minspont_920_50024_narrow_without-000_d1_256_d2_256_d3_1_order_F_frames_16951_.mmap']:
    #                      data_object.create_mot_corrected_kalman_tiff(correct=True)                   
    #              for dataset, data_object in mouse.all_mouse_acquisitions_datasets.items():
    #                  if not any(x in data_object.dataset_full_file_path for x in to_ignore):                   
    #                      data_object.create_mot_corrected_kalman_tiff(correct=True)                    
    #      print('processing finished')           
    # # this does caiman on a selected dataset            
                
    # def process_selected_acquisition(self):
    #     dataset=self.frame2.selected_aquisition_frame_widget.selected_dataset.get()
    #     data_object=self.frame2.selected_aquisition_frame_widget.mouse_all_fovs_datasets_dic[dataset]
    #     caim=CaimanExtraction(data_object)
    #     caim.apply_caiman()

    # # this does caiman on all datasets of mouse    

    # def process_all_acquisition_from_mouse(self): 
    #     mouse_datasets=self.frame2.selected_aquisition_frame_widget.mouse_all_fovs_datasets_dic                         
    #     for dataset, data_object in mouse_datasets.items():
    #             caim=CaimanExtraction(data_object)
    #             caim.apply_caiman()