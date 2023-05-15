# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 18:43:38 2022

@author: sp3660 
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 09:18:34 2022

@author: sp3660
"""
import matplotlib as mpl
import os

import tkinter as tk
from tkinter import Canvas,END, Label, RAISED, Text, WORD, StringVar, Button, ttk, Listbox, Scrollbar
from tkinter import ttk
import  matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from project_manager.ProjectManager import ProjectManager
try :
    from .scrollbarFrame import ScrollbarFrame
except:
    from scrollbarFrame import ScrollbarFrame
import time

import numpy as np
import shutil
import sys
from pprint import pprint
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b",'g','y','c','m', 'tab:brown']) 

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

#%% class definition
class MouseDatasetsPanel(tk.Toplevel):
    def __init__(self,  gui, session_date=None, datamanaging=None):
        tk.Toplevel.__init__(self, gui) #inst
        if datamanaging:
            self.datamanaging=datamanaging
        elif gui:
            self.gui_ref=gui 
            self.datamanaging=self.gui_ref.datamanaging
        self.geometry("+2555+0")
        self.selected_acquisition_object=None
        self.imaging_session=None
        
#%%'Imaging Session' 
        # self.vmax_default=1
        # self.vmax=self.vmax_default
        self.frames_names=['Session Info Frame',
                           'Acquisition Frame', 
                           # 'Message Box'
                           ]
        self.frames={}
        for i in range(len(self.frames_names)):
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4, relief='groove')
              
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=3)
        # self.grid_rowconfigure(2, weight=1)

    
        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe")         
        self.frames[self.frames_names[1]].grid(row=1, column=0, sticky="nswe")        
        # self.frames[self.frames_names[2]].grid(row=2, column=0, sticky="nswe")        


        
#%%'Imaging Session Info Frame' 
       
        self.frame1=self.frames[self.frames_names[0]]
        
        self.frame1.frames_names=['Selections Frame',
                            'Metadata-Info Frame',    
                            'Widefield Frame'
                            ]
        self.frame1.frames={}
        for i in range(len(self.frame1.frames_names)):
              # self.frame1.frames[self.frame1.frames_names[i]]=ttk.Frame(self.frame1, borderwidth = 4, relief='groove')
              self.frame1.frames[self.frame1.frames_names[i]]=ScrollbarFrame(self.frame1, borderwidth = 4, relief='groove')

              
        self.frame1.grid_columnconfigure(0, weight=2)
        self.frame1.grid_columnconfigure(1, weight=1)
        self.frame1.grid_columnconfigure(2, weight=2)
        self.frame1.grid_rowconfigure(0, weight=1)
        # self.frame1.grid_rowconfigure(1, weight=1)

        
        self.frame1.frames[self.frame1.frames_names[0]].grid(row=0, column=0, sticky="nswe")         
        self.frame1.frames[self.frame1.frames_names[1]].grid(row=0, column=1, sticky="nswe") 
        self.frame1.frames[self.frame1.frames_names[2]].grid(row=0, column=2, sticky="nswe") 

        
        
#%%'Imaging Session Info Frame SELECTIONS frame' 

        self.frame1.frame1=self.frame1.frames[self.frame1.frames_names[0]].scrolled_frame
    
        self.frame1.frame1.litbox_names=['Mouse_selection', 
                      'acquisition_selection', 
                      'dataset_selection1',
                       'dataset_selection2',
                       'dataset_selection3',
                       'dataset_selection4',
                       'mock'
                      ]

        self.frame1.frame1.listboxes={}
        self.frame1.frame1.listbox_variables={}
        self.frame1.frame1.scrollbar ={}
        
        for i, litbox_name in enumerate(self.frame1.frame1.litbox_names):
            self.frame1.frame1.listbox_variables[litbox_name]=StringVar()  
            self.frame1.frame1.listboxes[litbox_name]=Listbox(self.frame1.frame1, listvariable=self.frame1.frame1.listbox_variables[litbox_name], width=10, height=10, exportselection=0)
            self.frame1.frame1.scrollbar [litbox_name]= Scrollbar(self.frame1.frame1) 
            self.frame1.frame1.listboxes[litbox_name].config(yscrollcommand =  self.frame1.frame1.scrollbar [litbox_name].set)
            self.frame1.frame1.scrollbar[litbox_name].config(command = self.frame1.frame1.listboxes[litbox_name].yview)
            
        self.get_mouse_imaged() 
        self.get_all_objects()

        
        self.frame1.frame1.listboxes[ self.frame1.frame1.litbox_names[0]].bind('<<ListboxSelect>>', self.get_mouse_info_and_data)
        self.frame1.frame1.listboxes[self.frame1.frame1.litbox_names[1]].bind('<<ListboxSelect>>', self.get_acquisition_info)
        self.frame1.frame1.listboxes[self.frame1.frame1.litbox_names[2]].bind('<<ListboxSelect>>', self.get_dataset_info)
        self.frame1.frame1.listboxes[self.frame1.frame1.litbox_names[3]].bind('<<ListboxSelect>>', self.get_dataset_info)
        self.frame1.frame1.listboxes[self.frame1.frame1.litbox_names[4]].bind('<<ListboxSelect>>', self.get_dataset_info)
        self.frame1.frame1.listboxes[self.frame1.frame1.litbox_names[5]].bind('<<ListboxSelect>>', self.get_dataset_info)




        for i, litbox in    enumerate(self.frame1.frame1.listboxes.values()):
            if i==0:
                litbox.grid(column=0, row=0)
            elif i==1:
                litbox.grid(column=0, row=1)
            elif i==2:
                litbox.grid(column=2, row=0)
            elif i==3:
                litbox.grid(column=2, row=1) 
            elif i==4:
                litbox.grid(column=2, row=2)
            elif i==5:
                litbox.grid(column=2, row=3)
            elif i==6:
                litbox.grid(column=0, row=4) 
                
                
        for i, scrolbar in    enumerate(self.frame1.frame1.scrollbar.values()):
            if i==0:
                scrolbar.grid(column=1, row=0)
            elif i==1:
                scrolbar.grid(column=1, row=1)
            elif i==2:
                scrolbar.grid(column=3, row=0)
            elif i==3:
                scrolbar.grid(column=3, row=1) 
            elif i==4:
                scrolbar.grid(column=3, row=2)
            elif i==5:
                scrolbar.grid(column=3, row=3)  
            elif i==6:
                scrolbar.grid(column=0, row=4) 


        # self.frame1.frame1.buttons={}
        # self.frame1.frame1.buttons_names=['Open acquisition directory',
        #                            'Swicth tomato(if 3plane ac)',  
        #                            'Open dataset directory'
        #                            ]
        # self.frame1.frame1.buttons_commands=[self.open_directory_button, 
        #                                      self.switch_red_button,
        #                                      self.open_dataset_button,
        #                                    ]   
        # for i, butt in enumerate( self.frame1.frame1.buttons_names):
        #       self.frame1.frame1.buttons[ butt]= ttk.Button( self.frame1.frame1 , text= self.frame1.frame1.buttons_names[i], command= self.frame1.frame1.buttons_commands[i])


        # for i, name in enumerate(self.frame1.frame1.buttons_names):
        #        self.frame1.frame1.buttons[name].grid(column=0, row=i+3)     


#%%'Imaging Session Info Frame METADATA frame' 
 
        self.frame1.frame2=self.frame1.frames[self.frame1.frames_names[1]].scrolled_frame
     
  
        self.frame1.frame2.labels={}
        self.frame1.frame2.labels_names=['MouseCode', 
                                  'Age',
                                  'Sex',
                                  'Line',
                                  'Genotype',
                                  'Cage',
                                  'Project',
                                  'Ear Mark',
                                  'Injection Date',
                                  'Days From Injection',
                                  'Window Date',
                                  'Days From Window',
                                  'Virus Combination',
                                  'Dilution Sensor 1',
                                  'Dilution Sensor 2',
                                  'Dilution Opto',
                                  'CoverType',
                                  'Damaged Areas',
                                  'InjectionSite1bleeding',
                                  'InjectionSite2bleeding',
                                  'Notes',
                                  'Injection Notes',
                                  'Window Notes',
                                  ]
        
        for i in range(len(self.frame1.frame2.labels_names)):
            self.frame1.frame2.labels[self.frame1.frame2.labels_names[i]]=ttk.Label(self.frame1.frame2, text=self.frame1.frame2.labels_names[i], width=25)
               
                
        for i, label in enumerate(self.frame1.frame2.labels_names):
            self.frame1.frame2.labels[label].grid(column=0, row=i)            
      
        self.frame1.frame2.labels_values={}
        self.frame1.frame2.labels_values_variables={}
      
        for i ,label in enumerate(self.frame1.frame2.labels_names):
            self.frame1.frame2.labels_values_variables[label]=StringVar()
            self.frame1.frame2.labels_values[label]=ttk.Label(self.frame1.frame2, textvariable= self.frame1.frame2.labels_values_variables[label], width=80)
      
        for i, label in enumerate(self.frame1.frame2.labels_names):
            self.frame1.frame2.labels_values[label].grid(column=1, row=i) 

#%%'Imaging Session Info Frame WIDEFIELD frame' 
        self.frame1.frame3=self.frame1.frames[self.frame1.frames_names[2]].scrolled_frame
        
        self.frame1.frame3.fig = Figure(figsize=(4, 4), dpi=100)
        self.frame1.frame3.ax=self.frame1.frame3.fig.add_axes([0.1,0.1,0.8,0.8])
        self.frame1.frame3.canvas = FigureCanvasTkAgg(self.frame1.frame3.fig, master=self.frame1.frame3)  # A tk.DrawingArea.
        self.frame1.frame3.canvas.draw()
        self.frame1.frame3.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)        
        self.frame1.frame3.toolbar = NavigationToolbar2Tk(self.frame1.frame3.canvas, self.frame1.frame3)
        self.frame1.frame3.toolbar.update()
        self.frame1.frame3.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        

#%%'AcquisitionFrame' 
        self.frame2=self.frames[self.frames_names[1]]
        
        self.frame2.frames_names=['Facecam_Frame',
                            'Voltage_Frame',    
                            'Metadata_Frame',
                            'Planes_Frame',
                            ]
        
        self.frame2.frames={}
        for i in range(len(self.frame2.frames_names)):
              self.frame2.frames[self.frame2.frames_names[i]]=ScrollbarFrame(self.frame2, borderwidth = 4, relief='groove')
              
        self.frame2.grid_columnconfigure(0, weight=1)
        self.frame2.grid_columnconfigure(1, weight=5)
        self.frame2.grid_rowconfigure(0, weight=4)
        self.frame2.grid_rowconfigure(1, weight=1)
        self.frame2.grid_rowconfigure(2, weight=10)
        

        self.frame2.frames[self.frame2.frames_names[0]].grid(row=0, column=0, sticky="nswe")         
        self.frame2.frames[self.frame2.frames_names[1]].grid(row=1, column=0, sticky="nswe") 
        self.frame2.frames[self.frame2.frames_names[2]].grid(row=2, column=0, sticky="nswe") 
        self.frame2.frames[self.frame2.frames_names[3]].grid(row=0, column=1, sticky="nswe", rowspan=3) 


         
#%%'AcquisitionFrame FACECAM frame' 
        self.frame2.frame1=self.frame2.frames[self.frame2.frames_names[0]].scrolled_frame

        self.frame2.frame1.buttons={}
        self.frame2.frame1.buttons_names=['Open facecamera on pyhton',
                                   'Open kalman on pyhton', 
                                   'Open acquisition directory',
                                   'Open dataset directory',
                                   'transfer kalman to desktop',
                                   'transfer eye video to desktop',
                                   'transfer caiman to desktop',
                                   'reset contrast',
                                   'increase contrast 1',
                                   'increase contrast 2',
                                   'open raw directory',
                                   'open slow mouse dir',
                                   'plot signals',
                                   'get acq metadata',
                                   'unload all metdata'
                                    # 'Swicth tomato(if 3plane ac)',  
                                                              ]
                             
        self.frame2.frame1.buttons_commands=[self.open_facecamera_button, 
                                             self.open_kalman_button,
                                             self.open_directory_button, 
                                             self.open_dataset_button,
                                             self.transfer_kalman_to_desktop_button,
                                             self.transfer_facecam_to_desktop_button,
                                             self.transfer_caiman_to_desktop_button,
                                             self.de_enhance,
                                             self.enhance_level_1,
                                             self.enhance_level_2,
                                             self.open_raw_acquisition_directory_button,
                                             self.open_slow_mouse_directory_button,
                                             self.plot_signals_button,
                                             self.get_acq_metdata_button,
                                             self.unload_all_metadata_button
                                             # self.switch_red_button,

                                           ]   
        for i, butt in enumerate( self.frame2.frame1.buttons_names):
              self.frame2.frame1.buttons[ butt]= ttk.Button( self.frame2.frame1 , text= self.frame2.frame1.buttons_names[i], command= self.frame2.frame1.buttons_commands[i])

        for i, name in enumerate(self.frame2.frame1.buttons_names):
              if (i % 2) == 0: 
                 self.frame2.frame1.buttons[name].grid(row=int(i/2)+1, column=0, sticky="nswe")     
              else:  
                  self.frame2.frame1.buttons[name].grid(row=int((i-1)/2)+1, column=1, sticky="nswe")


        self.frame2.frame1.litbox_names=['Dataset to copy selection',                      
                      ]
      
        self.frame2.frame1.listboxes={}
        self.frame2.frame1.listbox_variables={}
        for litbox_name in self.frame2.frame1.litbox_names:
            self.frame2.frame1.listbox_variables[litbox_name]=StringVar()  
            self.frame2.frame1.listboxes[litbox_name]=Listbox(self.frame2.frame1, listvariable=self.frame2.frame1.listbox_variables[litbox_name], width=10, height=10)
       

        self.frame2.frame1.listboxes[self.frame2.frame1.litbox_names[0]].bind('<<ListboxSelect>>', self.select_dataset_to_copy)


        for i, litbox in   enumerate(self.frame2.frame1.listboxes.values()):
           litbox.grid(column=i, row=0)

    



#%%'AcquisitionFrame VOLTAGE frame' 
        self.frame2.frame2=self.frame2.frames[self.frame2.frames_names[1]].scrolled_frame

        # self.frame2.frame2.test_label=ttk.Label( self.frame2.frame2, text='Test voltage', width=20)
        # self.frame2.frame2.test_label.grid(row=10, column=10, sticky="snew")

        self.frame2.frame2.fig = Figure(figsize=(7, 10), dpi=100)
        
        

        self.frame2.frame2.canvas = FigureCanvasTkAgg(self.frame2.frame2.fig, master= self.frame2.frame2)  # A tk.DrawingArea.
        self.frame2.frame2.canvas.draw()
        self.frame2.frame2.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)        
        self.frame2.frame2.toolbar = NavigationToolbar2Tk( self.frame2.frame2.canvas,  self.frame2.frame2)
        self.frame2.frame2.toolbar.update()
        self.frame2.frame2.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
  
        self.frame2.frame2.axs=[]
        for i in range(7):
            self.frame2.frame2.axs.append(self.frame2.frame2.fig.add_subplot(7, 1, i+1))


#%%'AcquisitionFrame METADATA frame' 

        self.frame2.frame3=self.frame2.frames[self.frame2.frames_names[2]].scrolled_frame
        # self.frame2.frame3.frames_names=['Metadata Frame',
        #                     # 'Ref Images Frame',    
        #                     ]        
        # self.frame2.frame3.frames={}
        # for i in range(len(self.frame2.frame3.frames_names)):
        #       self.frame2.frame3.frames[self.frame2.frame3.frames_names[i]]=ScrollbarFrame(self.frame2.frame3)

        # self.frame2.frame3.grid_columnconfigure(0, weight=1)
        # self.frame2.frame3.grid_rowconfigure(0, weight=1)
        # self.frame2.frame3.grid_rowconfigure(1, weight=1)
   
        # self.frame2.frame3.frames[self.frame2.frame3.frames_names[1]].grid(row=1, column=0, sticky="nswe") 
        

        self.frame2.frame3.labels={}
        self.frame2.frame3.labels_names={
                    'PowerSetting',
                    'Objective',
                    'PMT1GainRed',
                    'PMT2GainGreen',
                    'FrameAveraging',
                    'ObjectivePositions',
                    'ETLPositions',
                    'PlaneNumber',
                    'TotalVolumes',
                    'IsETLStack',
                    'IsObjectiveStack',
                    'InterFramePeriod',
                    'FinalVolumePeriod',
                    'FinalFrequency',
                    'TotalFrames',
                    'FOVNumber',
                    'ExcitationWavelength',
                    'CoherentPower',
                    'CalculatedPower',
                    'Comments',
                    'IsChannel1Red',
                    'IsChannel2Green',
                    'IsGalvo',
                    'IsResonant',   
                    'Resolution',
                    'DwellTime',
                    'Multisampling',
                    'BitDepth',
                    'LinePeriod',
                    'FramePeriod',
                    'FullAcquisitionTime',    
                    'RedFilter',
                    'GreenFilter',
                    'DichroicBeamsplitter',
                    'IsBlockingDichroic',
                    'OverlapPercentage',
                    'AtlasOverlap',
                    'OverlapPercentageMetadata',
                    'AtlasDirection',
                    'AtlasZStructure',
                    'AtlasGridSize',
                    'CorrectedObjectivePositions',
                    'CorrectedETLPositions',
                    'ImagingTime',
                    'IsVoltageRecording',
                    'MicronsPerPixelX',
                    'MicronsPerPixelY',
                    'Xpositions',
                    'Ypositions',
                    'Zoom',
                    'VoltageRecordingChannels',
                    'VoltageRecordingFrequency',
                    'ToDoDeepCaiman'}
        
        self.frame2.frame3.labels_names=sorted( self.frame2.frame3.labels_names)

        for i in range(len(self.frame2.frame3.labels_names)):
            self.frame2.frame3.labels[self.frame2.frame3.labels_names[i]]=ttk.Label(self.frame2.frame3, text=self.frame2.frame3.labels_names[i], width=25)
       
        for i, label in enumerate(self.frame2.frame3.labels_names):
           if i<26:            
               self.frame2.frame3.labels[label].grid(column=0, row=i)            
           else:              
               self.frame2.frame3.labels[label].grid(column=2, row=i-26)
          
        self.frame2.frame3.labels_values={}
        self.frame2.frame3.labels_values_variables={}
     
        for i ,label in enumerate(self.frame2.frame3.labels_names):
            self.frame2.frame3.labels_values_variables[label]=StringVar()
            self.frame2.frame3.labels_values[label]=ttk.Label(self.frame2.frame3, textvariable= self.frame2.frame3.labels_values_variables[label], width=25)

     
        for i, label in enumerate(self.frame2.frame3.labels_names):
           if i<26:            
               self.frame2.frame3.labels_values[label].grid(column=1, row=i) 
     
           else:              
               self.frame2.frame3.labels_values[label].grid(column=3, row=i-26)
      
    
   
#%%'AcquisitionFrame datasets frame' 
    
        self.frame2.frame4=self.frame2.frames[self.frame2.frames_names[3]].scrolled_frame
        
        self.frame2.frame4.frames_names=['Dataset1 Frame' ,
                            'Dataset2 Frame',    
                            'Dataset3 Frame',
                            'Dataset4 Frame'
                            ]
        
        self.frame2.frame4.frames={}
        for i, val in enumerate(self.frame2.frame4.frames_names):
              self.frame2.frame4.frames[val]=ttk.Frame(self.frame2.frame4, borderwidth = 4, relief='groove')
              
        self.frame2.frame4.grid_columnconfigure(0, weight=1)
        self.frame2.frame4.grid_columnconfigure(1, weight=1)
        self.frame2.frame4.grid_rowconfigure(0, weight=1)
        self.frame2.frame4.grid_rowconfigure(1, weight=1)
        
        for i, val in enumerate(self.frame2.frame4.frames_names):
            if (i % 2) == 0: 
                self.frame2.frame4.frames[val].grid(row=int(i/2), column=0, sticky="nswe")     
            else:  
                self.frame2.frame4.frames[val].grid(row=int((i-1)/2), column=1, sticky="nswe")     
 #%%'AcquisitionFrame dataset frame images' 
        for dataset_frame in  self.frame2.frame4.frames.values():
            dataset_frame.frames_names=['Std proj',    
                                        'average proj'                                 
                                        ]
            dataset_frame.frames={}
            for i, val in enumerate(dataset_frame.frames_names):
                  dataset_frame.frames[val]=ttk.Frame(dataset_frame, borderwidth = 4, relief='groove')
                  
            dataset_frame.grid_columnconfigure(0, weight=1)
            dataset_frame.grid_columnconfigure(1, weight=1)
            dataset_frame.grid_rowconfigure(0, weight=1)
           
            
            for i, val in enumerate(dataset_frame.frames_names):
                 dataset_frame.frames[val].grid(row=0, column=i, sticky="nswe")  
           
            for frame in dataset_frame.frames.values():
                frame.fig = Figure(figsize=(4, 4), dpi=100)
                frame.ax= frame.fig.add_axes([0.1,0.1,0.8,0.8])
                frame.canvas = FigureCanvasTkAgg( frame.fig, master= frame)  # A tk.DrawingArea.
                frame.canvas.draw()
                frame.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)        
                frame.toolbar = NavigationToolbar2Tk( frame.canvas,  frame)
                frame.toolbar.update()
                frame.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

#%% methods  
    def get_all_objects(self):
        # this is the slowest step, load al metdatas

        self.mice_objects={mouse:{'Object':self.datamanaging.all_experimetal_mice_objects[mouse]} for mouse in self.mice}
        for mouse, m_dic in self.mice_objects.items():
            m_dic['Acquisitions']={acq: {'Object':acq_ob} for acq, acq_ob in  m_dic['Object'].all_mouse_acquisitions.items()}

            if  m_dic['Acquisitions'].items():
                for acq, ac_dic  in  m_dic['Acquisitions'].items():
                    # # print('laoding metdata'+ acq)
                    # # ac_dic['Object'].load_metadata_slow_working_directories()
                    # if not  ac_dic['Object'].voltage_signal_object.no_voltage_signals:
                    #     print('laoding signals'+ acq)
                    #     # ac_dic['Object'].voltage_signal_object.load_slow_storage_voltage_signals()
                    #     # ac_dic['Object'].voltage_signal_object.signal_extraction_object()

                    ac_dic['Datasets']={dtset:{'Object':dtset_ob} for dtset, dtset_ob in  ac_dic['Object'].all_datasets.items()}


 
    def get_mouse_imaged(self):     
        
        self.mice=sorted(list(self.datamanaging.all_imaged_mice_objects.keys()))
        self.frame1.frame1.listbox_variables[self.frame1.frame1.litbox_names[0]].set(self.mice)
 
    def get_all_mouse_acquisitions(self):

        acquisitions=list(self.mice_objects[self.mouse_code]['Acquisitions'].keys())
        self.frame1.frame1.listbox_variables[self.frame1.frame1.litbox_names[1]].set(acquisitions)   
        self.frame1.frame1.listboxes[self.frame1.frame1.litbox_names[1]].config(width=0,height=10)
        self.sessions_with_acquisitions=set([i[0:8] for i in acquisitions])
        self.get_widefield_only_sessions()

        
    def get_widefield_only_sessions(self):
        all_sessions=self.mice_objects[self.mouse_code]['Object'].imaging_sessions_objects
        set(all_sessions.keys())
        
        
        
        if not self.sessions_with_acquisitions or len(self.sessions_with_acquisitions)!=len(all_sessions):
            missing_sessions=self.sessions_with_acquisitions ^ set(all_sessions.keys())
            
            if self.sessions_with_acquisitions:
                # add non aquisition session to aquisition list       
                    self.frame1.frame1.listbox_variables[self.frame1.frame1.litbox_names[1]].set(list(self.mice_objects[self.mouse_code]['Acquisitions'].keys()) + list(missing_sessions))
            else:
                self.frame1.frame1.listbox_variables[self.frame1.frame1.litbox_names[1]].set(list(missing_sessions))

            
        

            
        
        
    def get_acquisition_datasets(self):   
        selected_acq_datasets= list(self.mice_objects[self.mouse_code]['Acquisitions'][self.selected_acquisition]['Datasets'].keys())
        # for listbox_var in   self.frame1.frame1.listbox_variables[self.frame1.frame1.litbox_names[2:]]:
        for var_name in self.frame1.frame1.litbox_names[2:]:
           listbox_var =self.frame1.frame1.listbox_variables[var_name]
           listbox_var.set(selected_acq_datasets)   
           self.frame1.frame1.listboxes[var_name].config(width=0,height=0)

        self.frame2.frame1.listbox_variables[list(self.frame2.frame1.listbox_variables.keys())[0]].set(selected_acq_datasets) 
        self.frame2.frame1.listboxes[list(self.frame2.frame1.listboxes.keys())[0]].config(width=0,height=0)

    def get_dataset_objects(self):
        # self.selected_dataset_object=self.selected_acquisition_object.all_datasets[self.selected_dataset]
        self.selected_dataset_objects={}
        for key, val in  self.selected_datasets.items():
            self.selected_dataset_objects[key]=self.mice_objects[self.mouse_code]['Acquisitions'][self.selected_acquisition]['Datasets'][val]['Object']
                     
    def load_selected_datasets(self):
        
        # self.selected_acquisition_object.load_vis_stim_info()
        
        for key, val in   self.selected_dataset_objects.items():
            val.summary_images_object.load_projections()
        
    def unload_all_datasets(self):
        for key, val in   self.selected_dataset_objects.items():
            val.unload_dataset()
            
    def get_mouse_object(self):
        
        self.selected_mouse_object= self.mice_objects[self.mouse_code]['Object']
        
    def get_acquisition_object(self):
        self.selected_acquisition_object=self.mice_objects[self.mouse_code]['Acquisitions'][self.selected_acquisition]['Object']
        
    def get_mouse_info_and_data(self, event):
        self.reset_dataset_selections_and_projections()
        selection = self.frame1.frame1.listboxes[self.frame1.frame1.litbox_names[0]].curselection()
        if selection:
            index = selection[0]
            self.mouse_code = event.widget.get(index)
            self.get_mouse_object()
            self.get_all_mouse_acquisitions()
            self.get_all_mouse_database_info()
            self.plot_widefield()

        else:
            pass
             
    def get_acquisition_info(self, event):
        self.reset_dataset_selections_and_projections()
        selection = self.frame1.frame1.listboxes[self.frame1.frame1.litbox_names[1]].curselection()
        if selection:
            index = selection[0]
            self.selected_acquisition = event.widget.get(index)
            if len(self.selected_acquisition)>9:
                self.get_acquisition_object()
                self.get_acquisition_datasets()  
                self.get_mouse_imaging_session_from_acquisition()
                self.plot_widefield()
            else:
              self.imaging_session=self.mice_objects[self.mouse_code]['Object'].imaging_sessions_objects[self.selected_acquisition]
              self.selected_acquisition_object=None
              self.plot_widefield()


            # self.get_acq_metadata()
           
            # self.get_ac_facecam()
            # self.get_ac_voltages()
            # self.plot_signals()

        else:
            pass 

    def get_acq_metadata(self):
        # self.selected_acquisition_object.load_metadata_slow_working_directories()
        self.selected_acquisition_object.load_metadata_from_database()       
        self.metadata_dict=self.selected_acquisition_object.metadata_object.translated_imaging_metadata

        for i, label in enumerate(self.frame2.frame3.labels_names):  
            
            if self.metadata_dict[label] is not np.nan: 
                self.frame2.frame3.labels_values_variables[label].set( self.metadata_dict[label])
            else:
                self.frame2.frame3.labels_values_variables[label].set('NA')

    def get_dataset_info(self, event):
        self.selected_datasets={}
        
        for i, var_name in enumerate(self.frame1.frame1.litbox_names[2:]):
    
                listbox=self.frame1.frame1.listboxes[var_name]
      
                selection = listbox.curselection()
                if selection:
                    index = selection[0]
                    self.selected_datasets[self.frame1.frame1.litbox_names[i+2]] = event.widget.get(index)
                    print(self.selected_datasets)
                else:
                    pass
        self.get_dataset_objects()
        self.load_selected_datasets()
        self.plot_dataset_projections()
  
    def get_all_mouse_database_info(self):

        query_all_animals_recovery="""
            SELECT 
                Code, 
                round(round(julianday('now') - julianday(b.DOB))) AS Age,
                aa.Sex_types,
                Line_short,
                G2C_table.Genotypes_types AS G2C,
                Cage,
                d.Projects,
                Labels_types,
                e.InjDate,
                round(round(julianday('now') - julianday(Injection1Date))) AS DaysFromInjection,
                f.WindDate,
                round(round(julianday('now') - julianday(WindowDate))) AS DaysFromWindow,
                Combination,
                e.DilutionSensor1,
                e.DilutionSensor2,
                e.DilutionOpto,
                ab.WindowType,
                f.DamagedAreas,
                e.InjectionSite1bleeding,
                e.InjectionSite2bleeding, 
                a.Notes,
                e.Notes AS InjectionNotes, 
                f.Notes AS WindowNotes,                
                k.Sensors AS Sensors1,
                l.Optos AS Optos1,          
                o.Sensors AS Sensors2,           
                r.Optos AS Optos3, 
                G2C_table.Genotypes_types AS G2C,
                Ai14_table.Genotypes_types AS Ai14,
                Ai75_table.Genotypes_types AS Ai75,
                VRC_table.Genotypes_types AS VRC,
                SLF_table.Genotypes_types AS SLF,
                PVF_table.Genotypes_types AS PVF,
                Ai65_table.Genotypes_types AS Ai65,
                Ai80_table.Genotypes_types AS Ai80,
                VGC_table.Genotypes_types AS VGC,
                Ai162_table.Genotypes_types AS Ai162,
                Ai148_table.Genotypes_types AS Ai148           
            FROM ExperimentalAnimals_table a
            LEFT JOIN MICE_table b ON a.Mouse_ID  = b.ID        
            LEFT JOIN Lines_table c ON c.ID=b.Line
            LEFT JOIN Projects_table d ON d.ID=a.Project
            LEFT JOIN Injections_table e ON e.ExpID = a.ID
            LEFT JOIN Windows_table f ON f.ExpID = a.ID
            LEFT JOIN VirusCombinations_table g ON g.ID=e.VirusCombination
            LEFT JOIN Virus_table h ON h.ID=g.Virus1
            LEFT JOIN Virus_table i ON i.ID=g.Virus2
            LEFT JOIN Virus_table j ON j.ID=g.Virus3
            LEFT JOIN Sensors_table k ON k.ID=h.Sensor
            LEFT JOIN Optos_table l ON l.ID=h.Opto
            LEFT JOIN Promoter_table m ON m.ID=h.Promoter
            LEFT JOIN Recombinase_table n ON n.ID=h.Recombinase
            LEFT JOIN Sensors_table o ON o.ID=i.Sensor
            LEFT JOIN Promoter_table p ON p.ID=i.Promoter
            LEFT JOIN Recombinase_table q ON q.ID=i.Recombinase
            LEFT JOIN Optos_table r ON r.ID=j.Opto
            LEFT JOIN Promoter_table s ON s.ID=j.Promoter
            LEFT JOIN Recombinase_table t ON t.ID=j.Recombinase
            LEFT JOIN Genotypes_table AS G2C_table   ON b.G2C   = G2C_table.ID
            LEFT JOIN Genotypes_table AS Ai14_table   ON b.Ai14   = Ai14_table.ID
            LEFT JOIN Genotypes_table AS Ai75_table   ON b.Ai75   = Ai75_table.ID
            LEFT JOIN Genotypes_table AS VRC_table   ON b.VRC   = VRC_table.ID
            LEFT JOIN Genotypes_table AS SLF_table   ON b.SLF   = SLF_table.ID
            LEFT JOIN Genotypes_table AS PVF_table   ON b.PVF   = PVF_table.ID
            LEFT JOIN Genotypes_table AS Ai65_table   ON b.Ai65   = Ai65_table.ID
            LEFT JOIN Genotypes_table AS Ai80_table   ON b.Ai80   = Ai80_table.ID
            LEFT JOIN Genotypes_table AS VGC_table   ON b.VGC  = VGC_table.ID
            LEFT JOIN Genotypes_table AS Ai162_table   ON b.Ai162   = Ai162_table.ID
            LEFT JOIN Genotypes_table AS Ai148_table   ON b.Ai148  = Ai148_table.ID
            LEFT JOIN Labels_table z on z.ID=a.EarMark 
            LEFT JOIN Sex_table aa on aa.ID=b.Sex
            LEFT JOIN Covertype_table ab on ab.ID=f.CoverType

            WHERE Code IN(?)""" 
    
        params=tuple([self.mouse_code,])
        self.micebrought_info=self.datamanaging.Database_ref.arbitrary_query_to_df(query_all_animals_recovery, params)
        
        for i, label in enumerate(self.frame1.frame2.labels_names):    
            if self.micebrought_info.iloc[0, i]:
                self.frame1.frame2.labels_values_variables[label].set(self.micebrought_info.iloc[0, i])           
            else:
                 self.frame1.frame2.labels_values_variables[label].set('NA')  
                     
    def open_directory_button(self):
        self.open_acquisition_directory_action()
        
    def open_raw_acquisition_directory_button(self):
        self.open_raw_acquisition_directory_action()
        
    def open_slow_mouse_directory_button(self):
        self.open_slow_mouse_directory_action()
        
    def open_dataset_button(self):
        self.open_dataset_directory_action()
            
    def transfer_kalman_to_desktop(self):        
            try:
                if os.path.isfile(self.selected_dataset_to_copy_object.kalman_object.kalman_path):
                    print('trasnefering to desktop kalman')
                    shutil.copyfile(self.selected_dataset_to_copy_object.kalman_object.kalman_path, os.path.join(r'C:\Users\sp3660\Desktop\DatabaseAddingTemporal',os.path.split(self.selected_dataset_to_copy_object.kalman_object.kalman_path)[1]))
                    print('kalman transfered')

                else:
                     print('no klamna')

            except:
                print('kalman not transfered')
       
    def transfer_kalman_to_desktop_button(self):
        self.transfer_kalman_to_desktop()

    def transfer_caiman_to_desktop(self):
            try:
                if os.path.isfile(self.selected_dataset_to_copy_object.most_updated_caiman.caiman_path):

                    print('trasnefering to desktop caiman')
                    shutil.copyfile(self.selected_dataset_to_copy_object.most_updated_caiman.caiman_path, os.path.join(r'C:\Users\sp3660\Desktop\DatabaseAddingTemporal',os.path.split(self.selected_dataset_to_copy_object.most_updated_caiman.caiman_path)[1]))
                    print('caiman transfered')
                else:
                    print('no caiman')

            except:
                print('caiman not transfered')
     
    def transfer_caiman_to_desktop_button(self):
        self.transfer_caiman_to_desktop()

    def transfer_facecam_to_desktop(self):
       
            try:
                if os.path.isfile(self.selected_acquisition_object.face_camera.working_camera_full_path):
                    print('trasnefering to desktop facecam')
                    shutil.copyfile(self.selected_acquisition_object.face_camera.working_camera_full_path,  os.path.join(r'C:\Users\sp3660\Desktop\DatabaseAddingTemporal',os.path.split(self.selected_acquisition_object.face_camera.working_camera_full_path)[1]))
                    print('facecam transfered')
                else:
                    print('no facecam')

            except:
                print('facecam not transfered')

    def transfer_facecam_to_desktop_button(self):
        self.transfer_facecam_to_desktop()

    def select_dataset_to_copy(self, event):

       selection = self.frame2.frame1.listboxes[self.frame2.frame1.litbox_names[0]].curselection()
       if selection:
           index = selection[0]
           self.selected_dataset_to_copy = event.widget.get(index)
           self.selected_dataset_to_copy_object=self.selected_acquisition_object.all_datasets[self.selected_dataset_to_copy]
    
       else:
           pass 
      
    def open_acquisition_directory_action(self):
        
        os.startfile(self.selected_acquisition_object.mouse_aquisition_path)
        
    def open_slow_mouse_directory_action(self):
        
        os.startfile( os.path.join(self.selected_mouse_object.mouse_slow_subproject_path,'imaging', self.imaging_session_name))
               
    def open_raw_acquisition_directory_action(self):
        
        os.startfile(self.selected_acquisition_object.acquisition_database_info['AcquisitonRawPath'][0])
        
    def open_dataset_directory_action(self):
        
        os.startfile(self.selected_dataset_to_copy_object.selected_dataset_mmap_path)

    def open_facecamera_button(self):
        pass
        
    def open_kalman_button(self):
        pass

    def switch_red_button(self):
        pass

    def plot_widefield(self, *a):
        latest_imaging_session=self.selected_mouse_object.imaging_sessions_objects[list(self.selected_mouse_object.imaging_sessions_objects.keys())[-1]]
        widefieldob=None
        if latest_imaging_session.widefield_image:
            widefieldob=latest_imaging_session.widefield_image[list(latest_imaging_session.widefield_image.keys())[0]]      
        
        if self.selected_acquisition_object:
            selected_imaging_session=self.selected_acquisition_object.mouse_imaging_session_object
            if  selected_imaging_session.widefield_image:                                                                                 
                widefieldob=selected_imaging_session.widefield_image[list(selected_imaging_session. widefield_image.keys())[0]]
            
            
        elif self.imaging_session:
            if self.imaging_session.widefield_image:
                
                widefieldob=self.imaging_session. widefield_image[list(self.imaging_session.widefield_image.keys())[0]]
            
             
        if widefieldob:     
            widefieldob.load_all()
            self.frame1.frame3.ax.clear()
            widefieldob.plot_image(self.frame1.frame3.ax)
            widefieldob.plot_rois(self.frame1.frame3.ax)
        
            self.frame1.frame3.canvas.draw()  
        else:
            self.frame1.frame3.ax.clear()
            self.frame1.frame3.canvas.draw()  


                
        
    def plot_signals_button(self):
        t=time.time()
        print('Plotting signals')
        self.selected_acquisition_object.load_voltage_signals()
        self.selected_acquisition_object.voltage_signal_object.signal_extraction_object()


        self.plot_signals()
        elapsed=time.time()-t

        print('Signlas pltted '+ str(elapsed) )
        pass
        
    def plot_signals(self):

        for i in self.frame2.frame2.axs:
            i.clear()
            
        self.frame2.frame2.canvas.draw()  
        dats=[]
        if not self.selected_acquisition_object.voltage_signal_object.no_voltage_signals:
            if hasattr(self.selected_acquisition_object.voltage_signal_object.extraction_object, 'rectified_speed_array'):               
                dats.append(self.selected_acquisition_object.voltage_signal_object.extraction_object.rectified_speed_array['Prairire']['Locomotion'])
            else:
               dats.append(np.array([False]) )
               
            for signal in self.selected_acquisition_object.voltage_signal_object.voltage_signals_dictionary.keys():
                if signal!= 'Time':
                    dats.append(self.selected_acquisition_object.voltage_signal_object.extraction_object.get_specific_signal(signal)['Prairire'][signal])

            # for n, ax in  enumerate(self.frame2.frame2.axs):
            for n, dat in  enumerate(dats):
                    self.frame2.frame2.axs[n].clear()
                    self.frame2.frame2.canvas.draw()  
                    self.frame2.frame2.axs[n].plot(dat)
        else:      
           dats.append(np.array([False]))
           self.frame2.frame2.axs[0].clear()
           self.frame2.frame2.axs[0].plot(dats[0])
           
        self.frame2.frame2.canvas.draw()   
   
    def plot_dataset_projections(self, *a):
        
        for i, dataset in enumerate(self.selected_dataset_objects.values()):
            
            for j, image_frame in enumerate(self.frame2.frame4.frames[self.frame2.frame4.frames_names[i]].frames.values()):
  
                dataset_final=np.array([False])
                image_frame.ax.clear()
                image_frame.canvas.draw()      

                if j==0:
                    dataset_final=dataset.summary_images_object.projection_dic['std_projection_path']
                else:
                    dataset_final=dataset.summary_images_object.projection_dic['average_projection_path']

                if dataset_final.any():
                    image_frame.ax.clear()         
                    image_frame.ax.imshow(dataset_final, cmap='inferno', aspect='equal')
                    image_frame.canvas.draw()  
                    
    def reset_dataset_selections_and_projections(self):
        
        selected_acq_datasets= []
        # for listbox_var in   self.frame1.frame1.listbox_variables[self.frame1.frame1.litbox_names[2:]]:
        for var_name in self.frame1.frame1.litbox_names[2:]:
           listbox_var =self.frame1.frame1.listbox_variables[var_name]
           listbox_var.set(selected_acq_datasets)   
           self.frame1.frame1.listboxes[var_name].config(width=0,height=0)

        self.frame2.frame1.listbox_variables[list(self.frame2.frame1.listbox_variables.keys())[0]].set(selected_acq_datasets) 
        self.frame2.frame1.listboxes[list(self.frame2.frame1.listboxes.keys())[0]].config(width=0,height=0)
        for i in range(len(self.frame2.frame4.frames_names)):
            for j, image_frame in enumerate(self.frame2.frame4.frames[self.frame2.frame4.frames_names[i]].frames.values()):
     
                dataset_final=np.array([False])
                image_frame.ax.clear()
                image_frame.canvas.draw()      







          
    def enhance_level_1(self,*a): 
        self.get_dataset_objects()
        for i, dataset in enumerate(self.selected_dataset_objects.values()):
            
            for j, image_frame in enumerate(self.frame2.frame4.frames[self.frame2.frame4.frames_names[i]].frames.values()):
                dataset_final=np.array([False])
                image_frame.ax.clear()
                image_frame.canvas.draw()      

                if j==0:
                    dataset_final=dataset.summary_images_object.projection_dic['std_projection_path']
                else:
                    dataset_final=dataset.summary_images_object.projection_dic['average_projection_path']

                if dataset_final.any():
                    image_frame.ax.clear()
                    dataset_final.min()    
                    average=dataset_final.mean()
                    std_im=dataset_final.std()
                    vmin=average-8*std_im
                    vmax=average+8*std_im
                    if vmin< dataset_final.min() : vmin = dataset_final.min()
                    if vmax> dataset_final.max() : vmax =  dataset_final.max()
                    image_frame.ax.imshow(dataset_final, cmap='inferno', aspect='equal', vmin=vmin, vmax=vmax) 
                    image_frame.canvas.draw()
                    
    def enhance_level_2(self,*a): 
        self.get_dataset_objects()
        for i, dataset in enumerate(self.selected_dataset_objects.values()):
            
            for j, image_frame in enumerate(self.frame2.frame4.frames[self.frame2.frame4.frames_names[i]].frames.values()):
                dataset_final=np.array([False])
                image_frame.ax.clear()
                image_frame.canvas.draw()      
    
                if j==0:
                    dataset_final=dataset.summary_images_object.projection_dic['std_projection_path']
                else:
                    dataset_final=dataset.summary_images_object.projection_dic['average_projection_path']
    
                if dataset_final.any():
                    image_frame.ax.clear()
                    dataset_final.min()    
                    average=dataset_final.mean()
                    std_im=dataset_final.std()
                    vmin=average-6*std_im
                    vmax=average+6*std_im
                    if vmin< dataset_final.min() : vmin = dataset_final.min()
                    if vmax> dataset_final.max() : vmax =  dataset_final.max()
                    image_frame.ax.imshow(dataset_final, cmap='inferno', aspect='equal', vmin=vmin, vmax=vmax) 
                    image_frame.canvas.draw()
            
    def de_enhance(self,*a): 
        self.get_dataset_objects()
        for i, dataset in enumerate(self.selected_dataset_objects.values()):
            
            for j, image_frame in enumerate(self.frame2.frame4.frames[self.frame2.frame4.frames_names[i]].frames.values()):

                dataset_final=np.array([False])
                image_frame.ax.clear()
                image_frame.canvas.draw()      

                if j==0:
                    dataset_final=dataset.summary_images_object.projection_dic['std_projection_path']
                else:
                    dataset_final=dataset.summary_images_object.projection_dic['average_projection_path']

                if dataset_final.any():
                    image_frame.ax.clear()
                    image_frame.ax.imshow(dataset_final, cmap='inferno', aspect='equal')
                    image_frame.canvas.draw()   
                    
                    
                    
    def get_acq_metdata_button(self):
        self.get_acq_metadata()


    def get_mouse_imaging_session_from_acquisition(self):
        self.imaging_session=self.selected_acquisition_object.mouse_imaging_session_object
        self.imaging_session_name=self.imaging_session.imaging_session_name
        
        
    def unload_all_metadata_button(self): 
        self.unload_all_metadata()
        
    def unload_all_metadata(self):
        self.selected_acquisition_object.unload_all()    
                    
#%% main startup
if __name__ == "__main__":
    from pathlib import Path
    import tkinter as tk
    from sys import platform
    import socket
    from project_manager.ProjectManager import ProjectManager
    import urllib3
    import os
    import pandas as pd


    house_PC='DESKTOP-V1MT0U5'
    lab_PC='DESKTOP-OKLQSQS'
    small_laptop_ubuntu='samuel-XPS-13-9380'
    small_laptop_kali='samuel-XPS-13-9380'
    big_laptop_ubuntu='samuel-XPS-15-9560'
    big_laptop_arch='samuel-XPS-15-9560'

    if platform == "win32":
        if socket.gethostname()==house_PC:
            githubtoken_path=r'C:\Users\Samuel\Documents\Github\GitHubToken.txt'
            computer=house_PC
        elif socket.gethostname()==lab_PC:
            githubtoken_path=r'C:\Users\sp3660\Documents\Github\GitHubToken.txt'
            computer=lab_PC
            
    elif platform == "linux" or platform == "linux2":
        if socket.gethostname()==small_laptop_ubuntu:
            computer=small_laptop_ubuntu
            githubtoken_path='/home/samuel/Documents/Github/GitHubToken.txt'
            # Path('/home/samuel/Documents/Github/GitHubToken.txt')
            print('TO DO')

    ProjectManager=ProjectManager(githubtoken_path, computer, platform)
    gui=0
    lab=ProjectManager.initialize_a_project('LabNY', gui)   
    lab.do_datamanaging()
    datamanaging=lab.datamanaging
    # for msession not in database
 
 
#%% opening the vis app
    root = tk.Tk()
    app = MouseDatasetsPanel(root, datamanaging=datamanaging)
    root.mainloop()
        # test=app.micebrought_info
