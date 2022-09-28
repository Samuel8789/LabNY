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
from .imagingSessionPanel import ImagingSessionPanel
from .databasaedatasetsExplorer import MouseDatasetsPanel


# from ....data_pre_processing.caimanExtraction import CaimanExtraction

class DataProcessingTab(tk.Frame):
    def __init__(self, gui_object, gui_tab_control):
        super().__init__(gui_tab_control)
        self.gui_ref=gui_object        
        #%%TAB 5 'Data Processing',                        
   
        
        self.frames_names=['InitialPreprocessing',
                           'Deep Processing',
                           ]
        self.frames={}
        for i in range(len(self.frames_names)):
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4,relief='groove')
              
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
              
        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe") 
        self.frames[self.frames_names[1]].grid(row=0, column=1, sticky="nswe") 


#%%'Imaging Data Cleaning Up'  this is for sesssion not in database                      
        
        self.frame1= self.frames[self.frames_names[0]]  
        self.frame1.title=ttk.Label(self.frame1, text=self.frames_names[0], width=40)
        self.frame1.title.grid(column=0, row=0)       

        
        self.frame1.buttons={}
        self.frame1.buttons_names=['Copy ImagingSSH',
                                   'Proces Permanent Folders',
                                   'Open Mouse Session Viewer',
                                   'Add Session To Database',
                                   'Open dataset explorer'
                                     ]
        
        self.frame1.buttons_commands=[self.open_ssh_notepad_and_cmd, 
                                      self.clean_up_raw_imaging_folders_button,
                                      self.open_image_session_viewer_button,
                                      self.add_new_imaging_session_button,
                                      self.open_dataset_explorer_button
                                           ]
        
        for i in range(len( self.frame1.buttons_names)):
              self.frame1.buttons[self.frame1.buttons_names[i]]= ttk.Button(self.frame1 , text=self.frame1.buttons_names[i], command=self.frame1.buttons_commands[i])


        # self.frame1.entries={}
        # self.frame1.entries_names=['Session Path']
        # for i in range(len( self.frame1.entries_names)):
        #     self.frame1.entries[ self.frame1.entries_names[i]]=ttk.Entry( self.frame1 , text='', width=45)

        self.frame1.labels={}
        self.frame1.labels_names=['Unprocessed Sessions']
        for i in range(len( self.frame1.labels_names)):
            self.frame1.labels[ self.frame1.labels_names[i]]=ttk.Label(self.frame1, text=self.frame1.labels_names[i], width=30)


        self.frame1.labels[self.frame1.labels_names[0]].grid(column=0, row=1)       
        self.frame1.buttons[self.frame1.buttons_names[0]].grid(column=0, row=3)
        self.frame1.buttons[self.frame1.buttons_names[1]].grid(column=0, row=4)
        self.frame1.buttons[self.frame1.buttons_names[2]].grid(column=0, row=5)
        self.frame1.buttons[self.frame1.buttons_names[3]].grid(column=0, row=6)
        self.frame1.buttons[self.frame1.buttons_names[4]].grid(column=0, row=7)

    

        self.session_to_process=tk.StringVar()
        values=[]
        self.prairie_session_combobox=ttk.Combobox(self.frame1, values=values, textvariable=self.session_to_process, width=60, postcommand = self.update_datamanaging)   
        # self.prairie_session_combobox=ttk.Combobox(self.frame1, values=list(self.gui_ref.datamanaging.all_new_unprocessed_session.keys()), textvariable=self.session_to_process, width=60)   

        self.prairie_session_combobox.grid(column=0, row=2)
        

        #%% procesing database sessions, add the mouse explore here also
        
        self.frame2= self.frames[self.frames_names[1]]  
        self.frame2.title=ttk.Label(self.frame1, text=self.frames_names[0], width=40)
        self.frame2.title.grid(column=0, row=0)       

        
        self.frame2.buttons={}
        self.frame2.buttons_names=['Proces Permanent Folders',
                                   'Open Mouse Session Viewer',
                                   'Open dataset explorer'
                                     ]
        
        self.frame2.buttons_commands=[self.clean_up_raw_imaging_folders_button,
                                      self.open_image_session_viewer_button,
                                      self.open_dataset_explorer_button
                                       ]
        
        for i in range(len( self.frame2.buttons_names)):
              self.frame2.buttons[ self.frame2.buttons_names[i]]= ttk.Button( self.frame2 , text= self.frame2.buttons_names[i], command= self.frame2.buttons_commands[i])


        # self.frame2.entries={}
        # self.frame2.entries_names=['Session Path']
        # for i in range(len( self.frame2.entries_names)):
        #     self.frame2.entries[ self.frame2.entries_names[i]]=ttk.Entry( self.frame2 , text='', width=45)

        self.frame2.labels={}
        self.frame2.labels_names=['Unprocessed Sessions']
        for i in range(len( self.frame2.labels_names)):
            self.frame2.labels[ self.frame2.labels_names[i]]=ttk.Label( self.frame2, text= self.frame2.labels_names[i], width=30)


        self.frame2.labels[ self.frame2.labels_names[0]].grid(column=0, row=1)       
        self.frame2.buttons[ self.frame2.buttons_names[0]].grid(column=0, row=3)
        self.frame2.buttons[ self.frame2.buttons_names[1]].grid(column=0, row=4)
        self.frame2.buttons[ self.frame2.buttons_names[2]].grid(column=0, row=5)

    

        self.database_session_to_process=tk.StringVar()
        values=[]
        self.prairie_database_session_combobox=ttk.Combobox( self.frame2, values=values, textvariable=self.database_session_to_process, width=60, postcommand = self.update_datamanaging_database)   
        # self.prairie_database_session_combobox=ttk.Combobox( self.frame2, values=list(self.gui_ref.datamanaging.all_new_unprocessed_session.keys()), textvariable=self.database_session_to_process, width=60)   

        self.prairie_database_session_combobox.grid(column=0, row=2)
        
        



#%% buttons
    def reprocess_database_session(self):
        pass

    def update_datamanaging_database(self):
        if self.gui_ref.datamanaging:
            self.prairie_database_session_combobox['values']=list(self.gui_ref.datamanaging.all_database_session.keys())
        else:
            self.prairie_database_session_combobox['values']=[]

    def update_datamanaging(self):
        if self.gui_ref.datamanaging:
            self.prairie_session_combobox['values']=list(self.gui_ref.datamanaging.all_new_unprocessed_session.keys())
        else:
            self.prairie_session_combobox['values']=[]
        

    def open_ssh_notepad_and_cmd(self):
        '''semimanually at the moment
        later add the pipeilen to process automatically
        '''
        session_name=os.path.split( self.session_to_process.get())[1]
        os.startfile(r'G:\Projects\TemPrairireSSH\TranfseringFilesFromPrairireRig.txt')
        os.system("start /B start cmd.exe @cmd /k mycommand...")
        os.startfile(os.path.join(r'G:\Projects\TemPrairireSSH', session_name))

    def clean_up_raw_imaging_folders_button(self):
        print('Started Processing Raw Permanent Folders')
        session_name=self.session_to_process.get()
        prairie_session=self.gui_ref.lab.datamanaging.all_existing_sessions_not_database_objects[session_name]
        # this is the celan up and org, this has to be done first
        prairie_session.process_all_imaged_mice()
        # button_update_database(self.gui_ref)
        print('Finsihed Processing Raw Permanent Folders')

        os.startfile(os.path.join(self.gui_ref.lab.datamanaging.all_new_unprocessed_session[session_name],'Mice'))

    def open_image_session_viewer_button(self):
        session_name=os.path.split( self.session_to_process.get())[1]
        prairie_session=self.gui_ref.lab.datamanaging.all_existing_sessions_not_database_objects[session_name]
        prairie_session.read_all_yet_to_database_mice()
        ImagingSessionPanel(self.gui_ref, session_date=session_name, )
        
    def open_dataset_explorer_button(self):
        MouseDatasetsPanel(self.gui_ref )
 

    def add_new_imaging_session_button(self):
        session=self.session_to_process.get()
        
        self.gui_ref.MouseDat.ImagingDatabase_class.add_new_session_to_database( self.gui_ref, self.gui_ref.lab.datamanaging.all_new_unprocessed_session[session])
        button_update_database(self.gui_ref)
        
    def open_mouse_imaging_viewer_button(self):
        # maybe move to data exploration
        pass
    
    
    def clean_up_all_non_database_sessions(self):
        for k,i in self.gui_ref.lab.datamanaging.all_existing_sessions_not_database_objects.items() :
                i.process_all_imaged_mice()
    
    def reclean_up_all_database_sessions(self):
       
        for i in self.gui_ref.lab.datamanaging.all_existing_sessions_database_objects.values() :
            i.process_all_imaged_mice()
        
        
    #%% doing and checking deep caiman
    # mice_list=['SPKJ']
    # datamanaging.do_deep_caiman_of_mice_datasets(mice_list)

    # datamanaging.all_deep_caiman_objects
    # caimanobject=datamanaging.all_deep_caiman_objects[list(datamanaging.all_deep_caiman_objects.keys())[0]]
    # caimanobject.load_results_object()
    #%%
    # datamanaging.copy_all_mouse_with_data_to_working_path(['SPJC','SPIN' ,'SPKG','SPIL','SPIK'])
    # caimanobject.CaimanResults_object.open_caiman_sorter()

     
#%% explanations
'''

datamanaging.all_existing_sessions
datamanaging.all_existing_unprocessed_sessions

datamanaging.all_existing_sessions_database
datamanaging.all_new_unprocessed_session
datamanaging.all_existing_sessions_database_objects
datamanaging.all_existing_sessions_not_database_objects


datamanaging.all_imaged_mice
datamanaging.all_imaged_mouse_dbinfo
datamanaging.all_imaged_mice_objects

datamanaging.all_experimental_mice
datamanaging.all_exp_mouse_dbinfo


datamanaging.all_experimetal_mice_objects this combine mice in database and mice not in database


# this cekcs session not in daba adn makes a list of the mice there 
datamanaging.non_database_imaged_mice 
# and thus compares with mice already in dab keeping those whic are not in db yes, keeping oinly the new ones
sorted(datamanaging.new_mouse_codes)
#this are the corresponding mice objects. this shouldn have any imaging session inside
datamanaging.all_non_imaged_mice_objects
datamanaging.all_non_imaged_mice_objects['SPJY']
spjy=datamanaging.all_non_imaged_mice_objects['SPJY']





datamanaging.mouse_data_structure_paths
datamanaging.mouse_data_structure_paths_mouse_codes
datamanaging.new_working_directories
# datamanaging.all_working_directories
    #%% pre database image sesionprocesing
    #prairiesessions
    datamanaging.all_new_unprocessed_session
    datamanaging.all_existing_sessions_not_database_objects
    
    prairses=datamanaging.all_existing_sessions_not_database_objects['20220223']
    
    # this is the celan up and org, this has to be done first
    prairses.process_all_imaged_mice
        # gets mice and does mouse.add_prairie_session()
            #mouseimsess.raw_session_preprocessing() this is where all teh procesing is doen

    # this is necessary for the ImagingSessionGui
    prairses.read_all_yet_to_database_mice
        #mouse.read_processed_imaging_session_not_in_database() this loads the mouseimagingsessionobjects
            # read mouseimagingsession
    #this wil ll load everythin very slow, optioanl
    prairses.load_all_yet_to_database_mice
        # mouse.load_processed_imaging_session_not_in_database()
            # acq.load(all)

    prairses.read_all_yet_to_database_mice
    session_name
    ImagingSessionPanel(root, session_date=session_name, datamanaging=datamanaging)


    #mices
    datamanaging.non_database_imaged_mice 
    sorted(datamanaging.new_mouse_codes)
    datamanaging.all_non_imaged_mice_objects
'''
