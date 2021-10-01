# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 12:25:16 2021

@author: sp3660
"""




import tkinter as tk
from tkinter import ttk

class WidgetSelectAcquisition:
        def __init__(self,gui, frame):

            self.frame=frame
            self.gui_ref=gui
            
            self.project=tk.StringVar()   
            self.project.trace_add('write', self.update_mice) #
            
            self.frame.project_label=ttk.Label(self.frame, text='Project', width=20)
            self.frame.project_label.grid(column=0, row=0)
        
            self.frame.project_selection=ttk.Combobox(self.frame, values=self.gui_ref.MouseDat.allprojects.values.tolist(),      textvariable=self.project, width=30)   
            self.frame.project_selection.grid(column=0, row=1)
            
                 
            
            self.selected_mouse=tk.StringVar()
            self.selected_mouse.trace_add('write', self.update_acquisitons) ## Add   
            
            
            self.frame.mouse_label=ttk.Label(self.frame, text='Experimental Mouse', width=20)
            self.frame.mouse_label.grid(column=1, row=0)
            
            self.frame.mouse_selection=ttk.Combobox(self.frame, values=[], textvariable=self.selected_mouse, width=30)   
            self.frame.mouse_selection.grid(column=1, row=1)
            
      
            self.selected_session=tk.StringVar()
            self.selected_session.trace_add('write', self.update_acquisitons) ## Add   
            
            
            self.frame.session_label=ttk.Label(self.frame, text='Imaging Session', width=20)
            self.frame.session_label.grid(column=2, row=0)
            
            self.frame.session_selection=ttk.Combobox(self.frame, values=[], textvariable=self.selected_session, width=30)   
            self.frame.session_selection.grid(column=2, row=1)
            
            
            
            self.selected_dataset=tk.StringVar()
            # self.selected_session.trace_add('write', self.update_aquisitions) ## Add   
            
            
            self.frame.dataset_label=ttk.Label(self.frame, text='Dataset', width=20)
            self.frame.dataset_label.grid(column=0, row=2)
            
            self.frame.dataset_selection=ttk.Combobox(self.frame, values=[], textvariable=self.selected_dataset, width=200)   
            self.frame.dataset_selection.grid(column=0, row=3)
            
            
            
            
            
        def update_mice(self, *a):
       
            project= int(self.project.get()[0])
            if project ==4:
                 self.frame.mouse_selection['values']=self.gui_ref.MouseDat.ImagingDatabase_class.interneuron_imaging_imaged_mice['Code'].tolist()
            if project ==5:
                 self.frame.mouse_selection['values']=self.gui_ref.MouseDat.ImagingDatabase_class.interneuron_opto_imaged_mice['Code'].tolist()
            if project ==2:
                 self.frame.mouse_selection['values']=self.gui_ref.MouseDat.ImagingDatabase_class.chandelier_imaging_imaged_mice['Code'].tolist()
            if project ==3:
                self.frame.mouse_selection['values']= self.gui_ref.MouseDat.ImagingDatabase_class.chandelier_opto_imaged_mice['Code'].tolist()     
            if project ==7:
                self.frame.mouse_selection['values']=self.gui_ref.MouseDat.ImagingDatabase_class.tigres_imaged_mice['Code'].tolist()

        def update_sessions(self, *a):       
            mouse= self.selected_mouse.get()
            mouse_object=self.gui_ref.datamanaging.all_experimetal_mice_objects[mouse]
            mouse_imaging_sessions=list(mouse_object.imaging_sessions_objects.keys())
            self.frame.session_selection['values']=mouse_imaging_sessions
            
        def update_acquisitons(self, *a):
            mouse_object=self.gui_ref.datamanaging.all_experimetal_mice_objects[self.selected_mouse.get()]
            mouse_object.get_all_mouse_acquisitions_datasets()
            mouse_object.get_all_mouse_FOVdata_datasets()
            self.mouse_all_fovs_datasets_dic=mouse_object.all_mouse_FOVdata_datasets
            # mouse_all_datasets=mouse_object.all_mouse_acquisitions_datasets
            mouse_all_fovs_list=list(self.mouse_all_fovs_datasets_dic.keys())
            self.frame.dataset_selection['values']=[x for x in mouse_all_fovs_list if 'Red' not in x]