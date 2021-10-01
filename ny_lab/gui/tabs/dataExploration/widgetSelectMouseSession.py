# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:18:39 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk

class WidgetSelectMouseSession(ttk.Frame):
        def __init__(self, gui, frame):
            ttk.Frame.__init__(self, frame) #instead of super
            self.gui_ref=gui
            
            # project selection
            self.project=tk.StringVar()   
            self.project.trace_add('write', self.update_mice) #
            self.project_label=ttk.Label(self, text='Project', width=20)
            self.project_label.grid(column=0, row=0)
            self.project_selection=ttk.Combobox(self, values=self.gui_ref.MouseDat.allprojects.values.tolist(),  textvariable=self.project, width=30)   
            self.project_selection.grid(column=0, row=1)
            # mouse selection
            self.selected_mouse=tk.StringVar()
            self.selected_mouse.trace_add('write', self.update_sessions) ## Add   
            self.mouse_label=ttk.Label(self, text='Experimental Mouse', width=20)
            self.mouse_label.grid(column=1, row=0)
            self.mouse_selection=ttk.Combobox(self, values=[], textvariable=self.selected_mouse, width=30)   
            self.mouse_selection.grid(column=1, row=1)
            
            # session selection
            self.selected_session=tk.StringVar()
            # self.selected_session.trace_add('write', self.get_session) ## Add   
            self.session_label=ttk.Label(self, text='Imaging Session', width=20)
            self.session_label.grid(column=2, row=0)
            self.session_selection=ttk.Combobox(self, values=[], textvariable=self.selected_session, width=30)   
            self.session_selection.grid(column=2, row=1)


        def update_mice(self, *a):
       
            project= int(self.project.get()[0])
            if project ==4:
                self.mouse_selection['values']=self.gui_ref.MouseDat.ImagingDatabase_class.interneuron_imaging_imaged_mice['Code'].tolist()
            if project ==5:
                self.mouse_selection['values']=self.gui_ref.MouseDat.ImagingDatabase_class.interneuron_opto_imaged_mice['Code'].tolist()
            if project ==2:
                self.mouse_selection['values']=self.gui_ref.MouseDat.ImagingDatabase_class.chandelier_imaging_imaged_mice['Code'].tolist()
            if project ==3:
               self.mouse_selection['values']= self.gui_ref.MouseDat.ImagingDatabase_class.chandelier_opto_imaged_mice['Code'].tolist()     
            if project ==7:
               self.mouse_selection['values']=self.gui_ref.MouseDat.ImagingDatabase_class.tigres_imaged_mice['Code'].tolist()

        def update_sessions(self, *a):       
            mouse= self.selected_mouse.get()
            mouse_object=self.gui_ref.datamanaging.all_experimetal_mice_objects[mouse]
            mouse_imaging_sessions=list(mouse_object.imaging_sessions_objects.keys())
            self.session_selection['values']=mouse_imaging_sessions
      