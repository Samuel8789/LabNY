# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:23:01 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk

class DataAnalysisTab(tk.Frame):
    def __init__(self, gui_object, gui_tab):
        super().__init__(gui_tab)
        
        #%%TAB 6  'Data Analysis'
   
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.frames_names=['TO DO']
        self.frames={}
        for i in range(len(self.frames_names)):
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4)
              
        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe") 
        
        
        
        """
        
        This is for
            grating tuning
            locomotion tuning
            data plots
            similarity 
            ensemble anlaysis
            patter completion cells
            pupil tuning
            whisker tuning
            
        """