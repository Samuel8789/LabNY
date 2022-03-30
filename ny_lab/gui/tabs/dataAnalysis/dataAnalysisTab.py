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
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4,relief='groove')
              
        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe") 
        
        
        
        """
        select acquisition acq and 
        # acq.load_results_analysis()
        
        
        # acq.analysis_object.signals_object.process_all_signals()

        
        # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='MCMC', plane='All')
        # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='dfdt', plane='All')
        # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='MCMC', plane='Plane1')
        # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='dfdt', plane='Plane1')
        # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='MCMC', plane='Plane2')
        # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='dfdt', plane='Plane2')
        # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='MCMC', plane='Plane3')
        # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='dfdt', plane='Plane3')

        
        #%% geting tomato identiti
        # acq.analysis_object.identify_in_pyr()
            
        """
        
        