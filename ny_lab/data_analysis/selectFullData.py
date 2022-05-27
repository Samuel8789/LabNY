# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:26:49 2022

@author: sp3660
"""

import tkinter as Tkinter
import random
from tkinter import *
import pandas as pd
import tkinter as tk
import numpy as np

class SelectFullData(tk.Tk):

    
    def __init__(self, analysis_object=None):       
        super().__init__()
        self.title('To DO Select')
        self.geometry('1100x50')
        self.analysis_object=analysis_object
        self.full_data_paths=self.analysis_object.full_data_list

        var = StringVar()
        self.sel_box=ttk.Combobox(self, values= self.full_data_paths, textvariable=var, width=200)       
        self.sel_box.var = tk.StringVar()
        self.sel_box['textvariable'] = self.sel_box.var
        self.sel_box.var.set(self.full_data_paths[0])
        self.sel_box.grid(row=0, column=0) 
        
        self.accept_button=ttk.Button(self, text='Select', command=self.button_selec_full_data)
        self.accept_button.grid(row=1, column=0)    

    def button_selec_full_data(self):
   
        self.selected_full_data= self.sel_box.get()


if __name__ == "__main__":
    
    app = SelectFullData('dd')
    root.mainloop()
    get_values=app.selected_full_data
