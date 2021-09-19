# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:25:53 2021

@author: sp3660
"""
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk 

def folder_raw_data_selection(number_of_folders):

    folders_to_process=[]
    for videos in range(number_of_folders):
        root = tk.Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory(parent=root,
                                      initialdir="F:\Imaging",
                                      title='Please select a directory')
        folders_to_process.append(folder_selected)
    return folders_to_process
