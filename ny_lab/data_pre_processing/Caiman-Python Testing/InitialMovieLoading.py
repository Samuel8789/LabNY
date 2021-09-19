# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:02:04 2021

@author: sp3660
"""
import caiman as cm
from PathStructure import PathStructure
import os
from tkinter import filedialog
from tkinter import *

folders_to_process=[]
number_of_folders=4
for videos in range(number_of_folders):
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(parent=root,
                                      initialdir="F:\Imaging",
                                      title='Please select a directory')
    folders_to_process.append(folder_selected)
    
for i, video in enumerate(folders_to_process):
    video_name=os.path.basename(video)
    folder_selected_list = os.listdir(video)    

    directory_green=(video + os.sep + 'Ch2Green')
    for 
    
    
    gcampframes=os.listdir(directory_green)
    

    projectCodePath, projectName, projectDataPath, projectRAWPath, projectTempRAWPath =PathStructure()   
    
    temp_directory=projectTempRAWPath + os.sep + video_name
    if not os.path.exists(temp_directory):
        os.mkdir(temp_directory)
    
    movieframes= [directory_green + os.sep + aquisition for aquisition in gcampframes]
    mv=cm.load(movieframes)
    #%%
    ouput_movie_path=temp_directory + os.sep + video_name+'_Full_GCaMP.tiff'
    mv.save(ouput_movie_path)
    #mv.play()
