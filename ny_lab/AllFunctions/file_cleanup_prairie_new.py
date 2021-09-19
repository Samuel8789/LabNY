# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:01:11 2021

@author: sp3660
"""

import os
from LabNY.AllFunctions.check_channels_and_planes import check_channels_and_planes
from LabNY.AllFunctions.channel_and_plane_of_image import channel_and_plane_of_image
from LabNY.AllFunctions.move_files import move_files

def file_cleanup_prairie_new(aquisition_path):
       
    file_list = os.listdir(aquisition_path)


# check channle and plane structure     and current folders
    directory_red=aquisition_path + os.sep + 'Ch1Red'   
    directory_green=aquisition_path + os.sep + 'Ch2Green'
    


    correction=False  
    ChannelPaths=[directory_red, directory_green]

    ChannelRedExists=False
    ChannelGreenExists=False
    
    if os.path.exists(directory_red) or os.path.exists(directory_green):
        
          correction=True
          
          if os.path.exists(directory_red):
              ChannelRedExists=True            
              folder_selected_list_red = os.listdir(directory_red)

              if any('plane' in file_name  for file_name in folder_selected_list_red if os.path.isdir(directory_red + os.sep + file_name)):
                  
                  PlaneNumber=len(folder_selected_list_red)
                  if PlaneNumber>1:
                      Multiplane=True

              else:
                  ChannelRedExists, _, Multiplane, PlaneNumber=check_channels_and_planes(directory_red)
              
 
          if os.path.exists(directory_green):
              ChannelGreenExists=True           
              folder_selected_list_green = os.listdir(directory_green)  

              if any('plane' in file_name  for file_name in folder_selected_list_green if os.path.isdir(directory_green + os.sep + file_name)):
                  PlaneNumber=len(folder_selected_list_green)
                  if PlaneNumber>1:
                      Multiplane=True
              else:
                  _, ChannelGreenExists, Multiplane, PlaneNumber=check_channels_and_planes(directory_green)
         
    else:
        ChannelRedExists, ChannelGreenExists, Multiplane, PlaneNumber=check_channels_and_planes(aquisition_path)
        
    ImagedChannels=['lolo','lolo']
    if ChannelRedExists:
        ImagedChannels[0]='Ch1Red'
    if ChannelGreenExists:
        ImagedChannels[1]='Ch2Green'
        
    
 # create necessary folders 
    all_image_sequence_paths=[]

    
    PlanePaths=[os.sep +'plane'+str(i+1) for i in range(PlaneNumber)]
    
    
    
    for ch in ImagedChannels:
        for i, channel_path in enumerate(ChannelPaths):
            if ch in channel_path :
                for n in range(PlaneNumber):
                    all_image_sequence_paths.append(ChannelPaths[i]+PlanePaths[n])
                    if not os.path.exists(ChannelPaths[i]+PlanePaths[n]):
                        os.makedirs(ChannelPaths[i]+PlanePaths[n])

# move files                
        
    if correction:
        if len(file_list)>15:
            move_files(aquisition_path,ChannelPaths,PlanePaths, Multiplane) 
        

        for channel_folder in ChannelPaths:
            if os.path.isdir(channel_folder):
                file_list_channel = os.listdir(channel_folder)
                
                if len (file_list_channel)>3:
                     move_files(channel_folder,ChannelPaths,PlanePaths, Multiplane) 
                elif len(file_list_channel)<3:  
                     for plane_folder in file_list_channel:
                         if os.path.isdir(plane_folder):
                             file_list_plane=os.listdir(plane_folder)
                             move_files(plane_folder,ChannelPaths,PlanePaths, Multiplane) 
              
    else:       
        move_files(aquisition_path,ChannelPaths,PlanePaths, Multiplane) 
          
            

    return  [ImagedChannels, PlaneNumber, all_image_sequence_paths]
       
       
            
    

