# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 18:13:55 2023

@author: sp3660
"""
import os
import glob
import shutil
import numpy as np
import pandas as pd
import gc
import sys
import tifffile
import time
import itertools
import sys
sys.path.append(r'C:/Users/sp3660/Documents/Github/LabNY/ny_lab/dataManaging/functions')


from functionsDataOrganization import check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories

def file_cleanup_prairie_dataset(Prairireaqpath):
             
  
  # check channle and plane structure     and current folders
      directory_red=os.path.join(Prairireaqpath,'Ch1Red')
      directory_green=os.path.join(Prairireaqpath, 'Ch2Green')
      
      correction=False  
      ChannelPaths=[directory_red, directory_green]  
      ChannelRedExists=False
      ChannelGreenExists=False
      PlaneNumber=False
      
      # IF Already have done some processing it can be a numbe rof things
      
      if os.path.exists(directory_red) or os.path.exists(directory_green):   
            correction=True  
            #here is soem what probmelatic
            aq_info=check_channels_and_planes(Prairireaqpath, correction)
            
            if os.path.exists(directory_red):
                ChannelRedExists=True            
                folder_selected_list_red = os.listdir(directory_red)
                if any('plane' in file_name  for file_name in folder_selected_list_red if os.path.isdir(os.path.join(directory_red , file_name))):
                    
                    if aq_info[9]:
                        PlaneNumber= aq_info[9]
                    else:
                        PlaneNumber=len(folder_selected_list_red) + aq_info[9]

                elif any('Volume' in file_name  for file_name in folder_selected_list_red if os.path.isdir(os.path.join(directory_red , file_name))):
                    last_cycle=len(folder_selected_list_red) + aq_info[9]
                    PlaneNumber=len(glob.glob(os.path.join(directory_red,folder_selected_list_red[0])+'\\**'))     
                    Multiplane=False
    
                else:
                    aq_info =check_channels_and_planes(directory_red, correction)
                
            if os.path.exists(directory_green):
                ChannelGreenExists=True           
                folder_selected_list_green = os.listdir(directory_green)  
  
                if any('plane' in file_name  for file_name in folder_selected_list_green if os.path.isdir(directory_green + os.sep + file_name)):
                    # PlaneNumber=len(folder_selected_list_green) + aq_info[10]
                    
                    if aq_info[10]:
                        PlaneNumber= aq_info[10]
                    else:
                        PlaneNumber=len(folder_selected_list_green) + aq_info[9]
            
                        
                elif any('Volume' in file_name  for file_name in folder_selected_list_green if os.path.isdir(os.path.join(directory_green , file_name))):
                    last_cycle=len(folder_selected_list_green) + aq_info[10]
                    PlaneNumber=len(glob.glob(os.path.join(directory_green,folder_selected_list_green[0])+'\\**'))     
                
                else:
                   aq_info=check_channels_and_planes(directory_green, correction)
           
  
      else:
          aq_info = check_channels_and_planes(Prairireaqpath, correction)
          
      if aq_info[0]:
            ChannelRedExists=1
            PlaneNumber=aq_info[9]     
            first_cycle=  int(aq_info[3][5:] ) 
            last_cycle=  int(aq_info[4][5:]   )

      if aq_info[1]:
            ChannelGreenExists=1
            PlaneNumber=aq_info[10]
            first_cycle=  int(aq_info[6][5:])
            last_cycle=  int(aq_info[7] [5:] )  
            
      Multiplane=False
      if PlaneNumber>1:
          Multiplane=True     

      ImagedChannels=['No','No']
      if ChannelRedExists:
          ImagedChannels[0]='Ch1Red'
      if ChannelGreenExists:
          ImagedChannels[1]='Ch2Green'
  
       # create necessary folders     
      if (ChannelRedExists or ChannelGreenExists) and PlaneNumber<6:     
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
          # module_logger.info('Moving Files')   
          time.sleep(10)
          if correction:
              if glob.glob(Prairireaqpath+'\\**.tif', recursive=False):             
                  move_files(Prairireaqpath,ChannelPaths,PlanePaths, Multiplane,aq_info[-1] ) 
              for channel_folder in ChannelPaths:
                  if os.path.isdir(channel_folder):
                      file_list_channel = os.listdir(channel_folder)
                      
                      if len (file_list_channel)>3:
                           move_files(channel_folder,ChannelPaths,PlanePaths, Multiplane, aq_info[-1]) 
                      elif len(file_list_channel)<3:  
                           for plane_folder in file_list_channel:
                               if os.path.isdir(plane_folder):
                                   # file_list_plane=os.listdir(plane_folder)
                                   move_files(plane_folder,ChannelPaths,PlanePaths, Multiplane,aq_info[-1] ) 
                
          else:       
              move_files(Prairireaqpath,ChannelPaths,PlanePaths, Multiplane,aq_info[-1]) 
  
          return  [ImagedChannels, PlaneNumber, all_image_sequence_paths]     
      
      elif (ChannelRedExists or ChannelGreenExists) and PlaneNumber>6:    
            all_image_sequence_paths=[]

            CyclesPaths=[os.sep +'Volume'+str(i+1) for i in range(last_cycle)]       
            for ch in ImagedChannels:
                for i, channel_path in enumerate(ChannelPaths):
                      if ch in channel_path :
                          for n,j in enumerate(CyclesPaths):
                              all_image_sequence_paths.append(ChannelPaths[i]+CyclesPaths[n])
                              if not os.path.exists(ChannelPaths[i]+CyclesPaths[n]):
                                  os.makedirs(ChannelPaths[i]+CyclesPaths[n])
                         
            Multiplane=aq_info[8]
            if correction:
                if glob.glob(Prairireaqpath+'\\**.tif', recursive=False):             
                    move_files(Prairireaqpath,ChannelPaths,CyclesPaths, Multiplane,aq_info[-1] ) 
                for channel_folder in ChannelPaths:
                    if os.path.isdir(channel_folder):
                        file_list_channel = os.listdir(channel_folder)
                        
                        if len (file_list_channel)>last_cycle:
                              move_files(channel_folder,ChannelPaths,CyclesPaths, Multiplane, aq_info[-1]) 
                        elif len(file_list_channel)<3:  
                              for volume_folder in file_list_channel:
                                  if os.path.isdir(os.path.join(channel_folder, volume_folder)):
                                      # file_list_plane=os.listdir(plane_folder)
                                      pass
                                      # move_files(os.path.join(channel_folder, volume_folder),ChannelPaths,CyclesPaths, Multiplane,aq_info[-1] ) 
                  
            else:       
                move_files(Prairireaqpath, ChannelPaths, CyclesPaths, Multiplane, aq_info[-1],  is_highstack=True) 
            
            VolumeNumber=last_cycle
    
            return  [ImagedChannels, VolumeNumber, all_image_sequence_paths]     
        
      else:        
          return  [[], False, []]  
      

if __name__ == "__main__":

    file_cleanup_prairie_dataset(r'F:\Projects\LabNY\Imaging\2022\20220523Hakim\Mice\SPKU')