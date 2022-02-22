# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:51:53 2022

@author: sp3660
"""
import os

def transform_path(path, fast_output=False, slow_ouput=False):
    
    if path[0]=='K':
        fast_path1='\\\\?\\'+ r'G:\Projects\LabNY\Working_Mice_Data_2'+path[46:]
        fast_path2='\\\\?\\'+ r'C:\Users\sp3660\Documents\Projects\LabNY\Working_Mice_Data_1'+path[46:]
        
        if fast_output and not slow_ouput:
            if os.path.isfile(fast_path1) or os.path.isdir(fast_path1):
                new_path=fast_path1
            elif os.path.isfile(fast_path2) or os.path.isdir(fast_path2):
                new_path=fast_path2
        else:
            new_path=path
          
    else:
        slow_path='\\\\?\\'+ r'K:\Projects\LabNY\Full_Mice_Pre_Processed_Data' +fast_path1[fast_path1.find('Working_Mice')+19:]
        if slow_ouput and not fast_output:
            new_path=slow_path
        else:
            new_path=path

    return new_path
   