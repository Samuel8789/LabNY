# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:32:34 2021

@author: sp3660
"""
import pandas as pd
import os
import numpy as np
def load_voltage_recordings(voltage_group, aquisition_to_process, voltage_signal):
    
        
    for j, files_aq in enumerate( os.listdir(aquisition_to_process)):
            if os.path.isfile(aquisition_to_process + os.sep + files_aq):
                if 'csv' in files_aq:
                    df = pd.read_csv(aquisition_to_process + os.sep + files_aq)
                    
                                   
                    index = df.index
                    number_of_rows = len(index)
                    if voltage_signal=='locomotion':
                        df_label=2
                    if voltage_signal== 'visual stim':
                        df_label=1
                    if voltage_signal== 'photoDiode':
                        df_label=3
                
                    df1 = df.iloc[:, np.r_[0, df_label]]
                    nparray=df1.to_numpy()
                    nparray2=nparray.astype('double')
                
                
                    level6_groups={}
                
                    
                    level6_groups[voltage_signal]= voltage_group.create_dataset(voltage_signal,(number_of_rows,2),'double', data=nparray2)
    
    
    
