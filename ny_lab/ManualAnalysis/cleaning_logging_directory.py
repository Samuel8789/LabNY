# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:59:59 2023

@author: sp3660
"""
import os
import glob
from pathlib import Path
import shutil
from datetime import date



def clean_up_logging_dir():
    today = date.today()
    path=r'C:\Users\sp3660\Dropbox\Projects\LabNY\Logging'
    



    all_error_logs=glob.glob(os.path.join(path+'\\app_errors**.log'))
    last_log_number = max(all_error_logs, key=os.path.getctime)[58:-4]

    
    alllogs=glob.glob(os.path.join(path+'\\app_**.log'))
    
    nonerror=[ log for log in all_error_logs if os.path.getsize(log)==0]
    
    nonerrorinex= [log[58:-4] for log in nonerror]
    
    logs_to_delete=[glob.glob(os.path.join(path+f'\\app_{i}.log'))[0] for i in nonerrorinex]
    
    
    if not os.path.isdir(os.path.join(path,today.strftime("%Y%m%d")+'_review')):
        os.mkdir(os.path.join(path,today.strftime("%Y%m%d")+'_review'))
    if not os.path.isdir(os.path.join(path,today.strftime("%Y%m%d")+'_delete')):
        os.mkdir(os.path.join(path,today.strftime("%Y%m%d")+'_delete'))
    
    all_to_delete=nonerror+ logs_to_delete
    
    for i in all_to_delete:
        if last_log_number not in i:
            try:
                shutil.move(i,  os.path.join(os.path.join(path,today.strftime("%Y%m%d")+'_delete'),os.path.split(i)[1]))
            except:
                print(f'File in use {i}')
        
    error_logs=glob.glob(os.path.join(path+'\\app_**.log'))
    
    for i in error_logs:
        if last_log_number not in i:
            try:
        
                shutil.move(i,  os.path.join(os.path.join(path,today.strftime("%Y%m%d")+'_review'),os.path.split(i)[1]))
            except:
                print(f'File in use {i}')
    
    
    
    
    pass
if __name__ == "__main__":
    clean_up_logging_dir()
    
    