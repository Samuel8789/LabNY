# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 13:48:28 2021

@author: sp3660
"""
import dateutil.parser as dparser
import os
from LabNY.AllFunctions.select_values_gui import select_values_gui
from LabNY.AllFunctions.manually_get_some_metadata import manually_get_some_metadata


def further_aquisition_structuring(aquisition_to_process ,image_sequence_paths, mouse_imaging_sessions_path, aquisition_imaging_metadata):

    
    if aquisition_imaging_metadata:
         print('procesing aq metadata')
         aquisition_date=dparser.parse(aquisition_imaging_metadata[0]['Date'],fuzzy=True)
         formated_aquisition_date=aquisition_date.strftime('%Y%m%d')   
         session_path=os.path.join(mouse_imaging_sessions_path, formated_aquisition_date)
         if not os.path.exists(session_path):
              os.mkdir(session_path)
             
         aquisition_name=aquisition_imaging_metadata[0]['AquisitionName']
         aquisition_path=os.path.join(session_path, aquisition_name)
         if not os.path.exists(aquisition_path):
              os.mkdir(session_path)
         
         
         multiplane=aquisition_imaging_metadata[1]['MultiplanePrompt']
         number_of_planes=aquisition_imaging_metadata[1]['Plane Number']
         
         
     
         
     
        
     
        
    else:  
        print('getting manual metadata')
        aquisition_imaging_metadata=[{},{},[]]
        
        aquisition_date=aquisition_to_process[aquisition_to_process.find('/SP')-8:aquisition_to_process.find('/SP')-1]
        formated_aquisition_date=aquisition_date
        aquisition_imaging_metadata[0]['Date']=formated_aquisition_date
        aquisition_imaging_metadata[0]['AquisitionName']=aquisition_to_process[aquisition_to_process.find('/SP')+1:]
        aquisition_name=aquisition_imaging_metadata[0]['AquisitionName']
        aquisition_imaging_metadata[1]['MultiplanePrompt']=select_values_gui(["TSeries ZSeries Element","TSeries ImageSequence Element"])
        mtdata=manually_get_some_metadata()
        aquisition_imaging_metadata[1]['Plane Number']=int(mtdata[0])
        aquisition_imaging_metadata[1]['Volume Number']=int(mtdata[2])
        aquisition_imaging_metadata[1]['Frame Number']=int(mtdata[2])
        aquisition_imaging_metadata[0]['Lines per Frame']=mtdata[1]
        aquisition_imaging_metadata[0]['Pixels Per Line']=mtdata[1]
        

        
        print('finsihed with manual metadata')
        
    return