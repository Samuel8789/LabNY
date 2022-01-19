# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 18:27:42 2022

@author: sp3660
"""

# get_aqu_slow_storage_path_from_raw
import os
import glob



def get_slow_working_paths(acquisition_path, slow_mouse_path, working_mouse_path=''):

    SessionDate=acquisition_path[acquisition_path.find('Mice')-9:acquisition_path.find('Mice')-1]
    Prairieimagingname=os.path.split(acquisition_path)[1]
    WideFieldFileName=os.path.split(glob.glob(os.path.join(acquisition_path[:acquisition_path.find('Mice')+9], 'Widefield')+'\\**.tif')[0])[1]
    
    
    
    VisStim_path=os.path.join(os.path.split(acquisition_path)[0],'VisStim')
    VisStimLog=glob.glob( VisStim_path+ '\\**.mat', recursive=False)
    if VisStimLog:
        VisStimLogName=os.path.split(VisStimLog[0])[1]
    else:
        VisStimLogName=''

    mousesessionslowstoragepath= os.path.join(slow_mouse_path,'imaging', SessionDate)
    mousesessionworkingstoragepath= os.path.join(working_mouse_path,'imaging', SessionDate)
    

    
    widefieldslowstoragepath=os.path.join(mousesessionslowstoragepath,'widefield image',WideFieldFileName)   
    widefieldworkingstoragepath=os.path.join(mousesessionworkingstoragepath,'widefield image',WideFieldFileName)  
    
    
    if glob.glob(acquisition_path+'\\**.env', recursive=False):
        imaging_path= os.path.split(glob.glob( acquisition_path+'\\**.env', recursive=False)[0])[0]
               

    
    if 'NonImagingAcquisitions' in acquisition_path:
        if imaging_path:
            aqslowstoragepath=os.path.join(mousesessionslowstoragepath,'nonimaging acquisitions',Prairieimagingname)   
            aqsworkingstoragepath=os.path.join(mousesessionworkingstoragepath,'nonimaging acquisitions',Prairieimagingname)  
        
        else:           
            aqslowstoragepath=os.path.join(mousesessionslowstoragepath,'nonimaging acquisitions', 'Aq_1_NonImaging')   
            aqsworkingstoragepath=os.path.join(mousesessionworkingstoragepath,'nonimaging acquisitions','Aq_1_NonImaging')  
    
       
    if 'Atlas' in acquisition_path:
        aqslowstoragepath=os.path.join(mousesessionslowstoragepath,'atlases',Prairieimagingname)   
        aqsworkingstoragepath=os.path.join(mousesessionworkingstoragepath,'atlases',Prairieimagingname)   
        
        if 'Overview' in acquisition_path:
            aqslowstoragepath=os.path.join(os.path.split(aqslowstoragepath)[0],'Overview',Prairieimagingname)   
            aqsworkingstoragepath=os.path.join(os.path.split(aqsworkingstoragepath)[0],'Overview',Prairieimagingname)    
        elif 'Preview' in acquisition_path:
            aqslowstoragepath=os.path.join(os.path.split(aqslowstoragepath)[0],'Preview',Prairieimagingname)   
            aqsworkingstoragepath=os.path.join(os.path.split(aqsworkingstoragepath)[0],'Preview',Prairieimagingname)     
        elif 'Volume' in acquisition_path:
            aqslowstoragepath=os.path.join(os.path.split(aqslowstoragepath)[0],'AtlasVolume',Prairieimagingname)   
            aqsworkingstoragepath=os.path.join(os.path.split(aqsworkingstoragepath)[0],'AtlasVolume',Prairieimagingname)     
    
    # OptogeneticsID=''
    if 'Calibrations' in acquisition_path:
        IsCalibration=1
        # to add geting sample ID
        
    if 'TestAcquisitions' in acquisition_path:
        aqslowstoragepath=os.path.join(mousesessionslowstoragepath,'test aquisitions',Prairieimagingname)   
        aqsworkingstoragepath=os.path.join(mousesessionworkingstoragepath,'test aquisitions',Prairieimagingname)   
    
    if '0CoordinateAcquisiton' in acquisition_path:
        aqslowstoragepath=os.path.join(mousesessionslowstoragepath,'0Coordinate acquisition',Prairieimagingname)   
        aqsworkingstoragepath=os.path.join(mousesessionworkingstoragepath,'0Coordinate acquisition',Prairieimagingname)                
        
    if 'FOV_' in acquisition_path:
        acquisition_path[acquisition_path.find('FOV_'):acquisition_path.find('FOV_')+5]
        aqslowstoragepath=os.path.join(mousesessionslowstoragepath,'data aquisitions',acquisition_path[acquisition_path.find('FOV_'):acquisition_path.find('FOV_')+5], Prairieimagingname)   
        aqsworkingstoragepath=os.path.join(mousesessionworkingstoragepath,'data aquisitions',acquisition_path[acquisition_path.find('FOV_'):acquisition_path.find('FOV_')+5], Prairieimagingname) 
    
        
        if 'SurfaceImage' in acquisition_path:
            aqslowstoragepath=os.path.join(os.path.split(aqslowstoragepath)[0],'SurfaceImage',Prairieimagingname)   
            aqsworkingstoragepath=os.path.join(os.path.split(aqsworkingstoragepath)[0],'SurfaceImage',Prairieimagingname)                       
        if '1050_Tomato'in acquisition_path:    
            aqslowstoragepath=os.path.join(os.path.split(aqslowstoragepath)[0],'1050_Tomato',Prairieimagingname)   
            aqsworkingstoragepath=os.path.join(os.path.split(aqsworkingstoragepath)[0],'1050_Tomato',Prairieimagingname)      
        if '1050_3PlaneTomato'in acquisition_path:  
            aqslowstoragepath=os.path.join(os.path.split(aqslowstoragepath)[0],'1050_3PlaneTomato',Prairieimagingname)   
            aqsworkingstoragepath=os.path.join(os.path.split(aqsworkingstoragepath)[0],'1050_3PlaneTomato',Prairieimagingname)                                  
        if '1050_HighResStackTomato'in acquisition_path:  
            aqslowstoragepath=os.path.join(os.path.split(aqslowstoragepath)[0],'1050_HighResStackTomato',Prairieimagingname)   
            aqsworkingstoragepath=os.path.join(os.path.split(aqsworkingstoragepath)[0],'1050_HighResStackTomato',Prairieimagingname)                                                  
        if 'HighResStackGreen' in acquisition_path:  
            aqslowstoragepath=os.path.join(os.path.split(aqslowstoragepath)[0],'HighResStackGreen',Prairieimagingname)   
            aqsworkingstoragepath=os.path.join(os.path.split(aqsworkingstoragepath)[0],'HighResStackGreen',Prairieimagingname)                                                                               
        if 'OtherAcq' in acquisition_path:                
            aqslowstoragepath=os.path.join(os.path.split(aqslowstoragepath)[0],'OtherAcq',Prairieimagingname)   
            aqsworkingstoragepath=os.path.join(os.path.split(aqsworkingstoragepath)[0],'OtherAcq',Prairieimagingname)    
            
            
    acq_name=os.path.split(aqslowstoragepath)[1]
    EyeCameraFilename_processed=acq_name+'_full_face_camera.tiff'   
    EyeCameraFilename_processed_metadata=acq_name+'_full_face_camera_metadata.json'               
            
    eyecameraslowstoragepath=os.path.join(aqslowstoragepath,'eye camera', EyeCameraFilename_processed)  
    eyecamerametadataslowstoragepath=os.path.join(aqslowstoragepath,'eye camera', EyeCameraFilename_processed_metadata)  
    
    eyecameraworkingstoragepath=os.path.join(aqsworkingstoragepath,'eye camera', EyeCameraFilename_processed)  
     
    
    visstimslowstoragepath=os.path.join(aqslowstoragepath,'visual stim', VisStimLogName)   
    visstimworkingstoragepath=os.path.join(aqsworkingstoragepath,'visual stim', VisStimLogName)        
    
    
    return {'Aq':aqslowstoragepath, 'WideField':widefieldslowstoragepath,  'EyeCamera':eyecameraslowstoragepath, 'EyeCameraMeta':eyecamerametadataslowstoragepath, 'VisStim':visstimslowstoragepath}


if __name__ == "__main__":
    acquisition_path=r'C:\Users\sp3660\Desktop\20211111\Mice\SPJM\FOV_1\Aq_1\211111_SPJM_FOV1_2planeAllenA_25x_920_50024_narrow_with-000'
    slow_mouse_path=r'K:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Chandelier_Optogenetics\VRC\SLF\Ai65\SPJM'
    test=get_slow_working_paths(acquisition_path,slow_mouse_path)
