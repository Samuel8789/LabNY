# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 17:57:08 2021

@author: sp3660
"""

import logging 
logger = logging.getLogger(__name__)


import sys
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/ny_lab')

import ny_lab





gui=1
lab=ny_lab.RunNYLab(gui)
"""
# current errors and excepions
    copiyn to F instead of D
    SPHQ ignored when loading mmaps 
    SPJO/P day 26 had corrupted csv files I CREATED EMPTY VOLTAGE FILES
    added two lines to create metdata and reference folder in create dir structure for datasets
    for motioncorrected calman igonred this list to_ignore=['SurfaceImage','0Coordinate', 'nonimaging','etl','MaxResMech','Tomato','1050', '\Red']  
    test aquisition are indieed processed
    
    TO DO
    read metadata from database
    
    
"""

#%% datamaniging lists

if not gui:
    MouseDat=lab.database
    datamanaging=lab.datamanaging
    
    
    mice_codes=['SPJA', 'SPJC']
    datamanaging.copy_all_mouse_with_data_to_working_path(mice_codes)
    
    
    
    all_prairie_sessions=datamanaging.all_existing_sessions
    all_database_sessions_objects=datamanaging.all_existing_sessions_database_objects
    all_database_sessions=datamanaging.all_existing_sessions_database
    
    all_experimetal_mice_objects=datamanaging.all_experimetal_mice_objects
    all_imaged_mice_df=datamanaging.all_imaged_mice
    primary_data_mice_codes=datamanaging.primary_data_mice_codes
    primary_data_mice_paths=datamanaging.primary_data_mice_paths
    secondary_data_mice_codes=datamanaging.secondary_data_mice_codes
    secondary_data_mice_paths=datamanaging.secondary_data_mice_paths
    secondary_data_mice_projects=datamanaging.secondary_data_mice_projects
    
    
    selected_mouse='SPJA'
    
    selected_mouse_info={'Project':secondary_data_mice_projects[selected_mouse], 
                         'Path': secondary_data_mice_paths[selected_mouse], 
                         'Code':selected_mouse,
                         'Mouse_object': all_experimetal_mice_objects[selected_mouse],
                         'imaging_sessions':all_experimetal_mice_objects[selected_mouse].imaging_sessions_objects
                             }
    selected_mouse_info['Mouse_object'].get_all_mouse_FOVdata_datasets()
    mooom=selected_mouse_info['Mouse_object'].all_mouse_FOVdata_datasets
    session=selected_mouse_info['imaging_sessions']['20210624']  
    fov=session.all_FOVs['FOV_1']                   
    fov.all_existing_1050tomato
    fov.all_existing_1050tomato
    dataset=fov.all_aquisitions[list(fov.all_aquisitions.keys())[0]]
    
    '''  
    
    1st clean up the raw folder
    then add the session to database
    then trnasform raw tiff to mmap and transfer all other files
       
     parairie_session_functions
        load_all_images_as_mmap this is for a single prairie_imaging_session, it create a mouse imaging session and lead all images as mmaps. I still have to solev the issue with very large multiplanes mostly SPHQ and furture multiplanes, se how to organize in raw and also hot to convert to mmap
        
        
        
        
        '''
    

    
    
    
    
    
    #%% git managing TO WORK ON IT
    
    # labRepo=lab.repo_object
    
    # assert not labRepo.is_dirty()  # check the dirty state
    # labRepo.untracked_files  
    
    
    # assert os.path.isdir(labRepo.working_tree_dir)                   # directory with your work files
    # assert labRepo.git_dir.startswith(labRepo.working_tree_dir)  # directory containing the git repository
    
    #               # the commit pointed to by head called master
    # index = labRepo.index
    # count_modified_files = len(labRepo.index.diff(None))
    # count_staged_files = len(labRepo.index.diff("HEAD"))
    # commits = list(labRepo.iter_commits('master'))
    #%% 