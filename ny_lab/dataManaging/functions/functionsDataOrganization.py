# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:19:22 2021

@author: sp3660
"""

import os
import xml.etree.ElementTree as ET
import glob
import shutil
import xml
import gc

from distutils.dir_util import copy_tree
from pyometiff import OMETIFFReader
import pathlib



    
def move_all_files_from_source_to_dest(source,dest):
    file_list = glob.glob(source+os.sep+'**')
    for fname in file_list:
        shutil.move(fname,os.path.join(dest,os.path.split(fname)[1]))


def open_directory(foldername):
    systems = {
        'nt': lambda foldername: os.startfile(foldername),
        'posix': lambda foldername: os.system('xdg-open "%s"' % foldername),
        'os2': lambda foldername: os.system('open "%s"' % foldername)
         }
    
    systems.get(os.name)(foldername)
    return foldername

    
def read_mouse_name_from_aquisition(aquisition_folder):
    aquisition_name=os.path.basename(aquisition_folder)
    mouse_name=aquisition_name[0:4]
    return mouse_name

def recursively_copy_changed_files_and_directories_from_slow_to_fast(slow_path, fast_path):

    if os.path.isdir(slow_path) and not os.path.isdir(fast_path):  
        copy_tree(slow_path,fast_path)
        
    if os.path.isdir(slow_path) and os.path.isdir(fast_path):

        directories_slow=[i for i in os.listdir(slow_path) if os.path.isdir(os.path.join(slow_path, i))]
        files_slow = [ file for file in os.listdir(slow_path) if os.path.isfile(os.path.join(slow_path, file))]
        directories_fast=[ i for i in os.listdir(fast_path) if os.path.isdir(os.path.join(fast_path, i))]
        files_fast = [ file for file in os.listdir(fast_path) if os.path.isfile(os.path.join(fast_path, file))]


        for file in files_slow:
            if file not in files_fast:
                shutil.copyfile(os.path.join(slow_path,file), os.path.join(fast_path,file))
        for direc in directories_slow:
             if direc not in directories_fast:
                 recursively_copy_changed_files_and_directories_from_slow_to_fast(os.path.join(slow_path,direc), os.path.join(fast_path,direc))

def recursively_eliminate_empty_folders(path_to_check):
    # print('removing empty folders')
    if os.path.isdir(path_to_check):      
              
        is_files=any(os.path.isfile(os.path.join(path_to_check, i)) for i in os.listdir(path_to_check))           
        is_directories=any(os.path.isdir(os.path.join(path_to_check, i)) for i in os.listdir(path_to_check))
        directories=[os.path.join(path_to_check, i) for i in os.listdir(path_to_check) if os.path.isdir(os.path.join(path_to_check, i))]
        
        if is_directories:
            for dirs in directories:
                recursively_eliminate_empty_folders(dirs)        
            is_directories=any(os.path.isdir(os.path.join(path_to_check, i)) for i in os.listdir(path_to_check)) 
            
        if not (is_files or is_directories):
            os.rmdir(path_to_check)
            
def recursively_delete_back_directories(directory):

    if len(os.listdir(directory))== 0:
         os.rmdir(directory) 
         recursively_delete_back_directories(os.path.split(directory)[0]) 
             
def short_metadata_type_check(aq_path):
    # print('Fast checking metadata')
    imaging_metadata_file=os.path.join(aq_path,os.path.split(aq_path)[1]+'.xml')  
    
    
    try: 
        tree = ET.parse( imaging_metadata_file)   
          
        root = tree.getroot()
        seqinfo=root.find('Sequence')  
        aquisition_type=seqinfo.attrib['type']
        if aquisition_type=='TSeries ZSeries Element':
            framenumber=len(root.findall('Sequence'))
            planenumber=len(seqinfo.findall('Frame'))
        elif aquisition_type=='AtlasVolume': 
            framenumber=len(root.findall('Sequence'))
            planenumber=len(seqinfo.findall('Frame'))     
        elif aquisition_type!='TSeries ZSeries Element':  
            planenumber=1
            framenumber=len(seqinfo.findall('Frame'))   
            
    except:
        print(f'Corrupted imaging netadata{imaging_metadata_file}')

        aquisition_type=''
        framenumber=0
        planenumber=0

    return aquisition_type, framenumber, planenumber, 

def channel_and_plane_of_image(single_image_full_path, multiplane, aq_type):
    Ch=False
    Plane=False
    if '_Ch1_' in single_image_full_path:
        Ch='Ch1Red'
    if '_Ch2_' in single_image_full_path:
        Ch='Ch2Green'
        
    if aq_type=='TSeries ZSeries Element' or  aq_type=='ZSeries':
        Plane=int(single_image_full_path[-14:-8])          
    elif aq_type=='AtlasVolume':
        Plane=int(single_image_full_path[-14:-8])
    else:
        Plane=1   
        
    cycle=int((single_image_full_path)[-24:-19])
         
    return Ch, Plane, cycle


def move_files(current_directory, ChannelPaths, PlanePaths, Multiplane, aq_type, is_highstack=False):
    
    file_list = os.listdir(current_directory)
    
    if not is_highstack:
        for i, fname in enumerate(file_list):
            file_path=os.path.join(current_directory,fname)
            if os.path.isfile(file_path):
                if file_path.endswith('.tif'):
                    Ch, Plane,_ = channel_and_plane_of_image(current_directory + os.sep + fname, Multiplane, aq_type)
                    if Ch:
                        ChannelPath = [i for i in ChannelPaths if Ch in i] 
                        
                        if Plane<=len(PlanePaths):
                            new_directory=ChannelPath[0] + PlanePaths[Plane-1]                            
                            shutil.move(os.path.join(current_directory,fname),os.path.join(new_directory,fname))
    else:
        for i, fname in enumerate(file_list):
            file_path=os.path.join(current_directory,fname)
            if os.path.isfile(file_path):
                if file_path.endswith('.tif'):
                    Ch, Plane, cycle = channel_and_plane_of_image(current_directory + os.sep + fname, Multiplane, aq_type)
                    if Ch:
                        ChannelPath = [i for i in ChannelPaths if Ch in i] 
                        if cycle==len(PlanePaths):
                            new_directory=ChannelPath[0] + PlanePaths[cycle-1]
                            shutil.move(os.path.join(current_directory,fname),os.path.join(new_directory,fname))
                        
                        
        pass
                            



def check_channels_and_planes(image_sequence_directory_full_path, correction=False):
    
    if image_sequence_directory_full_path==r'F:\Projects\LabNY\Imaging\2022\20221021\Mice\SPNN\FOV_1\Aq_1\221021_SPNN_FOV1_AllenA_25x_940_51020_60745_with-000':
        print('Error')
    
    # print('Cheking file structure')    
    sq_type=[]
    sq_type=list(short_metadata_type_check(image_sequence_directory_full_path))
    
    cleaneduplist=glob.glob(image_sequence_directory_full_path+os.sep+'**Cycle**.tif')

    # if 'TSeries ZSeries Element' separate by plane
    # if 'Single'
    # if 'TSeries Timed Element' separate by channles
    # if 'ZSeries' separate by channels only
       
    ChannelRedExists=False
    ChannelGreenExists=False
    Multiplane=False
    RedPlaneNumber=0
    GreenPlaneNumber=0   
    ChannelRedExists=any(glob.glob(image_sequence_directory_full_path+os.sep+'**_Ch1_**', recursive=False))
    ChannelGreenExists=any(glob.glob(image_sequence_directory_full_path+os.sep+'**_Ch2_**', recursive=False))  
       
    possible_channels=2
    if sq_type[0]!='':
        pass
    else:
        alltiffs=glob.glob(image_sequence_directory_full_path+os.sep+'**.tif')

        if cleaneduplist and os.path.getsize(cleaneduplist[0])>200000:
            
            
    
            img_fpath = pathlib.Path(alltiffs[0])
    
            
        elif os.path.isdir(os.path.join(image_sequence_directory_full_path,'Ch1Red')):
            
            img_fpath = pathlib.Path(glob.glob(os.path.join(image_sequence_directory_full_path,'Ch1Red','plane1')+os.sep+'**.tif')[0])
            
        else :
            
            img_fpath = pathlib.Path(glob.glob(os.path.join(image_sequence_directory_full_path,'Ch2Green','plane1')+os.sep+'**.tif')[0])
    
    
            
            
        reader = OMETIFFReader(fpath=img_fpath)

        img_array, metadata, xml_metadata = reader.read()
        del reader, img_array, xml_metadata
        gc.collect()

        

        if metadata['SizeZ']>1:
            sq_type[0]='ZSeries' 
            sq_type[0]='TSeries ZSeries Element' 
           
        else:
            
            sq_type[0]='Single' 
            sq_type[0]='TSeries Timed Element' 
       
        sq_type[2]=metadata['SizeZ']
        if alltiffs:
            if alltiffs[-1][-9]==4 or  alltiffs[-1][-10]:
                alltiffs.pop(-1)
            elif alltiffs[-1][-10]==4:
                alltiffs.pop(-1)
                alltiffs.pop(-1)
            """
            HERE TO IMPROE AND MAKE THIS GOOD FOR
            """
            sq_type[1]=len(alltiffs)// metadata['SizeZ']//metadata['SizeC']
            sq_type[1]=int(alltiffs[-1][-24:-19])
        else:
            sq_type[1]=0
            
       
    
            
        

        
        
    possible_frames=sq_type[1]
    possible_planes=sq_type[2]
    if sq_type[0]=='TSeries ZSeries Element' or sq_type[0]=='ZSeries':
        Multiplane=True
   
            
    
    moviestructure={}


   
        
    RedPlaneNumber=0
    FirstRedPlane=0
    LastRedPlane= 0
    GreenPlaneNumber=0
    FirstGreenPlane=0
    LastGreenPlane=0    
    RedFrameNumber=0
    RedFirstFrame=0
    RedLastFrame=0
    GreenFrameNumber=0
    GreenFirstFrame=0
    GreenLastFrame=0
                

    
    # here I have to add an option to detect the extra files when doing the multiplane
    if cleaneduplist:
        if len(cleaneduplist)<4: 
            if cleaneduplist[0][cleaneduplist[0].find('.ome')-1]=='4' and possible_planes==3:
                cleaneduplist=[]
            elif cleaneduplist[0][cleaneduplist[0].find('.ome')-1]=='3' and possible_planes==2:
                cleaneduplist=[]

        
    if cleaneduplist:
        for channel in range(1,possible_channels+1):
            chlist=[file_name for file_name in cleaneduplist if  '_Ch{}_'.format(str(channel)) in file_name]
            if chlist:
                moviestructure['Ch'+str(channel)]={}
            
                cycles_truth=[]
                # print('Creting movistructure')
                moviestructure['Ch'+str(channel)]['cycles']={}
                moviestructure['Ch'+str(channel)]['aqs']={}

                if correction:
                    if sq_type[0]!='TSeries ZSeries Element'  :
                        for  frame in chlist:
                            for i in range(1,possible_frames+1):
                                if 'Cycle{}'.format(str(i).zfill(5)) in frame:
                                    cycles_truth.append('Cycle{}'.format(str(i).zfill(5)))
                                    break
                    elif sq_type[0]=='TSeries ZSeries Element':        
                        for j, frame in enumerate(chlist):
                            for i in range(j+1,possible_frames+1):
                                if 'Cycle{}'.format(str(i).zfill(5)) in frame:
                                    cycles_truth.append('Cycle{}'.format(str(i).zfill(5)))
                                    break   
                                
                    cycles_truth=list(set(cycles_truth))
                    cycles_truth.sort()       
                    moviestructure['Ch'+str(channel)]['cycles']['cyclesnumber']= len(cycles_truth)
                    moviestructure['Ch'+str(channel)]['cycles']['firstcycle']=cycles_truth[0]
                    moviestructure['Ch'+str(channel)]['cycles']['lastcycle']=cycles_truth[-1]
                    
                # elif sq_type[1]*sq_type[2]==len(chlist):
                #    moviestructure['Ch'+str(channel)]['cycles']['cyclesnumber']= sq_type[1]
                #    moviestructure['Ch'+str(channel)]['cycles']['firstcycle']='Cycle00001'
                #    moviestructure['Ch'+str(channel)]['cycles']['lastcycle']='Cycle{}'.format(str(sq_type[1]).zfill(5))
    
            
            
                
                if correction:
                    aq_truth=[]
                    if sq_type[0]!='TSeries ZSeries Element' and correction:
                        for j, frame in enumerate(chlist):
                            for i in range(j+1,possible_frames+1):
                                if 'Ch{}_{}'.format(str(channel),str(i).zfill(6)) in frame:
                                    aq_truth.append('Ch{}_{}'.format(str(channel),str(i).zfill(6)))  
                                    break
                    elif sq_type[0]=='TSeries ZSeries Element' and correction:
                        for frame in chlist:
                            for i in range(1,possible_planes+1):
                                if 'Ch{}_{}'.format(str(channel),str(i).zfill(6)) in frame:
                                    aq_truth.append('Ch{}_{}'.format(str(channel),str(i).zfill(6)))  
                                    break
                                      
                    aq_truth=list(set(aq_truth))
                    aq_truth.sort()  
                    
                    
                    moviestructure['Ch'+str(channel)]['aqs']['aqsnumber']=len(aq_truth)
                    moviestructure['Ch'+str(channel)]['aqs']['firstaq']=aq_truth[0]
                    moviestructure['Ch'+str(channel)]['aqs']['lastaq']=aq_truth[-1]
                    # print('finsihed movistructure')
        
            
                # elif sq_type[1]*sq_type[2]==len(chlist):
                #     moviestructure['Ch'+str(channel)]['aqs']['aqsnumber']=sq_type[1]
                #     moviestructure['Ch'+str(channel)]['aqs']['firstaq']='Ch{}_{}'.format(str(channel),str(1).zfill(6))
                #     moviestructure['Ch'+str(channel)]['aqs']['lastaq']='Ch{}_{}'.format(str(channel),str(sq_type[2]).zfill(6))
        

        
        if correction:
            if sq_type[0]=='TSeries ZSeries Element' or sq_type[0]=='ZSeries':    
                
                if 'Ch1' in moviestructure:
                    
                    RedPlaneNumber=moviestructure['Ch1']['aqs']['aqsnumber']
                    FirstRedPlane=moviestructure['Ch1']['aqs']['firstaq']
                    LastRedPlane= moviestructure['Ch1']['aqs']['lastaq']
                    RedFrameNumber=moviestructure['Ch1']['cycles']['cyclesnumber']
                    RedFirstFrame=moviestructure['Ch1']['cycles']['firstcycle']
                    RedLastFrame=moviestructure['Ch1']['cycles']['lastcycle']
                    
                if 'Ch2' in moviestructure:    
                    GreenPlaneNumber=moviestructure['Ch2']['aqs']['aqsnumber']
                    FirstGreenPlane=moviestructure['Ch2']['aqs']['firstaq']
                    LastGreenPlane=  moviestructure['Ch2']['aqs']['lastaq']
                    GreenFrameNumber=moviestructure['Ch2']['cycles']['cyclesnumber']
                    GreenFirstFrame=moviestructure['Ch2']['cycles']['firstcycle']
                    GreenLastFrame=moviestructure['Ch2']['cycles']['lastcycle']
                    
            elif sq_type[0]=='TSeries Timed Element' or  sq_type[0]=='Single':
                
                if 'Ch1' in moviestructure:      
                    RedPlaneNumber=moviestructure['Ch1']['cycles']['cyclesnumber']
                    FirstRedPlane=moviestructure['Ch1']['cycles']['firstcycle']
                    LastRedPlane= moviestructure['Ch1']['cycles']['lastcycle']
                    RedFrameNumber=moviestructure['Ch1']['aqs']['aqsnumber']
                    RedFirstFrame=moviestructure['Ch1']['aqs']['firstaq']
                    RedLastFrame=moviestructure['Ch1']['aqs']['lastaq']
                    
                if 'Ch2' in moviestructure:    
                    GreenPlaneNumber=moviestructure['Ch2']['cycles']['cyclesnumber']
                    FirstGreenPlane=moviestructure['Ch2']['cycles']['firstcycle']
                    LastGreenPlane=  moviestructure['Ch2']['cycles']['lastcycle']
                    GreenFrameNumber=moviestructure['Ch2']['aqs']['aqsnumber']
                    GreenFirstFrame=moviestructure['Ch2']['aqs']['firstaq']
                    GreenLastFrame=moviestructure['Ch2']['aqs']['lastaq']
        else:
                if 'Ch2' in moviestructure:
                    GreenPlaneNumber=sq_type[2]
                    FirstGreenPlane='Ch1_{}'.format(str(1).zfill(6))
                    LastGreenPlane= 'Ch1_{}'.format(str(sq_type[2]).zfill(6))
                    GreenFrameNumber=sq_type[1]
                    GreenFirstFrame='Cycle00001'
                    GreenLastFrame='Cycle{}'.format(str(sq_type[1]).zfill(5))
                    
                if 'Ch1' in moviestructure:    
                    RedPlaneNumber=sq_type[2]
                    FirstRedPlane='Ch2_{}'.format(str(1).zfill(6))
                    LastRedPlane= 'Ch2_{}'.format(str(sq_type[2]).zfill(6))
                    RedFrameNumber=sq_type[1]
                    RedFirstFrame='Cycle00001'
                    RedLastFrame='Cycle{}'.format(str(sq_type[1]).zfill(5))


                    
        return [ChannelRedExists, ChannelGreenExists,
                RedFrameNumber, RedFirstFrame, RedLastFrame ,
                GreenFrameNumber, GreenFirstFrame, GreenLastFrame  ,
                Multiplane, RedPlaneNumber, GreenPlaneNumber, 
                FirstRedPlane, LastRedPlane, FirstGreenPlane, LastGreenPlane, sq_type[0]]
    
    
    else:        
        return [False, False, 
                0, False, False, 
                0, False, False, 
                0, 0, 0, 
                False,False,False,False,sq_type[0]]        


'''
def restore_initial_org():
    import os
    import glob
    from functionsDataOrganization import check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories

    sphaqpath=r'F:\Projects\LabNY\Imaging\2021\20210520\Mice\SPHQ\TestAcquisitions'
    aq='Aq_2'
    aq_name='210520_SPHQ_MaxResMechZSTack_940-000'
    ch1='Ch1Red'
    ch2='Ch2Green'

    images1=glob.glob(os.path.join(sphaqpath,aq,aq_name,ch1)+os.sep+'**\\**.tif')
    images2=glob.glob(os.path.join(sphaqpath,aq,aq_name,ch2)+os.sep+'**\\**.tif')

    for i in images1:
        os.rename(i, os.path.join( sphaqpath,aq,aq_name,os.path.split(i)[1]))

    for i in images2:
        os.rename(i, os.path.join( sphaqpath,aq,aq_name,os.path.split(i)[1]))
            
        

    recursively_eliminate_empty_folders(os.path.join(sphaqpath,aq,aq_name,ch1))
    recursively_eliminate_empty_folders(os.path.join(sphaqpath,aq,aq_name,ch2))
    '''