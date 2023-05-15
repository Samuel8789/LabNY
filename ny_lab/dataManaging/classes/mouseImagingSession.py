# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 19:14:58 2021

@author: sp3660
"""
import os
import time
import glob
import datetime
import logging 
module_logger = logging.getLogger(__name__)
from ...AllFunctions.create_dir_structure import create_dir_structure

from .fov import FOV
from .atlas import Atlas

from .acquisitonVariants import Coordinate0Aquisition, TestAquisition, NonimagingAquisition
from .wideFieldImage import WideFieldImage

# from .mouse import Mouse

from ..functions.functionsDataOrganization import check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories
# from functionsDataOrganization import check_channels_and_planes, recursively_eliminate_empty_folders, move_files, recursively_copy_changed_files_and_directories_from_slow_to_fast, recursively_delete_back_directories

from ..functions.select_face_camera import select_face_camera
# from select_face_camera import select_face_camera

import shutil



import tkinter as tk
from tkinter import ttk



class MouseImagingSession():
    
    def __init__(self,  imaging_session_name=None, imaging_session_ID=None, raw_imaging_session_path=None, mouse_object=None, yet_to_add=None):   
        module_logger.info('Loading Mouse Imaging Session ' + imaging_session_name)

        self.imaging_session_name=imaging_session_name
        self.sess_date=datetime.datetime.strptime(self.imaging_session_name, '%Y%m%d')
        self.imaging_session_ID=imaging_session_ID
        self.mouse_object=mouse_object
        self.mouse_session_path=os.path.join(self.mouse_object.mouse_slow_subproject_path,
                                              'imaging',
                                              self.imaging_session_name)
        self.yet_to_add=yet_to_add
        
        self.all_raw_nonimaging_Aquisitions={}
        self.all_raw_0coordinate_Aquisitions={}
        self.all_raw_FOVs={}

        
        self.all_raw_Test_Aquisitions={}
        self.all_Test_Aquisitions={}
        if not os.path.isdir(self.mouse_session_path):
            os.mkdir(self.mouse_session_path)
        imaging_session_struc={'calibrations',
                                'test aquisitions',
                                'data aquisitions',
                                'microscope state',  
                                'widefield image',  
                                'widefield activity mapping',     
                                'widefield landmarking',
                                '0Coordinate acquisition',
                                'nonimaging acquisitions',
                                'atlases'
                                }
       
            # create_dir_structure(self.mouse_session_path, imaging_session_struc)
        #tem[porary to create atlas folders]
        create_dir_structure(self.mouse_session_path, imaging_session_struc)

        # if not os.path.isdir(os.path.join(self.mouse_session_path,'atlases')):
        #     os.mkdir(os.path.join(self.mouse_session_path,'atlases'))    
        if raw_imaging_session_path:
            self.mouse_code=self.mouse_object.mouse_name
            module_logger.info('adding prairie session '+ self.mouse_code+' '+self.imaging_session_name)
            # if self.imaging_session_name=='20210522':
            #     module_logger.info('wait')
            self.imaging_session_path=raw_imaging_session_path           
            self.mouse_raw_imaging_session_path=os.path.join(self.imaging_session_path,'Mice',
                                              self.mouse_object.mouse_name)
            
            module_logger.info('pre processing session '+ self.mouse_code+' '+self.imaging_session_name)



            # this organizez folders in permanent directiory
            self.raw_session_folder_org()
            print('Finished Folder Organization'+  raw_imaging_session_path +  self.mouse_code)
           
            #then create acquisitoin objects and copy to slow , this is slow because of metadta read
            # self.raw_session_preprocessing()



        elif self.imaging_session_ID:
            quey_imaging_session="""SELECT b.ImagingDate AS SessionDate, a.*, b.* ,c.*, d.*
                            FROM ImagedMice_table a 
                            LEFT JOIN ImagingSessions_table b ON b.ID=a.SessionID 
                            LEFT JOIN ExperimentalAnimals_table c ON c.ID=a.ExpID 
                            LEFT JOIN MICE_table d ON d.ID=c.Mouse_ID 
                            WHERE c.Code=? AND a.ID=?
                            """
                            
              
            params=(self.mouse_object.mouse_name, self.imaging_session_ID)
            self.ImagedMiceinfo=mouse_object.Database_ref.arbitrary_query_to_df(quey_imaging_session, params)       
            self.session_slowstoragepath= self.ImagedMiceinfo.SlowStoragePath
            self.session_workingstoragepath=self.ImagedMiceinfo.WorkingStoragePath
            self.get_acq_IDs_from_database()
            
            self.load_existing_FOVs()
            self.load_existing_Test_Aquisitions()
            self.load_existing_widefield_image()
            self.load_existing_nonimaging_Aquisitions()
            self.load_existing_0coordinate_Aquisitions()
            # self.load_existing_atlas()
            
        elif self.yet_to_add:
            
            self.mouse_code=self.mouse_object.mouse_name
            module_logger.info('adding prairie session '+ self.mouse_code+' '+self.imaging_session_name)
            # if self.imaging_session_name=='20210522':
            #     module_logger.info('wait')

            self.load_existing_FOVs()
            self.load_existing_Test_Aquisitions()
            self.load_existing_widefield_image()
            self.load_existing_nonimaging_Aquisitions()
            self.load_existing_0coordinate_Aquisitions()
        
            
            
         
   
    
#%% methods   


   
    def raw_session_preprocessing(self):  
        
        if os.path.isdir( self.mouse_raw_imaging_session_path):
            try:
                module_logger.info('adding raw fovs')
                self.load_raw_FOVs()
                module_logger.info('adding raw testaq')
                self.load_raw_Test_Aquisitions()
                module_logger.info('adding raw widefield')
                self.load_raw_widefield_image()
                module_logger.info('adding raw nonimaging')
                self.load_raw_nonimaging_Aquisitions()
                module_logger.info('adding raw 0coordinates')
                self.load_raw_0coordinate_Aquisitions()
                # self.load_raw_atlas()
            except:
                module_logger.exception('Something wrong with loading acquisitions ' +   self.mouse_raw_imaging_session_path)
            print('Finished Session Preprocessing')
        else:
            module_logger.info('no imaging done')
        
    def raw_session_folder_org(self):
        try:
            self.organize_acquisition_folders()  
        except:
            module_logger.exception('Something wrong with raw session preprocessing ' +   self.mouse_raw_imaging_session_path)
                   
    def organize_acquisition_folders(self):
        module_logger.info('organizing folders slow')

        self.raw_Coordinate0path=os.path.join(self.mouse_raw_imaging_session_path,'0CoordinateAcquisiton')   
        self.raw_Noniima=os.path.join(self.mouse_raw_imaging_session_path,'NonImagingAcquisitions')  
        self.Testacquisitionspath=os.path.join(self.mouse_raw_imaging_session_path,'TestAcquisitions') 
        self.correctComplexacquisitonsNames('FOV_')
        self.correctComplexacquisitonsNames('Atlas_')
        self.all_fovs=glob.glob(self.mouse_raw_imaging_session_path +'\\FOV_**', recursive=False) 
        self.all_atlases=glob.glob(self.mouse_raw_imaging_session_path +'\\Atlas_**', recursive=False) 
        
        self.all_simple_Aq_folders=[self.raw_Coordinate0path, self.raw_Noniima, self.Testacquisitionspath]
        
        self.all_FOVs_all_Aq_folders={}       
        for fov_path in self.all_fovs:
    
            Plane3Tomato1050=os.path.join(fov_path, '1050_3PlaneTomato')
            HighResStackTomato1050=os.path.join(fov_path, '1050_HighResStackTomato' )  
            Tomato1050=os.path.join(fov_path, '1050_Tomato')
            HighResStackGreen=os.path.join(fov_path, 'HighResStackGreen')
            SurfaceImage =os.path.join(fov_path, 'SurfaceImage' )
            OtherAcq=os.path.join(fov_path, 'OtherAcq')
            single_FOVs_Aq_folders=[Plane3Tomato1050, HighResStackTomato1050, Tomato1050, HighResStackGreen, SurfaceImage, OtherAcq, fov_path]  
            self.all_FOVs_all_Aq_folders[fov_path]=single_FOVs_Aq_folders
            
        self.all_atlases_all_Aq_folders={}
        for atlas in self.all_atlases:
              Overview=os.path.join(atlas, 'Overview')
              Preview=os.path.join(atlas, 'Preview' )  
              Volumes=os.path.join(atlas, 'Volumes')
              Coordinates=os.path.join(atlas, 'Coordinates')
              if os.path.isdir(Coordinates):
                if  len(os.listdir(Coordinates))==0:
                    recursively_eliminate_empty_folders(Coordinates)  
              single_atlas_Aq_folders=[Overview, Preview, Volumes] 
              self.all_atlases_all_Aq_folders[atlas]=single_atlas_Aq_folders
              
          
        self.all_aquisitions=self.all_simple_Aq_folders+  [y for x in list(self.all_FOVs_all_Aq_folders.values()) for y in x] + [y for x in list(self.all_atlases_all_Aq_folders.values()) for y in x]
               
        for aq_folder in self.all_aquisitions:
            module_logger.info('creating directory structures' + aq_folder)

            self.createAqFolders(aq_folder)
            allaq=glob.glob(aq_folder +'\\Aq_**', recursive=False)  

            for aq in allaq:
                module_logger.info('processing acquisitions'+ aq_folder)
                self.process_aquisition_folder(aq)
                module_logger.info('removing empty final'+ aq_folder)

            recursively_eliminate_empty_folders(aq_folder) 
            
            
        recursively_eliminate_empty_folders(self.mouse_raw_imaging_session_path) 
        
        
        

    def correctComplexacquisitonsNames(self, ComplexAcqname):
        
        # no detect fovs with fov number highr thant totAL NIMBER OF FOVS AND CHANGE IT        
        all_compacqs=glob.glob(self.mouse_raw_imaging_session_path +'\\'+ComplexAcqname+'**', recursive=False) 
        unnumbered_compacqs=glob.glob(self.mouse_raw_imaging_session_path +'\\'+ComplexAcqname, recursive=False) 

        if unnumbered_compacqs:
            named_compacqs_paths=[i for i in all_compacqs if i not in unnumbered_compacqs]
        else:
            named_compacqs_paths=all_compacqs
        named_compacqs_names=[os.path.split(i)[1] for i in named_compacqs_paths]
        # number_current_named_compacqs=len(named_compacqs_paths)
        number_total_compacqs=len(all_compacqs)
        
        wanted_compacq_numbers=[i+1 for i in range(number_total_compacqs)]
        # this is to detect fov numbers highr than the total fovs
           
            
        if named_compacqs_paths:
            named_compacq_number=[int(i[i.find('_')+1:]) for i in named_compacqs_names]
            to_change=[i for i in named_compacq_number if i not in wanted_compacq_numbers ]
            to_change_to=[i for i in wanted_compacq_numbers if i not in named_compacq_number ]
            for compacq in named_compacqs_paths:
                if to_change:
                   if compacq.find(ComplexAcqname + str(to_change[0]))!=-1: 
                       os.rename(compacq, compacq[:compacq.find(ComplexAcqname)+len(ComplexAcqname)]+str(to_change_to[0]))   
                       to_change.remove(to_change[0])
                       to_change_to.remove(to_change_to[0])
           
        for compacq in unnumbered_compacqs:
            if not glob.glob( compacq+'\\**\\**.env', recursive=True) and not glob.glob( compacq+'\\**\\**.xy', recursive=True) :

                recursively_eliminate_empty_folders(compacq) 
            else:
                if not named_compacqs_paths:
                    to_change_to=[1]   
                    os.rename(compacq, compacq+str(to_change_to[0]))    

                else:
                #this will change empty fov to the minimum fov number avilabel  
                    os.rename(compacq, compacq+str(to_change_to[0]))    
                                           
    
    def createAqFolders(self, generic_aq):
        generic_aq_folder_prairieaq=glob.glob(generic_aq +'\\**\\**.env', recursive=False) 
                       
        allaq=glob.glob(generic_aq +'\\Aq_**', recursive=False) 
        emptyaq=glob.glob(generic_aq +'\\Aq_', recursive=False) 

        Aq_number=[os.path.split(i)[1] for i in allaq if i not in emptyaq]
        # currentaq=[i for i in allaq if i not in emptyaq]
        curent_good_aq=len(Aq_number)

        if generic_aq_folder_prairieaq:
            for i, aq_path in enumerate(generic_aq_folder_prairieaq):  
                destination=os.path.join(generic_aq,'Aq_'+str(1+i+curent_good_aq))
                os.mkdir(destination) 
                shutil.move(os.path.split(aq_path)[0],destination)
                
                
    def read_unprocessed_extra_data(self):
         
         self.UnprocessedFaceCameras=os.path.join(self.mouse_raw_imaging_session_path,'UnprocessedFaceCameras')   
         self.UnprocessedVisStim=os.path.join(self.mouse_raw_imaging_session_path,'UnprocessedVisStim')   
         self.UnprocessedDaqRecording=os.path.join(self.mouse_raw_imaging_session_path,'UnprocessedDaq')   

         self.UnprocessedFaceCameraspaths=glob.glob(self.UnprocessedFaceCameras +'\\**\\**Default.ome.tif', recursive=False)
         self.UnprocessedFaceCameraspaths2=glob.glob(self.UnprocessedFaceCameras +'\\**\\Default\\**000000000_z000.tif', recursive=False)

         self.UnprocessedVisStimpaths=glob.glob(self.UnprocessedVisStim +'\\**.mat', recursive=False)
         self.UnprocessedDaqRecordingpaths=glob.glob(self.UnprocessedDaqRecording +'\\**.mat', recursive=False)
         self.UnprocessedFaceCamerasnames=[]
         self.UnprocessedVisStimnames=[]
         self.UnprocessedDaqRecordingnames=[]
                     
         if not self.UnprocessedFaceCameraspaths and not self.UnprocessedFaceCameraspaths2:
             if os.path.isdir(self.UnprocessedFaceCameras):
                 recursively_eliminate_empty_folders(self.UnprocessedFaceCameras)
         elif self.UnprocessedFaceCameraspaths:
             self.UnprocessedFaceCamerasnames=[os.path.split(self.UnprocessedFaceCameraspath)[1] for self.UnprocessedFaceCameraspath in self.UnprocessedFaceCameraspaths]
             self.UnprocessedFaceCamerasnames=['None']+self.UnprocessedFaceCamerasnames
         elif self.UnprocessedFaceCameraspaths2:
             self.UnprocessedFaceCamerasnames=[os.path.split(os.path.split(os.path.split(self.UnprocessedFaceCameraspath)[0])[0])[1] for self.UnprocessedFaceCameraspath in self.UnprocessedFaceCameraspaths2]
             self.UnprocessedFaceCamerasnames=['None']+self.UnprocessedFaceCamerasnames
             
             
             
         if not self.UnprocessedVisStimpaths:
             if os.path.isdir(self.UnprocessedVisStim):
                 recursively_eliminate_empty_folders(self.UnprocessedVisStim) 
         else:
             self.UnprocessedVisStimnames=[os.path.split(self.UnprocessedVisStimpath)[1] for self.UnprocessedVisStimpath in self.UnprocessedVisStimpaths]
             self.UnprocessedVisStimnames=['None']+self.UnprocessedVisStimnames
             if glob.glob(self.UnprocessedVisStim +'\\**.mat.mat', recursive=False) :
                 for mat in glob.glob(self.UnprocessedVisStim +'\\**.mat.mat', recursive=False):      
                     os.rename(mat,mat[:-4])

         if not self.UnprocessedDaqRecordingpaths:
             if os.path.isdir(self.UnprocessedDaqRecording):
                 recursively_eliminate_empty_folders(self.UnprocessedDaqRecording) 
         else:
             self.UnprocessedDaqRecordingnames=[os.path.split(self.UnprocessedDaqRecordingpath)[1] for self.UnprocessedDaqRecordingpath in self.UnprocessedDaqRecordingpaths]
             self.UnprocessedDaqRecordingnames=['None']+self.UnprocessedDaqRecordingnames
             if glob.glob(self.UnprocessedDaqRecording +'\\**.mat.mat', recursive=False) :
                 for mat in glob.glob(self.UnprocessedDaqRecording +'\\**.mat.mat', recursive=False):      
                     os.rename(mat,mat[:-4])            
                
    def process_aquisition_folder(self, aq):
        
        self.read_unprocessed_extra_data()
        if self.UnprocessedFaceCameraspaths or self.UnprocessedVisStimpaths or self.UnprocessedDaqRecordingpaths or self.UnprocessedFaceCameraspaths2: 
            self.addFaceCamsVisStim(aq)
        self.read_unprocessed_extra_data()

        recursively_eliminate_empty_folders(aq) 
                         
        
    def addFaceCamsVisStim(self, aq): 
        module_logger.info('moving camera and mat files'+ aq)

        
        if not  self.UnprocessedFaceCamerasnames:
            self.UnprocessedFaceCamerasnames=['None']
        if not  self.UnprocessedVisStimnames:
            self.UnprocessedVisStimnames=['None']
        if not  self.UnprocessedDaqRecordingnames:
            self.UnprocessedDaqRecordingnames=['None']
    
                
        if not self.mouse_object.data_managing_object.LabProjectObject.gui:
           self.guiref=tk.Tk()
        else:
           self.guiref=self.mouse_object.data_managing_object.LabProjectObject.gui
                
        print('Openeing extra data selection')   
        self.select_face_camera_window=select_face_camera(self.guiref, os.path.split(glob.glob(aq +'\\**', recursive=False)[0])[1], self.UnprocessedFaceCamerasnames, self.UnprocessedVisStimnames, self.UnprocessedDaqRecordingnames)
        self.select_face_camera_window.wait_window()
        get_values= self.select_face_camera_window.values
        self.select_face_camera_window.destroy()

        if get_values[1][1] and get_values[1][1]!='None':
            facecameradir=os.path.join(aq, 'FaceCamera')  
            if not os.path.isdir(facecameradir):
                os.mkdir(facecameradir)
            # unprocessedfacecameraname= os.path.split(os.path.split([name for name in UnprocessedFaceCameraspaths if get_values[1][1] in name][0])[0])[1]
            if self.UnprocessedFaceCameraspaths:
                unprocessedfacecamerafullpath=os.path.split([name for name in self.UnprocessedFaceCameraspaths if get_values[1][1] in name][0])[0]
                files = glob.glob(unprocessedfacecamerafullpath+'\\**' )
                for f in files:
                      shutil.move(f, facecameradir)               
            
            elif self.UnprocessedFaceCameraspaths2:
                unprocessedfacecamerafullpath2=os.path.split(os.path.split([name for name in self.UnprocessedFaceCameraspaths2 if get_values[1][1] in name][0])[0])[0]
                files2 = glob.glob(unprocessedfacecamerafullpath2+'\\**' )
                for f in files2:
                    if os.path.isfile(f):
                        shutil.move(f, facecameradir)       
                    elif os.path.isdir(f):
                        shutil.move(f, os.path.join(facecameradir, 'Default'))       



            
        if get_values[2][1] and get_values[2][1]!='None':
            visstimdir=os.path.join(aq, 'VisStim')   
            unprocessedvisstim= [name for name in self.UnprocessedVisStimpaths if get_values[2][1] in name][0]
            if not os.path.isdir(visstimdir):
                os.mkdir(visstimdir)
            file = unprocessedvisstim
            shutil.move(file, visstimdir)   
                    
        if get_values[3][1] and get_values[3][1]!='None':
            daqdir=os.path.join(aq, 'ExtraDaq')   
            unprocesseddaq= [name for name in self.UnprocessedDaqRecordingpaths if get_values[3][1] in name][0]
            if not os.path.isdir(daqdir):
                os.mkdir(daqdir)
            
            file = unprocesseddaq
            shutil.move(file, daqdir)   
  

        recursively_eliminate_empty_folders(aq) 

#%% raw loading functions 

    def load_raw_nonimaging_Aquisitions(self):   
        if os.path.isdir(os.path.join( self.mouse_raw_imaging_session_path ,'NonImagingAcquisitions')):           
            for i, aqu in enumerate(os.listdir(os.path.join(self.mouse_raw_imaging_session_path, 'NonImagingAcquisitions'))):
                 if (os.path.isdir(os.path.join( self.mouse_raw_imaging_session_path ,'NonImagingAcquisitions', aqu)) and glob.glob(os.path.join( self.mouse_raw_imaging_session_path ,'NonImagingAcquisitions', aqu)+'\\**\\**.env', recursive=False)):
                    self.all_raw_nonimaging_Aquisitions[str(aqu)]=NonimagingAquisition(os.path.split(glob.glob(os.path.join( self.mouse_raw_imaging_session_path ,'NonImagingAcquisitions', aqu)+'\\**\\**.env', recursive=False)[0])[0],
                                                                              self,
                                                                              os.path.join( self.mouse_raw_imaging_session_path ,'NonImagingAcquisitions', aqu)) 
                 else:
                     self.all_raw_nonimaging_Aquisitions[str(aqu)]=NonimagingAquisition(os.path.join(self.mouse_raw_imaging_session_path ,'NonImagingAcquisitions', aqu)+'_NonImaging',
                                                                      self,
                                                                      os.path.join( self.mouse_raw_imaging_session_path ,'NonImagingAcquisitions', aqu),
                                                                      non_imaging=True)                               
                                     
        self.load_existing_nonimaging_Aquisitions()
        
        
    def load_raw_FOVs(self):
        self.all_raw_FOVs={str(directory):FOV(str(directory), 
                                          os.path.join(self.mouse_raw_imaging_session_path, directory),
                                          self) 
                       for i, directory in enumerate(os.listdir(self.mouse_raw_imaging_session_path))
                       if 'FOV' in directory}
        self.load_existing_FOVs()
        
        
    def load_raw_Test_Aquisitions(self):   
         if os.path.isdir(os.path.join( self.mouse_raw_imaging_session_path ,'TestAcquisitions')):
         
             self.all_raw_Test_Aquisitions={str(aqu):TestAquisition(glob.glob(os.path.join( self.mouse_raw_imaging_session_path ,'TestAcquisitions', aqu)+'\\**',recursive=False)[0],
                                                                       self,
                                                                       os.path.join( self.mouse_raw_imaging_session_path ,'TestAcquisitions', aqu)) 
                                        
                                        for i, aqu in enumerate(os.listdir(os.path.join(self.mouse_raw_imaging_session_path, 'TestAcquisitions')))
                                        if os.path.isdir(os.path.join( self.mouse_raw_imaging_session_path ,'TestAcquisitions', aqu))}
         self.load_existing_Test_Aquisitions()    
        
    def load_raw_widefield_image(self):
       if os.path.isdir(os.path.join(self.mouse_raw_imaging_session_path, 'Widefield')): 
            self.widefield_image={self.mouse_object.mouse_name+ self.imaging_session_name+'_Widefield': WideFieldImage(widefield, 
                                                                                                                       mouse_imaging_session_object=self,
                                                                                         
                                                                                         raw_input_path=os.path.join(self.mouse_raw_imaging_session_path,'Widefield', widefield)
                                                                                         )
                                  for i, widefield in enumerate(os.listdir(os.path.join(self.mouse_raw_imaging_session_path, 'Widefield')))
                                  if '.tif' in widefield
                                  }
            self.load_existing_widefield_image()  
            
            
    def load_raw_0coordinate_Aquisitions(self): 
         
         if os.path.isdir(os.path.join( self.mouse_raw_imaging_session_path ,'0CoordinateAcquisiton')):
         
             self.all_raw_0coordinate_Aquisitions={str(aqu):Coordinate0Aquisition(glob.glob(os.path.join( self.mouse_raw_imaging_session_path ,'0CoordinateAcquisiton', aqu)+'\\**',recursive=False)[0],
                                                                       self,
                                                                       os.path.join( self.mouse_raw_imaging_session_path ,'0CoordinateAcquisiton', aqu)) 
                                        
                                         for i, aqu in enumerate(os.listdir(os.path.join(self.mouse_raw_imaging_session_path, '0CoordinateAcquisiton')))
                                         if os.path.isdir(os.path.join( self.mouse_raw_imaging_session_path ,'0CoordinateAcquisiton', aqu))}
             
         self.load_existing_0coordinate_Aquisitions() 
         
         
    def load_raw_atlas(self):
         self.all_raw_atlas={str(directory):Atlas(str(directory), 
                                           os.path.join(self.mouse_raw_imaging_session_path, directory),
                                           self) 
                        for  directory in os.listdir(self.mouse_raw_imaging_session_path)
                        if 'Atlas' in directory}
         self.load_existing_atlas()     
         
#%% database loading functions        
    def load_existing_nonimaging_Aquisitions(self):  
        self.all_nonimaging_Aquisitions={str(aqu):NonimagingAquisition(os.path.join( self.mouse_session_path ,'nonimaging acquisitions', aqu),self) 
                         
                        for i, aqu in enumerate(os.listdir(os.path.join(self.mouse_session_path, 'nonimaging acquisitions')))
                        if os.path.isdir(os.path.join( self.mouse_session_path ,'nonimaging acquisitions', aqu))} 

    def load_existing_FOVs(self):
          self.all_FOVs={}
    
          self.all_FOVs={str(directory):FOV(str(directory), mouse_imaging_session_object=self) 
                         
                       for i, directory in enumerate(os.listdir(os.path.join(self.mouse_session_path, 'data aquisitions')))
                       if 'FOV' in directory} 
        
   
        
    def load_existing_Test_Aquisitions(self): 
        self.all_Test_Aquisitions={}
        self.all_Test_Aquisitions={str(aqu):TestAquisition(os.path.join( self.mouse_session_path ,'test aquisitions', aqu),self) 
                         
                       for i, aqu in enumerate(os.listdir(os.path.join(self.mouse_session_path, 'test aquisitions')))
                       if os.path.isdir(os.path.join( self.mouse_session_path ,'test aquisitions', aqu))} 
        
    
        
    def load_existing_widefield_image(self):    
        if os.path.isdir(os.path.join(self.mouse_session_path,'widefield image')): 
            self.widefield_image={self.mouse_object.mouse_name+ self.imaging_session_name+'_Widefield': WideFieldImage(widefield, mouse_imaging_session_object=self)
                                  
                          for i, widefield in enumerate(os.listdir(os.path.join(self.mouse_session_path,'widefield image')))
                          if '.tif' in widefield}

   
        
    def load_existing_0coordinate_Aquisitions(self): 
        self.all_0coordinate_Aquisitions={}
        self.all_0coordinate_Aquisitions={str(aqu):Coordinate0Aquisition(os.path.join( self.mouse_session_path ,'0Coordinate acquisition', aqu), self) 
                         
                        for i, aqu in enumerate(os.listdir(os.path.join(self.mouse_session_path, '0Coordinate acquisition')))
                        if os.path.isdir(os.path.join( self.mouse_session_path ,'0Coordinate acquisition', aqu))}     


   
        
    def load_existing_atlas(self):
    
          self.all_atlas={str(directory):Atlas(str(directory), mouse_imaging_session_object=self) 
                     
                       for  directory in os.listdir(os.path.join(self.mouse_session_path, 'atlases'))
                       if 'Atlas' in directory} 
#%% get all aqcuisitions from database


    def get_acq_IDs_from_database(self):
        imaged_mice_id=self.mouse_object.imaging_sessions_database.iloc[:,1].tolist()
        quey_acquisitions_ID="""SELECT ID,ImagedMouseID, SlowDiskPath
                                    FROM Acquisitions_table  
                                    WHERE ImagedMouseID IN(%s)""" % ','.join('?' for i in imaged_mice_id) 
                                    
        params=tuple(imaged_mice_id)
        self.database_acquisitions=self.mouse_object.Database_ref.arbitrary_query_to_df(quey_acquisitions_ID, params)      
                
                
        