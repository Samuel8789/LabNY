# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 19:14:58 2021

@author: sp3660
"""
import os
import glob
import datetime
import logging 
logger = logging.getLogger(__name__)
from ...AllFunctions.create_dir_structure import create_dir_structure

from .fov import FOV
from .atlas import Atlas

from .acquisitonVariants import Coordinate0Aquisition, TestAquisition, NonimagingAquisition
from .wideFieldImage import WideFieldImage

class MouseImagingSession():
    
    def __init__(self,  imaging_session_name, imaging_session_ID=None, raw_imaging_session_path=None, mouse_object=None):   
        
        self.imaging_session_name=imaging_session_name
        self.sess_date=datetime.datetime.strptime(self.imaging_session_name, '%Y%m%d')
        self.imaging_session_ID=imaging_session_ID
        self.mouse_object=mouse_object
        self.mouse_session_path=os.path.join(self.mouse_object.mouse_slow_subproject_path,
                                              'imaging',
                                              self.imaging_session_name)
        
        
        
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
       
            create_dir_structure(self.mouse_session_path, imaging_session_struc)
        #tem[porary to create atlas folders]
        if not os.path.isdir(os.path.join(self.mouse_session_path,'atlases')):
            os.mkdir(os.path.join(self.mouse_session_path,'atlases'))    
        if raw_imaging_session_path:
            print('adding prairie session'+self.imaging_session_name)
            # if self.imaging_session_name=='20210522':
            #     print('wait')
            self.imaging_session_path=raw_imaging_session_path           
            self.mouse_imaging_session_path=os.path.join(self.imaging_session_path,'Mice',
                                              self.mouse_object.mouse_name)
            # print('adding raw fovs')
            self.load_raw_FOVs()
            # print('adding raw testaq')
            self.load_raw_Test_Aquisitions()
            # print('adding raw widefield')
            self.load_raw_widefield_image()
            # print('adding raw nonimaging')
            self.load_raw_nonimaging_Aquisitions()
            # print('adding raw 0coordinates')
            self.load_raw_0coordinate_Aquisitions()
            # self.load_raw_atlas()

        else:
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
            
            
            self.load_existing_FOVs()
            self.load_existing_Test_Aquisitions()
            self.load_existing_widefield_image()
            self.load_existing_nonimaging_Aquisitions()
            self.load_existing_0coordinate_Aquisitions()
            # self.load_existing_atlas()
    
    def load_raw_nonimaging_Aquisitions(self):   
        self.all_raw_nonimaging_Aquisitions={}
        if os.path.isdir(os.path.join( self.mouse_imaging_session_path ,'NonImagingAcquisitions')):           
            for i, aqu in enumerate(os.listdir(os.path.join(self.mouse_imaging_session_path, 'NonImagingAcquisitions'))):
                 if (os.path.isdir(os.path.join( self.mouse_imaging_session_path ,'NonImagingAcquisitions', aqu)) and glob.glob(os.path.join( self.mouse_imaging_session_path ,'NonImagingAcquisitions', aqu)+'\\**\\**.env', recursive=False)):
                    self.all_raw_nonimaging_Aquisitions[str(aqu)]=NonimagingAquisition(os.path.split(glob.glob(os.path.join( self.mouse_imaging_session_path ,'NonImagingAcquisitions', aqu)+'\\**\\**.env', recursive=False)[0])[0],
                                                                              self,
                                                                              os.path.join( self.mouse_imaging_session_path ,'NonImagingAcquisitions', aqu)) 
                 else:
                     self.all_raw_nonimaging_Aquisitions[str(aqu)]=NonimagingAquisition(os.path.join(self.mouse_imaging_session_path ,'NonImagingAcquisitions', aqu)+'_NonImaging',
                                                                      self,
                                                                      os.path.join( self.mouse_imaging_session_path ,'NonImagingAcquisitions', aqu),
                                                                      non_imaging=True)                               
                                     
        self.load_existing_nonimaging_Aquisitions()
        
    def load_existing_nonimaging_Aquisitions(self):  
        self.all_nonimaging_Aquisitions={str(aqu):NonimagingAquisition(os.path.join( self.mouse_session_path ,'nonimaging acquisitions', aqu),self) 
                         
                        for i, aqu in enumerate(os.listdir(os.path.join(self.mouse_session_path, 'nonimaging acquisitions')))
                        if os.path.isdir(os.path.join( self.mouse_session_path ,'nonimaging acquisitions', aqu))} 
        
    def load_raw_FOVs(self):
        self.all_raw_FOVs={str(directory):FOV(str(directory), 
                                          os.path.join(self.mouse_imaging_session_path, directory),
                                          self) 
                       for i, directory in enumerate(os.listdir(self.mouse_imaging_session_path))
                       if 'FOV' in directory}
        self.load_existing_FOVs()
        
    def load_existing_FOVs(self):
    
          self.all_FOVs={str(directory):FOV(str(directory), mouse_imaging_session_object=self) 
                         
                       for i, directory in enumerate(os.listdir(os.path.join(self.mouse_session_path, 'data aquisitions')))
                       if 'FOV' in directory} 
        
    def load_raw_Test_Aquisitions(self):   
        if os.path.isdir(os.path.join( self.mouse_imaging_session_path ,'TestAcquisitions')):
        
            self.all_raw_Test_Aquisitions={str(aqu):TestAquisition(glob.glob(os.path.join( self.mouse_imaging_session_path ,'TestAcquisitions', aqu)+'\\**',recursive=False)[0],
                                                                      self,
                                                                      os.path.join( self.mouse_imaging_session_path ,'TestAcquisitions', aqu)) 
                                       
                                       for i, aqu in enumerate(os.listdir(os.path.join(self.mouse_imaging_session_path, 'TestAcquisitions')))
                                       if os.path.isdir(os.path.join( self.mouse_imaging_session_path ,'TestAcquisitions', aqu))}
        self.load_existing_Test_Aquisitions()
        
    def load_existing_Test_Aquisitions(self):   
        self.all_Test_Aquisitions={str(aqu):TestAquisition(os.path.join( self.mouse_session_path ,'test aquisitions', aqu),self) 
                         
                       for i, aqu in enumerate(os.listdir(os.path.join(self.mouse_session_path, 'test aquisitions')))
                       if os.path.isdir(os.path.join( self.mouse_session_path ,'test aquisitions', aqu))} 
        
    def load_raw_widefield_image(self):
       if os.path.isdir(os.path.join(self.mouse_imaging_session_path, 'Widefield')): 
            self.widefield_image={self.mouse_object.mouse_name+ self.imaging_session_name+'_Widefield': WideFieldImage(self,
                                                                                         widefield, 
                                                                                         os.path.join(self.mouse_imaging_session_path,'Widefield', widefield)
                                                                                         )
                                  for i, widefield in enumerate(os.listdir(os.path.join(self.mouse_imaging_session_path, 'Widefield')))
                                  if '.tif' in widefield
                                  }
            self.load_existing_widefield_image()
        
    def load_existing_widefield_image(self):    
        if os.path.isdir(os.path.join(self.mouse_session_path,'widefield image')): 
            self.widefield_image={self.mouse_object.mouse_name+ self.imaging_session_name+'_Widefield': WideFieldImage(self, widefield)
                                  
                          for i, widefield in enumerate(os.listdir(os.path.join(self.mouse_session_path,'widefield image')))
                          if '.tif' in widefield}

    def load_raw_0coordinate_Aquisitions(self):   
        if os.path.isdir(os.path.join( self.mouse_imaging_session_path ,'0CoordinateAcquisiton')):
        
            self.all_raw_0coordinate_Aquisitions={str(aqu):Coordinate0Aquisition(glob.glob(os.path.join( self.mouse_imaging_session_path ,'0CoordinateAcquisiton', aqu)+'\\**',recursive=False)[0],
                                                                      self,
                                                                      os.path.join( self.mouse_imaging_session_path ,'0CoordinateAcquisiton', aqu)) 
                                       
                                        for i, aqu in enumerate(os.listdir(os.path.join(self.mouse_imaging_session_path, '0CoordinateAcquisiton')))
                                        if os.path.isdir(os.path.join( self.mouse_imaging_session_path ,'0CoordinateAcquisiton', aqu))}
            
        self.load_existing_0coordinate_Aquisitions()
        
    def load_existing_0coordinate_Aquisitions(self): 
        
        self.all_0coordinate_Aquisitions={str(aqu):Coordinate0Aquisition(os.path.join( self.mouse_session_path ,'0Coordinate acquisition', aqu), self) 
                         
                        for i, aqu in enumerate(os.listdir(os.path.join(self.mouse_session_path, '0Coordinate acquisition')))
                        if os.path.isdir(os.path.join( self.mouse_session_path ,'0Coordinate acquisition', aqu))}     


    def load_raw_atlas(self):
        self.all_raw_atlas={str(directory):Atlas(str(directory), 
                                          os.path.join(self.mouse_imaging_session_path, directory),
                                          self) 
                       for  directory in os.listdir(self.mouse_imaging_session_path)
                       if 'Atlas' in directory}
        self.load_existing_atlas()
        
    def load_existing_atlas(self):
    
          self.all_atlas={str(directory):Atlas(str(directory), mouse_imaging_session_object=self) 
                     
                       for  directory in os.listdir(os.path.join(self.mouse_session_path, 'atlases'))
                       if 'Atlas' in directory} 
