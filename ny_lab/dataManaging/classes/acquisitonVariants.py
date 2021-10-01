# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:19:35 2021

@author: sp3660
"""


import os
from .aquisition import Aquisition


class TestAquisition(Aquisition):
    def __init__(self, aqu_name, mouse_imaging_session_object, raw_input_path=None, ):
        Aquisition.__init__(self, aqu_name, raw_input_path, mouse_imaging_session_object=mouse_imaging_session_object,subaq_object='TestAquisition')
                
class Coordinate0Aquisition(Aquisition):
    def __init__(self, aqu_name, mouse_imaging_session_object, raw_input_path=None, ):
        Aquisition.__init__(self, aqu_name, raw_input_path, mouse_imaging_session_object=mouse_imaging_session_object,subaq_object='Coordinate0Aquisition')
                
class NonimagingAquisition(Aquisition):
    def __init__(self, aqu_name, mouse_imaging_session_object, raw_input_path=None, non_imaging=False):
        aqu_name=os.path.split(aqu_name)[1]
        Aquisition.__init__(self, aqu_name, raw_input_path, mouse_imaging_session_object=mouse_imaging_session_object, subaq_object='NonimagingAquisition',non_imaging=non_imaging)
 
class AtlasOverview(Aquisition):
    def __init__(self, aqu_name, atlas_object, raw_input_path=None):
        aqu_name=os.path.split(aqu_name)[1]
        Aquisition.__init__(self, aqu_name, raw_input_path, atlas_object=atlas_object, subaq_object='AtlasOverview')
class AtlasPreview(Aquisition):
    def __init__(self, aqu_name, atlas_object, raw_input_path=None):
        aqu_name=os.path.split(aqu_name)[1]
        Aquisition.__init__(self, aqu_name, raw_input_path, atlas_object=atlas_object, subaq_object='AtlasPreview')
class AtlasVolume(Aquisition):
    def __init__(self, aqu_name, atlas_object, raw_input_path=None):
        aqu_name=os.path.split(aqu_name)[1]
        Aquisition.__init__(self, aqu_name, raw_input_path, atlas_object=atlas_object, subaq_object='AtlasVolume')         
        
        
        
class Tomato1050Acquisition(Aquisition):
    def __init__(self, aqu_name, FOV_object, raw_input_path=None, ):
        Aquisition.__init__(self, aqu_name, raw_input_path, FOV_object=FOV_object, subaq_object='Tomato1050Acquisition')
               
class Tomato3Plane1050Acquisition(Aquisition):
    def __init__(self, aqu_name, FOV_object, raw_input_path=None, ):
        Aquisition.__init__(self, aqu_name, raw_input_path, FOV_object=FOV_object,subaq_object='Tomato3Plane1050Acquisition')
                 
class TomatoHighResStack1050Acquisition(Aquisition):
    def __init__(self, aqu_name, FOV_object, raw_input_path=None, ):
        Aquisition.__init__(self, aqu_name, raw_input_path, FOV_object=FOV_object,subaq_object='TomatoHighResStack1050Acquisition')
        
class HighResStackGreenAcquisition(Aquisition):
    def __init__(self, aqu_name, FOV_object, raw_input_path=None, ):
        Aquisition.__init__(self, aqu_name, raw_input_path, FOV_object=FOV_object,subaq_object='HighResStackGreenAcquisition')
               
class SurfaceImageAquisition(Aquisition):
    def __init__(self, aqu_name, FOV_object, raw_input_path=None, ):
        Aquisition.__init__(self, aqu_name, raw_input_path, FOV_object=FOV_object,subaq_object='SurfaceImageAquisition')
                 
class OtherAcqAquisition(Aquisition):
    def __init__(self, aqu_name, FOV_object, raw_input_path=None, ):
        Aquisition.__init__(self, aqu_name, raw_input_path, FOV_object=FOV_object,subaq_object='OtherAcqAquisition')
        