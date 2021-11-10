# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:08:27 2021

@author: sp3660
"""
import glob
import os
import time
import numpy as np
import xml.etree.ElementTree as ET
import scipy as sp
import pandas as pd
import shutil
import matplotlib.pyplot as plt

from recursively_read_metadata import recursively_read_metadata



from select_values_gui import select_values_gui
from manually_get_some_metadata import manually_get_some_metadata


#%%
class Metadata():
    
    def __init__(self,  aq_metadataPath=None, 
                 photostim_metadataPath=None,
                 voltagerec_metadataPath=None, 
                 face_camera_metadata_path=False, 
                 voltageoutput_metadataPath=False, 
                 imaging_database_row=None,
                 temporary_path=None,
                 acquisition_directory_raw=None):
        print('Processing Metadata')
        
        self.temporary_path=temporary_path

        
        self.acquisition_directory_raw=acquisition_directory_raw

        self.all_frames=[]
        self.all_volumes=[]
        self.video_params=[]
        self.params=[]
        self.imaging_metadata_file=aq_metadataPath
        self.photostim_file=photostim_metadataPath
        self.voltage_file=voltagerec_metadataPath
        self.voltage_output=voltageoutput_metadataPath
        self.check_metadata_in_folder()

        
        
        
        if  self.imaging_metadata_file:     
            if os.path.isfile(self.imaging_metadata_file):
                    # print('getting metadata')
                    self.process_metadata()
        if self.voltage_file:
            if os.path.isfile(self.voltage_file):        
                    self.process_voltage_recording()
   
   
        # else:
        #     self.add_metadata_manually()
        #     if self.photostim_metadataPath!=None:
        #         if os.path.isfile(self.photostim_file):
        #             self.photstim_metadata=self.process_photostim_metadata()                  
        #             self.photostim_extra_metadata=self.create_photostim_sequence(self.photostim_metadata, self.imaging_metadata)#,
        #             self.photostim_metadata['PhotoStimSeriesArtifact']=self.photostim_extra_metadata

        if self.temporary_path: 
            self.transfer_metadata()   
            
        # if self.video_params:    
        #     self.plotting()    
            
    def process_voltage_recording(self):
        tree = ET.parse(self.voltage_file)
        root = tree.getroot()
        self.full_voltage_recording_metadata=recursively_read_metadata(root)  

        
        voltage_aq_time=root[3].text[root[3].text.find('T')+1:]
        voltage_aq_time=time.strptime(voltage_aq_time[:-9], '%X.%f')
        self.voltage_aq_time=time.strftime('%H:%M:%S', voltage_aq_time)
        
        ExperimentInfo=self.full_voltage_recording_metadata['Experiment']          
        # recorded_channels=[elem['Childs']['VisibleSignals']['VRecSignalPerPlotProperties']['Childs']['SignalId']['Value']['Description'] for elem in ExperimentInfo['Childs']["PlotConfigList"].values() if elem['Childs']['VisibleSignals']['VRecSignalPerPlotProperties']]
        recorded_channels=[elem['Childs']['VisibleSignals']['VRecSignalPerPlotProperties']['Childs']['SignalId']['Value']['Description'] if elem['Childs']['VisibleSignals'] else '' for elem in ExperimentInfo['Childs']["PlotConfigList"].values() ]

        recorded_channels2=[chan[chan.find(')')-1] if chan else '' for chan in recorded_channels ]
        self.recorded_signals=[elem['Childs']['Name']['Description'] for elem in ExperimentInfo['Childs']["SignalList"].values() if elem['Childs']['Channel']['Description'] in recorded_channels2]

        pth=((os.path.splitext(self.voltage_file)[0])+'.csv')
        voltage_rec_csv= pd.read_csv(pth,header=0)
        recorded_signals_csv=list(voltage_rec_csv.columns)
        recorded_signals_csv.pop(0)
        self.recorded_signals_csv=[i.strip() for i in recorded_signals_csv]

    def process_metadata(self):

        tree = ET.parse( self.imaging_metadata_file)       
        root = tree.getroot()
        self.full_metadata=recursively_read_metadata(root)  
        if not self.full_metadata:
            return []
        else:
            MicroscopeInfo=self.full_metadata['PVStateShard']
            seqinfo=self.full_metadata['Sequence']

            self.params={'ImagingTime':time.strftime('%H:%M:%S',time.strptime(self.full_metadata['date'],'%m/%d/%Y %X %p')),
                    'Date':self.full_metadata['date'],
                    'AquisitionName':os.path.splitext(os.path.basename(self.imaging_metadata_file))[0]}
              
            for element in MicroscopeInfo['Childs'].values():
                
                if element['key']=='activeMode':
                    self.params['ScanMode']= element['value']
                    
                if element['key']=='bitDepth':
                    self.params['BitDepth']=int(element['value'])
                    
                if element['key']=='dwellTime':
                    self.params['dwellTime']=float(element['value'])
                    
                if element['key']=='framePeriod':
                    self.params['framePeriod']=float(element['value'] )
                    
                if element['key']=='laserPower':
                    self.params['ImagingLaserPower']=float(element['IndexedValue']['value'])
                    
                if element['key']=='laserPower':
                    self.params['UncagingLaserPower']=float(element['IndexedValue_1']['value'] )
                    
                if element['key']=='linesPerFrame':
                    self.params['LinesPerFrame']=int(element['value'])
                    
                if element['key']=='micronsPerPixel':
                    self.params['MicronsPerPixelX']=float(element['IndexedValue']['value'])
                    
                if element['key']=='micronsPerPixel':
                    self.params['MicronsPerPixelY']=float(element['IndexedValue_1']['value'])
                    
                if element['key']=='objectiveLens':
                    self.params['Objective']=element['value']
                    
                if element['key']=='objectiveLensMag':
                    self.params['ObjectiveMag']=int(element['value'])
                    
                if element['key']=='objectiveLensNA':
                    self.params['ObjectiveNA']=float(element['value'])
                    
                if element['key']=='opticalZoom':
                    self.params['OpticalZoom']=float(element['value'] )
                    
                if element['key']=='pixelsPerLine':
                    self.params['PixelsPerLine']=int(element['value'])
                    
                if element['key']=='pmtGain':
                    self.params['PMTGainRed']=float(element['IndexedValue']['value'] )
                    
                if element['key']=='pmtGain':
                    self.params['PMTGainGreen']=float(element['IndexedValue_1']['value'])
                    
                if element['key']=='positionCurrent':
                    self.params['PositionX']=float(element['SubindexedValues']['Childs']['SubindexedValue']['value'])
                    
                if element['key']=='positionCurrent':
                    self.params['PositionY']=float(element['SubindexedValues_1']['Childs']['SubindexedValue']['value'])
                    
                if element['key']=='positionCurrent':
                    self.params['PositionZphysical']=float(element['SubindexedValues_2']['Childs']['SubindexedValue']['value'])
                    
                if element['key']=='positionCurrent':
                    self.params['PositionZETL']=float(element['SubindexedValues_2']['Childs']['SubindexedValue_1']['value'])
                    
                if element['key']=='rastersPerFrame':
                    self.params['RasterAveraging']=int(element['value'])
                    
                if element['key']=='resonantSamplesPerPixel':
                    self.params['ResonantSampling']=int(element['value'] )
                    
                if element['key']=='scanLinePeriod':
                    self.params['ScanLinePeriod']=float(element['value'])
                    
                if element['key']=='zDevice':
                    self.params['ZDevice']=int(element['value'] )
                    
            self.video_params={'MultiplanePrompt':seqinfo['type'],                                  
                          'ParameterSet':seqinfo['Childs']["Frame"]['parameterSet'],
                          'RedChannelName':'No Channel',
                          'GreenChannelName':'No Channel',           
                          'FrameNumber':int(len( [x for x in seqinfo['Childs'] if 'Frame' in x])),
                          'PlaneNumber':'Single',
                          'PlanePositionsOBJ':self.params['PositionZphysical'],
                          'PlanePositionsETL':self.params['PositionZETL'],           
                          'Planepowers':self.params['ImagingLaserPower'], 
                          'XPositions':self.params['PositionX'],
                          'YPositions':self.params['PositionY'],
                          'pmtGains_Red':self.params['PMTGainRed'],
                          'pmtGains_Green':self.params['PMTGainGreen'],
                         
                          }

            # MultiPlane=0
            SingleChannel=0
 #%% here to create the full frame by frame volume by volume 
            if len(list(root))>3 and (self.video_params['MultiplanePrompt']=="TSeries ZSeries Element" or self.video_params['MultiplanePrompt']=="AtlasVolume"):
                FirstVolumeMetadat=root[2]
                del self.video_params['FrameNumber']
                self.video_params['VolumeNumber']=int(len(list(root.findall('Sequence'))))
                # MultiPlane=1
                self.all_volumes=[]            
                volumes={key:volume for key, volume in self.full_metadata.items() if 'Sequence' in  key}
                
                
                if self.video_params['MultiplanePrompt']=="AtlasVolume":
                    self.video_params['StageGridYOverlap']=volumes[list(volumes.keys())[0]]['xYStageGridYOverlap']
                    self.video_params['StageGridXOverlap']=volumes[list(volumes.keys())[0]]['xYStageGridXOverlap']
                    self.video_params['StageGridOverlapPercentage']=volumes[list(volumes.keys())[0]]['xYStageGridOverlapPercentage']
                    self.video_params['StageGridNumYPositions']=volumes[list(volumes.keys())[0]]['xYStageGridNumYPositions']
                    self.video_params['StageGridNumXPositions']=volumes[list(volumes.keys())[0]]['xYStageGridNumXPositions']
                
                for i, volume in enumerate(volumes.values()):
                    all_planes={}
                    planes={ key:plane for key, plane in volume['Childs'].items() if 'Frame' in key}
                    for i, plane in enumerate(planes.values()):
                         iplane={}
                         iplane['scanLinePeriod']='Default_' + str(self.params['ScanLinePeriod'])
                         iplane['absoluteTime']=float(plane['absoluteTime'])
                         iplane['index']=int(plane['index'])
                         iplane['relativeTime']=float(plane['relativeTime'])
                         
                         iplane['LastGoodFrame']=int(plane['ExtraParameters']['lastGoodFrame'])
                         ExtraMetadata=plane['PVStateShard']
                         iplane['ImagingSlider']=self.params['ImagingLaserPower']
                         iplane['UncagingSlider']=self.params['UncagingLaserPower']
                         iplane['XAxis']='Default_' + str(self.params['PositionX'])
                         iplane['YAxis']='Default_' + str(self.params['PositionY'])
                         iplane['ObjectiveZ']='Default_' +str(self.video_params['PlanePositionsOBJ'])
                         iplane['ETLZ']='Default_' +str(self.video_params['PlanePositionsETL'])


                            
                        # plane['File']['channelName']
                        # plane['File_1']['channelName']


                         for element in ExtraMetadata['Childs'].values():
                             
                             if 'framePeriod' in element.values():
                                 iplane['framePeriod']=float(element['value'])
                                 
                             if 'scanLinePeriod' in element.values():
                                 iplane['scanLinePeriod']=float(element['value'])
                                 
                             if 'laserPower' in element.values():
                                 for element2 in  element.values():
                                     if isinstance(element2,dict):
                                         if 'Imaging' in element2.values():
                                            iplane['ImagingSlider']=float(element2['value'])
                                         if 'Uncaging' in element2.values():
                                            iplane['UncagingSlider']=float(element2['value'])
                                                                   
                             if 'positionCurrent' in element.values():
                                 for element2 in  element.values():
                                     if isinstance(element2,dict):
                                         if 'XAxis' in element2.values():
                                             iplane['XAxis']=float(element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
                                         if 'YAxis' in element2.values():
                                             iplane['YAxis']=float(element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
    
                                         if 'ZAxis' in element2.values():
                                             iplane['ObjectiveZ']=float(element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
                                             iplane['ETLZ']=float(element2['Childs'][list(element2['Childs'].keys())[1]]['value'])
                                         
                             if 'pmtGain' in element.values():
                                 for element2 in element.values():
                                     if isinstance(element2, dict):
                                         iplane['pmtGain_'+element2['description']]=float(element2['value'])
        
                         all_planes[i]=iplane
                    # this is because sometime sthe first xy positions were workng     
                    all_planes[list(all_planes.keys())[0]]['XAxis']=all_planes[list(all_planes.keys())[1]]['XAxis']
                    all_planes[list(all_planes.keys())[0]]['YAxis']=all_planes[list(all_planes.keys())[1]]['YAxis']  
                    self.all_volumes.append(all_planes)  

                    # this is because i had metadata with 1 extra volume with 2 planes only
                if len(self.all_volumes[-1])!=len(self.all_volumes[-2]):
                    self.all_volumes.pop(-1)
                    self.video_params['VolumeNumber']=self.video_params['VolumeNumber']-1

            elif len(list(root))<=3 and self.video_params['MultiplanePrompt']=="TSeries ZSeries Element" :
                FirstVolumeMetadat=root[2]
                del self.video_params['FrameNumber']
                self.video_params['VolumeNumber']=int(len(list(root.findall('Sequence'))))
                # MultiPlane=1
                self.all_volumes=[]            
                volumes={key:volume for key, volume in self.full_metadata.items() if 'Sequence' in  key}
                
                
                if self.video_params['MultiplanePrompt']=="AtlasVolume":
                    self.video_params['StageGridYOverlap']=volumes[list(volumes.keys())[0]]['xYStageGridYOverlap']
                    self.video_params['StageGridXOverlap']=volumes[list(volumes.keys())[0]]['xYStageGridXOverlap']
                    self.video_params['StageGridOverlapPercentage']=volumes[list(volumes.keys())[0]]['xYStageGridOverlapPercentage']
                    self.video_params['StageGridNumYPositions']=volumes[list(volumes.keys())[0]]['xYStageGridNumYPositions']
                    self.video_params['StageGridNumXPositions']=volumes[list(volumes.keys())[0]]['xYStageGridNumXPositions']
                
                for i, volume in enumerate(volumes.values()):
                    all_planes={}
                    planes={ key:plane for key, plane in volume['Childs'].items() if 'Frame' in key}
                    for i, plane in enumerate(planes.values()):
                         iplane={}
                         iplane['absoluteTime']=float(plane['absoluteTime'])
                         iplane['index']=int(plane['index'])
                         iplane['relativeTime']=float(plane['relativeTime'])
                         iplane['framePeriod']='Default_' + str(self.params['framePeriod'])

                         iplane['LastGoodFrame']='Default_0'
                         iplane['scanLinePeriod']='Default_' + str(self.params['ScanLinePeriod'])
 
                         iplane['LastGoodFrame']=int(plane['ExtraParameters']['lastGoodFrame'])
                         ExtraMetadata=plane['PVStateShard']
                         iplane['ImagingSlider']=self.params['ImagingLaserPower']
                         iplane['UncagingSlider']=self.params['UncagingLaserPower']
                         iplane['XAxis']='Default_' + str(self.params['PositionX'])
                         iplane['YAxis']='Default_' + str(self.params['PositionY'])
                         iplane['ObjectiveZ']='Default_' +str(self.video_params['PlanePositionsOBJ'])
                         iplane['ETLZ']='Default_' +str(self.video_params['PlanePositionsETL'])
          
          
                            
                        # plane['File']['channelName']
                        # plane['File_1']['channelName']
          
          
                         for element in ExtraMetadata['Childs'].values():
                             
                             if 'framePeriod' in element.values():
                                 iplane['framePeriod']=float(element['value'])
                                 
                             if 'scanLinePeriod' in element.values():
                                 iplane['scanLinePeriod']=float(element['value'])
                                 
                             if 'laserPower' in element.values():
                                 for element2 in  element.values():
                                     if isinstance(element2,dict):
                                         if 'Imaging' in element2.values():
                                            iplane['ImagingSlider']=float(element2['value'])
                                         if 'Uncaging' in element2.values():
                                            iplane['UncagingSlider']=float(element2['value'])
                                                                   
                             if 'positionCurrent' in element.values():
                                 for element2 in  element.values():
                                     if isinstance(element2,dict):
                                         if 'XAxis' in element2.values():
                                             iplane['XAxis']=float(element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
                                         if 'YAxis' in element2.values():
                                             iplane['YAxis']=float(element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
          
                                         if 'ZAxis' in element2.values():
                                             iplane['ObjectiveZ']=float(element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
                                             iplane['ETLZ']=float(element2['Childs'][list(element2['Childs'].keys())[1]]['value'])
                                         
                             if 'pmtGain' in element.values():
                                 for element2 in element.values():
                                     if isinstance(element2, dict):
                                         iplane['pmtGain_'+element2['description']]=float(element2['value'])
          
                         all_planes[i]=iplane
                    # this is because sometime sthe first xy positions were workng     
                    all_planes[list(all_planes.keys())[0]]['XAxis']=all_planes[list(all_planes.keys())[1]]['XAxis']
                    all_planes[list(all_planes.keys())[0]]['YAxis']=all_planes[list(all_planes.keys())[1]]['YAxis']  
                    self.all_volumes.append(all_planes)  
          

            elif self.video_params['MultiplanePrompt']=='AtlasPreview' or self.video_params['MultiplanePrompt']=='AtlasOverview':
                
                seqinfo=self.full_metadata['Sequence']
                self.all_frames=[]
                self.video_params['StageGridYOverlap']=seqinfo['xYStageGridYOverlap']
                self.video_params['StageGridXOverlap']=seqinfo['xYStageGridXOverlap']
                self.video_params['StageGridOverlapPercentage']=seqinfo['xYStageGridOverlapPercentage']
                self.video_params['StageGridNumYPositions']=seqinfo['xYStageGridNumYPositions']
                self.video_params['StageGridNumXPositions']=seqinfo['xYStageGridNumXPositions']
                
                for key, frame in seqinfo['Childs'].items() :
                    if 'Frame' in  key:
                        iframe={}
                        iframe['LastGoodFrame']='Default_0'
                        iframe['scanLinePeriod']='Default_' + str(self.params['ScanLinePeriod'])
                        iframe['absoluteTime']=float(frame['absoluteTime'])
                        iframe['index']=int(frame['index'])
                        iframe['relativeTime']=float(frame['relativeTime'])
                        if 'ExtraParameters' in frame:     
                            iframe['LastGoodFrame']=int(frame['ExtraParameters']['lastGoodFrame'])
                        ExtraMetadata=frame['PVStateShard']
                        
                        for element in ExtraMetadata['Childs'].values():
                        
                            if 'framePeriod' in element.values():
                                iframe['framePeriod']=float(element['value'])
                                
                            if 'scanLinePeriod' in element.values():
                                iframe['scanLinePeriod']=float(element['value'])
                                
                            if 'laserPower' in element.values():
                                for element2 in  element.values():
                                    if isinstance(element2,dict):
                                        if 'Imaging' in element2.values():
                                           iframe['ImagingSlider']=float(element2['value'])
                                        if 'Uncaging' in element2.values():
                                           iframe['UncagingSlider']=float(element2['value'])
                                                                  
                            if 'positionCurrent' in element.values():
                                for element2 in  element.values():
                                    if isinstance(element2,dict):
                                        if 'XAxis' in element2.values():
                                            iframe['XAxis']=float(element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
                                        if 'YAxis' in element2.values():
                                            iframe['YAxis']=float(element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
    
                                        if 'ZAxis' in element2.values():
                                            iframe['ObjectiveZ']=float(element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
                                            iframe['ETLZ']=float(element2['Childs'][list(element2['Childs'].keys())[1]]['value'])
                                        
                        self.all_frames.append(iframe)
                        
 
            else:
                seqinfo=self.full_metadata['Sequence']
                self.all_frames=[]
                for key, frame in seqinfo['Childs'].items() :
                    if 'Frame' in  key:
                        iframe={}
                        iframe['framePeriod']='Default_' + str(self.params['framePeriod'])
                        iframe['LastGoodFrame']='Default_0'
                        iframe['scanLinePeriod']='Default_' + str(self.params['ScanLinePeriod'])
                        iframe['absoluteTime']=float(frame['absoluteTime'])
                        iframe['index']=int(frame['index'])
                        iframe['relativeTime']=float(frame['relativeTime'])
                        if 'ExtraParameters' in frame:     
                            iframe['LastGoodFrame']=int(frame['ExtraParameters']['lastGoodFrame'])
                            
                        if 'PVStateShard' in  frame:
                            ExtraMetadata=frame['PVStateShard']
                            iframe['framePeriod']=float(ExtraMetadata['Childs']['PVStateValue']['value'])
                            if len(ExtraMetadata['Childs'])>1:
                                if 'value' in ExtraMetadata['Childs'][list(ExtraMetadata['Childs'].keys())[-1]]:
                                    iframe['scanLinePeriod']=float(ExtraMetadata['Childs'][list(ExtraMetadata['Childs'].keys())[-1]]['value'])                               
                        self.all_frames.append(iframe)

 #%% doind video params 
 
 
            if self.video_params['MultiplanePrompt']=="AtlasVolume":
                  self.video_params['XPositions']=[position[list(position.keys())[0]]['XAxis']  for position in self.all_volumes]
                  self.video_params['YPositions']=[position[list(position.keys())[0]]['YAxis']  for position in self.all_volumes]                 
              
            elif self.video_params['MultiplanePrompt']=="TSeries ZSeries Element":
                  self.video_params['XPositions']=self.all_volumes[0][list(self.all_volumes[0].keys())[0]]['XAxis'] 
                  self.video_params['YPositions']=self.all_volumes[0][list(self.all_volumes[0].keys())[0]]['YAxis']     
                  
                  
            elif self.video_params['MultiplanePrompt']=='AtlasPreview' or self.video_params['MultiplanePrompt']=='AtlasOverview'  :
                self.video_params['XPositions']=[frame['XAxis']  for frame in self.all_frames]
                self.video_params['YPositions']=[frame['YAxis']  for frame in self.all_frames]
                


            if self.video_params['MultiplanePrompt']=="TSeries ZSeries Element" or self.video_params['MultiplanePrompt']=="AtlasVolume": 
                     
                self.video_params['FullAcquisitionTime']=self.all_volumes[-1][list(self.all_volumes[-1].keys())[-1]]['relativeTime']  
                self.video_params['PlaneNumber']=len(list(FirstVolumeMetadat.findall('Frame')))                           
                self.video_params['PlanePositionsOBJ']=[self.all_volumes[0][key]['ObjectiveZ'] for key in self.all_volumes[0].keys()]
                self.video_params['PlanePositionsETL']=[self.all_volumes[0][key]['ETLZ'] for key in self.all_volumes[0].keys()]
                self.video_params['Planepowers']= [self.all_volumes[0][key]['ImagingSlider'] for key in self.all_volumes[0].keys()]
                self.video_params['pmtGains_Red']= [self.all_volumes[0][key]['pmtGain_Red'] for key in self.all_volumes[0].keys() if 'pmtGain_Red' in self.all_volumes[0][key].keys() ]
                self.video_params['pmtGains_Green']= [self.all_volumes[0][key]['pmtGain_Green'] for key in self.all_volumes[0].keys() if 'pmtGain_Green' in self.all_volumes[0][key].keys() ]
                
                if not   self.video_params['pmtGains_Red']:
                    self.video_params['pmtGains_Red']=self.params['PMTGainRed']
                if not  self.video_params['pmtGains_Green']:
                        self.video_params['pmtGains_Green']=self.params['PMTGainGreen']
      
                self.video_params['relativeTimes']= [[position[key]['relativeTime'] for key in self.all_volumes[0].keys()] for position in self.all_volumes]
                self.video_params['absoluteTimes']= [[position[key]['absoluteTime'] for key in self.all_volumes[0].keys()] for position in self.all_volumes]
                self.video_params['scanLinePeriods']= [[position[key]['scanLinePeriod'] for key in self.all_volumes[0].keys()] for position in self.all_volumes]
                self.video_params['framePeriods']= [[position[key]['framePeriod'] for key in self.all_volumes[0].keys()] for position in self.all_volumes]
            
                self.video_params['lastGoodFrames']= [[position[key]['LastGoodFrame'] for key in self.all_volumes[0].keys()] for position in self.all_volumes]

            else:
 
               self.video_params['FullAcquisitionTime']=self.all_frames[-1]['relativeTime']
               self.video_params['relativeTimes']= [position['relativeTime']  for position in self.all_frames]
               self.video_params['absoluteTimes']= [position['absoluteTime']  for position in self.all_frames]
               self.video_params['scanLinePeriods']= [position['scanLinePeriod']  for position in self.all_frames]
               self.video_params['framePeriods']= [position['framePeriod']  for position in self.all_frames]
               self.video_params['lastGoodFrames']= [position['LastGoodFrame']  for position in self.all_frames]
            

            if seqinfo['Childs']['PVStateShard'] and self.video_params['MultiplanePrompt']=="AtlasVolume":
                SingleChannel=0 
                files={key:val for key, val in seqinfo['Childs']['Frame'].items() if 'File' in key}
                for chan in files.values():
                    ChannelName=chan['filename']
                    if 'Green' in chan.values() :              
                        self.video_params['GreenChannelName']=ChannelName
                    elif 'Red' in chan.values():
                        self.video_params['RedChannelName']=ChannelName
                if not all( self.video_params['RedChannelName'] and self.video_params['GreenChannelName']):
                    SingleChannel=1
            
            elif seqinfo['Childs']['PVStateShard'] and not self.video_params['MultiplanePrompt']=="AtlasVolume":             
                SingleChannel=0 
                files={key:val for key, val in seqinfo['Childs']['Frame'].items() if 'File' in key}
                for chan in files.values():
                    ChannelName=chan['filename']
                    if 'Green' in chan.values() :              
                        self.video_params['GreenChannelName']=ChannelName
                    elif 'Red' in chan.values():
                        self.video_params['RedChannelName']=ChannelName
                if not all( self.video_params['RedChannelName'] and self.video_params['GreenChannelName']):
                    SingleChannel=1
   
            else:
                SingleChannel=0 
                files={key:val for key, val in seqinfo['Childs']['Frame'].items() if 'File' in key}
                for chan in files.values():
                    ChannelName=chan['filename']
                    if 'Green' in chan.values() :              
                        self.video_params['GreenChannelName']=ChannelName
                    elif 'Red' in chan.values():
                        self.video_params['RedChannelName']=ChannelName
                if not all( self.video_params['RedChannelName'] and self.video_params['GreenChannelName']):
                    SingleChannel=1
                    
                    
        if self.all_frames:        
            self.imaging_metadata=[self.params, self.video_params, self.all_frames]
        if self.all_volumes:        
            self.imaging_metadata=[self.params, self.video_params, self.all_volumes]
  #%% Plotting
    def plotting(self):
        
            f, axs= plt.subplots(5, 1, sharex=True)
            axs[0].plot(np.array(self.video_params['relativeTimes']).flatten())
            axs[1].plot(np.array(self.video_params['absoluteTimes']).flatten())
            axs[2].plot(np.array(self.video_params['scanLinePeriods']).flatten())
            axs[3].plot(np.array(self.video_params['framePeriods']).flatten())
            axs[4].plot(np.array(self.video_params['lastGoodFrames']).flatten())

            
            f, axs= plt.subplots(2, 1, sharex=True)
            axs[0].plot(np.array(self.video_params['XPositions']).flatten(), 'x')
            axs[1].plot(np.array(self.video_params['YPositions']).flatten(), 'x')
            
            f, axs= plt.subplots(3, 1, sharex=True)
            axs[0].plot(np.array(self.video_params['Planepowers']).flatten(), 'x')
            axs[1].plot(np.array(self.video_params['pmtGains_Red']).flatten(), 'x')
            axs[2].plot(np.array(self.video_params['pmtGains_Green']).flatten(), 'x')

#%% manual and photostim metadata
    def add_metadata_manually(self):   
        # print('getting manual meta1data')
        
        self.imaging_metadata=[{},{},[]]
        aquisition_to_process=os.path.split(self.imaging_metadata_file)[0]
        temp_path=os.path.split(os.path.split(self.imaging_metadata_file)[0])[0]
        aquisition_date=aquisition_to_process[temp_path.find('\SP')-8:temp_path.find('\SP')-1]
        formated_aquisition_date=aquisition_date
        
        self.imaging_metadata[0]['Date']=formated_aquisition_date
        self.imaging_metadata[0]['AquisitionName']=os.path.split(aquisition_to_process)[1]

        self.imaging_metadata[1]['MultiplanePrompt']=select_values_gui(["TSeries ZSeries Element","TSeries ImageSequence Element"])
        
        mtdata=manually_get_some_metadata()
        
        self.imaging_metadata[1]['Plane Number']=int(mtdata[0])
        self.imaging_metadata[1]['Volume Number']=int(mtdata[2])
        self.imaging_metadata[1]['Frame Number']=int(mtdata[2])
        self.imaging_metadata[0]['Lines per Frame']=mtdata[1]
        self.imaging_metadata[0]['Pixels Per Line']=mtdata[1]
        
    def process_photostim_metadata(self):

        tree = ET.parse(self.photostim_file)
        root = tree.getroot()

        Experiment={ 'Iterations': root.attrib['Iterations'],
                               'Iteration Delay': root.attrib['IterationDelay'],
                              'PhotoStimSeries':{}}
        
        i=0
        
        for photostim in root:   
            Experiment['PhotoStimSeries']['PhotostimExperiment'+str(i+1)]={}
            sequence={'Point Order': photostim[0].attrib['Indices'],
                        'StimDuration':photostim[0].attrib['Duration'],
                        'InterpointDuration':photostim[0].attrib['InterPointDelay'],
                        'Repetitions':photostim.attrib['Repetitions'],                                                                        
                        'RelativeDelay':photostim[0].attrib['InitialDelay'],
                        'SpiralRevolutions':photostim[0].attrib['SpiralRevolutions'],
                        }
            Experiment['PhotoStimSeries']['PhotostimExperiment'+str(i+1)]['sequence']=sequence
            points=[] 
            
            for point in photostim[0]:
                points.append({'index':point.attrib['Index'],
                             'x_pos':point.attrib['X'],
                             'y_pos':point.attrib['Y'],
                             'spiral':point.attrib['IsSpiral'],
                             'spiral_width':point.attrib['SpiralWidth'],
                             'spiral_height':point.attrib['SpiralHeight'],
                             'spiral_size_microns':point.attrib['SpiralSizeInMicrons']})
                Experiment['PhotoStimSeries']['PhotostimExperiment'+str(i+1)]['points']=points
            i=i+1   
            points=[]    
          
        return Experiment                                       

    def create_photostim_sequence(photmet, aqumet, pokels_signal=[]):
                                                                
        timestamps = [tuple(l) for l in aqumet[2]]
                
        dt=np.dtype('double,double,double')
        zz=np.array(timestamps,dtype=dt)
        zz.dtype.names=['Frame','AbsTime','RelTime']
        dd=zz.view((float, len(zz.dtype.names)))
      
        ddd=dd[:,[0,2]]
        ddd_miliseconds=ddd
        ddd_miliseconds[:,1]=ddd_miliseconds[:,1]*1000
        stim_delays={}
        for key in photmet['PhotoStimSeries']:
            stim_delays[key]= photmet['PhotoStimSeries'][key]['sequence']['RelativeDelay']
            stim_delays[key]=int(stim_delays[key])
            stim_delays[key]=stim_delays[key]/1000
        exp_total_duration={}
        for key in photmet['PhotoStimSeries']:    
            exp_total_duration[key]= (int(photmet['PhotoStimSeries'][key]['sequence']['StimDuration']) + 
                                    int(photmet['PhotoStimSeries'][key]['sequence']['InterpointDuration'])) * \
                                    len((photmet['PhotoStimSeries'][key]['points'])) *\
                                    int(photmet['PhotoStimSeries'][key]['sequence']['Repetitions'])
     
    
            
            
        def dict_zip(*dicts):
            all_keys = {k for d in dicts for k in d.keys()}
            return {k: [d[k] for d in dicts if k in d] for k in all_keys}
        
        zzz=dict_zip(stim_delays,exp_total_duration)
        
        sortedDict = dict( sorted(zzz.items(), key=lambda x: x[0].lower()) )
        
        zzz_sorted_list=list(sortedDict.values())
        stim_table=[]
        
        
        for i,j in enumerate(zzz_sorted_list):
            if i==0:
                stim_info=[]
                stim_info.append(0)
                stim_info.append(zzz_sorted_list[i][0])
                stim_info.append(stim_info[0]+stim_info[1] )    
                stim_info.append(round(zzz_sorted_list[i][1]/1000,2))   
                stim_info.append(stim_info[2]+round(stim_info[3],2))
                stim_table.append(stim_info)
                
            else:  
                stim_info=[]
                stim_info.append(round(stim_table[i-1][4],2))
                stim_info.append(zzz_sorted_list[i][0])
                stim_info.append(stim_info[0]+ stim_info[1] )    
                stim_info.append(round(zzz_sorted_list[i][1]/1000,2))   
                stim_info.append(round(stim_info[2]+stim_info[3],2))
                stim_table.append(stim_info)
                
            
    #%% fill artifical pockels cell signal
        if not pokels_signal:
            time_s=round(int(aqumet[1]['Frame Number'])*float(aqumet[1]['FramePeriod'])*int(aqumet[0]['RasterAveraging']))#s
            freq_s=1000#hz
            samples=time_s*freq_s
            high_freq_signals=np.linspace(0,samples,samples, dtype='int_')
            high_freq_signals
            stim_table
            stim_table_ms=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
            
            for i, j in enumerate(stim_table):
                for k, l in enumerate(stim_table[i]):
                    stim_table_ms[i][k]=stim_table[i][k]*1000
                    
            index_to_fill=[] 
            len(high_freq_signals)   
            for m, n in enumerate(stim_table_ms):         
                index_to_fill.append(np.arange(stim_table_ms[m][2],stim_table_ms[m][4]).astype(int))
            
            
                  
            new_column =  np.zeros (len(high_freq_signals)).astype(int)
            high_freq_signals_full=np.vstack((high_freq_signals,new_column)).transpose()
            cum_indx_fil=[]
            for o, p in enumerate(index_to_fill):
                cum_indx_fil=index_to_fill[o][len(index_to_fill[o])-1]
                if cum_indx_fil>samples:
                    break
                else:
                    high_freq_signals_full[index_to_fill[o],1]=5
        
        # fig, ax = plt.subplots()  # Create a figure containing a single axes.
        # ax.plot(high_freq_signals_full[:,0], high_freq_signals_full[:,1])
        resampled_laser=sp.signal.resample(high_freq_signals_full[:,1],len(ddd))
        # fig, ax = plt.subplots()  # Create a figure containing a single axes.
        # ax.plot(resampled_laser[:])
        filtered_resampled_laser=sp.signal.medfilt(resampled_laser,5)
        # fig, ax = plt.subplots()  # Create a figure containing a single axes.
        # ax.plot(filtered_resampled_laser[:])
        artifact_idx = np.where(filtered_resampled_laser>1)
        PhotoStimSeriesArtifact={'artifact_idx':artifact_idx, 'stim_table_ms':stim_table_ms,'Processed_Pockels_Signal' :filtered_resampled_laser}
        
        return PhotoStimSeriesArtifact   
 #%%   new functions
    def transfer_metadata(self):    

        self.metadata_raw_files_full_path=[file for file in glob.glob(self.acquisition_directory_raw+'\\**', recursive=False) if '.xml' in file  ]
        self.transfered_metadata_paths=[]
        for file in self.metadata_raw_files_full_path:
            shutil.copy(file, self.temporary_path)
            self.transfered_metadata_paths.append(os.path.join(self.temporary_path, file))
            
    def check_metadata_in_folder(self):
     if self.acquisition_directory_raw:
            xmlfiles=glob.glob(self.acquisition_directory_raw+'\\**.xml')
            for xml in xmlfiles:
                if 'Cycle' not in xml:
                    self.imaging_metadata_file=xml
                if 'VoltageRecording' in xml:
                    self.voltage_file=xml 
                # if 'VoltageRecording' in xml:
                #      self.photostim_file=xml 
                # if 'VoltageRecording' in xml:
                #     face_camera_metadata_path=xml 
                if 'VoltageOutput' in xml:
                    self.voltage_output=xml 

    def read_metadata_from_database(self):
        print('TO DO')
    
    
    
#%%    
if __name__ == "__main__":
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIFFrameNumberK3planeallen\Plane3'
    # path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20211007\Mice\SPIK\FOV_1\Aq_1\211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000'   
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\Plane3'
    # path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20211015\Mice\SPKG\FOV_1\Aq_1\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop'
    # path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20210917\Mice\SPGT\Atlas_1\Volumes\Aq_1\20210917_SPGT_Atlas_1050_50024_607_without_105_135z_5z_1ol-005'
    # path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20210930\Mice\SPKI\FOV_1\Aq_1\210930_SPKI_FOV1_AllenA_920_50024_narrow_without-000'
    path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20211022\Mice\SPKS\FOV_1\Aq_1\211022_SPKS_FOV1_AllenA_20x_920_50024_narrow_with-000'

    # path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20210929\Mice\SPIC\TestAcquisitions\Aq_1\210929_SPIC_TestVideo5min_920_50024_narrow_without-000'
    # path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20210916\Mice\SPGT\Atlas_1\Preview\Aq_1\20210916_SPGT_Atlas_920_50024_607_without-Preview-000'
    # path='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20210914\Mice\SPGT\Atlas_1\Overview\Aq_1\20210914_SPGT_AtlasOverview_930_50024_607_without-Overview-000'
    # temporary_path1='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000'
    temporary_path1='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211022_SPKS_FOV1_AllenA_20x_920_50024_narrow_with-000'
    temporary_path1='\\\\?\\'+r'F:\Projects\LabNY\Imaging\2021\20211015\Mice\SPKJ\FOV_1\1050_3PlaneTomato\Aq_1\211015_SPKJ_FOV1_1050tomato3plane_990_50024_narrow_without-000'



    
    xmlfile_voltage=[]
    xmlfiles=glob.glob(path+'\\**.xml')
    xmlfile_imaging=xmlfiles[0]
    for xml in xmlfiles:
        if 'VoltageRecording' in xml:
            xmlfile_voltage=xml

    
    # meta =Metadata(aq_metadataPath=image_sequence_directory_full_path, temporary_path=temporary_path)
    # meta =Metadata(aq_metadataPath=xmlfile_imaging, voltagerec_metadataPath=xmlfile_voltage)
    # meta =Metadata(voltagerec_metadataPath=xmlfile_voltage)
    meta =Metadata(acquisition_directory_raw=temporary_path1)
    # meta =Metadata(acquisition_directory_raw=path, temporary_path=temporary_path1)





   