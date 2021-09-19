# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:08:27 2021

@author: sp3660
"""
import sys
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/AllFunctions')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/ProcessingScripts')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/MainClasses')
import os
import time
import numpy as np
import xml.etree.ElementTree as ET
import scipy as sp



from ...AllFunctions.select_values_gui import select_values_gui
from ...AllFunctions.manually_get_some_metadata import manually_get_some_metadata



import pandas as pd
#%%
class Metadata():
    
    def __init__(self,  aq_metadataPath=None, photostim_metadataPath=None, voltagerec_metadataPath=None, face_camera_metadata_path=False, voltageoutput_metadataPath=False, imaging_database_row=None):
        
        if aq_metadataPath:        
                self.imaging_metadata_file=aq_metadataPath
                self.photostim_file=photostim_metadataPath
                self.voltage_file=voltagerec_metadataPath
                self.voltage_output=voltageoutput_metadataPath
                
                if os.path.isfile(self.imaging_metadata_file):
                        print('getting metadata')
                        self.imaging_metadata=self.process_metadata()
                if  voltagerec_metadataPath:       
                    if os.path.isfile(self.voltage_file):        
                            self.voltage_recording_metadata=self.process_voltage_recording()
                    else: 
                         self.voltage_recording_metadata=[]
        else:
            self.add_metadata_manually()
            if self.photostim_metadataPath!=None:
                if os.path.isfile(self.photostim_file):
                    self.photstim_metadata=self.process_photostim_metadata()
                   
                    self.photostim_extra_metadata=self.create_photostim_sequence(self.photostim_metadata, self.imaging_metadata)#,
                    self.photostim_metadata['PhotoStimSeriesArtifact']=self.photostim_extra_metadata
                        
                    
    def process_voltage_recording(self):
        tree = ET.parse(self.voltage_file)
        root = tree.getroot()
        
        voltage_aq_time=root[3].text[root[3].text.find('T')+1:]
        voltage_aq_time=time.strptime(voltage_aq_time[:-9], '%X.%f')
        voltage_aq_time=time.strftime('%H:%M:%S', voltage_aq_time)
        
        ExperimentInfo=root[0]          
        recorded_channels=[elem.find('VisibleSignals').find('VRecSignalPerPlotProperties').find('SignalId').find('Value').text for elem in ExperimentInfo.find("PlotConfigList") if elem.find('VisibleSignals').find('VRecSignalPerPlotProperties')]
        recorded_channels2=[chan[chan.find(')')-1] for chan in recorded_channels]
        recorded_signals=[elem.find('Name').text for elem in ExperimentInfo.find("SignalList") if elem.find('Channel').text in recorded_channels2]

        pth=((os.path.splitext(self.voltage_file)[0])+'.csv')
        voltage_rec_csv= pd.read_csv(pth,header=0)
        recorded_signals_csv=list(voltage_rec_csv.columns)
        recorded_signals_csv.pop(0)
        recorded_signals_csv=[i.strip() for i in recorded_signals_csv]
        
       
        return [voltage_aq_time, recorded_signals, recorded_signals_csv]

    def process_metadata(self):

        tree = ET.parse( self.imaging_metadata_file)       
        root = tree.getroot()
        if not root.find('PVStateShard'):
            return []
        else:
            MicroscopeInfo=root.find('PVStateShard')
            seqinfo=root.find('Sequence')
            
                
            
            params={'ImagingTime':time.strftime('%H:%M:%S',time.strptime(root.attrib['date'],'%m/%d/%Y %X %p')),
                    'Date':root.attrib['date'],
                    'AquisitionName':os.path.splitext(os.path.basename(self.imaging_metadata_file))[0]}
            
            for element in MicroscopeInfo:
                
                if element.attrib['key']=='activeMode':
                    params['ScanMode']= element.attrib['value']
                    
                if element.attrib['key']=='bitDepth':
                    params['BitDepth']=int(element.attrib['value'])
                    
                if element.attrib['key']=='dwellTime':
                    params['dwellTime']=float(element.attrib['value'])
                    
                if element.attrib['key']=='framePeriod':
                    params['framePeriod']=float(element.attrib['value'] )
                    
                if element.attrib['key']=='laserPower':
                    params['ImagingLaserPower']=float(element[0].attrib['value'])
                    
                if element.attrib['key']=='laserPower':
                    params['UncagingLaserPower']=float(element[1].attrib['value'] )
                    
                if element.attrib['key']=='linesPerFrame':
                    params['LinesPerFrame']=int(element.attrib['value'])
                    
                if element.attrib['key']=='micronsPerPixel':
                    params['MicronsPerPixelX']=float(element[0].attrib['value'])
                    
                if element.attrib['key']=='micronsPerPixel':
                    params['MicronsPerPixelY']=float(element[1].attrib['value'])
                    
                if element.attrib['key']=='objectiveLens':
                    params['Objective']=element.attrib['value']
                    
                if element.attrib['key']=='objectiveLensMag':
                    params['ObjectiveMag']=int(element.attrib['value'])
                    
                if element.attrib['key']=='objectiveLensNA':
                    params['ObjectiveNA']=float(element.attrib['value'])
                    
                if element.attrib['key']=='opticalZoom':
                    params['OpticalZoom']=float(element.attrib['value'] )
                    
                if element.attrib['key']=='pixelsPerLine':
                    params['PixelsPerLine']=int(element.attrib['value'])
                    
                if element.attrib['key']=='pmtGain':
                    params['PMTGainRed']=float(element[0].attrib['value'] )
                    
                if element.attrib['key']=='pmtGain':
                    params['PMTGainGreen']=float(element[1].attrib['value'])
                    
                if element.attrib['key']=='positionCurrent':
                    params['PositionX']=float(element[0][0].attrib['value'])
                    
                if element.attrib['key']=='positionCurrent':
                    params['PositionY']=float(element[1][0].attrib['value'])
                    
                if element.attrib['key']=='positionCurrent':
                    params['PositionZphysical']=float(element[2][0].attrib['value'])
                    
                if element.attrib['key']=='positionCurrent':
                    params['PositionZETL']=float(element[2][1].attrib['value'])
                    
                if element.attrib['key']=='rastersPerFrame':
                    params['RasterAveraging']=int(element.attrib['value'])
                    
                if element.attrib['key']=='resonantSamplesPerPixel':
                    params['ResonantSampling']=int(element.attrib['value'] )
                    
                if element.attrib['key']=='scanLinePeriod':
                    params['ScanLinePeriod']=float(element.attrib['value'])
                    
                if element.attrib['key']=='zDevice':
                    params['ZDevice']=int(element.attrib['value'] )
                    
                    
    
                
            video_params={'MultiplanePrompt':seqinfo.attrib['type'],                                  
                          'ParameterSet':seqinfo.find("Frame").attrib['parameterSet'],
                          'RedChannelName':'No Channel',
                          'GreenChannelName':'No Channel',           
                          'FrameNumber':int(len(list(seqinfo.findall('Frame')))),
                          'PlaneNumber':'Single',
                          'PlanePositionsOBJ':params['PositionZphysical'],
                          'PlanePositionsETL':params['PositionZETL'],           
                          'Planepowers':params['ImagingLaserPower'],
                          
                          }
            
     
            # MultiPlane=0
            SingleChannel=0
           
            if len(list(root))>3 and video_params['MultiplanePrompt']=="TSeries ZSeries Element":
                FirstVolumeMetadat=root[2]
                del video_params['FrameNumber']
                video_params['VolumeNumber']=int(len(list(root.findall('Sequence'))))
                # MultiPlane=1
                all_volumes=[]            
                            
                for volume in root.findall('Sequence'):
                    
                    all_planes={}
                    
                    for i, plane in enumerate(volume.findall('Frame')):
                         iplane={}
                         iplane['absoluteTime']=float(plane.attrib['absoluteTime'])
                         iplane['index']=int(plane.attrib['index'])
                         iplane['relativeTime']=float(plane.attrib['relativeTime'])
                         
                         iplane['LastGoodFrame']=int(plane.find('ExtraParameters').attrib['lastGoodFrame'])
                         ExtraMetadata=plane.find('PVStateShard') 
                         iplane['ImagingSlider']=params['ImagingLaserPower']
                         iplane['UncagingSlider']=params['UncagingLaserPower']
                         iplane['XAxis']='Default_' + str(params['PositionX'])
                         iplane['YAxis']='Default_' + str(params['PositionY'])
                         iplane['ObjectiveZ']='Default_' +str(video_params['PlanePositionsOBJ'])
                         iplane['ETLZ']='Default_' +str(video_params['PlanePositionsETL'])
                         
                         
                         for element in ExtraMetadata:
                             
                             if 'framePeriod' in element.attrib.values():
                                 iplane['framePeriod']=float(element.attrib['value'])
                                 
                             if 'scanLinePeriod' in element.attrib.values():
                                 iplane['scanLinePeriod']=float(element.attrib['value'])
                                 
                             if 'laserPower' in element.attrib.values():
                                 for element2 in element:
                                     if 'Imaging' in element2.attrib.values():
                                        iplane['ImagingSlider']=float(element2.attrib['value'])
                                     if 'Uncaging' in element2.attrib.values():
                                        iplane['UncagingSlider']=float(element2.attrib['value'])
                                                                   
                             if 'positionCurrent' in element.attrib.values():
                                 for element2 in element:
                                     if 'XAxis' in element2.attrib.values():
                                         iplane['XAxis']=float(element2[0].attrib['value'])
                                     if 'YAxis' in element2.attrib.values():
                                         iplane['YAxis']=float(element2[0].attrib['value'])
                                     if 'ZAxis' in element2.attrib.values():
                                         iplane['ObjectiveZ']=float(element2[0].attrib['value'])
                                         iplane['ETLZ']=float(element2[1].attrib['value'])
                                                       
                         if i==0:
                            all_planes['TopPlane']=iplane
                         if i==1:
                            all_planes['MediumPlane']=iplane
        
                         if i==2:
                            all_planes['BottomPlane']=iplane
                            
                    all_volumes.append(all_planes)   
                video_params['FullAcquisitionTime']=all_volumes[-1]['BottomPlane']['relativeTime']  
                video_params['PlaneNumber']=len(list(FirstVolumeMetadat.findall('Frame')))               
                video_params['PlanePositionsOBJ']=[all_volumes[0]['TopPlane']['ObjectiveZ'], all_volumes[0]['MediumPlane']['ObjectiveZ'], all_volumes[0]['BottomPlane']['ObjectiveZ']]
                video_params['PlanePositionsETL']=[all_volumes[0]['TopPlane']['ETLZ'], all_volumes[0]['MediumPlane']['ETLZ'], all_volumes[0]['BottomPlane']['ETLZ']]
                video_params['Planepowers']= [all_volumes[0]['TopPlane']['ImagingSlider'], all_volumes[0]['MediumPlane']['ImagingSlider'], all_volumes[0]['BottomPlane']['ImagingSlider']]          
                                     
            else:
                SequenceInfo=root.find('Sequence')
                seqinfo=SequenceInfo
                
                all_frames=[]
                for frame in SequenceInfo.findall('Frame'):
                    iframe={}
                    iframe['LastGoodFrame']='Default_0'
                    iframe['scanLinePeriod']='Default_' + str(params['ScanLinePeriod'])
                    iframe['absoluteTime']=float(frame.attrib['absoluteTime'])
                    iframe['index']=int(frame.attrib['index'])
                    iframe['relativeTime']=float(frame.attrib['relativeTime'])
                    if frame.find('ExtraParameters'):     
                        iframe['LastGoodFrame']=int(frame.find('ExtraParameters').attrib['lastGoodFrame'])
                    ExtraMetadata=frame.find('PVStateShard')    
                    iframe['framePeriod']=float(ExtraMetadata[0].attrib['value'])
                    if len(ExtraMetadata)>1:
                        iframe['scanLinePeriod']=float(ExtraMetadata[1].attrib['value'])
                    all_frames.append(iframe)
                video_params['FullAcquisitionTime']=all_frames[-1]['relativeTime']
    
            
            if seqinfo.find('PVStateShard').attrib:
                SingleChannel=1
                SingleColorPMTInfo=seqinfo.find('PVStateShard').find('PVStateValue').findall('IndexedValue')
                SingleColorPMTinfo2=[pmt.attrib for pmt in  SingleColorPMTInfo] 
                if 'Green' in  seqinfo.find('Frame').find('File').attrib.values():
                    ChannelName=seqinfo.find('Frame').find('File').attrib['filename']
                    video_params['GreenChannelName']=ChannelName
            else:
                SingleChannel=0 
                for chan in seqinfo.find('Frame').findall('File'):
                    ChannelName=chan.attrib['filename']
                    if 'Green' in chan.attrib.values() :              
                        video_params['GreenChannelName']=ChannelName
                    elif 'Red' in chan.attrib.values():
                        video_params['RedChannelName']=ChannelName
                
           
            if video_params['MultiplanePrompt']=="TSeries ZSeries Element":
                return [params, video_params, all_volumes]
    
            else:
                return  [params, video_params, all_frames]
            
   
    def add_metadata_manually(self):   
        print('getting manual metadata')
        
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
        

        print('finsihed with manual metadata')
         
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
        #%%
        resampled_laser=sp.signal.resample(high_freq_signals_full[:,1],len(ddd))
        # fig, ax = plt.subplots()  # Create a figure containing a single axes.
        # ax.plot(resampled_laser[:])
        filtered_resampled_laser=sp.signal.medfilt(resampled_laser,5)
        # fig, ax = plt.subplots()  # Create a figure containing a single axes.
        # ax.plot(filtered_resampled_laser[:])
        artifact_idx = np.where(filtered_resampled_laser>1)
        PhotoStimSeriesArtifact={'artifact_idx':artifact_idx, 'stim_table_ms':stim_table_ms,'Processed_Pockels_Signal' :filtered_resampled_laser}
        
        return PhotoStimSeriesArtifact   
    
#%%    
if __name__ == "__main__":
    path=r'F:\Projects\LabNY\Imaging\2021\20210525\Mice\SPIL\FOV_1\OtherAcq\Aq_1\20210525_SPIL_FOV1_3planeredonly1000nm_3reps-000'
    xmlfile='20210525_SPIL_FOV1_3planeredonly1000nm_3reps-000.xml'
    image_sequence_directory_full_path='\\\?\\'+ os.path.join(path,xmlfile)
    voltxmlfile='210615_SPHV_FOV1_10minspont_50024_narrored_920_with-000_Cycle00001_VoltageRecording_001.xml'
    volt_directory_full_path='\\\?\\'+ os.path.join(path,voltxmlfile)
    
    meta =Metadata(aq_metadataPath=image_sequence_directory_full_path)
    zz=meta.imaging_metadata
    zzz  =meta.voltage_recording_metadata               