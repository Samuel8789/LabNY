# # -*- coding: utf-8 -*-
# """
# Created on Sat Jul 31 09:59:15 2021

# @author: sp3660
# """
# import sys
# sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/AllFunctions')
# sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/MainClasses')
# from MetadataClass import Metadata
# import numpy as np
# import os

# # imaging_path=r'F:\Projects\LabNY\Imaging\2021\20210521\Mice\SPJO\FOV_1\Aq_1\210521_SPJO_HabituationDay1_940_WideFilter-000\210521_SPJO_HabituationDay1_940_WideFilter-000.xml'
# # metadata=Metadata(aq_metadataPath=imaging_path)

# #%%


# #%%
# for imgID in range(7,15):
#     params1=(imgID,)
#     query="""
#             SELECT ImagingFullFilePath, ImagingFilename
#             FROM Imaging_table
#             WHERE ID=?
#         """
#     info=MouseDat.arbitrary_query_to_df(query,params1)

#     imaging_path=os.path.join(info.ImagingFullFilePath[0],info.ImagingFilename[0])+'.xml'

#     metadata=Metadata(aq_metadataPath=imaging_path)
       
#     PowerSetting=np.nan
#     Objective=np.nan
#     PMT1GainRed=np.nan
#     PMT2GainGreen=np.nan
#     FrameAveraging=np.nan
#     ObjectivePositions=np.nan
#     ETLPositions=np.nan
#     PlaneNumber=np.nan
#     TotalVolumes=np.nan
#     IsETLStack=np.nan
#     IsObjectiveStack=np.nan
#     InterFramePeriod=np.nan
#     FinalVolumePeriod=np.nan
#     FinalFrequency=np.nan
#     TotalFrames=np.nan
#     FOVNumber=np.nan
#     ExcitationWavelength=np.nan
#     CoherentPower=np.nan
#     CalculatedPower=np.nan
#     Comments=np.nan
#     IsChannel1Red=np.nan
#     IsChannel2Green=np.nan
#     IsGalvo=np.nan
#     IsResonant=np.nan   
#     Resolution=np.nan
#     DwellTime=np.nan
#     Multisampling=np.nan
#     BitDepth=np.nan
#     LinePeriod=np.nan
#     FramePeriod=np.nan
#     FullAcquisitionTime=np.nan    
#     RedFilter=np.nan
#     GreenFilter=np.nan
#     DichroicBeamsplitter=np.nan
#     IsBlockingDichroic=np.nan
    
#     PowerSetting=metadata.imaging_metadata[1]['Planepowers']    
#     Objective=metadata.imaging_metadata[0]['Objective']       
#     PMT1GainRed=metadata.imaging_metadata[0]['PMTGainRed']
#     PMT2GainGreen=metadata.imaging_metadata[0]['PMTGainGreen']
#     FrameAveraging=metadata.imaging_metadata[0]['RasterAveraging']   
#     ObjectivePositions=metadata.imaging_metadata[1]['PlanePositionsOBJ']
#     ETLPositions=metadata.imaging_metadata[1]['PlanePositionsETL'] 
    
        
        
#     if metadata.imaging_metadata[1]['PlaneNumber']=='Single':
#             IsETLStack=0
#             IsObjectiveStack=0
#             PlaneNumber=1
#             TotalFrames=metadata.imaging_metadata[1]['FrameNumber']
#             InterFramePeriod=metadata.imaging_metadata[0]['framePeriod']*FrameAveraging
#             FinalVolumePeriod=InterFramePeriod
#             FinalFrequency=1/InterFramePeriod
#             TotalVolumes=TotalFrames
#     else:
#         TotalVolumes=metadata.imaging_metadata[1]['VolumeNumber']
#         IsETLStack=0
#         IsObjectiveStack=0
#         PlaneNumber=metadata.imaging_metadata[1]['PlaneNumber']
       
#         InterFramePeriod=metadata.imaging_metadata[0]['framePeriod']
#         FinalVolumePeriod=metadata.imaging_metadata[2][0]['TopPlane']['framePeriod']*PlaneNumber
#         FinalFrequency=1/FinalVolumePeriod
#         TotalFrames=TotalVolumes*PlaneNumber
#         PowerSetting=str(PowerSetting)
#         correctedObjectivePositions=[float(i[8:]) if isinstance(i, str) else i for i in ObjectivePositions]
#         correctedETLPositions=[float(i[8:]) if isinstance(i, str) else i for i in ETLPositions]
#         if not all(element == correctedObjectivePositions[0] for element in correctedObjectivePositions):
#             IsObjectiveStack=1
#         if not all(element == correctedETLPositions[0] for element in correctedETLPositions):
#             IsETLStack=1
    
#     FOVNumber=np.nan
#     if 'FOV_' in imaging_path:            
#          FOVNumber=imaging_path[ imaging_path.index('FOV_')+4]
         
    
    
    	
#     RedFilter=1
#     GreenFilter=1
#     DichroicBeamsplitter=1
#     filtervalues=[RedFilter,GreenFilter,DichroicBeamsplitter]
#     filtercodes=filtervalues
    
#     IsBlockingDichroic=1   
    
        
#     ExcitationWavelength=920
#     CalculatedPower=np.nan
#     Comments=''
       
    
#     IsChannel1Red=0
#     IsChannel2Green=0
    
#     if not metadata.imaging_metadata[1]['RedChannelName']=='No Channel':
#         IsChannel1Red=1
#     if not metadata.imaging_metadata[1]['GreenChannelName']=='No Channel':
#         IsChannel2Green=1
#     IsGalvo=1
#     IsResonant=0
#     if 'Resonant' in  metadata.imaging_metadata[0]['ScanMode']:
#          IsResonant=1
#          IsGalvo=0
    
#     Resolution=str(metadata.imaging_metadata[0]['LinesPerFrame'])+'x'+ str(metadata.imaging_metadata[0]['PixelsPerLine'])
#     DwellTime=metadata.imaging_metadata[0]['dwellTime']
#     Multisampling=metadata.imaging_metadata[0]['ResonantSampling']
#     BitDepth=metadata.imaging_metadata[0]['BitDepth']
        
#     LinePeriod=metadata.imaging_metadata[0]['ScanLinePeriod']
#     FramePeriod=metadata.imaging_metadata[0]['framePeriod']
#     FullAcquisitionTime=metadata.imaging_metadata[1]['FullAcquisitionTime']
    
    
    
#     IsVoltageRecording=0
#     VoltageRecordingChannels=np.nan
#     VoltageRecordingFrequency=np.nan
    
        
#     query_add_imaging="""
#             UPDATE Imaging_table
#             SET
#                 RedFilter=?,
#                 GreenFilter=?,
#                 DichroicBeamsplitter=?,
#                 IsBlockingDichroic=?,
#                 FOVNumber=?,
#                 IsETLStack=?,
#                 IsObjectiveStack=?,
#                 PlaneNumber=?,
#                 Objective=?,
#                 ObjectivePositions=?,
#                 ETLPositions=?,
#                 PMT1GainRed=?,
#                 PMT2GainGreen=?,
#                 IsChannel1Red=?,
#                 IsChannel2Green=?,
#                 ExcitationWavelength=?,
#                 CoherentPower=?,
#                 PowerSetting=?,
#                 CalculatedPower=?,
#                 IsGalvo=?,
#                 IsResonant=?,
#                 Resolution=?,
#                 DwellTime=?,
#                 Multisampling=?,
#                 BitDepth=?,
#                 FrameAveraging=?,
#                 LinePeriod=?,
#                 FramePeriod=?,
#                 InterFramePeriod=?,
#                 FinalVolumePeriod=?,
#                 FinalFrequency=?,
#                 FullAcquisitionTime=?,
#                 TotalFrames=?,
#                 TotalVolumes=?,
#                 IsVoltageRecording=?,
#                 VoltageRecordingChannels=?,
#                 VoltageRecordingFrequency=?,
#                 Comments=?
                
#          WHERE ID=?
#             """    
#     params10=( int(filtercodes[0]),
#             int(filtercodes[1]),
#             int(filtercodes[2]),
#             IsBlockingDichroic,
#             FOVNumber,
#             IsETLStack,
#             IsObjectiveStack,
#             PlaneNumber,
#             Objective,
#             str(ObjectivePositions),
#             str(ETLPositions),
#             PMT1GainRed,
#             PMT2GainGreen,
#             IsChannel1Red,
#             IsChannel2Green,
#             ExcitationWavelength,
#             CoherentPower,
#             PowerSetting,
#             CalculatedPower,
#             IsGalvo,
#             IsResonant,
#             Resolution,
#             DwellTime,
#             Multisampling,
#             BitDepth,
#             FrameAveraging,
#             LinePeriod,
#             FramePeriod,
#             InterFramePeriod,
#             FinalVolumePeriod,
#             FinalFrequency,
#             FullAcquisitionTime,
#             TotalFrames,
#             TotalVolumes,
#             IsVoltageRecording,
#             VoltageRecordingChannels,
#             VoltageRecordingFrequency,
#             Comments,
#             imgID)
  
#     MouseDat.arbitrary_updating_record(query_add_imaging, params10, commit=True )   
