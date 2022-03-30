# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:04:49 2021

@author: sp3660
"""

import sys
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/AllFunctions')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/ProcessingScripts')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/MainClasses')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/MainClasses/Visual Stimulation/Allen_Vis_Stim/WarpedVisualStim-master')
import os
import WarpedVisualStim.DisplayLogAnalysis as dla
import NeuroAnalysisTools.NwbTools as nt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import scipy as sp

#%% load data
log_path='\\\\?\\'+r'F:\Imaging\2021\20210408\SPIC\FOV1\SPIC_FOV1_FilterWide_WL940_ChGreen-000\VisStim\210408123114-CombinedStimuli-MSPIC_FOV1_000-SAM-Grating + spont intercalated-notTriggered-complete.pkl'
csv_path='\\\\?\\'+r'F:\Imaging\2021\20210408\SPIC\FOV1\SPIC_FOV1_FilterWide_WL940_ChGreen-000\SPIC_FOV1_FilterWide_WL940_ChGreen-000_Cycle00001_VoltageRecording_001.csv'


log_folder, log_fn = os.path.split(log_path)
stim_log = dla.DisplayLogAnalyzer(log_path)


log_dict=stim_log.log_dict

voltage=pd.read_csv(csv_path)
for col in voltage.columns:
    print(col)
photodiode=voltage[' Photodiode']
phlist=photodiode.tolist()
#%%

stim_info=log_dict['stimulation']
multiple_stim_sequence=stim_info['stimuli_sequence']
frames_unique=stim_info['frames_unique']
index_to_display=stim_info['index_to_display']
individual_logs=stim_info['individual_logs']

presentation_stm_info=log_dict['presentation']['displayed_frames']
indexes_displayed=[ frames_unique.index(frame) for frame in presentation_stm_info   ]

test=individual_logs['000_UniformContrast']
#%%

fig, ax = plt.subplots()

ys2= indexes_displayed
ys = index_to_display

threshold = 2797

ax.axhline(y=threshold, color='r', linestyle=':')
ax.plot(ys)

greater_than_threshold = [i for i, val in enumerate(ys) if val>threshold]
diffff=np.diff(greater_than_threshold)

ax.plot(greater_than_threshold,  [ys[i] for i in greater_than_threshold], 
        linestyle='none', color='r', marker='o')

plt.show()
#%%
fig, ax = plt.subplots()
ax.axhline(y=threshold, color='r', linestyle=':')
ax.plot(ys2)

greater_than_threshold = [i for i, val in enumerate(ys) if val>threshold]
diffff=np.diff(greater_than_threshold)

ax.plot(greater_than_threshold,  [ys[i] for i in greater_than_threshold], 
        linestyle='none', color='r', marker='o')

plt.show()
#%%


#%%
z=0
j=1000
downsampled[336:339]
339-336
(339-336)*volume_period
20941-20924
phlist[20924:20941]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(np.diff(periods))
# plt.ylim([0.02, 0.2])
plt.show()
for i in range(int(len(downsampled)/j)):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(downsampled[i*j+1:j*(i+1)])
    plt.ylim([0.02, 0.18])
    plt.show()
#%%
dataset_metadata[3][i][3]
periods=[ float(dataset_metadata[3][i][3]) for i,j in enumerate(dataset_metadata[3])]

#%%


dataset_metadata=pp
volateg_frew=1000
framePeriod=float(dataset_metadata[0]['framePeriod'])
etl_frame_period=float(dataset_metadata[1]['FramePeriod'])
rastersPerFrame=int(dataset_metadata[0]['RasterAveraging'])
plane_period=float(framePeriod*rastersPerFrame)
number_planes=dataset_metadata[1]['Plane Number']
if number_planes=='Single':
    number_planes=1
    
volume_period=number_planes*etl_frame_period
  
stim_rec=60
frame_numer=40260
downsampled=np.interp(np.arange(0, len(phlist), volateg_frew/(1/volume_period)), np.arange(0, len(phlist)), phlist)



fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(phlist[20920:20975])
plt.ylim([0.02, 0.18])
plt.show()


fig, ax = plt.subplots()
ax.plot(ys2[:12180])

plt.show()

#%%
fig, ax = plt.subplots()
ax.plot(test['index_to_display'])

plt.show()
#%%%

"""
to do 
    get full stimuls sequence
    get duration of each stiml

"""


print(multiple_stim_sequence)



stim_durations={}
for stim in multiple_stim_sequence:
    if 'duration' in  individual_logs[stim].keys():
        stim_durations[stim]=[individual_logs[stim]['duration']+ individual_logs[stim]['postgap_dur'] +individual_logs[stim]['pregap_dur'],
                              (individual_logs[stim]['duration']+ individual_logs[stim]['postgap_dur'] +individual_logs[stim]['pregap_dur'])/(1/60)]
    else:
        if not individual_logs[stim]['is_blank_block']:
            stim_present_numer=len(individual_logs[stim]['tf_list'])*len(individual_logs[stim]['sf_list'])* len(individual_logs[stim]['dire_list']) 
            times_iteration=individual_logs[stim]['iteration']*(stim_present_numer)
 
            stimonly_present_time=times_iteration*individual_logs[stim]['block_dur']        
            
            inter_stim_presentations=times_iteration-1
            inter_stim_duration=  inter_stim_presentations*individual_logs[stim]['midgap_dur']
              
            
            inter_exp=individual_logs[stim]['postgap_dur']+individual_logs[stim]['pregap_dur']
            inter_exp_frames=round(inter_exp/(1/60))
            stim_durations[stim]=[stimonly_present_time+inter_exp+inter_stim_duration,(stimonly_present_time+inter_exp+inter_stim_duration)/(1/60)]

                                  
        elif individual_logs[stim]['is_blank_block']:  
            
            stim_present_numer=len(individual_logs[stim]['tf_list'])*len(individual_logs[stim]['sf_list'])* len(individual_logs[stim]['dire_list']) 
            times_iteration=individual_logs[stim]['iteration']*(stim_present_numer+1)
 
            stimonly_present_time=times_iteration*individual_logs[stim]['block_dur']        
            
            inter_stim_presentations=times_iteration-1
            inter_stim_duration=  inter_stim_presentations*individual_logs[stim]['midgap_dur']
              
            
            inter_exp=individual_logs[stim]['postgap_dur']+individual_logs[stim]['pregap_dur']
            inter_exp_frames=round(inter_exp/(1/60))
            stim_durations[stim]=[stimonly_present_time+inter_exp+inter_stim_duration,(stimonly_present_time+inter_exp+inter_stim_duration)/(1/60)]





#%%
for i in range(len(multiple_stim_sequence)):
    if i==0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ys2[int(i*stim_durations[multiple_stim_sequence[i]][1]):int(stim_durations[multiple_stim_sequence[i]][1])])
        print(int(i*stim_durations[multiple_stim_sequence[i]][1]),int(stim_durations[multiple_stim_sequence[i]][1]))
        plt.show()
        added=int(stim_durations[multiple_stim_sequence[i]][1])
        
    else:    
        fig, ax = plt.subplots(figsize=(12, 6))       
        ax.plot(ys2[added+1: int(stim_durations[multiple_stim_sequence[i]][1]+added)])  
        print(added+1, int(stim_durations[multiple_stim_sequence[i]][1]+added))

        added=added+int(stim_durations[multiple_stim_sequence[i]][1])
        plt.show()
#%%
testin=stim_info['individual_logs']['003_DriftingGratingCircle']['index_to_display'] 
stimonly= testin[90:len(testin)-90]   
stimonly.count(0)

difference=np.diff(stimonly)
np.unique(difference)
"""
-59 -29,  -14, -6,   -3,
"""

#%%
plotrange=difference
fig, ax = plt.subplots(figsize=(12, 6)) 

# ax.plot(testin[:90])  
# ax.plot(testin[len(testin)-90:])  
threshold=3
ax.axhline(y=threshold, color='r', linestyle=':')
ax.plot(plotrange)

greater_than_threshold = [i for i, val in enumerate(plotrange) if val==threshold]

ax.plot(greater_than_threshold,  [plotrange[i] for i in greater_than_threshold], 
        linestyle='none', color='r', marker='o')
plt.show()

#%%

presentations_sum=0
modif=stimonly.copy()
modif.insert(0,0)
firstplotrange=np.diff(modif)

stim_present_numer=len(individual_logs[stim]['tf_list'])*len(individual_logs[stim]['sf_list'])* len(individual_logs[stim]['dire_list']) 
times_iteration=individual_logs[stim]['iteration']*(stim_present_numer+1)
theoretical_presentations_sum=times_iteration

last_value=difference[len(plotrange)-120]

frame_sum=0
summary_list={}



summary_list={}
for val in np.unique(difference):
    if val>1:
        if val!=last_value and val!=firstplotrange[0]:
            threshold=val
            greater_than_threshold = [j for j, val in enumerate(plotrange) if val==threshold]
    
            summary_list[str(val)]=[[len(list(range(i+1,i+121))),i, stimonly[i],i+1,stimonly[i+1],i+120,stimonly[i+120],i+121,stimonly[i+121],list(range(i+1,i+121))] for i in greater_than_threshold] 
            presentations_sum=presentations_sum+len(summary_list[str(val)])
            for i in summary_list[str(val)]:
                frame_sum=frame_sum+i[0]
                if not (i[2]==0 and i[8]==0): 
                    print('bad', val, i)
        
        elif val==last_value:
            threshold=val
            greater_than_threshold = [i for i, val in enumerate(plotrange) if val==threshold]
            summary_list[str(val)]=[]
            for z in greater_than_threshold:
                if z!=36718:
                 summary_list[str(val)].append([len(list(range(z+1,z+121))),z, stimonly[z],z+1,stimonly[z+1],z+120,stimonly[z+120],z+120,stimonly[z+120],list(range(z+1,z+121))] )

                else:
                 summary_list[str(val)].append([len(list(range(z+1,z+121))),z, stimonly[z],z+1,stimonly[z+1],z+120,stimonly[z+120],z+120,stimonly[z+120],list(range(z+1,z+121))] )
            presentations_sum=presentations_sum+len(summary_list[str(val)])     
            for i in summary_list[str(val)]:
                frame_sum=frame_sum+i[0]

                    
        elif val==firstplotrange[0]:
            threshold=val
            greater_than_threshold = [j for j, val in enumerate(firstplotrange) if val==threshold]
            for i in greater_than_threshold:
                if i==0:
                    summary_list[str(val)]=[[len(list(range(i,i+120))),i, stimonly[i],i,stimonly[i],i+119,stimonly[i+119],i+120,stimonly[i+120],list(range(i,i+120))] ]
                else:
                    i=i-1
                    summary_list[str(val)].append([len(list(range(i+1,i+121))),i, stimonly[i],i+1,stimonly[i+1],i+120,stimonly[i+120],i+121,stimonly[i+121],list(range(i+1,i+121))])
            presentations_sum=presentations_sum+len(summary_list[str(val)])
            for i in summary_list[str(val)]:
                frame_sum=frame_sum+i[0]

                    
                    
                    
threshold=3
greater_than_threshold = [i for i, val in enumerate(stimonly) if val==threshold]
del greater_than_threshold[1::2]

summary_list[str(1)]=[[len(list(range(i-2,i+118))),i-3, stimonly[i-3],i-2,stimonly[i-2],i+117,stimonly[i+117],i+118,stimonly[i+118],list(range(i-2,i+118))] for i in greater_than_threshold] 
presentations_sum=presentations_sum+len(summary_list[str(1)])
for i in summary_list[str(1)]:
                frame_sum=frame_sum+i[0]
                if not (i[2]==0 and i[8]==0): 
                    print('bad', 1, i)    
    

#%%
interstim_list={}
interstim_frame_sum=0
interstim_presentations=0
for i,val in enumerate(np.unique(difference)):
    if val<-59:
        threshold=val
        greater_than_threshold = [i for i, val in enumerate(plotrange) if val==threshold]

        interstim_list[str(val)]=[[len(list(range(i+1,i+61))),i, stimonly[i],i+1,stimonly[i+1],i+60,stimonly[i+60],i+61,stimonly[i+61],list(range(i+1,i+61))] for i in greater_than_threshold] 
        interstim_presentations=interstim_presentations+len(interstim_list[str(val)])

        for i in interstim_list[str(val)]:
                interstim_frame_sum=interstim_frame_sum+i[0]



len(stimonly)
interstim_frame_sum+frame_sum
#%%
for key in summary_list.keys():
    dictionary[new_key] = summary_list[key]
    del dictionary[old_key]
    
    dest=[val for val in frames_unique if 'DriftingGrating' in val[0]]
    
#%%    
frames_unique
testin=index_to_display

integer_indexes=[int(idx) for idx in index_to_display]

testin=integer_indexes
stimonly= testin
stimonly.count(4)

difference=np.diff(stimonly)
np.unique(difference)
plotrange=difference
fig, ax = plt.subplots(figsize=(12, 6)) 

# ax.plot(testin[:90])  
# ax.plot(testin[len(testin)-90:])  
# threshold=3
# ax.axhline(y=threshold, color='r', linestyle=':')
ax.plot(difference)

# greater_than_threshold = [i for i, val in enumerate(plotrange) if val==threshold]

# ax.plot(greater_1than_threshold,  [plotrange[i] for i in greater_than_threshold], 
        # linestyle='none', color='r', marker='o')
plt.show()

#%%
add a list of unique stimuli
frames_unique
multiple_stim_sequence


#%%
stim_list={}
for i, stim in enumerate(frames_unique):
    
    if 'UniformContrast' in stim[0]:
        if stim[1]==0:
            stim_list[stim[0]+ ' OFF']=[i,stim]
        if stim[1]==1:
            stim_list[stim[0]+' ON']=[i,stim]
            
    elif 'Drifting' in stim[0]:   
        if stim[1]==0:
            stim_list[stim[0]+ ' OFF']=[i,stim]
        elif stim[2]==1: 
            if stim[6]!=0:
                feq_angle_combination=( stim[4], stim[5])
                stim_list[stim[0]+str(feq_angle_combination)]=[i,stim]
            elif stim[-1]==1:
                stim_list[stim[0]+' BLANK FIRST']=[i,stim]
            elif stim[-1]==0:
                stim_list[stim[0]+' BLANK REST']=[i,stim]
                
combine_stim_list={}            
for key, val in stim_list.items():
    combined_keys=key[4:]
    if combined_keys in combine_stim_list.keys():
        combine_stim_list[combined_keys][0].append(val[0])
    else:
        combine_stim_list[combined_keys]=[[val[0]],val[1]]



     
        
 #%%   
    
    
    












