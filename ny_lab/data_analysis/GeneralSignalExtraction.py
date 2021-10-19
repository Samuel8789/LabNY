# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 08:57:34 2021

@author: sp3660
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import matplotlib
%matplotlib qt
## load full csv signals
temp_folder='\\\\?\\'+ r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000'
# # file_to_proces=r'210521_SPJO_HabituationDay1_940_WideFilter-000_Cycle00001_VoltageRecording_001.csv'
# file_to_proces=r'210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'
# full_path=os.path.join(temp_folder,file_to_proces)
# voltage_signals = pd.read_csv(full_path)

# voltage_signals.columns


# #%% LOCOMOTION
# # locomotion_df=voltage_signals[' Locomotion']
# # locomotion_aray=locomotion_df.to_numpy()
# # rectified_speed_array=np.absolute(np.diff(np.around(locomotion_aray,4)))
# # rectified_speed_array=np.insert(rectified_speed_array,0,0)

# # #%% VISUAL SIGNAL
# # visualstim_df=voltage_signals[' VisStim']
# # visualstim_aray=visualstim_df.to_numpy()

#%% PHOTODIODE
# photodiode_df=voltage_signals[' PhotoDiode']
# photodiode_aray=photodiode_df.to_numpy()

#%% MEMORY RELEASE 
# del voltage_signals
#%% DIFFERENT TIME SCALES
# voltage_rate=1000;
# milisecondscale=np.arange(1,locomotion_df.size+1)
# second_scale=milisecondscale/voltage_rate
# minutes_scale=second_scale/60


#%% PLOTTING
fig, axs = plt.subplots(2)
fig.suptitle('Locomotion')
axs[0].plot(second_scale,rectified_speed_array)
# axs[1].plot(second_scale,visualstim_aray)
# axs[2].plot(second_scale,photodiode_aray)
mplcursors.cursor(axs) # or just mplcursors.cursor()

fig, axs = plt.subplots(2)
fig.suptitle('Visual Stimulation')
# axs[0].plot(second_scale,rectified_speed_array)
axs[0].plot(second_scale,visualstim_aray)
# axs[2].plot(second_scale,photodiode_aray)
mplcursors.cursor(axs) # or

#%% ALLEN A
rounded_vis_stim=np.around(visualstim_aray,1)
df_rounded_vis_stim = np.diff(rounded_vis_stim)
k=np.argwhere(df_rounded_vis_stim<-6.5).flatten()
pp=np.argwhere(df_rounded_vis_stim>5).flatten()
lst_down=np.argwhere(df_rounded_vis_stim<-0.5).flatten()



first_drifitng_set_first=k[0]+1
first_drifitng_set_last=pp[1]+1
second_drifitng_set_first=k[5]+1
second_drifitng_set_last=pp[8]+1
third_drifitng_set_first=k[8]+1
third_drifitng_set_last=lst_down[-1]+2

first_movie_set_first=k[2]+1
first_movie_set_last=pp[4]+1
second_movie_set_first=k[6]+1
second_movie_set_last=pp[9]+1

short_movie_set_first=k[4]+1
short_movie_set_last=pp[7]+1

spont_first=pp[8]+1
spont_last=k[6]-2


first_drifitng_set=rounded_vis_stim[first_drifitng_set_first:first_drifitng_set_last]
second_drifitng_set=rounded_vis_stim[second_drifitng_set_first:second_drifitng_set_last]
third_drifitng_set=rounded_vis_stim[third_drifitng_set_first:third_drifitng_set_last]


first_movie_set=rounded_vis_stim[first_movie_set_first:first_movie_set_last]
second_movie_set=rounded_vis_stim[second_movie_set_first:second_movie_set_last]

short_movie_set=rounded_vis_stim[short_movie_set_first:short_movie_set_last]

spont=rounded_vis_stim[spont_first:spont_last]



fig, axs = plt.subplots(8)
fig.suptitle('VisStiss')
axs[0].plot(first_drifitng_set)
axs[1].plot(second_drifitng_set)
axs[2].plot(third_drifitng_set)
axs[3].plot(first_movie_set)
axs[4].plot(second_movie_set)
axs[5].plot(short_movie_set)
axs[6].plot(spont)
axs[7].plot(rounded_vis_stim)
mplcursors.cursor(axs) # or just mplcursors.cursor()
#%% concatenated just to make sure
%matplotlib qt
%matplotlib inline
#%%
combined_gratings=np.concatenate((first_drifitng_set, second_drifitng_set, third_drifitng_set))
df_combined_gratings = np.diff(combined_gratings)

orientations=np.round(np.linspace(0.1,4,40),1)
blanksweepvoltage=4.5
repetitions=15
idxx=8
test_down=np.argwhere(df_combined_gratings==orientations[idxx]).flatten()
test_upn=np.argwhere(np.logical_and(df_combined_gratings>0.05, df_combined_gratings<4.1)).flatten()

# range_to_plot=np.arange(k[8]+1,pp[10]+1)
# plt.plot(second_scale[range_to_plot],rounded_vis_stim[range_to_plot])
range_to_plot=np.arange(k[8]+1,lst_down[-1]+2)
fig, axs = plt.subplots(2)
axs[0].plot(combined_gratings)
axs[0].plot(test_down,combined_gratings[test_down+1],'o')
axs[1].plot(df_combined_gratings)
axs[1].plot(test_down,df_combined_gratings[test_down],'o')

mplcursors.cursor(axs) # or just mplcursors.cursor()


totaltrials=orientations.shape[0]*repetitions
period_tuning=combined_gratings
orientations_boolean_starts=np.zeros((orientations.shape[0],period_tuning.size))
orientations_boolean_end=np.zeros((orientations.shape[0],period_tuning.size))
df_period_tuning = np.diff(period_tuning)
df_period_tuning_rounded=np.around(df_period_tuning,1)
#correcting slower voltage transitions
for i in range(2,df_period_tuning_rounded.size-2):
    if df_period_tuning_rounded[i+1]!=0 and df_period_tuning_rounded[i]!=0:
       df_period_tuning_rounded[i+1]= df_period_tuning_rounded[i+1]+ df_period_tuning_rounded[i];
       df_period_tuning_rounded[i]=0;


#getting bolean indexing vectors
for i in range(0,df_period_tuning_rounded.size):
    for j, voltage in enumerate(orientations):
        if df_period_tuning_rounded[i] == voltage:
            orientations_boolean_starts[j,i+1] = 1;
            
for i in range(0,df_period_tuning_rounded.size):
    for j, voltage in enumerate(orientations):
        if df_period_tuning_rounded[i] == -voltage:
            orientations_boolean_end[j,i+1] = 1;

# get indexes
stim_on_indexes_tuning_period=np.zeros((orientations.size, repetitions)).astype('int64')
for row in range(0,orientations.size):
    stim_on_indexes_tuning_period[row,:]=np.argwhere(orientations_boolean_starts[row,:]).flatten()
    
    
    
stim_off_indexes_tuning_period=np.zeros((orientations.size,repetitions)).astype('int64')
for row in range(0,orientations.size):
    if np.argwhere(orientations_boolean_end[row,:]).flatten().size==9:
        tosave= np.argwhere(orientations_boolean_end[row,:]).flatten() 
        tosave=np.append(tosave,period_tuning.size)
        stim_off_indexes_tuning_period[row,:]=tosave-1
    else:
        stim_off_indexes_tuning_period[row,:]=np.argwhere(orientations_boolean_end[row,:]).flatten()-1
#%%
fig, axs = plt.subplots(1)
axs.plot(combined_gratings)
axs.plot(stim_on_indexes_tuning_period,combined_gratings[stim_on_indexes_tuning_period],'o')
axs.plot(stim_off_indexes_tuning_period,combined_gratings[stim_off_indexes_tuning_period],'x')



#%% correcting for full video
first_length=first_drifitng_set.shape[0]
second_length=second_drifitng_set.shape[0]
mivies_indexes=[first_drifitng_set_first,first_drifitng_set_last,second_drifitng_set_first,second_drifitng_set_last,third_drifitng_set_first,third_drifitng_set_last]

def correctindex(indx, first_length, second_length, mivies_indexes0, mivies_indexes2, mivies_indexes4,):
    if indx < first_length:
        return indx + mivies_indexes0
    elif np.logical_and(indx<first_length+second_length,indx>first_length) :
        return indx + mivies_indexes2-first_length
    elif indx > second_length:
        return indx + mivies_indexes4-first_length-second_length
    
vfunc = np.vectorize(correctindex)

tuning_stim_on_index_full_recording = vfunc(stim_on_indexes_tuning_period, first_length, second_length,  mivies_indexes[0], mivies_indexes[2], mivies_indexes[4])
tuning_stim_off_index_full_recording = vfunc(stim_off_indexes_tuning_period, first_length, second_length,  mivies_indexes[0], mivies_indexes[2], mivies_indexes[4])



fig, axs = plt.subplots(1)
axs.plot(rounded_vis_stim)
axs.plot(tuning_stim_on_index_full_recording,rounded_vis_stim[tuning_stim_on_index_full_recording],'o')
axs.plot(tuning_stim_off_index_full_recording,rounded_vis_stim[tuning_stim_off_index_full_recording],'x')

file_path=dataset_path+"\\diftingindexon.mat"
scipy.io.savemat(file_path, {'diftingindexon': tuning_stim_on_index_full_recording})
file_path=dataset_path+"\\diftingindexoff.mat"
scipy.io.savemat(file_path, {'diftingindexoff': tuning_stim_off_index_full_recording})


movie_rate=16.10383676648614 #hz
milisecond_period=1000/movie_rate
movie_frames_tuning_on=tuning_stim_on_index_full_recording/milisecond_period
movie_frames_tuning_off=tuning_stim_off_index_full_recording/milisecond_period


drifting_1_frame_index_start=np.round(first_drifitng_set_first/milisecond_period).astype('uint16')
drifting_1_frame_index_end=np.round(first_drifitng_set_last/milisecond_period).astype('uint16')
drifting_2_frame_index_start=np.round(second_drifitng_set_first/milisecond_period).astype('uint16')
drifting_2_frame_index_end=np.round(second_drifitng_set_last/milisecond_period).astype('uint16')
drifting_3_frame_index_start=np.round(third_drifitng_set_first/milisecond_period).astype('uint16')
drifting_3_frame_index_end=np.round(third_drifitng_set_last/milisecond_period).astype('uint16')



#%%
fig, axs = plt.subplots(3)
fig.suptitle('VoltageS Signals')
axs[0].plot(second_scale,rounded_vis_stim)
axs[1].plot(second_scale[0:-1],df_rounded_vis_stim)
# axs[2].plot(second_scale,photodiode_aray)
mplcursors.cursor(axs) # or just mplcursors.cursor()


# figure()
# plot(vistim)
# indexup=find(vistim>6.5);
# tocorrectup=find(diff(indexup)<20);
# indexup(tocorrectup)=[];
#%% HABITUATION GOOD
#Global
prestim = 29*voltage_rate
poststim = 100*voltage_rate
nsession = 5;
rounded_vis_stim=np.around(visualstim_aray,1)
df_rounded_vis_stim = np.diff(rounded_vis_stim)
k=np.argwhere(df_rounded_vis_stim<0).flatten()

first_spont_indx=k[0];
last_spont_indx=k[1];
first_habituation_indx=last_spont_indx+1;
last_habituation_indx=k[6];
first_grating_indx=last_habituation_indx+1;
last_grating_indx=k[-1];

period_spontaneous=rounded_vis_stim[first_spont_indx:last_spont_indx+1]
period_habituation=rounded_vis_stim[first_habituation_indx:last_habituation_indx+1]
period_tuning=rounded_vis_stim[first_grating_indx:last_grating_indx+1]


#%%Habituation period extrct indexes
for i in range(0,period_habituation.size):
    if period_habituation[i]<0.2:
        period_habituation[i]=0
    else:
        period_habituation[i]=1
df_period_habituation = np.diff(period_habituation)
df_period_habituation_rounded=np.around(df_period_habituation)
stim_on_index_habituation_period = np.argwhere(df_period_habituation_rounded>0.9).flatten()
stim_on_index_full_recording = stim_on_index_habituation_period + (first_habituation_indx)
#proces locomotion form indexes
habituation_periods_locomotion=np.zeros((5,len(list(range(stim_on_index_full_recording[0]-prestim,stim_on_index_full_recording[0]+poststim)))))
for i in range(0,5):
    habituation_periods_locomotion[i,:]=rectified_speed_array[stim_on_index_full_recording[i]-prestim:stim_on_index_full_recording[i]+poststim]
# normalization
mean_prestim = np.mean(habituation_periods_locomotion[:,0:prestim],1)
norm_locomotion_speed=np.divide(habituation_periods_locomotion.T,mean_prestim).T
# results
full_trial_averaged_normalized_locomotion=np.mean(norm_locomotion_speed,0);
stim_on_mean_normalized_locomotion=np.mean(norm_locomotion_speed[:,prestim:],1)
stim_on_trial_averaged_normalized_locomotion = np.mean(stim_on_mean_normalized_locomotion);
#%% tuning period extract index
orientations=np.linspace(0.5,4,8)
repetitions=10
orientations_boolean_starts=np.zeros((8,period_tuning.size))
orientations_boolean_end=np.zeros((8,period_tuning.size))
df_period_tuning = np.diff(period_tuning)
df_period_tuning_rounded=np.around(df_period_tuning,1)

#correcting slower voltage transitions
for i in range(2,df_period_tuning_rounded.size-2):
    if df_period_tuning_rounded[i+1]!=0 and df_period_tuning_rounded[i]!=0:
       df_period_tuning_rounded[i+1]= df_period_tuning_rounded[i+1]+ df_period_tuning_rounded[i];
       df_period_tuning_rounded[i]=0;


#getting bolean indexing vecors
for i in range(0,df_period_tuning_rounded.size):
    for j, voltage in enumerate(orientations):
        if df_period_tuning_rounded[i] == voltage:
            orientations_boolean_starts[j,i+1] = 1;
            
for i in range(0,df_period_tuning_rounded.size):
    for j, voltage in enumerate(orientations):
        if df_period_tuning_rounded[i] == -voltage:
            orientations_boolean_end[j,i+1] = 1;

# get indexes

stim_on_indexes_tuning_period=np.zeros((orientations.size,repetitions)).astype('int64')
for row in range(0,orientations.size):
    stim_on_indexes_tuning_period[row,:]=np.argwhere(orientations_boolean_starts[row,:]).flatten()
    
stim_off_indexes_tuning_period=np.zeros((orientations.size,repetitions)).astype('int64')
for row in range(0,orientations.size):
    if np.argwhere(orientations_boolean_end[row,:]).flatten().size==9:
        tosave= np.argwhere(orientations_boolean_end[row,:]).flatten() 
        tosave=np.append(tosave,period_tuning.size)
        stim_off_indexes_tuning_period[row,:]=tosave-1
    else:
        stim_off_indexes_tuning_period[row,:]=np.argwhere(orientations_boolean_end[row,:]).flatten()-1


tuning_stim_on_index_full_recording = stim_on_indexes_tuning_period + (first_grating_indx)
tuning_stim_off_index_full_recording = stim_off_indexes_tuning_period + (first_grating_indx)


movie_rate=16.10383676648614 #hz
milisecond_period=1000/movie_rate
movie_frames_tuning_on=tuning_stim_on_index_full_recording/milisecond_period
movie_frames_tuning_off=tuning_stim_off_index_full_recording/milisecond_period



#%% plotting multiple

fig, ax = plt.subplots(nrows=8,sharey=True)

# fig.set_title('Snapping cursor')
for i in range(0,8):
    line, = ax[i].plot(orientations_boolean_starts[i])
     
snap_cursor = SnappingCursor(ax[0], line)

fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
mplcursors.cursor(line) # or just mplcursors.cursor()

plt.show()






#%% plotting single

fig, ax = plt.subplots(nrows=1,sharey=True)
     
ax.plot(rounded_vis_stim)
ax.plot(tuning_stim_off_index_full_recording[0,:]-1,rounded_vis_stim[tuning_stim_off_index_full_recording[0,:]-1],'o')
snap_cursor = SnappingCursor(ax, line)

fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
mplcursors.cursor(line) # or just mplcursors.cursor()

plt.show()




















































