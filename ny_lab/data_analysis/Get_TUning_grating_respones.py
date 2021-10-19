# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 12:15:17 2021

@author: sp3660
"""
# def Get_Grating_Responses(activity_matrix,grating_start_indexes,grating_end_indexes, pre_frames, post_frames):
    
import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from TestPLot import SnappingCursor
import mplcursors
from numpy import exp, abs, angle
from scipy import stats
import scipy.io

#%%
dataset_path='\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000'   
file_paths=glob.glob(dataset_path+'\**\**.mat', recursive=True)
# plane1=h5py.File(file_paths[0], 'r')
# plane2=h5py.File(file_paths[1], 'r')
# plane3=h5py.File(file_paths[2], 'r')
chand=h5py.File(file_paths[0], 'r')

chandgoodcells=chand["proc"]['idx_components'][:,:]
chandC=chand['est']['C'][:,chandgoodcells]
chanddfdt=chand['proc']['deconv']['smooth_dfdt']['S'][:,chandgoodcells]
chandfoopsi=chand['proc']['deconv']['c_foopsi']['S'][:,chandgoodcells]
file_path=dataset_path+"\\alchandnesC.mat"
scipy.io.savemat(file_path, {'allplanesC': chandC})
file_path=dataset_path+"\\allchanddfdt.mat"
scipy.io.savemat(file_path, {'alchandnedfdt': chanddfdt})

# plane1goodcells=plane1["proc"]['idx_components'][:,:]
# plane2goodcells=plane2["proc"]['idx_components'][:,:]
# plane3goodcells=plane3["proc"]['idx_components'][:,:]
# plane1goodcells.shape[0]


# plane1C=plane1['est']['C'][:,plane1goodcells]
# plane2C=plane2['est']['C'][:,plane2goodcells]
# plane3C=plane3['est']['C'][:,plane3goodcells]
# allplanesC=np.concatenate((plane1C,plane2C,plane3C), axis=1).T
# plt.imshow(allplanesC,vmin=0, vmax=1,aspect='auto',cmap='inferno')
# plt.colorbar()

# file_path=dataset_path+"\\allplanesC.mat"
# scipy.io.savemat(file_path, {'allplanesC': allplanesC})



# plane1dfdt=plane1['proc']['deconv']['smooth_dfdt']['S'][:,plane1goodcells]
# plane2dfdt=plane2['proc']['deconv']['smooth_dfdt']['S'][:,plane2goodcells]
# plane3dfdt=plane3['proc']['deconv']['smooth_dfdt']['S'][:,plane3goodcells]
# allplanesdfdt=np.concatenate((plane1dfdt,plane2dfdt,plane3dfdt), axis=1).T
# plt.imshow(allplanesdfdt,vmin=0, vmax=0.1,aspect='auto',cmap='inferno')
# plt.colorbar()
# file_path=dataset_path+"\\allplanedfdt.mat"
# scipy.io.savemat(file_path, {'allplanedfdt': allplanesdfdt})


# plane1foopsi=plane1['proc']['deconv']['c_foopsi']['S'][:,plane1goodcells]
# plane1foopsigood=np.zeros((plane1goodcells.shape[0],plane1C.shape[0]))
# for i in range(plane1goodcells.shape[0]):
#     plane1foopsigood[i,:]=plane1[plane1foopsi[0,i]][:,:].T

# plane2foopsi=plane2['proc']['deconv']['c_foopsi']['S'][:,plane2goodcells]
# plane2foopsigood=np.zeros((plane2goodcells.shape[0],plane2C.shape[0]))
# for i in range(plane2goodcells.shape[0]):
#     plane2foopsigood[i,:]=plane2[plane2foopsi[0,i]][:,:].T
    
# plane3foopsi=plane3['proc']['deconv']['c_foopsi']['S'][:,plane3goodcells]
# plane3foopsigood=np.zeros((plane3goodcells.shape[0],plane3C.shape[0]))
# for i in range(plane3goodcells.shape[0]):
#     if i!=31:
#         plane3foopsigood[i,:]=plane3[plane3foopsi[0,i]][:,:].T
    
# allplanesfoopsi=np.concatenate((plane1foopsigood,plane2foopsigood,plane3foopsigood))
# plt.imshow(allplanesfoopsi,vmin=0, vmax=0.1,aspect='auto',cmap='inferno')
# plt.colorbar()   
    
# file_path=dataset_path+"\\allplanefoopsi.mat"
# scipy.io.savemat(file_path, {'allplanefoopsi': allplanesfoopsi})
    
# # plane1MCMC=plane1['proc']['deconv']['MCMC']['S'][:,plane1goodcells]
# # plane1MCMCgood=np.zeros((plane1goodcells.shape[0],plane1C.shape[0]))
# # for i in range(plane1goodcells.shape[0]):
# #     plane1MCMCgood[i,:]=plane1[plane1MCMC[0,i]][:,:].T
    
# # plane2MCMC=plane2['proc']['deconv']['MCMC']['S'][:,plane2goodcells]
# # plane2MCMCgood=np.zeros((plane2goodcells.shape[0],plane2C.shape[0]))
# # for i in range(plane2goodcells.shape[0]):
# #     if i not in [8,9,12,23,28,39,40,44,45]:
# #         plane2MCMCgood[i,:]=plane2[plane2MCMC[0,i]][:,:].T
    
# # plane3MCMC=plane3['proc']['deconv']['MCMC']['S'][:,plane3goodcells]
# # plane3MCMCgood=np.zeros((plane3goodcells.shape[0],plane3C.shape[0]))
# # for i in range(plane3goodcells.shape[0]):
# #     plane3MCMCgood[i,:]=plane3[plane3MCMC[0,i]][:,:].T

# # allplanesMCMC=np.concatenate((plane1MCMCgood,plane2MCMCgood,plane2MCMCgood), axis=1).T
# # plt.imshow(allplanesMCMC,vmin=0, vmax=0.1,aspect='auto',cmap='inferno')
# # plt.colorbar()   
    
# allplanesfoopsi=np.concatenate((plane1foopsigood,plane2foopsigood,plane3foopsigood))
# plt.imshow(allplanesfoopsi,vmin=0, vmax=0.1,aspect='auto',cmap='inferno')
# plt.colorbar()   
    
fig, axs = plt.subplots(2)
fig.suptitle('Locomotion')
axs[0].plot(second_scale,rectified_speed_array)
axs[0].set_ylim(0, max(rectified_speed_array))
axs[1].imshow(allplanesfoopsi,vmin=0, vmax=0.1,aspect='auto',cmap='inferno')



#%%

activity_matrix=allplanesfoopsi.T

gratin1_periods_only=activity_matrix[drifting_1_frame_index_start:drifting_1_frame_index_end,:]
gratin2_periods_only=activity_matrix[drifting_2_frame_index_start:drifting_2_frame_index_end,:]
gratin3_periods_only=activity_matrix[drifting_3_frame_index_start:drifting_3_frame_index_end,:]

allplanesfoopsigrating=np.concatenate((gratin1_periods_only,gratin2_periods_only,gratin3_periods_only))
plt.imshow(gratin1_periods_only.T,vmin=0, vmax=0.1,aspect='auto',cmap='inferno')
plt.colorbar()  

file_path=dataset_path+"\\allplanefoopsigrating.mat"
scipy.io.savemat(file_path, {'allplanefoopsigrating': allplanesfoopsigrating})
    

%matplotlib inline
grating_start_indexes=movie_frames_tuning_on
grating_end_indexes=movie_frames_tuning_off
isi_time=1
stim_time=2
movie_rate=16.10383676648614 #hz
milisecond_period=1000/movie_rate

pre_frames=16
post_frames=16




grating_number=grating_start_indexes.shape[0]
repetitions=grating_start_indexes.shape[1]
frame_number=np.arange(grating_start_indexes[0,0]-pre_frames,grating_end_indexes[0,0]+post_frames).size


grating_response_all_cells=np.zeros((repetitions,frame_number,grating_number,activity_matrix.shape[1]));
for cell in range(0,activity_matrix.shape[1]):
    for grat in range(0,grating_number):
        grating_select=grat
        selected_grating_starts=grating_start_indexes[grating_select,:]
        selected_grating_ends=grating_end_indexes[grating_select,:]
        for rept in range(0,repetitions):           
            frame_window=np.arange(selected_grating_starts[rept]-pre_frames,selected_grating_ends[rept]+post_frames).astype('int64')
            grating_response_all_cells[rept,:,grat,cell]=activity_matrix[frame_window,cell]
            
plt.imshow(grating_response_all_cells[:,:,0,0],vmin=0, vmax=0.1,aspect='auto',cmap='inferno')
plt.plot(grating_response_all_cells[0,:,0,:].T)

plt.colorbar()  


            
trial_averaged_activity= np.mean(grating_response_all_cells,0)
plt.imshow(trial_averaged_activity[:,0,:].T,vmin=0, vmax=0.1,aspect='auto',cmap='inferno')
plt.colorbar()  
plt.plot(trial_averaged_activity[:,0,0].T)


mean_response_to_gratings=np.mean(trial_averaged_activity,2)
plt.imshow(mean_response_to_gratings.T,vmin=0, vmax=0.05,aspect='auto',cmap='inferno')
plt.colorbar()  

mean_reponse_per_cell=np.mean(trial_averaged_activity,1)
plt.imshow(mean_reponse_per_cell.T,vmin=0, vmax=0.05,aspect='auto',cmap='inferno')
plt.colorbar()  

#%%
df_fpercentage=np.zeros(grating_response_all_cells.shape);
for cell in range(grating_response_all_cells.shape[3]):
    for grat in range(0,grating_response_all_cells.shape[2]):
        for rept in range(0,grating_response_all_cells.shape[0]):      
            baseline_segment=grating_response_all_cells[rept,0:16,grat,cell]
            baseline_F0=np.nanmean(baseline_segment)
            if baseline_F0<0.0001:
                df_fpercentage[rept,:,grat,cell]=grating_response_all_cells[rept,:,grat,cell]-0
            else:
                df_fpercentage[rept,:,grat,cell]=(grating_response_all_cells[rept,:,grat,cell]-baseline_F0)/baseline_F0
fig, ax =plt.subplots(2)
ax[0].imshow(df_fpercentage[:,:,0,0],aspect='auto',cmap='inferno')
ax[1].imshow(grating_response_all_cells[:,:,0,0],aspect='auto',cmap='inferno')
#%%

trial_averaged_df_f=np.nanmean(df_fpercentage,0)
plt.imshow(trial_averaged_df_f[:,2,:].T,vmin=0, vmax=10,aspect='auto',cmap='inferno')
plt.colorbar()  
mean_df_f_to_gratings=np.nanmean(trial_averaged_df_f,2)
plt.imshow(mean_df_f_to_gratings.T,vmin=0, vmax=10,aspect='auto',cmap='inferno')
plt.colorbar()  
mean_df_f_per_cell=np.nanmean(trial_averaged_df_f,1)
plt.imshow(mean_df_f_per_cell.T,vmin=0, vmax=10,aspect='auto',cmap='inferno')
plt.colorbar()  

trial_averaged_df_f_evoked=trial_averaged_df_f[16:-16,:,:]
mean_evoked_df_d=np.nanmean(trial_averaged_df_f[16:-16,:,:],0)
trial_evoked_df_d=df_fpercentage[:,16:-16,:,:]
mean_trial_evoked_df_d=np.nanmean(df_fpercentage[:,16:-16,:,:],1)

fig, ax =plt.subplots(4)
ax[0].imshow(trial_averaged_df_f_evoked[:,0,:],vmin=0, vmax=10,aspect='auto',cmap='inferno')
ax[1].imshow(mean_evoked_df_d,vmin=0, vmax=10,aspect='auto',cmap='inferno')
ax[2].imshow(trial_evoked_df_d[:,:,0,0],vmin=0, vmax=10,aspect='auto',cmap='inferno')
ax[3].imshow(mean_trial_evoked_df_d[:,:,0],vmin=0, vmax=10,aspect='auto',cmap='inferno')


angles=np.linspace(0,360,9)[:-1]
freqencies=np.array([1,2,4,8,15])
fullcom = (5, 8)
x = angles
y = freqencies
xv, yv = np.meshgrid(x, y)
anglevalues = np.reshape(np.arange(1,41), (5, 8))

prefered_combination=mean_evoked_df_d.argmax(axis=0)

coord=np.argwhere(anglevalues==prefered_combination[i])[0]



prefered_angle=np.zeros(prefered_combination.shape[0])
prefered_frequency=np.zeros(prefered_combination.shape[0])
for z, i in enumerate(prefered_combination):
    coord=np.argwhere(anglevalues==prefered_combination[i]+1).flatten()
    coord_idx = [yv[coord[0],0] , xv[0,coord[1]]]
    prefered_angle[z]=coord_idx[1]
    prefered_frequency[z]=coord_idx[0]


def ortho(x):
    if x<180:
       return x+180
    elif x>=180:
       return x-180
vortho=np.vectorize(ortho)
orthogonal_angle=vortho(prefered_angle)

mean_evoked_df_d_prefered=np.zeros((1,mean_evoked_df_d.shape[1]))
mean_evoked_df_d_ortho=np.zeros((1,mean_evoked_df_d.shape[1]))

for i in range(mean_evoked_df_d.shape[1]):
    mean_evoked_df_d_prefered[0,i]=np.nanmean(mean_evoked_df_d[anglevalues[:,np.argwhere(xv==prefered_angle[i])[0][1]]-1,i],0)
    mean_evoked_df_d_ortho[0,i]=np.nanmean(mean_evoked_df_d[anglevalues[:,np.argwhere(xv==orthogonal_angle[i])[0][1]]-1,i],0)
    
osi=np.squeeze((mean_evoked_df_d_prefered-mean_evoked_df_d_ortho)/(mean_evoked_df_d_prefered+mean_evoked_df_d_ortho))
filtered_osi=osi[ np.logical_and(osi<2,osi>0)]




mean_evoked_df_d_exponential=np.zeros((8,mean_evoked_df_d.shape[1]))
mena_accc=np.zeros((8,mean_evoked_df_d.shape[1]))
for cell in range(mean_evoked_df_d.shape[1]):
    cell_prefered_frequency=prefered_frequency[cell]
    for grat in range(8):
        angless=anglevalues[np.argwhere(yv==cell_prefered_frequency)[0][0],:]-1
        mena_accc[grat,cell]=mean_evoked_df_d[angless[grat],cell]
        mean_evoked_df_d_exponential[grat,cell]=mena_accc[grat,cell]*np.exp(2.j*angles[grat])

gosi=np.sum(mean_evoked_df_d_exponential,0)/np.sum(mena_accc,0)


fig = plt.figure()
plt.hist(gosi)
fig = plt.figure()
plt.hist(filtered_osi)
#%% DSI



#%%
idx = (-osi).argsort()[:5]

cell=idx[2]
cell=150

angles_matrix=np.array([np.deg2rad(angles),]*mean_trial_evoked_df_d.shape[2]).transpose()
full_angles_matrix=np.array([angles_matrix,]*5)

temporl=freqencies

temporl_matrix=np.array([temporl,]*8)
full_temporl_matrix=np.array([temporl_matrix,]*mean_trial_evoked_df_d.shape[2]).transpose()

single_cell_area=mean_trial_evoked_df_d[:,:,cell]


single_cell_area_reshaped=np.reshape(single_cell_area,(15,5,8))


single_cell_angle=full_angles_matrix[:,:,cell]
single_cell_radius=full_temporl_matrix[:,:,cell]



textstr = '\n'.join((
    # 'pAngle={}°'.format(prefered_angle[cell]),
    'OSI={}'.format(np.round(osi[cell],1)),
    'gOSI={}'.format(np.round(gosi[cell],1))
    ))

colors=single_cell_angle
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
for i in range(15):
    noise = np.reshape(np.random.normal(0,0.05,40),(5,8))
    ax.scatter(single_cell_angle+noise,single_cell_radius+noise, single_cell_area_reshaped[i,:,:]*15, c=colors, cmap='hsv')
    
    
    
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0, 1.1, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


#%% signal correlations


signal_corelations=np.zeros((mean_evoked_df_d.shape[1],mean_evoked_df_d.shape[1]))

for cell_target in range(mean_evoked_df_d.shape[1]):
    for cell_other in range(mean_evoked_df_d.shape[1]):
        sum_correlations=0
        for orient in range(mean_evoked_df_d.shape[0]):
            singlecorrelation=stats.spearmanr(trial_averaged_df_f_evoked[:,orient,cell_target], trial_averaged_df_f_evoked[:,orient,cell_other])
            sum_correlations=sum_correlations+singlecorrelation[0]
        avera_sum_correlations=sum_correlations/mean_evoked_df_d.shape[0]
        signal_corelations[cell_target,cell_other]=avera_sum_correlations
plt.hist(signal_corelations[cell,:])




#%% noise correlations

noise_corelations=np.zeros((mean_evoked_df_d.shape[1],mean_evoked_df_d.shape[1]))

for x in range(7):
    if x==0:
        concatenated_trial_activity =np.concatenate([df_fpercentage[:,:,x,:], df_fpercentage[:,:,x+1,:]])
    else:
        concatenated_trial_activity =np.concatenate([concatenated_trial_activity, df_fpercentage[:,:,x+1,:]])

for cell_target in range(mean_evoked_df_d.shape[1]):
    for cell_other in range(mean_evoked_df_d.shape[1]):
        sum_correlations=0
        for trial in range(concatenated_trial_activity.shape[0]):   
            singlecorrelation=stats.spearmanr(concatenated_trial_activity[trial,:,cell_target], concatenated_trial_activity[trial,:,cell_other])
            sum_correlations=sum_correlations+singlecorrelation[0]
        avera_sum_correlations=sum_correlations/concatenated_trial_activity.shape[0]
        noise_corelations[cell_target,cell_other]=avera_sum_correlations



corrected_signal_correlations=signal_corelations
corrected_noise_correlations=noise_corelations
corrected_signal_correlations[np.diag_indices(corrected_signal_correlations.shape[0])]=0
corrected_noise_correlations[np.diag_indices(corrected_noise_correlations.shape[0])]=0

triang_signal_corelations=signal_corelations[np.triu_indices(corrected_signal_correlations.shape[0])]
triang_signal_corelations = triang_signal_corelations[triang_signal_corelations != 1]
triang_noise_corelations=noise_corelations[np.triu_indices(corrected_noise_correlations.shape[0])]
triang_noise_corelations = triang_noise_corelations[triang_noise_corelations != 1]

test=np.array(list(zip(triang_signal_corelations,triang_noise_corelations)))
plt.scatter(test[:,1],test[:,0])





#%% representational similarity matrix
represnet_similarty_matrix=np.zeros((trial_averaged_df_f_evoked.shape[1],trial_averaged_df_f_evoked.shape[1]))
for orient in range(trial_averaged_df_f_evoked.shape[1]):
    for comp_orient in range(trial_averaged_df_f_evoked.shape[1]):
        sperm_cell_correlations=np.zeros((trial_averaged_df_f_evoked.shape[2],1))
        for cell in range(trial_averaged_df_f_evoked.shape[2]):
            singlecorrelation=stats.spearmanr(trial_averaged_df_f_evoked[:,orient,cell], trial_averaged_df_f_evoked[:,comp_orient,cell])
            sperm_cell_correlations[cell,0]=singlecorrelation[0]
        represnet_similarty_matrix[orient,comp_orient]=np.nanmean(sperm_cell_correlations)
scaled_diagonal_represnet_similarty_matrix=represnet_similarty_matrix
scaled_diagonal_represnet_similarty_matrix[np.diag_indices_from(scaled_diagonal_represnet_similarty_matrix)]=np.mean(np.triu(scaled_diagonal_represnet_similarty_matrix))

plt.imshow(1-scaled_diagonal_represnet_similarty_matrix,aspect='auto')



#%% running modulation
rectified_speed_array
tuning_stim_on_index_full_recording
tuning_stim_off_index_full_recording
voltage_post_frames=np.around(milisecond_period*post_frames).astype('int64')
voltage_pre_frames=np.around(milisecond_period*pre_frames).astype('int64')
voltage_frame_number=np.arange(tuning_stim_on_index_full_recording[0,0]-voltage_pre_frames,tuning_stim_off_index_full_recording[0,0]+voltage_post_frames).size
voltage_frame_number=5015+voltage_post_frames+voltage_pre_frames

grating_locomotion_all_cells=np.zeros((repetitions,voltage_frame_number,grating_number,activity_matrix.shape[1]));
for cell in range(0,activity_matrix.shape[1]):
    for grat in range(0,grating_number):
        grating_select=grat
        selected_grating_starts=tuning_stim_on_index_full_recording[grating_select,:]
        selected_grating_ends=tuning_stim_off_index_full_recording[grating_select,:]
        for rept in range(0,repetitions):           
            frame_window=np.arange(selected_grating_starts[rept]-voltage_pre_frames,selected_grating_ends[rept]+voltage_post_frames).astype('int64')
            if frame_window.size!=7127:
                toadd=7127-frame_window.size
                for i in range(toadd):
                    frame_window=np.append(frame_window,frame_window[-1]+1)
                
    
            grating_locomotion_all_cells[rept,:,grat,cell]=rectified_speed_array[frame_window]



evoked_grating_locomotion_all_cells=grating_locomotion_all_cells[:,voltage_pre_frames:5015+voltage_pre_frames,:,:]
mean_evoked_grating_locomotion_all_cells=np.mean(evoked_grating_locomotion_all_cells,1)


cell=15
cell_prefered_grating=prefered_angle[cell]
test=mean_evoked_grating_locomotion_all_cells[:,cell_prefered_grating,cell]

test>0.002



plt.plot(evoked_grating_locomotion_all_cells[9,:,cell_prefered_grating,cell])
plt.hist(test)



#%



#%% plotting


nrows=8
ncols=10
fig, ax = plt.subplots(nrows=nrows,ncols=ncols,sharey=True)
# fig.set_title('Snapping cursor')
for cell in range(0,ncols):
    for grating in range(0,nrows):
        line, = ax[grating,cell].plot(trial_averaged_df_f[:,grating,cell+20])
     
# snap_cursor = SnappingCursor(ax[0], line)

# fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
# mplcursors.cursor(line) # or just mplcursors.cursor()

plt.show()


#%%



plt.imshow(noise_corelations.T,aspect='auto')
plt.imshow(mean_response_to_gratings.T,aspect='auto')



plt.hist(test)