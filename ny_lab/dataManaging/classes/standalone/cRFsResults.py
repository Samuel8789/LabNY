# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:05:37 2021

@author: sp3660
"""

import matplotlib.pyplot as plt
import matplotlib as mlp

# import h5py
# import scipy.io
import mat73
import numpy as np
import copy

class CRFsResults():
    
    def __init__(self, mat_results_file, plane1_cell_number=None, plane2_cell_number=None, plane3_cell_number=None,
                 best_model_file=None, model_collection_file=None, model_parameters_file=None, structures_file=None):
             
        self.stimuli_labels=['0_2hz','45_2hz','90_2hz','135_2hz','180_2hz','225_2hz','270_2hz','315_2hz']
        self.mat_results_file = mat_results_file
        self.results = mat73.loadmat(self.mat_results_file)
        self.core_crf=self.results['results']['core_crf']
        self.PCNs=self.results['PCNs']
        self.ensembles={ self.stimuli_labels[i]:ensemble.astype(int)  for i, ensemble in enumerate(self.core_crf)}
        self.PatternCompletionCells={ self.stimuli_labels[i]:PCN.astype(int) for i, PCN in enumerate(self.PCNs)}
        self.activity_matrix=self.results['params']['data'].T
        self.stimulus_matrix=self.results['params']['UDF'].T
        self.cell_number=  self.activity_matrix.shape[0]
        self.plane1_cell_number=plane1_cell_number
        self.plane2_cell_number=plane2_cell_number
        self.plane3_cell_number=plane3_cell_number
        self.correct_indexes_by_plane()
        self.all_frame_rates=[]
        self.get_stim_indexes()
        self.transform_binary_activity_to_frame_rate()
        self.get_stimuli_activity()
        self.calculate_df_over_prestim()
        self.get_activities_by_stimuli()

        # self.plotting_general()
        # self.plotting_traces_and_rates()
        # self.plotting_normalized_dff()
        
    def correct_indexes_by_plane(self):
        
        self.ensembles_plane_full_indexes={}
        for grating_angle, ensemble in  self.ensembles.items():
           self.ensembles_plane_full_indexes[grating_angle]={}
           self.ensembles_plane_full_indexes[grating_angle]['Plane1']=ensemble[ensemble<= self.plane1_cell_number]
           self.ensembles_plane_full_indexes[grating_angle]['Plane2']=ensemble[np.logical_and(ensemble >  self.plane1_cell_number, ensemble <=  self.plane1_cell_number+ self.plane2_cell_number)]- self.plane1_cell_number
           self.ensembles_plane_full_indexes[grating_angle]['Plane3']=ensemble[ensemble> self.plane1_cell_number+ self.plane2_cell_number ]-(  self.plane1_cell_number + self.plane2_cell_number )
           
        self.PCCs_plane_full_indexes={}
        for grating_angle, PCN_ensemble in  self.PatternCompletionCells.items():
           self.PCCs_plane_full_indexes[grating_angle]={}
           self.PCCs_plane_full_indexes[grating_angle]['Plane1']=PCN_ensemble[PCN_ensemble<= self.plane1_cell_number]
           self.PCCs_plane_full_indexes[grating_angle]['Plane2']=PCN_ensemble[np.logical_and(PCN_ensemble >  self.plane1_cell_number, PCN_ensemble <=  self.plane1_cell_number+ self.plane2_cell_number)]- self.plane1_cell_number
           self.PCCs_plane_full_indexes[grating_angle]['Plane3']=PCN_ensemble[PCN_ensemble> self.plane1_cell_number+ self.plane2_cell_number ]-(  self.plane1_cell_number + self.plane2_cell_number )

    def transform_binary_activity_to_frame_rate(self):
        print('calculating_frame_rates')
        self.all_frame_rates=np.zeros(self.activity_matrix.shape)
        for cell in range(self.activity_matrix.shape[0]):      
            frame_rate=[]
            for i, frame in  enumerate(self.activity_matrix[cell,:]):
                if i>0 and i<len(self.activity_matrix[cell,:])-1:
                    frame_rate.append(sum([self.activity_matrix[cell,i-1], frame, self.activity_matrix[cell,i+1]])/0.18)
                if i==0: 
                    frame_rate.append(sum([frame,self.activity_matrix[cell,i+1]])/0.18)
                if i==len(self.activity_matrix[cell,:])-1:
                    frame_rate.append(sum([self.activity_matrix[cell,i-1], frame])/0.18)
            self.all_frame_rates[cell,:]=frame_rate
            
        print('frame_rates_done')

    def get_stim_indexes(self):
        
        self.start_indexes={}
        self.stim_ranges={}
        for  i, grating_angle in enumerate(self.ensembles.keys()):   
            self.start_indexes[grating_angle]=np.argwhere(np.diff(self.stimulus_matrix[i,:])==1).flatten()+1 
            self.stim_ranges[grating_angle]=[range(stim-12,stim+40) for stim in self.start_indexes[grating_angle]]
        
    def get_stimuli_activity(self):
        
        self.cells_grating_rates=[]
        self.cells_grating_spikes=[]
        for cell  in range(self.cell_number):
            cell_rates_to_grating={}
            cell_spikes_to_grating={}
            for i, grating_angle in enumerate(self.stimuli_labels):
                cell_rates_to_grating[grating_angle]=np.zeros((len(self.stim_ranges[grating_angle]),len(self.stim_ranges[grating_angle][0])))
                cell_spikes_to_grating[grating_angle]=np.zeros((len(self.stim_ranges[grating_angle]),len(self.stim_ranges[grating_angle][0])))
                for trial_number, trial in enumerate(self.stim_ranges[grating_angle]):
                    cell_rates_to_grating[grating_angle][trial_number,:]=self.all_frame_rates[cell, trial]
                    cell_spikes_to_grating[grating_angle][trial_number,:]=self.activity_matrix[cell,trial]
                   
            self.cells_grating_rates.append(cell_rates_to_grating)
            self.cells_grating_spikes.append(cell_spikes_to_grating)


        self.trial_stimuli={}
        for i, grating_angle in enumerate(self.stimuli_labels):
             self.trial_stimuli[grating_angle]=np.zeros((len(self.stim_ranges[grating_angle]),len(self.stim_ranges[grating_angle][0])))
             for trial_number, trial in enumerate(self.stim_ranges[grating_angle]):
                 self.trial_stimuli[grating_angle][trial_number,:]=self.stimulus_matrix[i, trial]


    def calculate_df_over_prestim(self):
        
        self.relative_cells_grating_rates=copy.deepcopy(self.cells_grating_rates)
        self.relative_cells_grating_spikes=copy.deepcopy(self.cells_grating_spikes)

        def df_f_trial(trial_activity):

          prestim_activity=trial_activity[0:12]
          prestim_mean=np.mean(prestim_activity)
          if prestim_mean==0:
              prestim_mean=1
          df_f_trial=trial_activity/ prestim_mean

          return df_f_trial

        for i,cell in enumerate(self.relative_cells_grating_rates):
            for grating, trial_responses in cell.items():            
                self.relative_cells_grating_rates[i][grating]=np.apply_along_axis(df_f_trial, 1, trial_responses)  
        for i,cell in enumerate(self.relative_cells_grating_spikes):
            for grating, trial_responses in cell.items():
                self.relative_cells_grating_spikes[i][grating]=np.apply_along_axis(df_f_trial, 1, trial_responses)
 
        self.mean_relative_cells_grating_rates=copy.deepcopy(self.relative_cells_grating_rates)
        self.mean_relative_cells_grating_spikes=copy.deepcopy(self.relative_cells_grating_spikes)
        
        for i,cell in enumerate(self.mean_relative_cells_grating_rates):
            for grating, trial_responses in cell.items():
                self.mean_relative_cells_grating_rates[i][grating]=np.mean(trial_responses,axis=0)               
        for i,cell in enumerate(self.mean_relative_cells_grating_spikes):
            for grating, trial_responses in cell.items():
                self.mean_relative_cells_grating_spikes[i][grating]=np.mean(trial_responses,axis=0)


    def get_activities_by_stimuli(self):
 
        self.grating_activity={}
        for stimulus_label in self.mean_relative_cells_grating_rates[0].keys():
            non_ensemble_cells_indexes= [x for x in range(self.cell_number) if x not in self.ensembles[stimulus_label]]
            ensemble_indexes=self.ensembles[stimulus_label].tolist()
            PAterncompletion_cell_indexes=self.PatternCompletionCells[stimulus_label].tolist()
            
            activities=[self.mean_relative_cells_grating_rates, self.relative_cells_grating_rates, self.cells_grating_rates]
            indexes=[ensemble_indexes, non_ensemble_cells_indexes, PAterncompletion_cell_indexes]
            cell_types=['Ensembles', 'Non_enembles', 'PCNs']
            activity_types=['mean_relative','relative','rates']
            
            self.grating_activity[stimulus_label]={}
         
            for i, cell_type in enumerate(cell_types):
                self.grating_activity[stimulus_label][cell_type]={}
                for j, activity_type in enumerate (activity_types):
                    self.grating_activity[stimulus_label][cell_type][activity_type]=[activities[j][index-1][stimulus_label] for index in indexes[i]]
        
        
        
    def plotting_general(self, cell_range=None,sequence_range=None, vmax1=None, vmax2=None, norm1=None, norm2=None):
        norm1=mlp.colors.Normalize(0, 1)
        norm2=mlp.colors.LogNorm(vmin=self.all_frame_rates.min(), vmax=self.all_frame_rates.max())

        if not cell_range:
             cell_range=slice(0,self.activity_matrix.shape[0],1)
        if not cell_range:
             sequence_range=slice(0,self.activity_matrix.shape[1],1)

        if norm1:
            norm1=norm1
        if norm2:
            norm2=norm2
        if vmax1 :
            norm1=mlp.colors.Normalize(0, vmax1)
        if vmax2 :
            norm2=mlp.colors.Normalize(0, vmax2)
                
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.imshow(self.activity_matrix[cell_range,sequence_range], cmap='binary', aspect='auto', norm=norm1)   
        ax2.imshow(self.stimulus_matrix[:,sequence_range], cmap='binary', aspect='auto')
        ax3.imshow(self.all_frame_rates[cell_range,sequence_range], cmap='binary', aspect='auto', norm=norm2)
        plt.figure()
        plt.imshow(self.activity_matrix[cell_range,sequence_range], cmap='binary', aspect='auto', norm=norm1)  
        plt.figure()
        plt.imshow(self.stimulus_matrix[:,sequence_range], cmap='binary', aspect='auto', norm=norm1)  
        plt.figure()
        plt.imshow(self.all_frame_rates[cell_range,sequence_range], cmap='binary', aspect='auto', norm=norm2)  
        print(np.amax(self.all_frame_rates))
        print(np.amax(self.activity_matrix))
        
        # f, axs= plt.subplots(2, 1, sharex=True)
        # n_bins=20
        # axs[0].hist(self.all_frame_rates, bins=n_bins)
        # axs[1].hist(self.activity_matrix, bins=n_bins)




        
    def plotting_traces_and_rates(self,vmax1=2, vmax2=200, vmax3=2, vmax4=200):
         
        for i, (grating_angle, spikes) in enumerate(self.ensemble_specific_spikes.items()):
               
            f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
            ax1.plot(self.stimulus_matrix[i])
            ax2.imshow(self.ensemble_specific_spikes[grating_angle], cmap='inferno', aspect='auto', vmax=vmax1)
            ax3.plot(np.mean(self.ensemble_specific_spikes[grating_angle],axis=0))
            ax4.imshow(self.PCNs_specific_spikes[grating_angle], cmap='inferno', aspect='auto', vmax=vmax2)
            ax5.plot(np.mean(self.PCNs_specific_spikes[grating_angle],axis=0))

            f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
            ax1.plot(self.stimulus_matrix[i])
            ax2.imshow(self.ensemble_specific_rates[grating_angle], cmap='inferno', aspect='auto', vmax=vmax3)
            ax3.plot(np.mean(self.ensemble_specific_rates[grating_angle],axis=0))
            ax4.imshow(self.PCNs_specific_rates[grating_angle], cmap='inferno', aspect='auto', vmax=vmax4)
            ax5.plot(np.mean(self.PCNs_specific_rates[grating_angle],axis=0))
           
        for i, (grating_angle, ensemble) in enumerate(self.PCNs_responses_to_stimuli.items()): 
            rowws=len(ensemble)+1
            fig, rows = plt.subplots(rowws, 1, sharex=True)
            fig2, rows2 = plt.subplots(rowws, 1, sharex=True)
            rows[0].plot(np.mean(self.trial_stimuli[grating_angle], axis=0))
            rows2[0].plot(np.mean(self.trial_stimuli[grating_angle], axis=0))

            for cell, trials in ensemble.items():
                rows[cell+1].plot(np.mean(self.PCNs_responses_to_stimuli[grating_angle][cell], axis=0))
                rows2[cell+1].plot(np.mean(self.PCNs_rates_to_stimuli[grating_angle][cell], axis=0))
                     
    def plotting_normalized_dff(self, vmax1=2, vmax2=2):
        for i, (grating_angle, ensemble) in enumerate(self.normalized_PCNs_rates_to_stimuli.items()): 
            rowws=len(ensemble)+1
            fig, rows = plt.subplots(rowws, 1, sharex=True)
            rows[0].plot(np.mean(self.trial_stimuli[grating_angle], axis=0))
            for cell, trials in ensemble.items():
                rows[cell+1].imshow(self.normalized_PCNs_rates_to_stimuli[grating_angle][cell], aspect='auto', vmax=vmax1)         
            
        for i, (grating_angle, ensemble) in enumerate(self.normalized_ensemble_rates_to_stimuli.items()): 
            rowws=len(ensemble)+1
            fig, rows = plt.subplots(rowws, 1, sharex=True)
            rows[0].plot(np.mean(self.trial_stimuli[grating_angle], axis=0))
            for cell, trials in ensemble.items():
                rows[cell+1].imshow(self.normalized_ensemble_rates_to_stimuli[grating_angle][cell], aspect='auto', vmax=vmax2)         
                
                
    def plotting_mean_normalized_dff(self, vmax1=2, vmax2=2):
            
                
        for grating_angle, angle_gratings_activity in self.mean_normalized_trials_ensembles.items():
            fig, rowws = plt.subplots(2, 1, sharex=True)
            rowws[0].plot(np.mean(self.trial_stimuli[grating_angle], axis=0))
            rowws[1].imshow(angle_gratings_activity, aspect='auto')
        
        for grating_angle, angle_gratings_activity in self.mean_normalized_trials_PCNs.items():
            fig, rowws = plt.subplots(2, 1, sharex=True)
            rowws[0].plot(np.mean(self.trial_stimuli[grating_angle], axis=0))
            rowws[1].imshow(angle_gratings_activity, aspect='auto')
                
                
     

if __name__ == "__main__":
    
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIK3planeallen\Plane1'
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\SAMUEL_BIG_RUN_2Hz'
    # this results are for the 2hz frequency only
    SPJA_0702_allen_CRFS=CRFsResults(temporary_path+ r'\results.mat',
                                     plane1_cell_number=110,
                                     plane2_cell_number=96 ,
                                     plane3_cell_number=65
                                     )
    res=SPJA_0702_allen_CRFS
#%%

# stim=res.stimuli_labels[0]
# non_stim_stim=res.stimuli_labels[1:]

# ensemble_indexes=res.ensembles[stim]
# non_ensemble_cells_indexes=np.array([x for x in range(res.cell_number) if x not in res.ensembles[stim]])
# pcns_indexes=res.PatternCompletionCells[stim]



# other_ensembles=[[res.ensembles[x].tolist()] for x in non_stim_stim]
# other_ensembles = [item for sublist in other_ensembles for item in sublist]
# other_ensembles = set([item for sublist in other_ensembles for item in sublist])

# uniqe_ensembles_cells_indexes=np.array([cell for cell in ensemble_indexes if cell not in other_ensembles])
# unique_PCns=np.array([cell for cell in pcns_indexes if cell in uniqe_ensembles_cells_indexes])






# #%%
# %matplotlib inline
# color= ['blue', 'orange', 'green']
# for stim in res.stimuli_labels:
#     stimactivity=res.grating_activity
#     degres0=stimactivity[stim]
#     res.ensembles[stim]
#     non_ensemble_cells_indexes= [x for x in range(res.cell_number) if x not in res.ensembles[stim]]

    
#     ensembleactivty=degres0['Ensembles']['relative']
#     nonensembleactivty=degres0['Non_enembles']['relative']
#     pcnsactivty=degres0['PCNs']['relative']
    
#     meanensembleactivty=np.vstack(degres0['Ensembles']['mean_relative'])
#     meannonensembleactivty=np.vstack(degres0['Non_enembles']['mean_relative'])
#     meanpcnsactivty=np.vstack(degres0['PCNs']['mean_relative'])
    
#     rowes=6
#     fig, rowes = plt.subplots(rowes, 1, sharex=True)
#     rowes[0].plot(np.mean(res.trial_stimuli[stim], axis=0))
#     rowes[1].imshow(meanensembleactivty, aspect='auto')
#     rowes[4].plot(np.mean(meanensembleactivty, axis=0), color=color[0], label='Ensemble')
#     rowes[5].plot(np.mean( meanensembleactivty/np.linalg.norm(meanensembleactivty, axis=1)[:,np.newaxis], axis=0), color=color[0], label='Ensemble')
#     rowes[2].imshow(meanpcnsactivty, aspect='auto')
#     rowes[4].plot(np.mean(meanpcnsactivty, axis=0), color=color[1], label='PCCs')
#     rowes[5].plot(np.mean(meanpcnsactivty/np.linalg.norm(meanpcnsactivty, axis=1)[:,np.newaxis], axis=0), color=color[1], label='PCCs')
#     rowes[3].imshow(meannonensembleactivty[6:11,:], aspect='auto')
#     rowes[4].plot(np.mean(meannonensembleactivty[6:11,:], axis=0), color=color[2], label='Non Ensemble')
#     rowes[5].plot(np.mean(meannonensembleactivty[6:11,:]/np.linalg.norm(meannonensembleactivty[0:5,:], axis=1)[:,np.newaxis], axis=0), color=color[2], label='Non Ensemble')

#     rowes[0].set_title('Stimulus')
#     rowes[1].set_title('Ensemble')
#     rowes[2].set_title('PCCs')
#     rowes[3].set_title('Non Ensemble')
#     rowes[4].set_title('Mean Activity')
#     rowes[5].set_title('Normalized Mean Activity')
#     fig.suptitle(stim, fontsize=16)
#     fig.set_size_inches(20, 20)
#     fig.tight_layout
#     rowes[4].legend( loc='upper left')
#     rowes[5].legend(loc='upper left')


# #%%
# fig, axes = plt.subplots(rowes, columns, sharex=True, sharey=True)
# axes[0][0].plot(np.transpose(aucs[res.ensembles[stim],0]))
# axes[0][1].imshow(aucs[res.ensembles[stim],:], aspect='auto')

# plt.plot(aucs[non_ensemble_cells_indexes,0])
# plt.imshow(aucs[non_ensemble_cells_indexes,:], aspect='auto')


# plt.plot(aucs[:,0])

# meannonensembleactivty
# test3=np.linalg.norm(test, axis=1)
# test2=test/test3[:,np.newaxis]


# fig, axes = plt.subplots(4, 1, sharex=True)
# axes[0].plot(np.mean(meanensembleactivty, axis=0))
# axes[1].imshow(meanensembleactivty, aspect='auto')
# axes[2].plot(np.mean(meannonensembleactivty, axis=0))
# axes[3].imshow(meannonensembleactivty, aspect='auto')



# fig, axes = plt.subplots(4, 1, sharex=True)
# axes[0].plot(np.mean(test2, axis=0))
# axes[1].imshow(test2, aspect='auto')
# axes[2].plot(np.mean(    meannonensembleactivty/np.linalg.norm(meannonensembleactivty, axis=1)[:,np.newaxis]
# , axis=0))
# axes[3].imshow(    meannonensembleactivty/np.linalg.norm(meannonensembleactivty, axis=1)[:,np.newaxis]
# , aspect='auto')



# plt.plot(aucs[non_ensemble_cells_indexes,0])
# plt.imshow(aucs[non_ensemble_cells_indexes,:], aspect='auto')
# plt.imshow(test2, aspect='auto')
# plt.imshow(test, aspect='auto')


# rowes=15
# for stim in res.stimuli_labels:
#     fig, rowes = plt.subplots(rowes, 1, sharex=True)
#     for i, row in enumerate(res.trial_stimuli[stim]):
#         rowes[i].plot(row)

    
    
# aucs=res.results['results']['auc']   
# connectivity_matrix=res.results['best_model']['structure']
# # cellensembleactivity=[res.cells_grating_rates[i] for i in res.ensembles[stim]]


# pixel_per_bar = 4
# dpi = 100
# fig = plt.figure(figsize=(200 * pixel_per_bar / dpi, 2), dpi=dpi)
# ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
# ax.set_axis_off()
# ax.imshow(res.all_frame_rates, cmap='binary', aspect='auto',
#           interpolation='nearest', norm=mlp.colors.Normalize(0, 1))
# #%%
# # non_ensemble_angle=270
# # fig, rowws = plt.subplots(2, 1, sharex=True)
# # rowws[0].plot(np.mean(SPJA_0702_allen_CRFS.trial_stimuli[0], axis=0))
# # rowws[1].imshow(SPJA_0702_allen_CRFS.mean_normalized_trials_PCNs[0], aspect='auto')  


# fig, rowws = plt.subplots(2, 1, sharex=True)
# rowws[0].plot(np.mean(SPJA_0702_allen_CRFS.trial_stimuli[0], axis=0))
# rowws[1].imshow(np.vstack(mean_normalized_PCNs_rates_to_orthogonal_stimuli[non_ensemble_angle]), aspect='auto')  
# #%%

# for grating_angle, angle_gratings_activity in SPJA_0702_allen_CRFS.mean_normalized_trials_PCNs.items():
#     fig, rowws = plt.subplots(2, 1, sharex=True)
#     rowws[0].plot(np.mean(SPJA_0702_allen_CRFS.trial_stimuli[grating_angle], axis=0))
#     rowws[1].plot(np.mean(angle_gratings_activity,  axis=0))
    
# for grating_angle, angle_gratings_activity in SPJA_0702_allen_CRFS.mean_normalized_trials_PCNs.items():
#     fig, rowws = plt.subplots(2, 1, sharex=True)
#     rowws[0].plot(np.mean(SPJA_0702_allen_CRFS.trial_stimuli[grating_angle], axis=0))
#     rowws[1].imshow(angle_gratings_activity, aspect='auto')  

#%%


