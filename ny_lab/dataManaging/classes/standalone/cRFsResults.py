# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:05:37 2021

@author: sp3660
 """

import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# import h5py
# import scipy.io
import mat73
import numpy as np
import copy
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, RocCurveDisplay,  PrecisionRecallDisplay, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import pandas as pd

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b",'g','y','c','m', 'tab:brown']) 

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

class CRFsResults():
    
    def __init__(self, mat_results_file, plane1_cell_number=None, plane2_cell_number=None, plane3_cell_number=None,
                 best_model_file=None, model_collection_file=None, model_parameters_file=None, structures_file=None):
        print('Processing CRFResults')

             
        self.mat_results_file = mat_results_file

        self.all_frame_rates=np.zeros(1)
        [path, file_name]=os.path.split(self.mat_results_file)
        frame_rates_file_name=os.path.splitext(file_name)[0]+'_frame_rates'
        self.all_frame_rates_to_save_path= os.path.join(path,frame_rates_file_name)
        
        self.stimuli_labels=['0_2hz','45_2hz','90_2hz','135_2hz','180_2hz','225_2hz','270_2hz','315_2hz']
        self.results = mat73.loadmat(self.mat_results_file)
        self.core_crf=self.results['results']['core_crf']
        self.PCNs=self.results['PCNs']
        self.ensembles={ self.stimuli_labels[i]:ensemble.astype(int)  for i, ensemble in enumerate(self.core_crf)}
        self.PatternCompletionCells={ self.stimuli_labels[i]:PCN.astype(int) for i, PCN in enumerate(self.PCNs)}
        self.ensembles_pyhton_corrected={ self.stimuli_labels[i]:ensemble.astype(int)-1  for i, ensemble in enumerate(self.core_crf)}
        self.PatternCompletionCells_pyhton_corrected={ self.stimuli_labels[i]:PCN.astype(int)-1 for i, PCN in enumerate(self.PCNs)}
        self.activity_matrix=self.results['params']['data'].T
        self.stimulus_matrix=self.results['params']['UDF'].T
        self.cell_number=  self.activity_matrix.shape[0]
        self.plane1_cell_number=plane1_cell_number
        self.plane2_cell_number=plane2_cell_number
        self.plane3_cell_number=plane3_cell_number
        self.calculate_extra_indexes()
        self.correct_indexes_by_plane()
        self.get_stim_indexes()
        
        if os.path.isfile(self.all_frame_rates_to_save_path+'.npy'):
            self.load_frame_rates()

        if not self.all_frame_rates.any():       
            self.transform_binary_activity_to_frame_rate()
            self.save_frame_rates()
        
        self.get_stimuli_activity()
        self.calculate_df_over_prestim()
        self.get_activities_by_stimuli()

        # self.plotting_general()
        # self.plotting_traces_and_rates()
        # self.plotting_normalized_dff()
        # self.plotting_mean_normalized_dff()
        # self.plotting_mean_normalized_dff_rasters()
        # self.plotting_mean_normalized_dff_rasters_uniques()
      
        
      
    def calculate_extra_indexes(self):

        self.extra_indexes={}
        
        for grating in self.stimuli_labels:
            self.extra_indexes[grating]={}
            non_stim_stim=[non_grating for non_grating in self.stimuli_labels if non_grating!=grating]
            
            ensemble_indexes=self.ensembles_pyhton_corrected[grating]
            
            self.extra_indexes[grating]['Ensembles']=np.array(ensemble_indexes)
            # non_ensemble_cells_indexes=np.array([x for x in range(self.cell_number) if x not in self.ensembles_pyhton_corrected[grating]])
            pcns_indexes=self.PatternCompletionCells_pyhton_corrected[grating]
            self.extra_indexes[grating]['PCNs']=np.array(pcns_indexes)
            
            self.extra_indexes[grating]['Cell_Outside_Ensemble']=np.array([x for x in range(self.cell_number) if x not in self.ensembles_pyhton_corrected[grating]])
    
            other_ensembles = [[self.ensembles_pyhton_corrected[x].tolist()] for x in non_stim_stim]
            other_ensembles = [item for sublist in other_ensembles for item in sublist]
            self.extra_indexes[grating]['Cells_In_Other_Ensembles'] = np.array(list(set([item for sublist in other_ensembles for item in sublist])))
            
            self.extra_indexes[grating]['unique_ensembles_cells_indexes']=np.array([cell for cell in ensemble_indexes if cell not in self.extra_indexes[grating]['Cells_In_Other_Ensembles']])
            self.extra_indexes[grating]['unique_PCns']=np.array([cell for cell in pcns_indexes if cell in  self.extra_indexes[grating]['unique_ensembles_cells_indexes']])

            self.extra_indexes[grating]['cells_in_no_ensemble']=np.array([cell for cell in  self.extra_indexes[grating]['Cell_Outside_Ensemble'] if cell not in  self.extra_indexes[grating]['Cells_In_Other_Ensembles']])

      
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
    
    def save_frame_rates(self):
        np.save(self.all_frame_rates_to_save_path, self.all_frame_rates)

    def load_frame_rates(self):
        self.all_frame_rates=np.load(self.all_frame_rates_to_save_path+'.npy')

    def get_stim_indexes(self):
        
        self.start_indexes={}
        self.stim_ranges={}
        for  i, grating_angle in enumerate(  self.stimuli_labels):   
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
          prestim_mean=np.nanmean(prestim_activity)
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
                self.mean_relative_cells_grating_rates[i][grating]=np.nanmean(trial_responses,axis=0)               
        for i,cell in enumerate(self.mean_relative_cells_grating_spikes):
            for grating, trial_responses in cell.items():
                self.mean_relative_cells_grating_spikes[i][grating]=np.nanmean(trial_responses,axis=0)


    def get_activities_by_stimuli(self):
 
        self.grating_activity={}
        for stimulus_label in self.mean_relative_cells_grating_rates[0].keys():
            non_ensemble_cells_indexes= [x for x in range(self.cell_number) if x not in   self.ensembles_pyhton_corrected[stimulus_label]]
            # ensemble_indexes=  self.ensembles_pyhton_corrected[stimulus_label].tolist()
            # PAterncompletion_cell_indexes=self.PatternCompletionCells_pyhton_corrected[stimulus_label].tolist()
            
            activities=[self.mean_relative_cells_grating_rates, self.relative_cells_grating_rates, self.cells_grating_rates, self.cells_grating_spikes]
            # indexes=[ensemble_indexes, non_ensemble_cells_indexes, PAterncompletion_cell_indexes]
            indexes=[self.extra_indexes[stimulus_label]['Ensembles'], 
                     self.extra_indexes[stimulus_label]['PCNs'], 
                     self.extra_indexes[stimulus_label]['Cell_Outside_Ensemble'],
                     self.extra_indexes[stimulus_label]['Cells_In_Other_Ensembles'],
                     self.extra_indexes[stimulus_label]['cells_in_no_ensemble'],
                     self.extra_indexes[stimulus_label]['unique_ensembles_cells_indexes'],
                     self.extra_indexes[stimulus_label]['unique_PCns']]

            cell_types=['Ensembles', 'PCNs', 'Cell_Outside_Ensemble','Cells_In_Other_Ensembles','cells_in_no_ensemble','unique_ensembles_cells_indexes','unique_PCns']
            activity_types=['mean_relative','relative','rates', 'spikes']
            
            self.grating_activity[stimulus_label]={}
         
            for i, cell_type in enumerate(cell_types):
                self.grating_activity[stimulus_label][cell_type]={}
                if indexes[i].any():
                    self.grating_activity[stimulus_label][cell_type]['full_activity_rates']=self.all_frame_rates[indexes[i],:]
                    self.grating_activity[stimulus_label][cell_type]['full_activity_spikes']=self.activity_matrix[indexes[i],:] 
                    for j, activity_type in enumerate (activity_types):
                        self.grating_activity[stimulus_label][cell_type][activity_type]=[activities[j][index][stimulus_label] for index in indexes[i]]
                    self.grating_activity[stimulus_label][cell_type]['mean_relative']=np.vstack(self.grating_activity[stimulus_label][cell_type]['mean_relative'])
                else:
                    self.grating_activity[stimulus_label][cell_type]['full_activity_rates']=np.zeros((2,self.all_frame_rates.shape[1]))
                    self.grating_activity[stimulus_label][cell_type]['full_activity_spikes']=np.zeros((2,self.activity_matrix.shape[1]))        
                    for j, activity_type in enumerate (activity_types):
                        self.grating_activity[stimulus_label][cell_type][activity_type]=np.zeros((2,    activities[1][0][stimulus_label].shape[1]))
                    self.grating_activity[stimulus_label][cell_type]['mean_relative']=np.zeros((2,  activities[1][0][stimulus_label].shape[1]))

    def plotting_general(self, cell_range=None,sequence_range=None, vmax1=None, vmax2=None, norm1=None, norm2=None):
        norm1=mpl.colors.Normalize(0, 1)
        norm2=mpl.colors.LogNorm(vmin=self.all_frame_rates.min(), vmax=self.all_frame_rates.max())

        if not cell_range:
             cell_range=slice(0,self.activity_matrix.shape[0],1)
        if not sequence_range:
             sequence_range=slice(0,self.activity_matrix.shape[1],1)

        if norm1:
            norm1=norm1
        if norm2:
            norm2=norm2
        if vmax1 :
            norm1=mpl.colors.Normalize(0, vmax1)
        if vmax2 :
            norm2=mpl.colors.Normalize(0, vmax2)
                
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
        
        
        pixel_per_bar = 4
        dpi = 100
        fig = plt.figure(figsize=(200 * pixel_per_bar / dpi, 2), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
        ax.set_axis_off()
        ax.imshow(self.all_frame_rates, cmap='binary', aspect='auto',
                  interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        
        
        
        # f, axs= plt.subplots(2, 1, sharex=True)
        # n_bins=20
        # axs[0].hist(self.all_frame_rates, bins=n_bins)
        # axs[1].hist(self.activity_matrix, bins=n_bins)

        
    def plotting_traces_and_rates(self,vmax1=2, vmax2=10, vmax3=2, vmax4=10):
         
        for i, (grating_angle, activity) in enumerate(self.grating_activity.items()):
            f, axs = plt.subplots(5, 1, sharex=True)
            axs[0].plot(self.stimulus_matrix[i,:])
            axs[1].imshow(activity['Ensembles']['full_activity_rates'], cmap='binary', aspect='auto', vmax=vmax1)
            axs[2].plot(np.nanmean(activity['Ensembles']['full_activity_rates'],axis=0))
            axs[3].imshow(activity['PCNs']['full_activity_rates'], cmap='binary', aspect='auto', vmax=vmax2)
            axs[4].plot(np.nanmean(activity['PCNs']['full_activity_rates'],axis=0))

        for i, (grating_angle, activity) in enumerate(self.grating_activity.items()):
            f, axs = plt.subplots(5, 1, sharex=True)
            axs[0].plot(self.stimulus_matrix[i,:])
            axs[1].imshow(activity['Ensembles']['full_activity_spikes'], cmap='binary', aspect='auto', vmax=vmax3)
            axs[2].plot(np.nanmean(activity['Ensembles']['full_activity_spikes'],axis=0))
            axs[3].imshow(activity['PCNs']['full_activity_spikes'], cmap='binary', aspect='auto', vmax=vmax4)
            axs[4].plot(np.nanmean(activity['PCNs']['full_activity_spikes'],axis=0))
           
        for i, (grating_angle, activity) in enumerate(self.grating_activity.items()): 
            rowws=len(activity['PCNs']['rates'])+1
            fig, rows = plt.subplots(rowws, 1, sharex=True)
            fig2, rows2 = plt.subplots(rowws, 1, sharex=True)
            rows[0].plot(np.nanmean(self.trial_stimuli[grating_angle], axis=0))
            rows2[0].plot(np.nanmean(self.trial_stimuli[grating_angle], axis=0))

            for cell, trials  in enumerate(activity['PCNs']['rates']):
                rows[cell+1].plot(np.nanmean(trials, axis=0))
                rows2[cell+1].plot(np.nanmean(activity['PCNs']['spikes'][cell], axis=0))
                     
    def plotting_normalized_dff(self, vmax1=2, vmax2=2):
        
        for i, (grating_angle, activity) in enumerate(self.grating_activity.items()):
            rowws=len(activity['PCNs']['relative'])+1
            fig, rows = plt.subplots(rowws, 1, sharex=True)
            rows[0].plot(np.nanmean(self.trial_stimuli[grating_angle], axis=0))
            for cell, trials in enumerate(activity['PCNs']['relative']):
                rows[cell+1].imshow(np.vstack(trials), aspect='auto', vmax=vmax1) 
        
        for i, (grating_angle, activity) in enumerate(self.grating_activity.items()):
            rowws=len(activity['Ensembles']['relative'])+1
            fig, rows = plt.subplots(rowws, 1, sharex=True)
            rows[0].plot(np.nanmean(self.trial_stimuli[grating_angle], axis=0))
            for cell, trials in enumerate(activity['Ensembles']['relative']):
                rows[cell+1].imshow(np.vstack(trials), aspect='auto', vmax=vmax1) 
        
                
    def plotting_mean_normalized_dff(self, vmax1=2, vmax2=2):
               
        for i, (grating_angle, activity) in enumerate(self.grating_activity.items()):
            fig, rowws = plt.subplots(2, 1, sharex=True)
            rowws[0].plot(np.nanmean(self.trial_stimuli[grating_angle], axis=0))
            rowws[1].imshow(activity['PCNs']['mean_relative'], aspect='auto')
            
        for i, (grating_angle, activity) in enumerate(self.grating_activity.items()):
            fig, rowws = plt.subplots(2, 1, sharex=True)
            rowws[0].plot(np.nanmean(self.trial_stimuli[grating_angle], axis=0))
            rowws[1].imshow(activity['Ensembles']['mean_relative'], aspect='auto')

     
    def plotting_mean_normalized_dff_rasters(self):


        color= ['blue', 'orange', 'green']
        for stim in self.stimuli_labels:
            stimactivity=self.grating_activity
            degres0=stimactivity[stim]
            self.ensembles[stim]
            # non_ensemble_cells_indexes= [x for x in range(self.cell_number) if x not in self.ensembles_pyhton_corrected[stim]]
            meanensembleactivty=degres0['Ensembles']['mean_relative']
            meannonensembleactivty=degres0['Cell_Outside_Ensemble']['mean_relative']
            meanpcnsactivty=degres0['PCNs']['mean_relative']
            
            rowes=6
            fig, rowes = plt.subplots(rowes, 1, sharex=True)
            rowes[0].plot(np.nanmean(self.trial_stimuli[stim], axis=0))
            rowes[1].imshow(meanensembleactivty, aspect='auto')
            rowes[4].plot(np.nanmean(meanensembleactivty, axis=0), color=color[0], label='Ensemble')
            rowes[5].plot(np.nanmean( meanensembleactivty/np.linalg.norm(meanensembleactivty, axis=1)[:,np.newaxis], axis=0), color=color[0], label='Ensemble')
            rowes[2].imshow(meanpcnsactivty, aspect='auto')
            rowes[4].plot(np.nanmean(meanpcnsactivty, axis=0), color=color[1], label='PCCs')
            rowes[5].plot(np.nanmean(meanpcnsactivty/np.linalg.norm(meanpcnsactivty, axis=1)[:,np.newaxis], axis=0), color=color[1], label='PCCs')
            rowes[3].imshow(meannonensembleactivty, aspect='auto')
            rowes[4].plot(np.nanmean(meannonensembleactivty, axis=0), color=color[2], label='Non Ensemble')
            rowes[5].plot(np.nanmean(meannonensembleactivty/np.linalg.norm(meannonensembleactivty, axis=1)[:,np.newaxis], axis=0), color=color[2], label='Non Ensemble')
        
            rowes[0].set_title('Stimulus')
            rowes[1].set_title('Ensemble')
            rowes[2].set_title('PCCs')
            rowes[3].set_title('Non Ensemble')
            rowes[4].set_title('Mean Activity')
            rowes[5].set_title('Normalized Mean Activity')
            fig.suptitle(stim, fontsize=16)
            fig.set_size_inches(20, 20)
            fig.tight_layout
            rowes[4].legend( loc='upper left')
            rowes[5].legend(loc='upper left')
            
    def plotting_mean_normalized_dff_rasters_uniques(self):
    
    #          self.extra_indexes[stimulus_label]['PCNs'], 
    #          self.extra_indexes[stimulus_label]['Cell_Outside_Ensemble'],
    #          self.extra_indexes[stimulus_label]['Cells_In_Other_Ensembles'],
    #          self.extra_indexes[stimulus_label]['cells_in_no_ensemble'],
    #          self.extra_indexes[stimulus_label]['unique_ensembles_cells_indexes'],
    #          self.extra_indexes[stimulus_label]['unique_PCns']]
    
         color= ['blue', 'orange', 'green']
         for stim in self.stimuli_labels:
             stimactivity=self.grating_activity
             degres0=stimactivity[stim]
             self.ensembles[stim]
             # non_ensemble_cells_indexes= [x for x in range(self.cell_number) if x not in self.ensembles_pyhton_corrected[stim]]
             meanensembleactivty=degres0['unique_ensembles_cells_indexes']['mean_relative']
             meannonensembleactivty=degres0['cells_in_no_ensemble']['mean_relative']
             meanpcnsactivty=degres0['unique_PCns']['mean_relative']
             
             
             
             
             rowes=6
             fig, rowes = plt.subplots(rowes, 1, sharex=True)
             rowes[0].plot(np.nanmean(self.trial_stimuli[stim], axis=0))
             rowes[1].imshow(meanensembleactivty, aspect='auto')
             rowes[4].plot(np.nanmean(meanensembleactivty, axis=0), color=color[0], label='Unique_Ensemble')
             rowes[5].plot(np.nanmean( meanensembleactivty/np.linalg.norm(meanensembleactivty, axis=1)[:,np.newaxis], axis=0), color=color[0], label='Unique_Ensemble')
             rowes[2].imshow(meanpcnsactivty, aspect='auto')
             rowes[4].plot(np.nanmean(meanpcnsactivty, axis=0), color=color[1], label='Unique_PCCs')
             rowes[5].plot(np.nanmean(meanpcnsactivty/np.linalg.norm(meanpcnsactivty, axis=1)[:,np.newaxis], axis=0), color=color[1], label='Unique_PCCs')
             rowes[3].imshow(meannonensembleactivty, aspect='auto')
             rowes[4].plot(np.nanmean(meannonensembleactivty, axis=0), color=color[2], label='Cell_In_No_Ensemble')
             rowes[5].plot(np.nanmean(meannonensembleactivty/np.linalg.norm(meannonensembleactivty, axis=1)[:,np.newaxis], axis=0), color=color[2], label='Cell_In_No_Ensemble')
         
             rowes[0].set_title('Stimulus')
             rowes[1].set_title('Unique_Ensemble')
             rowes[2].set_title('Unique_PCCs')
             rowes[3].set_title('Cell_In_No_Ensemble')
             rowes[4].set_title('Mean Activity')
             rowes[5].set_title('Normalized Mean Activity')
             fig.suptitle(stim, fontsize=16)
             fig.set_size_inches(20, 20)
             fig.tight_layout
             rowes[4].legend( loc='upper left')
             rowes[5].legend(loc='upper left')
             
             
             
    def plotting_ensembles_pattern_completion(self):
        
        graph=self.results['results']['best_model']['structure']
        g=self.results['results']['best_model']['theta']['G']
        phi_names=['00', '01', '10', '11']

        phi_potentials={}
        epsums={}
        for phipot in range(4):
            num_node = graph.shape[0]
            trilgraph = np.tril(graph)
            num_edge = sum(sum(trilgraph))
            edge_list = np.zeros((num_edge,2))
            [edge_list[:,1],edge_list[:,0]] = np.where(trilgraph)
            edge_list=edge_list[np.lexsort((edge_list[:,1], edge_list[:,0]))]
            G_on = np.zeros((num_node,num_node))
            for i in range(num_edge):
                node_1 = edge_list[i,0].astype(int)
                node_2 = edge_list[i,1].astype(int)
                G_on[node_1,node_2] = g[phipot,i]#+G[0,i]-G[1,i]-G[2,i]
                
           
            phi_potentials[phi_names[phipot]]=G_on+G_on.T
            epsums[phi_names[phipot]]=sum(phi_potentials[phi_names[phipot]],0)
            epsums[phi_names[phipot]][sum(graph,1)==0] = np.nan
            fig, axs=plt.subplots(2, sharex=True)
            im=axs[1].imshow(phi_potentials[phi_names[phipot]], aspect='auto', cmap='binary')
            fig.colorbar(im, ax=axs[:])
            axs[0].plot(epsums[phi_names[phipot]])
            axs[0].set_xlabel('Cell' );
            axs[0].set_ylabel('Node Strength Sum' )
            axs[1].set_xlabel('Cell' );
            axs[1].set_ylabel('Cell' )
            axs[0].set_title('Phi nodes:'  + phi_names[phipot])


    
        
        num_stim = self.results['params']['UDF'].shape[1]
        core_crf = self.results['results']['core_crf']
        coords=self.results['params']['coords']
        auc = self.results['results']['auc']
        auc_ens = self.results['results']['auc_ens']
        
        
        good_structure=self.results['results']['best_model']['structure'][:-8,:-8]
        good_edge_potentials=self.results['results']['best_model']['theta']['edge_potentials'][:-8,:-8]
        good_auc=auc[:-8]
        
        time_span=1
        nodesz = 30
        nsmi = 0
        nsma =  1
        aucmi =  0
        aucma =  1
        
        for phipot in phi_names:
            epsum=epsums[phipot]
            epsum2 = (epsum-min(epsum))/(max(epsum)-min(epsum))
            good_epsum2 =epsum2[:-8]
            node_matrix=good_epsum2

            for ii in range(num_stim):
       
                fig, ax = plt.subplots(nrows=1, figsize=(10,6))          
                ax.scatter(node_matrix, good_auc[:,ii], s=nodesz, marker='o', facecolors='none', edgecolors='k')
                ax.scatter(node_matrix[self.results['results']['core_crf'][ii].astype(int)-1],good_auc[self.results['results']['core_crf'][ii].astype(int)-1,ii],  s=nodesz, marker='o', facecolors='b', edgecolors='k')
                ax.scatter(node_matrix[self.results['PCNs'][ii].astype(int)-1],good_auc[self.results['PCNs'][ii].astype(int)-1,ii],  s=nodesz, marker='o', facecolors='r', edgecolors='k')
                
              
                ax.plot([nsmi, nsma],np.mean(auc_ens[ii])*np.array([1,1]),'k--')
                ax.plot([nsmi, nsma],(np.mean(auc_ens[ii])+np.std(auc_ens[ii]))*np.array([1,1]),'--', color='tab:gray')
                ax.plot([nsmi, nsma],(np.mean(auc_ens[ii])-np.std(auc_ens[ii]))*np.array([1,1]),'--', color='tab:gray')
                ax.plot(np.mean(node_matrix)*np.array([1,1]),[aucmi, aucma],'k--');
                # ax.set_xlim([nsmi ,nsma])
                # ax.set_ylim([aucmi, aucma])
                ax.plot((np.mean(node_matrix)+np.std(node_matrix))*np.array([1,1]),[aucmi, aucma],'--',color='tab:gray');
                ax.plot((np.mean(node_matrix)-np.std(node_matrix))*np.array([1,1]),[aucmi, aucma],'--',color='tab:gray');
                ax.set_xlabel('Node Strength: phi' + phipot );
                ax.set_ylabel('AUC Ensemble: ' + str(ii+1))
                ax.set_title('Pattern Completors of Ensemble: '  + str(ii+1))
                X = [(np.mean(node_matrix)-np.std(node_matrix)), (np.mean(node_matrix)+np.std(node_matrix)), (np.mean(node_matrix)+np.std(node_matrix)), (np.mean(node_matrix)-np.std(node_matrix))]
                Y = [(np.mean(auc_ens[ii])-np.std(auc_ens[ii])), (np.mean(auc_ens[ii])-np.std(auc_ens[ii])), (np.mean(auc_ens[ii])+np.std(auc_ens[ii])), (np.mean(auc_ens[ii])+np.std(auc_ens[ii]))]
                
                left, bottom, width, height = (X[0], Y[0], X[1]-X[0], Y[2]-Y[0])
                rect = plt.Rectangle((left, bottom), width, height,
                                      facecolor='tab:gray', alpha=0.25)
                ax.add_patch(rect)
                
                fig.savefig('Node_Strength_phi' + phipot+ '_Ensemble_'+str(ii+1)+".pdf")

                # set2=np.argwhere(node_matrix>np.mean(node_matrix)+np.std(node_matrix))
                # set1=np.argwhere(good_auc[:, ii]>np.mean(auc_ens[ii])+np.std(auc_ens[ii]))
                # good=np.array([x for x in set2.flatten().tolist() if x in set1.flatten().tolist()])
                # good=np.array([1, 42, 44, 147, 168 ])-1
              
                # ax.scatter(node_matrix[good],good_auc[good,ii],  s=nodesz, marker='o', facecolors='r', edgecolors='k')
    
    def plot_auc_curves(self):
        results=self.results['results']
        params=self.results['params']

        numClass = params['UDF'].shape[1]
        true_label =  params['UDF'].T
        gratings=list(range(0,numClass))
        Xcell = list(range(0,numClass))
        LL_cell = list(range(0,numClass))
        
        self.roc_curves={}
        self.precision_recall={}

        self.figures=[]
        for StimNum in gratings:
            
            fig=plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(1, 2, 1)    
            ax2 = fig.add_subplot(1, 2, 2)   
            ax=[ax1,ax2]
            ax[0].plot([0,1],[0,1],'--k')
            fig.suptitle('Grating: '  +  self.stimuli_labels[StimNum])
            # print(ax[0].legend)
            # print(ax[0].get_legend)
            # print(ax[0].get_legend_handles_labels)


            self.roc_curves[self.stimuli_labels[StimNum]]={}
            self.precision_recall[self.stimuli_labels[StimNum]]={}
            self.figures.append(ax)
            for a in Xcell:
                #get LL
                
                ensembleIdx = results['ens_nodes'][a][0][0].astype(int)-1
                ll = results['LL_on'][ensembleIdx,:]
                ll2 = ll.sum(axis=0)
                LL_cell[a]=ll2;
             
                # %find auc
                fpr, tpr, thr = roc_curve(true_label[StimNum,:], ll2, pos_label=1, drop_intermediate=False)
                roc_auc = auc(fpr, tpr)
                # roc_auc2=roc_auc_score(fpr, tpr)
                self.roc_curves[self.stimuli_labels[StimNum]]['Ensemble_'+str(a+1)] = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Ensemble: '+str(a+1))
                self.roc_curves[self.stimuli_labels[StimNum]]['Ensemble_'+str(a+1)].plot(self.figures[StimNum][0], linewidth=0.5)
                
                # def Find_Optimal_Cutoff(target, predicted):
                #     fpr, tpr, threshold = roc_curve(target, predicted, pos_label=1, drop_intermediate=False)
                #     i = np.arange(len(tpr)) 
                #     roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
                #     roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
                
                #     return list(roc_t['threshold']) 
        
                # thresholdss = Find_Optimal_Cutoff(true_label[StimNum,:],ll2)
         
                # test=precision_recall_curve(true_label[StimNum,:], ll2)
                
                self.precision_recall[self.stimuli_labels[StimNum]]['Ensemble_'+str(a+1)]=precision_recall_curve(true_label[StimNum,:], ll2, ) 
                self.figures[StimNum][1].plot( self.precision_recall[self.stimuli_labels[StimNum]]['Ensemble_'+str(a+1)][1][:-1],  self.precision_recall[self.stimuli_labels[StimNum]]['Ensemble_'+str(a+1)][0][:-1], linewidth=0.5, label='Ensemble '+str(a+1))
                
                
            self.figures[StimNum][1].set_xlabel('Recall')
            self.figures[StimNum][1].set_ylabel('Precision')
            self.figures[StimNum][1].set_xlim(left=0, right=1)
            self.figures[StimNum][1].set_ylim(bottom=0)
            # self.figures[StimNum][0].get_legend().remove()
            self.figures[StimNum][0].set_xlim(left=0.00000001, right=1)
            self.figures[StimNum][0].set_ylim(bottom=0)
            fig.savefig( self.stimuli_labels[StimNum]   +".pdf")
            
        
        tuned_ensembles=[None]*8   
        for j, (key, grating) in enumerate(self.precision_recall.items()):
            tuned_ensembles[j]=[]    
            for i, array in enumerate(grating[list(grating.keys())[j]]):
                tuned_ensembles[j].append(array)
                
                
        fig2=plt.figure(figsize=(10,6))
        ax = fig2.add_subplot(1, 1, 1)    
        fig2.suptitle('Precision-Recall Tuned Ensembles')
 
 
        for i in tuned_ensembles:
            ax.plot(i[1], i[0], linewidth=0.5)
            
                
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision') 
        fig2.savefig("TunedRecalPrecisiom.pdf")

  
            

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


    plt.close('all')

    res.plotting_ensembles_pattern_completion()
    # res.plot_auc_curves()





#%%this is for network analysises of crf results
# for a in range(N):
#     for b in range(N):
#         if good_structure[a,b]:
#             s.append(a)
#             t.append(b)
#             b=b+1;
#     a=a+1;
# edge_wt = good_edge_potentials[np.nonzero(good_edge_potentials)]
# normalized_edge_wt=((edge_wt-min(edge_wt))*(1-(-1))/(max(edge_wt)-min(edge_wt)))+(-1)

# test=nx.Graph(incoming_graph_data=list(zip(s,t)))
# nx.set_edge_attributes(test, values = edge_wt.tolist(), name = 'weight')
# test.edges(data = True)

# nx.draw(test, with_labels=True, font_weight='bold')

# net=Network()
# net.from_nx(test)
# net.show('test.html')

# # MDL = graph[s,t,normalized_edge_wt]
# # MDL.Edges.EdgeColors = MDL.Edges.Weight/max(MDL.Edges.Weight);
# # MDL.Edges.LWidths = 1*normalize(MDL.Edges.Weight,'range', [0.01 0.99])/max(normalize(MDL.Edges.Weight,'range',[0.01 0.99]));

#     f = figure; set(gcf,'color','w')
#     f.Name = sprintf('Ensemble %d', ii);
#     f.WindowState = 'maximized';
    
#     subplot(2,2,1)
#     MODEL = plot(MDL,'XData',coords(:,1),'YData',coords(:,2),'ZData',coords(:,3));
#     MODEL.EdgeCData = MDL.Edges.EdgeColors;
#     MODEL.LineWidth = MDL.Edges.LWidths;
#     MODEL.NodeLabel = {};
#     MODEL.Marker='o';
#     xlabel('X Pixels'); ylabel('Y Pixels'); zlabel('Z Pixels');
#     xlim([0 512]);ylim([0 512]);zlim([0 150]);
#     title('Pattern Completors Within Network Functional Connectivity Map')
#     colormap(winter)
#     colorbar
#     highlight(MODEL,[1:length(good_auc)], 'NodeColor',[0 0 0]);
#     highlight(MODEL,[PCNs{ii}],'NodeColor','r');
    

#  subplot(2,2,3:4)
#  plotGraphHighlight(coords,mod(PCNs{ii}-1, num_orig_neuron)+1, 'red', 1);
#  xlabel('X Pixels'); ylabel('Y Pixels'); zlabel('Z Pixels');
#  xlim([0 512]);ylim([0 512]);zlim([0 150]);
#  title('Coordinates of Pattern Completors for Optogenetic Targeting');
#  hold off
# end



