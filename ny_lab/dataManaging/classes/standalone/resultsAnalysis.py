# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:09:58 2021

@author: sp3660
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
# import torch
# import torch.nn as nn
from random import sample, random
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy import signal
from scipy.io import loadmat, savemat
import skimage.io
import math
from caimanSorterYSResults import CaimanSorterYSResults
from cRFsResults import CRFsResults
from voltageSignalsExtractions import VoltageSignalsExtractions
from metadata import Metadata
import os
from sklearn.preprocessing import normalize

import mplcursors
import matplotlib as mlp
# from TestPLot import SnappingCursor
import matplotlib as mpl
import scipy.signal as sig
from numpy import exp, abs, angle
from scipy import stats, interpolate
import scipy.io
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from matplotlib import pyplot as plt
import pandas as pd

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b",'g','y','c','m', 'tab:brown']) 

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'



class ResultsAnalysis():
    
    def __init__(self,  plane1_caiman_sorter_results_object=None, plane2_caiman_sorter_results_object=None,plane3_caiman_sorter_results_object=None, crf_results_object=None, acquisition_voltage_signals_object=None, metadata_object=None ):
        
        self.metadata_object=metadata_object
        self.plane1_results=plane1_caiman_sorter_results_object
        self.plane2_results=plane2_caiman_sorter_results_object
        self.plane3_results=plane3_caiman_sorter_results_object
        self.crf_results=crf_results_object
        self.acquisition_voltage_signals=acquisition_voltage_signals_object
        
        
        [path, file_name]=os.path.split(self.acquisition_voltage_signals.voltage_excel_path)
        noise_correlations_filename=os.path.splitext(file_name)[0]+'_noise_correlations'
        signal_correlations_filename=os.path.splitext(file_name)[0]+'_signal_correlations'
        represent_similarty_matrix_filename=os.path.splitext(file_name)[0]+'_represent_similart'
        self.noise_correlations_to_save_path=os.path.join(path,noise_correlations_filename)
        self.signal_correlations_to_save_path=os.path.join(path,signal_correlations_filename)
        self.represent_similarty_matrix_to_save_path=os.path.join(path,represent_similarty_matrix_filename)
        
        
        
        self.movie_frames= self.plane1_results.C_matrix.shape[1]
       
        self.movie_rate=1/(self.metadata_object.video_params['framePeriods'][0][0]*self.metadata_object.video_params['PlaneNumber']) #hz
        self.milisecond_period=1000/self.movie_rate #ms
       
        self.allplanesraw=np.concatenate((self.plane1_results.raw,
                                    self.plane2_results.raw,
                                    self.plane3_results.raw), axis=0)

        self.allplanesC=np.concatenate((self.plane1_results.C_matrix,
                                    self.plane2_results.C_matrix,
                                    self.plane3_results.C_matrix), axis=0)

        self.allplanesdfdt=np.concatenate((self.plane1_results.dfdt_matrix,
                                      self.plane2_results.dfdt_matrix,
                                      self.plane3_results.dfdt_matrix), axis=0)
        if self.plane1_results.foopsi_matrix.any():
            self.allplanesfoopsi=np.concatenate((self.plane1_results.foopsi_matrix,
                                            self.plane2_results.foopsi_matrix,
                                            self.plane3_results.foopsi_matrix))
        else:
            self.allplanesfoopsi=np.zeros( self.allplanesdfdt.shape)
                    
        if self.plane1_results.MCMC_matrix.any():
            self.allplanesMCMC=np.concatenate((self.plane1_results.MCMC_matrix,
                                            self.plane2_results.MCMC_matrix,
                                            self.plane3_results.MCMC_matrix))
            
        else:
            self.allplanesMCMC=np.zeros( self.allplanesdfdt.shape)
            
        self.cell_number=self.allplanesC.shape[0]
                      
        self.resample_voltage_matrices()
        self.resample_grating_indexes_and_matrix()
        self.slice_borders()
        self.analyze_drifting_gratings()
        self.calculate_windows_and_ranges()
        self.slice_responses_by_gratings()
        self.calculate_df_f()
        self.trial_averaging_and_evoked_activity()
        self.orientation_tuning()
        self.create_stimuli_meshes()
        self.frequency_discriminability()
        self.response_reliability()
        self.signal_noise_correlations()
        
        self.running_modulation()
        self.deal_with_crf_stuff()
        
        
    def resample_voltage_matrices(self):
        
        self.resampled_vistim_matrix=self.resample(self.acquisition_voltage_signals.rounded_vis_stim[:], factor=self.milisecond_period, kind='linear').squeeze()
        self.resampled_speed_matrix=self.resample(self.acquisition_voltage_signals.rectified_speed_array[:], factor=self.milisecond_period, kind='linear').squeeze()
        self.resampled_acceleration_matrix=self.resample(self.acquisition_voltage_signals.acceleration_array    [:], factor=self.milisecond_period, kind='linear').squeeze()    
      
    def resample(self, x, factor, kind='linear'):
        n = int(np.ceil(x.size / factor))
        f = interpolate.interp1d(np.linspace(0, 1, x.size), x, kind)
        return f(np.linspace(0, 1, n))       
            
    def resample_grating_indexes_and_matrix(self):    
        
        
        self.resampled_stim_binary_matrix=np.zeros(( self.acquisition_voltage_signals.full_stimuli_binary_matrix.shape[0],
                                                    self.movie_frames))
        for row in range(self.acquisition_voltage_signals.full_stimuli_binary_matrix.shape[0]) :
            self.resampled_stim_binary_matrix[row,:]=np.ceil(self.resample(self.acquisition_voltage_signals.full_stimuli_binary_matrix[row,:], factor=self.milisecond_period, kind='linear'))

        self.resampled_grating_start_indexes=np.ceil(self.acquisition_voltage_signals.tuning_stim_on_index_full_recording/self.milisecond_period).astype('uint16')
        self.resampled_grating_end_indexes=np.ceil(self.acquisition_voltage_signals.tuning_stim_off_index_full_recording/self.milisecond_period).astype('uint16')
            

        self.ranges_resampled={'resampled_grating_start_indexes':self.resampled_grating_start_indexes,
                               'resampled_grating_end_indexes':self.resampled_grating_end_indexes,
                                   }

        self.activity_matrixes_resampled={'grating_binary_matrix':self.resampled_stim_binary_matrix,
                                  'cmatrix':self.allplanesC,
                                  'dfdtmatrix':self.allplanesdfdt,
                                  'foopsimatrx':self.allplanesfoopsi,
                                  'mcmcmatrix':self.allplanesMCMC,
                                  'rawmatrix':self.allplanesraw
                                   }

        self.voltage_traces_resampled={'vistim_trace':self.resampled_vistim_matrix,
                                  'speed_trace':self.resampled_speed_matrix,
                                  'acceleration_trace':self.resampled_acceleration_matrix,
                                   }
        
    def slice_borders(self):
        
        self.pre_cut=60 # this is bases on the cut I did for crfs, might have to change it, howver fisrt grating is bad
        self.post_cut=1616
        self.border_slice=slice(self.pre_cut-1, self.movie_frames-self.post_cut,1)
        
        self.activity_matrixes_resampled_bordercuts={key: value[:,self.border_slice] for key, value in  self.activity_matrixes_resampled.items()}
        self.voltage_traces_resampled_bordercuts={key:value[self.border_slice] for key, value in  self.voltage_traces_resampled.items()}
        self.ranges_resampled_bordercuts={key:(array-(self.pre_cut+1)) for key, array  in  self.ranges_resampled.items()}
        
    def analyze_drifting_gratings(self):
        

        self.isi_time=1000     #ms
        self.stim_time=2000    #ms
        self.pre_time=350     #ms
        self.post_time=350      #ms
        self.pre_frames=np.ceil(self.pre_time/self.milisecond_period).astype(int)
        self.post_frames=np.ceil(self.post_time/self.milisecond_period).astype(int)
        self.grating_number=self.ranges_resampled_bordercuts['resampled_grating_start_indexes'].shape[0]
        
        self.grating_repetitions=self.ranges_resampled_bordercuts['resampled_grating_start_indexes'].shape[1]
        self.grating_frame_number=np.arange(self.resampled_grating_start_indexes[0,0]-self.pre_frames, self.resampled_grating_end_indexes[0,0]+self.post_frames).size

        self.angles=np.linspace(0,360,9)[:-1]
        self.angle_numbers=len(self.angles)
        self.frequencies=np.array([1,2,4,8,15])
        self.frequency_numbers=len(self.frequencies)

        self.angles_xv, self.frequencies_yv = np.meshgrid(self.angles, self.frequencies)
        self.anglevalues = np.reshape(np.arange(1,41), (5, 8))
    
    def calculate_windows_and_ranges(self):
        
        self.frame_windows={}
        for grat in range( self.grating_number):
            temp=[]
            selected_grating_starts=self.ranges_resampled_bordercuts['resampled_grating_start_indexes'][grat,:]
            selected_grating_ends=self.ranges_resampled_bordercuts['resampled_grating_end_indexes'][grat,:]
            for rept in range( self.grating_repetitions): 
                frame_window=np.arange(selected_grating_starts[rept]-self.pre_frames, selected_grating_ends[rept]+self.post_frames).astype('int64')
                if len(frame_window)!=self.grating_frame_number:
                    # temp.append(np.append(frame_window, frame_window[-1]+1))
                    temp.append(np.insert(frame_window, 0, frame_window[0]-1))
                else:
                    temp.append(frame_window)
            self.frame_windows[grat]=np.vstack(temp)    

        self.ranges={}
        for key, grat in self.frame_windows.items():
            self.ranges[key]=[]
            for trial in grat:
                self.ranges[key].append(range(trial[0],trial[-1]+1))
                
                
        self.extended_ranges={}
        for key, grat in self.frame_windows.items():
            self.extended_ranges[key]=[]
            for trial in grat:
                self.extended_ranges[key].append(range(trial[0]-100,trial[-1]+100))
                
                
        for i in range(self.grating_number):       
            zzz=np.ceil(self.activity_matrixes_resampled_bordercuts['grating_binary_matrix'][i,self.frame_windows[i]])
            zzzz=np.diff(zzz)
            firsidex=np.argwhere(zzzz[:,:11])[:,1]
            tocorrect=np.argwhere(abs(firsidex-np.nanmean(firsidex))==max(abs(firsidex-np.nanmean(firsidex))))
            self.frame_windows[i][tocorrect.flatten()]=self.frame_windows[i][tocorrect.flatten()]-1
          
            # zzz2=np.ceil(activity_matrixes_resampled_bordercuts['grating_binary_matrix'][i,frame_windows[i]])
            # zzzz2=np.diff(zzz2)
            # firsidex2=np.argwhere(zzzz2[:,:11])[:,1]
            # tocorrect2=np.argwhere(abs(firsidex2-np.nanmean(firsidex2))==max(abs(firsidex2-np.nanmean(firsidex2))))

          
    def slice_responses_by_gratings(self):       

        self.grating_sliced_arrays={}
        self.grating_sliced_arrays['dfdtmatrix']=np.zeros((self.grating_repetitions,self.grating_frame_number, self.grating_number, self.cell_number))
        self.grating_sliced_arrays['rawmatrix']=np.zeros((self.grating_repetitions,self.grating_frame_number, self.grating_number, self.cell_number))
        self.grating_sliced_arrays['mcmcmatrix']=np.zeros((self.grating_repetitions,self.grating_frame_number, self.grating_number, self.cell_number))
        
        self.grating_sliced_traces={}
        self.grating_sliced_traces['speed_trace']=np.zeros((self.grating_repetitions,self.grating_frame_number, self.grating_number))
        self.grating_sliced_traces['vistim_trace']=np.zeros((self.grating_repetitions,self.grating_frame_number, self.grating_number))



        self.grating_extended_sliced_arrays={}
        self.grating_extended_sliced_arrays['dfdtmatrix']=np.zeros((self.grating_repetitions,len(self.extended_ranges[0][0]), self.grating_number, self.cell_number))
        self.grating_extended_sliced_arrays['rawmatrix']=np.zeros((self.grating_repetitions,len(self.extended_ranges[0][0]), self.grating_number, self.cell_number))
        self.grating_extended_sliced_arrays['mcmcmatrix']=np.zeros((self.grating_repetitions,len(self.extended_ranges[0][0]), self.grating_number, self.cell_number))
        
        self.grating_extended_sliced_traces={}
        self.grating_extended_sliced_traces['speed_trace']=np.zeros((self.grating_repetitions,len(self.extended_ranges[0][0]), self.grating_number))
        self.grating_extended_sliced_traces['vistim_trace']=np.zeros((self.grating_repetitions,len(self.extended_ranges[0][0]), self.grating_number))


        for key, value in self.grating_sliced_arrays.items():
          for cell in range(0,self.cell_number):
              for grat, trials in   self.frame_windows.items():
                  for row, trial in enumerate(trials):
                      value[row,:,grat,cell]=self.activity_matrixes_resampled_bordercuts[key][cell,trial]
         
        for key, value in self.grating_sliced_traces.items():
          for grat, trials in   self.frame_windows.items():
              for row, trial in enumerate(trials):
                  value[row,:,grat]=self.voltage_traces_resampled_bordercuts[key][trial]  
                  
        self.grating_extended_sliced_arrays={}
        self.grating_extended_sliced_arrays['dfdtmatrix']=np.zeros((self.grating_repetitions,len(self.extended_ranges[0][0]), self.grating_number, self.cell_number))
        self.grating_extended_sliced_arrays['rawmatrix']=np.zeros((self.grating_repetitions,len(self.extended_ranges[0][0]), self.grating_number, self.cell_number))
        self.grating_extended_sliced_arrays['mcmcmatrix']=np.zeros((self.grating_repetitions,len(self.extended_ranges[0][0]), self.grating_number, self.cell_number))
        
        self.grating_extended_sliced_traces={}
        self.grating_extended_sliced_traces['speed_trace']=np.zeros((self.grating_repetitions,len(self.extended_ranges[0][0]), self.grating_number))
        self.grating_extended_sliced_traces['vistim_trace']=np.zeros((self.grating_repetitions,len(self.extended_ranges[0][0]), self.grating_number))
     

        for key, value in self.grating_extended_sliced_arrays.items():
          for cell in range(0,self.cell_number):
              for grat, trials in   self.extended_ranges.items():
                  for row, trial in enumerate(trials):
                      value[row,:,grat,cell]=self.activity_matrixes_resampled_bordercuts[key][cell,trial]
         
        for key, value in self.grating_extended_sliced_traces.items():
          for grat, trials in   self.extended_ranges.items():
              for row, trial in enumerate(trials):
                  value[row,:,grat]=self.voltage_traces_resampled_bordercuts[key][trial]  


    def df_f_trial(self,trial_activity):
    
      prestim_activity=trial_activity[:self.pre_frames+1]
      prestim_mean=np.nanmean(prestim_activity)
      # put something here for std 
      if prestim_mean==0 or prestim_mean<0.002:
          prestim_mean=1
      df_f_trial_activity=(trial_activity-prestim_mean)/ prestim_mean
    
      return df_f_trial_activity
    
    def calculate_df_f(self):

        self.grating_df_f_percentages={}
        self.grating_df_f_percentages['dfdtmatrix']=np.zeros(self.grating_sliced_arrays['dfdtmatrix'].shape);
        self.grating_df_f_percentages['rawmatrix']=np.zeros(self.grating_sliced_arrays['rawmatrix'].shape);
        self.grating_df_f_percentages['mcmcmatrix']=np.zeros(self.grating_sliced_arrays['mcmcmatrix'].shape);
        
        for key, value in  self.grating_df_f_percentages.items():
            for cell in range(self.cell_number):
                for grat in range(self.grating_number):
                    for rept in range(self.grating_repetitions):               
                           value[rept,:,grat,cell]=self.df_f_trial(self.grating_sliced_arrays[key][rept,:,grat,cell])
                           
   
    def trial_averaging_and_evoked_activity(self):
        
        self.grating_df_f_percentages_trial_averaged={}
        self.grating_activity_trial_averaged={}
        self.extended_grating_activity_trial_averaged={}


        for key, value in self.grating_df_f_percentages.items():
            
            self.grating_activity_trial_averaged[key]=np.nanmean(self.grating_extended_sliced_arrays[key],0)
            self.grating_activity_trial_averaged[key]=np.nanmean(self.grating_sliced_arrays[key],0)
            self.grating_df_f_percentages_trial_averaged[key]=np.nanmean(self.grating_df_f_percentages[key],0)

        # shape is  frames, grating, cell
        self.mean_response_per_gratings=np.nanmean(self.grating_activity_trial_averaged['dfdtmatrix'],2).T
        self.mean_reponse_per_cell=np.nanmean(self.grating_activity_trial_averaged['dfdtmatrix'],1).T
        self.mean_df_f_per_gratings=np.nanmean(self.grating_df_f_percentages_trial_averaged['dfdtmatrix'],2).T
        self.mean_df_f_per_cell=np.nanmean(self.grating_df_f_percentages_trial_averaged['dfdtmatrix'],1).T
        self.slice_evoked_activity()
        
        
    def slice_evoked_activity(self):
      self.evoked_slice=slice(self.pre_frames+1,-self.post_frames+2)
      self.evoked_all_activities={}
      for key, value in  self.grating_df_f_percentages.items():
          self.evoked_all_activities[key]={}
          self.evoked_all_activities[key]['evoked_trial_df_f']=self.grating_df_f_percentages[key][:,self.evoked_slice,:,:]
          self.evoked_all_activities[key]['evoked_trial_df_f_mean']=np.nanmean( self.evoked_all_activities[key]['evoked_trial_df_f'],1)
          self.evoked_all_activities[key]['evoked_trial_averaged_df_f']=  self.grating_df_f_percentages_trial_averaged[key][self.evoked_slice,:,:]
          self.evoked_all_activities[key]['evoked_trial_averaged_df_f_mean']=np.nanmean( self.evoked_all_activities[key]['evoked_trial_averaged_df_f'],0).T
          
      self.evoked_all_activities['speed_trace']={}
      self.evoked_all_activities['speed_trace']['evoked_locomotion']=self.grating_sliced_traces['speed_trace'][:,self.evoked_slice,:]
      self.evoked_all_activities['speed_trace']['evoked_locomotion_mean']=np.nanmean(self.evoked_all_activities['speed_trace']['evoked_locomotion'], axis=1)


    def orientation_tuning(self):
        print('doing')
         
        self.prefered_combinations=self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'].argmax(axis=1)
        self.prefered_angles=np.zeros(self.cell_number)
        self.prefered_frequencies=np.zeros(self.cell_number)
        self.prefered_directions=np.zeros(self.cell_number)

        for z, i in enumerate( self.prefered_combinations):
              coord=np.argwhere(self.anglevalues== self.prefered_combinations[z]+1).flatten()
              coord_idx = [self.frequencies_yv[coord[0],0] , self.angles_xv[0,coord[1]]]
              self.prefered_angles[z]=coord_idx[1]
              self.prefered_frequencies[z]=coord_idx[0]

        def ortho(x):
            if x<180:
                return x+90
            elif x>=180:
                return x-90
        vortho=np.vectorize(ortho)
        self.orthogonal_angle=vortho(self.prefered_angles)
        
        def opposite_direction(x):
            if x<180:
                return x+180
            elif x>=180:
                return x-180
        vopposite=np.vectorize(opposite_direction)
        self.opposite_angle=vopposite(self.prefered_angles)
        self.opposite_orthogonal_angle=vopposite(self.orthogonal_angle)

        self.prefered_orientations=np.array(list(zip( self.prefered_angles,  self.opposite_angle)))
        self.orthogonal_orientations=np.array(list(zip( self.orthogonal_angle,  self.opposite_orthogonal_angle)))
        self.prefered_directions= self.prefered_angles
        self.opposite_directions= self.opposite_angle
     
        self.opposite_direction_mean_evoked_df_f=np.zeros((self.cell_number))
        self.prefered_directions_mean_evoked_df_f=np.zeros((self.cell_number))
        self.prefered_orientations_mean_evoked_df_f=np.zeros((self.cell_number))
        self.orthogonal_orientations_mean_evoked_df_f=np.zeros((self.cell_number))

        for i in range(self.cell_number):
    
            pref_angles=np.argwhere(np.logical_or(self.angles_xv==self.prefered_orientations[i][0], self.angles_xv==self.prefered_orientations[i][1]))[0:2,1]
            ortho_angles=np.argwhere(np.logical_or(self.angles_xv==self.orthogonal_orientations[i][0], self.angles_xv==self.orthogonal_orientations[i][1]))[0:2,1]
            angles_prefered=self.anglevalues[:,slice(pref_angles[0], pref_angles[1],3)]
            angles_ortho=self.anglevalues[:,slice(ortho_angles[0], ortho_angles[1],3)]                                       
            self.prefered_orientations_mean_evoked_df_f[i]=     np.nanmean(self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][i, (angles_prefered-1).flatten()])
            self.orthogonal_orientations_mean_evoked_df_f[i]=   np.nanmean(self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][i, (angles_ortho-1).flatten()])
            
            self.prefered_directions_mean_evoked_df_f[i]=       np.nanmean(self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][i, self.anglevalues[:,np.argwhere(self.angles_xv==self.prefered_directions[i])[0][1]]-1])
            self.opposite_direction_mean_evoked_df_f[i]=        np.nanmean(self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][i, self.anglevalues[:,np.argwhere(self.angles_xv==self.opposite_directions[i])[0][1]]-1])
    
        self.osi=np.squeeze((self.prefered_orientations_mean_evoked_df_f-self.orthogonal_orientations_mean_evoked_df_f)/(self.prefered_orientations_mean_evoked_df_f+self.orthogonal_orientations_mean_evoked_df_f))
        self.filtered_osi=self.osi[np.logical_and(self.osi<2,self.osi>0)]
        # filtered_osi=osi[osi>0]
    
        self.dsi=np.squeeze((self.prefered_directions_mean_evoked_df_f-self.opposite_direction_mean_evoked_df_f)/(self.prefered_directions_mean_evoked_df_f+self.opposite_direction_mean_evoked_df_f))
        self.filtered_dsi=self.dsi[np.logical_and(self.dsi<2,self.dsi>0)]
    
    
        self.mean_evoked_df_f_exponential=np.zeros((self.angle_numbers,self.cell_number))
        self.mean_accc=np.zeros((self.angle_numbers,self.cell_number))
        for cell in range(self.cell_number):
            cell_prefered_frequency=self.prefered_frequencies[cell]
            for grat in range(self.angle_numbers):
                angless=self.anglevalues[np.argwhere(self.frequencies_yv==cell_prefered_frequency)[0][0],:]-1
                self.mean_accc[grat,cell]=self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][cell, angless[grat]]
                self.mean_evoked_df_f_exponential[grat,cell]=self.mean_accc[grat,cell]*np.exp(2.j*np.deg2rad(self.angles[grat]))
    
        self.gosi=np.sum(self.mean_evoked_df_f_exponential,0)/np.sum(self.mean_accc,0)
        self.filtered_gosi=abs(self.gosi[(abs(self.gosi))<2])

    def create_stimuli_meshes(self):
        
        self.angles_matrix=np.array([np.deg2rad(self.angles),]*self.cell_number).transpose()
        self.full_angles_matrix=np.array([self.angles_matrix,]*5)

        self.temporal_matrix=np.array([self.frequencies,]*8)
        self.full_temporal_matrix=np.array([self.temporal_matrix,]*self.cell_number).transpose()

    def frequency_discriminability(self):
        
        self.max_temp_frequency_response_mean_evoked_df_f=np.zeros((self.cell_number))
        self.min_temp_frequency_response_mean_evoked_df_f=np.zeros((self.cell_number))
        self.SSE=np.zeros((self.cell_number))

        for i in range(self.cell_number):
            
            angless=self.anglevalues[:,np.argwhere(self.angles_xv==self.prefered_directions[i])[0][0]]-1
    
            self.max_temp_frequency_response_mean_evoked_df_f[i]=self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][i, self.prefered_combinations[i]]
            self.min_temp_frequency_response_mean_evoked_df_f[i]=min(self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][i, angless])
            self.SSE[i]=np.sum(((self.evoked_all_activities['dfdtmatrix']['evoked_trial_df_f_mean'][:,self.prefered_combinations[i],i])-self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][i, self.prefered_combinations[i]])**2)
            
        self.tfdi=(self.max_temp_frequency_response_mean_evoked_df_f-self.min_temp_frequency_response_mean_evoked_df_f)/\
            (self.max_temp_frequency_response_mean_evoked_df_f-self.min_temp_frequency_response_mean_evoked_df_f+2*(np.sqrt(self.SSE/(self.grating_repetitions-self.frequency_numbers))))

    def response_reliability(self):
        
        self.trial_correlations=np.zeros((self.cell_number,self.grating_repetitions, self.grating_repetitions))
        self.reliability=np.zeros((self.cell_number))

        for cell in range(self.cell_number):
            for trial_target in range(self.grating_repetitions):
                for trial_other in range(self.grating_repetitions):
                    self.trial_correlations[cell, trial_target, trial_other]=np.corrcoef( self.evoked_all_activities['dfdtmatrix']['evoked_trial_df_f'][trial_target,:,self.prefered_combinations[cell], cell],
                                                                                            self.evoked_all_activities['dfdtmatrix']['evoked_trial_df_f'][trial_other,:, self.prefered_combinations[cell], cell])[0,1]
            np.fill_diagonal(self.trial_correlations[cell, :, :], 0)
            self.reliability[cell]=np.sum(self.trial_correlations[cell,:,:])

      
    def signal_noise_correlations(self):
        self.noise_correlations=np.zeros(1)
        self.signal_correlations=np.zeros(1)
        self.represent_similarty_matrix=np.zeros(1)

        self.load_correlations()
        
        if not self.signal_correlations.any():
            print('Doing Signal Correlations')
            signal_correlations=np.zeros((self.cell_number,self.cell_number,self.grating_number ))
            for cell_target in range(self.cell_number):
                for cell_other in range(cell_target,self.cell_number):
                    for orient in range(self.grating_number):
                        signal_correlations[cell_target,cell_other,orient]=stats.spearmanr(self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f'][:,orient,cell_target], 
                                                                                                self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f'][:,orient,cell_other]).correlation 
                
            squaredmatrix=  signal_correlations+np.transpose( signal_correlations, (1, 0, 2))  
            for orient in range(self.grating_number):      
                  np.fill_diagonal(squaredmatrix[:,:,orient], 1)    
    
            self.signal_correlations=np.sum(squaredmatrix, axis=2)/self.grating_number
            self.signal_correlations_mean=np.nanmean(squaredmatrix, axis=2)
            np.fill_diagonal(self.signal_correlations, 0)
            np.save(self.signal_correlations_to_save_path, self.signal_correlations)
            plt.imshow(self.signal_correlations)

        if not self.represent_similarty_matrix.any():
            print('Doing Represent Similarity Matrix')

            represent_similarty_matrix_cells=np.zeros((self.grating_number,self.grating_number, self.cell_number))
            for orient in range(self.grating_number):
                for comp_orient in range(orient, self.grating_number):
                    for cell in range(self.cell_number):
       
                        represent_similarty_matrix_cells[orient,comp_orient,cell]=stats.spearmanr(self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f'][:,orient,cell], 
                                                                                                self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f'][:,comp_orient,cell]).correlation   
            squaredmatrix= represent_similarty_matrix_cells+np.transpose(represent_similarty_matrix_cells, (1, 0, 2))  
            for cell in range(self.cell_number):     
                np.fill_diagonal(squaredmatrix[:,:,cell], 1)
            
            self.represent_similarty_matrix_cells=np.sum(squaredmatrix, axis=2)/(self.cell_number)
            np.fill_diagonal(self.represent_similarty_matrix_cells, 0)
            np.save(self.represent_similarty_matrix_to_save_path, self.represent_similarty_matrix_cells)
            plt.imshow(self.represent_similarty_matrix_cells)


        # if not self.noise_correlations.any():
        #     print('Doing Noise Correlations')
        
        #     noise_correlations=np.zeros((self.cell_number,self.cell_number,self.grating_number*self.grating_repetitions ))
        #     for x in range(self.grating_number-1):
        #         if x==0:
        #             concatenated_trial_activity =np.concatenate([self.df_f_percentage_array[:,:,x,:],self.grating_df_f_percentages['dfdtmatrix'][:,:,x+1,:]])
        #         else:
        #             concatenated_trial_activity =np.concatenate([concatenated_trial_activity,self.grating_df_f_percentages['dfdtmatrix'][:,:,x+1,:]])   
        
        #     for cell_target in range(self.cell_number):
        #         for cell_other in  range(cell_target, self.cell_number):
        #             for trial in range(self.grating_number*self.grating_repetitions):   
        #                 noise_correlations[cell_target, cell_other, trial]=stats.spearmanr(concatenated_trial_activity[trial,:,cell_target], 
        #                                                                                        concatenated_trial_activity[trial,:,cell_other]).correlation
           
        #         squaredmatrix= noise_correlations+np.transpose(noise_correlations, (1, 0, 2))  
        #         for trial in range(self.grating_number*self.grating_repetitions):      
        #             np.fill_diagonal(squaredmatrix[:,:,trial], 1)
                
        #         self.noise_correlations=np.sum(squaredmatrix, axis=2)/(self.grating_number*self.grating_repetitions)
        #         np.fill_diagonal(self.noise_correlations, 0)
        #         plt.imshow(self.noise_correlations)

        

        if not os.path.isfile(self.noise_correlations_to_save_path+'.npy'):
            self.save_correlations()
      
        # triang_signal_corelations= signal_corelations[np.triu_indices(corrected_signal_correlations.shape[0])]
        # triang_signal_corelations= triang_signal_corelations[triang_signal_corelations != 1]
        # triang_noise_corelations= noise_corelations[np.triu_indices(corrected_noise_correlations.shape[0])]
        # triang_noise_corelations= triang_noise_corelations[triang_noise_corelations != 1]
      
        # test=np.array(list(zip(triang_signal_corelations,triang_noise_corelations)))
        # plt.scatter(test[:,1],test[:,0])
      
        # plt.imshow(1-scaled_diagonal_represnet_similarty_matrix,aspect='auto')
        


    def save_correlations(self):
       
        np.save(self.noise_correlations_to_save_path, self.noise_correlations)
        np.save(self.signal_correlations_to_save_path, self.signal_correlations)
        np.save(self.represent_similarty_matrix_to_save_path, self.represent_similarty_matrix)

    def load_correlations(self):
        if os.path.isfile(self.noise_correlations_to_save_path+'.npy'):
            self.noise_correlations=np.load(self.noise_correlations_to_save_path+'.npy')
        if os.path.isfile(self.signal_correlations_to_save_path+'.npy'):
            self.signal_correlations=np.load(self.signal_correlations_to_save_path+'.npy')
        if os.path.isfile(self.represent_similarty_matrix_to_save_path+'.npy'):
            self.represent_similarty_matrix=np.load(self.represent_similarty_matrix_to_save_path+'.npy')


    def running_modulation(self):
        self.running_threshold=np.mean(self.voltage_traces_resampled_bordercuts['speed_trace'])+1.5*np.std(self.voltage_traces_resampled_bordercuts['speed_trace'])
        c_constant= self.evoked_all_activities['speed_trace']['evoked_locomotion_mean'].copy()
        self.c_constant=np.where(c_constant>self.running_threshold, 1, -1)
        self.running_modulation_matrix=np.zeros((self.cell_number, self.grating_number, self.grating_repetitions))
        for cell in range(self.cell_number):
            for orient in range(self.grating_number):
                max_index=np.argmax(self.evoked_all_activities['dfdtmatrix']['evoked_trial_df_f_mean'][:,orient,cell])
                min_index=np.argmin(self.evoked_all_activities['dfdtmatrix']['evoked_trial_df_f_mean'][:,orient,cell])
                if np.logical_or((len(np.where(self.c_constant[:, orient]==1))/self.grating_repetitions )>0.9, len(np.where(self.c_constant[:, orient]==1))/self.grating_repetitions <0.1 ):
                    self.running_modulation_matrix[cell, orient, :]=0
                else:
                    for trial in range(self.grating_repetitions):         
                       self.running_modulation_matrix[orient, trial]=c_constant[orient,trial]*((self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][max_index]-self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][min_index])/abs(self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][min_index]))

        # self.maxmin_matrix=np.array(self.cell_number, self.grating_number, 2)) 


    def deal_with_crf_stuff(self):
        print('somethinf')
        # 'grating_crf_matrix':analysis.crf_results.stimulus_matrix,
        # 'binary_activity_crf_matrix':analysis.crf_results.activity_matrix,
        # 'rates_activity_crf_matrix':analysis.crf_results.all_frame_rates,
        
        # 'grating_crf_matrix':analysis.crf_results.stimulus_matrix,
        # 'binary_activity_crf_matrix':analysis.crf_results.activity_matrix,
        # 'rates_activity_crf_matrix':analysis.crf_results.all_frame_rates,
        
  
        # redoing dfdt matrix used in crfs perfer the binary matrix might have been  1 index displaced
        # trace=self.activity_matrixes_resampled['dfdtmatrix']
        # zzz=np.sqrt(np.nanmean(np.square(trace), axis=1))
        # z_thresh=2*zzz
        
        # thresholdeddfdt=np.zeros(trace.shape)
        # plt.figure()
        # plt.imshow(trace, vmax=0.5, aspect='auto', cmap='binary')
        # for i, cell in enumerate(trace):
        #     thresholdeddfdt[i,:]=np.where(cell<z_thresh[i], 0, 1 ).astype(int)
        # plt.figure()
        # plt.imshow(thresholdeddfdt, aspect='auto', cmap='binary')
        # plt.imshow(self.activity_matrixes_resampled['binary_activity_crf_matrix'], aspect='auto', cmap='binary')
        # np.array_equal(thresholdeddfdt, self.activity_matrixes_resampled['binary_activity_crf_matrix'])
        
        # test1=activity_matrixes_resampled_bordercuts['grating_binary_matrix'][9,:]
        # test2=activity_matrixes_resampled['grating_crf_matrix'][1,:]
        # fig, ax=plt.subplots(1)
        # ax.plot(test1-test2)
        # np.array_equal(test1, test2)
        
        # #%% crf activity
        #     stimuli_labels=analysis.crf_results.stimuli_labels
        #     stim_ranges=analysis.crf_results.stim_ranges
        #     cell_number=activity_matrixes_resampled['cmatrix'].shape[0]
           
        #     response_all_cells={}
        #     for cell  in range(cell_number):
        #           response_all_cells[cell]={}
        #           for grating, trials in stim_ranges.items():
        #               response_all_cells[cell][grating]={}
        #               for trial_number, trial in enumerate(trials):
        #                   response_all_cells[cell][grating][trial_number]=activity_matrixes_resampled['cmatrix'][cell, trial]
        
        # CRF deep
        
        # crfres=analysis.crf_results.results
        # plt.imshow(crfres['results']['best_model']['structure'])
        # plt.imshow(crfres['results']['best_model']['theta']['edge_potentials'])
        # plt.imshow(crfres['results']['best_model']['theta']['G'])
        # plt.imshow(crfres['results']['best_model']['theta']['F'])
        # plt.imshow(crfres['results']['best_model']['ep_on'])

        # plt.imshow(crfres['results']['auc'],aspect='auto')
        # plt.plot(crfres['results']['epsum'])
        # plt.imshow(crfres['results']['LL_on'],aspect='auto')
        # plt.imshow(crfres['results']['LL_frame'][:,:,0],aspect='auto')
        # plt.imshow(crfres['results']['LL_frame'][:,:,1],aspect='auto')


        if self.crf_results:
            self.PCNS=self.crf_results.PatternCompletionCells_pyhton_corrected
            self.ensembles=self.crf_results.ensembles_pyhton_corrected
    
        
    
    # plotting
   
    def plot_selectivity_histograms(self):
        
        fig, axss=plt.subplots(3)      

        axss[0].hist(self.gosi)
        axss[1].hist(self.osi)
        axss[2].hist(self.dsi)
        
        fig, axss=plt.subplots(3, sharex=True)  
        axss[0].hist(self.filtered_gosi)
        axss[1].hist(self.filtered_osi)
        axss[2].hist(self.filtered_dsi)
        
        fig, axss=plt.subplots(1, sharex=True) 
        axss.hist(self.tfdi)


    def plotting_corrected_ranges(self):
        
        for i in range(self.grating_number):       
            fig, axss=plt.subplots( self.grating_repetitions, sharex=True)
            for j in range(self.grating_repetitions):               
                # axss[j].plot(activity_matrixes_resampled_bordercuts['grating_binary_matrix'][i,frame_windows[i][j]]) 
                axss[j].plot(self.voltage_traces_resampled_bordercuts['vistim_trace'][self.frame_windows[i][j]]) 
            fig.suptitle('Grating : '  + str(i+1))
        

        fig, axs=plt.subplots(3, sharex=True)
        axs[1].imshow(self.activity_matrixes_resampled_bordercuts['dfdtmatrix'],   
                      vmax=np.nanmean(self.activity_matrixes_resampled_bordercuts['dfdtmatrix'])+2*np.std(self.activity_matrixes_resampled_bordercuts['dfdtmatrix']),     aspect='auto', cmap='binary') 
        axs[0].plot(self.voltage_traces_resampled_bordercuts['vistim_trace'])    
        axs[2].plot(self.voltage_traces_resampled_bordercuts['speed_trace'])   
        for i in range(40):
            for j in range(15):
                color = tuple(np.random.choice(range(256), size=3)/256)
                axs[0].plot(self.ranges[i][j], self.voltage_traces_resampled_bordercuts['vistim_trace'][self.frame_windows[i][j]],'x', color=color) 
     
            fig, axss=plt.subplots(1, sharex=True)
            for j in range(15):
                axss.plot(self.voltage_traces_resampled_bordercuts['vistim_trace'][self.frame_windows[i][j]])  
        

    
    def plot_activity_matrixes(self):
        fig, ax = plt.subplots(4, 1, sharex=True)
        # norm1=mlp.colors.Normalize(0, 1)
        ax[0].imshow(self.allplanesC,     vmax=np.nanmean(self.allplanesC)+2*np.std(self.allplanesC),     aspect='auto', cmap='binary')
        ax[1].imshow(self.allplanesdfdt,  vmax=np.nanmean(self.allplanesdfdt)+2*np.std(self.allplanesdfdt),     aspect='auto', cmap='binary')
        ax[2].imshow(self.allplanesfoopsi,vmax=np.nanmean(self.allplanesfoopsi)+2*np.std(self.allplanesfoopsi),     aspect='auto', cmap='binary')
        ax[3].imshow(self.allplanesMCMC,  vmax=np.nanmean(self.allplanesMCMC)+2*np.std(self.allplanesMCMC),     aspect='auto', cmap='binary')

    def plot_activity_matrix_with_signals(self, activity_matrix):
        
        fig, axs=plt.subplots(3, sharex=True)
        axs[1].imshow(activity_matrix,   vmax=np.nanmean(activity_matrix)+2*np.std(activity_matrix),     aspect='auto', cmap='binary') 
        axs[0].plot(self.resampled_vistim_matrix)    
        axs[2].plot(self.resampled_speed_matrix)   
        for i in range(40):
            color = tuple(np.random.choice(range(256), size=3)/256)
            axs[0].plot(np.argwhere(self.resampled_stim_binary_matrix[i,:]) , self.resampled_vistim_matrix[np.argwhere(self.resampled_stim_binary_matrix[i,:])],'x', color=color) 
    
    def plotting2(self):
        fig, axs = plt.subplots(3)
        fig.suptitle('Locomotion')
        axs[0].plot(self.acquisition_voltage_signals.second_scale, self.acquisition_voltage_signals.rectified_speed_array)
        axs[1].plot(self.acquisition_voltage_signals.second_scale, self.acquisition_voltage_signals.voltage_signals['VisStim'])
        axs[2].plot(self.acquisition_voltage_signals.second_scale, self.acquisition_voltage_signals.voltage_signals['PhotoDiode'])
        mplcursors.cursor(axs) # or just mplcursors.cursor()
        
    def plot_activity_matrix_with_signals_and_grating_ranges(self, activity_matrix=None):
        
        fig, axs=plt.subplots(3, sharex=True)
        axs[1].imshow(self.activity_matrixes_resampled_bordercuts['dfdtmatrix'],   
                      vmax=np.nanmean(self.activity_matrixes_resampled_bordercuts['dfdtmatrix'])+2*np.std(self.activity_matrixes_resampled_bordercuts['dfdtmatrix']),     aspect='auto', cmap='binary') 
        axs[0].plot(self.voltage_traces_resampled_bordercuts['vistim_trace'])    
        axs[2].plot(self.voltage_traces_resampled_bordercuts['speed_trace'])   
        for i in range(40):
            color = tuple(np.random.choice(range(256), size=3)/256)
            axs[0].plot(np.argwhere(self.activity_matrixes_resampled_bordercuts['grating_binary_matrix'][i,:]) ,
                        self.voltage_traces_resampled_bordercuts['vistim_trace'][np.argwhere(self.activity_matrixes_resampled_bordercuts['grating_binary_matrix'][i,:])],'x', color=color) 
        
    def plotting_resampling_accuracy_of_ranges(self):
        # fig, axs = plt.subplots(1,1)
        # axs.imshow(self.gratin1_periods_only,vmin=0, vmax=0.1,aspect='auto',cmap='inferno')
                 
        for stim in range( self.grating_number):
             fig, axs=plt.subplots(2, sharex=True)
             axs[0].plot(self.activity_matrixes_resampled_bordercuts['grating_binary_matrix'][stim,:].T) 
             axs[1].plot(self.voltage_traces_resampled_bordercuts['vistim_trace']) 
             color = tuple(np.random.choice(range(256), size=3)/256)
             axs[0].plot( self.ranges_resampled_bordercuts['resampled_grating_start_indexes'][stim,:]+1, 
                           self.activity_matrixes_resampled_bordercuts['grating_binary_matrix'][stim,self.ranges_resampled_bordercuts['resampled_grating_start_indexes'][stim,:]+1]      ,'x', color=color)      
             axs[0].plot( self.ranges_resampled_bordercuts['resampled_grating_start_indexes'][stim,:]+2, 
                           self.activity_matrixes_resampled_bordercuts['grating_binary_matrix'][stim,self.ranges_resampled_bordercuts['resampled_grating_start_indexes'][stim,:]+2]      ,'o', color=color) 
             axs[1].plot( self.ranges_resampled_bordercuts['resampled_grating_start_indexes'][stim,:]+1, 
                           self.voltage_traces_resampled_bordercuts['vistim_trace'][self.ranges_resampled_bordercuts['resampled_grating_start_indexes'][stim,:]+1]      ,'x', color=color)      
             axs[1].plot( self.ranges_resampled_bordercuts['resampled_grating_start_indexes'][stim,:]+2, 
                           self.voltage_traces_resampled_bordercuts['vistim_trace'][self.ranges_resampled_bordercuts['resampled_grating_start_indexes'][stim,:]+2]      ,'o', color=color)    
             
    def more_plotting(self, cell, grating, trial):
         
         fig, axs = plt.subplots(nrows=3, sharex=True)          
         # one cell response to 15 trial of single gratin
         axs[0].imshow(self.grating_sliced_arrays['dfdtmatrix'][:,:,grating,cell],   vmax=np.nanmean(self.grating_sliced_arrays['dfdtmatrix'])+2*np.std(self.grating_sliced_arrays['dfdtmatrix']),     aspect='auto', cmap='binary')
         # response of all cells to a single repetion of s sigle stoimuli
         axs[1].imshow(self.grating_sliced_arrays['dfdtmatrix'][trial,:,grating,:].T, vmax=np.nanmean(self.grating_sliced_arrays['dfdtmatrix'])+2*np.std(self.grating_sliced_arrays['dfdtmatrix']),     aspect='auto', cmap='binary')
         # response a cell to a single repetition of all stimuli
         axs[2].imshow(self.grating_sliced_arrays['dfdtmatrix'][trial,:,:,cell],   vmax=np.nanmean(self.grating_sliced_arrays['dfdtmatrix'])+2*np.std(self.grating_sliced_arrays['dfdtmatrix']),     aspect='auto', cmap='binary')

         #trial averaged activctu for all cells  and for a single cell

         fig, axs = plt.subplots(nrows=2,  sharex=True)            
         axs[0].imshow(self.trial_averaged_activity[:,grating,:].T, vmax=np.nanmean(self.trial_averaged_activity)+2*np.std(self.trial_averaged_activity),     aspect='auto', cmap='binary')
         axs[1].plot(self.trial_averaged_activity[:,grating,cell])


         fig, axs = plt.subplots(nrows=2)            
         axs[0].imshow(self.mean_response_per_gratings, vmax=np.nanmean(self.mean_response_per_gratings)+2*np.std(self.mean_response_per_gratings),     aspect='auto', cmap='binary')
         axs[1].imshow(self.mean_reponse_per_cell,  vmax=np.nanmean(self.mean_reponse_per_cell)+2*np.std(self.mean_reponse_per_cell),     aspect='auto', cmap='binary')


    def plot_all_tuning_single_cell(self, celltoplot):
                        
           
            for gratingtoplot in range(self.grating_number):
                  
                fig, ax =plt.subplots(6,sharex=True)
                ax[0].imshow(self.grating_sliced_arrays['dfdtmatrix'][:,:,gratingtoplot,celltoplot], 
                             vmax=np.nanmean(self.grating_sliced_arrays['dfdtmatrix'][:,:,gratingtoplot,celltoplot])+2*np.std(self.grating_sliced_arrays['dfdtmatrix'][:,:,gratingtoplot,celltoplot]), aspect='auto',cmap='binary')
                ax[1].plot(np.nanmean(self.grating_sliced_arrays['dfdtmatrix'], axis=0)[:,gratingtoplot,celltoplot])
            
                ax[2].imshow(self.df_f_percentage_array[:,:,gratingtoplot,celltoplot],   
                             vmax=np.nanmean(self.df_f_percentage_array[:,:,gratingtoplot,celltoplot])+2*np.std(self.df_f_percentage_array[:,:,gratingtoplot,celltoplot]),        aspect='auto',cmap='binary')
                ax[3].plot(np.nanmean(self.df_f_percentage_array, axis=0)[:,gratingtoplot,celltoplot])
                ax[4].plot(np.nanmean(self.grating_sliced_traces['vistim_trace'], axis=0)[:,gratingtoplot])
                ax[5].plot(np.nanmean(self.grating_sliced_traces['speed_trace'], axis=0)[:,gratingtoplot])
                fig.suptitle('Grating : '  + str(gratingtoplot))

                
    def plot_trial_averaged_single_cell(self, celltoplot):
                        
            for gratingtoplot in range(self.grating_number):        

                fig, ax =plt.subplots(3)
                ax[0].imshow( self.grating_df_f_percentages_trial_averaged['dfdtmatrix'][:,gratingtoplot,:].T, 
                             vmax=np.nanmean( self.grating_df_f_percentages_trial_averaged['dfdtmatrix'][:,gratingtoplot,:])+2*np.std( self.grating_df_f_percentages_trial_averaged['dfdtmatrix'][:,gratingtoplot,:]),   aspect='auto',cmap='binary')
            
                ax[1].imshow(self.mean_df_f_per_gratings[25:30,:],     
                              aspect='auto',cmap='binary')
                ax[2].imshow(self.mean_df_f_per_cell,       
                              vmax=np.nanmean(self.mean_df_f_per_cell)+2*np.std(self.mean_df_f_per_cell),   aspect='auto',cmap='binary')
                fig.suptitle('Grating trial averaged : '  + str(gratingtoplot))

    def plot_evoked_tuning_single_cell(self, celltoplot):
        
        
        for z, i in enumerate(self.prefered_combinations):
            coord=np.argwhere(self.anglevalues==self.prefered_combinations[z]+1).flatten()
            coord_idx = [self.frequencies_yv[coord[0],0] , self.angles_xv[0,coord[1]]]

        for gratingtoplot in range(self.grating_number):   

            fig, ax =plt.subplots(4, sharex=True)
            ax[0].imshow(self.evoked_all_activities['dfdtmatrix']['evoked_trial_df_f'][:,:,gratingtoplot,celltoplot],
                         vmax=np.nanmean(self.evoked_all_activities['dfdtmatrix']['evoked_trial_df_f'][:,:,gratingtoplot,celltoplot])+2*np.std(self.evoked_all_activities['dfdtmatrix']['evoked_trial_df_f'][:,:,gratingtoplot,celltoplot]),   aspect='auto',cmap='binary')
            ax[1].plot( self.grating_df_f_percentages_trial_averaged['dfdtmatrix'][self.evoked_slice,gratingtoplot,celltoplot])
            ax[2].plot(np.nanmean(self.grating_sliced_traces['speed_trace'], axis=0)[self.evoked_slice,gratingtoplot])
            ax[3].plot(np.nanmean(self.grating_sliced_traces['vistim_trace'], axis=0)[self.evoked_slice,gratingtoplot])
            fig.suptitle('Grating Evoked : '  + str(gratingtoplot))

    def pcns_plotting(self):
        # analysis.PCNS

        grats=[8,9,10,11,12,13,14,15]
        for kk, (grating, cells) in enumerate(self.PCNS.items()):
            grat=grats[kk]
            fig1, ax1 =plt.subplots(cells.shape[0]+2, sharex=True, figsize=(10,6))
            fig1.suptitle('DFDT_DF_F:'  + str(grating))
            fig2, ax2 =plt.subplots(cells.shape[0]+2, sharex=True, figsize=(10,6))
            fig2.suptitle('RAW:' + str(grating))
            fig3, ax3 =plt.subplots(cells.shape[0]+2, sharex=True, figsize=(10,6))
            fig3.suptitle('DFDT:' + str(grating))
            fig4, ax4 =plt.subplots(cells.shape[0]+2, sharex=True, figsize=(10,6))
            fig4.suptitle('MCMC:' + str(grating))
            fig5, ax5 =plt.subplots(cells.shape[0]+2, sharex=True, figsize=(10,6))
            fig5.suptitle('RAW_DF_F' + str(grating))
            fig6, ax6 =plt.subplots(cells.shape[0]+2, sharex=True, figsize=(10,6))
            fig6.suptitle('MCMC-DF_F:' + str(grating))



            for j, cell in enumerate(cells):
                ax1[j].plot( self.grating_df_f_percentages_trial_averaged['dfdtmatrix'][:,grat,cell])
                ax1[j].set_title('Cell '+ str(cell))
                ax2[j].plot(  self.grating_activity_trial_averaged['rawmatrix'][:,grat,cell])
                ax2[j].set_title('Cell '+ str(cell))
                ax3[j].plot(  self.grating_activity_trial_averaged['dfdtmatrix'][:,grat,cell])
                ax3[j].set_title('Cell '+ str(cell))
                ax4[j].plot( self.grating_activity_trial_averaged['mcmcmatrix'][:,grat,cell])
                ax4[j].set_title('Cell '+ str(cell))
                ax5[j].plot( self.grating_df_f_percentages_trial_averaged['rawmatrix'][:,grat,cell])
                ax5[j].set_title('Cell '+ str(cell))
                ax6[j].plot( self.grating_df_f_percentages_trial_averaged['mcmcmatrix'][:,grat,cell])
                ax6[j].set_title('Cell '+ str(cell))





            ax1[-2].plot(np.mean(self.grating_sliced_traces['speed_trace'], axis=0)[:,grat])
            ax1[-1].plot(np.mean(self.grating_sliced_traces['vistim_trace'], axis=0)[:,grat])
            ax2[-2].plot(np.mean(self.grating_sliced_traces['speed_trace'], axis=0)[:,grat])
            ax2[-1].plot(np.mean(self.grating_sliced_traces['vistim_trace'], axis=0)[:,grat])
            ax3[-2].plot(np.mean(self.grating_sliced_traces['speed_trace'], axis=0)[:,grat])
            ax3[-1].plot(np.mean(self.grating_sliced_traces['vistim_trace'], axis=0)[:,grat])
            ax4[-2].plot(np.mean(self.grating_sliced_traces['speed_trace'], axis=0)[:,grat])
            ax4[-1].plot(np.mean(self.grating_sliced_traces['vistim_trace'], axis=0)[:,grat])
            ax5[-2].plot(np.mean(self.grating_sliced_traces['speed_trace'], axis=0)[:,grat])
            ax5[-1].plot(np.mean(self.grating_sliced_traces['vistim_trace'], axis=0)[:,grat])
            ax6[-2].plot(np.mean(self.grating_sliced_traces['speed_trace'], axis=0)[:,grat])
            ax6[-1].plot(np.mean(self.grating_sliced_traces['vistim_trace'], axis=0)[:,grat])
            
            
            fig1.savefig('DFDT_DF_F_'  + str(grating)+".pdf")
            fig2.savefig('RAW_'  + str(grating)+".pdf")
            fig3.savefig('DFDT_'  + str(grating)+".pdf")
            fig4.savefig('MCMC_'  + str(grating)+".pdf")
            fig5.savefig('RAW_DF_F_'  + str(grating)+".pdf")
            fig6.savefig('MCMC_DF_F_'  + str(grating)+".pdf")


            
            
 
        # self.self.grating_sliced_arrays['dfdtmatrix']
        # self.grating_sliced_arrays['rawmatrix']
        #self.grating_df_f_percentages['dfdtmatrix']
        
      



    def polar_plot(self, cell) :

        single_cell_area=self.evoked_all_activities['dfdtmatrix']['evoked_trial_df_f_mean'][:,:,cell]
        single_cell_area_reshaped=np.reshape(single_cell_area,(self.grating_repetitions,self.frequency_numbers,   self.angle_numbers))
        single_cell_angle=self.full_angles_matrix[:,:,cell]
        single_cell_radius=self.full_temporal_matrix[:,:,cell]
        
        colors=single_cell_angle
        fig= plt.figure()
        ax = fig.add_subplot(projection='polar')
        for i in range(self.grating_repetitions):
            noise=np.reshape(np.random.normal(0,0.1,40),(5,8,))
            ax.scatter(single_cell_angle +noise, 
                       single_cell_radius +noise, 
                       single_cell_area_reshaped[i,:,:]*20, edgecolors='k',c=colors, cmap='hsv')
            
            
        textstr = '\n'.join((
            # 'pAngle={}°'.format(prefered_angle[cell]),
            'OSI={}'.format(np.round(self.osi[cell],1)),
            'gOSI={}'.format(np.round(self.gosi[cell],1)),
            'DSI={}'.format(np.round(self.dsi[cell],1))
            ))
      
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0, 1.12, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
 
        textst2 = '\n'.join((
            # 'pAngle={}°'.format(prefered_angle[cell]),
            'PreferedAngle={}'.format(np.round(self.prefered_angles[cell],1)),
            'PreferedFrequency={}'.format(np.round(self.prefered_frequencies[cell],1))
            ))
        
        props2 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.6, 1.12, textst2, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props2)
        
    def polar_plot_single_temporal(self, cell) :   
        
        single_cell_angle=self.full_angles_matrix[:,:,cell]        
        test=self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][cell,:]

        idx=np.zeros((8,5)) 
        for i in range(8):
            idx[i,:]=np.arange(i,40,8)
        idx=idx.astype(int)
            
        
        meantofreq=np.zeros((8)) 

        for i in range(8) :
        
            meantofreq[i]=np.mean(test[idx[i,:]])
        correctedmeantofreq=np.append(meantofreq, meantofreq[0])
        
        only2hzidx=idx[:,1]
        only2hz=test[only2hzidx]
        cell_angle=single_cell_angle[0,:]
        correctedonly2hz=np.append(only2hz, only2hz[0])
        correcetdcell_angle=np.append(cell_angle, cell_angle[0])

        orientationidx=np.zeros((4,2)) 
        for i in range(4):
            orientationidx[i,:]=np.arange(i,8,4)
        orientationidx=orientationidx.astype(int)
        
        
        orienttuning=np.zeros(4)
        for i in range(4):
                orienttuning[i]=np.mean(only2hz[orientationidx[i]])

        
        fig= plt.figure()
        ax = fig.add_subplot(projection='polar')
        ax.plot( np.append(cell_angle[::2], cell_angle[0]), 
                       np.append(orienttuning, orienttuning[0]), color='k')
        ax.yaxis.grid(False)
        ax.set_yticklabels([])
        
        

        fig= plt.figure()
        ax = fig.add_subplot(projection='polar')
        ax.plot(correcetdcell_angle, 
                       correctedonly2hz, color='k')
        ax.yaxis.grid(False)
        ax.set_yticklabels([])

        fig= plt.figure()
        ax = fig.add_subplot(projection='polar')
        
        ax.plot(correcetdcell_angle, 
                       correctedmeantofreq, color='k')
        ax.yaxis.grid(False)
        ax.set_yticklabels([])
        
        fig= plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.bar(list(range(8)), only2hz, width=0.2)
        angles=['0','45','90','135','180','225','270','315']
        plt.xticks(list(range(8)), angles)

    
    def polar_plot_multiplecells_temporal(self, cells, grating_name='') :   

         fig4= plt.figure(figsize=(10,6))
         ax4 = fig4.add_subplot(projection='polar')
         ax4.yaxis.grid(False)
         ax4.set_yticklabels([])
         fig4.suptitle('Orientation Tuning Averaged TF'+grating_name)
         
         fig1= plt.figure(figsize=(10,6))
         ax1 = fig1.add_subplot(projection='polar')
         ax1.yaxis.grid(False)
         ax1.set_yticklabels([])
         fig1.suptitle('Orientation Tuning 2Hz'+grating_name)

         fig2= plt.figure(figsize=(10,6))
         ax2 = fig2.add_subplot(projection='polar')      
         ax2.yaxis.grid(False)
         ax2.set_yticklabels([])
         fig2.suptitle('Direction Tuning 2Hz'+grating_name)

         fig3= plt.figure(figsize=(10,6))
         ax3 = fig3.add_subplot(projection='polar')
         ax3.yaxis.grid(False)
         ax3.set_yticklabels([])
         fig3.suptitle('Direction Tuning Averaged TF'+grating_name)
         
 
         for cell in cells:
        
             single_cell_angle=self.full_angles_matrix[:,:,cell]        
             test=self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][cell,:]

             idx=np.zeros((8,5)) 
             for i in range(8):
                 idx[i,:]=np.arange(i,40,8)
             idx=idx.astype(int)
                 
             
             meantofreq=np.zeros((8)) 
             for i in range(8) :
                 meantofreq[i]=np.mean(test[idx[i,:]])
             correctedmeantofreq=np.append(meantofreq, meantofreq[0])
             
             only2hzidx=idx[:,1]
             only2hz=test[only2hzidx]
             cell_angle=single_cell_angle[0,:]
             correctedonly2hz=np.append(only2hz, only2hz[0])
             correcetdcell_angle=np.append(cell_angle, cell_angle[0])

             orientationidx=np.zeros((4,2)) 
             for i in range(4):
                 orientationidx[i,:]=np.arange(i,8,4)
             orientationidx=orientationidx.astype(int)
             orienttuningaveraged=np.zeros(4)
             orienttuning=np.zeros(4)
             for i in range(4):
                    orienttuning[i]=np.mean(only2hz[orientationidx[i]])
                    orienttuningaveraged[i]=np.mean(meantofreq[orientationidx[i]])

             ax1.plot( np.append(cell_angle[::2], cell_angle[0]), 
                           np.append(orienttuning, orienttuning[0]), label='PCCs'+str(cell+1))
            
             ax2.plot(correcetdcell_angle, 
                           correctedonly2hz, label='PCCs'+str(cell+1))
             
             ax3.plot(correcetdcell_angle, 
                            correctedmeantofreq, label='PCCs'+str(cell+1))

             ax4.plot( np.append(cell_angle[::2], cell_angle[0]), 
                           np.append(orienttuningaveraged, orienttuningaveraged[0]),label='PCCs'+str(cell+1))
             ax1.legend()
             ax2.legend()
             ax3.legend()
             ax4.legend()


            

             fig= plt.figure()
             ax = fig.add_subplot(1,1,1)
             ax.bar(list(range(8)), only2hz, width=0.2)
             angles=['0','45','90','135','180','225','270','315']
             
             
             fig1.savefig('PCCs Orientation Tuning 2Hz'+grating_name+".pdf")
             fig2.savefig('PCCs Direction Tuning 2Hz'+grating_name+".pdf")
             fig3.savefig('PCCs Direction Tuning Averaged TF'+grating_name+".pdf")
             fig4.savefig('PCCs Orientation Tuning Averaged TF'+grating_name+".pdf")


    def polar_plot_auc(self) :   

        self.crf_results.plot_auc_curves()
        cell_angle=self.full_angles_matrix[:,:,0][0,:]
        fig= plt.figure(figsize=(10,6))
        ax = fig.add_subplot(projection='polar')
        ax.yaxis.grid(False)
        ax.set_yticklabels([])
        fig.suptitle('Ensembles AUC Tuning')

        aucs=np.zeros((8,8))
        for i,(key, grating) in enumerate(analysis.crf_results.roc_curves.items()):
   
            for j, (key2, ensemble) in enumerate(grating.items()):
           
                aucs[i][j]=ensemble.roc_auc 
        for k in range(8) :   
            ax.plot( np.append(cell_angle, cell_angle[0]), 
                      np.append(aucs[:,k], aucs[:,k][0]), label='Ensemble'+str(k+1))
        ax.legend()
       
        fig.savefig('Ensembles AUC Tuning'+".pdf")    
                
    def polar_plot_aucs_pcns(self) :   
        auc =  self.crf_results.results['results']['auc']
        good_auc=auc[:-8]
        cell_angle=self.full_angles_matrix[:,:,0][0,:]

                 
        for i, (key, grating) in enumerate(analysis.crf_results.roc_curves.items()):
            fig= plt.figure(figsize=(10,6))
            ax = fig.add_subplot(projection='polar')
            ax.yaxis.grid(False)
            ax.set_yticklabels([])
            fig.suptitle('PCCs AUC Tuning '+key)
            
        
            for key2, ensemble in grating.items():
           
                PCCs_indexes=self.crf_results.results['PCNs'][i].astype(int)-1
                aucs=good_auc[PCCs_indexes]

            for h, cell in enumerate(aucs):
                ax.plot( np.append(cell_angle, cell_angle[0]), 
                          np.append(aucs[h], aucs[h][0]), label='PCCs'+str(PCCs_indexes[h]+1))
            ax.legend()

            fig.savefig('PCCs AUC Tuning Ensemble'+key+".pdf")  
            
        
    def polar_plot_average_ensemble(self) :   
       
        fig4= plt.figure(figsize=(10,6))
        ax4 = fig4.add_subplot(projection='polar')
        ax4.yaxis.grid(False)
        ax4.set_yticklabels([])
        fig4.suptitle('Orientation Tuning Averaged TF')
        
        fig1= plt.figure(figsize=(10,6))
        ax1 = fig1.add_subplot(projection='polar')
        ax1.yaxis.grid(False)
        ax1.set_yticklabels([])
        fig1.suptitle('Orientation Tuning 2Hz')

        fig2= plt.figure(figsize=(10,6))
        ax2 = fig2.add_subplot(projection='polar')      
        ax2.yaxis.grid(False)
        ax2.set_yticklabels([])
        fig2.suptitle('Direction Tuning 2Hz')

        fig3= plt.figure(figsize=(10,6))
        ax3 = fig3.add_subplot(projection='polar')
        ax3.yaxis.grid(False)
        ax3.set_yticklabels([])
        fig3.suptitle('Direction Tuning Averaged TF') 
       
        allll=self.ensembles
        cell_angle=self.full_angles_matrix[0,:,0]  


        for mm,(key, value) in enumerate(self.ensembles.items()):
            
            test=np.mean(self.evoked_all_activities['dfdtmatrix']['evoked_trial_averaged_df_f_mean'][value,:], axis=0)
    
            idx=np.zeros((8,5)) 
            for i in range(8):
                idx[i,:]=np.arange(i,40,8)
            idx=idx.astype(int)
                
            
            meantofreq=np.zeros((8)) 
            for i in range(8) :
                meantofreq[i]=np.mean(test[idx[i,:]])
            correctedmeantofreq=np.append(meantofreq, meantofreq[0])
            
            only2hzidx=idx[:,1]
            only2hz=test[only2hzidx]
            correctedonly2hz=np.append(only2hz, only2hz[0])
            correcetdcell_angle=np.append(cell_angle, cell_angle[0])
    
            orientationidx=np.zeros((4,2)) 
            for i in range(4):
                orientationidx[i,:]=np.arange(i,8,4)
            orientationidx=orientationidx.astype(int)
            orienttuningaveraged=np.zeros(4)
            orienttuning=np.zeros(4)
            for i in range(4):
                   orienttuning[i]=np.mean(only2hz[orientationidx[i]])
                   orienttuningaveraged[i]=np.mean(meantofreq[orientationidx[i]])   
    
       
            ax1.plot( np.append(cell_angle[::2], cell_angle[0]), 
                          np.append(orienttuning, orienttuning[0]), label='Full Ensemble'+str(mm+1))
           
            ax2.plot(correcetdcell_angle, 
                          correctedonly2hz, label='Full Ensemble'+str(mm+1))
            
            ax3.plot(correcetdcell_angle, 
                           correctedmeantofreq, label='Full Ensemble'+str(mm+1))
    
            ax4.plot( np.append(cell_angle[::2], cell_angle[0]), 
                          np.append(orienttuningaveraged, orienttuningaveraged[0]),label='Full Ensemble'+str(mm+1))
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        fig1.savefig('Ensembles Orientation Tuning 2Hz'+".pdf")
        fig2.savefig('Ensembles Direction Tuning 2Hz'+".pdf")
        fig3.savefig('Ensembles Direction Tuning Averaged TF'+".pdf")
        fig4.savefig('Ensembles Orientation Tuning Averaged TF'+".pdf")


                
            
#%% run the necessary classes
if __name__ == "__main__":
    
    plt.close('all')
    # sorter results
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIK3planeallen\Plane1'
    
    # linux_temp='/home/samuel/Desktop/SPJAFUllAllen/'
    # windowstemp='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000'
    windowstemp='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'

    plane1=os.sep+'Plane1'+os.sep
    plane2=os.sep+'Plane2'+os.sep
    plane3=os.sep+'Plane3'+os.sep
    temporary_path1=windowstemp+plane1
    temporary_path2=windowstemp+plane2
    temporary_path3=windowstemp+plane3
    # temporary_path1=linux_temp+plane1
    # temporary_path2=linux_temp+plane2
    # temporary_path3=linux_temp+plane3

    # SPJA_0702_allen_plane1=CaimanSorterYSResults(temporary_path1+ '210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_d1_256_d2_256_d3_1_order_F_frames_64416_cnmf_results_sort.mat')
    # SPJA_0702_allen_plane2=CaimanSorterYSResults(temporary_path2+ '210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_d1_256_d2_256_d3_1_order_F_frames_64416_cnmf_results_sort.mat')
    # SPJA_0702_allen_plane3=CaimanSorterYSResults(temporary_path3+ '210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_d1_256_d2_256_d3_1_order_F_frames_64416_cnmf_results_sort.mat')
    SPKG_1015_allen_plane1=CaimanSorterYSResults(temporary_path1+ '211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_MC_OnACID_20211020-013317_cnmf_results_sort.mat')
    SPKG_1015_allen_plane2=CaimanSorterYSResults(temporary_path2+ '211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_MC_OnACID_20211020-123307_cnmf_results_sort.mat')
    SPKG_1015_allen_plane3=CaimanSorterYSResults(temporary_path3+ '211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_MC_OnACID_20211020-164443_cnmf_results_sort.mat')
# crf results

    # temporary_path=linux_temp +os.sep+'SAMUEL_BIG_RUN_2Hz'
#     temporary_path=windowstemp +os.sep+'SAMUEL_BIG_RUN_2Hz'
#     # this results are for the 2hz frequency only
#     SPJA_0702_allen_CRFS=CRFsResults(temporary_path+ os.sep+'results.mat',
#                                      plane1_cell_number=SPJA_0702_allen_plane1.accepted_cells,
#                                      plane2_cell_number=SPJA_0702_allen_plane2.accepted_cells ,
#                                      plane3_cell_number=SPJA_0702_allen_plane3.accepted_cells
#                                      )
# #
    # temporary_path1='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000'
    temporary_path1='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'

    meta =Metadata(acquisition_directory_raw=temporary_path1)


# voltage signals

    # temporary_path1=linux_temp +os.sep+'210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'
    temporary_path1=windowstemp +os.sep+'211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'

    voltagesignals=VoltageSignalsExtractions(temporary_path1)

#%% RUN ANALYSIS CLASS

    # analysis=ResultsAnalysis(SPJA_0702_allen_plane1, SPJA_0702_allen_plane2, SPJA_0702_allen_plane3, SPJA_0702_allen_CRFS, voltagesignals, meta)
    analysis=ResultsAnalysis(SPKG_1015_allen_plane1, SPKG_1015_allen_plane2, SPKG_1015_allen_plane3, acquisition_voltage_signals_object=voltagesignals, metadata_object=meta )

    #%% plotting
    plt.close('all') 
 
    
    # analysis.plot_activity_matrixes()
    # analysis.plot_activity_matrix_with_signals(analysis.allplanesdfdt)
    # analysis.plot_activity_matrix_with_signals_and_grating_ranges()
    # analysis.plotting_corrected_ranges()
    # analysis.plotting_resampling_accuracy_of_ranges()
    # analysis.more_plotting(0,0,0)
    # analysis.plot_all_tuning_single_cell(204)
    # analysis.plot_trial_averaged_single_cell(0)
    # analysis.plot_evoked_tuning_single_cell(0)
    # analysis.plot_selectivity_histograms()
    # analysis.polar_plot(0)
    # analysis.polar_plot_single_temporal(0)
    # analysis.ensembles
    # analysis.PCNS['0_2hz']
    
    # for cell in analysis.PCNS['0_2hz']:
    #     analysis.polar_plot(cell)
    # for cell in analysis.PCNS['0_2hz']:
    #     analysis.plot_trial_averaged_single_cell(cell)
    # for cell in analysis.PCNS['0_2hz']:
    #     analysis.plot_evoked_tuning_single_cell(cell)
    
    # analysis.polar_plot_single_temporal(204)
    # for cell in analysis.PCNS['0_2hz']:
    #     analysis.polar_plot_single_temporal(cell)
 
    # analysis.polar_plot_auc()
    # analysis.polar_plot_aucs_pcns()
    # for key, ensemble in  analysis.PCNS.items():
    #     analysis.polar_plot_multiplecells_temporal(ensemble,  grating_name=key)
    # analysis.polar_plot_average_ensemble()
    
    
    analysis.pcns_plotting()
#%%
   
    
    # def seriation(Z,N,cur_index):
    #     '''
    #         input:
    #             - Z is a hierarchical tree (dendrogram)
    #             - N is the number of points given to the clustering process
    #             - cur_index is the position in the tree for the recursive traversal
    #         output:
    #             - order implied by the hierarchical tree Z
                
    #         seriation computes the order implied by a hierarchical tree (dendrogram)
    #     '''
    #     if cur_index < N:
    #         return [cur_index]
    #     else:
    #         left = int(Z[cur_index-N,0])
    #         right = int(Z[cur_index-N,1])
    #         return (seriation(Z,N,left) + seriation(Z,N,right))
    
    # flat_dist_met = pdist( analysis.activity_matrixes_resampled_bordercuts['dfdtmatrix'], metric='cosine');
    # cs = 1- squareform(flat_dist_met)
    # res_linkage = linkage(flat_dist_met, method='ward')
    # fig = plt.figure(figsize=(10, 6))
    # dn = dendrogram(res_linkage)
    # plt.show()
    # N = 271
    # res_ord = seriation(res_linkage,N, N + N -2)
    # cs_ord = 1- squareform(pdist(analysis.activity_matrixes_resampled_bordercuts['dfdtmatrix'][res_ord,:], metric='cosine'));
    # fi,ax=plt.subplots(ncols=2, figsize=(10,6))
    # ax[1].imshow(cs_ord,vmin=0.3, vmax=0.4)
    # ax[0].imshow(cs,vmin=0.3, vmax=0.4)
    # fi.suptitle('Cosine Similarity')
    # ax[1].set_xlabel('Cell');  
    # ax[0].set_xlabel('Cell');  
    # ax[1].set_ylabel('Cell');  
    # ax[0].set_ylabel('Cell');  
    # ax[0].set_title('Unsorted')
    # ax[1].set_title('Ward\'s Linkage Sorted')
    # fi.savefig('Cosine Similarity Dfdt'+".pdf")



    
 #%%
    fig,axing =plt.subplots(nrows=5, figsize=(10,6))
    
    grating=5
    cell=3
    axing[0].plot(np.mean(analysis.grating_extended_sliced_arrays['dfdtmatrix'],0)[:,grating,cell])
    axing[1].plot(np.mean(analysis.grating_extended_sliced_arrays['rawmatrix'],0)[:,grating,cell])
    axing[2].plot(np.mean(analysis.grating_extended_sliced_arrays['mcmcmatrix'],0)[:,grating,cell])

    axing[3].plot(np.mean(analysis.grating_extended_sliced_traces['speed_trace'],0)[:,grating])
    axing[4].plot(np.mean(analysis.grating_extended_sliced_traces['vistim_trace'],0)[:,grating])

    fig,axing =plt.subplots(nrows=2, figsize=(10,6))
    axing[0].plot(analysis.voltage_traces_resampled_bordercuts['speed_trace'])
    axing[1].plot(analysis.voltage_traces_resampled_bordercuts['speed_trace'])
    axing[1].set_ylim(np.mean(analysis.voltage_traces_resampled_bordercuts['speed_trace'])+1.5*np.std(analysis.voltage_traces_resampled_bordercuts['speed_trace']),1)

