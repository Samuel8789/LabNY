# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:56:22 2022

@author: sp3660
"""

import networkx as nx
import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt
from matplotlib import gridspec
# import torch
from math import sqrt

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
from allensdk.brain_observatory.drifting_gratings import DriftingGratings
try:
    from .allenDatasetEquivalent import AllenDatasetEquivalent
except:
    from allenDatasetEquivalent import AllenDatasetEquivalent

from matplotlib import pyplot as plt
import pandas as pd
import scipy.stats as st
import pandas as pd
import numpy as np
import h5py
from math import sqrt
import logging
from . import observatory_plots as oplots
from . import circle_plots as cplots

    
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b",'g','y','c','m', 'tab:brown']) 



class AllenAnalysis():
    
    def __init__(self, analysis_object=None,  plane1_caiman_sorter_results_object=None, plane2_caiman_sorter_results_object=None,plane3_caiman_sorter_results_object=None, 
                 acquisition_voltage_signals_object=None, metadata_object=None):
     
        if analysis_object:
            self.analysis_object=analysis_object
        
        
            self.path_managing()
            self.milisecond_period=self.analysis_object.milisecond_period
            self.full_data=self.analysis_object.full_data
            self.movie_rate=self.full_data['imaging_data']['Frame_rate']
            self.stimulus_table={}

            '''
            steps
                plot visstim signal with speed
                plottransitions framse
                plot individual paradimgs with speed
                Selecte ALLENA
                    Define drifting grating parameters
                    get slicnf ranges already corrected and aligne
                        ?what to do if ranges have different size
                            align to start
                    get sliced grating matrix
                    calculate dff over trials
                    calculate trial avergae activity
                    plot and check everythin here
                    
                    do teh rest of analysys
                            
                
                
                
            # '''
            # self.organize_signals_and_traces()

            
            # if self.analysis_object.signals_object.vis_stim_protocol =='AllenA':
            #     self.set_up_drifting_gratings_parameters()
            #     self.do_AllenA_analysis()

                
            # self.calculate_windows_and_ranges()
            
            
            # self.slice_responses_by_gratings(  self.activity_matrixes_resampled,  self.ranges_resampled,     self.voltage_traces_resampled)
            # self.calculate_df_f()
            # self.trial_averaging_and_evoked_activity()


            
        
        else:
                
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
                          
            # self.resample_voltage_matrices()
            # self.resample_grating_indexes_and_matrix()
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



#%% Allen analsysys
    def path_managing(self):
        self.allen_results_path=self.analysis_object.data_paths['allen_runs_path']
        noise_correlations_filename=self.analysis_object.acquisition_object.aquisition_name+'_noise_correlations'
        signal_correlations_filename=self.analysis_object.acquisition_object.aquisition_name+'_signal_correlations'
        represent_similarty_matrix_filename=self.analysis_object.acquisition_object.aquisition_name+'_represent_similarty'
        self.noise_correlations_to_save_path=os.path.join(self.allen_results_path,noise_correlations_filename)
        self.signal_correlations_to_save_path=os.path.join(self.allen_results_path,signal_correlations_filename)
        self.represent_similarty_matrix_to_save_path=os.path.join(self.allen_results_path,represent_similarty_matrix_filename)
        
        


   
    def do_AllenA_analysis(self, plane, matrix):
        
        self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table']=self.stimulus_table['drifting_gratings']
        
        
        self.mock_allen_dataset=AllenDatasetEquivalent(self.full_data, plane)
        self.drift_obj=DriftingGratings(self.mock_allen_dataset)
        self.sweeplength = 33
        self.interlength = 16
        self.extralength = 0
        self.drift_obj.sweeplength = self.sweeplength
        self.drift_obj.interlength = self.interlength
        

        
        self.sweep_response, self.mean_sweep_response,  self.pval = self.deconvolved_get_sweep_response( plane, matrix)
        # self.response=self.get_response()
        # self.peak=self.get_peak()



        
        
        
    def deconvolved_get_sweep_response(self, plane, matrix):
        def do_mean(x):
            # +1])
            return np.mean(
                x[self.interlength:
                  self.interlength + self.sweeplength + self.extralength])

        def do_p_value(x):
            (_, p) = \
                st.f_oneway(
                    x[:self.interlength],
                    x[self.interlength:
                      self.interlength + self.sweeplength + self.extralength])
            return p

        C_mat=self.analysis_object.full_data['imaging_data'][plane]['Traces'][matrix]
        self.numbercells=len(C_mat)
        self.celltraces=C_mat
        
        _,self.dxcm=self.mock_allen_dataset.get_running_speed()
        sweep_response = pd.DataFrame(index=self.stimulus_table['drifting_gratings'].index.values,
                                      columns=list(map(str, range(
                                          self.numbercells + 1))))
        
        sweep_response.rename(
            columns={str(self.numbercells): 'dx'}, inplace=True)
        
        
        for index, row in self.stimulus_table['drifting_gratings'].iterrows():
            start = int(row['start'] - self.interlength)
            end = int(row['start'] + self.sweeplength + self.interlength)

            for nc in range(self.numbercells):
                temp = self.celltraces[int(nc), start:end]
                sweep_response[str(nc)][index] =temp 
            sweep_response['dx'][index] = self.dxcm[start:end]
            
        mean_sweep_response = sweep_response.applymap(do_mean)

        pval = sweep_response.applymap(do_p_value)
        return sweep_response, mean_sweep_response, pval
    
       


    def get_response(self):
        ''' Computes the mean response for each cell to each stimulus
        condition.  Return is
        a (# orientations, # temporal frequencies, # cells, 3) np.ndarray.
        The final dimension
        contains the mean response to the condition (index 0), standard
        error of the mean of the response
        to the condition (index 1), and the number of trials with a
        significant response (p < 0.05)
        to that condition (index 2).

        Returns
        -------
        Numpy array storing mean responses.
        '''
        DriftingGratings._log.info("Calculating mean responses")

        response = np.empty(
            (self.drift_obj.number_ori, self.drift_obj.number_tf, self.numbercells + 1, 3))

        def ptest(x):
            if x.empty:
                return np.nan
            return len(np.where(x < (0.05 / (8 * 5)))[0])

        for ori in self.drift_obj.orivals:
            ori_pt = np.where(self.drift_obj.orivals == ori)[0][0]
            for tf in self.drift_obj.tfvals:
                tf_pt = np.where(self.drift_obj.tfvals == tf)[0][0]
                subset_response = self.mean_sweep_response[
                    (self.drift_obj.stim_table.temporal_frequency == tf) & (
                                self.drift_obj.stim_table.orientation == ori)]
                subset_pval = self.pval[
                    (self.drift_obj.stim_table.temporal_frequency == tf) & (
                            self.drift_obj.stim_table.orientation == ori)]
                response[ori_pt, tf_pt, :, 0] = subset_response.mean(axis=0)
                response[ori_pt, tf_pt, :, 1] = subset_response.std(
                    axis=0) / sqrt(len(subset_response))
                response[ori_pt, tf_pt, :, 2] = subset_pval.apply(
                    ptest, axis=0)
        return response
        
        
    def get_peak(self):
        ''' Computes metrics related to each cell's peak response condition.
 
        Returns
        -------
        Pandas data frame containing the following columns (_dg suffix is
        for drifting grating):
            * ori_dg (orientation)
            * tf_dg (temporal frequency)
            * reliability_dg
            * osi_dg (orientation selectivity index)
            * dsi_dg (direction selectivity index)
            * peak_dff_dg (peak dF/F)
            * ptest_dg
            * p_run_dg
            * run_modulation_dg
            * cv_dg (circular variance)
        '''
        DriftingGratings._log.info('Calculating peak response properties')
 
        peak = pd.DataFrame(index=range(self.drift_obj.numbercells),
                            columns=('ori_dg', 'tf_dg', 'reliability_dg',
                                     'osi_dg', 'dsi_dg', 'peak_dff_dg',
                                     'ptest_dg', 'p_run_dg',
                                     'run_modulation_dg',
                                     'cv_os_dg', 'cv_ds_dg', 'tf_index_dg',
                                     'cell_specimen_id'))
        # cids = self.drift_obj.data_set.get_cell_specimen_ids()
        cids=np.arange(self.drift_obj.numbercells)
        orivals_rad = np.deg2rad(self.drift_obj.orivals)
        for nc in range(self.drift_obj.numbercells):
            cell_peak = np.where(self.response[:, 1:, nc, 0] == np.nanmax(
                self.response[:, 1:, nc, 0]))
            prefori = cell_peak[0][0]
            preftf = cell_peak[1][0] + 1
            peak.cell_specimen_id.iloc[nc] = cids[nc]
            peak.ori_dg.iloc[nc] = prefori
            peak.tf_dg.iloc[nc] = preftf
 
            pref = self.response[prefori, preftf, nc, 0]
            orth1 = self.response[np.mod(prefori + 2, 8), preftf, nc, 0]
            orth2 = self.response[np.mod(prefori - 2, 8), preftf, nc, 0]
            orth = (orth1 + orth2) / 2
            null = self.response[np.mod(prefori + 4, 8), preftf, nc, 0]
 
            tuning = self.response[:, preftf, nc, 0]
            tuning = np.where(tuning > 0, tuning, 0)
            # new circular variance below
            CV_top_os = np.empty((8), dtype=np.complex128)
            CV_top_ds = np.empty((8), dtype=np.complex128)
            for i in range(8):
                CV_top_os[i] = (tuning[i] * np.exp(1j * 2 * orivals_rad[i]))
                CV_top_ds[i] = (tuning[i] * np.exp(1j * orivals_rad[i]))
            peak.cv_os_dg.iloc[nc] = np.abs(CV_top_os.sum()) / tuning.sum()
            peak.cv_ds_dg.iloc[nc] = np.abs(CV_top_ds.sum()) / tuning.sum()
 
            peak.osi_dg.iloc[nc] = (pref - orth) / (pref + orth)
            peak.dsi_dg.iloc[nc] = (pref - null) / (pref + null)
            peak.peak_dff_dg.iloc[nc] = pref
 
            groups = []
            for ori in self.drift_obj.orivals:
                for tf in self.drift_obj.tfvals[1:]:
                    groups.append(
                        self.mean_sweep_response[
                            (self.drift_obj.stim_table.temporal_frequency == tf) &
                            (self.drift_obj.stim_table.orientation == ori)][str(nc)])
            groups.append(self.mean_sweep_response[
                              self.drift_obj.stim_table.temporal_frequency == 0][
                              str(nc)])
            _, p = st.f_oneway(*groups)
            peak.ptest_dg.iloc[nc] = p
 
            subset = self.mean_sweep_response[
                (self.drift_obj.stim_table.temporal_frequency == self.drift_obj.tfvals[preftf]) &
                (self.drift_obj.stim_table.orientation == self.drift_obj.orivals[prefori])]
 
            # running modulation
            subset_stat = subset[subset.dx < 1000]
            subset_run = subset[subset.dx >= 1000]
            if (len(subset_run) > 2) & (len(subset_stat) > 2):
                (_, peak.p_run_dg.iloc[nc]) = st.ttest_ind(subset_run[str(nc)],
                                                           subset_stat[
                                                               str(nc)],
                                                           equal_var=False)
 
                if subset_run[str(nc)].mean() > subset_stat[str(nc)].mean():
                    peak.run_modulation_dg.iloc[nc] = (subset_run[
                                                           str(nc)].mean() -
                                                       subset_stat[
                                                           str(nc)].mean()) \
                                                      / np.abs(
                        subset_run[str(nc)].mean())
                elif subset_run[str(nc)].mean() < subset_stat[str(nc)].mean():
                    peak.run_modulation_dg.iloc[nc] = \
                        (-1 * (subset_stat[str(nc)].mean() -
                               subset_run[str(nc)].mean()) /
                         np.abs(subset_stat[str(nc)].mean()))
 
            else:
                peak.p_run_dg.iloc[nc] = np.NaN
                peak.run_modulation_dg.iloc[nc] = np.NaN
 
            # reliability
            subset = self.sweep_response[
                (self.drift_obj.stim_table.temporal_frequency == self.drift_obj.tfvals[preftf]) &
                (self.drift_obj.stim_table.orientation == self.drift_obj.orivals[prefori])]
            corr_matrix = np.empty((len(subset), len(subset)))
            for i in range(len(subset)):
                for j in range(len(subset)):
                    r, p = st.pearsonr(subset[str(nc)].iloc[i][30:90],
                                       subset[str(nc)].iloc[j][30:90])
                    corr_matrix[i, j] = r
            mask = np.ones((len(subset), len(subset)))
            for i in range(len(subset)):
                for j in range(len(subset)):
                    if i >= j:
                        mask[i, j] = np.NaN
            corr_matrix *= mask
            peak.reliability_dg.iloc[nc] = np.nanmean(corr_matrix)
 
            # TF index
            tf_tuning = self.response[prefori, 1:, nc, 0]
            trials = self.mean_sweep_response[
                (self.drift_obj.stim_table.temporal_frequency != 0) &
                (self.drift_obj.stim_table.orientation == self.drift_obj.orivals[prefori])
            ][str(nc)].values
            SSE_part = np.sqrt(
                np.sum((trials - trials.mean()) ** 2) / (len(trials) - 5))
            peak.tf_index_dg.iloc[nc] = (np.ptp(tf_tuning)) / (
                        np.ptp(tf_tuning) + 2 * SSE_part)
 
        return peak
    
    def row_from_cell_id(self, csid=None, idx=None):

        if csid is not None and not np.isnan(csid):
            return self.drift_obj.data_set.get_cell_specimen_ids().tolist().index(csid)
        elif idx is not None:
            return idx
        else:
            raise Exception("Could not find row for csid(%s) idx(%s)"
                            % (str(csid), str(idx)))
    
    def open_star_plot(self, cell_specimen_id=None, include_labels=False,
                       cell_index=None):
        cell_index = self.row_from_cell_id(cell_specimen_id, cell_index)

        df = self.mean_sweep_response[str(cell_index)]
        st = self.drift_obj.data_set.get_stimulus_table('drifting_gratings')
        mask = st.dropna(subset=['orientation']).index

        data = df.values

        cmin = self.response[0, 0, cell_index, 0]
        cmax = max(cmin, data.mean() + data.std() * 3)

        fp = cplots.FanPlotter.for_drifting_gratings()
        fp.plot(r_data=st.temporal_frequency.loc[mask].values,
                angle_data=st.orientation.loc[mask].values,
                data=df.loc[mask].values,
                clim=[cmin, cmax])
        fp.show_axes(closed=True)

        if include_labels:
            fp.show_r_labels()
            fp.show_angle_labels()
        plt.show()
        
    def plot_orientation_selectivity(self,
                                     si_range=oplots.SI_RANGE,
                                     n_hist_bins=oplots.N_HIST_BINS,
                                     color=oplots.STIM_COLOR,
                                     p_value_max=oplots.P_VALUE_MAX,
                                     peak_dff_min=oplots.PEAK_DFF_MIN,
                                     cell_type='All'):
        if cell_type=='All':
            vis_cells = ( self.peak.ptest_dg < p_value_max) & (
                        self.peak.peak_dff_dg > peak_dff_min)

        else:
        # responsive cells
            vis_cells = ( self.peak.ptest_dg < p_value_max) & (
                        self.peak.peak_dff_dg > peak_dff_min) & (self.peak.Tomato==cell_type)

        # orientation selective cells
        osi_cells = vis_cells & ( self.peak.osi_dg > si_range[0]) & (
                    self.peak.osi_dg < si_range[1])

        peak_osi =  self.peak.loc[osi_cells]
        osis = peak_osi.osi_dg.values

        oplots.plot_selectivity_cumulative_histogram(osis,
                                                     "orientation "
                                                     "selectivity index",
                                                     si_range=si_range,
                                                     n_hist_bins=n_hist_bins,
                                                     color=color)
        plt.show()


    def plot_direction_selectivity(self,
                                   si_range=oplots.SI_RANGE,
                                   n_hist_bins=oplots.N_HIST_BINS,
                                   color=oplots.STIM_COLOR,
                                   p_value_max=oplots.P_VALUE_MAX,
                                   peak_dff_min=oplots.PEAK_DFF_MIN,
                                   cell_type='All'):
        if cell_type=='All':
            vis_cells = (self.peak.ptest_dg < p_value_max) & (
                        self.peak.peak_dff_dg > peak_dff_min) 
        else:
        # responsive cells
            vis_cells = (self.peak.ptest_dg < p_value_max) & (
                        self.peak.peak_dff_dg > peak_dff_min) & (self.peak.Tomato==cell_type)

        # direction selective cells
        dsi_cells = vis_cells & (self.peak.dsi_dg > si_range[0]) & (
                    self.peak.dsi_dg < si_range[1])

        peak_dsi = self.peak.loc[dsi_cells]
        dsis = peak_dsi.dsi_dg.values

        oplots.plot_selectivity_cumulative_histogram(dsis,
                                                     "direction selectivity "
                                                     "index",
                                                     si_range=si_range,
                                                     n_hist_bins=n_hist_bins,
                                                     color=color)
        plt.show()


    def plot_preferred_direction(self,
                                 include_labels=False,
                                 si_range=oplots.SI_RANGE,
                                 color=oplots.STIM_COLOR,
                                 p_value_max=oplots.P_VALUE_MAX,
                                 peak_dff_min=oplots.PEAK_DFF_MIN,
                                 cell_type='All'):
        if cell_type=='All':
            vis_cells = ( self.peak.ptest_dg < p_value_max) & (
                        self.peak.peak_dff_dg > peak_dff_min) 
        else:
            vis_cells = ( self.peak.ptest_dg < p_value_max) & (
                        self.peak.peak_dff_dg > peak_dff_min) & (self.peak.Tomato==cell_type)
            
        pref_dirs =  self.peak.loc[vis_cells].ori_dg.values
        pref_dirs = [self.drift_obj.orivals[pref_dir] for pref_dir in pref_dirs]

        angles, counts = np.unique(pref_dirs, return_counts=True)
        oplots.plot_radial_histogram(angles,
                                     counts,
                                     include_labels=include_labels,
                                     all_angles=self.drift_obj.orivals,
                                     direction=-1,
                                     offset=0.0,
                                     closed=True,
                                     color=color)
        plt.show()


    def plot_preferred_temporal_frequency(self,
                                          si_range=oplots.SI_RANGE,
                                          color=oplots.STIM_COLOR,
                                          p_value_max=oplots.P_VALUE_MAX,
                                          peak_dff_min=oplots.PEAK_DFF_MIN,
                                          cell_type='All'):
        if cell_type=='All':
            vis_cells = (self.peak.ptest_dg < p_value_max) & (
                        self.peak.peak_dff_dg > peak_dff_min) 
        else:
            vis_cells = (self.peak.ptest_dg < p_value_max) & (
                    self.peak.peak_dff_dg > peak_dff_min) & (self.peak.Tomato==cell_type)
            
        pref_tfs = self.peak.loc[vis_cells].tf_dg.values
        
     
            
            
            

        oplots.plot_condition_histogram(pref_tfs,
                                        self.drift_obj.tfvals[1:])

        plt.xlabel('temporal frequency (Hz)')
        plt.ylabel('number of cells')
        plt.show()
     
    # peak_info=self.drift_obj.get_peak() #gives error
    # reponse_info=self.drift_obj.get_response() # not proper dimensions
    # noise_cor=self.drift_obj.get_noise_correlation()
    # rep_sim=self.drift_obj.get_representational_similarity()
    # sig_cor=self.drift_obj.get_signal_correlation()
    # # open_star_plot(cell_specimen_id=)
    
    # plt.imshow(sig_cor[0])
    # plt.imshow(rep_sim[0])
    # plt.imshow(noise_cor[0][:,:,0,0])
    # plt.imshow(noise_cor[2])
    # plt.imshow(noise_cor[3])
    
    # cell=2
    # cell_resp=reponse_info[:,1:,cell,:]
    # goodresp=reponse_info[:,1:,:,2]
    # plt.imshow(goodresp[:,:,cell], aspect='auto')
    
    # test=self.mock_allen_dataset.get_corrected_fluorescence_traces()
    # test2=self.mock_allen_dataset.get_cell_specimen_ids()
    # plt.plot(test[0], test[1][cell,:])
    
    
    
    # filt=self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table']['orientation']==45
    # self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'][filt]
    # filt2=self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'][filt]['temporal_frequency']==8
    # filtered=self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'][filt][filt2]
    
    # trial=558
    
    
    # plt.plot(test[0][filtered.start[trial]-10:filtered.end[trial]+10], test[1][cell, filtered.start[trial]-10:filtered.end[trial]+10])



        
#%% DRIFTING GRATINGS
    def set_up_drifting_gratings_parameters(self):
        
        
        self.isi_time=1000     #ms
        self.stim_time=2000    #ms
        self.pre_time=350     #ms
        self.post_time=350      #ms
        self.pre_frames=np.ceil(self.pre_time/self.milisecond_period).astype(int)
        self.post_frames=np.ceil(self.post_time/self.milisecond_period).astype(int)
        
        
        self.grating_number=self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'].shape[0]
        
        
        self.grating_repetitions=self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'].shape[1]
        self.grating_frame_number=np.arange(self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'][0,0]-self.pre_frames, self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_off'][0,0]+self.post_frames).size
      
        self.angles=np.linspace(0,360,9)[:-1]
        self.angle_numbers=len(self.angles)
        self.frequencies=np.array([1,2,4,8,15])
        self.frequency_numbers=len(self.frequencies)
      
        self.angles_xv, self.frequencies_yv = np.meshgrid(self.angles, self.frequencies)
        self.anglevalues = np.reshape(np.arange(1,41), (5, 8))
        
        all_rows=[]
        for ori in range(1,41):
  
            angled=self.angles_xv[:,np.where(self.anglevalues==ori)[1][0]][0]
            freq=float(self.frequencies[np.where(self.anglevalues==ori)[0][0]])
            
            
            indexes=list(zip(self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'][ori-1,:], self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_off'][ori-1,:]))
            for i in indexes:
               all_rows.append((np.float32(freq),np.float32(angled), np.float32(0),np.int32(i[0]), np.int32(i[1]) ))
              
              
        blankindexes=list(zip(self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Blank_sweep_on'], self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Blank_sweep_off']))

        for i in blankindexes:
          all_rows.append((np.float32(np.nan), np.float32(np.nan), np.float32(1),np.int32(i[0][0]), np.int32(i[1][0]) ))
 
        df = pd.DataFrame(all_rows, columns =['temporal_frequency','orientation', 'blank_sweep', 'start', 'end'])
        sorted_df=df.sort_values(by=['start'])
        
        self.stimulus_table['drifting_gratings']=sorted_df.reset_index(drop=True)

        
    
    def calculate_windows_and_ranges(self):
        
        self.frame_windows={}
        for grat in range( self.grating_number):
            temp=[]
            selected_grating_starts=self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'][grat,:]
            selected_grating_ends=self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_off'][grat,:]
            for rept in range( self.grating_repetitions): 
                frame_window=np.arange(selected_grating_starts[rept]-self.pre_frames, selected_grating_ends[rept]+self.post_frames).astype('int64')
                if len(frame_window)<self.grating_frame_number:
                    # temp.append(np.append(frame_window, frame_window[-1]+1))
                    temp.append(np.insert(frame_window, 0, frame_window[0]-1))
                elif len(frame_window)>self.grating_frame_number:
                    temp.append(frame_window[:-1])
                else:
                    temp.append(frame_window)
            self.frame_windows[grat]=np.vstack(temp)    
  
        self.ranges={}
        for key, grat in self.frame_windows.items():
            self.ranges[key]=[]
            for trial in grat:
                self.ranges[key].append(np.arange(trial[0],trial[-1]+1))
                
            self.ranges[key]=np.vstack( self.ranges[key])
                
                
        self.extended_ranges={}
        for key, grat in self.frame_windows.items():
            self.extended_ranges[key]=[]
            for trial in grat:
                self.extended_ranges[key].append(np.arange(trial[0]-50,trial[-1]+50))
            self.extended_ranges[key]=np.vstack(self.extended_ranges[key])

                
                
        for i in range(self.grating_number):       
            zzz2=np.ceil(self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_downsampled'][i,self.frame_windows[i].squeeze()])
            zzz=np.ceil(self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_recreated'][i,self.frame_windows[i].squeeze()])

            zzzz=np.diff(zzz)
            firsidex=np.argwhere(zzzz[:,:11])[:,1]
            tocorrect=np.argwhere(abs(firsidex-np.nanmean(firsidex))==max(abs(firsidex-np.nanmean(firsidex))))
            self.frame_windows[i][tocorrect.flatten()]=self.frame_windows[i][tocorrect.flatten()]-1
          
            # zzz2=np.ceil(activity_matrixes_resampled['grating_binary_matrix'][i,frame_windows[i]])
            # zzzz2=np.diff(zzz2)
            # firsidex2=np.argwhere(zzzz2[:,:11])[:,1]
            # tocorrect2=np.argwhere(abs(firsidex2-np.nanmean(firsidex2))==max(abs(firsidex2-np.nanmean(firsidex2))))
  
          
  
    
    def slice_matrix_by_indexes(self, matrix, slicing_range_matrix):
        pass
        # stim=indexes_on.shape[0]
        # reps=indexes_on.shape[1]
        # sliced_matrix=np.zeros((matrix.shape[0],stim, reps, self.grating_frame_number))
        
    
  
    def slice_responses_by_gratings(self, activity_dic, indexes, voltage_dictionary):   
        
        
        self.grating_sliced_dictionary={}
        self.extended_grating_sliced_dictionary={}
        
        for activity_matrix  in activity_dic.keys():
            self.grating_sliced_dictionary[activity_matrix]={}
            self.extended_grating_sliced_dictionary[activity_matrix]={}

            for plane in activity_dic[activity_matrix]:
                cell_number=activity_dic[activity_matrix][plane].shape[0]
                self.grating_sliced_dictionary[activity_matrix][plane]=np.zeros((cell_number,self.grating_number, self.grating_repetitions,self.grating_frame_number))
                self.extended_grating_sliced_dictionary[activity_matrix][plane]=np.zeros((cell_number,self.grating_number ,self.grating_repetitions, len(self.extended_ranges[0][0]) ))
                             
                
               
  
        for activity, planes in self.grating_sliced_dictionary.items():
            for plane, matrix  in planes.items():
                for cell in range(matrix.shape[0]):
                    for grat, trials in  self.ranges.items():
                        for row, trial in enumerate(trials):
                            matrix[cell,grat,row,:]=activity_dic[activity][plane][cell,trial]
  
        for activity, planes in self.extended_grating_sliced_dictionary.items():
            for plane, matrix  in planes.items():
                for cell in range(matrix.shape[0]):
                    for grat, trials in  self.extended_ranges.items():
                        for row, trial in enumerate(trials):
                            matrix[cell,grat,row,:]=activity_dic[activity][plane][cell,trial]
   
                
        self.grating_sliced_voltages_dictionary={}
        self.extended_grating_sliced_voltages_dictionary={}
        
        for activity_matrix  in voltage_dictionary.keys():
            self.grating_sliced_voltages_dictionary[activity_matrix]=np.zeros((self.grating_repetitions,self.grating_frame_number, self.grating_number))
            self.extended_grating_sliced_voltages_dictionary[activity_matrix]=np.zeros((self.grating_repetitions,len(self.extended_ranges[0][0]), self.grating_number))
            
            self.grating_sliced_voltages_dictionary[activity_matrix]=np.zeros((self.grating_number,self.grating_repetitions,self.grating_frame_number ))
            self.extended_grating_sliced_voltages_dictionary[activity_matrix]=np.zeros((self.grating_number,self.grating_repetitions,len(self.extended_ranges[0][0])))

        for key, value in self.grating_sliced_voltages_dictionary.items():
          for grat, trials in   self.ranges.items():
              for row, trial in enumerate(trials):
                  self.grating_sliced_voltages_dictionary[key][grat,row,:]=voltage_dictionary[key][trial]  

        for key, value in self.extended_grating_sliced_voltages_dictionary.items():
          for grat, trials in   self.extended_ranges.items():
              for row, trial in enumerate(trials):
                  self.extended_grating_sliced_voltages_dictionary[key][grat,row,:]=voltage_dictionary[key][trial]  
  
  
    def df_f_trial(self,trial_activity):
    
      prestim_activity=trial_activity[:self.pre_frames+1]
      prestim_mean=np.nanmean(prestim_activity)
      prestim_mean2=prestim_mean
      # put something here for std 
      if prestim_mean==0 or prestim_mean<0.002:
          prestim_mean2=1
      df_f_trial_activity=(trial_activity-prestim_mean)/ prestim_mean2
    
      return df_f_trial_activity
    
    def calculate_df_f(self):
  
        self.grating_df_f_percentages={}
        
        for activity_matrix  in  self.activity_matrixes_resampled.keys():
            self.grating_df_f_percentages[activity_matrix]={}
            for plane in self.activity_matrixes_resampled[activity_matrix]:
                self.grating_df_f_percentages[activity_matrix][plane]=np.zeros(self.grating_sliced_dictionary[activity_matrix][plane].shape);

        for activity, planes in self.grating_df_f_percentages.items():
            for plane, matrix  in planes.items():
                for cell in range(self.grating_sliced_dictionary[activity][plane].shape[0]):
                    for grat in range(self.grating_number):
                        for rept in range(self.grating_repetitions):               
                               matrix[cell,grat,rept,:]=self.df_f_trial(self.grating_sliced_dictionary[activity][plane][cell,grat,rept,:])
                           
   
    def trial_averaging_and_evoked_activity(self):
        
        self.grating_df_f_percentages_trial_averaged={}
        self.grating_activity_trial_averaged={}
        self.extended_grating_activity_trial_averaged={}
  
  
        for activity, planes in self.grating_df_f_percentages.items():
            self.extended_grating_activity_trial_averaged[activity]={}
            self.grating_activity_trial_averaged[activity]={}
            self.grating_df_f_percentages_trial_averaged[activity]={}
            for plane, matrix  in planes.items():
            
                self.extended_grating_activity_trial_averaged[activity][plane]=np.nanmean(self.extended_grating_sliced_dictionary[activity][plane],2)
                self.grating_activity_trial_averaged[activity][plane]=np.nanmean(self.grating_sliced_dictionary[activity][plane],2)
                self.grating_df_f_percentages_trial_averaged[activity][plane]=np.nanmean(self.grating_df_f_percentages[activity][plane],2)
  
        # shape is  frames, grating, cell
        
        
        self.mean_response_per_gratings={}
        self.mean_reponse_per_cell={}
        self.mean_df_f_per_gratings={}
        self.mean_df_f_per_cell={}
        
        for activity, planes in self.grating_df_f_percentages.items():
            self.mean_response_per_gratings[activity]={}
            self.mean_reponse_per_cell[activity]={}
            self.mean_df_f_per_gratings[activity]={}
            self.mean_df_f_per_cell[activity]={}

            for plane, matrix  in planes.items():
                
                self.mean_response_per_gratings[activity][plane]=np.nanmean(self.grating_activity_trial_averaged[activity][plane],1).T
                self.mean_reponse_per_cell[activity][plane]=np.nanmean(self.grating_activity_trial_averaged[activity][plane],0).T
                self.mean_df_f_per_gratings[activity][plane]=np.nanmean(self.grating_df_f_percentages_trial_averaged[activity][plane],1).T
                self.mean_df_f_per_cell[activity][plane]=np.nanmean(self.grating_df_f_percentages_trial_averaged[activity][plane],0).T

        self.slice_evoked_activity()
                
        
    def slice_evoked_activity(self):
        self.evoked_slice=slice(self.pre_frames+1,-self.post_frames+2)
        self.evoked_all_activities={}
        
        for activity, planes in self.grating_df_f_percentages.items():
            self.evoked_all_activities[activity]={}
      
            for plane, matrix  in planes.items():
      
                self.evoked_all_activities[activity][plane]={}
                self.evoked_all_activities[activity][plane]['evoked_trial_df_f']=self.grating_df_f_percentages[activity][plane][:,:,:,self.evoked_slice]
                self.evoked_all_activities[activity][plane]['evoked_trial_df_f_mean']=np.nanmean( self.evoked_all_activities[activity][plane]['evoked_trial_df_f'],3)
                self.evoked_all_activities[activity][plane]['evoked_trial_averaged_df_f']=  self.grating_df_f_percentages_trial_averaged[activity][plane][:,:,self.evoked_slice]
                self.evoked_all_activities[activity][plane]['evoked_trial_averaged_df_f_mean']=np.nanmean( self.evoked_all_activities[activity][plane]['evoked_trial_averaged_df_f'],2)
  
        self.evoked_all_activities['speed_trace']={}
        self.evoked_all_activities['speed_trace']['evoked_locomotion']=self.grating_sliced_voltages_dictionary['speed_trace'][:,:,self.evoked_slice]
        self.evoked_all_activities['speed_trace']['evoked_locomotion_mean']=np.nanmean(self.evoked_all_activities['speed_trace']['evoked_locomotion'], axis=2)
    
  
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
        self.running_threshold=np.mean(self.voltage_traces_resampled['speed_trace'])+1.5*np.std(self.voltage_traces_resampled['speed_trace'])
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
        
        # for i in range(self.grating_number):  
        for i in range(5):  
            fig, axss=plt.subplots( self.grating_repetitions, sharex=True)
            for j in range(self.grating_repetitions):               
                # axss[j].plot(activity_matrixes_resampled['grating_binary_matrix'][i,frame_windows[i][j]]) 
                # axss[j].plot(self.voltage_traces_resampled['vistim_trace'][self.ranges[i][j]]) 
                axss[j].plot(self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_recreated'][i,self.ranges[i][j]]) 
            fig.suptitle('Grating : '  + str(i+1))
        

        fig, axs=plt.subplots(3, sharex=True)
        axs[1].imshow(self.activity_matrixes_resampled['dfdtmatrix']['Plane1'],   
                      vmax=np.nanmean(self.activity_matrixes_resampled['dfdtmatrix']['Plane1'])+2*np.std(self.activity_matrixes_resampled['dfdtmatrix']['Plane1']),     aspect='auto', cmap='binary') 
        axs[0].plot(self.voltage_traces_resampled['vistim_trace'])    
        axs[2].plot(self.voltage_traces_resampled['speed_trace'])   
        for i in range(40):
            for j in range(15):
                color = tuple(np.random.choice(range(256), size=3)/256)
                axs[0].plot(self.ranges[i][j], self.voltage_traces_resampled['vistim_trace'][self.frame_windows[i][j]],'x', color=color) 
     
            fig, axss=plt.subplots(1, sharex=True)
            for j in range(15):
                axss.plot(self.voltage_traces_resampled['vistim_trace'][self.frame_windows[i][j]])  
        

    
    def plot_activity_matrixes(self):
        fig, ax = plt.subplots(2, 1, sharex=True)
        # norm1=mlp.colors.Normalize(0, 1)
        # ax[0].imshow(self.allplanesC,     vmax=np.nanmean(self.allplanesC)+2*np.std(self.allplanesC),     aspect='auto', cmap='binary')
        ax[0].imshow(self.activity_matrixes_resampled['dfdtmatrix']['Plane1'],  vmax=np.nanmean(self.activity_matrixes_resampled['dfdtmatrix']['Plane1'])+2*np.std(self.activity_matrixes_resampled['dfdtmatrix']['Plane1']),     aspect='auto', cmap='binary')
        # ax[2].imshow(self.allplanesfoopsi,vmax=np.nanmean(self.allplanesfoopsi)+2*np.std(self.allplanesfoopsi),     aspect='auto', cmap='binary')
        ax[1].imshow(self.activity_matrixes_resampled['dfdtmatrix']['Plane1'],  vmax=np.nanmean(self.activity_matrixes_resampled['dfdtmatrix']['Plane1'])+2*np.std(self.activity_matrixes_resampled['dfdtmatrix']['Plane1']),     aspect='auto', cmap='binary')
        
        
        

    def plot_activity_matrix_with_signals(self, activity_matrix):
        activity_matrix=self.activity_matrixes_resampled['dfdtmatrix']['Plane1']
        
        fig, axs=plt.subplots(3, sharex=True)
        axs[1].imshow(activity_matrix,   vmax=np.nanmean(activity_matrix)+2*np.std(activity_matrix),     aspect='auto', cmap='binary') 
        axs[0].plot(self.voltage_traces_resampled['vistim_trace'])    
        axs[2].plot(self.voltage_traces_resampled['speed_trace'])   
        for i in range(40):
            color = tuple(np.random.choice(range(256), size=3)/256)
            axs[0].plot(np.argwhere(self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_recreated'][i,:]) , self.voltage_traces_resampled['vistim_trace'][np.argwhere(self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_recreated'][i,:])],'x', color=color) 
    
    
    
    
    def plotting2(self):
        fig, axs = plt.subplots(3)
        fig.suptitle('Locomotion')
        axs[0].plot(self.analysis_object.signals_object.second_scale, self.analysis_object.signals_object.rectified_speed_array)
        axs[1].plot(self.analysis_object.signals_object.second_scale, self.analysis_object.signals_object.voltage_signals['VisStim'])
        axs[2].plot(self.analysis_object.signals_object.second_scale, self.analysis_object.signals_object.voltage_signals['PhotoDiode'])
        
        mplcursors.cursor(axs) # or just mplcursors.cursor()
        
    def plot_activity_matrix_with_signals_and_grating_ranges(self, activity_matrix=None):
        
        fig, axs=plt.subplots(3, sharex=True)
        axs[1].imshow(self.activity_matrixes_resampled['dfdtmatrix']['Plane1'],   
                      vmax=np.nanmean(self.activity_matrixes_resampled['dfdtmatrix']['Plane1'])+2*np.std(self.activity_matrixes_resampled['dfdtmatrix']['Plane1']),     aspect='auto', cmap='binary') 
        axs[0].plot(self.voltage_traces_resampled['vistim_trace'])    
        axs[2].plot(self.voltage_traces_resampled['speed_trace'])   
        for i in range(40):
            color = tuple(np.random.choice(range(256), size=3)/256)
            axs[0].plot(np.argwhere(self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_recreated'][i,:]) ,
                        self.voltage_traces_resampled['vistim_trace'][np.argwhere(self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_recreated'][i,:])],'x', color=color) 
        
    def plotting_resampling_accuracy_of_ranges(self):
        # fig, axs = plt.subplots(1,1)
        # axs.imshow(self.gratin1_periods_only,vmin=0, vmax=0.1,aspect='auto',cmap='inferno')
                 
        for stim in range(self.grating_number):
             fig, axs=plt.subplots(2, sharex=True)
             axs[0].plot(self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_downsampled'][stim,:]) 
             axs[1].plot(self.voltage_traces_resampled['vistim_trace']) 
             color = tuple(np.random.choice(range(256), size=3)/256)
             axs[0].plot( self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'][stim,:], 
                           self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_downsampled'][stim,self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'][stim,:]]      ,'x', color=color)      
             axs[0].plot( self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'][stim,:]+1, 
                           self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_downsampled'][stim,self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'][stim,:]+1]      ,'o', color=color) 
             axs[1].plot( self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'][stim,:], 
                           self.voltage_traces_resampled['vistim_trace'][self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'][stim,:]]      ,'x', color=color)      
             axs[1].plot( self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'][stim,:]+1, 
                           self.voltage_traces_resampled['vistim_trace'][self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'][stim,:]+1]      ,'o', color=color)    
             plt.show()
             
    def more_plotting(self, cell, grating, trial):
         
         fig, axs = plt.subplots(nrows=3, sharex=True)          
         # one cell response to 15 trial of single gratin
         axs[0].imshow(self.grating_sliced_dictionary['dfdtmatrix']['Plane1'][:,:,grating,cell],   vmax=np.nanmean(self.grating_sliced_dictionary['dfdtmatrix']['Plane1'])+2*np.std(self.grating_sliced_dictionary['dfdtmatrix']['Plane1']),     aspect='auto', cmap='binary')
         # response of all cells to a single repetion of s sigle stoimuli
         axs[1].imshow(self.grating_sliced_dictionary['dfdtmatrix']['Plane1'][trial,:,grating,:].T, vmax=np.nanmean(self.grating_sliced_dictionary['dfdtmatrix']['Plane1'])+2*np.std(self.grating_sliced_dictionary['dfdtmatrix']['Plane1']),     aspect='auto', cmap='binary')
         # response a cell to a single repetition of all stimuli
         axs[2].imshow(self.grating_sliced_dictionary['dfdtmatrix']['Plane1'][trial,:,:,cell],   vmax=np.nanmean(self.grating_sliced_dictionary['dfdtmatrix']['Plane1'])+2*np.std(self.grating_sliced_dictionary['dfdtmatrix']['Plane1']),     aspect='auto', cmap='binary')

         #trial averaged activctu for all cells  and for a single cell

         fig, axs = plt.subplots(nrows=2,  sharex=True)            
         axs[0].imshow(self.grating_activity_trial_averaged['dfdtmatrix']['Plane1'][:,grating,:], vmax=np.nanmean(self.grating_activity_trial_averaged['dfdtmatrix']['Plane1'])+2*np.std(self.grating_activity_trial_averaged['dfdtmatrix']['Plane1']),     aspect='auto', cmap='binary')
         axs[1].plot(self.grating_activity_trial_averaged['dfdtmatrix']['Plane1'][cell,grating,:])


         fig, axs = plt.subplots(nrows=2)            
         axs[0].imshow(self.mean_response_per_gratings['dfdtmatrix']['Plane1'].T, vmax=np.nanmean(self.mean_response_per_gratings['dfdtmatrix']['Plane1'])+2*np.std(self.mean_response_per_gratings['dfdtmatrix']['Plane1']),     aspect='auto', cmap='binary')
         axs[1].imshow(self.mean_reponse_per_cell['dfdtmatrix']['Plane1'].T,  vmax=np.nanmean(self.mean_reponse_per_cell['dfdtmatrix']['Plane1'])+2*np.std(self.mean_reponse_per_cell['dfdtmatrix']['Plane1']),     aspect='auto', cmap='binary')


    def plot_all_tuning_single_cell(self, celltoplot):
                        
            for gratingtoplot in range(self.grating_number):
                  
                fig, ax =plt.subplots(6,sharex=True)
                ax[0].imshow(self.grating_sliced_dictionary['dfdtmatrix']['Plane1'][:,:,gratingtoplot,celltoplot], 
                             vmax=np.nanmean(self.grating_sliced_dictionary['dfdtmatrix']['Plane1'][:,:,gratingtoplot,celltoplot])+2*np.std(self.grating_sliced_dictionary['dfdtmatrix']['Plane1'][:,:,gratingtoplot,celltoplot]), aspect='auto',cmap='binary')
                ax[1].plot(np.nanmean(self.grating_sliced_dictionary['dfdtmatrix']['Plane1'], axis=0)[:,gratingtoplot,celltoplot])
            
                ax[2].imshow(self.grating_df_f_percentages['dfdtmatrix']['Plane1'][:,:,gratingtoplot,celltoplot],   
                             vmax=np.nanmean(self.grating_df_f_percentages['dfdtmatrix']['Plane1'][:,:,gratingtoplot,celltoplot])+2*np.std(self.grating_df_f_percentages['dfdtmatrix']['Plane1'][:,:,gratingtoplot,celltoplot]),        aspect='auto',cmap='binary')
                ax[3].plot(np.nanmean(self.grating_df_f_percentages['dfdtmatrix']['Plane1'], axis=0)[:,gratingtoplot,celltoplot])
                ax[4].plot(np.nanmean(self.grating_sliced_voltages_dictionary['vistim_trace'], axis=0)[:,gratingtoplot])
                ax[5].plot(np.nanmean(self.grating_sliced_voltages_dictionary['speed_trace'], axis=0)[:,gratingtoplot])
                fig.suptitle('Grating : '  + str(gratingtoplot))

                
    def plot_trial_averaged_single_cell(self, celltoplot):
                        # self.grating_number
            for gratingtoplot in range(2):        

                fig, ax =plt.subplots(3)
                ax[0].imshow( self.grating_df_f_percentages_trial_averaged['dfdtmatrix']['Plane1'][:,gratingtoplot,:].T, 
                             vmax=np.nanmean( self.grating_df_f_percentages_trial_averaged['dfdtmatrix']['Plane1'][:,gratingtoplot,:])+2*np.std( self.grating_df_f_percentages_trial_averaged['dfdtmatrix']['Plane1'][:,gratingtoplot,:]),   aspect='auto',cmap='binary')
            
                ax[1].imshow(self.mean_df_f_per_gratings['dfdtmatrix']['Plane1'][25:30,:],     
                              aspect='auto',cmap='binary')
                ax[2].imshow(self.mean_df_f_per_cell['dfdtmatrix']['Plane1'],       
                              vmax=np.nanmean(self.mean_df_f_per_cell['dfdtmatrix']['Plane1'])+2*np.std(self.mean_df_f_per_cell['dfdtmatrix']['Plane1']),   aspect='auto',cmap='binary')
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



# cell=0
# trial=0
# self.drift_obj.sweep_response[cell][trial]

# drift_obj.drift_obj.sweep_response
# drift_obj.stim_table
# mean_response_info=reponse_info[:,1:,:,0]
# trialresponsive_response_info=reponse_info[:,1:,:,2]

# plt.imshow(mean_response_info[0,:,:], aspect='auto')
# plt.imshow(trialresponsive_response_info[0,:,:], aspect='auto')


# for cell in range(reponse_info.shape[2]):
#     plt.figure()
#     plt.imshow(mean_response_info[:,:,cell], aspect='auto')
#     plt.show()




# # for str(cell) in range(drift_obj.sweep_response.shape[1]):
# for cell in range(10):

#     fig, ax=plt.subplots(1)
#     for trial in drift_obj.stim_table[drift_obj.stim_table.orientation==45].index:

#         ax.plot(drift_obj.sweep_response[str(cell)][trial])
#     plt.show()
    
    
    
# fig, ax=plt.subplots(8)

# for i, ori in enumerate(drift_obj.orivals):
    
#     for trial in drift_obj.stim_table[(drift_obj.stim_table.orientation==ori) & (drift_obj.stim_table.blank_sweep==0)].index:
        
    
#         ax[i].plot(drift_obj.sweep_response[str(cell)][trial])
        
#     ax[i].set_title("Angle: "+str(ori), y=0.8)
#     ax[i].axvspan(0, drift_obj.interlength, facecolor='g', alpha=0.1)
#     ax[i].axvspan(drift_obj.interlength, drift_obj.interlength + drift_obj.sweeplength, facecolor='b', alpha=0.1)






# fig, ax=plt.subplots(8)

# for i, ori in enumerate(drift_obj.orivals):
    
#     for trial in drift_obj.stim_table[(drift_obj.stim_table.orientation==ori) & (drift_obj.stim_table.blank_sweep==0)].index:
        
    
#         ax[i].plot(drift_obj.mean_sweep_response[str(cell)][drift_obj.stim_table[drift_obj.stim_table.orientation==ori].index].tolist())
        
#     ax[i].set_title("Angle: "+str(ori), y=0.8)

                    
        