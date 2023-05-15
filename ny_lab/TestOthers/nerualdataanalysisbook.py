# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:09:58 2021

@author: sp3660
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
# from .caimanSorterYSResults_legacy import CaimanSorterYSResults
# from .voltageSignalsExtractions import VoltageSignalsExtractions
# from .metadata import Metadata
import os
from sklearn.preprocessing import normalize
import numpy as np
import mplcursors
import matplotlib as mlp
# from TestPLot import SnappingCursor
import scipy.signal as sig
from numpy import exp, abs, angle
from scipy import stats, interpolate
import scipy.io
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from .allenAnalysis import AllenAnalysis
from .ensemblesYuriy import EnsemblesYuriy
from .jesusEnsemblesResults import JesusEnsemblesResults
from .sVDEnsemblesResults import SVDEnsemblesResults
from .cRFsResults import CRFsResults
import time
from pprint import pprint

from matplotlib import pyplot as plt
import pandas as pd
import matplotlib as mpl
import glob
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b",'g','y','c','m', 'tab:brown']) 

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'



class ResultsAnalysis():
    """
    This is the clas to integrate several acquisition data and start running the analysis pipelines
    HEre nfor each acqusition do
        Integration of multiple planes
            COncatentae planes
            Integrate tomato and gcamp signlas
            Integrate with other FOV cquisitions
            
        Alignmenet of voltage signals and imageing
        Create a visstim class to integrate info form voltage sigblas and visstim mat files
        Integrate daq signals and prairire voltage sognals
        Integrate with facecam
        
    
    
    """
    
    def __init__(self,  plane1_caiman_sorter_results_object=None, plane2_caiman_sorter_results_object=None,plane3_caiman_sorter_results_object=None, 
                 crf_results_object=None, acquisition_voltage_signals_object=None, metadata_object=None, caiman_results_object=None, allen_results_object=None, acquisition_object=None ):
        
      
        self.caiman_results_object=caiman_results_object
        self.acquisition_object=acquisition_object
        self.jesus_runs={}
        
        
        if self.caiman_results_object:
            self.allen_results_object=allen_results_object
            self.acquisition_voltage_signals=acquisition_voltage_signals_object
            self.metadata_object=metadata_object
            self.plane1_results=plane1_caiman_sorter_results_object
            self.pane2_results=plane2_caiman_sorter_results_object
            self.plane3_results=plane3_caiman_sorter_results_object
            self.crf_results_object=crf_results_object
            self.corrected_traces=self.caiman_results_object.C_matrix
            self.spike_traces=self.caiman_results_object.dfdt_thesholded_accepted_matrix
            self.binary_spikes= self.caiman_results_object.binarized_dfdt
            # savemat(r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\data_analysis\Jesus\Ensemblex\Ensemblex\test_raster_{time}.mat'.format(time= time.strftime("%Y_%m_%d_%H_%M_%S")),{'raster':self.binary_spikes})
            
            '''# TO ADD
            get other planes 
            integrate with voltage
            separta rasters in oparadigms
            separate raster by trials
            separta eby pyramidals and interneurons
            define wich visual stim is
            input is goinfg to be aquisition obejcy load all planes caiman results
            also load red images and masks
            
            
     
            '''
            pass
        
        elif self.acquisition_object:
            #%% acquisition
            
            self.set_up_some_paths()

            # load metadata, voltage signals, reference images, vis stim info in the acquisition object, 
            # this create the objects and loads the voltage signal data  to memory
            self.load_all_acquisition_subobjects()
            self.load_voltage_signals()

            #%% datsets
            self.split_channel_datasets()
            self.load_calcium_extractions()
            self.combine_planes_rough()
            self.signals_object.process_all_signals()

            # self.identify_in_pyr()
            # self.load_jesus_analysis(binary_raster_to_proces='MCMC', plane='All')
  
        else:
           pass
       
        
       
        
       
        
    def set_up_some_paths(self):
        mouse_slow_path=self.acquisition_object.mouse_imaging_session_object.mouse_object.mouse_slow_subproject_path
        self.jesus_runs_path=os.path.join(mouse_slow_path,'data','JesusRuns')
        self.caiman_runs_path=os.path.join(mouse_slow_path,'data','')
        self.nmf_runs_path=os.path.join(mouse_slow_path,'data','')
        self.allen_runs_path=os.path.join(mouse_slow_path,'data','')
        
        
       
    def signal_alignment_testing(self, ):
        
        self.fr=self.acquisition_object.metadata_object.translated_imaging_metadata['FinalFrequency']
        self.separated_planes_combined_gratings_binarized_mcmc={}
        self.separated_planes_combined_gratings_binarized_dfdt={}
        self.interplane_period=3/1000+self.acquisition_object.metadata_object.translated_imaging_metadata['InterFramePeriod']
        

        self.all_planes_timestamps_equivalences={}
    
        for dataset_name in list(self.caiman_results.keys()):
            if 'Plane1' in dataset_name:
                plane=     'Plane1'   
                plane_number=0
            elif 'Plane2' in dataset_name:
                plane=     'Plane2' 
                plane_number=1
            elif 'Plane3' in dataset_name:
                plane=     'Plane3'   
                plane_number=2
            
            selected_plane_caiman=  self.caiman_results[dataset_name]
            selected_binarydfdt=selected_plane_caiman.binarized_dfdt
            selected_binarymcmc=selected_plane_caiman.binarized_MCMC
            selected_plane_prairie_timetamps=self.all_planes_timestamps[plane][0:-1]
            calculated_timestamps= (np.arange(0,len(selected_plane_prairie_timetamps))/self.fr)+self.interplane_period*plane_number
            plt.figure()
            plt.plot(selected_plane_prairie_timetamps, selected_binarydfdt[0,:])
            plt.show()
            self.timestamp_equivalence=np.array([selected_plane_prairie_timetamps, calculated_timestamps])
            plt.figure()

            plt.plot(self.timestamp_equivalence[0,:])
            plt.plot(self.timestamp_equivalence[1,:])
            plt.show()

        self.transfrom_voltage_ms_index_to_frame_index()
        self.resample_voltage_matrixes()
            
    def transfrom_voltage_ms_index_to_frame_index(self):        
            
            self.paradigm_frame_transitions_dictionary={key:(np.abs(self.timestamp_equivalence[0] - index/1000)).argmin() for key, index in self.signals_object.transitions_dictionary.items()}

            if self.signals_object.vis_stim_protocol =='AllenA':
                # firtfirts=(np.abs(selected_plane_prairie_timetamps - self.signals_object.transitions_dictionary['first_drifting_set_first']/1000)).argmin()
                # firstlast=(np.abs(selected_plane_prairie_timetamps - self.signals_object.transitions_dictionary['first_drifting_set_last']/1000)).argmin()
                # secondfirts=(np.abs(selected_plane_prairie_timetamps - self.signals_object.transitions_dictionary['second_drifting_set_first']/1000)).argmin()
                # secondlast=(np.abs(selected_plane_prairie_timetamps - self.signals_object.transitions_dictionary['second_drifting_set_last']/1000)).argmin()
                # thirdfirts=(np.abs(selected_plane_prairie_timetamps - self.signals_object.transitions_dictionary['third_drifting_set_first']/1000)).argmin()
                # thirdlast=(np.abs(selected_plane_prairie_timetamps - self.signals_object.transitions_dictionary['third_drifting_set_last']/1000)).argmin()
                self.sliced_grating_on_indexes=np.vstack([[(np.abs(self.timestamp_equivalence[0] - rep/1000)).argmin()   for rep in ori] for ori in self.signals_object.drifting_on_transition_indexes])
                self.sliced_grating_off_indexes=np.vstack([[(np.abs(self.timestamp_equivalence[0] - rep/1000)).argmin()   for rep in ori] for ori in self.signals_object.drifting_off_transition_indexes])

            elif self.signals_object.vis_stim_protocol =='AllenC':
                # firtfirts=(np.abs(selected_plane_prairie_timetamps - self.signals_object.transitions_dictionary['first_noise_set_first']/1000)).argmin()
                # firstlast=(np.abs(selected_plane_prairie_timetamps - self.signals_object.transitions_dictionary['first_noise_set_last']/1000)).argmin()
                # secondfirts=(np.abs(selected_plane_prairie_timetamps - self.signals_object.transitions_dictionary['second_noise_set_first']/1000)).argmin()
                # secondlast=(np.abs(selected_plane_prairie_timetamps - self.signals_object.transitions_dictionary['second_noise_set_last']/1000)).argmin()
                # thirdfirts=(np.abs(selected_plane_prairie_timetamps - self.signals_object.transitions_dictionary['third_noise_set_first']/1000)).argmin()
                # thirdlast=(np.abs(selected_plane_prairie_timetamps - self.signals_object.transitions_dictionary['third_noise_set_last']/1000)).argmin()
                self.sliced_noise_on_indexes=np.vstack([[(np.abs(self.timestamp_equivalence[0] - rep/1000)).argmin()   for rep in ori] for ori in self.signals_object.noise_on_transition_indexes])
                self.sliced_noise_off_indexes=np.vstack([[(np.abs(self.timestamp_equivalence[0] - rep/1000)).argmin()   for rep in ori] for ori in self.signals_object.noise_off_transition_indexes])
                
    def resample_voltage_matrixes(self):
        
            self.milisecond_period=1000/self.fr

            self.resampled_vistim_matrix=self.resample(self.signals_object.rounded_vis_stim[:], factor=self.milisecond_period, kind='linear').squeeze()
            self.resampled_speed_matrix=self.resample(self.signals_object.rectified_speed_array[:], factor=self.milisecond_period, kind='linear').squeeze()
            self.resampled_acceleration_matrix=self.resample(self.signals_object.acceleration_array    [:], factor=self.milisecond_period, kind='linear').squeeze()    
            
            if self.signals_object.vis_stim_protocol =='AllenA':
                self.resampled_gratings_matrix=self.resample(np.concatenate((self.signals_object.first_drifting_set, self.signals_object.second_drifting_set, self.signals_object.third_drifting_set)), factor=self.milisecond_period, kind='linear').squeeze()
                self.resampled_slices_speed_matrix=self.resample(np.concatenate((self.signals_object.first_drifting_set, self.signals_object.second_drifting_set, self.signals_object.third_drifting_set)), factor=self.milisecond_period, kind='linear').squeeze()
                self.resampled_stim_matrix=self.resampled_gratings_matrix
                
            elif self.signals_object.vis_stim_protocol =='AllenC':

                self.resampled_noise_matrix=self.resample(np.concatenate((self.signals_object.first_noise_set, self.signals_object.second_noise_set, self.signals_object.third_noise_set)), factor=self.milisecond_period, kind='linear').squeeze()
                self.resampled_slices_speed_matrix=self.resample(np.concatenate((self.signals_object.first_noise_set, self.signals_object.second_noise_set, self.signals_object.third_noise_set)), factor=self.milisecond_period, kind='linear').squeeze()
                self.resampled_stim_matrix=self.resampled_noise_matrix


    def slice_activity_matrix(self):
        pass
       
            # self.separated_planes_combined_gratings_binarized_dfdt[dataset_name]=np.concatenate((selected_binarydfdt[:,firtfirts:firstlast], selected_binarydfdt[:,secondfirts:secondlast], selected_binarydfdt[:,thirdfirts:thirdlast]), axis=1)
            # self.separated_planes_combined_gratings_binarized_mcmc[dataset_name]=np.concatenate((selected_binarymcmc[:,firtfirts:firstlast], selected_binarymcmc[:,secondfirts:secondlast], selected_binarymcmc[:,thirdfirts:thirdlast]), axis=1)



            # for ori in range(1):
            #     for rep in range(15):
            #         self.grating_0_rep_0_activity=selected_binarydfdt[:,sliced_grating_on_indexes[ori,rep]-16:sliced_grating_off_indexes[ori,rep]+16]

            #         pixel_per_bar = 4
            #         dpi = 100
                    
            #         fig,ax = plt.subplots(2)
            #         ax[0].plot(self.resampled_stim_matrix[sliced_grating_on_indexes[ori,rep]-15:sliced_grating_off_indexes[ori,rep]+15])
            #         ax[1].imshow( self.grating_0_rep_0_activity, aspect='auto')
            #         ax[1].set_xlabel('Time(s)')
            #         fig.supylabel('Cell Number')
            #         fig.suptitle('Raster dfdt')
         
          
            # # movie1_binarized_activity=np.concatenate((selected_binarydfdt[:,firtfirts:firstlast], selected_binarydfdt[:,secondfirts:secondlast], selected_binarydfdt[:,thirdfirts:thirdlast]), axis=1)
            # # combined_movie3_binarized_activity=np.concatenate((selected_binarydfdt[:,firtfirts:firstlast], selected_binarydfdt[:,secondfirts:secondlast], selected_binarydfdt[:,thirdfirts:thirdlast]), axis=1)
            # # spont_binarized_activity=np.concatenate((selected_binarydfdt[:,firtfirts:firstlast], selected_binarydfdt[:,secondfirts:secondlast], selected_binarydfdt[:,thirdfirts:thirdlast]), axis=1)
            
            # pixel_per_bar = 4
            # dpi = 100
            
            # # fig = plt.figure(figsize=(6+(200*pixel_per_bar/dpi), 10), dpi=dpi)
            # fig = plt.figure(figsize=(16,16))
            # ax1 = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
            # ax1.plot(self.resampled_vistim_matrix)
            # ax1.margins(x=0)
            # plt.show()

            # fig = plt.figure(figsize=(10,10))

            # ax2 = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
            # ax2.plot(self.resampled_speed_matrix)
            # ax2.margins(x=0)
            # plt.show()

            # fig = plt.figure(figsize=(16,16))
            # ax3 = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
            # ax3.imshow(selected_binarydfdt, aspect='auto')
            # ax3.set_xlabel('Time(s)')
            # fig.supylabel('Cell Number')
            # fig.suptitle('Raster dfdt')
            # plt.show()

            
            # fig = plt.figure(figsize=(16,16))
            # ax3 = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
            # ax3.imshow(selected_binarymcmc, aspect='auto')
            # ax3.set_xlabel('Time(s)')
            # fig.supylabel('Cell Number')
            # fig.suptitle('Raster MCMC')
            # plt.show()

            
            
            # pixel_per_bar = 4
            # dpi = 100
            
            # # fig = plt.figure(figsize=(6+(200*pixel_per_bar/dpi), 10), dpi=dpi)
            # fig = plt.figure(figsize=(10,10))
            # ax1 = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
            
           
            
            # ax1.plot(self.resampled_stim_matrix)
            # ax1.margins(x=0)
            # ax1.set_xlabel('Time(s)')
            
            # plt.show()
            # fig = plt.figure(figsize=(10,10))

            # ax2 = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
            # ax2.plot(self.resampled_slices_speed_matrix)
            # ax2.margins(x=0)
            # ax2.set_xlabel('Time(s)')
            # plt.show()


            # fig = plt.figure(figsize=(10,10))

            # ax3 = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
            # ax3.imshow(self.separated_planes_combined_gratings_binarized_dfdt[dataset_name], aspect='auto')
            # ax3.set_xlabel('Time(s)')
            # fig.supylabel('Cell Number')
            # plt.show()

            # fig = plt.figure(figsize=(10,10))

            # ax3 = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
            # ax3.imshow(self.separated_planes_combined_gratings_binarized_mcmc[dataset_name], aspect='auto')
            # ax3.set_xlabel('Time(s)')
            # fig.supylabel('Cell Number')
            # fig.suptitle('Sliced  Activity')
            # plt.show()

            







    def some_ploting(self):
        dataset_name=list(self.caiman_results.keys())[0]
        fig, ax=plt.subplots(3)
        
        
        
        
        
        ax[0].plot(self.resampled_stim_matrix)
        ax[0].margins(x=0)
        ax[1].imshow(self.separated_planes_combined_gratings_binarized_mcmc[dataset_name], aspect='auto')
        ax[2].imshow(self.separated_planes_combined_gratings_binarized_dfdt[dataset_name], aspect='auto')

    def plot_full_signals_and_raster(self):
        
        pass
        

        
             
    def resample(self, x, factor, kind='linear'):
        n = int(np.ceil(x.size / factor))
        f = interpolate.interp1d(np.linspace(0, 1, x.size), x, kind)
        return f(np.linspace(0, 1, n))       
                   
    def load_calcium_extractions(self):
        
        self.caiman_extractions={key:dataset.most_updated_caiman for key,dataset in self.calcium_datasets.items()}
        for caiman_extr in  self.caiman_extractions.values(): caiman_extr.load_results_object()
        self.caiman_results={key:caiman_extr.CaimanResults_object for key,caiman_extr in self.caiman_extractions.items()}
        for caiman_res in  self.caiman_results.values(): caiman_res.load_all()
          
            
    def load_all_acquisition_subobjects(self):
        self.acquisition_object.load_all(camera=False, kalman=False)
        self.acquisition_object.metadata_object.get_timestamps()
        self.all_planes_timestamps= self.acquisition_object.metadata_object.timestamps


     
    def split_channel_datasets(self):
        self.calcium_datasets={key:values for key,values in   self.acquisition_object.all_datasets.items() if 'Green' in key }
        self.tomato_datasets={key:values for key,values in   self.acquisition_object.all_datasets.items() if 'Red' in key }
        
    def load_voltage_signals(self):

        #%getting voltageextractions
        # reference to voltage extraction object
        volt_object= self.acquisition_object.voltage_signal_object
        # load voltage signaks proicesiing object
        volt_object.signal_extraction_object()
        # reference to voltage extraction oject
        self.signals_object=volt_object.extraction_object
        #get timescale in seconds(not in miliseconds())
        self.signals_object.choose_time_scale('secs')
        # reference to speed and vis stim arrasr
        self.speed=self.signals_object.rectified_speed_array
        self.visstim=self.signals_object.visualstim_array
        

    def combine_planes_rough(self):
        self.combine_plane_binarized()
        self.combine_plane_deconvolved()
        self.combine_plane_denoised_traces()
        
    def combine_plane_binarized(self):
        
        self.binarized_dfdt_combined_planes=np.empty((0,list(self.caiman_results.values())[0].binarized_dfdt.shape[1]))
        self.binarized_mcmc_combined_planes=np.empty((0,list(self.caiman_results.values())[0].binarized_MCMC.shape[1]))
        
  
        for key, caiman_res in  self.caiman_results.items():
            self.binarized_dfdt_combined_planes=np.concatenate((self.binarized_dfdt_combined_planes,  caiman_res.binarized_dfdt), axis=0)
            self.binarized_mcmc_combined_planes=np.concatenate((self.binarized_mcmc_combined_planes,  caiman_res.binarized_MCMC), axis=0)

    
    def combine_plane_deconvolved(self):
        
        self.dfdt_combined_planes=np.empty((0,list(self.caiman_results.values())[0].dfdt_accepted_matrix.shape[1]))
        self.mcmc_combined_planes=np.empty((0,list(self.caiman_results.values())[0].convolved_MCMC.shape[1]))    
        for key, caiman_res in  self.caiman_results.items():
            self.dfdt_combined_planes=np.concatenate((self.dfdt_combined_planes,  caiman_res.dfdt_accepted_matrix))
            self.mcmc_combined_planes=np.concatenate((self.mcmc_combined_planes,  caiman_res.convolved_MCMC))



    def combine_plane_denoised_traces(self):
        self.denoised_C_combined_planes=np.empty((0,list(self.caiman_results.values())[0].C_matrix.shape[1]))
        
        for key, caiman_res in  self.caiman_results.items():
            self.denoised_C_combined_planes=np.concatenate((self.denoised_C_combined_planes,  caiman_res.C_matrix))

    def combine_plane_denoised_traces(self):
        self.demixed_CYr_combined_planes=self.denoised_C_combined_planes=np.empty((0,list(self.caiman_results.values())[0].raw.shape[1]))
        for key, caiman_res in  self.caiman_results.items():
            self.demixed_CYr_combined_planes=np.concatenate((self.demixed_CYr_combined_planes,  caiman_res.raw))
    
    def identify_in_pyr(self):
            
        for key,dataset in self.calcium_datasets.items():
            dataset.find_associated_fov_tomato_dataset()
            dataset.find_associated_channel_dataset()
        
            if hasattr(dataset, 'associated_channel_dataset_object'):
                dataset.associated_tomato_dataset=dataset.associated_channel_dataset_object
            elif hasattr(dataset, 'associated_fov_tomato_dataset_red_object'):
                dataset.associated_tomato_dataset=dataset.associated_fov_tomato_dataset_red_object
            else:
                print('No associated channel')
        for caiman_res in  self.caiman_results.values(): caiman_res.plot_registered_two_color_projections()
            

       

    def separate_cell_types_rasters_by_plane(self):
        
        pass
    
    def separate_paradigm_rasters(self):
        pass
  
    def load_allen_analysis(self):
        self.allen_analysis=AllenAnalysis(self)
              
    def load_CRFs_analysis(self):
        self.crf_analysis=CRFsResults(self)
        pass
    
    def load_yuriy_analysis(self):
        self.yuriy_analysis=EnsemblesYuriy(self)
        pass
    
    def load_jesus_analysis(self, binary_raster_to_proces, plane, segment):   
        # self.load_jesus_results()

        if binary_raster_to_proces=='MCMC':
            if plane=='All':
                selected_raster= self.binarized_mcmc_combined_planes
            else:
                for dataset_name in list(self.caiman_results.keys()):
                    if plane in dataset_name:
                        if segment=='Full':
                            selected_raster=self.caiman_results[dataset_name].binarized_MCMC
                        elif segment=='DriftingGratings':
                            selected_raster=  self.separated_planes_combined_gratings_binarized_mcmc[dataset_name]

        elif binary_raster_to_proces=='dfdt':
            if plane=='All':
                selected_raster= self.binarized_dfdt_combined_planes
                
            else:
                 for dataset_name in list(self.caiman_results.keys()):
                     if plane in dataset_name:
                         if segment=='Full':
                             selected_raster=self.caiman_results[dataset_name].binarized_dfdt
                         elif segment=='DriftingGratings':
                             selected_raster=  self.separated_planes_combined_gratings_binarized_dfdt[dataset_name]
   
            
        elif binary_raster_to_proces=='foops':
            pass

        self.jesus_binary_spikes=selected_raster
        self.jesus_analysis=JesusEnsemblesResults(self)
        # self.jesus_runs[ self.run_number+'_'+self.acquisition_object.aquisition_name+'_'+binary_raster_to_proces+'_'+plane+'_'+segment]=[binary_raster_to_proces, plane, segment, self.jesus_binary_spikes, self.jesus_analysis.analysis]
        self.jesus_run=[binary_raster_to_proces, plane, segment, self.jesus_analysis.analysis]
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.jesus_runs_path_name='_'.join([self.acquisition_object.aquisition_name, timestr,binary_raster_to_proces, plane, segment,'jesus_results.pkl'])  


        self.save_jesus_runs()
        
    def save_jesus_runs(self):
        print('Saving jesus run')
        datapath=os.path.join(self.jesus_runs_path, self.jesus_runs_path_name)
    
        with open(datapath, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.jesus_run, f, pickle.HIGHEST_PROTOCOL)
        
    def check_all_jesus_results(self) :
        self.jesus_results_list=glob.glob(self.jesus_runs_path+'\\**', recursive=False)
            
        
    def load_jesus_results(self, path):
        if path:
            with open( path, 'rb') as file:
                self.jesus_runs[os.path.split(path)[1]]= JesusEnsemblesResults(self, path)
        else:
            self.jesus_runs={}
            print('no previous jesus runs')
        
    def load_classicalsvd_analysis(self):
        self.svd_analysis=SVDEnsemblesResults(self)
        pass

      
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
    # SPKG_1015_allen_plane1=CaimanSorterYSResults(temporary_path1+ '211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_MC_OnACID_20211020-013317_cnmf_results_sort.mat')
    # SPKG_1015_allen_plane2=CaimanSorterYSResults(temporary_path2+ '211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_MC_OnACID_20211020-123307_cnmf_results_sort.mat')
    # SPKG_1015_allen_plane3=CaimanSorterYSResults(temporary_path3+ '211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_MC_OnACID_20211020-164443_cnmf_results_sort.mat')
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

    # meta =Metadata(acquisition_directory_raw=temporary_path1)


# voltage signals

    # temporary_path1=linux_temp +os.sep+'210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'
    temporary_path1=windowstemp +os.sep+'211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'

    # voltagesignals=VoltageSignalsExtractions(temporary_path1)

#% RUN ANALYSIS CLASS

    # analysis=ResultsAnalysis(SPJA_0702_allen_plane1, SPJA_0702_allen_plane2, SPJA_0702_allen_plane3, SPJA_0702_allen_CRFS, voltagesignals, meta)
    # analysis=ResultsAnalysis(SPKG_1015_allen_plane1, SPKG_1015_allen_plane2, SPKG_1015_allen_plane3, acquisition_voltage_signals_object=voltagesignals, metadata_object=meta )
# 
    #% plotting
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
    
    
    # analysis.pcns_plotting()
#%
   
    
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



    
 #%
    # fig,axing =plt.subplots(nrows=5, figsize=(10,6))
    
    # grating=5
    # cell=3
    # axing[0].plot(np.mean(analysis.grating_extended_sliced_arrays['dfdtmatrix'],0)[:,grating,cell])
    # axing[1].plot(np.mean(analysis.grating_extended_sliced_arrays['rawmatrix'],0)[:,grating,cell])
    # axing[2].plot(np.mean(analysis.grating_extended_sliced_arrays['mcmcmatrix'],0)[:,grating,cell])

    # axing[3].plot(np.mean(analysis.grating_extended_sliced_traces['speed_trace'],0)[:,grating])
    # axing[4].plot(np.mean(analysis.grating_extended_sliced_traces['vistim_trace'],0)[:,grating])

    # fig,axing =plt.subplots(nrows=2, figsize=(10,6))
    # axing[0].plot(analysis.voltage_traces_resampled_bordercuts['speed_trace'])
    # axing[1].plot(analysis.voltage_traces_resampled_bordercuts['speed_trace'])
    # axing[1].set_ylim(np.mean(analysis.voltage_traces_resampled_bordercuts['speed_trace'])+1.5*np.std(analysis.voltage_traces_resampled_bordercuts['speed_trace']),1)

