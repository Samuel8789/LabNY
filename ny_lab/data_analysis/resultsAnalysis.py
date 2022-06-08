# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:09:58 2021

@author: sp3660
"""
import networkx as nx
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import glob

import tkinter as Tkinter
import random
from tkinter import *
import pandas as pd
import tkinter as tk
import numpy as np


from matplotlib import gridspec
# import torch
# import torch.nn as nn
from random import sample, random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, pearsonr, spearmanr, ttest_ind, zscore
import copy
from scipy import interpolate
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage.filters import gaussian_filter1d
from scipy.io import loadmat, savemat
import skimage.io
import math
# from .caimanSorterYSResults_legacy import CaimanSorterYSResults
# from .voltageSignalsExtractions import VoltageSignalsExtractions
# from .metadata import Metadata
from sklearn.preprocessing import normalize
import mplcursors
# from TestPLot import SnappingCursor
from numpy import exp, abs, angle
from scipy import stats, interpolate
import scipy.io
import warnings
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation 
from IPython.display import HTML


try:
    from .allenAnalysis import AllenAnalysis
    from .ensemblesYuriy import EnsemblesYuriy
    from .jesusEnsemblesResults import JesusEnsemblesResults
    from .sVDEnsemblesResults import SVDEnsemblesResults
    from .cRFsResults import CRFsResults
    from .selectFullData import SelectFullData
    
except:
    from allenAnalysis import AllenAnalysis
    from ensemblesYuriy import EnsemblesYuriy
    from jesusEnsemblesResults import JesusEnsemblesResults
    from sVDEnsemblesResults import SVDEnsemblesResults
    from cRFsResults import CRFsResults
    from selectFullData import SelectFullData

        
import time
from pprint import pprint
from allensdk.brain_observatory.dff import calculate_dff
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import umap
import seaborn as sns

import matplotlib as mpl
import glob

import importlib.util
import sys


mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b",'g','y','c','m', 'tab:brown']) 
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'


#%%

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
        
      attributes
           data objects:
               self.acquisition_object
               self.volt_object
               self.signals_object
               self.calcium_datasets
               self.tomato_datasets
               associated_tomato_aquisitions TO DO
           outputs
               self.full_data
               self.pyr_int_identification
               
               
           self.full_data['imaging_data']['Plane1']['CellIds'] is pythonic index
           
        
    
    
    """
    
    def __init__(self,  
                 acquisition_voltage_signals_object=None,
                 metadata_object=None,
                 acquisition_object=None, 
                 full_data_path=None ,
                 nondatabase=None,
                 new_full_data=False,
                 allen_BO_tuple=False
                 ):      
        self.allen_BO_tuple=allen_BO_tuple
        self.acquisition_object=acquisition_object
        self.nondatabase=nondatabase
        self.new_full_data=new_full_data
        self.acquisition_voltage_signals_object=acquisition_voltage_signals_object
        self.metadata_object=metadata_object
        
        self.jesus_runs={}
        self.temporary_processing=r'C:\Users\sp3660\Desktop\TemporaryProcessing'
        self.caiman_results=None
        self.pyr_int_ids_and_indexes=None
        self.set_up_some_paths()

        
        
        if self.acquisition_object:
            self.metadata_object=self.acquisition_object.metadata_object
            self.acquisition_object.voltage_signal_object
            self.volt_object=self.acquisition_object.voltage_signal_object
            #% acquisition
            # load metadata, voltage signals, reference images, vis stim info in the acquisition object, 
            # this create the objects and loads the voltage signal data  to memory
            self.load_all_data_objects()

      
            #% combine all datasets
            self.signals_object.process_all_signals()
            self.create_full_data_container()
            self.create_stim_table()

      
            #%  pyr int identification

            self.check_pyr_int_identif_files()
            self.load_pyr_int_identif()
            self.get_pyr_int_indexing_dict()
            
            
            # subanalysis, this should be mode to sub anlayis object
            self.check_all_jesus_results()

        elif self.allen_BO_tuple:
            

            self.allen, self.data_set, self.spikes=self.allen_BO_tuple
            
            self.cell_info, self.traces, self.neuropilinfo, self.locomotion_info, self.metadata, self.pupilinfo,self.masks, self.projection=self.allen.dataset_exploration(self.data_set)
            # test=allen.allen_manifest.get_cell_specimens(ids=cell_info[1][0])
            self.stim_info=self.allen.exploring_stimulus(self.data_set)
        
            self.allen.plotting_traces_and_stim(self.data_set, self.traces[-1], self.locomotion_info[0])
            self.drifting=self.allen.explore_drifting_analysis( self.data_set, self.traces[-1], self.spikes, selected_cell_index=0, plot=False)
            self.spont=self.allen.explore_spontaneous_activity( self.data_set, self.traces[-1], self.spikes, selected_cell_index=0, plot=False)
            self.movies=self.allen.explore_natural_movie_analysis( self.data_set, self.traces[-1], self.spikes, selected_cell_index=0, plot=False)
            
            
        else:   
            
            if self.acquisition_voltage_signals_object:
                self.volt_object=self.acquisition_voltage_signals_object
                self.load_voltage_signals()

            if self.metadata_object:
                self.create_full_data_container()
#%% path managing
    def set_up_some_paths(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")

        if self.acquisition_object:
            mouse_slow_path=self.acquisition_object.mouse_imaging_session_object.mouse_object.mouse_slow_subproject_path
            self.jesus_runs_path=os.path.join(mouse_slow_path,'data','JesusRuns')
            self.caiman_runs_path=os.path.join(mouse_slow_path,'data','CaimanRuns')
            self.nmf_runs_path=os.path.join(mouse_slow_path,'data','NMFAnalysis')
            self.allen_runs_path=os.path.join(mouse_slow_path,'data','AllenAnalysis')
            self.data_analysis_path=os.path.join(mouse_slow_path,'data')
            self.full_data_path_name='_'.join([self.acquisition_object.aquisition_name, timestr,'full_data.pkl'])  
            self.pyr_int_identif_path_name='_'.join([self.acquisition_object.aquisition_name, timestr,'pyr_int_identification.pkl'])  
            
        else:
            self.jesus_runs_path=self.temporary_processing
            self.caiman_runs_path=self.temporary_processing
            self.nmf_runs_path=self.temporary_processing
            self.allen_runs_path=self.temporary_processing
            self.data_analysis_path=self.temporary_processing
            self.full_data_path_name=None
            self.pyr_int_identif_path_name=None
            
#%% loading aquisition objects(calcium results and signals)
    def load_all_data_objects(self):
        self.load_all_acquisition_subobjects()
        self.load_calcium_extractions()
        # self.check_caiman_rasters()
        self.load_voltage_signals()

    def load_all_acquisition_subobjects(self):
        self.acquisition_object.load_all(camera=False, kalman=False)
        self.acquisition_object.metadata_object.get_timestamps()
        self.all_planes_timestamps= self.acquisition_object.metadata_object.timestamps
        self.split_channel_datasets()

    def split_channel_datasets(self):
        self.calcium_datasets={key:values for key,values in   self.acquisition_object.all_datasets.items() if 'Green' in key }
        self.tomato_datasets={key:values for key,values in   self.acquisition_object.all_datasets.items() if 'Red' in key }

    def load_calcium_extractions(self):
        self.caiman_extractions={key:dataset.most_updated_caiman for key,dataset in self.calcium_datasets.items()}
        for caiman_extr in  self.caiman_extractions.values(): caiman_extr.load_results_object()
        self.caiman_results={key:caiman_extr.CaimanResults_object for key,caiman_extr in self.caiman_extractions.items()}
        for caiman_res in  self.caiman_results.values(): caiman_res.load_all()
          
    def load_voltage_signals(self):
        # load voltage signaks proicesiing object
        self.volt_object.signal_extraction_object()
        # reference to voltage extraction oject
        self.signals_object=self.volt_object.extraction_object
        self.signals_object.update_frame_rates_with_metadata(self.metadata_object.translated_imaging_metadata['VoltageRecordingFrequency'], 1000)
        self.signals_object.choose_time_scale('secs')

    def update_voltage_signals_frame_rates(self,prairire_frame_rate, daq_frame_rate):
        
        self.signals_object.update_frame_rates_with_metadata(prairire_frame_rate, daq_frame_rate)
        
        
#%% checking datasets
    def check_caiman_rasters(self):
    
        for dataset in self.caiman_extractions.values():
               dataset.CaimanResults_object.plot_final_rasters()
     
#%% full data organization       
    def create_full_data_container(self):
        
        self.check_full_data()
        self.load_full_data()


        if not self.full_data or self.new_full_data:
            
            self.full_data={'imaging_data':{'Frame_rate':self.metadata_object.translated_imaging_metadata['FinalFrequency'],
                                            'Interplane_period':''
                                            },
                            'voltage_traces':{},
                            'visstim_info':{}}
            
       
            if self.metadata_object.translated_imaging_metadata['InterFramePeriod']:
                self.full_data['imaging_data']['Interplane_period']=3/1000+self.metadata_object.translated_imaging_metadata['InterFramePeriod']
                

                
            if self.caiman_results:
                self.extract_calcium_traces()
                
            if self.signals_object:
                self.extrac_voltage_visstim_signals()
            
            if not self.nondatabase:
                self.save_full_data()
            
        self.milisecond_period=1000/self.full_data['imaging_data']['Frame_rate']

#%% loading and saving full data dictionary
    def reload_other_full_data(self, index):
        
        self.select_full_data(index)
        self.create_stim_table()
        self.check_pyr_int_identif_files()
        self.load_pyr_int_identif()
        self.get_pyr_int_indexing_dict()
        self.check_all_jesus_results()


    def select_full_data(self, index):

        # app = SelectFullData(self)
        # app.mainloop()
        # app.wait_window()

        # self.selected_full_data_path=app.selected_full_data
        self.selected_full_data_path= self.full_data_list[index]

    def save_full_data(self):
        datapath=os.path.join(self.data_analysis_path, self.full_data_path_name)
        if not os.path.isfile(datapath) or self.new_full_data:
            with open(datapath, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self.full_data, f, pickle.HIGHEST_PROTOCOL)
                
    def check_full_data(self) :
        self.full_data_list=glob.glob(self.data_analysis_path+'\\**full_data**', recursive=False)

    def load_full_data(self):
        if self.full_data_list:
            self.select_full_data(-1)
            with open( self.selected_full_data_path, 'rb') as file:
                self.full_data=  pickle.load(file)
        else:
            self.full_data={}
            print('no previous fulldata')
            
#%% calcium data managing
    def extract_calcium_traces(self):
        self.pyr_int_identification={}
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
            self.pyr_int_identification[plane]={'interneuron':{'matlab':'',
                                                               'python':'',
                                                               },
                                                'pyramidals':{'matlab':'',
                                                              'python':'',
                                                              }}
            self.full_data['imaging_data'][plane]={}
            selected_plane_caiman=  self.caiman_results[dataset_name]
            
            plane_traces={'demixed':selected_plane_caiman.raw, 
                          'denoised':selected_plane_caiman.C_matrix, 
                           # 'df/f_demixed':calculate_dff(selected_plane_caiman.raw), 
                           # 'df/f_denoised':calculate_dff(selected_plane_caiman.C_matrix), 
                          'dfdt_raw':selected_plane_caiman.dfdt_unsmoothed, 
                          'dfdt_smoothed':selected_plane_caiman.dfdt_thesholded_accepted_matrix, 
                          'dfdt_binary':selected_plane_caiman.binarized_dfdt,
                          'foopsi_raw':selected_plane_caiman.foopsi_matrix, 
                          'foopsi_smoothed':selected_plane_caiman.convolved_foopsi,
                          'foopsi_binary':selected_plane_caiman.binarized_foospi,
                          'mcmc_raw':selected_plane_caiman.MCMC_matrix,
                          'mcmc_smoothed': selected_plane_caiman.convolved_MCMC,
                          'mcmc_binary':selected_plane_caiman.binarized_MCMC}
            
            self.full_data['imaging_data'][plane]['Traces']=plane_traces
            self.full_data['imaging_data'][plane]['Timestamps']=(self.all_planes_timestamps[plane][0:-1], (np.arange(0,len(self.all_planes_timestamps[plane][0:-1]))/self.full_data['imaging_data']['Frame_rate'])+self.full_data['imaging_data']['Interplane_period']*plane_number)
            self.full_data['imaging_data'][plane]['CellNumber']=selected_plane_caiman.accepted_cells_number
            self.full_data['imaging_data'][plane]['CellIds']=selected_plane_caiman.final_accepted_cells_matlabcorrected_indexes
            # self.full_data['imaging_data'][plane]['Pyr_Int_Ident']= TO DO
            
            if plane_number==0:
                self.full_data['imaging_data']['All_planes_rough']={}
                self.full_data['imaging_data']['All_planes_rough']['Traces']= {key:np.empty((0, len(self.full_data['imaging_data']['Plane1']['Timestamps'][0]))) for key in plane_traces.keys()}
                # self.full_data['imaging_data']['All_planes_rough']['CellIds']='' TO DO
                self.full_data['imaging_data']['All_planes_timestamped']={}
                self.full_data['imaging_data']['All_planes_timestamped']['Traces']= {key:np.empty((0, len(self.full_data['imaging_data']['Plane1']['Timestamps'][0]))) for key in plane_traces.keys()}
            
            for matrix_name, matrix in  self.full_data['imaging_data'][plane]['Traces'].items():
                if matrix.any():
                    self.full_data['imaging_data']['All_planes_rough']['Traces'][matrix_name]=np.concatenate((self.full_data['imaging_data']['All_planes_rough']['Traces'][matrix_name],  matrix), axis=0)
        

#%% signal data managing
    def resample(self, x, factor, kind='linear'):
        n = int(np.ceil(x.size / factor))
        f = interpolate.interp1d(np.linspace(0, 1, x.size), x, kind)
        return f(np.linspace(0, 1, n))     

    def extrac_voltage_visstim_signals(self):
        self.milisecond_period=1000/self.full_data['imaging_data']['Frame_rate']
        voltagerate=self.metadata_object.translated_imaging_metadata['VoltageRecordingFrequency']
        self.full_data['voltage_traces']['Speed']=self.resample(self.signals_object.rectified_speed_array['Prairire']['Locomotion'][:], factor=self.milisecond_period, kind='linear').squeeze()
        self.full_data['voltage_traces']['Acceleration']=self.resampled_acceleration_matrix=self.resample(self.signals_object.rectified_acceleration_array['Prairire']['Locomotion'][:], factor=self.milisecond_period, kind='linear').squeeze() 
        if self.signals_object.rounded_vis_stim:
            self.full_data['voltage_traces']['VisStim']=self.resample(self.signals_object.rounded_vis_stim['Prairire']['VisStim'][:], factor=self.milisecond_period, kind='linear').squeeze()
        self.full_data['voltage_traces']['Photodiode']=''
        self.full_data['voltage_traces']['Start_End']=''
        self.full_data['voltage_traces']['LED']=''
        self.full_data['voltage_traces']['Optopockels']=''
        self.full_data['voltage_traces']['OptoTrigger']=''

        if self.signals_object.vis_stim_protocol and self.signals_object.transitions_dictionary:
            self.full_data['visstim_info']['Paradigm_Indexes']={key:(np.abs(self.full_data['imaging_data']['Plane1']['Timestamps'][0] - index/1000)).argmin() for key, index in self.signals_object.transitions_dictionary.items()}
            self.full_data['visstim_info']['Movie1']={'Indexes':'',
                                                      'Binary_Maytrix':''
                                                        }
            self.full_data['visstim_info']['Spontaneous']={}
            self.full_data['visstim_info']['Spontaneous']['stimulus_table']= pd.DataFrame( ([self.full_data['visstim_info']['Paradigm_Indexes']['spont_first'],self.full_data['visstim_info']['Paradigm_Indexes']['spont_last']] ,), columns =['start', 'end'])
    
            if self.signals_object.vis_stim_protocol =='AllenA':
            
                self.full_data['visstim_info']['Drifting_Gratings']={'Indexes':{'Drift_on':np.vstack([[(np.abs( self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.tuning_stim_on_index_full_recording]),
                                                                                'Drift_off':np.vstack([[(np.abs(self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.tuning_stim_off_index_full_recording]),
                                                                                'Blank_sweep_on':np.vstack([[(np.abs( self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.blank_sweep_on_index_full_recording]),
                                                                                'Blank_sweep_off':np.vstack([[(np.abs( self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.blank_sweep_off_index_full_recording])
                                                                                },
                                                                     'Binary_Maytrix_downsampled':np.vstack([self.resample(self.signals_object.full_stimuli_binary_matrix[srtim], 
                                                                                                                           factor=self.milisecond_period, kind='linear').squeeze() for srtim in range (self.signals_object.full_stimuli_binary_matrix.shape[0])]),
                                                                     'Binary_Maytrix_recreated':'',
                                                                     'Resampled_sliced_speed':self.resample(np.concatenate((self.signals_object.first_drifting_set_speed, 
                                                                                                                            self.signals_object.second_drifting_set_speed, 
                                                                                                                            self.signals_object.third_drifting_set_speed)), factor=self.milisecond_period, kind='linear').squeeze(),
                                                                     'Resampled_sliced_visstim':self.resample(np.concatenate((self.signals_object.first_drifting_set, 
                                                                                                                              self.signals_object.second_drifting_set,
                                                                                                                              self.signals_object.third_drifting_set)), factor=self.milisecond_period, kind='linear').squeeze()
                                                                     
                                                                     }
                self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_recreated']=np.zeros((self.signals_object.full_stimuli_binary_matrix.shape ))
                for i, row in enumerate(self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on']):
                    for j, trial in enumerate(row):
                        self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_recreated'][i,self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'][i,j]:self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_off'][i,j]]=1
                
            
                self.full_data['visstim_info']['Movie3']={'Indexes':'',
                                                          'Binary_Maytrix':''
                                                            }
              
            elif self.signals_object.vis_stim_protocol =='AllenC':
            
                self.full_data['visstim_info']['Sparse_Noise']={'Indexes':{'Noise_on':np.vstack([[(np.abs( self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.noise_on_transition_indexes]),
                                                                           'Noise_off':np.vstack([[(np.abs(self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.noise_off_transition_indexes]),
                                                                                },
                                                          'Binary_Maytrix':'',
                                                          'Ref_matrix':'',
                                                          
                                                          'Resampled_sliced_speed':self.resample(np.concatenate((self.signals_object.first_noise_set, self.signals_object.second_noise_set, self.signals_object.third_noise_set)), factor=self.milisecond_period, kind='linear').squeeze(),
                                                          'Resampled_sliced_visstim':self.resample(np.concatenate((self.signals_object.first_noise_set, self.signals_object.second_noise_set, self.signals_object.third_noise_set)), factor=self.milisecond_period, kind='linear').squeeze()
                                                          
                                                          
                                                          }
                
                
                self.full_data['visstim_info']['Movie2']={'Indexes':'',
                                                          'Binary_Maytrix':''
                                                            }
                
            elif self.signals_object.vis_stim_protocol =='AllenB':    
                
                self.full_data['visstim_info']['Static_Gratings']={'Indexes':'',
                                                          'Binary_Maytrix':'',
                                                          'Ref_matrix':''
                                                            }
                self.full_data['visstim_info']['Natural_Images']={'Indexes':'',
                                                          'Binary_Maytrix':'',
                                                          'Ref_matrix':''
                                                            }

    def create_stim_table(self):
        self.stimulus_table={}
        self.stimulus_table['drifting_gratings']=self.drifting_grating_stim_table()

    def drifting_grating_stim_table(self):
        if (not self.nondatabase) and self.signals_object.vis_stim_protocol:
    
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
            
            return sorted_df.reset_index(drop=True)
            
#%% plotting
    def some_ploting(self):
        dataset_name=list(self.caiman_results.keys())[0]
        fig, ax=plt.subplots(3)
    
        ax[0].plot(self.resampled_stim_matrix)
        ax[0].margins(x=0)
        ax[1].imshow(self.separated_planes_combined_gratings_binarized_mcmc[dataset_name], aspect='auto')
        ax[2].imshow(self.separated_planes_combined_gratings_binarized_dfdt[dataset_name], aspect='auto')

    def plot_full_signals_and_raster(self):
        
        pass


    def do_some_plotting(self, cell, trace_type, plane, matlabcell=None):
        
        # plt.close('all')
        
        # plane='Plane1'
        # cell=4
        # trace_type='dfdt_smoothed'
        # trace_type='denoised'
        # trace_type='mcmc_smoothed'
        self.preframes=16
        self.stim=33
        self.postframes=16
        
        # self.preframes=25
        # self.stim=50
        # self.postframes=25
        # ,matlabcell=4 21 75# inhibited by stimulus
        #cell 3 13
        # matlabcell=45# bimodal
        # matlabcell=47 49# very sharp dip at begining of stim


        if matlabcell:
            cell=np.argwhere(self.full_data['imaging_data'][plane]['CellIds']==matlabcell)[0][0]
        matlabcell=self.full_data['imaging_data'][plane]['CellIds'][cell]
            
        if self.pyr_int_ids_and_indexes:
            pyr=np.argwhere(self.pyr_int_ids_and_indexes['pyr'][1]).flatten()
            inter=np.argwhere(self.pyr_int_ids_and_indexes['int'][1]).flatten()
            if  cell in    pyr:
                celltype='Pyramidal Cell'
            elif  cell in    inter:
                celltype='Interneuron'
                
        celltype='Interneuron'

        print(plane+'\nMatlab cell: '+str( matlabcell)+'\nPython cell :'+str(cell)+'\n' + celltype)
        
        #  self.plot_blank_sweeps(cell,trace_type,plane)
        #  self.plot_directions(cell,trace_type,plane)
        # self.plot_orientation(cell,trace_type,plane)
        C_mat=self.full_data['imaging_data'][plane]['Traces'][trace_type]
        

        pixel_per_bar = 4
        dpi = 100
        
               
        # fig,ax = plt.subplots(1,  figsize=(16,9), dpi=dpi)
        # ax.imshow( C_mat, cmap='binary', aspect='auto',
        #     interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        
        # ax.set_xlabel('Time(s)')
        # fig.supylabel('Cell Number')
        # fig.suptitle('Raster dfdt')
        # plt.show()
 
        return cell


    def plot_blank_sweeps(self, cell, trace_type, plane):
       
        C_mat=self.full_data['imaging_data'][plane]['Traces'][trace_type]
        

        
        cell_trace=C_mat[cell,:]
        fig,ax=plt.subplots(4)
        ax[0].plot(cell_trace)
   
        test=np.vstack(
            [cell_trace[self.stimulus_table['drifting_gratings'][self.stimulus_table['drifting_gratings'].blank_sweep==1].start.values.astype( 'uint16' )[sweep]-self.preframes:
                        self.stimulus_table['drifting_gratings'][self.stimulus_table['drifting_gratings'].blank_sweep==1].end.values.astype( 'uint16' )[sweep]+self.postframes]
             for sweep in range(len(self.stimulus_table['drifting_gratings'][self.stimulus_table['drifting_gratings'].blank_sweep==1].start.values.astype( 'uint16' )))])
        
        
        meantraces=np.mean(test, axis=0)
        
        for i, row in self.stimulus_table['drifting_gratings'][self.stimulus_table['drifting_gratings'].blank_sweep==1].iterrows():
            ax[1].plot(cell_trace[int(row.start)-self.preframes:int(row.end)+self.postframes], 'k', alpha=0.1)
            
        ax[1].axvspan(0,self.preframes, facecolor='g', alpha=0.3)
        ax[1].axvspan(self.preframes,self.preframes+self.stim, facecolor='b', alpha=0.3)
        ax[1].plot(meantraces,'k')
        ax[2].imshow(test,cmap='binary', aspect='auto')
        ax[3].plot(meantraces,'k')
            

            
    def plot_directions(self, cell, trace_type, plane):
        
        for ori in np.linspace(0,360-45,8):
            fig,ax=plt.subplots(4)

         
            C_mat=self.full_data['imaging_data'][plane]['Traces'][trace_type]
            cell_trace=C_mat[cell,:]
            ax[0].plot(cell_trace)
     
    
            trial=np.vstack([cell_trace[int(row.start-self.preframes):int(row.end+self.postframes)]
             if row.end-row.start+self.preframes+self.postframes==self.preframes+self.stim+self.postframes
             else  cell_trace[int(row.start-self.preframes):int(row.end+self.postframes+1)] for i, row in self.stimulus_table['drifting_gratings'][self.stimulus_table['drifting_gratings'].orientation==ori].iterrows() ])

            meantraces=np.mean(trial, axis=0)
                
            for i, row in self.stimulus_table['drifting_gratings'][(self.stimulus_table['drifting_gratings'].orientation==ori)&(self.stimulus_table['drifting_gratings'].blank_sweep==0)].iterrows():
                ax[1].plot(cell_trace[int(row.start)-self.preframes:int(row.end)+self.postframes], 'k', alpha=0.1)
            ax[1].axvspan(0,self.preframes, facecolor='g', alpha=0.3)
            ax[1].axvspan(self.preframes,self.preframes+self.stim, facecolor='b', alpha=0.3)

            ax[1].plot(meantraces,'k')
            ax[2].imshow(trial,cmap='binary', aspect='auto')
            ax[3].plot(meantraces,'k')

            plt.show()
            
    def plot_orientation(self, cell, trace_type, plane):
        
        pyr=np.argwhere(self.pyr_int_ids_and_indexes['pyr'][1]).flatten()
        inter=np.argwhere(self.pyr_int_ids_and_indexes['int'][1]).flatten()
        if  cell in    pyr:
            celltype='Pyramidal Cell'
        elif  cell in    inter:
            celltype='Interneuron'
        
        mean_evoked_2s=[]
        mean_evoked_1s=[]

        
        fig = plt.figure(constrained_layout=True)
        subfigs = fig.subfigures(2, 3, wspace=0.07, width_ratios=[1, 1,1])
        fig.suptitle('Cell: '+str(cell)+' '+celltype )

        axs=['','','','']
        for n, ori in enumerate(np.linspace(0,180-(180/4),4)):
            axs[n]=subfigs[int(np.floor(n/2)),int(n%2)].subplots(3)
            # trace_type='denoised'
            C_mat=self.full_data['imaging_data'][plane]['Traces'][trace_type]
            cell_trace=C_mat[cell,:]
            
          
    
            trial=np.vstack([cell_trace[int(row.start-self.preframes):int(row.end+self.postframes)]
             if row.end-row.start+self.preframes+self.postframes==self.preframes+self.stim+self.postframes
             else  cell_trace[int(row.start-self.preframes):int(row.end+self.postframes+1)] 
             for i, row in self.stimulus_table['drifting_gratings'][self.stimulus_table['drifting_gratings'].orientation.isin([ori, ori+180])].iterrows() ])

            meantraces=np.mean(trial, axis=0)
                
            for i, row in self.stimulus_table['drifting_gratings'][(self.stimulus_table['drifting_gratings'].orientation==ori)&(self.stimulus_table['drifting_gratings'].blank_sweep==0)].iterrows():
                axs[n][0].plot(cell_trace[int(row.start)-self.preframes:int(row.end)+self.postframes], 'k', alpha=0.1)
            axs[n][0].set_title(ori)

                
                
            axs[n][0].axvspan(0,self.preframes, facecolor='g', alpha=0.3)
            axs[n][0].axvspan(self.preframes,self.preframes+self.stim, facecolor='b', alpha=0.3)
            mean_evoked_2s.append(np.mean(meantraces[self.preframes:self.preframes+self.stim]))
            mean_evoked_1s.append(np.mean(meantraces[self.preframes:int(np.floor((self.preframes+self.stim)/2))]))


            axs[n][0].plot(meantraces,'k')
            axs[n][1].imshow(trial,cmap='binary', aspect='auto')
            axs[n][2].plot(meantraces,'k')


                
                
        # ax1=subfigs[0,2].subplots(1)    
        # ax1.plot(cell_trace)
        # ax2=subfigs[1,2].subplots(1)    
        # ax2.plot([0,45,90,135],mean_evoked_2s, 'r', label='Full Stim')
        # ax2.plot([0,45,90,135],mean_evoked_1s, 'b', label='First Half Stim')
        # ax2.legend()

        # ax2.set_title('Mean Evoked Activity')

        # ax2.set_ylim(0,2*np.max(mean_evoked_2s))
        # # ax2.set_ylim(0,0.2)

        # plt.show()
        
        return mean_evoked_2s, mean_evoked_1s


#%% int pyr ident            
   
    def check_pyr_int_identif_files(self) :
        self.pyr_int_identif_list=glob.glob(self.data_analysis_path+'\\**pyr_int_identif**', recursive=False)

    def load_pyr_int_identif(self):
        if self.pyr_int_identif_list:
            with open( self.pyr_int_identif_list[0], 'rb') as file:
               self.pyr_int_identification=  pickle.load(file)
        else:
            plane='Plane1'
            self.pyr_int_identification={}
            self.pyr_int_identification[plane]={'interneuron':{'matlab':'',
                                                          'python':'',
                                                          },
                                           'pyramidals':{'matlab':'',
                                                         'python':'',
                                                         }
                                           }
            if not  self.pyr_int_identification[plane]['interneuron']['python']:
                self.pyr_int_identification[plane]['interneuron']['python']=self.full_data['imaging_data']['Plane1']['CellIds']
            print('no previous pyr ident')
            
    def save_pyr_int_identif(self):
        datapath=os.path.join(self.data_analysis_path, self.pyr_int_identif_path_name)
        if not os.path.isfile(datapath):
            with open(datapath, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self.pyr_int_identification, f, pickle.HIGHEST_PROTOCOL)


    def get_pyr_int_indexing_dict(self):
        if not self.nondatabase and   self.pyr_int_identification:
            self.pyr_int_identification['Plane1']['pyramidals']['python']=np.setxor1d(self.pyr_int_identification['Plane1']['interneuron']['python'],self.full_data['imaging_data']['Plane1']['CellIds'])
            
            self.pyr_int_ids_and_indexes={'pyr':(self.pyr_int_identification['Plane1']['pyramidals']['python'], 
                                    np.in1d(self.full_data['imaging_data']['Plane1']['CellIds'], self.pyr_int_identification['Plane1']['pyramidals']['python'])),
                             'int':(np.array(self.pyr_int_identification['Plane1']['interneuron']['python']),
                                    np.in1d(self.full_data['imaging_data']['Plane1']['CellIds'], self.pyr_int_identification['Plane1']['interneuron']['python']))}

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
        
#%% activity slicing
    def slice_matrix_by_paradigm_indexes(self, matrix, indexes):
        
      
        sliced_matrix=np.empty((matrix.shape[0],0))
        for segment in indexes:
    
            sliced_matrix=np.concatenate((sliced_matrix ,matrix[:,segment[0]:segment[1]]), axis=1)

        return sliced_matrix
    
    def separate_cell_types_rasters_by_plane(self):
        
        pass
    
    def separate_paradigm_rasters(self):
        pass
  
    def slice_activity_matrix(self):
        pass
       
          

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

        
#%% loading and saving subanalysis objects
#%% umap tsne

    # def umap_test(self):
    #     reducer = umap.UMAP()
    #     embedding = reducer.fit_transform(self.full_data['imaging_data']['Plane1']['Traces']['dfdt_binary'])
    #     embedding.shape
        # plt.scatter(
        # embedding[:, 0],
        # embedding[:, 1],
        # color=[sns.color_palette()[x] for x in penguins.species_short.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})])
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.title('UMAP projection of the Penguin dataset', fontsize=24)
       
#%% PCA
    def do_PCA(self, mean_sweep_response, sweep_response, driftgrattable):
        
        trial_mean_peak=mean_sweep_response
        trial_traces=sweep_response
        filename=r'C:/Users/sp3660/Desktop/PCAanim.mp4'
        
        start_stim=0
        end_stim=2
        frames_pre_stim = 30
        frames_post_stim = 30
        
        full_trials=np.zeros((trial_traces.shape[0],trial_traces.shape[1],trial_traces.iloc[0,0].shape[0]))
        
        for i in range(trial_traces.shape[0]): #iterate over rows
            for j in range(trial_traces.shape[1]):
                full_trials[i,j,:]=trial_traces.iloc[i,j]
                
       
        trials=[]
        for i in range(full_trials.shape[0]):
            trials.append(full_trials[i,:,:])
       
        trial_type=driftgrattable['orientation'].values.tolist()
        trial_types=np.arange(0,360,int(360/8))
        time=np.linspace(-1,3,120)
        trial_size   = trials[0].shape[1]
        Nneurons     = trials[0].shape[0]
        t_type_ind = [np.argwhere(np.array(trial_type) == t_type)[:, 0] for t_type in trial_types]
        print('Number of trials: {}'.format(len(trials)))
        print('Types of trials (orientations): {}'.format(trial_types)) 
        print('Dimensions of single trial array (# neurons by # time points): {}'.format(trials[0].shape))
        print('Trial types (orientations): {}'.format(trial_types))
        print('Trial type of the first 3 trials: {}'.format(trial_type[0:3]))
       
       
        shade_alpha  = 0.2
        lines_alpha   = 0.8
        pal= sns.color_palette('husl', 8)
        # config InlineBackend.figure_format = 'svg'
        
        def add_stim_to_plot(ax, shade_alpha, lines_alpha):
            ax.axvspan(start_stim, end_stim, alpha=shade_alpha,
                       color='gray')
            ax.axvline(start_stim, alpha=lines_alpha, color='gray', ls='--')
            ax.axvline(end_stim, alpha=lines_alpha, color='gray', ls='--')
            
        def add_orientation_legend(ax):
            custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for
                            k in range(len(trial_types))]
            labels = ['{}$^\circ$'.format(t) for t in trial_types]
            ax.legend(custom_lines, labels,
                      frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout(rect=[0,0,0.9,1])
            
        def z_score(X):
           # X: ndarray, shape (n_features, n_samples)
           ss = StandardScaler(with_mean=True, with_std=True)
           Xz = ss.fit_transform(X.T).T
           return Xz
       
        def get_concatentaed_trial_PCA(pal, trials, trial_types, t_type_ind, frames_pre_stim, frames_post_stim):
            Xr = np.vstack([t[:, frames_pre_stim:-frames_post_stim].mean(axis=1) for t in trials]).T
            # or take the max
            # Xr = np.vstack([t[:, frames_pre_stim:-frames_post_stim].max(axis=1) for t in trials]).T
            # or the baseline-corrected mean
            # Xr = np.vstack([t[:, frames_pre_stim:-frames_post_stim].mean(axis=1) - t[:, 0:frames_pre_stim].mean(axis=1) for t in trials]).T
            Xr_sc = z_score(Xr)
            
            pca = PCA(n_components=15)
            Xp = pca.fit_transform(Xr_sc.T).T
            
            projections = [(0, 1), (1, 2), (0, 2), (2,3)]
            fig, axes = plt.subplots(1, len(projections), figsize=[9, 3], sharey='row', sharex='row')
            for ax, proj in zip(axes, projections):
                for t, t_type in enumerate(trial_types):
                    x = Xp[proj[0], t_type_ind[t]]
                    y = Xp[proj[1], t_type_ind[t]]
                    ax.scatter(x, y, c=pal[t], s=25, alpha=0.8)
                    ax.set_xlabel('PC {}'.format(proj[0]+1))
                    ax.set_ylabel('PC {}'.format(proj[1]+1))
            sns.despine(fig=fig, top=True, right=True)
            add_orientation_legend(axes[2])
            
            return Xr, Xr_sc, pca, Xp
            
                                           
        def get_concatentaed_trial_averaged_PCA(pal, trial_traces,trial_types,  time):
            
            trial_averages = []
            for ind in t_type_ind:
                trial_averages.append(np.array(trials)[ind].mean(axis=0))
            Xa = np.hstack(trial_averages)
            
            n_components = 15
            Xa = z_score(Xa) #Xav_sc = center(Xav)
            pca = PCA(n_components=n_components)
            Xa_p = pca.fit_transform(Xa.T).T
            plt.plot(pca.explained_variance_ratio_)

            
            
            comp_to_plot=3
            fig, axes = plt.subplots(1, comp_to_plot, figsize=[20, 2.8], sharey='row')
            for comp in range(comp_to_plot):
                ax = axes[comp]
                for kk, type in enumerate(trial_types):
                    x = Xa_p[comp, kk * trial_size :(kk+1) * trial_size]
                    x = gaussian_filter1d(x, sigma=3)
                    ax.plot(time, x, c=pal[kk])
                add_stim_to_plot(ax, shade_alpha, lines_alpha)
                ax.set_ylabel('PC {}'.format(comp+1))
            add_orientation_legend(axes[2])
            axes[1].set_xlabel('Time (s)')
            sns.despine(fig=fig, right=True, top=True)
            plt.tight_layout(rect=[0, 0, 0.9, 1])
                                            
            # find the indices of the three largest elements of the second eigenvector
            units = np.abs(pca.components_[1, :].argsort())[::-1][0:3]
            f, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey=False, sharex=True)
            for ax, unit in zip(axes, units):
                ax.set_title('Neuron {}'.format(unit))
                alldfs=[]
                for t, ind in enumerate(t_type_ind):
                    x = np.array(trials)[ind][:, unit, :]
                    df=pd.DataFrame(x.T)
                    df.reset_index(inplace=True)
                    df['index']=time
                    df2 = pd.melt(df, id_vars='index', value_vars=np.arange(0,73))
                    df2['Orientation'] =trial_types[t]
                    alldfs.append(df2)
                finaldf = pd.concat(alldfs)
                finaldf.rename(columns = {'index':'time'}, inplace = True)
                finaldf.reset_index(drop=True, inplace=True)
                sns.lineplot(x="time", y="value",hue='Orientation', data=finaldf, ax=ax, legend=False, palette=pal)
                    
            for ax in axes:
                add_stim_to_plot(ax, shade_alpha, lines_alpha)
                
            axes[1].set_xlabel('Time (s)')
            sns.despine(fig=f, right=True, top=True)
            add_orientation_legend(axes[2])
            
            return Xa, pca, Xa_p
       
            
        def get_trial_concatenated_PCA(pal, trial_traces, filename, trial_types, trial_type, time): 
            n_components = 15
       
            Xl = np.hstack(trials)
            Xl = z_score(Xl)
            pca = PCA(n_components=15)
            Xl_p = pca.fit_transform(Xl.T).T
            gt = {comp : {t_type : [] for t_type in trial_types} for comp in range(n_components)}
       
            for comp in range(n_components):
                for i, t_type in enumerate(trial_type):
                    if not np.isnan(t_type):
                        t = Xl_p[comp, trial_size * i: trial_size * (i + 1)]
                        gt[comp][t_type].append(t)
            
            for comp in range(n_components):
                for t_type in trial_types:
                    if not np.isnan(t_type):
                        gt[comp][t_type] = np.vstack(gt[comp][t_type])
                                         
            f, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey=False, sharex=True)
            for comp in range(3):
                ax = axes[comp]
                alldfs=[]
                for t, t_type in enumerate(trial_types):
                    x=gt[comp][t_type]
                    df=pd.DataFrame(x.T)
                    df.reset_index(inplace=True)
                    df['index']=time
                    df2 = pd.melt(df, id_vars='index', value_vars=np.arange(0,73))
                    df2['Orientation'] =t_type
                    alldfs.append(df2)
                finaldf = pd.concat(alldfs)
                finaldf.rename(columns = {'index':'time'}, inplace = True)
                finaldf.reset_index(drop=True, inplace=True)
                sns.lineplot(x="time", y="value",hue='Orientation', data=finaldf, ax=ax, legend=False, palette=pal)
                add_stim_to_plot(ax, shade_alpha, lines_alpha)
                ax.set_ylabel('PC {}'.format(comp+1))
            axes[1].set_xlabel('Time (s)')
            sns.despine(right=True, top=True)
            add_orientation_legend(axes[2])

        def get_hybrid_PCA(pal, trial_traces, filename, trial_types, trial_type):     
            # fit PCA on trial averages
            trial_averages = []
            for ind in t_type_ind:
                trial_averages.append(np.array(trials)[ind].mean(axis=0))
            Xav = np.hstack(trial_averages)
            
            ss = StandardScaler(with_mean=True, with_std=True)
            Xav_sc = ss.fit_transform(Xav.T).T
            pca = PCA(n_components=15) 
            pca.fit(Xav_sc.T) # only call the fit method
            
            projected_trials = []
            for trial in trials:
                # scale every trial using the same scaling applied to the averages 
                trial = ss.transform(trial.T).T
                # project every trial using the pca fit on averages
                proj_trial = pca.transform(trial.T).T
                projected_trials.append(proj_trial)
       
            gt = {comp: {t_type: [] for t_type in trial_types}
                  for comp in range(n_components)}
            
            for comp in range(n_components):
                for i, t_type in enumerate(trial_type   ):
                    if not np.isnan(t_type):
                        t = projected_trials[i][comp, :]
                        gt[comp][t_type].append(t)
                            
            f, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey=True, sharex=True)
            for comp in range(3):
                ax = axes[comp]
                for t, t_type in enumerate(trial_types):
                    x=np.vstack(gt[comp][t_type])
                    df=pd.DataFrame(x.T)
                    df.reset_index(inplace=True)
                    df['index']=time
                    df2 = pd.melt(df, id_vars='index', value_vars=np.arange(0,73))
                    df2['Orientation'] =t_type
                    alldfs.append(df2)
                finaldf = pd.concat(alldfs)
                finaldf.rename(columns = {'index':'time'}, inplace = True)
                finaldf.reset_index(drop=True, inplace=True)
                sns.lineplot(x="time", y="value",hue='Orientation', data=finaldf, ax=ax, legend=False, palette=pal)
                add_stim_to_plot(ax, shade_alpha, lines_alpha)
                ax.set_ylabel('PC {}'.format(comp+1))
            axes[1].set_xlabel('Time (s)')
            sns.despine(right=True, top=True)
            add_orientation_legend(axes[2])
            
        def get_3d_PCA(frames_pre_stim):
            # prepare trial averages
            trial_averages = []
            for ind in t_type_ind:
                trial_averages.append(np.array(trials)[ind].mean(axis=0))
            Xa = np.hstack(trial_averages)
            
            # standardize and apply PCA
            Xa = z_score(Xa) 
            pca = PCA(n_components=15)
            Xa_p = pca.fit_transform(Xa.T).T
            
            # pick the components corresponding to the x, y, and z axes
            component_x = 0
            component_y = 1
            component_z = 2
            component_x = 3

            
            # create a boolean mask so we can plot activity during stimulus as 
            # solid line, and pre and post stimulus as a dashed line
            stim_mask = ~np.logical_and(np.arange(trial_size) >= frames_pre_stim,
                           np.arange(trial_size) < (trial_size-frames_post_stim))
            
            # utility function to clean up and label the axes
            def style_3d_ax(ax):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('w')
                ax.yaxis.pane.set_edgecolor('w')
                ax.zaxis.pane.set_edgecolor('w')
                ax.set_xlabel('PC 1')
                ax.set_ylabel('PC 2')
                ax.set_zlabel('PC 3')
            
            sigma = 3 # smoothing amount
            
            # set up a figure with two 3d subplots, so we can have two different views
            fig = plt.figure(figsize=[9, 4])
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            axs = [ax1, ax2]
            
            for ax in axs:
                for t, t_type in enumerate(trial_types):
            
                    # for every trial type, select the part of the component
                    # which corresponds to that trial type:
                    x = Xa_p[component_x, t * trial_size :(t+1) * trial_size]
                    y = Xa_p[component_y, t * trial_size :(t+1) * trial_size]
                    z = Xa_p[component_z, t * trial_size :(t+1) * trial_size]
                    
                    # apply some smoothing to the trajectories
                    x = gaussian_filter1d(x, sigma=sigma)
                    y = gaussian_filter1d(y, sigma=sigma)
                    z = gaussian_filter1d(z, sigma=sigma)
            
                    # use the mask to plot stimulus and pre/post stimulus separately
                    z_stim = z.copy()
                    z_stim[stim_mask] = np.nan
                    z_prepost = z.copy()
                    z_prepost[~stim_mask] = np.nan
            
                    ax.plot(x, y, z_stim, c = pal[t])
                    ax.plot(x, y, z_prepost, c=pal[t], ls=':')
                    
                    # plot dots at initial point
                    ax.scatter(x[0], y[0], z[0], c=pal[t], s=14)
                    
                    # make the axes a bit cleaner
                    style_3d_ax(ax)
                    
            # specify the orientation of the 3d plot        
            ax1.view_init(elev=22, azim=30)
            ax2.view_init(elev=22, azim=110)
            plt.tight_layout()
     
        def animation_2d_scatter(projected_trials):    
            # smooth the single projected trials 
            for i in range(len(projected_trials)):
                for c in range(projected_trials[0].shape[0]):
                    projected_trials[i][c, :] = gaussian_filter1d(projected_trials[i][c, :], sigma=3)
            
            # for every time point (imaging frame) get the position in PCA space of every trial
            pca_frame = []
            for t in range(trial_size):
                # projected data for all trials at time t 
                Xp = np.hstack([tr[:, None, t] for tr in projected_trials]).T
                pca_frame.append(Xp)
                
            subspace = (1, 2) # pick the subspace given by the second and third components
                
            # set up the figure
            fig, ax = plt.subplots(1, 1, figsize=[6, 6]); plt.close()
            ax.set_xlim(( -16, 16))
            ax.set_ylim((-16, 16))
            ax.set_xlabel('PC 2')
            ax.set_xticks([-10, 0, 10])
            ax.set_yticks([-10, 0, 10])
            ax.set_ylabel('PC 3')
            sns.despine(fig=fig, top=True, right=True)
            
            # generate empty scatter plot to be filled by data at every time point
            scatters = []
            for t, t_type in enumerate(trial_types):
                scatter, = ax.plot([], [], 'o', lw=2, color=pal[t]);
                scatters.append(scatter)
            
            # red dot to indicate when stimulus is being presented
            stimdot, = ax.plot([], [], 'o', c='r', markersize=35, alpha=0.5)
            
            # annotate with stimulus and time information
            text     = ax.text(6.3, 9, 'Stimulus OFF \nt = {:.2f}'.format(time[0]), fontdict={'fontsize':14})
            
            # this is the function to be called at every animation frame
            def animate(i):
                for t, t_type in enumerate(trial_types):
                    # find the x and y position of all trials of a given type
                    x = pca_frame[i][t_type_ind[t], subspace[0]]
                    y = pca_frame[i][t_type_ind[t], subspace[1]]
                    # update the scatter
                    scatters[t].set_data(x, y)
                    
                # update stimulus and time annotation
                if (i > frames_pre_stim) and (i < (trial_size-frames_post_stim)):
                    stimdot.set_data(10, 14)
                    text.set_text('Stimulus ON \nt = {:.2f}'.format(time[i]))
                else:
                    stimdot.set_data([], [])
                    text.set_text('Stimulus OFF \nt = {:.2f}'.format(time[i]))
                return (scatter,)
            
            # generate the animation
            anim = animation.FuncAnimation(fig, animate, 
                                           frames=len(pca_frame), interval=30, 
                                           blit=False)
            
            anim.save(filename, writer = 'ffmpeg', fps = 10)
            
            
       
        def animate_3d_trial_averaged():
            
            pca_frame = []
            for t in range(trial_size):
                # projected data for all trials at time t 
                Xp = np.hstack([tr[:, None, t] for tr in projected_trials]).T
                pca_frame.append(Xp)
                
            subspace = (1, 2) # pick 

            # apply some smoothing to the trajectories
            for c in range(Xa_p.shape[0]):
                Xa_p[c, :] =  gaussian_filter1d(Xa_p[c, :], sigma=sigma)
            
            # create the figure
            fig = plt.figure(figsize=[9, 9]); plt.close()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            
            def animate(i):
                
                ax.clear() # clear up trajectories from previous iteration
                style_3d_ax(ax)
                ax.view_init(elev=22, azim=30)
            
                for t, t_type in enumerate(trial_types):
                
                    x = Xa_p[component_x, t * trial_size :(t+1) * trial_size][0:i]
                    y = Xa_p[component_y, t * trial_size :(t+1) * trial_size][0:i]
                    z = Xa_p[component_z, t * trial_size :(t+1) * trial_size][0:i]
                            
                    stim_mask = ~np.logical_and(np.arange(z.shape[0]) >= frames_pre_stim,
                                 np.arange(z.shape[0]) < (trial_size-frames_pre_stim))
            
                    z_stim = z.copy()
                    z_stim[stim_mask] = np.nan
                    z_prepost = z.copy()
                    z_prepost[~stim_mask] = np.nan
                    
                    ax.plot(x, y, z_stim, c = pal[t])
                    ax.plot(x, y, z_prepost, c=pal[t], ls=':')
            
                ax.set_xlim(( -8, 8))
                ax.set_ylim((-8, 8))
                ax.set_zlim((-6, 6))
            
                return []
            
            
            anim = animation.FuncAnimation(fig, animate,
                                           frames=len(pca_frame), interval=30
                                           )
            
            anim.save(filename, writer = 'ffmpeg', fps = 10)
            
        def animate_3d_single_trial():
   
            # set up a dictionary to color each line
            col = {trial_types[i] : pal[i] for i in range(len(trial_types))}
            
            
            fig = plt.figure(figsize=[9, 9]); plt.close()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            
            def animate(i):
                
                ax.clear()
                style_3d_ax(ax)
                ax.view_init(elev=22, azim=30)
                for t, (trial, t_type) in enumerate(zip(projected_trials[0:40], trial_type[0:40])):
                    
                    x = trial[component_x, :][0:i]
                    y = trial[component_y, :][0:i]
                    z = trial[component_z, :][0:i]
                    
                    stim_mask = ~np.logical_and(np.arange(z.shape[0]) >= frames_pre_stim,
                                 np.arange(z.shape[0]) < (trial_size-frames_pre_stim))
            
                    z_stim = z.copy()
                    z_stim[stim_mask] = np.nan
                    z_prepost = z.copy()
                    z_prepost[~stim_mask] = np.nan
                    
                    ax.plot(x, y, z_stim, c = col[t_type])
                    ax.plot(x, y, z_prepost, c=col[t_type], ls=':')
            
                ax.set_xlim(( -12, 12))
                ax.set_ylim((-12, 12))
                ax.set_zlim((-13, 13))
                ax.view_init(elev=22, azim=30)
            
                return []
            
            anim = animation.FuncAnimation(fig, animate, frames=len(pca_frame), interval=30,blit=True)
            
            anim.save(filename, writer = 'ffmpeg', fps = 10)
                        
            
        get_concatentaed_trial_PCA(pal, trials, trial_types, t_type_ind, frames_pre_stim, frames_post_stim)
        get_concatentaed_trial_averaged_PCA(pal, trial_traces, time)
        get_trial_concatenated_PCA(pal, trial_traces, filename, trial_types, trial_type)
        get_hybrid_PCA(pal, trial_traces, filename, trial_types, trial_type)
        animation_2d_scatter(projected_trials)
        get_3d_PCA(frames_pre_stim)
        animate_3d_trial_averaged()
        animate_3d_single_trial()

            
            
#%% allen
    def load_allen_analysis(self):
        self.allen_analysis=AllenAnalysis(self)
#%% CRFs
    def load_CRFs_analysis(self):
        self.crf_analysis=CRFsResults(self)
        pass
#%% yuriy
    def load_yuriy_analysis(self):
        self.yuriy_analysis=EnsemblesYuriy(self)
    
#%% jesus    
    def load_jesus_analysis(self, binary_raster_to_proces, plane, segment, cell_type):   
        # self.load_jesus_results()

        # get raster
        if binary_raster_to_proces=='MCMC':
            if plane=='All':
                selected_full_raster= self.full_data['imaging_data']['All_planes_rough']['Traces']['mcmc_binary']
            else:
                selected_full_raster=self.full_data['imaging_data'][plane]['Traces']['mcmc_binary']
        elif binary_raster_to_proces=='dfdt':
            if plane=='All':
                selected_full_raster= self.full_data['imaging_data']['All_planes_rough']['Traces']['dfdt_binary']
            else:
                selected_full_raster=self.full_data['imaging_data'][plane]['Traces']['dfdt_binary']   



        #slice by paradigm
        if segment=='Full':
            indexes=((0,selected_full_raster.shape[1]+1),)
        elif segment=='DriftingGratings':
            indexes=((self.full_data['visstim_info']['Paradigm_Indexes']['first_drifting_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['first_drifting_set_last']),
            (self.full_data['visstim_info']['Paradigm_Indexes']['second_drifting_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['second_drifting_set_last']),
            (self.full_data['visstim_info']['Paradigm_Indexes']['third_drifting_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['third_drifting_set_last'])) 
        elif segment=='Spontaneous':
            indexes_alt= ((self.full_data['visstim_info']['Spontaneous']['stimulus_table']['start'].values.tolist()[0], self.full_data['visstim_info']['Spontaneous']['stimulus_table']['end'].values.tolist()[0]))
            indexes=([self.full_data['visstim_info']['Paradigm_Indexes']['spont_first'],self.full_data['visstim_info']['Paradigm_Indexes']['spont_last']],)
            

        paradigm_sliced_raster= self.slice_matrix_by_paradigm_indexes(selected_full_raster, indexes)
        
        #slice bycells
        if self.pyr_int_ids_and_indexes:
            if cell_type=='Pyr':
                cell_indexes= self.pyr_int_ids_and_indexes['pyr'][1]
            elif cell_type=='Int':
                cell_indexes= self.pyr_int_ids_and_indexes['int'][1]
            elif cell_type=='All':
                cell_indexes= np.full(self.pyr_int_ids_and_indexes['int'][1].shape, True)
                cell_indexes=np.arange(paradigm_sliced_raster.shape[0])

        else:
            cell_indexes=np.arange(paradigm_sliced_raster.shape[0])

            


        final_raster=paradigm_sliced_raster[cell_indexes,:]
        plt.imshow(final_raster, aspect='auto')
        plt.title('_'.join([binary_raster_to_proces, plane, segment, cell_type]))
        plt.show()
        self.jesus_binary_spikes=final_raster
        self.jesus_analysis=JesusEnsemblesResults(self)
        # self.jesus_runs[ self.run_number+'_'+self.acquisition_object.aquisition_name+'_'+binary_raster_to_proces+'_'+plane+'_'+segment]=[binary_raster_to_proces, plane, segment, self.jesus_binary_spikes, self.jesus_analysis.analysis]
        self.jesus_run=[binary_raster_to_proces, plane, segment,cell_type, self.jesus_analysis.analysis]
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.jesus_runs_path_name='_'.join([self.acquisition_object.aquisition_name, timestr,binary_raster_to_proces, plane, segment,cell_type,'jesus_results.pkl'])  


        self.save_jesus_runs()
        
    def save_jesus_runs(self):
        print('Saving jesus run')
        datapath=os.path.join(self.jesus_runs_path, self.jesus_runs_path_name)
        
        if not os.path.isdir(self.jesus_runs_path):
            os.mkdir(self.jesus_runs_path)
            
    
        with open(datapath, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.jesus_run, f, pickle.HIGHEST_PROTOCOL)
        
    def check_all_jesus_results(self) :
        self.jesus_results_list=glob.glob(self.jesus_runs_path+'\\**', recursive=False)
            
        # pyr_results=[i for i in  analysis.jesus_results_list if 'Pyr' in i]
        # int_results=[i for i in  analysis.jesus_results_list if '_Int_' in i]
        # full_results=[i for i in  analysis.jesus_results_list if '_Full' in i]
        # drift_grat_results=[i for i in  analysis.jesus_results_list if 'DriftingGratings' in i]
        # mcmc_results=[i for i in  analysis.jesus_results_list if 'mcmc' in i]
        # dfdtresults=[i for i in  analysis.jesus_results_list if 'dfdt' in i]
        # all_results=[i for i in  analysis.jesus_results_list if '_All' in i]
        
        # deconv='MCMC'
        # deconv='dfdt'
        # all_divisions=0
        # index=0
        # cell_type='Pyr'
        # cell_type='Int'
        
      
        
        # def intersection(lst1, lst2):
        #     return list(set(lst1) & set(lst2))
        
        # pyr_grat=intersection(pyr_results, drift_grat_results)
        # int_grat=intersection(int_results, drift_grat_results)
        # all_grat=intersection(all_results, drift_grat_results)

        
        
    def load_jesus_results(self, path):
        if path:
            with open( path, 'rb') as file:
                self.jesus_runs[os.path.split(path)[1]]= JesusEnsemblesResults(self, path)
        else:
            self.jesus_runs={}
            print('no previous jesus runs')
        
#%% classical svd
    def load_classicalsvd_analysis(self):
        self.svd_analysis=SVDEnsemblesResults(self)
        pass

      
#%% if main
if __name__ == "__main__":
    
    
    analysis=ResultsAnalysis(  
                 acquisition_voltage_signals_object=voltagesignals,
                 metadata_object=meta,
                 nondatabase=True)
    
    analysis.signals_object.process_all_signals(vis_stim_protocol='AllenC')
    analysis.metadata_object
    

    plt.close('all')
    # sorter results

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

    # meta =Metadata(acquisition_directory_raw=temporary_path1)


# voltage signals

    # temporary_path1=linux_temp +os.sep+'210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'

    # voltagesignals=VoltageSignalsExtractions(temporary_path1)

#% RUN ANALYSIS CLASS

    # analysis=ResultsAnalysis(SPJA_0702_allen_plane1, SPJA_0702_allen_plane2, SPJA_0702_allen_plane3, SPJA_0702_allen_CRFS, voltagesignals, meta)
    # analysis=ResultsAnalysis(SPKG_1015_allen_plane1, SPKG_1015_allen_plane2, SPKG_1015_allen_plane3, acquisition_voltage_signals_object=voltagesignals, metadata_object=meta )
# 
    #% plotting
 
    
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

