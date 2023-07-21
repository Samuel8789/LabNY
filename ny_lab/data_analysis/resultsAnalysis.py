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
import copy
import glob
import gc
import sys
from matplotlib.backends.backend_pdf import PdfPages
import logging 
module_logger = logging.getLogger(__name__)
from operator import itemgetter
import tkinter as Tkinter
import random
from tkinter import *
import pandas as pd
import tkinter as tk
import numpy as np
import scipy.io as sio
import caiman as cm
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.ticker as mtick
from matplotlib import gridspec
# import torch
# import torch.nn as nn
from random import sample, random, randint
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, pearsonr, spearmanr, ttest_ind, zscore, mode
import copy
from scipy import interpolate
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage.filters import gaussian_filter1d
from scipy.io import loadmat, savemat
import skimage.io
import math
from statsmodels.nonparametric.smoothers_lowess import lowess

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
import caiman as cm
import os
import glob
import scipy.io as spio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import scipy.signal as sg
from scipy import signal
from scipy import interpolate

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
from numbers import Number
from scipy.signal import firwin, filtfilt
from typing import Optional
from numba import njit
from tqdm import tqdm

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
        Integrate daq signals and Prairie voltage sognals
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
                 acquisition_data_path=False,
                 allen_BO_tuple=False,
                 cloud=False,
                 ):      
        self.allen_BO_tuple=allen_BO_tuple
        self.acquisition_object=acquisition_object
        self.nondatabase=nondatabase
        self.new_full_data=new_full_data
        self.data_analysis_path=acquisition_data_path
        self.acquisition_voltage_signals_object=acquisition_voltage_signals_object
        self.metadata_object=metadata_object
        self.preframes=16
        self.stim=33
        self.postframes=16
        
        self.jesus_runs={}
        self.temporary_processing=r'C:\Users\sp3660\Desktop\TemporaryProcessing'
        self.caiman_results=None
        self.pyr_int_ids_and_indexes=None

        
        
        if self.acquisition_object:
            self.set_up_some_paths()
            self.correction_for_LED_clipping()


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
            self.set_up_some_paths()

            self.cell_info, self.traces, self.neuropilinfo, self.locomotion_info, self.metadata, self.pupilinfo,self.masks, self.projection=self.allen.dataset_exploration(self.data_set)
            # test=allen.allen_manifest.get_cell_specimens(ids=cell_info[1][0])
            self.stim_info=self.allen.exploring_stimulus(self.data_set)
            self.check_pyr_int_identif_files()

            self.create_full_data_container_allen()
            self.allen.plotting_traces_and_stim(self.data_set, self.traces[-1], self.locomotion_info[0])
            # self.drifting=self.allen.explore_drifting_analysis( self.data_set, self.traces[-1], self.spikes, selected_cell_index=0, plot=False)
            # self.spont=self.allen.explore_spontaneous_activity( self.data_set, self.traces[-1], self.spikes, selected_cell_index=0, plot=False)
            # self.movies=self.allen.explore_natural_movie_analysis( self.data_set, self.traces[-1], self.spikes, selected_cell_index=0, plot=False)
            self.load_pyr_int_identif()
            
            
        elif cloud:
            self.set_up_some_paths()

            self.create_full_data_container()
            self.create_stim_table()

      
            #%  pyr int identification

            self.check_pyr_int_identif_files()
            self.load_pyr_int_identif()
            self.get_pyr_int_indexing_dict()
            
            
            # subanalysis, this should be mode to sub anlayis object
            self.check_all_jesus_results()
        
        else:   
            self.set_up_some_paths()

            
            if self.acquisition_voltage_signals_object:
                self.volt_object=self.acquisition_voltage_signals_object
                self.load_voltage_signals()

            if self.metadata_object:
                self.create_full_data_container()
#%% path managing
    def set_up_some_paths(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        name_dict={'jesus_runs_path':'JesusRuns',
                   'caiman_runs_path':'CaimanRuns',
                   'nmf_runs_path':'NMFAnalysis',
                   'allen_runs_path':'AllenAnalysis',
                   'CRFs_runs_path':'CRFsResults',
                   'own_tuning_runs_path':'MyOwnTuningAnalysis',
                   'pca_runs_path':'PCA'}

        if self.acquisition_object:
            self.aquisition_id=self.acquisition_object.aquisition_name
            mouse_slow_path=self.acquisition_object.mouse_imaging_session_object.mouse_object.mouse_slow_subproject_path
            self.data_analysis_path=os.path.join(mouse_slow_path,'data',self.acquisition_object.aquisition_name)
            self.data_paths={key:os.path.join(self.data_analysis_path,val) for key,val in name_dict.items()}
            
            self.full_data_path_name='_'.join([self.aquisition_id, timestr,'full_data.pkl'])  
            self.pyr_int_identif_path_name='_'.join([self.aquisition_id, timestr,'pyr_int_identification.pkl'])  
            
        elif self.data_analysis_path:
            self.data_paths={key:os.path.join(self.data_analysis_path,val) for key,val in name_dict.items()}
            
        else:
            
            self.data_paths={key:os.path.join(self.temporary_processing,val) for key,val in name_dict.items()}
            self.full_data_path_name=None
            self.pyr_int_identif_path_name=None
            
        if self.allen_BO_tuple:
            self.aquisition_id='_'.join((str(self.data_set.get_metadata()['experiment_container_id']), str(self.data_set.get_metadata()['ophys_experiment_id'])))

            self.data_analysis_path=os.path.join(self.allen.main_directory,'Containers', str(self.data_set.get_metadata()['experiment_container_id']), str(self.data_set.get_metadata()['ophys_experiment_id']))
            self.data_paths={key:os.path.join(self.data_analysis_path,val) for key,val in name_dict.items()}
            
            self.full_data_path_name='_'.join([  self.aquisition_id, timestr,'full_data.pkl'])  
            self.pyr_int_identif_path_name='_'.join([  self.aquisition_id, timestr,'pyr_int_identification.pkl'])  
            
        for data_path in self.data_paths.values():
            if not os.path.isdir(data_path):
                os.makedirs(data_path)
            
            
           
#%% loading aquisition objects(calcium results and signals)
    def load_all_data_objects(self):
        self.load_all_acquisition_subobjects()
        self.load_calcium_extractions()
        # self.check_caiman_rasters()
        self.load_voltage_signals()

    def load_all_acquisition_subobjects(self):
        self.acquisition_object.load_all(camera=False, kalman=False)
        self.acquisition_object.metadata_object.get_timestamps()
        
        
        
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
        
        if self.signals_object.all_signals:
            self.are_signals=True

            self.signals_object.update_frame_rates_with_metadata(self.metadata_object.translated_imaging_metadata['VoltageRecordingFrequency'], 1000)
            self.signals_object.choose_time_scale('secs')
        else:
            self.are_signals=None

    def update_voltage_signals_frame_rates(self,Prairie_frame_rate, daq_frame_rate):
        
        self.signals_object.update_frame_rates_with_metadata(Prairie_frame_rate, daq_frame_rate)
        
        
    def load_visual_stim_protocol(self):
        pass
        
    def save_activity_array_to_matlab(self,activity_arrays):
        if isinstance(activity_arrays[-2][2],list):
            selectedcells='selected_cell_range_'+'_'.join([str(i) for i in activity_arrays[-2][2][:2]])+'___'+ '_'.join([str(i) for i in activity_arrays[-2][2][-2:]])

            
            
        else:
            selectedcells=activity_arrays[-2][2]
        name='_'.join([activity_arrays[-2][0], activity_arrays[-2][1], activity_arrays[-2][4], selectedcells,'binarization_threshold_'+str(activity_arrays[-2][-1])])
        sio.savemat( os.path.join(self.data_analysis_path,self.aquisition_id+'_'+name+'.mat'),{name:activity_arrays[3]})
        
    
#%% checking datasets
    def check_caiman_rasters(self):
    
        for dataset in self.caiman_extractions.values():
               dataset.CaimanResults_object.plot_final_rasters()
     
#%% full data organization   

    def binarization(self, threshold=0.1):
        
        binarized=np.copy(self.full_data['imaging_data']['Plane1']['Traces']['deconvolved'])

        binarized[binarized>threshold]=1
        binarized[binarized<threshold]=0

        self.full_data['imaging_data']['Plane1']['Traces']['binarized']=binarized
        self.full_data['imaging_data']['Plane1']['Traces']['binarization_threshold']=threshold

        

    def correction_for_LED_clipping(self):
        #things that require correction
        # timestamps
        # voltage synchroniztion
        
        
        self.start_frame, self.end_frame=self.acquisition_object.all_datasets[list(self.acquisition_object.all_datasets.keys())[0]].bidishift_object.load_LED_tips()
        
        
    def create_full_data_container_allen(self):
        self.check_full_data()
        self.load_full_data()
        
        
        module_logger.info('Creating full data container')
        
        if not self.full_data or self.new_full_data:
            
            self.full_data={'imaging_data':{'Frame_rate':'',
                                            'Interplane_period':'',
                                            'Frame_number':'',
                                            'Plane1':{'CellIds':np.arange(self.traces[2].shape[0]),
                                                      'CellNumber':self.traces[2].shape[0],
                                                      'Timestamps':(self.traces[1],self.traces[0]),
                                                      'Traces':{'raw_traces':self.traces[2],
                                                                'demixed_traces':self.traces[3],
                                                                'neuropil_traces':self.traces[4],
                                                                'corrected_traces':self.traces[5],
                                                                'dff_traces':self.traces[6],
                                                                'deconvolved':self.spikes,
                                                                'binarized':'',
                                                                'binarization_threshold':''
    
                                                                }
                                                      }
                                            },
                            'voltage_traces':{},
                            'visstim_info':{'Paradigm_Indexes':{
    
                                                                },
                                            'Drifting_Gratings':{'Resampled_sliced_speed':''
                                                }
                                
                                            }
                                    
                            }
        
        
            self.binarization()

            inter=[]
            pyr=[]
    
            if self.data_set.get_metadata()['cre_line'] in ['Pvalb-IRES-Cre', 'Sst-IRES-Cre','Vip-IRES-Cre']:
                inter=self.full_data['imaging_data']['Plane1']['CellIds'].tolist()
            else:
                pyr=self.full_data['imaging_data']['Plane1']['CellIds'].tolist()
               
            self.pyr_int_identification={}
            self.pyr_int_identification['Plane1']={'interneuron':{'matlab':[i+1 for i in inter],
                                                          'python':inter,
                                                          },
                                           'pyramidals':{'matlab':[i+1 for i in pyr],
                                                         'python':pyr,
                                                         }
                                           }
                
            self.pyr_int_ids_and_indexes={'Plane1':{'pyr':(pyr,np.ones(len(pyr), dtype=bool)),
                                                    'int':(inter,np.ones(len(inter), dtype=bool))
                                                        }
                                          }
                            
            epochtable=self.data_set.get_stimulus_epoch_table()
            
            
            drifdic=['first_drifting_set_first',
                     'first_drifting_set_last',
                     'second_drifting_set_first',
                     'second_drifting_set_last',
                     'third_drifting_set_first',
                     'third_drifting_set_last'
                     ]
            
            drifting_epochs=epochtable[epochtable['stimulus']=='drifting_gratings'].reset_index(drop=True)
            locomotions=[]
            for i in range(3):
                self.full_data['visstim_info']['Paradigm_Indexes'][drifdic[2*i]]=drifting_epochs['start'][i]
                self.full_data['visstim_info']['Paradigm_Indexes'][drifdic[2*i+1]]=drifting_epochs['end'][i]
                locomotions.append(self.locomotion_info[0][drifting_epochs['start'][i]:drifting_epochs['end'][i]])
                
                
            
            self.full_data['visstim_info']['Drifting_Gratings']['Resampled_sliced_speed']=np.hstack(locomotions)
            self.save_pyr_int_identif()
            self.save_full_data()

           
    def create_full_data_container(self):
        module_logger.info('Creating full data container')

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
                if self.are_signals:
                    self.extrac_voltage_visstim_signals()
            
            if not self.nondatabase:
                self.save_full_data()
            
        self.milisecond_period=1000/self.full_data['imaging_data']['Frame_rate']
        module_logger.info('Finished Creating full data container')


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
        self.all_planes_timestamps= copy.deepcopy(self.acquisition_object.metadata_object.timestamps)
        self.all_planes_clipped_timestamps=copy.deepcopy(self.all_planes_timestamps)
        self.clipped_timestamps=False
        if self.start_frame or self.end_frame:
            for key,val in  self.all_planes_timestamps.items():
                all_planes_timestamps_LED_clipped=np.array(val[self.start_frame:self.end_frame])
                self.all_planes_clipped_timestamps[key]=all_planes_timestamps_LED_clipped
                self.clipped_timestamps=True
        else:
            for key,val in  self.all_planes_timestamps.items():
                self.all_planes_clipped_timestamps[key]=np.array(val)

            
        self.all_planes_clipped_timestamps_shifted={}
        for k in self.all_planes_clipped_timestamps.keys():
            self.all_planes_clipped_timestamps_shifted[k]=self.all_planes_clipped_timestamps[k]-self.all_planes_clipped_timestamps[k][0]
        
        # Define dictionary names
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
                          'dfdt_raw':selected_plane_caiman.dfdt_accepted_matrix, 
                          'dfdt_smoothed':selected_plane_caiman.dfdt_thesholded_accepted_matrix, 
                          'dfdt_binary':selected_plane_caiman.binarized_dfdt,
                          'foopsi_raw':selected_plane_caiman.foopsi_matrix, 
                          'foopsi_smoothed':selected_plane_caiman.convolved_foopsi,
                          'foopsi_binary':selected_plane_caiman.binarized_foospi,
                          'mcmc_raw':selected_plane_caiman.MCMC_matrix,
                          'mcmc_smoothed': selected_plane_caiman.convolved_MCMC,
                          'mcmc_binary':selected_plane_caiman.binarized_MCMC,
                          'mcmc_scored_binary':selected_plane_caiman.z_scored_binarized_mcmc_binarized}
            
            self.full_data['imaging_data'][plane]['Traces']=plane_traces 
            if len(self.all_planes_clipped_timestamps[plane])==plane_traces['demixed'].shape[1]:

                self.full_data['imaging_data'][plane]['Timestamps']=(self.all_planes_clipped_timestamps[plane][0:],\
                                                                     (np.arange(0,len(self.all_planes_clipped_timestamps[plane][0:]))/self.full_data['imaging_data']['Frame_rate'])\
                                                                         +self.full_data['imaging_data']['Interplane_period']*plane_number)
            else:
                endclip=len(self.all_planes_clipped_timestamps[plane])-plane_traces['demixed'].shape[1]
     
                self.full_data['imaging_data'][plane]['Timestamps']=(self.all_planes_clipped_timestamps[plane][0:-endclip],\
                                                                     (np.arange(0,len(self.all_planes_clipped_timestamps[plane][0:-endclip]))/self.full_data['imaging_data']['Frame_rate'])\
                                                                         +self.full_data['imaging_data']['Interplane_period']*plane_number)
                
                
                
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
                if matrix.any() :
                    self.full_data['imaging_data']['All_planes_rough']['Traces'][matrix_name]=np.concatenate((self.full_data['imaging_data']['All_planes_rough']['Traces'][matrix_name],  matrix), axis=0)
        
        # cell_numbers={k:len(plane['CellIds']) for k,plane in self.full_data['imaging_data'].items() if 'Plane' in k }
        self.full_data['imaging_data']['Frame_number']= self.full_data['imaging_data'][plane]['Traces']['demixed'].shape[1]
        self.full_data['imaging_data']['All_planes_rough']['CellIds']={k:plane['CellIds'] for k,plane in self.full_data['imaging_data'].items() if 'Plane' in k }
        self.full_data['imaging_data']['All_planes_rough']['Timestamps']= self.full_data['imaging_data']['Plane1']['Timestamps']
        self.full_data['imaging_data']['All_planes_rough']['CellNumber']= sum(len(plane) for plane in  self.full_data['imaging_data']['All_planes_rough']['CellIds'].values())
        
        self.mov_timestamps_seconds={'raw':np.array(self.all_planes_timestamps['Plane1']), # from metadata
                        'clipped':self.full_data['imaging_data']['Plane1']['Timestamps'][0], # clipped to start frame
                        'shifted':self.all_planes_clipped_timestamps_shifted['Plane1'], #substracting first timstamp to clipped
                        'shifted_recalculated':self.full_data['imaging_data']['Plane1']['Timestamps'][1] # by taking shifted and doin linspace with imaging para,meters
                        }
        
        self.mov_timestamps_miliseconds={k:v*1000 for k,v in self.mov_timestamps_seconds.items()}

#%% signal data managing
    def resample(self, x, factor, kind='linear'):
        n = int(np.floor(x.size / factor))
        f = interpolate.interp1d(np.linspace(0, 1, x.size), x, kind)
        return f(np.linspace(0, 1, n))     
    
    
    # THis method I will move to acquisition when finsihed, for the moment keep it here. Ruuning during analysis but integrate in acquisition as weel as in the allen processing
    # THIs loads the rasw video to do the mean so it takes time to load
    # at the moment I have single planes only so I will have to dapat to multiplane somwhow
    # this hsould be done only once, after i will have to save the synchornization and clippinh somehow
    
    
     


    def review_aligned_signals_and_transitions(self,led_clipeed=False):
        
        self.mean_movie_path =self.calcium_datasets[list(self.calcium_datasets.keys())[0]].bidishift_object.mean_movie_path
        mean_mov=np.load(self.mean_movie_path)
        timestamps_voltage_signals=self.full_data['voltage_traces']['Full_signals']['Prairie']['LED_aligned']['traces']['PhotoTrig'].index.values
        mov=self.scale_signal(mean_mov)
        surrounddown=np.array([[i-1,i,i+1] for i in [self.start_frame]])
        surroundup=np.array([[i-1,i,i+1] for i in [self.end_frame]])
       
     

        
        deriv=lambda x:np.diff(x,prepend=x[0] )
        rectified=lambda x:np.absolute(x)
 
        signals_to_plot=[ 'VisStim',  'LED',  'PhotoStim',  'PhotoTrig',  'AcqTrig']
        self.full_data['voltage_traces']['Full_signals']['Prairie']['LED_aligned']['traces']
        
        
        #plot prairie daq alignment
        # plt.close('all')
        for sig in reversed(self.full_data['voltage_traces']['Full_signals']['Prairie']['Raw']['traces'].columns[:-1]): 
            daq_sig=sg.medfilt(np.squeeze( self.full_data['voltage_traces']['Full_signals']['daq']['Raw']['traces'][sig].values), kernel_size=1)
            prairie_sig=sg.medfilt(np.squeeze(  self.full_data['voltage_traces']['Full_signals']['Prairie']['Raw']['traces'][sig].values), kernel_size=1)           
            f,ax=plt.subplots()
            ax.plot(daq_sig)
            ax.plot(prairie_sig,'r')
            f,ax=plt.subplots()
            ax.plot( self.full_data['voltage_traces']['Full_signals']['daq']['Prairie_aligned']['traces'][sig].values)
            ax.plot(self.full_data['voltage_traces']['Full_signals']['Prairie']['Raw']['traces'][sig].values,'r')
        

        #plotting clipped signal to video length
        signals_to_plot=[  'LED',  'AcqTrig']
        for record_to_plot, process in self.full_data['voltage_traces']['Full_signals'].items():
            f,ax=plt.subplots()
            for sig in  signals_to_plot:
                ax.plot(self.scale_signal(process['Raw']['traces'][sig].values),label=sig)
                ax.plot(self.scale_signal(process['Movie_length_clipped']['traces'][sig].values),label=sig)   
            ax.plot(self.mov_timestamps_miliseconds['raw'],mov,label='Video')
            ax.legend()

      
        
        
        #plot LED aligned not clipped signals
        # plt.close('all')
        for record_to_plot, process in self.full_data['voltage_traces']['Full_signals'].items():
            f,ax=plt.subplots(len(signals_to_plot)+1,sharex=True)
            for k,sigk in enumerate(signals_to_plot):
                sig=self.scale_signal(process['LED_aligned']['traces'][sigk].values)
                locomotion=process['LED_aligned']['traces']['Locomotion'].values
                speed=self.scale_signal(rectified(deriv(locomotion)))
                speedtimestamps=process['LED_aligned']['traces']['Locomotion'].index.values
                upt=self.full_data['voltage_traces']['Full_signals_transitions'][sigk]['aligned'][record_to_plot]['up']
                downt=self.full_data['voltage_traces']['Full_signals_transitions'][sigk]['aligned'][record_to_plot]['down']
                upt_downsampled=self.full_data['voltage_traces']['Full_signals_transitions'][sigk]['aligned_downsampled'][record_to_plot]['up']
                downt_downsampled=self.full_data['voltage_traces']['Full_signals_transitions'][sigk]['aligned_downsampled'][record_to_plot]['down']
                surrounddown=np.array([[i-1,i,i+1] for i in downt_downsampled])
                surroundup=np.array([[i-1,i,i+1] for i in upt_downsampled])


                ax[k].plot(self.mov_timestamps_miliseconds['raw'], mov)
                ax[k].plot(speedtimestamps, speed)
                ax[k].vlines(self.mov_timestamps_miliseconds['raw'][self.start_frame],0,1,linestyles='solid',alpha=1)
                ax[k].vlines(self.mov_timestamps_miliseconds['raw'][self.end_frame],0,1,linestyles='solid',alpha=1)
                ax[k].vlines(self.mov_timestamps_miliseconds['raw'][surrounddown], 0,1,linestyles='dashed',alpha=0.2)
                if surroundup.any():
                    ax[k].vlines(self.mov_timestamps_miliseconds['raw'][surroundup], 0,1,linestyles='dashdot',alpha=0.2)
                ax[k].plot( timestamps_voltage_signals,sig )
                ax[k].plot( upt,sig[upt] ,'^')
                ax[k].plot( downt,sig[downt] ,'v')
                ax[k].plot(self.mov_timestamps_miliseconds['raw'][upt_downsampled].astype(int),sig[ self.mov_timestamps_miliseconds['raw'][upt_downsampled].astype(int)] ,'<')
                ax[k].plot(self.mov_timestamps_miliseconds['raw'][downt_downsampled].astype(int),sig[self.mov_timestamps_miliseconds['raw'][downt_downsampled].astype(int)] ,'>')
                ax[k].vlines(self.mov_timestamps_miliseconds['raw'][surrounddown], 0,1,linestyles='dashed',alpha=0.2)
                if surroundup.any():
                    ax[k].vlines(self.mov_timestamps_miliseconds['raw'][surroundup], 0,1,linestyles='dashdot',alpha=0.2)
            ax[-1].plot(self.mov_timestamps_miliseconds['raw'], mov)
            ax[-1].plot(speedtimestamps, speed,alpha=0.2) 
            ax[-1].set_ylim(0,0.02)            
              
                    
        #plot led clipeed signals
        signals_to_plot=[ 'VisStim', 'PhotoStim',  'PhotoTrig', ]
        for record_to_plot, process in self.full_data['voltage_traces']['Full_signals'].items():
            f,ax=plt.subplots(len(signals_to_plot)+1,sharex=True)
            for k,sigk in enumerate(signals_to_plot):
                sig=self.scale_signal(process['LED_clipped']['traces'][sigk].values)
                locomotion=process['LED_clipped']['traces']['Locomotion'].values
                speed=self.scale_signal(rectified(deriv(locomotion)))
                speedtimestamps=process['LED_clipped']['traces']['Locomotion'].index.values
                mov_clipped=self.scale_signal(mean_mov[self.start_frame:self.end_frame])
                upt=self.full_data['voltage_traces']['Full_signals_transitions'][sigk]['aligned_downsampled_LEDshifted'][record_to_plot]['up']
                downt=self.full_data['voltage_traces']['Full_signals_transitions'][sigk]['aligned_downsampled_LEDshifted'][record_to_plot]['down']
                surrounddown=np.array([[i-1,i,i+1] for i in downt])
                surroundup=np.array([[i-1,i,i+1] for i in upt])
                 
                ax[k].plot(self.mov_timestamps_miliseconds['shifted'], mov_clipped)
                ax[k].plot(speedtimestamps,sig )
                ax[k].plot(speedtimestamps,speed )
                ax[k].plot(self.mov_timestamps_miliseconds['shifted'][upt].astype(int),sig[ self.mov_timestamps_miliseconds['shifted'][upt].astype(int)] ,'<')
                ax[k].plot(self.mov_timestamps_miliseconds['shifted'][downt].astype(int),sig[self.mov_timestamps_miliseconds['shifted'][downt].astype(int)] ,'>')
                ax[k].vlines(self.mov_timestamps_miliseconds['shifted'][surrounddown], 0,1,linestyles='dashed',alpha=0.2)
                ax[k].vlines(self.mov_timestamps_miliseconds['shifted'][surroundup], 0,1,linestyles='dashdot',alpha=0.2)

            ax[-1].plot(self.mov_timestamps_miliseconds['shifted'], mov_clipped)
            ax[-1].plot(speedtimestamps, speed,alpha=0.2) 
                  

    
        plt.show()
            
        

        # def check_clip_voltages_to_movie_length(self):
        
       
        
       
            

  

    def extrac_voltage_visstim_signals(self):
 
    
        
        self.milisecond_period=1000/self.full_data['imaging_data']['Frame_rate']
        voltagerate=self.metadata_object.translated_imaging_metadata['VoltageRecordingFrequency']
        self.full_data['voltage_traces']['Speed']=self.resample(self.signals_object.rectified_speed_array['Prairie']['Locomotion'][:], factor=self.milisecond_period, kind='linear').squeeze()
        self.full_data['voltage_traces']['Acceleration']=self.resample(self.signals_object.rectified_acceleration_array['Prairie']['Locomotion'][:], factor=self.milisecond_period, kind='linear').squeeze() 
        if self.signals_object.correct_voltages:
            self.full_data['voltage_traces']['VisStim']=self.signals_object.all_final_signals['Prairie']['LED_clipped']['traces']['VisStim'].values
            self.full_data['voltage_traces']['Photodiode']=''
            self.full_data['voltage_traces']['Start_End']=self.signals_object.all_final_signals['Prairie']['LED_clipped']['traces']['AcqTrig'].values
            self.full_data['voltage_traces']['LED']=self.signals_object.all_final_signals['Prairie']['LED_clipped']['traces']['LED'].values
            self.full_data['voltage_traces']['Optopockels']=self.signals_object.all_final_signals['Prairie']['LED_clipped']['traces']['PhotoStim'].values
            self.full_data['voltage_traces']['OptoTrigger']=self.signals_object.all_final_signals['Prairie']['LED_clipped']['traces']['PhotoTrig'].values
            self.full_data['voltage_traces']['Full_signals']=self.signals_object.all_final_signals
            self.full_data['voltage_traces']['Full_signals_transitions']=self.signals_object.signal_transitions
            
        else:
            if self.signals_object.rounded_vis_stim:
                self.full_data['voltage_traces']['VisStim']=self.resample(self.signals_object.rounded_vis_stim['Prairie']['VisStim'][:], factor=self.milisecond_period, kind='linear').squeeze()
                self.full_data['voltage_traces']['Photodiode']=''
                self.full_data['voltage_traces']['Start_End']=''
                self.full_data['voltage_traces']['LED']=''
                self.full_data['voltage_traces']['Optopockels']=''
                self.full_data['voltage_traces']['OptoTrigger']=''


        if self.signals_object.vis_stim_protocol and self.signals_object.transitions_dictionary:
            self.full_data['visstim_info']['Paradigm_Indexes']={key:(np.abs(self.full_data['imaging_data']['Plane1']['Timestamps'][0] - index/1000)).argmin() for key, index in self.signals_object.transitions_dictionary.items()}
            
            frame_starts=np.zeros([10,900])  
            frame_ends=np.zeros([10,900])                                          
                                        
            it = np.nditer(self.signals_object.movie_one_frame_index_full_recording[:,:,0], flags=['multi_index'])                              
            for x in it:                
                frame_starts[it.multi_index]=np.abs(np.array(self.full_data['imaging_data']['Plane1']['Timestamps'][0]) -x/voltagerate).argmin()
                  
            it = np.nditer(self.signals_object.movie_one_frame_index_full_recording[:,:,1], flags=['multi_index'])                              
            for y in it:                
                frame_ends[it.multi_index]=np.abs( np.array(self.full_data['imaging_data']['Plane1']['Timestamps'][0]) -y/voltagerate).argmin()
            
            self.full_data['visstim_info']['Full']={ 'Resampled_sliced_speed':self.resample(self.signals_object.rectified_speed_array['Prairie']['Locomotion'], factor=self.milisecond_period, kind='linear').squeeze(),
                                                    'Resampled_sliced_visstim':self.resample(self.signals_object.rounded_vis_stim['Prairie']['VisStim'], factor=self.milisecond_period, kind='linear').squeeze(),
               }
            self.full_data['visstim_info']['Movie1']={'Frame_Starts':frame_starts,
                                                      'Frame_Ends':frame_ends,                                                    
                                                      'Resampled_sliced_speed':self.resample(self.signals_object.natural_movie_one_set_speed, factor=self.milisecond_period, kind='linear').squeeze(),
                                                      'Resampled_sliced_visstim':self.resample(self.signals_object.natural_movie_one_set, factor=self.milisecond_period, kind='linear').squeeze(),
                                                        }
     
            
            self.full_data['visstim_info']['Spontaneous']={}
            self.full_data['visstim_info']['Spontaneous']['stimulus_table']= pd.DataFrame( ([self.full_data['visstim_info']['Paradigm_Indexes']['spont_first'],self.full_data['visstim_info']['Paradigm_Indexes']['spont_last']] ,), columns =['start', 'end'])
    
            if self.signals_object.vis_stim_protocol =='AllenA':
            
                self.full_data['visstim_info']['Drifting_Gratings']={'Indexes':{'Drift_on':np.vstack([[(np.abs( self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.tuning_stim_on_index_full_recording]),
                                                                                'Drift_off':np.vstack([[(np.abs(self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.tuning_stim_off_index_full_recording]),
                                                                                'Blank_sweep_on':np.vstack([[(np.abs( self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.blank_sweep_on_index_full_recording]),
                                                                                'Blank_sweep_off':np.vstack([[(np.abs( self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.blank_sweep_off_index_full_recording])
                                                                                },
                                                                     # 'Binary_Maytrix_downsampled':np.vstack([self.resample(self.signals_object.full_stimuli_binary_matrix[srtim], 
                                                                     #                                                       factor=self.milisecond_period, kind='linear').squeeze() for srtim in range (self.signals_object.full_stimuli_binary_matrix.shape[0])]),
                                                                     # 'Binary_Maytrix_recreated':'',
                                                                     'Resampled_sliced_speed':self.resample(np.concatenate((self.signals_object.first_drifting_set_speed, 
                                                                                                                            self.signals_object.second_drifting_set_speed, 
                                                                                                                            self.signals_object.third_drifting_set_speed)), factor=self.milisecond_period, kind='linear').squeeze(),
                                                                     'Resampled_sliced_visstim':self.resample(np.concatenate((self.signals_object.first_drifting_set, 
                                                                                                                              self.signals_object.second_drifting_set,
                                                                                                                              self.signals_object.third_drifting_set)), factor=self.milisecond_period, kind='linear').squeeze()
                                                                     
                                                                     }
                # self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_recreated']=np.zeros((self.signals_object.full_stimuli_binary_matrix.shape ))
                # for i, row in enumerate(self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on']):
                #     for j, trial in enumerate(row):
                #         self.full_data['visstim_info']['Drifting_Gratings']['Binary_Maytrix_recreated'][i,self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_on'][i,j]:self.full_data['visstim_info']['Drifting_Gratings']['Indexes']['Drift_off'][i,j]]=1
                
            
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
                all_static_grating_onsets=np.sort(np.append(self.signals_object.static_grat_even_index_full_recording,self.signals_object.static_grat_odd_index_full_recording))
                all_natural_images_onsets=np.sort(np.append(self.signals_object.natural_image_even_index_full_recording,self.signals_object.natural_image_odd_index_full_recording))
            
  
                static_gratings_trial_structure=pd.DataFrame({'onsets':all_static_grating_onsets,
                                                              'grating_id': self.acquisition_object.allstaticindexes[:all_static_grating_onsets.shape[0]]})


                natural_images_trial_structure=pd.DataFrame({'onsets':all_natural_images_onsets,
                                                              'image_id': self.acquisition_object.allnatiuralindexes[:all_natural_images_onsets.shape[0]]})

                indexes={}
                for grating_id in range(0,120+1):
                    indexes[grating_id]=[]
                    for trial in range(static_gratings_trial_structure[static_gratings_trial_structure['grating_id']==grating_id].shape[0]):  
                        idx=static_gratings_trial_structure[static_gratings_trial_structure['grating_id']==grating_id].iloc[trial][0]
                        indexes[grating_id].append(np.abs( np.array(self.full_data['imaging_data']['Plane1']['Timestamps'][0]) -idx/voltagerate).argmin() )





                self.full_data['visstim_info']['Static_Gratings']={'Indexes':indexes,
                                                           'Resampled_sliced_speed':self.resample(np.concatenate((self.signals_object.first_static_set_speed, 
                                                                                                                  self.signals_object.second_static_set_speed, 
                                                                                                                  self.signals_object.third_static_set_speed)), factor=self.milisecond_period, kind='linear').squeeze(),
                                                           'Resampled_sliced_visstim':self.resample(np.concatenate((self.signals_object.first_static_set, 
                                                                                                                    self.signals_object.second_static_set,
                                                                                                                    self.signals_object.third_static_set)), factor=self.milisecond_period, kind='linear').squeeze()
                                                            }
                indexes={}
                for image_id in range(0,118+1):
                    indexes[image_id]=[]
                    for trial in range(natural_images_trial_structure[natural_images_trial_structure['image_id']==image_id].shape[0]):  
                        idx=natural_images_trial_structure[natural_images_trial_structure['image_id']==image_id].iloc[trial][0]
                        indexes[image_id].append(np.abs( np.array(self.full_data['imaging_data']['Plane1']['Timestamps'][0]) -idx/voltagerate).argmin() )
                
                self.full_data['visstim_info']['Natural_Images']={'Indexes':indexes,
                                                          'Resampled_sliced_speed':self.resample(np.concatenate((self.signals_object.first_images_set_speed, 
                                                                                                                 self.signals_object.second_images_set_speed, 
                                                                                                                 self.signals_object.third_images_set_speed)), factor=self.milisecond_period, kind='linear').squeeze(),
                                                          'Resampled_sliced_visstim':self.resample(np.concatenate((self.signals_object.first_images_set, 
                                                                                                                   self.signals_object.second_images_set,
                                                                                                                   self.signals_object.third_images_set)), factor=self.milisecond_period, kind='linear').squeeze()
                                                            }
         
            
            
            
    def scale_signal(self,signal,method='ZeroOne'):
        import scipy.stats as stats
        if method=='ZeroOne':
            scaled_signal=(signal-np.min(signal))/(np.max(signal)-np.min(signal))
                                            
        elif method=='ZScored':
            scaled_signal=stats.zscore(signal)
                             
        return scaled_signal

        
    def manually_setting_opto_times_and_cells(self):
        
        if os.path.split(self.acquisition_object.aquisition_name)[1]=='230321_SPRB_3MinTest_25x_920_51020_63075_with-001':
            plt.close('all')

            self.mean_movie_path =self.calcium_datasets[list(self.calcium_datasets.keys())[0]].bidishift_object.mean_movie_path
            mean_mov=np.load(self.mean_movie_path)
        
            cell=[0,4,6,9]
            # this come form the preliminary matlab analysis i did for rafa
            estimatedoptotimes=np.array([1003,1422, 1647, 2110, 2221, 2923, 3525, 3666, 3806, 4209])
            estimatedoptotimes=estimatedoptotimes-1
            
            estimatedlocomotionbouts=np.array([1230, 1985, 2698,3325,  4010, 4500])
            estimatedlocomotionbouts=estimatedlocomotionbouts-1
            prestim=20
            poststim=40
            
            
            
            backgroundcorrection=mean_mov[2786:2877]
            
            firstcorrectionlenght=len(np.arange(994,1079))
            secondtcorrectionlenght=len(np.arange(1432,1494))
            thirdcorrectionlenght=len(np.arange(3575,3619))
            fourthcorrectionlenght=len(np.arange(4214,4252))
            
            
            mean_mov[994:1079]=np.random.choice(backgroundcorrection, firstcorrectionlenght)
            mean_mov[1432:1494]=np.random.choice(backgroundcorrection, secondtcorrectionlenght)
            mean_mov[3575:3619]=np.random.choice(backgroundcorrection, thirdcorrectionlenght)
            mean_mov[4214:4252]=np.random.choice(backgroundcorrection, fourthcorrectionlenght)
            
            mean_mov[np.argmin(np.abs(fulltimevector-73)):np.argmin(np.abs(fulltimevector-83))]=\
                mean_mov[np.argmin(np.abs(fulltimevector-73)):np.argmin(np.abs(fulltimevector-83))]-4
            

            
            
            f,ax=plt.subplots(2, sharex=True)
            ax[0].plot(self.smooth_trace(mean_mov,10))
            baseline=140
            meanmovcop=copy.deepcopy(mean_mov)
            meanmovcop[meanmovcop<baseline]=0
            meanmovcop[meanmovcop>baseline]=1
            loc=self.smooth_trace(meanmovcop,10)
            ax[1].plot(loc)
            
            t=np.arange(len(mean_mov))
            threshold=baseline
            
            locomotionbouts=loc>0.8
            
            np.cumsum(locomotionbouts)
            
            loctrans=np.where(np.diff(locomotionbouts)>0.5)[0]
            u=loctrans.tolist()
            locbouts=list(zip(u[::2], u[1::2]))
            
            durations=[loc[1]-loc[0] for loc in locbouts]
            
            
            singleframelocs=np.where(np.array(durations)==1)[0]
            
            singleframelocspairs=np.array(locbouts)[singleframelocs]
            
            newloc=np.zeros(loc.shape)
            
            multiframelocs=np.where(np.array(durations)>11)[0]
            np.array(locbouts)[multiframelocs]
            np.array(durations)[multiframelocs]
            
            newloc[singleframelocspairs[:,0]+1]=2.5
            plt.plot(loc)
            plt.plot(newloc)
            
            
          
            toadd=[]
            for i,dur in enumerate(np.array(durations)[multiframelocs]):
                adding=[]
                currenttime=0
                j=0
                while currenttime<dur:
                    if currenttime>0:
                        currenttime=adding[j-1]+randint(1, 5)
                        adding.append(currenttime)
                        j=j+1
                    else:
                        adding.append(0)
                        currenttime=1
                        j=j+1
                toadd.append(adding)  
                
                
                
                
            for k,multi in enumerate(multiframelocs):
            
                newloc[np.array(toadd[k])+locbouts[multi][0]]=2.5
           
            plt.plot(np.abs(np.diff(newloc)))
                
                
                
            plt.close('all')
            f,ax=plt.subplots(4, sharex=True)
            ax[0].plot(self.smooth_trace(mean_mov,10))
            ax[1].plot(loc)
            ax[2].plot(newloc)
            ax[3].plot(np.abs(np.diff(newloc)))


            sigma=1
            thr_isi=0.2
            gx = np.arange(-3*sigma, 3*sigma, thr_isi)
            gaussian = np.exp(-(gx/sigma)**2/2)
            speed = np.convolve(np.abs(np.diff(newloc,prepend=0)), gaussian, mode='same')
            
            
    #%%
            #this i think is to creqate the locomotion based on the video signals
            # # Load the recorded voltage trace data
            # input_trace = mean_mov
            
            # # Define the threshold voltage value
            # threshold = 0.5
            
            # # Create a new output trace array initialized with all zeros
            # output_trace = np.zeros_like(input_trace)
            
            # # Define the parameters for the variable frequency oscillation
            # freq_range = (1, 10)  # Hz
            # amplitude = 1.0  # V
            # duration = 0.1  # seconds
            
            # # Iterate over the input trace array
            # for i, v in enumerate(input_trace):
            #     if v > threshold:
            #         # Set the corresponding segment in the output trace to the variable frequency oscillation
            #         freq = np.random.uniform(*freq_range)
            #         t = np.linspace(0, duration, int(duration * 1000), endpoint=False)
            #         oscillation = amplitude * np.sin(2 * np.pi * freq * t)
            #         output_trace[i:i+len(oscillation)] = oscillation
            #     else:
            #         # Set the corresponding segment in the output trace to zero
            #         output_trace[i] = 0
            
            # # Plot the input and output traces
            # plt.plot(input_trace, label='Input')
            # plt.plot(output_trace, label='Output')
            # plt.legend()
            # plt.show()
            
            
            stimnumber=5
            frequency=20#hz
            duration=20/1000#ms
            stimperiod=1/frequency #s
            isi=stimperiod-duration
            
            optotimes=np.arange(0,5*stimperiod,stimperiod)
    
            plt.rcParams["figure.figsize"] = [16, 5]
            plt.rcParams["figure.autolayout"] = True
            smoothwindows=10
            fr=self.metadata_object.translated_imaging_metadata['FinalFrequency']
            framen=self.full_data['imaging_data']['Plane1']['Traces']['demixed'].shape[1]
            
            
            lengthtime=framen/fr
            period=1/fr
            fulltimevector=np.arange(0,lengthtime,period)
    
            f,ax=plt.subplots(7)    
            for i in range(6):
                trace=self.full_data['imaging_data']['Plane1']['Traces']['demixed'][i,:]
                if i==3:
                    ax[i].plot(fulltimevector,self.smooth_trace(trace,10),c='y')
                else:
                    ax[i].plot(fulltimevector,self.smooth_trace(trace,10),c='g')
    
                for i,a in enumerate(ax):
                    a.margins(x=0)
                    if i<len(ax)-1:
                        a.axis('off')
                 
                    elif i==len(ax)-1:
                        
                        a.spines['top'].set_visible(False)
                        a.spines['right'].set_visible(False)
                        a.spines['left'].set_visible(False)
                        a.get_yaxis().set_ticks([])
                        
                        
            ax[-1].plot(fulltimevector,speed)

                        
            
            f,ax=plt.subplots(2,sharex=True)
            f.tight_layout()
            for i in range(3,4):
                trace=self.full_data['imaging_data']['Plane1']['Traces']['demixed'][i,:]
                ax[0].plot(fulltimevector,self.smooth_trace(trace,smoothwindows),'k')
                
                ax[0].margins(x=0)
                for j in range(len(estimatedoptotimes)):
                    ax[0].axvline(x=fulltimevector[estimatedoptotimes[j]],c='r')  
                ax[0].set_ylabel('Activity(a.u.)')
                f.suptitle('Optogenetic Stimulation of Chandelier Cells', fontsize=16)
                
                # ax[0].spines['bottom'].set_visible(False)

                # ax[-1].spines['top'].set_visible(False)

                
                ax[1].plot(fulltimevector,zscore(speed),'k')
                ax[-1].set_ylabel('Speed')
                ax[-1].set_xlabel('Time(s)')


    
               
            smoothedtraces=np.zeros_like(self.full_data['imaging_data']['Plane1']['Traces']['demixed'])        
            for i in range(6):
                smoothedtraces[i,:]=self.smooth_trace(self.full_data['imaging_data']['Plane1']['Traces']['demixed'][i,:],smoothwindows)
                  
            self.optotrialarraysmoothed=np.zeros((6,10,60))
            self.trialtimevector=np.arange(-period*prestim,period*poststim,period)
    
    
            for i,opto  in enumerate(estimatedoptotimes):
                for j in range(6):
                    self.optotrialarraysmoothed[j,i,:]=smoothedtraces[j,opto-prestim:opto+poststim]
            
            meanactivations=self.optotrialarraysmoothed[:,[0,4,6,9],:].mean(axis=1)
            meanalocomotion=self.optotrialarraysmoothed[:,7,:]
            meannonactivations=self.optotrialarraysmoothed[:,[1,2,3,5,8],:].mean(axis=1)
            
            self.optotrialarraysmoothedDF=np.zeros_like(self.optotrialarraysmoothed)
            meanbaselinesmoothed=self.optotrialarraysmoothed[:,:,0:20].mean(axis=2)
            for i in range(6):
                for j in range(10):
                    self.optotrialarraysmoothedDF[i,j,:]=(self.optotrialarraysmoothed[i,j,:]-meanbaselinesmoothed[i,j])


            f,ax=plt.subplots(5,2)
            for i,opto  in enumerate(estimatedoptotimes):
                for j in range(3,4):
                    row = i // 2  # determine the row index based on the iteration index
                    col = i % 2   # determine the column index based on the iteration index
                    ax[row, col].plot(self.trialtimevector,self.optotrialarraysmoothedDF[j,i,:],'k')
                    ax[row, col].axvline(x=0,c='r')  
                    ax[row, col].set_ylim(-1,3)
                    ax[row, col].margins(x=0)
                    for m in optotimes:
                        ax[row, col].add_patch(Rectangle((m, 2.8), 0.01, 0.2,color='r'))
                    ax[row, col].set_xlabel('Time(s)')
                    ax[row, col].set_ylabel('Activity(a.u.)')
                    
                f.suptitle('Single Trial Optogenetic Stimulation', fontsize=16)

                           
            #plotting all trials and mean for chandelier
            fullmeandf=self.optotrialarraysmoothedDF.mean(axis=1)
            cells=['Pyr1','Pyr2','Pyr3','Chand','Pyr4','Pyr5']
            for j in range(0,6):
                f,ax=plt.subplots(1)
                for i,opto  in enumerate(estimatedoptotimes):
                    ax.plot(self.trialtimevector,self.optotrialarraysmoothedDF[j,i,:],c='k',alpha=0.2)
                    ax.plot(self.trialtimevector,fullmeandf[j,:],c='k')
                    ax.axvline(x=0,c='r')  
                    ax.set_ylim(-1,3)
                    ax.margins(x=0)
                    
                    ax.set_xlabel('Time(s)')
                    ax.set_ylabel('Activity(a.u.)')
                    for m in optotimes:
                        ax.add_patch(Rectangle((m, 2.8), 0.01, 0.2,color='r'))
                    
                f.suptitle(f'Global Optostimulation PSTH {cells[j]}', fontsize=16)
                
                
            meanactivationsbasesub=self.optotrialarraysmoothedDF[:,[0,4,6,9],:].mean(axis=1)
            meanalocomotionbasesub=self.optotrialarraysmoothedDF[:,7,:]
            meannonactivationsbasesub=self.optotrialarraysmoothedDF[:,[1,2,3,5,8],:].mean(axis=1)       
            for i in range(3,4):
                f,ax=plt.subplots(3,sharex=True)
    
                ax[0].plot(self.trialtimevector,meanactivationsbasesub[i,:])
                ax[1].plot(self.trialtimevector,meannonactivationsbasesub[i,:])  
                ax[2].plot(self.trialtimevector,meanalocomotionbasesub[i,:])
                for k in [0,4,6,9]:
                    ax[0].plot(self.trialtimevector,self.optotrialarraysmoothedDF[i,k,:],c='k',alpha=0.2)
                for k in [7]:
                    ax[2].plot(self.trialtimevector,self.optotrialarraysmoothedDF[i,k,:],c='k',alpha=0.2)  
                for k in [1,2,3,5,8]:
                    ax[1].plot(self.trialtimevector,self.optotrialarraysmoothedDF[i,k,:],c='k',alpha=0.2)
                
                for j in range(3):
                    ax[j].margins(x=0)
                    ax[j].set_ylim(-1,3)
                    ax[j].axvline(x=0,c='r')  
                    for m in optotimes:
                        ax[j].add_patch(Rectangle((m, 2.8), 0.01, 0.2,color='r'))
                    
                    ax[j].set_ylabel('Activity(a.u.)')
                ax[-1].set_xlabel('Time(s)')
                ax[0].set_title('Responsive Trials')
                ax[1].set_title('Unresponsive Trials')
                ax[2].set_title('Running Trials')
                f.suptitle(f'Trial segmented PSTH {cells[i]}', fontsize=16)
                    
            f,ax=plt.subplots(1)
            for i in range(3,6):
                trace=self.full_data['imaging_data']['Plane1']['Traces']['demixed'][i,:]
                ax.plot(self.smooth_trace(trace,10))
                for j in range(len(estimatedlocomotionbouts)):
                    ax.axvline(x=estimatedlocomotionbouts[j],c='r')  
               
            smoothedtraces=np.zeros_like(self.full_data['imaging_data']['Plane1']['Traces']['demixed'])        
            for i in range(6):
                smoothedtraces[i,:]=self.smooth_trace(self.full_data['imaging_data']['Plane1']['Traces']['demixed'][i,:],10)
                  
                
            locomotiontrialarraysmoothed=np.zeros((3,6,60))
            locomotiontrialarraydenoised=np.zeros((3,6,60))

            f,ax=plt.subplots(3,2)
            for i,opto  in enumerate(estimatedlocomotionbouts):

                for j in range(3,6):
                    row = i // 2  # determine the row index based on the iteration index
                    col = i % 2   # determine the column index based on the iteration index
                    locomotiontrialarraysmoothed[j-3,i,:]=smoothedtraces[j,opto-prestim:opto+poststim]

                    ax[row, col].plot(locomotiontrialarraysmoothed[j-3,i,:])

                    ax[row, col].axvline(x=prestim,c='r')  
                    ax[row, col].set_ylim(-1,5)
                            
            locomeanactivations=locomotiontrialarraysmoothed[:,:,:].mean(axis=1)

            f,ax=plt.subplots(2)
            for i in range(locomeanactivations.shape[0]):
                ax[0].plot(locomeanactivations[i,:])
                ax[1].plot(meanactivations[i,:])

                ax[1].axvline(x=20,c='r')  
                ax[0].axvline(x=20,c='r')  
                ax[0].set_ylim(-1,4)
                ax[1].set_ylim(-1,4)
                 
            #% making some scatter for rafa
            
                        
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score
            import statsmodels.api as sm

            
            responsivetrials=[0,4,6,9]
            locomotiontrials=[7]
            nonresponsivetrials=[1,2,3,5,8]
            
            trislstruct=[responsivetrials,nonresponsivetrials,locomotiontrials]
            meantrislstructsubs=[meanactivationsbasesub,meannonactivationsbasesub,meanalocomotionbasesub]
            meantrislstruct=[meanactivations,meannonactivations,meanalocomotion]

            colors=['r','b','y','tab:orange','k']
            shape=['x','o','v','^','<']
            legend=['Pyr1','Pyr2']

            f,ax=plt.subplots(3)
            line_handles=[]
            for m in range(3):
                for j in range(1,3):
                    activity=self.optotrialarraysmoothed[:,trislstruct[m],:]
                    for i in range(activity.shape[1]):
                        scatter=ax[m].scatter(activity[0,i,:],activity[j,i,:],color=colors[i],marker=shape[j-1])
                        line_handles.append(scatter)
                        ax[m].set_ylim(-1,2.5)               
                    ax[0].set_title('Responsive Trials')
                    ax[1].set_title('Unresponsive Trials')
                    ax[2].set_title('Running Trials')            
                    
                    
                # ax[0].legend(,
                #    ,
                #    scatterpoints=1,
                #    loc='lower left',
                #    ncol=3,
                #    fontsize=8)
                    
            f.suptitle(f'Trial segmented Scatter Raw', fontsize=16)
            

          

            f,ax=plt.subplots(3)
            for m in range(3):
                for j in range(1,3):
                    activity=meantrislstruct[m]
                    ax[m].scatter(activity[0,:],activity[j,:],color=colors[j-1])
                    ax[m].set_ylim(-0.6,1.5)
                    ax[m].set_xlim(-0.5,2)
                    
                    x=activity[0,:].reshape((-1, 1))
                    y=activity[j,:]
                    model = LinearRegression().fit(x,y)
                    r_sq = model.score(x, y)
                    print(f"coefficient of determination: {r_sq}")

                    x_new=np.linspace(-0.5,2).reshape((-1, 1))
                    y_new = model.predict(x_new)

                    ax[m].plot(x_new,y_new,color=colors[j-1])
                    ax[m].set_xlabel('Time(s)')
                    ax[m].set_ylabel('Activity(a.u.)')
                    ax[m].legend(legend)
                    x = sm.add_constant(x)
                    #fit linear regression model
                    model = sm.OLS(y, x).fit()            
                    #view model summary
                    print(model.summary())
            ax[0].set_title('Responsive Trials')
            ax[1].set_title('Unresponsive Trials')
            ax[2].set_title('Running Trials')
            f.suptitle(f'Mean Trial segmented Scatter Raw', fontsize=16)
            

            f,ax=plt.subplots(3)
            for m in range(3):
                for j in range(1,3):
                    activity=self.optotrialarraysmoothedDF[:,trislstruct[m],:]
                    for i in range(activity.shape[1]):
                        ax[m].scatter(activity[0,i,:],activity[j,i,:],color=colors[i],marker=shape[j-1])
                        ax[m].set_ylim(-1,2.5)                
                    ax[0].set_title('Responsive Trials')
                    ax[1].set_title('Unresponsive Trials')
                    ax[2].set_title('Running Trials')   
            f.suptitle(f'Trial segmented Scatter Substracted', fontsize=16)


            f,ax=plt.subplots(3)
            for m in range(3):
                for j in range(1,3):
                    activity=meantrislstructsubs[m]
                    ax[m].scatter(activity[0,:],activity[j,:],color=colors[j-1])
                    ax[m].set_ylim(-0.5,1)
                    ax[m].set_xlim(-0.5,2.1)
                    
                    x=activity[0,:].reshape((-1, 1))
                    y=activity[j,:]
                    model = LinearRegression().fit(x,y)
                    r_sq = model.score(x, y)
                    print(f"coefficient of determination: {r_sq}")

                    x_new=np.linspace(-0.5,2).reshape((-1, 1))
                    y_new = model.predict(x_new)
                    ax[m].plot(x_new,y_new,color=colors[j-1])
                    
                    ax[m].set_xlabel('Time(s)')
                    ax[m].set_ylabel('Activity(a.u.)')
                    ax[m].legend(legend) 
            ax[0].set_title('Responsive Trials')
            ax[1].set_title('Unresponsive Trials')
            ax[2].set_title('Running Trials')
            f.suptitle(f'Mean Trial segmented Scatter Substracted', fontsize=16)               
                            
        
    def trace_screening(self,array_ids):
        smoothwindows=10
        demixed_traces=self.full_data['imaging_data']['All_planes_rough']['Traces']['demixed']

        plt.close('all')
        for cell_id in array_ids:

            n_subplots=4
            f,ax=plt.subplots(n_subplots,sharex=True)
            ax[0].plot(demixed_traces[cell_id,:])
            ax[1].plot(self.smooth_trace(demixed_traces[cell_id,:],smoothwindows))
            ax[2].plot(self.accepteddff[cell_id,:])
            ax[3].plot(self.smooth_trace(self.accepteddff[cell_id,:],smoothwindows))
        
            for i in range(n_subplots): 
                ax[i].margins(x=0)
                ax[i].set_xlabel('Time(s)')
                if i<2:
                    ax[i].set_ylabel('Fluorescence(a.u.)')
                    ax[i].set_title('Demixed Fluorescence')
                else:
                    ax[i].set_title('Detrended DF/F')
                    ax[i].set_ylabel('DF/F(%)')
                    ax[i].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        
            f.suptitle('Fluorescence_Detrended-DF/F comparison', fontsize=16)
            
            plt.show()
        
       

    def photostim_stim_table_and_optanalysisrafatemp(self):
        # plt.close('all')
        ext=self.caiman_extractions[list(self.caiman_results.keys())[0]]
        cnmf=ext.cnm_object
        dff=cnmf.estimates.detrend_df_f()
        self.accepteddff=dff.F_dff[self.full_data['imaging_data']['Plane1']['CellIds'],:]
        
        locomotion=self.full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['Locomotion'].values
        deriv=lambda x:np.diff(x,prepend=x[0] )
        rectified=lambda x:np.absolute(x)
        speed=rectified(deriv(locomotion))
        speedtimestamps=self.full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['Locomotion'].index.values
        
        
        self.full_data['voltage_traces']['Full_signals']
        self.full_data['voltage_traces']['Full_signals_transitions']
        self.signals_object.all_final_signals['Prairie']['LED_clipped']['traces']['PhotoStim']
        up=self.full_data['voltage_traces']['Full_signals_transitions']['PhotoStim']['aligned_downsampled_LEDshifted']['Prairie']['up']
        down=self.full_data['voltage_traces']['Full_signals_transitions']['PhotoStim']['aligned_downsampled_LEDshifted']['Prairie']['down']
        zz=self.acquisition_object.visstimdict
        
        ref_image=Image.fromarray(self.acquisition_object.reference_image_dic[[i for i in self.acquisition_object.reference_image_dic.keys() if 'Red-8bit' in i][0]])
        total_opt_number=len(self.full_data['voltage_traces']['Full_signals_transitions']['PhotoStim']['aligned_downsampled_LEDshifted']['Prairie']['up'])

        trial_delay=self.acquisition_object.visstimdict['opto']['intertrialtime']
        nTrials=self.acquisition_object.metadata_object.mark_points_experiment['Iterations']
        cell_number=len(self.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries'])
        repetitions=int(self.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries']['PhotostimExperiment_1']['sequence']['Repetitions'])
        frequency=self.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries']['PhotostimExperiment_1']['sequence']['RepFrequency']
        opto_duration=self.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries']['PhotostimExperiment_1']['sequence']['StimDuration']
        iteration_duration=self.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries']['PhotostimExperiment_1']['sequence']['StimDuration']
        inter_rep_time=self.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries']['PhotostimExperiment_1']['sequence']['RepTime']
        inter_point_time=self.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries']['PhotostimExperiment_1']['sequence']['InterpointDuration']
        
        metadata_total_opt_number=repetitions*cell_number*nTrials
    
        signals_to_review=['PhotoStim','PhotoTrig']
        metadata_transitions=[metadata_total_opt_number,nTrials]
        for sig in zip(signals_to_review,metadata_transitions):
            transitions=len(self.full_data['voltage_traces']['Full_signals_transitions'][sig[0]]['aligned_downsampled_LEDshifted']['Prairie']['up'])
            if sig[1]==transitions:
                good='Correct'
            else:
                good='Incorrect'
            print(f'{good} number of {sig[0]} transitions detected')
    
        
        if 1/frequency==inter_rep_time:
            print('Seems correct opto timings)')

            
        transition_array=np.zeros([cell_number,nTrials,repetitions,2],dtype=int)

        for it in range(nTrials):

            for cell in range(cell_number):
                se=cell_number*it+cell
                transition_array[cell,it,:,0]=up[se*repetitions:(se+1)*repetitions]
                transition_array[cell,it,:,1]=down[se*repetitions:(se+1)*repetitions]


        res=self.caiman_results[list(self.caiman_results.keys())[0]]
        
        
        coordinates=[]
        for key, v in   self.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries'].items():
        
          coordinates.append((float(v['points']['Point_1']['x_pos'])*256,float(v['points']['Point_1']['y_pos'])*256,float(v['points']['Point_1']['spiral_width'])*256))
          
        res.data['est']['contours']
        A=res.data['est']['A']

        nA = np.sqrt(np.ravel(A.power(2).sum(axis=0)))
        nA_inv_mat = scipy.sparse.spdiags(1. / nA, 0, nA.shape[0], nA.shape[0])
        A = A * nA_inv_mat
        A=A.toarray().reshape([256,256,len(res.data['est']['contours'])])
       
    
        ttt=res.centers_of_mass
        tttt=np.vstack(ttt)
 
        
        optocellids=np.zeros(len(coordinates),dtype=int)
        for cell in range(len(coordinates)):
            dist=[]
            for i in range(len(tttt[:,0])):
                dist.append(np.linalg.norm(np.array((tttt[i,0], tttt[i,1]))- np.array((coordinates[cell][0], coordinates[cell][1]))))
            optocellids[cell]=np.argmin(dist)
        
        optocellids.sort()
               

        f,ax=plt.subplots(1,3)
        ax[0].imshow(A[:,:,self.full_data['imaging_data']['Plane1']['CellIds']].sum(axis=2).T)
        ax[1].imshow(A[:,:,optocellids].sum(axis=2).T)
        ax[2].imshow(ref_image.resize((256, 256)))
        for coord in coordinates:
            circles=[]
            for i in range(3):
                ax[i].add_patch(plt.Circle((coord[0], coord[1]), coord[2]/2, color='r', fill=False))


            
     
            
        self.array_ids=np.searchsorted(self.full_data['imaging_data']['Plane1']['CellIds'] , optocellids)
        non_opto_ids=np.delete(self.full_data['imaging_data']['Plane1']['CellIds'],  self.array_ids)
        self.non_opto_ids=np.searchsorted(self.full_data['imaging_data']['Plane1']['CellIds'] , non_opto_ids)
        
        self.optotraces=self.full_data['imaging_data']['Plane1']['Traces']['demixed'][self.array_ids,:]
        self.non_opto_traces=self.full_data['imaging_data']['Plane1']['Traces']['demixed'][self.non_opto_ids,:]
        self.full_traces=self.full_data['imaging_data']['Plane1']['Traces']['demixed']
        
        
        
        
        # self.optotraces=self.accepteddff[self.array_ids,:]

        fr=self.full_data['imaging_data']['Frame_rate']   
        
        # window_size = int(np.round(30000/28))

        # def dff(a, n=3) :
        #     dff=np.zeros_like(a)
        #     for frame in range(len(a)):
        #         if frame>n:
        #             ret = np.percentile(a[frame-n:frame],1)
        #             ret = np.mean(a[frame-n:frame])

        #         else:
        #             ret = np.percentile(a[:frame+1],1)
        #             ret = np.mean(a[:frame+1])

        #         dff[frame]=(a[frame]-ret)/ret
        #     return dff

        # tt=dff(self.optotraces[0,:],window_size)
        # f,ax=plt.subplots(2,sharex=True)
        # ax[0].plot(self.smooth_trace(tt,10),'k')
        # ax[1].plot(self.smooth_trace(self.optotraces[0,:],10),'b')

       

       

         
        pretime=1#s
        posttime=3#S
        prestim=int(pretime*fr)
        poststim=int(posttime*fr)        

        self.trialtimevector=np.linspace(-pretime,posttime,prestim+poststim)
        
        prestimspeed=int(pretime*1000)
        poststimspeed=int(posttime*1000)  
        
        self.trialspeedtimevector=np.linspace(-pretime,posttime,prestimspeed+poststimspeed)
        interoptoframes=mode(np.diff(transition_array[0,0,:,0]))[0][0]

        print('Creating stim arrays')

        self.optotrialarray=np.zeros((cell_number,cell_number,nTrials,prestim+poststim))
        for cell in range(cell_number):
            for opto_idx in range(cell_number):
                for i, opto  in enumerate(transition_array[opto_idx,:,0,0]):
                    self.optotrialarray[cell,opto_idx,i,:]=self.optotraces[cell,opto-prestim:opto+poststim]

        self.fulloptotrialarray=np.zeros((self.full_traces.shape[0],cell_number,nTrials,prestim+poststim))
        for cell in range(self.full_traces.shape[0]):
            for opto_idx in range(cell_number):
                for i, opto  in enumerate(transition_array[opto_idx,:,0,0]):
                    self.fulloptotrialarray[cell,opto_idx,i,:]=self.full_traces[cell,opto-prestim:opto+poststim]

                    
        print('Scaling and smoothing speed')

        self.speedtrialrarray=np.zeros((cell_number,nTrials,prestimspeed+poststimspeed))
        for opto_idx in range(cell_number):
            for i, opto  in enumerate(transition_array[opto_idx,:,0,0]):
                voltopto= np.round(self.mov_timestamps_miliseconds['shifted'][opto]).astype(int)
                self.speedtrialrarray[opto_idx,i,:]=self.scale_signal(speed)[voltopto-prestimspeed:voltopto+poststimspeed]

        print('Substracting baseline')

        prestimsubstracted=np.zeros_like(self.optotrialarray)
        for cell in range(cell_number):
            for opto_idx in range(cell_number):
                for trial  in range(nTrials):
                    prestimsubstracted[cell,opto_idx,trial,:]=(self.optotrialarray[cell,opto_idx,trial,:]- self.optotrialarray[cell,opto_idx,trial,:prestim].mean())/self.optotrialarray[cell,opto_idx,trial,:prestim].mean()

        meanoptotracessubstracted=prestimsubstracted.mean(axis=2)
        meanoptotraces=self.optotrialarray.mean(axis=2)


        #smooth sigblas for plotting after processing
        print('Smoothing signals')
        smoothwindows=10
        smoothedoptotraces=np.zeros_like(self.optotrialarray)
        smoothedoptotracessubstracetd=np.zeros_like(prestimsubstracted)
        smoothedmeanoptotraces=np.zeros_like(meanoptotraces) 
        smoothedmeanoptotracessubstracted=np.zeros_like(meanoptotracessubstracted) 

        for cell in range(cell_number):
            for opto_idx in range(cell_number):
                smoothedmeanoptotraces[cell,opto_idx,:]=self.smooth_trace(meanoptotraces[cell,opto_idx,:],smoothwindows)
                smoothedmeanoptotracessubstracted[cell,opto_idx,:]=self.smooth_trace(meanoptotracessubstracted[cell,opto_idx,:],smoothwindows)
                for trial  in range(nTrials):
                    smoothedoptotraces[cell,opto_idx,trial,:]=self.smooth_trace(self.optotrialarray[cell,opto_idx,trial,:],smoothwindows)
                    smoothedoptotracessubstracetd[cell,opto_idx,trial,:]=self.smooth_trace(prestimsubstracted[cell,opto_idx,trial,:],smoothwindows)


        print('plotting')
        #plot all chandelier traces and first opt
        # plt.close('all')
        fig,ax=plt.subplots(self.optotraces.shape[0],sharex=True)
        fig.tight_layout()
        for i in range(self.optotraces.shape[0]):
            trace=self.optotraces[i,:]
            ax[i].plot(self.mov_timestamps_miliseconds['shifted'],self.smooth_trace(trace,smoothwindows),'k')
            ax[i].plot(speedtimestamps,self.scale_signal(speed),'r',alpha=0.5)
            ax[i].margins(x=0)
            for j in range(len(transition_array[i,:,0,0])):
                ax[i].axvline(x=self.mov_timestamps_miliseconds['shifted'][transition_array[i,j,0,0]])  
            ax[i].set_xlabel('Time(s)')
            ax[i].set_ylabel('Activity(a.u.)')
            fig.suptitle('Optogenetic Stimulation of Chandelier Cells', fontsize=16)

        #lpot clean traces of all opt and some non oppto cells
        
       
        plt.rcParams["figure.figsize"] = [16, 5]
        plt.rcParams["figure.autolayout"] = True
        non_opto_toplot=10
        totalcells=self.optotraces.shape[0]+non_opto_toplot
        f,ax=plt.subplots(totalcells+1,sharex=True)    
        for i in range(self.optotraces.shape[0]):
            trace=self.optotraces[i,:]
            ax[i].plot(self.mov_timestamps_seconds['shifted'],self.smooth_trace(trace,10),c='y')
            for j in range(len(transition_array[i,:,0,0])):
                ax[i].axvline(x=self.mov_timestamps_seconds['shifted'][transition_array[i,j,0,0]])  
            
        for i in range(non_opto_toplot):
            # trace=self.full_data['imaging_data']['Plane1']['Traces']['demixed'][self.non_opto_ids[i],:]
            trace=  self.non_opto_traces[self.non_opto_ids[i],:]

            ax[i+self.optotraces.shape[0]].plot(self.mov_timestamps_seconds['shifted'],self.smooth_trace(trace,10),c='g')
            
        ax[-1].plot(speedtimestamps/1000,speed,'r')
        for i,a in enumerate(ax):
            a.margins(x=0)
            if i<len(ax)-1:
                a.axis('off')
            elif i==len(ax)-1:
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
                a.spines['left'].set_visible(False)
                a.get_yaxis().set_ticks([])
                a.set_xlabel('Time(s)')
                
                
        #plot single optotirals individually 
        for l in range(cell_number):
            f,ax=plt.subplots(5,2)
            for i,opto  in enumerate(transition_array[l,:,0,0]):
                row = i // 2  # determine the row index based on the iteration index
                col = i % 2   # determine the column index based on the iteration index
                # ax[row, col].plot(self.trialtimevector,self.scale_signal(smoothedoptotraces[l,l,i,:]),'k')
                ax[row, col].plot(self.trialtimevector,smoothedoptotraces[l,l,i,:]),'k'

                # ax[row, col].plot(self.trialtimevector,smoothedoptotracessubstracetd[l,l,i,:],'b')
                ax[row, col].plot(self.trialspeedtimevector,self.speedtrialrarray[l,i,:],'r')
                ax[row, col].axvline(x=0)  
                # ax[row, col].set_ylim(-3,8)
                ax[row, col].margins(x=0)
                for m in range(repetitions):               
                    ax[row, col].add_patch(Rectangle((self.trialtimevector[prestim+interoptoframes*m], 0.8), 0.01, 0.2,color='r'))
    
                ax[row, col].set_xlabel('Time(s)')
                ax[row, col].set_ylabel('Activity(a.u.)')
                
            f.suptitle(f'Single Trial Optogenetic Stimulation Cell{str(l+1)}', fontsize=16)
                  
           
        
        #plot opto cells in tiled array with mean of all traces and then a individual figure for single trials
        
    
        f,ax=plt.subplots(cell_number,cell_number,sharex=True, sharey=True)
        for stim_cell_trials in range(cell_number):
            for cell_trace in range(cell_number):
                if stim_cell_trials==cell_trace:
                    color='r'
                else:
                    color='b'
    
                ax[stim_cell_trials, cell_trace].plot(self.trialtimevector,smoothedmeanoptotraces[cell_trace,stim_cell_trials,:],color)
                ax[stim_cell_trials, cell_trace].axvline(x=0)  
                ax[stim_cell_trials, cell_trace].axis('off')
  
        # plt.close('all')
        for trial in range(nTrials):
            f,ax=plt.subplots(cell_number,cell_number,sharex=True, sharey=True)
            for stim_cell_trials in range(cell_number):
                for cell_trace in range(cell_number):
                    if stim_cell_trials==cell_trace:
                        color='r'
                    else:
                        color='b'
           
                    ax[stim_cell_trials, cell_trace].plot(self.trialtimevector,smoothedoptotraces[cell_trace,stim_cell_trials,trial,:],color)
                    ax[stim_cell_trials, cell_trace].axvline(x=0)  
                    ax[stim_cell_trials, cell_trace].axis('off')
        



        #selecting traces by locomotion, active or inactive   based on single trial plootting
        
        
               # meanactivations=self.optotrialarraysmoothed[:,[0,4,6,9],:].mean(axis=1)
               # meanalocomotion=self.optotrialarraysmoothed[:,7,:]
               # meannonactivations=self.optotrialarraysmoothed[:,[1,2,3,5,8],:].mean(axis=1)
                
                
        plt.show()

    def smooth_trace(self, trace,window):
        framenumber=len(trace)
        frac=window/framenumber
        filtered = lowess(trace, np.arange(framenumber), frac=frac)
        
        return filtered[:,1]  
            
    def create_stim_table(self):
        
        if self.signals_object.vis_stim_protocol=='AllenA':
            self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table']=self.drifting_grating_stim_table()
        elif self.signals_object.vis_stim_protocol=='AllenB':
            self.full_data['visstim_info']['Static_Gratings']['stimulus_table']=self.static_gratings_stim_table()
            self.full_data['visstim_info']['Natural_Images']['stimulus_table']=self.natural_images_stim_table()
        elif self.signals_object.vis_stim_protocol=='AllenC':
            pass
                   
    
    def movie_one_stim_table(self):   
        
        
        frame1starts=self.full_data['visstim_info']['Movie1']['Frame_Starts'][:,0].astype('uint32')
        frame1ends=self.full_data['visstim_info']['Movie1']['Frame_Ends'][:,1].astype('uint32')
        
        if (not self.nondatabase) and self.signals_object.vis_stim_protocol:
            self.stim_time=2000  #ms
            self.pre_time=350    #ms
            self.post_time=350   #ms
                   
            
            self.pre_frames=np.ceil(self.pre_time/self.milisecond_period).astype(int)
            self.post_frames=np.ceil(self.post_time/self.milisecond_period).astype(int)
            self.stim_frames=np.ceil(self.stim_time/self.milisecond_period).astype(int)
            self.grating_number=len(self.full_data['visstim_info']['Natural_Images']['Indexes'])
            corrected_indexes=self.full_data['visstim_info']['Movie1']['Frame_Starts'].astype('uint32')
            
            df=pd.DataFrame({'Frame_ID':np.zeros(self.full_data['visstim_info']['Movie1']['Frame_Starts'].size),
                             'Trial_ID':np.zeros(self.full_data['visstim_info']['Movie1']['Frame_Starts'].size),
                         'start':np.zeros(self.full_data['visstim_info']['Movie1']['Frame_Starts'].size),
                         'end':np.zeros(self.full_data['visstim_info']['Movie1']['Frame_Starts'].size)} )
            
            
            
                         
            for i in np.arange(self.full_data['visstim_info']['Movie1']['Frame_Starts'].shape[0]):
                framenumber=np.arange(1,self.full_data['visstim_info']['Movie1']['Frame_Starts'][i,:].shape[0]+1)
                framestarts=self.full_data['visstim_info']['Movie1']['Frame_Starts'][i,:].astype('uint32')
                frameends=self.full_data['visstim_info']['Movie1']['Frame_Ends'][i,:].astype('uint32')
                
                df.iloc[900*i:900*(i+1)]=np.vstack((framenumber,np.zeros(self.full_data['visstim_info']['Movie1']['Frame_Starts'][i,:].shape[0]).astype('uint8')+i+1,framestarts,frameends)).T
   
            df = df.astype('int')

          
            
            return df
        
        
        
        
    def movie_two_stim_table(self):
        pass
    def movie_three_stim_table(self):
        pass
    
    def static_gratings_stim_table(self):
        if (not self.nondatabase) and self.signals_object.vis_stim_protocol:
            self.stim_time=250    #ms
            self.pre_time=350     #ms
            self.post_time=350      #ms
                   
            
            self.pre_frames=np.ceil(self.pre_time/self.milisecond_period).astype(int)
            self.post_frames=np.ceil(self.post_time/self.milisecond_period).astype(int)
            self.stim_frames=np.ceil(self.stim_time/self.milisecond_period).astype(int)
            self.grating_number=len(self.full_data['visstim_info']['Static_Gratings']['Indexes'])
            corrected_indexes=self.full_data['visstim_info']['Static_Gratings']['Indexes']
            
            
            df=pd.DataFrame({'Grating_ID':np.zeros(len(corrected_indexes[0])),
                         'start':corrected_indexes[0],
                         'end':corrected_indexes[0]+self.stim_frames           
                        })
            for i in range(1,len(corrected_indexes)):
                df2=pd.DataFrame({'Grating_ID':np.zeros(len(corrected_indexes[i]))+i,
                             'start':corrected_indexes[i],
                             'end':corrected_indexes[i]+self.stim_frames 
                            })
                
                df=df.append(df2, ignore_index=True)            
            
            
          
            sorted_df=df.sort_values(by=['start'])
            
            return sorted_df.reset_index(drop=True)
                      
    def natural_images_stim_table(self):
            
       if (not self.nondatabase) and self.signals_object.vis_stim_protocol:
           self.stim_time=2000    #ms
           self.pre_time=350     #ms
           self.post_time=350      #ms
                  
           
           self.pre_frames=np.ceil(self.pre_time/self.milisecond_period).astype(int)
           self.post_frames=np.ceil(self.post_time/self.milisecond_period).astype(int)
           self.stim_frames=np.ceil(self.stim_time/self.milisecond_period).astype(int)
           self.grating_number=len(self.full_data['visstim_info']['Natural_Images']['Indexes'])
           corrected_indexes=self.full_data['visstim_info']['Natural_Images']['Indexes']
           
           
           df=pd.DataFrame({'Image_ID':np.zeros(len(corrected_indexes[0])),
                        'start':corrected_indexes[0],
                        'end':corrected_indexes[0]+self.stim_frames           
                       })
           for i in range(1,len(corrected_indexes)):
               df2=pd.DataFrame({'Image_ID':np.zeros(len(corrected_indexes[i]))+i,
                            'start':corrected_indexes[i],
                            'end':corrected_indexes[i]+self.stim_frames 
                           })
               
               df=df.append(df2, ignore_index=True)            
           
           
         
           sorted_df=df.sort_values(by=['start'])
           
           return sorted_df.reset_index(drop=True)
                
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

    #%% PLOTTING MOVIE 1
    
   
        
        
        
        
        
        

    #%% PLOTTING GENERAL
    def general_raster_plotting(self, title):
        #method to define defaul options for a raster plot
        
        pixel_per_bar = 4
        dpi = 100
        fig, ax = plt.subplots(1,  figsize=(16,9), dpi=dpi, sharex=True) 
        ax.set_xlabel('Time(s)')
        fig.supylabel('Cell Number')
        fig.suptitle(title)
        fig.set_tight_layout(True)
        
        return fig, ax
    
    def plot_sliced_raster(self,trace_type,plane,selected_cells,paradigm='Full', locomotion=False):
        activity_arrays= self.get_raster_with_selections(trace_type,plane,selected_cells, paradigm)
        
        
        if not locomotion:
            fig, ax=self.general_raster_plotting(paradigm)
              
            centers = [activity_arrays[3].shape[0]+1, 1]
            dy, = -np.diff(centers)/(activity_arrays[3].shape[0]-1)
            extent=[activity_arrays[4][0]-activity_arrays[4][0],
                    activity_arrays[4][-1]-activity_arrays[4][0],
                    centers[0]+dy/2,
                    centers[1]-dy/2]
            
            ax.imshow(activity_arrays[3], cmap='binary', aspect='auto',
                interpolation='nearest', norm=mpl.colors.Normalize(0, 1),extent=extent)
        else:
            
            pixel_per_bar = 4
            dpi = 100
            fig, ax = plt.subplots(2,  figsize=(16,9), dpi=dpi, sharex=True) 
            ax[1].set_xlabel('Time(s)')
            fig.supylabel('Cell Number')
            fig.suptitle(title)
            fig.set_tight_layout(True)
            
            
        
        # ax[1].title.set_text('Binarized_Smoothed_thresholded_dfdt_{}_{}_{}'.format(0, self.dfdt_sigma, self.dfdt_std_threshold))
      
        
        # filename = os.path.join( os.path.split(self.data_paths['Movie1'])[0],f'allcells_Raster_scored_mcmc.pdf')
        # self.save_multi_image(filename)
        # plt.close('all')
        
        
    def plot_all_planes_by_cell_type(self):
        plt.close('all')

    
        dpi=100
        pyramidals=[   val['pyr'][1]    for plane,val in self.pyr_int_ids_and_indexes.items()  if 'rough' not in plane]
        allplanepyramidalindex=np.hstack(pyramidals)
        pyramidals=[   val['int'][1]    for plane,val in self.pyr_int_ids_and_indexes.items() if 'rough' not in plane]
        allplaneinterneuronindex=np.hstack(pyramidals)
        
        
        #all scored mcmc binary
        
        
        fig, ax = plt.subplots(1,  figsize=(16,9), dpi=dpi, sharex=True)    
        ax.imshow( self.full_data['imaging_data']['All_planes_rough']['Traces']['mcmc_scored_binary'][:,:], cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        # ax[1].title.set_text('Binarized_Smoothed_thresholded_dfdt_{}_{}_{}'.format(0, self.dfdt_sigma, self.dfdt_std_threshold))
        ax.set_xlabel('Time(s)')
        fig.supylabel('Cell Number')
        fig.suptitle('Raster_scored_mcmc')
        
        filename = os.path.join( os.path.split(self.data_paths['pca_runs_path'])[0],f'allcells_Raster_scored_mcmc.pdf')
        self.save_multi_image(filename)
        plt.close('all')



        #pyramidal dfdt
        
        
        fig, ax = plt.subplots(2,  figsize=(16,9), dpi=dpi, sharex=True)
        ax[0].imshow( self.full_data['imaging_data']['All_planes_rough']['Traces']['dfdt_smoothed'][allplanepyramidalindex.squeeze(),:], cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        # ax[0].title.set_text('Smoothed_thresholded_dfdt_{}_{}'.format(self.dfdt_sigma, self.dfdt_std_threshold))
        ax[1].imshow( self.full_data['imaging_data']['All_planes_rough']['Traces']['dfdt_binary'][allplanepyramidalindex.squeeze(),:], cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        # ax[1].title.set_text('Binarized_Smoothed_thresholded_dfdt_{}_{}_{}'.format(0, self.dfdt_sigma, self.dfdt_std_threshold))
        ax[1].set_xlabel('Time(s)')
        fig.supylabel('Cell Number')
        fig.suptitle('Raster_df/dt')

        
         
        #pyramidal mcmcm
        
        fig,ax = plt.subplots(3,  figsize=(16,9), dpi=dpi, sharex=True)
        ax[0].imshow( self.full_data['imaging_data']['All_planes_rough']['Traces']['mcmc_raw'][allplanepyramidalindex.squeeze(),:], cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        ax[0].title.set_text('Raw_MCMC')
        ax[1].imshow(  self.full_data['imaging_data']['All_planes_rough']['Traces']['mcmc_smoothed'][allplanepyramidalindex.squeeze(),:], cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        # ax[1].title.set_text('Smoothed_MCMC_{}'.format(self.MCMC_sigma))
        ax[2].imshow(  self.full_data['imaging_data']['All_planes_rough']['Traces']['mcmc_binary'][allplanepyramidalindex.squeeze(),:], cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        # ax[2].title.set_text('Binarized_smoothed_MCMC_{}_{}'.format(0, self.MCMC_sigma ))
        ax[-1].set_xlabel('Time(s)')
        fig.supylabel('Cell Number')
        fig.suptitle('Raster_MCMC')
        

        #interneuron dfdt
       
         
        fig, ax = plt.subplots(2,  figsize=(16,9), dpi=dpi, sharex=True)
        ax[0].imshow( self.full_data['imaging_data']['All_planes_rough']['Traces']['dfdt_smoothed'][allplaneinterneuronindex.squeeze(),:], cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        # ax[0].title.set_text('Smoothed_thresholded_dfdt_{}_{}'.format(self.dfdt_sigma, self.dfdt_std_threshold))
        ax[1].imshow( self.full_data['imaging_data']['All_planes_rough']['Traces']['dfdt_binary'][allplaneinterneuronindex.squeeze(),:], cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        # ax[1].title.set_text('Binarized_Smoothed_thresholded_dfdt_{}_{}_{}'.format(0, self.dfdt_sigma, self.dfdt_std_threshold))
        ax[1].set_xlabel('Time(s)')
        fig.supylabel('Cell Number')
        fig.suptitle('Raster_df/dt')
    


       
       #interneuron mcmcm
        fig,ax = plt.subplots(3,  figsize=(16,9), dpi=dpi, sharex=True)
        ax[0].imshow( self.full_data['imaging_data']['All_planes_rough']['Traces']['mcmc_raw'][allplaneinterneuronindex.squeeze(),:], cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        ax[0].title.set_text('Raw_MCMC')
        ax[1].imshow(  self.full_data['imaging_data']['All_planes_rough']['Traces']['mcmc_smoothed'][allplaneinterneuronindex.squeeze(),:], cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        # ax[1].title.set_text('Smoothed_MCMC_{}'.format(self.MCMC_sigma))
        ax[2].imshow(  self.full_data['imaging_data']['All_planes_rough']['Traces']['mcmc_binary'][allplaneinterneuronindex.squeeze(),:], cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        # ax[2].title.set_text('Binarized_smoothed_MCMC_{}_{}'.format(0, self.MCMC_sigma ))
        ax[-1].set_xlabel('Time(s)')
        fig.supylabel('Cell Number')
        fig.suptitle('Raster_MCMC')
               
    def plot_full_signals_and_raster(self):
        
        pass

    def do_some_plotting(self, cell, trace_type, plane, matlabcell=None):
        
        # plt.close('all')
        
        # plane='Plane1'
        # cell=4
        # trace_type='dfdt_smoothed'
        # trace_type='denoised'
        # trace_type='mcmc_smoothed'
         
        # self.preframes=25
        # self.stim=50
        # self.postframes=25
        # ,matlabcell=4 21 75# inhibited by stimulus
        #cell 3 13
        # matlabcell=45# bimodal
        # matlabcell=47 49# very sharp dip at begining of stim
        # if matlabcell:
        #     cell=np.argwhere(self.full_data['imaging_data'][plane]['CellIds']==matlabcell)[0][0]  
               
        # matlabcell=self.full_data['imaging_data'][plane]['CellIds'][cell]  
        if self.pyr_int_ids_and_indexes:
            pyr=np.argwhere(self.pyr_int_ids_and_indexes[plane]['pyr'][1]).flatten()
            inter=np.argwhere(self.pyr_int_ids_and_indexes[plane]['int'][1]).flatten()
            if  cell in    pyr:
                celltype='Pyramidal Cell'
            elif  cell in    inter:
                celltype='Interneuron'
                
        celltype='Interneuron'
        # print(plane+'\nMatlab cell: '+str( matlabcell)+'\nPython cell :'+str(cell)+'\n' + celltype)        
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
    #%% PLOTTING DRIFTING GRATINGS
        ax[3].plot(meantraces,'k')
                       
    def plot_directions(self, cell, trace_type, plane):
        
        for ori in np.linspace(0,360-45,8):
            fig,ax=plt.subplots(4)         
            C_mat=self.full_data['imaging_data'][plane]['Traces'][trace_type]
            cell_trace=C_mat[cell,:]
            ax[0].plot(cell_trace)       
            trial=np.vstack([cell_trace[int(row.start-self.preframes):int(row.end+self.postframes)]
             if row.end-row.start+self.preframes+self.postframes==self.preframes+self.stim+self.postframes
             else  cell_trace[int(row.start-self.preframes):int(row.end+self.postframes+1)] for i, row in self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'][self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'].orientation==ori].iterrows() ])
            meantraces=np.mean(trial, axis=0)               
            for i, row in self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'][(self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'].orientation==ori)&(self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'].blank_sweep==0)].iterrows():
                ax[1].plot(cell_trace[int(row.start)-self.preframes:int(row.end)+self.postframes], 'k', alpha=0.1)
            ax[1].axvspan(0,self.preframes, facecolor='g', alpha=0.3)
            ax[1].axvspan(self.preframes,self.preframes+self.stim, facecolor='b', alpha=0.3)
            ax[1].plot(meantraces,'k')
            ax[2].imshow(trial,cmap='binary', aspect='auto')
            ax[3].plot(meantraces,'k')
            plt.show()
            
    def plot_orientation(self, cell, trace_type, plane, plot=None):
        
        pyr=np.argwhere(self.pyr_int_ids_and_indexes[plane]['pyr'][1]).flatten()
        inter=np.argwhere(self.pyr_int_ids_and_indexes[plane]['int'][1]).flatten()
        if  cell in    pyr:
            celltype='Pyramidal Cell'
        elif  cell in    inter:
            celltype='Interneuron'
        
        mean_evoked_2s=[]
        mean_evoked_1s=[]
        for n, ori in enumerate(np.linspace(0,180-(180/4),4)):
            # trace_type='denoised'
            C_mat=self.full_data['imaging_data'][plane]['Traces'][trace_type]
            cell_trace=C_mat[cell,:]
            fluorescence=self.full_data['imaging_data'][plane]['Traces']['denoised'][cell,:]              
            trial=np.vstack([cell_trace[int(row.start-self.preframes):int(row.end+self.postframes)]
             if row.end-row.start+self.preframes+self.postframes==self.preframes+self.stim+self.postframes
             else  cell_trace[int(row.start-self.preframes):int(row.end+self.postframes+1)] 
             for i, row in self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'][self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'].orientation.isin([ori, ori+180])].iterrows() ])
            meantraces=np.mean(trial, axis=0)            
            mean_evoked_2s.append(np.mean(meantraces[self.preframes:self.preframes+self.stim]))
            mean_evoked_1s.append(np.mean(meantraces[self.preframes:int(np.floor((self.preframes+self.stim)/2))]))
            
        mean_evoked_2s=[]
        mean_evoked_1s=[]
        if plot: 
          
            fig = plt.figure(constrained_layout=True)
            subfigs = fig.subfigures(2, 3, wspace=0.07, width_ratios=[1, 1,1])
            fig.suptitle('Cell: '+str(cell)+' '+celltype )
          
            axs=['','','','']
            for n, ori in enumerate(np.linspace(0,180-(180/4),4)):
                axs[n]=subfigs[int(np.floor(n/2)),int(n%2)].subplots(3)
                C_mat=self.full_data['imaging_data'][plane]['Traces'][trace_type]
                cell_trace=C_mat[cell,:]
                fluorescence=self.full_data['imaging_data'][plane]['Traces']['denoised'][cell,:]
                
                trial=np.vstack([cell_trace[int(row.start-self.preframes):int(row.end+self.postframes)]
                 if row.end-row.start+self.preframes+self.postframes==self.preframes+self.stim+self.postframes
                 else  cell_trace[int(row.start-self.preframes):int(row.end+self.postframes+1)] 
                 for i, row in self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'][self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'].orientation.isin([ori, ori+180])].iterrows() ])   
                meantraces=np.mean(trial, axis=0)
                
                mean_evoked_2s.append(np.mean(meantraces[self.preframes:self.preframes+self.stim]))
                mean_evoked_1s.append(np.mean(meantraces[self.preframes:int(np.floor((self.preframes+self.stim)/2))]))
                   
                for i, row in self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'][(self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'].orientation==ori)&(self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'].blank_sweep==0)].iterrows():
                    axs[n][0].plot(cell_trace[int(row.start)-self.preframes:int(row.end)+self.postframes], 'k', alpha=0.1)
                axs[n][0].set_title(ori)
                    
                axs[n][0].axvspan(0,self.preframes, facecolor='g', alpha=0.3)
                axs[n][0].axvspan(self.preframes,self.preframes+self.stim, facecolor='b', alpha=0.3)
               
                axs[n][0].plot(meantraces,'k')
                axs[n][1].imshow(trial,cmap='binary', aspect='auto')
                axs[n][2].plot(meantraces,'k')
        
            ax1=subfigs[0,2].subplots(1)    
            ax1.plot(fluorescence)
            ax2=subfigs[1,2].subplots(1)    
            ax2.plot([0,45,90,135],mean_evoked_2s, 'r', label='Full Stim')
            ax2.plot([0,45,90,135],mean_evoked_1s, 'b', label='First Half Stim')
            ax2.legend()    
            ax2.set_title('Mean Evoked Activity')   
            ax2.set_ylim(0,2*np.max(mean_evoked_2s))
            ax2.set_ylim(0,0.2)    
            plt.show()
        
        return mean_evoked_2s, mean_evoked_1s

#%% INDEXING AND IDENTITI

    #this requires to manually copy the template stroed in desktop TemplateCellIdentity and marking at least wich cells are toimato+ and accpeted tomato+ the  use the method to read identity form ezcel
    # self.process_excel_pyr_int_identif_file()

    def convert_full_planes_idx_to_single_plane_final_indx(self,full_raster_pyhton_cell_idx, plane):
        
        # THIS IS TO CONVERT THE INDEXES FROM THE RASTER TO ACTUAL INDEXED FROM THE CAIMAN SORTER. IT DOESN WORK WITH CAIMAN SORTER INDEXES so if n=10 cells these indexex  have to be less than N
        caiman_sorter_idx=self.full_data['imaging_data'][plane]['CellIds']

        try:
            if 'All' in plane:
                total_cells=sum(len(plane) for plane in caiman_sorter_idx.values())
                first_idx_plane=np.concatenate((np.zeros((1)),np.cumsum([len(plane) for plane in caiman_sorter_idx.values()])[:-1])).astype('uint16')
                plane_asignation=[(full_raster_pyhton_cell_idx>=i,full_raster_pyhton_cell_idx<i) for i in first_idx_plane]
                plane_idx = len(plane_asignation) - next(i for i, val in enumerate(reversed(plane_asignation), 1) if val == (True, False)) 
                indexed_plane_sorter_idx=caiman_sorter_idx[f'Plane{plane_idx+1}']
                     
                single_plane_selected_pyhton_cell_idx=full_raster_pyhton_cell_idx-first_idx_plane[plane_idx]
                single_plane_sorter_pyhton_idx=indexed_plane_sorter_idx[single_plane_selected_pyhton_cell_idx]
                matlab_sorter_cell_id=single_plane_sorter_pyhton_idx+1
            else:

                total_cells=len(caiman_sorter_idx)
                plane_idx= int(plane[-1])-1
                indexed_plane_sorter_idx=caiman_sorter_idx
                # %this is for allen plane 1 only HAVETO DO FRO PLANE ! ON MY DATA
                single_plane_selected_pyhton_cell_idx=full_raster_pyhton_cell_idx
                single_plane_sorter_pyhton_idx=full_raster_pyhton_cell_idx
                matlab_sorter_cell_id=None

           
        
            return  f'Plane{plane_idx+1}', total_cells, full_raster_pyhton_cell_idx, single_plane_selected_pyhton_cell_idx, single_plane_sorter_pyhton_idx, matlab_sorter_cell_id
                 
        except:
            print('CHeck your indexes and planes')
            return None,None,None,None,None,None
                       
    def indetify_full_rater_idx_cell_identity(self, raster_pyhton_cell_idx, plane):
        
        #first check that the index is lower tha the toal number of cell in the given plane, then check if the index is in pyramidal or interneruon list
        if raster_pyhton_cell_idx<len(np.concatenate([self.pyr_int_ids_and_indexes[plane]['pyr'][0],self.pyr_int_ids_and_indexes[plane]['int'][0]])):
            if self.pyr_int_ids_and_indexes[plane]['pyr'][1][raster_pyhton_cell_idx]:
                cell_identity='Tomato -'
            elif  self.pyr_int_ids_and_indexes[plane]['int'][1][raster_pyhton_cell_idx]:
                cell_identity='Tomato +'
            else:
                print('Checking')
            return cell_identity
        else:
            print('CHeck your indexes and planes')
            return None
        
    def define_pre_post_frames(self, dur_frames, isi_frames):
        duration_dict={
            'preframes':isi_frames,
            'stim':dur_frames,
            'postframes':isi_frames}
        
        return   duration_dict
    
    
    def get_trial_from_drifting_angle(self, direction_list, temporal_frequency_list):
        
        selected_stim_table=self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'][(self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'].orientation.isin(direction_list)) & 
                                                                         (self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'].temporal_frequency.isin(temporal_frequency_list))]

        
        return selected_stim_table
    

    def extract_trial_indexing_array(self,selected_stim_table, duration_dict):
        
        isi_start=selected_stim_table['start'].values-duration_dict['preframes']
        post_isis_end=selected_stim_table['end'].values+duration_dict['postframes']
        
        trial_indexig_ranges=[]
        for i in range(len(isi_start)):
            trial_indexig_ranges.append(np.arange(isi_start[i],post_isis_end[i]))
            
        return    trial_indexig_ranges   
        


           
    def get_paradigm_range(self, paradigm):    

        if paradigm=='Full':
            paradimg_range=np.arange(0,self.full_data['imaging_data']['Frame_number'])
            
            
        elif paradigm=='Drifting_Gratings':
            
            indexes=(np.arange(self.full_data['visstim_info']['Paradigm_Indexes']['first_drifting_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['first_drifting_set_last']),
            np.arange(self.full_data['visstim_info']['Paradigm_Indexes']['second_drifting_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['second_drifting_set_last']),
            np.arange(self.full_data['visstim_info']['Paradigm_Indexes']['third_drifting_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['third_drifting_set_last'])) 
            
            paradimg_range=paradimg_range=np.r_[indexes]
            
        elif paradigm=='Spontaneous':
            paradigm_start=self.full_data['visstim_info']['Spontaneous']['stimulus_table'].iloc[[0]]['start'].values[0]
            paradigm_end=self.full_data['visstim_info']['Spontaneous']['stimulus_table'].iloc[[-1]]['end'].values[0]
            paradimg_range=np.arange(paradigm_start,paradigm_end+1)
            
        elif paradigm=='Movie1':
            
            paradimg_range=np.arange(self.full_data['visstim_info']['Paradigm_Indexes']['natural_movie_one_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['natural_movie_one_set_last'])
            
        elif paradigm=='Movie3':
            
            indexes=(np.arange(self.full_data['visstim_info']['Paradigm_Indexes']['natural_movie_three_first_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['natural_movie_three_first_set_last']),                       
             np.arange(self.full_data['visstim_info']['Paradigm_Indexes']['natural_movie_three_second_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['natural_movie_three_second_set_last']))
            paradimg_range=paradimg_range=np.r_[indexes]
            
        elif paradigm=='Movie2':
            
            paradimg_range=np.arange(self.full_data['visstim_info']['Paradigm_Indexes']['natural_movie_two_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['natural_movie_two_set_last'])
            
        elif paradigm=='Natural_Images':
            
            indexes=(np.arange(self.full_data['visstim_info']['Paradigm_Indexes']['first_images_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['first_images_set_last']),
            np.arange(self.full_data['visstim_info']['Paradigm_Indexes']['second_images_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['second_images_set_last']),
            np.arange(self.full_data['visstim_info']['Paradigm_Indexes']['third_images_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['third_images_set_last'])) 
            
            paradimg_range=paradimg_range=np.r_[indexes]
            
            
            
        elif paradigm=='Static_Gratings':
            
            indexes=(np.arange(self.full_data['visstim_info']['Paradigm_Indexes']['first_static_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['first_static_set_last']),
            np.arange(self.full_data['visstim_info']['Paradigm_Indexes']['second_static_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['second_static_set_last']),
            np.arange(self.full_data['visstim_info']['Paradigm_Indexes']['third_static_set_first'],
            self.full_data['visstim_info']['Paradigm_Indexes']['third_static_set_last'])) 
            
            paradimg_range=paradimg_range=np.r_[indexes]
        
        elif paradigm=='Sparse_Noise':
            
            pass
        return paradimg_range
    
    
       
    def create_cell_selection_ranges(self,selected_cells, plane) :

        if isinstance(selected_cells, list):
            
            cell_indexes=np.r_[selected_cells]
        elif self.pyr_int_ids_and_indexes:
        
            if selected_cells=='Pyramidal':
                cell_indexes= np.where(self.pyr_int_ids_and_indexes[plane]['pyr'][1].flatten())[0]
            elif selected_cells=='Interneurons':
                cell_indexes= np.where(self.pyr_int_ids_and_indexes[plane]['int'][1].flatten())[0]
            elif selected_cells=='All':
                cell_indexes=np.arange(len(self.pyr_int_ids_and_indexes[plane]['pyr'][1].flatten())+len(self.pyr_int_ids_and_indexes[plane]['int'][1].flatten()))
        else:
            cell_indexes=np.arange(len(self.pyr_int_ids_and_indexes[plane]['pyr'][1]))
            
        all_index_info=[ self.convert_full_planes_idx_to_single_plane_final_indx(full_raster_pyhton_cell_idx,plane)  for full_raster_pyhton_cell_idx in cell_indexes]
            
        all_index_info=[(cell, self.indetify_full_rater_idx_cell_identity(cell[2], plane)) for cell in all_index_info]
                
        return cell_indexes, all_index_info
    
    def get_raster_with_selections(self, trace_type, plane,selected_cells, paradigm='Full', drifting_options=False):
        selected_plane_activity_traces=None
        selected_cells_traces=None
        selected_plane_trials_traces=None
        selected_cells_paradigm_traces=None
        
        cell_range, all_index_info= self.create_cell_selection_ranges(selected_cells, plane) 
        
        selected_plane_activity_traces=self.full_data['imaging_data'][plane]['Traces'][trace_type]
        if selected_plane_activity_traces.any():
            selected_cells_traces=selected_plane_activity_traces[cell_range,:]
        else:    
            selected_cells_traces=np.zeros(self.full_data['imaging_data'][plane]['Traces']['demixed'].shape)
            
        selected_plane_and_cell_timestamps=self.full_data['imaging_data'][plane]['Timestamps'][1]
        bin_threshold=0
        if trace_type=='binarized':
            bin_threshold=self.full_data['imaging_data']['Plane1']['Traces']['binarization_threshold']
            
            
            
        if drifting_options:
            direction_list=drifting_options[0]
            temporal_frequency_list=drifting_options[1]
            seleceted_dur_frames=drifting_options[2]
            seleceted_dur_isi=drifting_options[3]
            selected_stim_table=self.get_trial_from_drifting_angle(direction_list, temporal_frequency_list)
            duration_dict=self.define_pre_post_frames(seleceted_dur_frames,seleceted_dur_isi)
            trial_indexig_ranges=self.extract_trial_indexing_array(selected_stim_table, duration_dict)
            
            selected_plane_trials_traces=[]
            
            for trial_range in trial_indexig_ranges:
                selected_plane_trials_traces.append(selected_cells_traces[:,trial_range])
                
        if paradigm:
            paradimg_range=self.get_paradigm_range(paradigm)
         
            selected_cells_paradigm_traces=selected_cells_traces[:,paradimg_range]
            selected_cells_paradigm_timestamps=selected_plane_and_cell_timestamps[paradimg_range]
            selected_cells_paradigm_framestamps=paradimg_range
            
            
            
        options_array=[trace_type,plane,selected_cells,all_index_info, paradigm, drifting_options,bin_threshold ]
        
        return selected_plane_activity_traces, \
                selected_cells_traces, \
                selected_plane_and_cell_timestamps, \
                selected_cells_paradigm_traces, \
                selected_cells_paradigm_timestamps, \
                selected_cells_paradigm_framestamps, \
                selected_plane_trials_traces, \
                options_array ,\
                self.full_data['visstim_info'][paradigm]['Resampled_sliced_speed']
        
   
    def check_pyr_int_identif_files(self) :
        self.pyr_int_identif_list=glob.glob(self.data_analysis_path+'\\**pyr_int_identif**', recursive=False)
        
        
    def process_excel_pyr_int_identif_file(self) :
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        identfile=self.acquisition_object.aquisition_name + '_Cell_Identity.xlsx'
        fullfilepath=os.path.join(self.data_analysis_path, identfile)
        pyr_int_identif_path_name='_'.join([self.acquisition_object.aquisition_name, timestr,'pyr_int_identification.pkl'])  

        
        planes=['Plane1', 'Plane2', 'Plane3']
        
        all_df= pd.read_excel(fullfilepath, sheet_name=None, engine="openpyxl")
        dfs=[df for df in all_df.values()]
        planes=[plane for plane in all_df.keys()]

        
        pyr_int_identification={}
        pyramidal_count={}
        interneuron_count={}

        for i,plane in enumerate(planes):
            pyr_int_identification[plane]={'interneuron':{'matlab':np.array(dfs[i][(dfs[i]['Accepted']=='+')& (dfs[i]['Tomato accepted only']=='+')].iloc[:,0].tolist()),
                                                           'python':np.array(dfs[i][(dfs[i]['Accepted']=='+')& (dfs[i]['Tomato accepted only']=='+')].iloc[:,0].tolist())-1
                                                           },
                                            'pyramidals':{'matlab':np.array(dfs[i][(dfs[i]['Accepted']=='+')& (dfs[i]['Tomato accepted only']=='-')].iloc[:,0].tolist()),
                                                          'python':np.array(dfs[i][(dfs[i]['Accepted']=='+')& (dfs[i]['Tomato accepted only']=='-')].iloc[:,0].tolist())-1
                                                          }
                                            }



            pyramidal_count[plane]=pyr_int_identification[plane]['pyramidals']['python'].shape[0]
            interneuron_count[plane]=pyr_int_identification[plane]['interneuron']['python'].shape[0]





        datapath=os.path.join(datapath, pyr_int_identif_path_name)
        if not os.path.isfile(datapath):
            with open(datapath, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(pyr_int_identification, f, pickle.HIGHEST_PROTOCOL)
        
       

    def load_pyr_int_identif(self):
        if self.pyr_int_identif_list:
            with open( self.pyr_int_identif_list[0], 'rb') as file:
               self.pyr_int_identification=  pickle.load(file)
        else:
            plane='Plane1'
            self.pyr_int_identification={}
            self.pyr_int_identification[plane]={'interneuron':{'matlab':np.full(1,False),
                                                          'python':np.full(1,False),
                                                          },
                                           'pyramidals':{'matlab':np.full(1,False),
                                                         'python':np.full(1,False),
                                                         }
                                           }
            if not  self.pyr_int_identification[plane]['interneuron']['python']:
                self.pyr_int_identification[plane]['interneuron']['python']=self.full_data['imaging_data']['Plane1']['CellIds']
            print('no previous pyr ident')
            
    def save_pyr_int_identif(self):
        datapath=os.path.join(self.data_analysis_path, self.pyr_int_identif_path_name)
        if not self.pyr_int_identif_list:
            with open(datapath, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self.pyr_int_identification, f, pickle.HIGHEST_PROTOCOL)

    # have to dredo dthis
    def get_pyr_int_indexing_dict(self):
        if not self.nondatabase and  self.pyr_int_identification:
            self.pyr_int_ids_and_indexes={}
            for plane in self.pyr_int_identification.keys():
                self.pyr_int_ids_and_indexes[plane]={'pyr':(np.array(self.pyr_int_identification[plane]['pyramidals']['python']), 
                                        np.in1d(self.full_data['imaging_data'][plane]['CellIds'], self.pyr_int_identification[plane]['pyramidals']['python'])),
                                  'int':(np.array(self.pyr_int_identification[plane]['interneuron']['python']),
                                        np.in1d(self.full_data['imaging_data'][plane]['CellIds'], self.pyr_int_identification[plane]['interneuron']['python']))}
                
             
                
            pyramidal_indexes=[[],[]] 
            for plane in self.pyr_int_ids_and_indexes.keys() :
                pyramidal_indexes[0].append(self.pyr_int_ids_and_indexes[plane]['pyr'][0])
                pyramidal_indexes[1].append(self.pyr_int_ids_and_indexes[plane]['pyr'][1])
            interneruosn_indexes=[[],[] ]  
            for plane in self.pyr_int_ids_and_indexes.keys() :
                interneruosn_indexes[0].append(self.pyr_int_ids_and_indexes[plane]['int'][0])
                interneruosn_indexes[1].append(self.pyr_int_ids_and_indexes[plane]['int'][1])
                
            self.pyr_int_ids_and_indexes['All_planes_rough']  = {'pyr':(np.hstack(pyramidal_indexes[0]).flatten(), 
                                    np.hstack(pyramidal_indexes[1]).flatten()),
                              'int':(np.hstack(interneruosn_indexes[0]).flatten(), 
                                                      np.hstack(interneruosn_indexes[1]).flatten())}
                
                  
      
            
            
            
            # self.pyr_int_identification['Plane1']['pyramidals']['python']=np.setxor1d(self.pyr_int_identification['Plane1']['interneuron']['python'],self.full_data['imaging_data']['Plane1']['CellIds'])
            
            # self.pyr_int_ids_and_indexes={'pyr':(self.pyr_int_identification['Plane1']['pyramidals']['python'], 
            #                         np.in1d(self.full_data['imaging_data']['Plane1']['CellIds'], self.pyr_int_identification['Plane1']['pyramidals']['python'])),
            #                  'int':(np.array(self.pyr_int_identification['Plane1']['interneuron']['python']),
            #                         np.in1d(self.full_data['imaging_data']['Plane1']['CellIds'], self.pyr_int_identification['Plane1']['interneuron']['python']))}

   

    # def identify_in_pyr(self):
            
    #     for key,dataset in self.calcium_datasets.items():
    #         dataset.find_associated_fov_tomato_dataset()
    #         dataset.find_associated_channel_dataset()
    #         if hasattr(dataset, 'associated_channel_dataset_object'):
    #             dataset.associated_tomato_dataset=dataset.associated_channel_dataset_object
    #         elif hasattr(dataset, 'associated_fov_tomato_dataset_red_object'):
    #             dataset.associated_tomato_dataset=dataset.associated_fov_tomato_dataset_red_object
    #         else:
    #             print('No associated channel')
    #     for caiman_res in  self.caiman_results.values(): caiman_res.plot_registered_two_color_projections()
        
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
    def do_PCA(self, mean_sweep_response, sweep_response, driftgrattable, params):
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.params=params
        trial_mean_peak=mean_sweep_response
        self.trial_traces=sweep_response
        filename=r'C:/Users/sp3660/Desktop/PCAanim.mp4'
        trial_frames_number=65
        self.start_stim=0
        self.end_stim=2
        self.frames_pre_stim = 16
        self.frames_post_stim = 16
        
        full_trials=np.zeros((self.trial_traces.shape[0],self.trial_traces.shape[1],self.trial_traces.iloc[0,0].shape[0]))
        
        for i in range(self.trial_traces.shape[0]): #iterate over rows
            for j in range(self.trial_traces.shape[1]):
                full_trials[i,j,:]=self.trial_traces.iloc[i,j]
                
       
        self.trials=[]
        for i in range(full_trials.shape[0]):
            self.trials.append(full_trials[i,:,:])
       
        self.trial_type=driftgrattable['orientation'].values.tolist()
        self.trial_types=np.arange(0,360,int(360/8))
        self.time=np.linspace(-1,3,trial_frames_number)
        self.trial_size   = self.trials[0].shape[1]
        Nneurons     = self.trials[0].shape[0]
        self.t_type_ind = [np.argwhere(np.array(self.trial_type) == t_type)[:, 0] for t_type in self.trial_types]
        print('Number of trials: {}'.format(len(self.trials)))
        print('Types of trials (orientations): {}'.format(self.trial_types)) 
        print('Dimensions of single trial array (# neurons by # time points): {}'.format(self.trials[0].shape))
        print('Trial types (orientations): {}'.format(self.trial_types))
        print('Trial type of the first 3 trials: {}'.format(self.trial_types[0:3]))
       
       
        self.shade_alpha  = 0.2
        self.lines_alpha   = 0.8
        self.pal= sns.color_palette("tab10", 8)
        # config InlineBackend.figure_format = 'svg'
        
        # Xr, Xr_sc, pca, Xp=self.get_concatentaed_trial_PCA()
        mean_response_pca=self.get_concatentaed_trial_PCA()
        # Xa, pca, Xa_p=self.get_concatentaed_trial_averaged_PCA()
        trial_averaged_pca=self.get_concatentaed_trial_averaged_PCA()
        # Xl, pca, Xl_p=self.get_trial_concatenated_PCA()
        single_trial_pca=self.get_trial_concatenated_PCA()
        Xav_sc, pca, projected_trials, gt=self.get_hybrid_PCA()
        hybrid_pca=self.get_hybrid_PCA()
        self.animation_2d_scatter()
        Xa, pca, Xa_p=self.get_3d_PCA()
        self.animate_3d_trial_averaged(Xa_p)
        self.animate_3d_single_trial()
        
        
        return mean_response_pca, trial_averaged_pca, single_trial_pca, hybrid_pca
        
        
    def add_stim_to_plot(self, ax):
        ax.axvspan(self.start_stim, self.end_stim, alpha=self.shade_alpha,
                   color='gray')
        ax.axvline(self.start_stim, alpha=self.lines_alpha, color='gray', ls='--')
        ax.axvline(self.end_stim, alpha=self.lines_alpha, color='gray', ls='--')
        
    def add_orientation_legend(self, ax):
        custom_lines = [Line2D([0], [0], color=self.pal[k], lw=4) for
                        k in range(len(self.trial_types))]
        labels = ['{}$^\circ$'.format(t) for t in self.trial_types]
        ax.legend(custom_lines, labels,
                  frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0,0,0.9,1])
        
    def z_score(self, X):
       # X: ndarray, shape (n_features, n_samples)
       ss = StandardScaler(with_mean=True, with_std=True)
       Xz = ss.fit_transform(X.T).T
       return Xz

    def get_concatentaed_trial_PCA(self):
        
        Xr = np.vstack([t[:, self.frames_pre_stim:-self.frames_post_stim].mean(axis=1) for t in self.trials]).T
        # or take the max
        # Xr = np.vstack([t[:, self.frames_pre_stim:-self.frames_post_stim].max(axis=1) for t in self.trials]).T
        # or the baseline-corrected mean
        # Xr = np.vstack([t[:, self.frames_pre_stim:-self.frames_post_stim].mean(axis=1) - t[:, 0:self.frames_pre_stim].mean(axis=1) for t in self.trials]).T
        Xr_sc = self.z_score(Xr)
        
        pca = PCA(n_components=15)
        Xp = pca.fit_transform(Xr_sc.T).T
        
        projections = [(0, 1), (1, 2), (0, 2)]
        fig, axes = plt.subplots(1, len(projections), figsize=[9, 3], sharey='row', sharex='row')
        for ax, proj in zip(axes, projections):
            for t, t_type in enumerate(self.trial_types):
                x = Xp[proj[0], self.t_type_ind[t]]
                y = Xp[proj[1], self.t_type_ind[t]]
                ax.scatter(x, y, c=self.pal[t], s=25, alpha=0.8)
                ax.set_xlabel('PC {}'.format(proj[0]+1))
                ax.set_ylabel('PC {}'.format(proj[1]+1))
        sns.despine(fig=fig, top=True, right=True)
        self.add_orientation_legend(axes[2])
        
        
        filename = os.path.join( self.data_paths[ 'pca_runs_path'],f'{"_".join(self.params)}_{self.timestr}_mean_trial_PCA.pdf')
        self.save_multi_image(filename)
        plt.close('all')
        return  Xr_sc, pca, Xp
                                                      
    def get_concatentaed_trial_averaged_PCA(self):
        
        trial_averages = []
        for ind in self.t_type_ind:
            trial_averages.append(np.array(self.trials)[ind].mean(axis=0))
        Xa = np.hstack(trial_averages)
        
        n_components = 15
        Xa = self.z_score(Xa) #Xav_sc = center(Xav)
        pca = PCA(n_components=n_components)
        Xa_p = pca.fit_transform(Xa.T).T
        plt.plot(pca.explained_variance_ratio_)
        comp_to_plot=3
        fig, axes = plt.subplots(1, comp_to_plot, figsize=[20, 2.8], sharey='row')
        for comp in range(comp_to_plot):
            ax = axes[comp]
            for kk, type in enumerate(self.trial_types):
                x = Xa_p[comp, kk * self.trial_size :(kk+1) * self.trial_size]
                x = gaussian_filter1d(x, sigma=3)
                ax.plot(self.time, x, c=self.pal[kk])
            self.add_stim_to_plot(ax)
            ax.set_ylabel('PC {}'.format(comp+1))
        self.add_orientation_legend(axes[2])
        axes[1].set_xlabel('Time (s)')
        sns.despine(fig=fig, right=True, top=True)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
                                        
        # find the indices of the three largest elements of the second eigenvector
        units = np.abs(pca.components_[1, :].argsort())[::-1][0:3]
        f, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey=False, sharex=True)
        for ax, unit in zip(axes, units):
            ax.set_title('Neuron {}'.format(unit))
            alldfs=[]
            for t, ind in enumerate(self.t_type_ind):
                x = np.array(self.trials)[ind][:, unit, :]
                df=pd.DataFrame(x.T)
                df.reset_index(inplace=True)
                df['index']=self.time
                df2 = pd.melt(df, id_vars='index', value_vars=np.arange(0,73))
                df2['Orientation'] =self.trial_types[t]
                alldfs.append(df2)
            finaldf = pd.concat(alldfs)
            finaldf.rename(columns = {'index':'time'}, inplace = True)
            finaldf.reset_index(drop=True, inplace=True)
            sns.lineplot(x="time", y="value",hue='Orientation', data=finaldf, ax=ax, legend=False, palette=self.pal)
                
        for ax in axes:
            self.add_stim_to_plot(ax)
            
        axes[1].set_xlabel('Time (s)')
        sns.despine(fig=f, right=True, top=True)
        self.add_orientation_legend(axes[2])
        
        filename = os.path.join( self.data_paths[ 'pca_runs_path'],f'{"_".join(self.params)}_{self.timestr}_trial_averaged_PCA.pdf')
        self.save_multi_image(filename)
        plt.close('all')
        return Xa, pca, Xa_p
             
    def get_trial_concatenated_PCA(self): 
        n_components = 15
        Xl = np.hstack(self.trials)
        Xl = self.z_score(Xl)
        pca = PCA(n_components=15)
        Xl_p = pca.fit_transform(Xl.T).T
        gt = {comp : {t_type : [] for t_type in self.trial_types} for comp in range(n_components)}
        plt.plot(pca.explained_variance_ratio_)
        for comp in range(n_components):
            for i, t_type in enumerate(self.trial_type):
                if not np.isnan(t_type):
                    t = Xl_p[comp, self.trial_size * i: self.trial_size * (i + 1)]
                    gt[comp][t_type].append(t)
        
        for comp in range(n_components):
            for t_type in self.trial_types:
                if not np.isnan(t_type):
                    gt[comp][t_type] = np.vstack(gt[comp][t_type])
                                     
        f, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey=False, sharex=True)
        f.suptitle('trial concatenated PCA' )
        for comp in range(3):
            ax = axes[comp]
            alldfs=[]
            for t, t_type in enumerate(self.trial_types):
                x=gt[comp][t_type]
                df=pd.DataFrame(x.T)
                df.reset_index(inplace=True)
                df['index']=self.time
                df2 = pd.melt(df, id_vars='index', value_vars=np.arange(0,73))
                df2['Orientation'] =t_type
                alldfs.append(df2)
            finaldf = pd.concat(alldfs)
            finaldf.rename(columns = {'index':'time'}, inplace = True)
            finaldf.reset_index(drop=True, inplace=True)
            sns.lineplot(x="time", y="value",hue='Orientation', data=finaldf, ax=ax, legend=False, palette=self.pal)
            self.add_stim_to_plot(ax)
            ax.set_ylabel('PC {}'.format(comp+1))
        axes[1].set_xlabel('Time (s)')
        sns.despine(right=True, top=True)
        self.add_orientation_legend(axes[2])
        
        filename = os.path.join( self.data_paths[ 'pca_runs_path'],f'{"_".join(self.params)}_{self.timestr}_single_trial_PCA.pdf')
        self.save_multi_image(filename)
        plt.close('all')
        return Xl, pca, Xl_p


    def get_hybrid_PCA(self):    
        n_components = 15
        # fit PCA on trial averages
        trial_averages = []
        for ind in self.t_type_ind:
            trial_averages.append(np.array(self.trials)[ind].mean(axis=0))
        Xav = np.hstack(trial_averages)
        
        ss = StandardScaler(with_mean=True, with_std=True)
        Xav_sc = ss.fit_transform(Xav.T).T
        pca = PCA(n_components=15) 
        pca.fit(Xav_sc.T) # only call the fit method
        plt.plot(pca.explained_variance_ratio_)
        self.projected_trials = []
        for trial in self.trials:
            # scale every trial using the same scaling applied to the averages 
            trial = ss.transform(trial.T).T
            # project every trial using the pca fit on averages
            proj_trial = pca.transform(trial.T).T
            self.projected_trials.append(proj_trial)
            
        gt = {comp: {t_type: [] for t_type in self.trial_types}
              for comp in range(n_components)}
        
        for comp in range(n_components):
            for i, t_type in enumerate(self.trial_type  ):
                if not np.isnan(t_type):
                    t = self.projected_trials[i][comp, :]
                    gt[comp][t_type].append(t)
                        
        f, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey=True, sharex=True)
        f.suptitle('Hybrid PCA' )
        for comp in range(3):
            alldfs=[]
            ax = axes[comp]
            for t, t_type in enumerate(self.trial_types):
                x=np.vstack(gt[comp][t_type])
                df=pd.DataFrame(x.T)
                df.reset_index(inplace=True)
                df['index']=self.time
                df2 = pd.melt(df, id_vars='index', value_vars=np.arange(0,73))
                df2['Orientation'] =t_type
                alldfs.append(df2)
            finaldf = pd.concat(alldfs)
            finaldf.rename(columns = {'index':'time'}, inplace = True)
            finaldf.reset_index(drop=True, inplace=True)
            sns.lineplot(x="time", y="value",hue='Orientation', data=finaldf, ax=ax, legend=False, palette=self.pal)
            self.add_stim_to_plot(ax)
            ax.set_ylabel('PC {}'.format(comp+1))
        axes[1].set_xlabel('Time (s)')
        sns.despine(right=True, top=True)
        self.add_orientation_legend(axes[2])
        
        filename = os.path.join( self.data_paths[ 'pca_runs_path'],f'{"_".join(self.params)}_{self.timestr}_hybrid_PCA.pdf')
        self.save_multi_image(filename)
        plt.close('all')
        
        return Xav_sc, pca,  self.projected_trials, gt

        
    def get_3d_PCA(self ):
        # prepare trial averages
        trial_averages = []
        for ind in self.t_type_ind:
            trial_averages.append(np.array(self.trials)[ind].mean(axis=0))
        Xa = np.hstack(trial_averages)
        
        # standardize and apply PCA
        Xa = self.z_score(Xa) 
        pca = PCA(n_components=15)
        Xa_p = pca.fit_transform(Xa.T).T
        
        # pick the components corresponding to the x, y, and z axes
        component_x = 0
        component_y = 1
        component_z = 2
        # create a boolean mask so we can plot activity during stimulus as 
        # solid line, and pre and post stimulus as a dashed line
        stim_mask = ~np.logical_and(np.arange(self.trial_size) >= self.frames_pre_stim,
                       np.arange(self.trial_size) < (self.trial_size-self.frames_post_stim))
        
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
            for t, t_type in enumerate(self.trial_types):
        
                # for every trial type, select the part of the component
                # which corresponds to that trial type:
                x = Xa_p[component_x, t * self.trial_size :(t+1) * self.trial_size]
                y = Xa_p[component_y, t * self.trial_size :(t+1) * self.trial_size]
                z = Xa_p[component_z, t * self.trial_size :(t+1) * self.trial_size]
                
                # apply some smoothing to the trajectories
                x = gaussian_filter1d(x, sigma=sigma)
                y = gaussian_filter1d(y, sigma=sigma)
                z = gaussian_filter1d(z, sigma=sigma)
        
                # use the mask to plot stimulus and pre/post stimulus separately
                z_stim = z.copy()
                z_stim[stim_mask] = np.nan
                z_prepost = z.copy()
                z_prepost[~stim_mask] = np.nan
        
                ax.plot(x, y, z_stim, c = self.pal[t])
                ax.plot(x, y, z_prepost, c=self.pal[t], ls=':')
                
                # plot dots at initial point
                ax.scatter(x[0], y[0], z[0], c=self.pal[t], s=14)
                
                # make the axes a bit cleaner
                style_3d_ax(ax)
                
        # specify the orientation of the 3d plot        
        ax1.view_init(elev=22, azim=30)
        ax2.view_init(elev=22, azim=110)
        plt.tight_layout()
        
        filename = os.path.join( self.data_paths[ 'pca_runs_path'],f'{"_".join(self.params)}_{self.timestr}_trial_averaged_PCA_3d.pdf')
        self.save_multi_image(filename)
        plt.close('all')
        return  Xa, pca, Xa_p
 
    def animation_2d_scatter(self):    
        # smooth the single projected trials 
        for i in range(len(self.projected_trials)):
            for c in range(self.projected_trials[0].shape[0]):
                self.projected_trials[i][c, :] = gaussian_filter1d(self.projected_trials[i][c, :], sigma=3)
        
        # for every time point (imaging frame) get the position in PCA space of every trial
        pca_frame = []
        for t in range(self.trial_size):
            # projected data for all trials at time t 
            Xp = np.hstack([tr[:, None, t] for tr in self.projected_trials]).T
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
        for t, t_type in enumerate(self.trial_types):
            scatter, = ax.plot([], [], 'o', lw=2, color=self.pal[t]);
            scatters.append(scatter)
        
        # red dot to indicate when stimulus is being presented
        stimdot, = ax.plot([], [], 'o', c='r', markersize=35, alpha=0.5)
        
        # annotate with stimulus and time information
        text     = ax.text(6.3, 9, 'Stimulus OFF \nt = {:.2f}'.format(self.time[0]), fontdict={'fontsize':14})
        
        # this is the function to be called at every animation frame
        def animate(i):
            for t, t_type in enumerate(self.trial_types):
                # find the x and y position of all trials of a given type
                x = pca_frame[i][self.t_type_ind[t], subspace[0]]
                y = pca_frame[i][self.t_type_ind[t], subspace[1]]
                # update the scatter
                scatters[t].set_data(x, y)
                
            # update stimulus and time annotation
            if (i > self.frames_pre_stim) and (i < (self.trial_size-self.frames_post_stim)):
                stimdot.set_data(10, 14)
                text.set_text('Stimulus ON \nt = {:.2f}'.format(self.time[i]))
            else:
                stimdot.set_data([], [])
                text.set_text('Stimulus OFF \nt = {:.2f}'.format(self.time[i]))
            return (scatter,)
        
        # generate the animation
        anim = animation.FuncAnimation(fig, animate, 
                                       frames=len(pca_frame), interval=30, 
                                       blit=False)
        
        filename = os.path.join( self.data_paths[ 'pca_runs_path'],f'{"_".join(self.params)}_{self.timestr}_mean_response_animation_PCA.mp4')
        anim.save(filename, writer = 'ffmpeg', fps = 10)
                   
    def animate_3d_trial_averaged(self,Xa_p):
        sigma = 3 # smoothing amount
        component_x = 0
        component_y = 1
        component_z = 2
                 
        pca_frame = []
        for t in range(self.trial_size):
            # projected data for all trials at time t 
            Xp = np.hstack([tr[:, None, t] for tr in self.projected_trials]).T
            pca_frame.append(Xp)           
        subspace = (1, 2) # pick 
        # apply some smoothing to the trajectories
        for c in range(Xa_p.shape[0]):
            Xa_p[c, :] =  gaussian_filter1d(Xa_p[c, :], sigma=sigma)
        
        # create the figure
        fig = plt.figure(figsize=[9, 9]); plt.close()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        
        def animate(i):
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
            
            ax.clear() # clear up trajectories from previous iteration
            style_3d_ax(ax)
            ax.view_init(elev=22, azim=30)
        
            for t, t_type in enumerate(self.trial_types):
            
                x = Xa_p[component_x, t * self.trial_size :(t+1) * self.trial_size][0:i]
                y = Xa_p[component_y, t * self.trial_size :(t+1) * self.trial_size][0:i]
                z = Xa_p[component_z, t * self.trial_size :(t+1) * self.trial_size][0:i]
                        
                stim_mask = ~np.logical_and(np.arange(z.shape[0]) >= self.frames_pre_stim,
                             np.arange(z.shape[0]) < (self.trial_size-self.frames_pre_stim))
        
                z_stim = z.copy()
                z_stim[stim_mask] = np.nan
                z_prepost = z.copy()
                z_prepost[~stim_mask] = np.nan
                
                ax.plot(x, y, z_stim, c = self.pal[t])
                ax.plot(x, y, z_prepost, c=self.pal[t], ls=':')
        
            ax.set_xlim(( -8, 8))
            ax.set_ylim((-8, 8))
            ax.set_zlim((-6, 6))
        
            return []
        
        
        anim = animation.FuncAnimation(fig, animate,
                                       frames=len(pca_frame), interval=30
                                       )
        filename = os.path.join( self.data_paths[ 'pca_runs_path'],f'{"_".join(self.params)}_{self.timestr}_trial_averaged_animation_PCA.mp4')
        anim.save(filename, writer = 'ffmpeg', fps = 10)
        
    def animate_3d_single_trial(self):
       
        pca_frame = []
        for t in range(self.trial_size):
            # projected data for all trials at time t 
            Xp = np.hstack([tr[:, None, t] for tr in self.projected_trials]).T
            pca_frame.append(Xp)
        
        component_x = 0
        component_y = 1
        component_z = 2
         # set up a dictionary to color each line
        col = {float(self.trial_types[i]) : self.pal[i] for i in range(len(self.trial_types))}
        
        fig = plt.figure(figsize=[9, 9]); plt.close()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        
        def animate(i):
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
            
            ax.clear()
            style_3d_ax(ax)
            ax.view_init(elev=22, azim=30)
            for t, (trial, t_type) in enumerate(zip(self.projected_trials, self.trial_types)):
                if not np.isnan(t_type):
                    
                    x = trial[component_x, :][0:i]
                    y = trial[component_y, :][0:i]
                    z = trial[component_z, :][0:i]
                    
                    stim_mask = ~np.logical_and(np.arange(z.shape[0]) >= self.frames_pre_stim,
                                 np.arange(z.shape[0]) < (self.trial_size-self.frames_pre_stim))
            
                    z_stim = z.copy()
                    z_stim[stim_mask] = np.nan
                    z_prepost = z.copy()
                    z_prepost[~stim_mask] = np.nan
                    
                    ax.plot(x, y, z_stim, c = col[int(t_type)])
                    ax.plot(x, y, z_prepost, c=col[int(t_type)], ls=':')
            
            ax.set_xlim(( -12, 12))
            ax.set_ylim((-12, 12))
            ax.set_zlim((-13, 13))
            ax.view_init(elev=22, azim=30)
        
            return []
        
        anim = animation.FuncAnimation(fig, animate, frames=len(pca_frame), interval=30,blit=True)
        filename = os.path.join( self.data_paths[ 'pca_runs_path'],f'{"_".join(self.params)}_{self.timestr}_single_trial_animation_PCA.mp4')
        anim.save(filename, writer = 'ffmpeg', fps = 10)
  
#%% allen
    def load_allen_analysis(self):
        self.allen_analysis=AllenAnalysis(self)
        
        
    def analyze_movie_one(self):
        pass
        
        
#%% CRFs
    def load_CRFs_analysis(self):
        self.crf_analysis=CRFsResults(self)
        pass
    
    def save_for_matlab_CRFs(self):
        pass
    
    def build_binary_grating_array(self):
        pass
        s
        
        
    
#%% yuriy
    def load_yuriy_analysis(self):
        self.yuriy_analysis=EnsemblesYuriy(self)
    
#%% jesus    
    def run_jesus_analysis(self, activity_arrays):   
        # self.load_jesus_results()
        if isinstance(activity_arrays[-2][2],list):
            selectedcells='selected_cell_range_'+'_'.join([str(i) for i in activity_arrays[-2][2][:2]])+'___'+ '_'.join([str(i) for i in activity_arrays[-2][2][-2:]])

        else:
            selectedcells=activity_arrays[-2][2]
            
            
        final_raster=activity_arrays[3]
        plt.imshow(final_raster, cmap='binary', aspect='auto', vmax=0.01)
        plt.title('_'.join([activity_arrays[-2][0], activity_arrays[-2][1], activity_arrays[-2][4], selectedcells]))
        plt.show()
        self.jesus_binary_spikes=final_raster
        pprint('_'.join([activity_arrays[-2][0], activity_arrays[-2][1], activity_arrays[-2][4], selectedcells]))
        self.jesus_analysis=JesusEnsemblesResults(self)
        pprint('_'.join([activity_arrays[-2][0], activity_arrays[-2][1], activity_arrays[-2][4], selectedcells]))

        # self.jesus_runs[ self.run_number+'_'+self.acquisition_object.aquisition_name+'_'+binary_raster_to_proces+'_'+plane+'_'+segment]=[binary_raster_to_proces, plane, segment, self.jesus_binary_spikes, self.jesus_analysis.analysis]
                
        self.jesus_run=[activity_arrays[-2][0], activity_arrays[-2][1], activity_arrays[-2][4], selectedcells, 'threshold_'+str(activity_arrays[-2][-1]), self.jesus_analysis.analysis]
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.jesus_runs_path_name='_'.join([self.aquisition_id, timestr, activity_arrays[-2][0], activity_arrays[-2][1], activity_arrays[-2][4],selectedcells,'binarization_threshold_'+str(activity_arrays[-2][-1]),'jesus_results.pkl'])  


        self.save_jesus_runs()
        
    def save_jesus_runs(self):
        print('Saving jesus run')
        datapath=os.path.join( self.data_paths[ 'jesus_runs_path'], self.jesus_runs_path_name)
        
        if not os.path.isdir( self.data_paths[ 'jesus_runs_path']):
            os.mkdir( self.data_paths[ 'jesus_runs_path'])
            
    
        with open(datapath, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.jesus_run, f, pickle.HIGHEST_PROTOCOL)
        
    def check_all_jesus_results(self) :
        
        self.jesus_results_list=glob.glob( self.data_paths[ 'jesus_runs_path']+'\\**.pkl', recursive=False)
        def intersection(lst1, lst2):
            return list(set(lst1) & set(lst2))

        
        pprint(self.jesus_results_list)
    

        #results by cell type
        pyr_results=[i for i in  self.jesus_results_list if (('Pyr' in i) and ('.pkl' in i))]
        int_results=[i for i in  self.jesus_results_list if (('_Interneurons_' in i) and ('.pkl' in i))]
        all_cells_results=[i for i in  self.jesus_results_list if (('_All_jes' in i) and ('.pkl' in i))]
        rangeof_cells_results=[i for i in  self.jesus_results_list if (('selected_cell_range' in i) and ('.pkl' in i))]

        #results by paradigm
        full_movie_results=[i for i in  self.jesus_results_list if (('_Full' in i) and ('.pkl' in i))]
        drift_grat_results=[i for i in  self.jesus_results_list if (('Drifting' in i) and ('.pkl' in i))]
        movie1_results=[i for i in  self.jesus_results_list if (('Movie1' in i) and ('.pkl' in i))]
        movie2_results=[i for i in  self.jesus_results_list if (('Movie3' in i) and ('.pkl' in i))]
        spont_results=[i for i in  self.jesus_results_list if (('Spont' in i) and ('.pkl' in i))]


        #results by trace type
        mcmcresults=[i for i in  self.jesus_results_list if (('mcmc_scored' in i) and ('.pkl' in i))]
        dfdtresults=[i for i in  self.jesus_results_list if (('dfdt' in i) and ('.pkl' in i))]
        scoredmcmcresults=[i for i in  self.jesus_results_list if (('mcmc_scored' in i) and ('.pkl' in i))]
        binarizedresults=[i for i in  self.jesus_results_list if (('binarized' in i) and ('.pkl' in i))]



         # gratings
         
        self.sorted_jesus_results={
            'selected_cells':rangeof_cells_results,
            'pyr_grat':intersection(pyr_results, drift_grat_results),
            'int_grat':intersection(int_results, drift_grat_results),
            'all_cells_grat':intersection(all_cells_results, drift_grat_results),
            'selected_cells_grat':intersection(rangeof_cells_results, drift_grat_results)
            }
        
        self.sorted_jesus_results['all_cells_grat_mcmcscored']=intersection(self.sorted_jesus_results['all_cells_grat'], scoredmcmcresults),
        self.sorted_jesus_results['all_cells_grat_binarized']=intersection(self.sorted_jesus_results['all_cells_grat'], binarizedresults),
        self.sorted_jesus_results['pyr_grat_mcmcscored']=intersection(self.sorted_jesus_results['pyr_grat'], scoredmcmcresults),
        self.sorted_jesus_results['int_grat_mcmcscored']=intersection(self.sorted_jesus_results['int_grat'], scoredmcmcresults),
       
     
    def unload_all_runs(self):
        
        for k,i in self.jesus_runs.items():
            del i
        del  self.jesus_runs
        gc.collect()
        sys.stdout.flush()
        self.jesus_runs={}
        print('runs unloaded')
        
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
    
    def save_multi_image(self, filename):
       pp = PdfPages(filename)
       fig_nums = plt.get_fignums()
       figs = [plt.figure(n) for n in fig_nums]
       for fig in figs:
          fig.savefig(pp, format='pdf')
       pp.close()
       plt.close('all')
      
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

