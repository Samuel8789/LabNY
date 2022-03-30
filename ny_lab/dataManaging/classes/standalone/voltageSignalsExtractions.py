# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:32:35 2021

@author: sp3660
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import mplcursors
import pickle
import math
import shutil
import scipy.io as sio
from scipy.signal import medfilt
from scipy import stats, interpolate


import gc
# from TestPLot import SnappingCursor
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b"]) 
import logging 
module_logger = logging.getLogger(__name__)


class VoltageSignalsExtractions():
    
    def __init__(self, voltage_excel_path=False, temporary_path=False, just_copy=False, acquisition_directory_raw=False, voltage_signals_object=None):
        module_logger.info('Processing Voltage Signals')
        
        self.voltage_excel_path=voltage_excel_path
        self.temporary_path=temporary_path
        self.acquisition_directory_raw=acquisition_directory_raw
        self.voltage_signals_object=voltage_signals_object


        if just_copy and self.voltage_excel_path:
            self.check_csv_in_folder()
            shutil.copy(self.voltage_excel_path, self.temporary_path)
            
        
        elif not just_copy and self.voltage_excel_path:
            self.check_csv_in_folder()

            self.voltage_signals_raw = pd.read_csv(self.voltage_excel_path)    
            self.voltage_signals={signal:self.voltage_signals_raw[signal].to_frame() for signal in self.voltage_signals_raw.columns.tolist()[1:]}
            
            self.transitions_dictionary={}
            [path, file_name]=os.path.split(self.voltage_excel_path)
            transition_index_file_name=os.path.splitext(file_name)[0]+'_transitions_indexes.pkl'
            self.transition_index_to_save_path= os.path.join(path,transition_index_file_name)
            self.correct_signals_names()
    
            locomotion_df=self.voltage_signals['Locomotion'].T
            self.locomotion_aray=locomotion_df.to_numpy()
            visualstim_df=self.voltage_signals['VisStim'].T
            self.visualstim_array=visualstim_df.to_numpy().squeeze()
            self.voltage_rate=1000;
            self.milisecondscale=np.arange(1,locomotion_df.size+1)
            self.second_scale=self.milisecondscale/self.voltage_rate
            self.minutes_scale=self.second_scale/60
            self.process_allenA_signals()
            
            if os.path.isfile(self.transition_index_to_save_path):
                self.load_indexes_from_file()
    
            # if not self.transitions_dictionary:       
                # self.get_paradign_indexes()
                # self.save_transition_indexes()
                
            # self.slice_visstim_by_paradigm()
            # self.get_drifting_gratings_indexes()
            self.process_locomotion()
            # self.slice_locomotion_by_paradigm()
            
            
            # self.plotting_paradigm_transitions()
            # self.plotting_grating_transitions()
        elif self.voltage_signals_object:
            
            transition_index_file_name=os.path.splitext(self.voltage_signals_object.acquisition_object.aquisition_name)[0]+'_transitions_indexes.pkl'
            drifting_gratings_index_file_name_on=os.path.splitext(self.voltage_signals_object.acquisition_object.aquisition_name)[0]+'_drifting_grating_indexes_on.pkl'
            drifting_gratings_index_file_name_off=os.path.splitext(self.voltage_signals_object.acquisition_object.aquisition_name)[0]+'_drifting_grating_indexes_off.pkl'
            drifting_gratings_sliced_index_file_name_on=os.path.splitext(self.voltage_signals_object.acquisition_object.aquisition_name)[0]+'_drifting_grating_sliced_indexes_on.pkl'
            drifting_gratings_sliced_index_file_name_off=os.path.splitext(self.voltage_signals_object.acquisition_object.aquisition_name)[0]+'_drifting_grating_sliced_indexes_off.pkl'


            self.transition_index_to_save_path= os.path.join(self.voltage_signals_object.acquisition_object.slow_storage_all_paths['visual stim'],transition_index_file_name)
            self.drifting_gratings_index_to_save_path_on= os.path.join(self.voltage_signals_object.acquisition_object.slow_storage_all_paths['visual stim'],drifting_gratings_index_file_name_on)
            self.drifting_gratings_index_to_save_path_off= os.path.join(self.voltage_signals_object.acquisition_object.slow_storage_all_paths['visual stim'],drifting_gratings_index_file_name_off)
            self.drifting_gratings_sliced_index_to_save_path_on= os.path.join(self.voltage_signals_object.acquisition_object.slow_storage_all_paths['visual stim'],drifting_gratings_sliced_index_file_name_on)
            self.drifting_gratings_sliced_index_to_save_path_off= os.path.join(self.voltage_signals_object.acquisition_object.slow_storage_all_paths['visual stim'],drifting_gratings_sliced_index_file_name_off)


            self.voltage_signals=self.voltage_signals_object.voltage_signals_dictionary
            self.voltage_signals_daq=self.voltage_signals_object.voltage_signals_dictionary_daq
            self.voltage_rate=1000;
            self.milisecondscale=np.arange(1,self.voltage_signals['Locomotion'].T.size+1)
            self.second_scale=self.milisecondscale/self.voltage_rate
            self.minutes_scale=self.second_scale/60
            self.time_scale=self.milisecondscale
            self.check_vis_stim_stimuli_in_database()
            self.process_visstim_signal()
            self.process_locomotion()
            self.process_all_signals()
            self.plotting_paradigm_transitions()
            self.plotting_grating_transitions()
            
            
        
    def process_all_signals(self):
        
        
        if 'Allen' in self.vis_stim_protocol:
            self.transitions_dictionary={}
            self.load_indexes_from_file()
            self.process_paradigms()
           
            if self.vis_stim_protocol=='AllenA':
              self.process_allenA_signals()
           
            if self.vis_stim_protocol=='AllenB':
               self.process_allenB_signals()
               
            if self.vis_stim_protocol=='AllenC':
               self.process_allenC_signals()
           
        if self.vis_stim_protocol=='Mistmatch':
           self.transitions_dictionary={}
           self.load_indexes_from_file()
           self.process_mistmatch_signals()
           
        if self.vis_stim_protocol=='Habituation':
           self.transitions_dictionary={}
           self.load_indexes_from_file()
           self.process_habituation_signals()
        
        
#%% common 

    def resample(self, x, factor, kind='linear'):
        n = int(np.ceil(x.size / factor))
        f = interpolate.interp1d(np.linspace(0, 1, x.size), x, kind)
        return f(np.linspace(0, 1, n))       

    def method_to_donwsample_all_signals_for_faster_plotting(self):
        self.all_downsampled_signals={}
        for key, signal in self.voltage_signals:
            self.all_downsampled_signals[key]=self.resample( signal, factor=self.milisecondscale, kind='linear').squeeze()
        
        
    def check_vis_stim_stimuli_in_database(self):
        if self.voltage_signals_object.acquisition_object.full_database_dictionary:
        
            self.database_VisStimInfo=self.voltage_signals_object.acquisition_object.full_database_dictionary['VisStim']
            if self.database_VisStimInfo['VisStimProtocol_name'][0]=='Allen Session A Version A':
                self.vis_stim_protocol='AllenA'
                self.vis_stim_protocol_version='Version A'
            elif self.database_VisStimInfo['VisStimProtocol_name'][0]=='Allen Session B Version A':
                self.vis_stim_protocol='AllenB'
                self.vis_stim_protocol_version='Version A'
            elif self.database_VisStimInfo['VisStimProtocol_name'][0]=='Allen Session C Version A':
                self.vis_stim_protocol='AllenC'
                self.vis_stim_protocol_version='Version A'

      
    
 
    def process_locomotion(self):
        locomotion_df=self.voltage_signals['Locomotion'].T
        self.locomotion_aray=locomotion_df.to_numpy().squeeze()
        # calculate movement speed in transitions per milisecond
        self.first_derivative_locomotion=np.diff(np.around(self.locomotion_aray,4))
        # recify(get all signals as positive)
        self.rectified_speed_array=np.absolute(self.first_derivative_locomotion)
        # add the datapoint lost during the diff
        self.rectified_speed_array=np.insert(self.rectified_speed_array,0,0) 
        #get acceleration
        self.acceleration_array=np.diff(np.around(self.rectified_speed_array,4))
        self.acceleration_array=np.absolute(self.acceleration_array)
        # add the datapoint lost during the diff
        self.acceleration_array=np.insert(self.acceleration_array,0,0) 
        
    def choose_time_scale(self, scale)   :
        if scale=='miliseconds':
            self.time_scale=self.milisecondscale
        elif scale=='secs':
            self.time_scale=self.second_scale
        elif scale=='min':
            self.time_scale=self.minutes_scale
        
        
    def process_visstim_signal(self):
        if 'VisStim' in self.voltage_signals.keys():
            visualstim_df=self.voltage_signals['VisStim'].T
            self.visualstim_array=visualstim_df.to_numpy().squeeze()
            self.rounded_vis_stim=np.around(self.visualstim_array, 1)
            self.dfdt_rounded_vis_stim = np.diff(self.rounded_vis_stim)
        


#%% Allen

    def process_paradigms(self):
        
        self.get_paradigm_indexes()
        self.slice_visstim_by_paradigm()
        self.slice_locomotion_by_paradigm()
        

    def process_allenA_signals(self):   
        module_logger.info('Analysing AllenA Gratings')
        # fist get transitions between stimulaton paradigms
        self.get_drifting_gratings_indexes()
        
    def process_allenB_signals(self):   
        module_logger.info('Analysing AllenA Gratings')
        
        # fist get transitions between stimulaton paradigms
        pass
        # self.get_drifting_gratings_indexes()
    
    def process_allenC_signals(self):   
        module_logger.info('Analysing AllenA Gratings')
        
        # fist get transitions between stimulaton paradigms
        pass
        # self.get_drifting_gratings_indexes()
        
    def get_paradigm_indexes(self):   
        '''
        # this has to be done semimanually 
        Acquisitions already done
            211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_transitions_indexes
            
        '''
        # %matplotlib qt
        
        
        
        if not self.transitions_dictionary:
        
            self.start_transitions=np.argwhere(self.dfdt_rounded_vis_stim<-6.5).flatten()+1
            self.start_transitions=np.delete(self.start_transitions, -1)
            self.end_transitions=np.argwhere(self.dfdt_rounded_vis_stim>4.9).flatten()+1   
            self.end_transitions=np.delete(self.end_transitions, 0)
            #eliminate first 
        
            self.spont_start_transitions=np.argwhere(np.logical_and(self.dfdt_rounded_vis_stim>5, self.dfdt_rounded_vis_stim<6.5)).flatten()+1
            self.last_down_transition=np.argwhere(self.dfdt_rounded_vis_stim<-0.5).flatten()[-1]+1
            
            
            # newest signals thre allen sessions
            # self.transitions_dictionary={'first_noise_set_first':self.start_transitions[0],
            #                             'first_noise_set_last': self.end_transitions[0],
            #                             'second_noise_set_first':self.start_transitions[4],
            #                             'second_noise_set_last':self.end_transitions[4],
            #                             'third_noise_set_first':self.start_transitions[8],
            #                             'third_noise_set_last':self.end_transitions[8],
            #                             'first_movie_set_first':self.start_transitions[2],
            #                             'first_movie_set_last': self.end_transitions[2],
            #                             'second_movie_set_first':self.start_transitions[6],
            #                             'second_movie_set_last': self.end_transitions[6],
            #                             'spont1_first':self.start_transitions[1],
            #                             'spont1_last':self.end_transitions[1],
            #                             'spont2_first':self.start_transitions[7],
            #                             'spont2_last':self.end_transitions[7],
            #                             }
            
            
        
     
            # for  211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_transitions_indexes
            # self.start_transitions=np.argwhere(self.dfdt_rounded_vis_stim<-6.5).flatten()+1
            # self.end_transitions=np.argwhere(self.dfdt_rounded_vis_stim>6.5).flatten()+1   
            # self.end_transitions=np.delete(self.end_transitions, 0)
            # #eliminate first 
        
            # self.spont_start_transitions=np.argwhere(np.logical_and(self.dfdt_rounded_vis_stim>5, self.dfdt_rounded_vis_stim<6.5)).flatten()+1
            # self.last_down_transition=np.argwhere(self.dfdt_rounded_vis_stim<-0.5).flatten()[-1]+1
            # self.transitions_dictionary={'first_drifting_set_first':self.start_transitions[0],
            #                             'first_drifting_set_last': self.end_transitions[0],
            #                             'second_drifting_set_first':self.start_transitions[6],
            #                             'second_drifting_set_last':self.end_transitions[5],
            #                             'third_drifting_set_first':self.start_transitions[10],
            #                             'third_drifting_set_last':self.last_down_transition+1,
            #                             'first_movie_set_first':self.start_transitions[2],
            #                             'first_movie_set_last': self.end_transitions[2],
            #                             'second_movie_set_first':self.start_transitions[8],
            #                             'second_movie_set_last': self.end_transitions[7],
            #                             'short_movie_set_first':self.start_transitions[4],
            #                             'short_movie_set_last':self.end_transitions[4],
            #                             'spont_first':self.spont_start_transitions[3],
            #                             'spont_last':self.end_transitions[6]-1,
            #                             }
 
            # for spja only
            # self.end_transitions=np.argwhere(self.dfdt_rounded_vis_stim<-6.5).flatten()+1 
            # self.start_transitions=np.argwhere(self.dfdt_rounded_vis_stim>5).flatten()+1   
            # self.last_down_transition=np.argwhere(self.dfdt_rounded_vis_stim<-0.5).flatten()[-1]+1
            # self.transitions_dictionary={'first_drifting_set_first':self.end_transitions[0],
            #                             'first_drifting_set_last': self.start_transitions[1],
            #                             'second_drifting_set_first':self.end_transitions[5],
            #                             'second_drifting_set_last':self.start_transitions[8],
            #                             'third_drifting_set_first':self.end_transitions[8],
            #                             'third_drifting_set_last':self.last_down_transition+1000,
            #                             'first_movie_set_first':self.end_transitions[2],
            #                             'first_movie_set_last': self.start_transitions[4],
            #                             'second_movie_set_first':self.end_transitions[6],
            #                             'second_movie_set_last': self.start_transitions[9],
            #                             'short_movie_set_first':self.end_transitions[4],
            #                             'short_movie_set_last':self.start_transitions[7],
            #                             'spont_first':self.start_transitions[8],
            #                             'spont_last':self.end_transitions[6]-3,
            #                             }
            
            fig, ax = plt.subplots(2)
            line, = ax[0].plot(self.time_scale,self.visualstim_array) 
            ax[0].plot(self.time_scale[self.start_transitions],self.visualstim_array[self.start_transitions],'rx') 
            ax[0].plot(self.time_scale[self.end_transitions-1],self.visualstim_array[self.end_transitions-1],'bo') 
            line, = ax[1].plot(self.dfdt_rounded_vis_stim) 
            
         
         
    def slice_visstim_by_paradigm (self):  
        
        if self.vis_stim_protocol =='AllenA':
            
            self.first_drifting_set=self.rounded_vis_stim[self.transitions_dictionary['first_drifting_set_first']:self.transitions_dictionary['first_drifting_set_last']]
            self.second_drifting_set=self.rounded_vis_stim[self.transitions_dictionary['second_drifting_set_first']:self.transitions_dictionary['second_drifting_set_last']]
            self.third_drifting_set=self.rounded_vis_stim[self.transitions_dictionary['third_drifting_set_first']:self.transitions_dictionary['third_drifting_set_last']]
            self.first_movie_set=self.rounded_vis_stim[self.transitions_dictionary['first_movie_set_first']:self.transitions_dictionary['first_movie_set_last']]
            self.second_movie_set=self.rounded_vis_stim[self.transitions_dictionary['second_movie_set_first']:self.transitions_dictionary['second_movie_set_last']]
            self.short_movie_set=self.rounded_vis_stim[self.transitions_dictionary['short_movie_set_first']:self.transitions_dictionary['short_movie_set_last']]     
            self.spont=self.rounded_vis_stim[self.transitions_dictionary['spont_first']:self.transitions_dictionary['spont_last']]
        
            self.first_drifting_set=self.visualstim_array[self.transitions_dictionary['first_drifting_set_first']:self.transitions_dictionary['first_drifting_set_last']]
            self.second_drifting_set=self.visualstim_array[self.transitions_dictionary['second_drifting_set_first']:self.transitions_dictionary['second_drifting_set_last']]
            self.third_drifting_set=self.visualstim_array[self.transitions_dictionary['third_drifting_set_first']:self.transitions_dictionary['third_drifting_set_last']]
            self.first_movie_set=self.visualstim_array[self.transitions_dictionary['first_movie_set_first']:self.transitions_dictionary['first_movie_set_last']]
            self.second_movie_set=self.visualstim_array[self.transitions_dictionary['second_movie_set_first']:self.transitions_dictionary['second_movie_set_last']]
            self.short_movie_set=self.visualstim_array[self.transitions_dictionary['short_movie_set_first']:self.transitions_dictionary['short_movie_set_last']]     
            self.spont=self.visualstim_array[self.transitions_dictionary['spont_first']:self.transitions_dictionary['spont_last']]
            
        elif self.vis_stim_protocol =='AllenC':
            
            self.first_noise_set=self.rounded_vis_stim[self.transitions_dictionary['first_noise_set_first']:self.transitions_dictionary['first_noise_set_last']]
            self.second_noise_set=self.rounded_vis_stim[self.transitions_dictionary['second_noise_set_first']:self.transitions_dictionary['second_noise_set_last']]
            self.third_noise_set=self.rounded_vis_stim[self.transitions_dictionary['third_noise_set_first']:self.transitions_dictionary['third_noise_set_last']]
            self.first_movie_set=self.rounded_vis_stim[self.transitions_dictionary['first_movie_set_first']:self.transitions_dictionary['first_movie_set_last']]
            self.second_movie_set=self.rounded_vis_stim[self.transitions_dictionary['second_movie_set_first']:self.transitions_dictionary['second_movie_set_last']]
            self.spont2=self.rounded_vis_stim[self.transitions_dictionary['spont2_first']:self.transitions_dictionary['spont2_last']]     
            self.spont1=self.rounded_vis_stim[self.transitions_dictionary['spont1_first']:self.transitions_dictionary['spont1_last']]
               

    def slice_locomotion_by_paradigm (self):  
         
        if self.vis_stim_protocol =='AllenA':
        
             self.first_drifting_set_speed=self.rectified_speed_array[self.transitions_dictionary['first_drifting_set_first']:self.transitions_dictionary['first_drifting_set_last']]
             self.second_drifting_set_speed=self.rectified_speed_array[self.transitions_dictionary['second_drifting_set_first']:self.transitions_dictionary['second_drifting_set_last']]
             self.third_drifting_set_speed=self.rectified_speed_array[self.transitions_dictionary['third_drifting_set_first']:self.transitions_dictionary['third_drifting_set_last']]
             self.first_movie_set_speed=self.rectified_speed_array[self.transitions_dictionary['first_movie_set_first']:self.transitions_dictionary['first_movie_set_last']]
             self.second_movie_set_speed=self.rectified_speed_array[self.transitions_dictionary['second_movie_set_first']:self.transitions_dictionary['second_movie_set_last']]
             self.short_movie_set_speed=self.rectified_speed_array[self.transitions_dictionary['short_movie_set_first']:self.transitions_dictionary['short_movie_set_last']]     
             self.spont_speed=self.rectified_speed_array[self.transitions_dictionary['spont_first']:self.transitions_dictionary['spont_last']]    
        
        elif self.vis_stim_protocol =='AllenC':
        
            self.first_noise_set_speed=self.rectified_speed_array[self.transitions_dictionary['first_noise_set_first']:self.transitions_dictionary['first_noise_set_last']]
            self.second_noise_set_speed=self.rectified_speed_array[self.transitions_dictionary['second_noise_set_first']:self.transitions_dictionary['second_noise_set_last']]
            self.third_noise_set_speed=self.rectified_speed_array[self.transitions_dictionary['third_noise_set_first']:self.transitions_dictionary['third_noise_set_last']]
            self.first_movie_set_speed=self.rectified_speed_array[self.transitions_dictionary['first_movie_set_first']:self.transitions_dictionary['first_movie_set_last']]
            self.second_movie_set_speed=self.rectified_speed_array[self.transitions_dictionary['second_movie_set_first']:self.transitions_dictionary['second_movie_set_last']]
            self.spont2_speed=self.rectified_speed_array[self.transitions_dictionary['spont2_first']:self.transitions_dictionary['spont2_last']]     
            self.spont1_speed=self.rectified_speed_array[self.transitions_dictionary['spont1_first']:self.transitions_dictionary['spont1_last']]
        
        
        
    def get_noise_trial_structure(self):
        pass
    def get_movies_trial_structure(self):
        pass
    def get_static_gratings_trial_structure(self):
        pass

    def get_natural_images_trial_structure(self):
        pass

    def correct_voltage_split_transitions(self, voltage_slice):
        # this is for transition that were split betwen 2 samples, I always get the inital transition to the sample with at tleast some 
        # voltage as voltages is send after the image and for the end transition i get the also the first as the image has chnaged before voltage change
        voltage_slice_filtered=medfilt(voltage_slice, kernel_size=29)
        voltage_slice_filtered_rounded=np.around(voltage_slice_filtered, 1)
        
        voltage_slice_filtered_rounded_corrected=np.copy(voltage_slice_filtered_rounded)

        diff_voltage_slice_filtered_rounded= np.diff(voltage_slice_filtered_rounded)
        diff_voltage_slice_filtered_rounded_rerounded =np.around(diff_voltage_slice_filtered_rounded, 1)

        #correcting  voltage transitions betwen samples
        errors_pairs=[]
        for i in range(0, diff_voltage_slice_filtered_rounded_rerounded.size-2):
            if diff_voltage_slice_filtered_rounded_rerounded[i+1]!=0 and diff_voltage_slice_filtered_rounded_rerounded[i]!=0:
                errors_pairs.append((i+1, i+2))
                voltage_slice_filtered_rounded_corrected[i+1]=voltage_slice_filtered_rounded_corrected[i+2]
                voltage_slice_filtered_rounded_corrected[i+2]=voltage_slice_filtered_rounded_corrected[i+3]
                        
        diff_voltage_slice_filtered_rounded_corrected = np.diff(voltage_slice_filtered_rounded_corrected)
        diff_voltage_slice_filtered_rounded_corrected_rerounded =np.around(diff_voltage_slice_filtered_rounded_corrected, 1)    

        fig,axa=plt.subplots(len(errors_pairs))
        for i ,j in  enumerate(errors_pairs) :  
            l1=axa[i].plot(diff_voltage_slice_filtered_rounded_rerounded[j[0]-10:j[0]+10])
            l2=axa[i].plot(diff_voltage_slice_filtered_rounded_corrected_rerounded[j[0]-10:j[0]+10])

        fig,axo=plt.subplots(len(errors_pairs))
        for i ,j in  enumerate(errors_pairs) :  
            axo[i].plot(voltage_slice_filtered_rounded[j[0]-10:j[0]+10])
            axo[i].plot(voltage_slice_filtered_rounded_corrected[j[0]-10:j[0]+10])
          
        return voltage_slice_filtered_rounded_corrected, diff_voltage_slice_filtered_rounded_corrected, diff_voltage_slice_filtered_rounded_corrected_rerounded, errors_pairs
    
    def get_drifting_gratings_indexes(self):
        
        # paradigm parameters
        self.orientations=np.round(np.linspace(0.1,4,40),1)
        blanksweepvoltage=4.5
        self.repetitions=15
        self.orientations_and_blank_sweep= np.append(self.orientations, blanksweepvoltage)
        self.total_grating_trials=self.orientations.shape[0]*self.repetitions

        self.tuning_stim_on_index_full_recording =np.zeros((1))
        self.tuning_stim_off_index_full_recording=np.zeros((1))
        self.load_drifting_grating_indexes()
        
        self.combined_gratings_raw=np.concatenate((self.first_drifting_set, self.second_drifting_set, self.third_drifting_set))

        
        # if not (self.tuning_stim_on_index_full_recording.any() and self.tuning_stim_off_index_full_recording.any()):
        if True:



            temp=np.diff(np.around(medfilt(self.combined_gratings_raw, kernel_size=29),1))
            temp2=np.around(temp,1)
            drifting_transition_indexes1=[np.argwhere(temp== voltage) for voltage in self.orientations_and_blank_sweep]
            drifting_transition_indexes2=[np.argwhere(temp2== voltage) for voltage in self.orientations_and_blank_sweep]
            
            self.drifting_voltage_slice_filtered_rounded_corrected,self.drifting_diff_voltage_slice_filtered_rounded_corrected, self.drifting_diff_voltage_slice_filtered_rounded_corrected_rerounded,self.drifting_errors_pairs=self.correct_voltage_split_transitions(self.combined_gratings_raw)

            all_on_transition_indexes=[np.argwhere(self.drifting_diff_voltage_slice_filtered_rounded_corrected== voltage).squeeze() for voltage in self.orientations_and_blank_sweep]
            all_on_transition_indexes2=[np.argwhere(self.drifting_diff_voltage_slice_filtered_rounded_corrected_rerounded== voltage) for voltage in self.orientations_and_blank_sweep]
            all_off_transition_indexes=[np.argwhere(self.drifting_diff_voltage_slice_filtered_rounded_corrected== -voltage).squeeze()  for voltage in self.orientations_and_blank_sweep]
            all_off_transition_indexes2=[np.argwhere(self.drifting_diff_voltage_slice_filtered_rounded_corrected_rerounded== -voltage) for voltage in self.orientations_and_blank_sweep]
            self.drifting_on_transition_indexes=np.vstack( all_on_transition_indexes[0:len(self.orientations)])+1
            self.drifting_off_transition_indexes=np.vstack( all_off_transition_indexes[0:len(self.orientations)])+1
            self.blank_on_transition_indexes=np.vstack( all_on_transition_indexes[-1])+1
            self.blank_off_transition_indexes=np.vstack( all_off_transition_indexes[-1])+1
            
            oritoplot=40
            fig,axo=plt.subplots(oritoplot, sharex=(True))
            for k, j in enumerate(range(oritoplot)):
                for i  in  range(15) :  
                    axo[k].plot(self.drifting_voltage_slice_filtered_rounded_corrected[self.drifting_on_transition_indexes[j,i]-600:self.drifting_off_transition_indexes[j,i]+600])
                    
         
            oritoplot=40
            fig,axo=plt.subplots(oritoplot, sharex=(True))
            for k, j in enumerate(range(oritoplot)):
                for i  in  range(15) :  
                    axo[k].plot(self.drifting_voltage_slice_filtered_rounded_corrected[self.drifting_off_transition_indexes[j,i]-5:self.drifting_off_transition_indexes[j,i]+5])

            fig,ax=plt.subplots(2)
            ax[0].plot(self.drifting_diff_voltage_slice_filtered_rounded_corrected)
            ax[0].plot(self.drifting_diff_voltage_slice_filtered_rounded_corrected_rerounded)


            stim_period_lengths=self.drifting_off_transition_indexes-self.drifting_on_transition_indexes
            plt.hist(stim_period_lengths.flatten(), bins=np.arange(2047, 2051), range=(2047,2051))
            plt.show()
       
            index_start=  self.drifting_on_transition_indexes[0]
            index_end= self.drifting_off_transition_indexes[0]
            
            fig,axa=plt.subplots(1)
            for i ,j in  enumerate(index_start) : 
                axa.plot(self.drifting_voltage_slice_filtered_rounded_corrected[j-10:j+2500])
                
            fig,axo=plt.subplots(1)
            for i ,j in  enumerate(index_end) :  
                axo.plot(self.drifting_voltage_slice_filtered_rounded_corrected[j-2500:j+10])

            self.correct_grating_indexes_for_full_movie()

            self.save_drifting_grating_indexes()
            
        self.correct_grating_indexes_for_full_movie()
        self.create_full_recording_grating_binary_matrix()



    def correct_grating_indexes_for_full_movie(self):
        
        first_length=self.first_drifting_set.shape[0]
        second_length=self.second_drifting_set.shape[0]
        mivies_indexes=[self.transitions_dictionary['first_drifting_set_first'],
                        self.transitions_dictionary['first_drifting_set_last'],
                        self.transitions_dictionary['second_drifting_set_first'],
                        self.transitions_dictionary['second_drifting_set_last'],
                        self.transitions_dictionary['third_drifting_set_first'],
                        self.transitions_dictionary['third_drifting_set_last']]
        
        def correctindex(indx, first_length, second_length, mivies_indexes0, mivies_indexes2, mivies_indexes4,):
            if indx < first_length:
                return indx + mivies_indexes0
            elif np.logical_and(indx<first_length+second_length,indx>first_length) :
                return indx + mivies_indexes2-first_length
            elif indx > second_length:
                return indx + mivies_indexes4-first_length-second_length
            
        vfunc = np.vectorize(correctindex)
       
        self.tuning_stim_on_index_full_recording = vfunc(self.drifting_on_transition_indexes, first_length, second_length,  mivies_indexes[0], mivies_indexes[2], mivies_indexes[4])
        self.tuning_stim_off_index_full_recording = vfunc(self.drifting_off_transition_indexes, first_length, second_length,  mivies_indexes[0], mivies_indexes[2], mivies_indexes[4])
        
        
        oritoplot=20
        fig,axo=plt.subplots(oritoplot, sharex=(True))
        for k, j in enumerate(range(oritoplot)):
            for i  in  range(15) :  
                axo[k].plot(self.rounded_vis_stim[self.tuning_stim_on_index_full_recording[j,i]-600:self.tuning_stim_off_index_full_recording[j,i]+600])
                
     
    def create_full_recording_grating_binary_matrix(self):

        self.full_stimuli_binary_matrix=np.zeros((self.orientations.size,self.visualstim_array.shape[0] ))
        for i, row in enumerate(self.tuning_stim_on_index_full_recording):
            for j, trial in enumerate(row):
                self.full_stimuli_binary_matrix[i, self.tuning_stim_on_index_full_recording[i,j]:self.tuning_stim_off_index_full_recording[i,j]]=1


        
        #%%

#%% misc
    def confirm_grating_indexes(self):
        module_logger.info('todo')
        # confirm number of trial
        # confirm length of ranges
        
    def check_csv_in_folder(self):
     if self.acquisition_directory_raw:
            csvfiles=glob.glob(self.acquisition_directory_raw+'\\**.csv')
            for csv in csvfiles:
                if 'VoltageRecording'  in csv:
                    self.voltage_excel_path=csv
                    
#%% mistmatch               
    def process_mistmatch(self):
        pass
#%% habituation

    def habituation_analysis(self):
     
        # #%% HABITUATION GOOD
        # #Global
        # prestim = 29*voltage_rate
        # poststim = 100*voltage_rate
        # nsession = 5;
        # rounded_vis_stim=np.around(visualstim_aray,1)
        # df_rounded_vis_stim = np.diff(rounded_vis_stim)
        # k=np.argwhere(df_rounded_vis_stim<0).flatten()
        
        # first_spont_indx=k[0];
        # last_spont_indx=k[1];
        # first_habituation_indx=last_spont_indx+1;
        # last_habituation_indx=k[6];
        # first_grating_indx=last_habituation_indx+1;
        # last_grating_indx=k[-1];
        
        # period_spontaneous=rounded_vis_stim[first_spont_indx:last_spont_indx+1]
        # period_habituation=rounded_vis_stim[first_habituation_indx:last_habituation_indx+1]
        # period_tuning=rounded_vis_stim[first_grating_indx:last_grating_indx+1]
        
        
        # #%%Habituation period extrct indexes
        # for i in range(0,period_habituation.size):
        #     if period_habituation[i]<0.2:
        #         period_habituation[i]=0
        #     else:
        #         period_habituation[i]=1
        # df_period_habituation = np.diff(period_habituation)
        # df_period_habituation_rounded=np.around(df_period_habituation)
        # stim_on_index_habituation_period = np.argwhere(df_period_habituation_rounded>0.9).flatten()
        # stim_on_index_full_recording = stim_on_index_habituation_period + (first_habituation_indx)
        # #proces locomotion form indexes
        # habituation_periods_locomotion=np.zeros((5,len(list(range(stim_on_index_full_recording[0]-prestim,stim_on_index_full_recording[0]+poststim)))))
        # for i in range(0,5):
        #     habituation_periods_locomotion[i,:]=rectified_speed_array[stim_on_index_full_recording[i]-prestim:stim_on_index_full_recording[i]+poststim]
        # # normalization
        # mean_prestim = np.mean(habituation_periods_locomotion[:,0:prestim],1)
        # norm_locomotion_speed=np.divide(habituation_periods_locomotion.T,mean_prestim).T
        # # results
        # full_trial_averaged_normalized_locomotion=np.mean(norm_locomotion_speed,0);
        # stim_on_mean_normalized_locomotion=np.mean(norm_locomotion_speed[:,prestim:],1)
        # stim_on_trial_averaged_normalized_locomotion = np.mean(stim_on_mean_normalized_locomotion);
        # #%% tuning period extract index
        # orientations=np.linspace(0.5,4,8)
        # repetitions=10
        # orientations_boolean_starts=np.zeros((8,period_tuning.size))
        # orientations_boolean_end=np.zeros((8,period_tuning.size))
        # df_period_tuning = np.diff(period_tuning)
        # df_period_tuning_rounded=np.around(df_period_tuning,1)
        
        # #correcting slower voltage transitions
        # for i in range(2,df_period_tuning_rounded.size-2):
        #     if df_period_tuning_rounded[i+1]!=0 and df_period_tuning_rounded[i]!=0:
        #        df_period_tuning_rounded[i+1]= df_period_tuning_rounded[i+1]+ df_period_tuning_rounded[i];
        #        df_period_tuning_rounded[i]=0;
        
        
        # #getting bolean indexing vecors
        # for i in range(0,df_period_tuning_rounded.size):
        #     for j, voltage in enumerate(orientations):
        #         if df_period_tuning_rounded[i] == voltage:
        #             orientations_boolean_starts[j,i+1] = 1;
                    
        # for i in range(0,df_period_tuning_rounded.size):
        #     for j, voltage in enumerate(orientations):
        #         if df_period_tuning_rounded[i] == -voltage:
        #             orientations_boolean_end[j,i+1] = 1;
        
        # # get indexes
        
        # stim_on_indexes_tuning_period=np.zeros((orientations.size,repetitions)).astype('int64')
        # for row in range(0,orientations.size):
        #     stim_on_indexes_tuning_period[row,:]=np.argwhere(orientations_boolean_starts[row,:]).flatten()
            
        # stim_off_indexes_tuning_period=np.zeros((orientations.size,repetitions)).astype('int64')
        # for row in range(0,orientations.size):
        #     if np.argwhere(orientations_boolean_end[row,:]).flatten().size==9:
        #         tosave= np.argwhere(orientations_boolean_end[row,:]).flatten() 
        #         tosave=np.append(tosave,period_tuning.size)
        #         stim_off_indexes_tuning_period[row,:]=tosave-1
        #     else:
        #         stim_off_indexes_tuning_period[row,:]=np.argwhere(orientations_boolean_end[row,:]).flatten()-1
        
        
        # tuning_stim_on_index_full_recording = stim_on_indexes_tuning_period + (first_grating_indx)
        # tuning_stim_off_index_full_recording = stim_off_indexes_tuning_period + (first_grating_indx)

        #
        module_logger.info('todo')

#%% saving indexes
    def save_transition_indexes(self):
        with open(  self.transition_index_to_save_path, 'wb') as f:
            pickle.dump(self.transitions_dictionary, f, pickle.HIGHEST_PROTOCOL)

    def load_indexes_from_file(self):
        if os.path.isfile(self.transition_index_to_save_path):
            with open(self.transition_index_to_save_path, 'rb') as f:
                self.transitions_dictionary= pickle.load(f)
                
    def save_drifting_grating_indexes(self):
   
        with open( self.drifting_gratings_index_to_save_path_on, 'wb') as f:
            pickle.dump(self.tuning_stim_on_index_full_recording, f, pickle.HIGHEST_PROTOCOL)

        with open(self.drifting_gratings_index_to_save_path_off, 'wb') as f:
            pickle.dump(self.tuning_stim_off_index_full_recording, f, pickle.HIGHEST_PROTOCOL)
            
        with open( self.drifting_gratings_sliced_index_to_save_path_on, 'wb') as f:
            pickle.dump(self.drifting_on_transition_indexes, f, pickle.HIGHEST_PROTOCOL)

        with open(self.drifting_gratings_sliced_index_to_save_path_off, 'wb') as f:
            pickle.dump(self.drifting_off_transition_indexes, f, pickle.HIGHEST_PROTOCOL)
            
    def load_drifting_grating_indexes(self):
        if os.path.isfile(self.drifting_gratings_index_to_save_path_on):
            with open(self.drifting_gratings_index_to_save_path_on, 'rb') as f:
                self.tuning_stim_on_index_full_recording=pickle.load(f)

        if os.path.isfile(self.drifting_gratings_index_to_save_path_off):
            with open( self.drifting_gratings_index_to_save_path_off, 'rb') as f:
                self.drifting_on_transition_indexes=pickle.load(f)
                
                
        if os.path.isfile(self.drifting_gratings_sliced_index_to_save_path_on):
            with open(self.drifting_gratings_sliced_index_to_save_path_on, 'rb') as f:
                self.tuning_stim_on_index_full_recording=pickle.load(f)

        if os.path.isfile(self.drifting_gratings_sliced_index_to_save_path_off):
            with open( self.drifting_gratings_sliced_index_to_save_path_off, 'rb') as f:
                self.drifting_off_transition_indexes=pickle.load(f)
   
           
        
    def save_movie1_frame_indexes(self):
        pass  
    def save_movie3_frame_indexes(self):
        pass
    #%%plotting
    
    def plot_all_basics(self):
        self.plot_vis_stim_trace()
        self.plot_stim_and_speed()
        self.plot_full_locomotion()
        self.plot_speed()
        
    def plot_processed_allen(self):
        self.plotting_paradigm_transitions()
        
    
    def plot_vis_stim_trace(self):
        fig, ax = plt.subplots(1)
        line, = ax.plot(self.time_scale,self.visualstim_array) 
        
    def plot_stim_and_speed(self):
        fig, ax = plt.subplots(2)
        line, = ax[0].plot(self.time_scale,self.visualstim_array) 
        line, = ax[1].plot(self.time_scale,self.rectified_speed_array) 
                
    def plot_full_locomotion(self):
  
        fig, ax = plt.subplots(nrows=3,sharey=True)
        # fig.set_title('Snapping cursor')
        for i in range(0,3):
            if i==0:
                line, = ax[i].plot(self.time_scale,self.locomotion_aray) 
            elif i==1:
                line, = ax[i].plot(self.time_scale,self.rectified_speed_array)  
            elif i==2:
                line, = ax[i].plot(self.time_scale,self.acceleration_array)  
         
        # snap_cursor = SnappingCursor(ax[0], line)  
        # fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
        mplcursors.cursor(line) # or just mplcursors.cursor()
        
    def plot_speed(self):
        fig, ax = plt.subplots(1)
        
        line, = ax.plot(self.time_scale,self.rectified_speed_array) 
        
    def plotting_paradigm_transitions(self):   
        
        if self.vis_stim_protocol=='AllenA':
            datasets_to_plot=[self.first_drifting_set, self.second_drifting_set, self.third_drifting_set, self.first_movie_set,self.second_movie_set,self.short_movie_set,self.spont]
        elif self.vis_stim_protocol=='AllenC':
            datasets_to_plot=[self.first_noise_set, self.second_noise_set, self.third_noise_set, self.first_movie_set,self.second_movie_set,self.spont1,self.spont2]

        
        datasets_to_plot.append(self.rounded_vis_stim)
         
        fig, axs = plt.subplots(1)
        fig.suptitle('VisStim Paradigm Transitions')
        
        axs.plot(self.rounded_vis_stim)     
        symbol_list=['x','o','<','^','v','s','>','+','d',]
        color_list=['r', 'g']
        n=2        
        indexes=list(self.transitions_dictionary.values())
        for i in range(0, len(indexes)-n+1, n):
            axs.plot(indexes[i], self.rounded_vis_stim[indexes[i]],symbol_list[i-int(i/2)],  color=color_list[i%2])
            axs.plot(indexes[i+1], self.rounded_vis_stim[indexes[i+1]],symbol_list[i-int(i/2)],  color=color_list[(i+1)%2])


        fig, axs = plt.subplots(len(datasets_to_plot))
        fig.suptitle('VisStim Paradigm Transitions')
        for i, ax in enumerate(axs):
            ax.plot(datasets_to_plot[i])
        mplcursors.cursor(axs) # 
        
        
        
                
    def plotting_grating_transitions(self): 
        
        pixel_per_bar = 10
        dpi = 200
        fig = plt.figure(figsize=(200 * pixel_per_bar / dpi, 2), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
        ax.set_axis_off()
        ax.imshow(self.full_stimuli_binary_matrix, vmax=0.0001, cmap='binary', aspect='auto')

        
         
        # fig, axs = plt.subplots(4, sharex=True)
        # axs[0].plot(self.combined_gratings_raw)
        # axs[0].plot(self.test_down+1, self.combined_gratings_raw[self.test_down+1],'bx')
        # axs[1].plot(self.combined_gratings_raw)
        # axs[1].plot(self.test_down, self.df_combined_gratings[self.test_down],'bo')
        # axs[2].plot(self.combined_gratings_raw)
        # axs[2].plot(self.test_upn+1, self.combined_gratings_raw[self.test_upn+1],'rx')
        # axs[3].plot(self.combined_gratings_raw)
        # axs[3].plot(self.test_upn, self.combined_gratings_raw[self.test_upn],'ro')
        # mplcursors.cursor(axs) # or just mplcursors.cursor()

        fig, axs = plt.subplots(1)
        axs.plot(self.combined_gratings_raw)
        axs.plot(self.drifting_on_transition_indexes,  self.combined_gratings_raw[self.drifting_on_transition_indexes],  'o')
        axs.plot(self.drifting_off_transition_indexes, self.combined_gratings_raw[self.drifting_off_transition_indexes], 'x')

        fig, axs = plt.subplots(3)
        fig.suptitle('VoltageS Signals')
        axs[0].plot(self.second_scale, self.rounded_vis_stim)
        axs[1].plot(self.second_scale[0:-1], self.dfdt_rounded_vis_stim)
        # axs[2].plot(second_scale,photodiode_aray)
        mplcursors.cursor(axs) # or just mplcursors.cursor()
        
        fig, axs = plt.subplots(1)
        axs.plot(self.rounded_vis_stim)
        axs.plot(self.tuning_stim_on_index_full_recording,  self.rounded_vis_stim[self.tuning_stim_on_index_full_recording],'o')
        axs.plot(self.tuning_stim_off_index_full_recording, self.rounded_vis_stim[self.tuning_stim_off_index_full_recording],'x')
  
        # figure()
        # plot(vistim)
        # indexup=find(vistim>6.5);
        # tocorrectup=find(diff(indexup)<20);
        # indexup(tocorrectup)=[];

        # fig, ax = plt.subplots(nrows=8,sharey=True)
        # # fig.set_title('Snapping cursor')
        # for i in range(0,8):
        #     line, = ax[i].plot(self.orientations_boolean_starts[i]) 
        # # snap_cursor = SnappingCursor(ax[0], line)  
        # # fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
        # mplcursors.cursor(line) # or just mplcursors.cursor()

        fig, ax = plt.subplots(nrows=1,sharey=True)            
        ax.plot(self.rounded_vis_stim)
        ax.plot(self.tuning_stim_off_index_full_recording[0,:]-1,self.rounded_vis_stim[self.tuning_stim_off_index_full_recording[0,:]-1],'o')
        # snap_cursor = SnappingCursor(ax, line)
        # # fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
        # mplcursors.cursor(line) # or just mplcursors.cursor()

        fig, axs=plt.subplots(1)
        axs.plot(self.rounded_vis_stim)
        
        for i in range(40):
            color = tuple(np.random.choice(range(256), size=3)/256)
            axs.plot(np.argwhere(self.full_stimuli_binary_matrix[i,:]) ,self.rounded_vis_stim[np.argwhere(self.full_stimuli_binary_matrix[i,:])],'x', color=color)
            
    def correct_signals_names(self):               
        signals=list(self.voltage_signals.keys())
        for signal in signals:
            if 'Locomotion' in signal:     
                self.voltage_signals['Locomotion']= self.voltage_signals[signal]
                del self.voltage_signals[signal]                
            if 'VisStim' in signal:           
                self.voltage_signals['VisStim']= self.voltage_signals[signal]
                del self.voltage_signals[signal]                
            if 'LED' in signal:               
                self.voltage_signals['LED']= self.voltage_signals[signal]
                del self.voltage_signals[signal]                
            if 'PhotoDiode' in signal:                    
                self.voltage_signals['PhotoDiode']= self.voltage_signals[signal]
                del self.voltage_signals[signal]    
                
                
       
if __name__ == "__main__":
    
    # temporary_path1='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'
    temporary_path1='\\\\?\\'+r'K:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Imaging\G2C\Ai14\SPJA\imaging\20210702\data aquisitions\FOV_1\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\raw_volatge_csv\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'

    voltagesignals=VoltageSignalsExtractions(temporary_path1)
    # voltagesignals.plot_full_locomotion()
    fig, ax = plt.subplots(nrows=2,sharex=True)
    line, = ax[0].plot(voltagesignals.locomotion_aray) 

    # fig.set_title('Snapping cursor')
    # for i in range(0,2):
    #     if i==0:
    #         line, = ax[i].plot(voltagesignals.locomotion_aray) 
    #     elif i==1:
    #         line, = ax[i].plot(voltagesignals.rectified_speed_array)  
    #     elif i==2:
    #         line, = ax[i].plot(voltagesignals.acceleration_array)  
     
    # # snap_cursor = SnappingCursor(ax[0], line)  
    # # fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
    # mplcursors.cursor(line) # or just mplcursors.cursor()
  