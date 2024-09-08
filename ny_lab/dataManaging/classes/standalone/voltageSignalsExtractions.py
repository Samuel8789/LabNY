# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:32:35 2021

@author: sp3660
"""
# lazy_import
import mplcursors
import math
import shutil
import scipy.io as sio
from scipy import interpolate
import os
import pandas as pd
from scipy.stats import  mode
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import glob
import scipy.signal as sg
import scipy.ndimage as ndi

import gc
# from TestPLot import SnappingCursor
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b"]) 
import logging 
module_logger = logging.getLogger(__name__)
from operator import itemgetter
import caiman as cm
import copy 
import json
from cycler import cycler
class VoltageSignalsExtractions():
    
    def __init__(self, voltage_excel_path=False, temporary_path=False, just_copy=False, acquisition_directory_raw=False, voltage_signals_object=None):
        module_logger.info('Processing Voltage Signals')
        
        self.voltage_excel_path=voltage_excel_path
        self.temporary_path=temporary_path
        self.acquisition_directory_raw=acquisition_directory_raw
        self.voltage_signals_object=voltage_signals_object
        self.visualstim_array={}
        self.vis_stim_protocol=None
        self.rounded_vis_stim=np.empty([0])
        self.transitions_dictionary={}
        self.signal_transitions={}
        self.all_final_signals=None
        self.all_signals=None


        if just_copy and self.voltage_excel_path:
            self.check_csv_in_folder()
            shutil.copy(self.voltage_excel_path, self.temporary_path)
            
        
        elif not just_copy and self.voltage_excel_path:
            self.check_csv_in_folder()
            self.update_frame_rates_with_metadata(1000,1000)

            self.voltage_signals_raw = pd.read_csv(self.voltage_excel_path)    
            self.voltage_signals={signal:self.voltage_signals_raw[signal].to_frame() for signal in self.voltage_signals_raw.columns.tolist()[1:]}
            
            self.transitions_dictionary={}
            [path, file_name]=os.path.split(self.voltage_excel_path)
            transition_index_file_name=os.path.splitext(file_name)[0]+'_transitions_indexes.pkl'
            self.indexes_full_file_paths_to_save[0]= os.path.join(path,transition_index_file_name)
            self.correct_signals_names()
    
            locomotion_df=self.voltage_signals['Locomotion'].T
            self.locomotion_array=locomotion_df.to_numpy()
            visualstim_df=self.voltage_signals['VisStim'].T
            self.visualstim_array['Prairie']['VisStim']=visualstim_df.to_numpy().squeeze()
            self.process_allenA_signals()
            
            if os.path.isfile(self.indexes_full_file_paths_to_save[0]):
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
        elif self.voltage_signals_object and self.voltage_signals_object.voltage_signals_dictionary:

            self.voltage_signals=self.voltage_signals_object.voltage_signals_dictionary
            self.voltage_signals_daq=self.voltage_signals_object.voltage_signals_dictionary_daq
            self.all_signals={'Prairie':self.voltage_signals,
                               'Daq':self.voltage_signals_daq
                }
            
            self.update_frame_rates_with_metadata(1000,1000)
            self.check_if_LED_to_align()

            
            
            #THIS IS THE MAIN PROCESSING SITE
            if self.voltage_signals_object.acquisition_object:
                self.acquisition_name=os.path.splitext(self.voltage_signals_object.acquisition_object.aquisition_name)[0]
                self.vis_stim_slow_storage_path=self.voltage_signals_object.acquisition_object.slow_storage_all_paths['visual stim']
                self.path_managing()
                self.check_vis_stim_stimuli_in_database()
                
                
                self.load_full_processed_signals()
                if not self.all_final_signals and self.correct_voltages:
                    self.align_daq_prairie_signals(plot=False)

                    

                    self.clip_voltages_to_movie_length()
                    # this processes led and acq trigger in aligna daq and paririe voltages clipped to the end of thwe movie (missing the aqtrigger end)
                    # seems to be stabe and easy
                    module_logger.info('Detecting trsmisition in unaligne signals')

                    self.proces_synch_signals() 
                    # this processesthe opto tirgger(signal if LED) and pockels (if 2p) on the movie clipped 
                    # seems to be stabe and easy
                    self.proces_opto_signals()
                    # this id pproblematis as the optodrift function uses final signals instead of signlas 
                    # self.proces_spont_visual_signals()
                    self.load_acquisition_info()
                    #this is to align the voltage signals with the movie based onthe lED artifcat and then redo the transitions
                    module_logger.info('aligning voltages to movie')

                    self.align_signals_based_on_LED()
                    module_logger.info('Detecting trsmisition inLED aligned signals')

                    # self.proces_synch_signals(aligned=True)
                    self.proces_opto_signals(aligned=True)
                    # self.proces_spont_visual_signals(aligned=True)
                    module_logger.info('This downsamples the alignes transition to movie')

                    self.downsample_transitions_times()
                    module_logger.info('Clipping voltages to led')

                    self.clip_all_signal_to_LED()
                    module_logger.info('Detecting trsmisition in LED clipped signals')

                    # self.proces_synch_signals(led_clipped=True)
                    self.proces_opto_signals(led_clipped=True)
                    self.proces_spont_visual_signals(led_clipped=True)
                    self.save_full_processed_signals()
                    module_logger.info('to check transition look at self.signal_transitions')

                self.process_locomotion() # to change kept to not break result analysis
                self.process_visstim_signal()


                
                
                self.process_all_signals(self.vis_stim_protocol)
                # self.plotting_paradigm_transitions()
                # self.plotting_grating_transitions()
            else:
                self.acquisition_name=self.voltage_signals_object.acq_temp_name
                self.vis_stim_slow_storage_path=self.voltage_signals_object.temporary_folder
                self.path_managing()
                self.process_visstim_signal()
                self.process_locomotion()
                # self.process_LED()
                # self.process_photottrigger()
                # self.process_optopokels()
                # self.process_startend()
                
    def check_if_LED_to_align(self):
        
        self.correct_voltages=True
        # plt.rcParams['axes.prop_cycle'] = plt.rcParamsDefault['axes.prop_cycle']

        # f,ax=plt.subplots(2,sharex=True)

        # for sig in self.all_signals['Prairie'].keys():
        #     if 'Time' not in sig:
        #         ax[0].plot(self.all_signals['Prairie'][sig],label=sig)
        #         ax[0].legend()
                
        # for sig in self.all_signals['Daq'].keys():
        #     if 'Time' not in sig:
        #         ax[1].plot(self.all_signals['Daq'][sig],label=sig)
        #         ax[1].legend()
                
        # plt.show(block = False)
        # plt.pause(1)
       
        raw_fluorescence_threshold = int(input('Shall I correct Voltages?: Yes:1, No:0 \n'))
        
        if raw_fluorescence_threshold==0:
            self.correct_voltages=False


        
  
            
    def path_managing(self):
        
        filenames_suffixes=['_transitions_indexes.pkl', 
         '_drifting_grating_indexes_on.pkl', 
         '_drifting_grating_indexes_off.pkl',
         '_drifting_grating_sliced_indexes_on.pkl',
         '_drifting_grating_sliced_indexes_off.pkl', 
         '_drifting_grating_blank_sweeps_indexes_on.pkl',
         '_drifting_grating_blank_sweeps_indexes_off.pkl',
         '_drifting_grating_sliced_blank_sweeps_indexes_on.pkl',
         '_drifting_grating_sliced_blank_sweeps_indexes_off.pkl', 
         '_static_grating_indexes_even.pkl', 
         '_static_grating_indexes_odd.pkl',
         '_static_grating_sliced_indexes_even.pkl',
         '_static_grating_sliced_indexes_odd.pkl', 
         '_natural_images_indexes_even.pkl', 
         '_natural_images_indexes_odd.pkl',
         '_natural_images_sliced_indexes_even.pkl',
         '_natural_images_sliced_indexes_odd.pkl', 
         '_movie_one_frame_indexes.pkl', 
         '_movie_one_frame_sliced_indexes.pkl',
         '_movie_two_frame_indexes.pkl', 
         '_movie_two_frame_sliced_indexes.pkl',
         '_movie_three_frame_indexes.pkl', 
         '_movie_three_frame_sliced_indexes.pkl',
         '_all_aligned_signals.pkl',
         '_voltage_signals_transitions.pkl',
         '_optodrift_info.pkl',
         '_movie_one_trial_indexes.pkl', 
         '_movie_two_trial_indexes.pkl', 
         '_movie_three_both_trial_indexes.pkl', 

         ]
        indexes_full_file_names=[self.acquisition_name+i for i in filenames_suffixes]
        self.indexes_full_file_paths_to_save=[os.path.join(self.vis_stim_slow_storage_path,i) for i in indexes_full_file_names]
        
        self.all_final_signals_datapath=os.path.join(self.voltage_signals_object.acquisition_object.slow_storage_all_paths['planes'],indexes_full_file_names[23]) 
        self.all_final_signals_transitions_datapath=os.path.join(self.voltage_signals_object.acquisition_object.slow_storage_all_paths['planes'],indexes_full_file_names[24]) 
        self.optodrift_info_datapath=os.path.join(self.voltage_signals_object.acquisition_object.slow_storage_all_paths['planes'],indexes_full_file_names[25]) 

        


    def process_all_signals(self, vis_stim_protocol=None):
        
        if vis_stim_protocol:
            protocol=vis_stim_protocol
        else:
            protocol=self.vis_stim_protocol

        
        if protocol :  
            
            if 'Allen' in protocol:   
                self.vis_stim_protocol=protocol

                self.load_indexes_from_file()
                self.process_allen_paradigms()
               
                if protocol=='AllenA':
                  self.process_allenA_signals()
               
                if protocol=='AllenB':
                   self.process_allenB_signals()
                   
                if protocol=='AllenC':
                   self.process_allenC_signals()
               
            if protocol=='Mistmatch':
               self.load_indexes_from_file()
               self.process_mistmatch_signals()
               
            if protocol=='Habituation':
               self.load_indexes_from_file()
               self.process_habituation_signals()
        
        
        
#%% LED ALIGNMENT AND CLIPPING 

    def align_daq_prairie_signals(self,plot=False):
        module_logger.info('Aligning prairiw ad daq signals')

        plt.close('all')
        alldelays={}
        corrected_daq_signals={}
        orderedsignals=['LED','AcqTrig','VisStim', 'PhotoTrig', 'PhotoStim', 'Locomotion']
        if not set(['LED','AcqTrig','VisStim', 'PhotoTrig', 'PhotoStim', 'Locomotion'])==set(list(self.all_signals['Prairie'].keys())[:-1]):
            orderedsignals=list(reversed(list(self.all_signals['Prairie'].keys())[:-1]))
            if 'LED' in orderedsignals:
                orderedsignals.remove('LED')
                orderedsignals.insert(0, 'LED')
            

        for sig in orderedsignals: 
            
            
            self.all_signals['Daq'][sig]= self.all_signals['Daq'][sig].astype('float32')
            self.all_signals['Prairie'][sig]= self.all_signals['Prairie'][sig].astype('float32')
            
            daq_sig=sg.medfilt(np.squeeze(self.all_signals['Daq'][sig]), kernel_size=1)
            prairie_sig=sg.medfilt(np.squeeze(self.all_signals['Prairie'][sig]), kernel_size=1)
            if plot:
                f,ax=plt.subplots(2)
                ax[0].plot(daq_sig,'r',label='daq')
                ax[0].plot(prairie_sig,'c',label='prairie')
                ax[0].legend()
                plt.show()

           
            if sig=='LED':
                
                transx=np.argwhere(np.diff(daq_sig)<-2).flatten()
                transy=np.argwhere(np.diff(prairie_sig)<-2).flatten()
                if transx.any() and transy.any():
                    scaling_factor = (transy[1]-transy[0])/(transx[1]-transx[0])
                    shift = np.round(transy[0] - (transx[0]*scaling_factor))
                    print(scaling_factor, shift)
                    volt_delay= int(abs(shift))
                    corrected_daq_signals[sig]=self.all_signals['Daq'][sig][volt_delay:].reset_index(drop=True)

                else:
                    scaling_factor=np.nan
                    shift=np.nan
                    volt_delay=np.nan
                    corrected_daq_signals[sig]=self.all_signals['Daq'][sig]
                    print('signlas not aligned')

                    
                       

                alldelays[sig]=[volt_delay,scaling_factor]
               
            else:
                try:
                    mode_delay=int(mode(np.array(list(zip(*list(alldelays.values())))[0]))[0] ) 
                    corrected_daq_signals[sig]=self.all_signals['Daq'][sig][mode_delay:].reset_index(drop=True)

                except ValueError as e:
                    mode_delay=np.nan
                    corrected_daq_signals[sig]=self.all_signals['Daq'][sig]
                    print('signlas not aligned')


                        
                alldelays[sig]=[mode_delay,1]
                print(scaling_factor, shift, 'locomotion')

            
                
            if plot:

                ax[1].plot(corrected_daq_signals[sig],'r',label='daq')
                ax[1].plot(prairie_sig,'c',label='prairie')
                ax[1].legend()
                plt.show()

        self.all_signals['Corrected_daq']= corrected_daq_signals
        self.all_signals['Corrected_daq_shifts']=alldelays

    def clip_voltages_to_movie_length(self): # this remove the extra voltage after acq trigger end by aligning timestamps of movie and volatge
        module_logger.info('Clipping voltages to movie lengths')

 
        timestamps_video_milisecond=np.array(self.voltage_signals_object.acquisition_object.metadata_object.timestamps['Plane1'])*1000
        timestamps_prairie_milisecond=self.all_signals['Prairie']['LED'].index.values
        timestamps_daq_milisecond= self.all_signals['Corrected_daq']['LED'].index.values
        timestamps_video_milisecond_last=timestamps_video_milisecond[-1]
    
        # clipping praire signla sto video lenght
        timestamp_prairie_first_index=0
        timestamp_prairie_last_index=np.abs(timestamps_prairie_milisecond -timestamps_video_milisecond_last).argmin()
        self.all_signals['Prairie_movie_length_clipped']={k:v[timestamp_prairie_first_index:timestamp_prairie_last_index+1] for k,v in self.all_signals['Prairie'].items()}
    
        # clipping daq signla sto video lenght
        timestamp_daq_first_index=0
        timestamp_daq_last_index=np.abs(timestamps_daq_milisecond -timestamps_video_milisecond_last).argmin()       
        self.all_signals['Corrected_daq_movie_length_clipped']={k:v[timestamp_daq_first_index:timestamp_daq_last_index+1] for k,v in self.all_signals['Corrected_daq'].items()}
        self.all_signals['Movie_length_clipping_index']={'Prairie':timestamp_prairie_last_index, 'daq':timestamp_daq_last_index}
        
    def load_acquisition_info(self):
        
        self.calcium_datasets={key:values for key,values in   self.voltage_signals_object.acquisition_object.all_datasets.items() if 'Green' in key }
        self.calcium_datasets[list(self.calcium_datasets.keys())[0]].bidishift_object.create_output_names_if_dont_exist()     
        self.mean_movie_path =self.calcium_datasets[list(self.calcium_datasets.keys())[0]].bidishift_object.mean_movie_path

        if not os.path.isfile(self.mean_movie_path):

            movie = cm.load(glob.glob(str(self.voltage_signals_object.acquisition_object.database_acq_raw_path / 'Ch2Green'/'plane1'/"**Ch2**.tif")))
            self.meanmov=movie.mean(axis=(1, 2)) 
            np.save(self.mean_movie_path,self.meanmov)
            del(movie)
        else:
            self.meanmov=np.load(self.mean_movie_path)
            # if len( self.meanmov)
        
        self.start_frame, self.end_frame=self.voltage_signals_object.acquisition_object.all_datasets[list(self.voltage_signals_object.acquisition_object.all_datasets.keys())[0]].bidishift_object.load_LED_tips()
        self.all_planes_timestamps= copy.deepcopy(self.voltage_signals_object.acquisition_object.metadata_object.timestamps)
        self.all_planes_clipped_timestamps=copy.deepcopy(self.all_planes_timestamps)
        if self.start_frame or self.end_frame:
            for key,val in  self.all_planes_timestamps.items():
                all_planes_timestamps_LED_clipped=np.array(val[self.start_frame:self.end_frame])
                self.all_planes_clipped_timestamps[key]=all_planes_timestamps_LED_clipped
                self.clipped_timestamps=True
        else:
            self.all_planes_clipped_timestamps=self.all_planes_timestamps
            self.clipped_timestamps=False
            
    def align_signals_based_on_LED(self):    
        module_logger.info('Aligning movie and voltages based on LED')

        
        transitions, result, scaling_factors, shifts, inverted_shifts=self.get_shifts_from_LED_alignment()
        
        self.all_signals['LED_alignment_info']= {'transitions':transitions, 'inverted_transitions':result, 'scaling_factors':scaling_factors, 'shifts':shifts, 'inverted_shits':inverted_shifts}
        
        shift_mean= np.round(np.mean(list(inverted_shifts['begining'].values())))
        shift=int(shift_mean)
        scaling_factor=np.mean(list(scaling_factors.values()))
            
        self.all_signals['Prairie_movie_length_clipped_aligned']={}
        self.all_signals['Corrected_daq_movie_length_clipped_aligned']={}
        for signal,df in self.all_signals['Prairie_movie_length_clipped'].items():
    
            timestamps,sig=self.shift_scale_voltage_trace(df.values.flatten(),shift,scaling_factor)
            if 'ocomotion' not in signal and 'Time' not in signal:
    
                corrected_signal,_,_,corrections=self.correct_voltage_split_transitions(sig,kernel_size=1)
            else:
                corrected_signal=sig
                corrections=[]
                
            signal_dataframe=pd.DataFrame(data=corrected_signal,    
                          index=timestamps)
    
            self.all_signals['Prairie_movie_length_clipped_aligned'][signal]=signal_dataframe
            self.all_signals['Prairie_movie_length_clipped_aligned'][signal+'_corrections']=corrections
    
            if signal in self.all_signals['Corrected_daq_movie_length_clipped'].keys():
                timestamps,sig=self.shift_scale_voltage_trace(self.all_signals['Corrected_daq_movie_length_clipped'][signal].values.flatten(),shift,scaling_factor)
                if 'ocomotion' not in signal and 'Time' not in signal:
        
                    corrected_signal,_,_,corrections=self.correct_voltage_split_transitions(sig,kernel_size=1)
                else:
                    corrected_signal=sig
                    corrections=[]
                    
                signal_dataframe=pd.DataFrame(data=corrected_signal,    
                              index=timestamps)
                self.all_signals['Corrected_daq_movie_length_clipped_aligned'][signal]=signal_dataframe
                self.all_signals['Corrected_daq_movie_length_clipped_aligned'][signal+'_corrections']=corrections    

    def get_shifts_from_LED_alignment(self, threshold=0.5,plot=False):
        # tthis is the fundamental function to align the LED voltage signal to the LED movie artifact
        #the alignment loos for the led transition in both signalk with split transtiinm correction
        manualcorrect=False
        not_led=False

        led_prairie=self.all_signals['Prairie_movie_length_clipped']['LED'].values
        mean_mov=sg.medfilt(np.squeeze(self.meanmov), kernel_size=1)
        # remove middle of the movie in case of led artifacts
        if manualcorrect:
            mean_mov[200:-200]=mean_mov[0]
        
        prairie_LED=sg.medfilt(np.squeeze(led_prairie), kernel_size=1)
        timestamps_video_milisecond=np.array(self.all_planes_timestamps['Plane1'])*1000
        if len(mean_mov)-len(timestamps_video_milisecond)==1:
            mean_mov=mean_mov[:-1]
        elif abs(len(mean_mov)-len(timestamps_video_milisecond))>1:
            module_logger.info('somthing wrong with movie length')
        else:
            pass

        #set the signlas betwen 0 and one
        mov=(mean_mov-np.min(mean_mov))/(np.max(mean_mov)-np.min(mean_mov))
        sig=(prairie_LED-np.min(prairie_LED))/(np.max(prairie_LED)-np.min(prairie_LED))
        split_corrected_movie,a,aa,corrections_mov=self.correct_voltage_split_transitions(mov,kernel_size=1)
        split_corrected_led,b,bb,corrections_vol=self.correct_voltage_split_transitions(sig,kernel_size=1)
        
        # self.check_split_transitions(mov,[split_corrected_movie,a,aa,corrections_mov])
        # self.check_split_transitions(sig, [split_corrected_led,b,bb,corrections_vol])

    
        transitions={'movie':{'up':'','down':''},'led':{'up':'','down':''} }
        tracenames=['movie','led']
        timestamps=[timestamps_video_milisecond,self.all_signals['Prairie_movie_length_clipped']['Time'].values.flatten()]
        
       
        for i,trace in enumerate([split_corrected_movie,split_corrected_led]):
            for thr in np.array([-1,1])*threshold:
                if thr==-threshold:
                    tr='down'
                    transitions[tracenames[i]][tr]=np.argwhere(np.diff(trace,prepend=mov[0])<thr).flatten()
                else:
                    tr='up'
                    transitions[tracenames[i]][tr]=np.argwhere(np.diff(trace,prepend=mov[0])>thr).flatten()


        # if there i no LED flash but ther is signal just align with 0 shift by selecting same timestampos as the signal
        if not transitions['movie']['up'].any() or not transitions['movie']['down'].any():   
            transitions['movie']['up']=np.array([np.argmin(np.abs(timestamps_video_milisecond- transitions['led']['up'][0])),np.argmin(np.abs(timestamps_video_milisecond- transitions['led']['up'][1]))])
            transitions['movie']['down']=np.array([np.argmin(np.abs(timestamps_video_milisecond- transitions['led']['down'][0])),np.argmin(np.abs(timestamps_video_milisecond- transitions['led']['down'][1]))])
            not_led=True

        if plot:
            f,ax=plt.subplots(2,sharex=True) 
            f.suptitle('THIS IS TO CHECK LED SIGNAL-MOVIE ALIGNMENT')
            titles=['Raw','SplitCorrected']
            for j,trac in enumerate([[mov,sig],[split_corrected_movie,split_corrected_led]]):
                ax[j].plot(timestamps_video_milisecond,trac[0],label='movie')
                ax[j].plot(self.all_signals['Prairie_movie_length_clipped']['Time'],trac[1],label='voltage')
                ax[j].legend()
                ax[j].set_title(titles[j])
                for i,trace in enumerate(trac):
                    for tran in ['up','down']:
                            ax[j].plot(timestamps[i][transitions[tracenames[i]][tran]],trace[transitions[tracenames[i]][tran]],'o')
            plt.show()


        transitions['movies_timestamps']={}
        for k,v in transitions['movie'].items():
            transitions['movies_timestamps'][k]=np.array([timestamps_video_milisecond[v][0], timestamps_video_milisecond[v][1] ])
            
      
   
        result = {}
        for k1, subdict in transitions.items():
            for k2, v in subdict.items():
                result.setdefault(k2, {})[k1] = v
   
        scaling_factors={}
        shifts={}
        for i in result.keys():
            shifts[i]={}
            scaling_factors[i]=(result[i]['led'][-1]-result[i]['led'][0])/(result[i]['movies_timestamps'][-1]-result[i]['movies_timestamps'][0])            
            shifts[i]['begining']=np.round(result[i]['led'][0] - (result[i]['movies_timestamps'][0]*scaling_factors[i]))
            shifts[i]['end']=np.round(result[i]['led'][1] - (result[i]['movies_timestamps'][1]*scaling_factors[i]))
            
            if not_led:
                shifts[i]['begining']=0
                shifts[i]['end']=0
   
        inverted_shifts = {}
        for k1, subdict in shifts.items():
            for k2, v in subdict.items():
                inverted_shifts.setdefault(k2, {})[k1] = v
   
        shift_up= list(inverted_shifts['begining'].values())[0]
        shift_end= list(inverted_shifts['begining'].values())[1]
        shift_mean= np.round(np.mean(list(inverted_shifts['begining'].values())))
           
        plt.close('all')
    
        sh=['shift_up','shift_end','shift_mean']
        for i,shift in enumerate([shift_up,shift_end,shift_mean]):
            shift=int(shift)
            scaling_factor=np.mean(list(scaling_factors.values()))
             
            shifted_voltage_timestamps, shifted_scaled_trace= self.shift_scale_voltage_trace(split_corrected_led,shift,scaling_factor)
            
            shifted_scaled_trace,_,_,corrections=self.correct_voltage_split_transitions(shifted_scaled_trace)
            
        if plot:

            for i,shift in enumerate([shift_up,shift_end,shift_mean]):
                shift=int(shift)
                scaling_factor=np.mean(list(scaling_factors.values()))
                 
                shifted_voltage_timestamps, shifted_scaled_trace= self.shift_scale_voltage_trace(split_corrected_led,shift,scaling_factor)
                
                shifted_scaled_trace,_,_,corrections=self.correct_voltage_split_transitions(shifted_scaled_trace)
        
                f,ax=plt.subplots(2,sharex=True)
                ax[0].plot( timestamps_video_milisecond,mov,label='Mov')
                ax[0].plot( self.all_signals['Prairie_movie_length_clipped']['Time'],sig,label='LED')
                ax[0].plot( timestamps_video_milisecond,split_corrected_movie,label='Split_Mov')
                ax[0].plot(  self.all_signals['Prairie_movie_length_clipped']['Time'],split_corrected_led,label='Split_LED')
        
                
                ax[1].plot( timestamps_video_milisecond,split_corrected_movie,label='Split_Mov')
                ax[1].plot( shifted_voltage_timestamps,shifted_scaled_trace,label='Shifted Scaled LED ')
                ax[0].legend()
                ax[1].legend()
                f.suptitle(f'{sh[i]}')
                
            plt.show()

   
        tracenames=['shifted_led']
        for i,trace in enumerate([shifted_scaled_trace]):
            transitions[tracenames[i]]={}
            for thr in np.array([-1,1])*threshold:
                if thr==-threshold:
                    tr='down'
                    transitions[tracenames[i]][tr]=np.argwhere(np.diff(trace,prepend=mov[0])<thr).flatten()
                else:
                    tr='up'
                    transitions[tracenames[i]][tr]=np.argwhere(np.diff(trace,prepend=mov[0])>thr).flatten()
   
        result = {}
        for k1, subdict in transitions.items():
            for k2, v in subdict.items():
                result.setdefault(k2, {})[k1] = v
                
        return transitions, result, scaling_factors, shifts, inverted_shifts

    def shift_scale_voltage_trace(self, trace, shift, scaling_factor):
       
        if scaling_factor==1:
            scaled_trace=trace
        else:
            x_original=np.arange(len(trace))*scaling_factor
            x_final=np.arange(0,int(len(trace)*scaling_factor))
            f = interpolate.interp1d(x_original,trace)
            scaled_trace = f(x_final) 
   
        if shift>0:
            shifted_scaled_trace = scaled_trace[shift:]
        elif shift<0:
            pad_val = scaled_trace[0]
            temp_padding = np.ones(abs(shift))*pad_val
            shifted_scaled_trace =np.concatenate([temp_padding,scaled_trace])
        else:
            shifted_scaled_trace=scaled_trace
            
        shifted_voltage_timestamps=np.arange(len(shifted_scaled_trace))
        
        return shifted_voltage_timestamps, shifted_scaled_trace

                
    def downsample_transitions_times(self,signal_rate=1000 ):
        timestamps_full_video_milisecond=np.array(self.all_planes_timestamps['Plane1'])*signal_rate
        keys_to_downsample=[k for k in self.signal_transitions.keys()  if 'aligned' in k]
        
        test_dict={}
        for sig in keys_to_downsample:
            v=  self.signal_transitions[sig]
            
            test_dict[sig+'_downsampled']={}

            for record,vv in v.items():
                test_dict[sig+'_downsampled'][record]={}
                for tran_type, vvv in vv.items():
                    test_dict[sig+'_downsampled'][record][tran_type]=[]

                    for i,tran in enumerate(vvv):
                        
                        test_dict[sig+'_downsampled'][record][tran_type].append(np.abs( np.array(timestamps_full_video_milisecond -tran)).argmin())
                    
        self.signal_transitions.update(test_dict)
        

    def clip_all_signal_to_LED(self):
      
        self.all_planes_clipped_timestamps_shifted={}
        for k in self.all_planes_clipped_timestamps.keys():
            self.all_planes_clipped_timestamps_shifted[k]=np.array(self.all_planes_clipped_timestamps[k])-self.all_planes_clipped_timestamps[k][0]
            
        keys_to_shift=[k for k in self.signal_transitions.keys()  if 'downsampled' in k]    
        if not self.start_frame:
            self.start_frame=0
        test_dict={}
        if keys_to_shift:
            for k in keys_to_shift:
               test_dict[k+'_LEDshifted']={}
               v=self.signal_transitions[k]
               for record,vv in v.items():
                    test_dict[k+'_LEDshifted'][record]={}
                    for tran_type, vvv in vv.items():
                        test_dict[k+'_LEDshifted'][record][tran_type]=[]
        
                        for i,tran in enumerate(vvv):
                        
                            test_dict[k+'_LEDshifted'][record][tran_type].append(tran-self.start_frame)
                
            self.signal_transitions.update(test_dict)
            
            self.led_clipped_signal_start=np.floor(self.all_planes_timestamps['Plane1'][self.start_frame]*1000).astype(int)
            self.led_clipped_signal_end=np.ceil(self.all_planes_timestamps['Plane1'][self.end_frame-1]*1000).astype(int)
            
            self.all_signals['Prairie_movie_length_clipped_aligned_LEDshifted']={}
            self.all_signals['Corrected_daq_movie_length_clipped_aligned_LEDshifted']={}
    
            for sig,v in self.all_signals['Prairie_movie_length_clipped_aligned'].items():
                if 'corrections' not in sig:
                    self.all_signals['Prairie_movie_length_clipped_aligned_LEDshifted'][sig]=pd.DataFrame(v.values[self.led_clipped_signal_start:self.led_clipped_signal_end])
                
            
                    if sig in self.all_signals['Corrected_daq_movie_length_clipped_aligned']:
                        self.all_signals['Corrected_daq_movie_length_clipped_aligned_LEDshifted'][sig]=pd.DataFrame(self.all_signals['Corrected_daq_movie_length_clipped_aligned'][sig].values[self.led_clipped_signal_start:self.led_clipped_signal_end])


    def load_full_processed_signals(self):
        

        if os.path.isfile(self.all_final_signals_datapath):
            with open(self.all_final_signals_datapath, 'rb') as fp:
                self.all_final_signals = pickle.load(fp)
                
        if os.path.isfile(self.all_final_signals_transitions_datapath):
            with open(self.all_final_signals_transitions_datapath, 'rb') as fp:
                self.signal_transitions = pickle.load(fp)
                
                
        self.load_optodrift_info()
                
   

    def save_full_processed_signals(self):
        #to save correcetd aligne and trans
        
        
        clustering_keys=[k for k in self.signal_transitions.keys() if '_' not in k]

        clustered_dict={}
        for k in clustering_keys:
            clustered_dict[k]={}
            for key, value in self.signal_transitions.items():
                if k in key:
                    if k==key:
                        key='Raw'
                    else:

                        clustered_dict[k][key]=value
                    
        self.signal_transitions=clustered_dict
 
        
        # simplify the sigbnals dictionry by combinen all dataframes into a single one with columns as signals names

        combined_df_prairie=pd.DataFrame()
        combined_df_daq=pd.DataFrame()
        for k,v in self.all_signals['Prairie'].items(): combined_df_prairie[k]=v 
        for k,v in self.all_signals['Daq'].items(): combined_df_daq[k]=v 
        
        combined_df_daq_cor=pd.DataFrame()
        for k,v in self.all_signals['Corrected_daq'].items(): combined_df_daq_cor[k]=v 
        
        combined_df_prairie_clip=pd.DataFrame()
        combined_df_daq_clip=pd.DataFrame()
        for k,v in self.all_signals['Prairie_movie_length_clipped'].items(): combined_df_prairie_clip[k]=v 
        for k,v in self.all_signals['Corrected_daq_movie_length_clipped'].items(): combined_df_daq_clip[k]=v 

        combined_df_prairie_ali=pd.DataFrame()
        combined_df_daq_clip_ali=pd.DataFrame()
        for k,v in self.all_signals['Prairie_movie_length_clipped_aligned'].items() :
            if 'correction' not in k : 
                combined_df_prairie_ali[k]=v 
        for k,v in self.all_signals['Corrected_daq_movie_length_clipped_aligned'].items() :
            if 'correction' not in k : 
                combined_df_daq_clip_ali[k]=v 
        
 

        combined_df_prairie_ledclip=pd.DataFrame()
        combined_df_daq_clip_ledclip=pd.DataFrame()
        for k,v in self.all_signals['Prairie_movie_length_clipped_aligned_LEDshifted'].items(): combined_df_prairie_ledclip[k]=v 
        for k,v in self.all_signals['Corrected_daq_movie_length_clipped_aligned_LEDshifted'].items(): combined_df_daq_clip_ledclip[k]=v 

        
        self.all_final_signals={'Prairie':{'Raw':{'corrections':'',
                                                  'traces':combined_df_prairie},
                                           'Movie_length_clipped':{'corrections':self.all_signals['Movie_length_clipping_index']['Prairie'],
                                                                   'traces':combined_df_prairie_clip},
                                           'LED_aligned':{'corrections':{'LED_alignment_info': self.all_signals['LED_alignment_info'],'split_corrections':{key: v for key,v in self.all_signals['Prairie_movie_length_clipped_aligned'].items()  if 'correction' in key}},
                                                          'traces':combined_df_prairie_ali},
                                           'LED_clipped':{'corrections':{'movie_LED':[self.start_frame, self.end_frame],'signal_LED':[self.led_clipped_signal_start,self.led_clipped_signal_end]},
                                                          'traces':combined_df_prairie_ledclip},
                                           
                                           },
                                'daq':{'Raw':{'corrections':'',
                                              'traces':combined_df_daq},
                                       'Prairie_aligned':{'corrections':self.all_signals['Corrected_daq_shifts'],
                                                          'traces':combined_df_daq_cor},
                                       'Movie_length_clipped':{'corrections':self.all_signals['Movie_length_clipping_index']['daq'],
                                                               'traces':combined_df_daq_clip},
                                       'LED_aligned':{'corrections':{'LED_alignment_info': 'Used Prairie Calculated','split_corrections':{key: v for key,v in self.all_signals['Corrected_daq_movie_length_clipped_aligned'].items()  if 'correction' in key}},
                                                      'traces':combined_df_daq_clip_ali},
                                       'LED_clipped':{'corrections':'Used Prairie Calculated',
                                                      'traces':combined_df_daq_clip_ledclip},
                                       
                                               }}
        
        
        
           
        if not os.path.isfile(self.all_final_signals_transitions_datapath):
            with open(self.all_final_signals_transitions_datapath, 'wb') as fp:
                pickle.dump(self.signal_transitions, fp)
        
        
        if not os.path.isfile(self.all_final_signals_datapath):
            with open(self.all_final_signals_datapath, 'wb') as fp:
                pickle.dump(self.all_final_signals, fp)
                
        
       
        
#%% utility to reorganize
    
    def get_specific_signal(self, selected_signal_name):
        
        signal_array_dict={}
        for signal_name, signals in self.all_signals.items():
            signal_array_dict[signal_name]={}
            for signal, df in signals.items():
                if selected_signal_name in signal and isinstance(df, pd.DataFrame):
                    signal_array_dict[signal_name][signal]=df.T.to_numpy().squeeze()
                
        return signal_array_dict

    
    
    def process_signal(self, array_dict, function='raw', round_factor=1, new_function=None):
        
        process_dict={}
        raw=lambda x:x
        deriv=lambda x:np.diff(x,prepend=x[0] )
        rectified=lambda x:np.absolute(x)
        rounded=lambda x:np.round(x,round_factor)
        median=lambda x:sg.medfilt(x, kernel_size=29)
        functions={'raw':raw,
                   'deriv':deriv,
                   'rectified':rectified,
                   'rounded':rounded,
                   'median': median
                       }
        
        for acq_name, signal in array_dict.items():
            process_dict[acq_name]={}
            for sig, array in signal.items():
                if array.any():
                    process_dict[acq_name][sig]=functions[function](array)
                else:
                    process_dict[acq_name][sig]=array

        return process_dict
    
    
    
    
    def proces_synch_signals(self,aligned=False, led_clipped=False):
        for signal_name in ['AcqTrig', 'LED']:
            self.extract_transitions_from_signal(signal_name,aligned=aligned, led_clipped=led_clipped)
            
    def proces_opto_signals(self,aligned=False, led_clipped=False):
        for signal_name in ['PhotoTrig', 'PhotoStim']:
            self.extract_transitions_from_signal(signal_name,aligned=aligned, led_clipped=led_clipped)
            # this i have to make general enough to detect all up and down transition, should work fine as it is but review
            
            
            
    def proces_spont_visual_signals(self,aligned=False, led_clipped=False):
        '''
        HERE IMPORTNAT I HAVE TO MANUALLY ADD THE VIS STIM PROTOCOL FTO THE SQL LITE DATABASE 
        VisualStimulations_Table add 7 to column (GIve error)
        havbe to do it through script 
        TESTOTHERS/PROCESSINGSCRIP/DATABASESCRIPT
        
        
        query_mice_cage_update="""
                        UPDATE VisualStimulations_table
                        SET VisualStimulationProtocolID=7
                        WHERE ID IN (137,138,139)
                    """        
        params=()   
        self.voltage_signals_object.acquisition_object.mouse_object.Database_ref.arbitrary_updating_record(query_mice_cage_update, params, commit=True)
        self.voltage_signals_object.acquisition_object.get_all_database_info()
        self.check_vis_stim_stimuli_in_database()

        '''

      
        if self.vis_stim_protocol==None:
            for signal_name in ['VisStim']:
                self.extract_transitions_from_signal(signal_name,aligned=aligned, led_clipped=led_clipped)
                #here i nedd t o create a switcher for differnet vis stim paradigm
        elif self.vis_stim_protocol=='OptoDrift':
            for signal_name in ['VisStim']:

                self.extract_transitions_optodrift(signal_name,aligned=aligned, led_clipped=led_clipped,plot=False)

        
        
    
                
        
    def extract_transitions_from_signal(self,signal_name,diff_thres=0.1, aligned=False, led_clipped=False,plot=False):
        
        
        if signal_name in self.all_signals['Prairie_movie_length_clipped'].keys():
            # if sig is acq trigger the ending should be clipped
            
            if aligned: # this is for LED aligned moves
                sig_daq=self.all_signals['Corrected_daq_movie_length_clipped_aligned'][signal_name].values.flatten()
                sig_prairie=self.all_signals['Prairie_movie_length_clipped_aligned'][signal_name].values.flatten()
                signal_name=signal_name+'_aligned'
                
            elif led_clipped: # this is for LED clipped moves
                sig_daq=self.all_signals['Corrected_daq_movie_length_clipped_aligned_LEDshifted'][signal_name].values.flatten()
                sig_prairie=self.all_signals['Prairie_movie_length_clipped_aligned_LEDshifted'][signal_name].values.flatten()
                signal_name=signal_name+'_aligned_LEDshifted'
                
            elif 'AcqTrig' in self.all_signals['Prairie_movie_length_clipped'].keys(): # this is for daq_prairie aligned and tail clipped
                sig_daq=self.all_signals['Corrected_daq_movie_length_clipped'][signal_name].values.flatten()
                sig_prairie=self.all_signals['Prairie_movie_length_clipped'][signal_name].values.flatten()
            else:
                sig_daq=self.all_signals['Corrected_daq_movie_length_clipped'][signal_name].values.flatten()
                sig_prairie=self.all_signals['Prairie_movie_length_clipped'][signal_name].values.flatten()
       
            sig_prairie=(sig_prairie-np.min(sig_prairie))/(np.max(sig_prairie)-np.min(sig_prairie))
            sig_daq=(sig_daq-np.min(sig_daq))/(np.max(sig_daq)-np.min(sig_daq))
            
            if plot:
                f,ax=plt.subplots()
                ax.plot(sig_prairie,'r',label='prairie')
                ax.plot(sig_daq,'c',label='daq')
                f.suptitle(f'{signal_name}')
                ax.legend()
                plt.show()
    
    
            
            signal_prairie_filtered_rounded_corrected,\
            signal_prairie_diff_filtered_rounded_corrected,\
            signal_prairie_diff_filtered_rounded_corrected_rerounded,\
            signal_prairie_errors_pairs = self.correct_voltage_split_transitions(sig_prairie,kernel_size=1)
            
            signal_daq_filtered_rounded_corrected,\
            signal_daq_diff_filtered_rounded_corrected,\
            signal_daq_diff_filtered_rounded_corrected_rerounded,\
            signal_daq_errors_pairs = self.correct_voltage_split_transitions(sig_daq,kernel_size=1)
    
            
            sigup_pra=np.argwhere(np.diff(signal_prairie_filtered_rounded_corrected)>0.8).flatten()                
            sigup_daq=np.argwhere(np.diff(signal_daq_filtered_rounded_corrected)>0.8).flatten()
            sigdown_pra=np.argwhere(np.diff(signal_prairie_filtered_rounded_corrected)<-0.8).flatten()              
            sigdown_daq=np.argwhere(np.diff(signal_daq_filtered_rounded_corrected)<-0.8).flatten()
    
    
            self.signal_transitions[signal_name]={'Prairie':{'up':sigup_pra,'down':sigdown_pra},'daq':{'up':sigup_daq,'down':sigdown_daq}}
            
            # for acq trig there should be up and down for daq and only down for prairie(prairire doesnt record the upvioltage trigger)
            # for LED there should 2 up and 2 down fro both signals and with a saml diffewrnece of 1 frame maybe)
            # for Photortige and else signal there should 20 up and 20 down fro both signals and with a saml diffewrnece of 1 frame maybe (60 and 60 if blanmk sweep opto )))
            # for Photostim and else signal there should 20*optoreps general 400 at 20hx for both up odown prairire and daq This have artound 5 ms misgalignemn t but is not relevant we dont use them and in movie time the are clumped in dame frames
            # for visstim only applys if no stim whgere ther is onl;y 2 transitions ofr the paradigm if not I had to add the vis stim protocol
            
            
        
        
        
    def process_LED(self):
        self.led_array=self.get_specific_signal('LED')
        self.dfdt_rounded_led =self.process_signal(self.process_signal(self.get_specific_signal('LED'),'rounded'), 'deriv')
        self.dfdt_rounded_led_median =self.process_signal(self.process_signal(self.process_signal(self.get_specific_signal('LED'),'median'),'rounded'), 'deriv')
        
        self.start_transitions=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']<-6.5).flatten()+1

        
    def process_photottrigger(self):
        self.photottrigger_array=self.get_specific_signal('PhotoTrig')
        self.dfdt_rounded_photottrigger =self.process_signal(self.process_signal(self.get_specific_signal('PhotoTrig'),'rounded'), 'deriv')
        self.dfdt_rounded_photottrigger_median =self.process_signal(self.process_signal(self.process_signal(self.get_specific_signal('PhotoTrig'),'median'),'rounded'), 'deriv')

        
    def process_optopokels(self):
        self.process_optopokels_array=self.get_specific_signal('PhotoStim')
        self.dfdt_rounded_process_optopokels =self.process_signal(self.process_signal(self.get_specific_signal('PhotoStim'),'rounded'), 'deriv')
        self.dfdt_rounded_process_optopokels_median =self.process_signal(self.process_signal(self.process_signal(self.get_specific_signal('PhotoStim'),'median'),'rounded'), 'deriv')

        
    def process_startend(self):
        self.startend_array=self.get_specific_signal('AcqTrig')
        self.dfdt_rounded_startend =self.process_signal(self.process_signal(self.get_specific_signal('AcqTrig'),'rounded'), 'deriv')
        self.dfdt_rounded_startend_median =self.process_signal(self.process_signal(self.process_signal(self.get_specific_signal('AcqTrig'),'median'),'rounded'), 'deriv')

            
    def process_locomotion(self):
        self.locomotion_array=self.get_specific_signal('Locomotion')
        self.rectified_speed_array=self.process_signal(self.process_signal(self.get_specific_signal('Locomotion'), 'deriv'), 'rectified')
        self.rectified_acceleration_array=self.process_signal(self.process_signal(self.process_signal(self.get_specific_signal('Locomotion'),'deriv'),'deriv'),'rectified')
        
    def process_visstim_signal(self):
        if 'VisStim' in self.voltage_signals.keys():
            self.visualstim_array=self.get_specific_signal('VisStim')
            self.rounded_vis_stim= self.process_signal( self.visualstim_array, 'rounded')
            self.dfdt_rounded_vis_stim =self.process_signal(self.process_signal(self.visualstim_array,'rounded'), 'deriv')
            self.dfdt_rounded_vis_stim_median =self.process_signal(self.process_signal(self.process_signal(self.get_specific_signal('VisStim'),'median'),'rounded'), 'deriv')

        

    def resample(self, x, factor, kind='linear'):
        n = int(np.ceil(x.size / factor))
        f = interpolate.interp1d(np.linspace(0, 1, x.size), x, kind)
        return f(np.linspace(0, 1, n))       

    def method_to_donwsample_all_signals_for_faster_plotting(self):
        self.all_downsampled_signals={}
        for key, signal in self.voltage_signals:
            self.all_downsampled_signals[key]=self.resample( signal, factor=self.milisecondscale['Prairie'], kind='linear').squeeze()
        
    def check_vis_stim_stimuli_in_database(self):
        if self.voltage_signals_object.acquisition_object.full_database_dictionary:
        
            self.database_VisStimInfo=self.voltage_signals_object.acquisition_object.full_database_dictionary['VisStim']
            if not self.database_VisStimInfo.empty:
                if self.database_VisStimInfo['VisStimProtocol_name'][0]=='Allen Session A Version A':
                    self.vis_stim_protocol='AllenA'
                    self.vis_stim_protocol_version='Version A'
                elif self.database_VisStimInfo['VisStimProtocol_name'][0]=='Allen Session B Version A':
                    self.vis_stim_protocol='AllenB'
                    self.vis_stim_protocol_version='Version A'
                elif self.database_VisStimInfo['VisStimProtocol_name'][0]=='Allen Session C Version A':
                    self.vis_stim_protocol='AllenC'
                    self.vis_stim_protocol_version='Version A'
                elif self.database_VisStimInfo['VisStimProtocol_name'][0]=='OptoDrift20':
                    self.vis_stim_protocol='OptoDrift'

        else:
            self.vis_stim_protocol=None
 
    def update_frame_rates_with_metadata(self, Prairie_frame_rate, daq_frame_rate):
        self.daqstim_voltagerate=daq_frame_rate
        self.Prairie_voltagerate=Prairie_frame_rate
        
        self.frame_rates={'Prairie':Prairie_frame_rate,
                          'Daq':daq_frame_rate,
            }
        
        timesignals=self.get_specific_signal('Time')
        if list(timesignals.values())[0]:
        
            self.second_scale={'Prairie':(1/self.frame_rates['Prairie'])*timesignals['Prairie']['Time'],
                              'Daq':(1/self.frame_rates['Daq'])*timesignals['Prairie']['Time'],
                }
            self.milisecondscale={'Prairie':1000*self.second_scale['Prairie'],
                            'Daq':1000*self.second_scale['Daq'],
              }
            self.minutes_scale={'Prairie':self.second_scale['Prairie']/60,
                              'Daq':self.second_scale['Daq']/60,
                }
    
            self.choose_time_scale('milisecond')

    def choose_time_scale(self, scale)   :
        if scale=='milisecond':
            self.time_scale=self.milisecondscale
              
        elif scale=='sec':
            self.time_scale= self.second_scale
                 
        elif scale=='min':
            self.time_scale=self.minutes_scale
            
            
#%% chandelier opto drifting gratings

    def downsample_optodrift_onsets(self,signal_rate=1000 ):
        self.load_acquisition_info()
        timestamps_full_video_milisecond=np.array(self.all_planes_timestamps['Plane1'])*signal_rate
        
        for stim in range(1,9):
            downsampledarray=np.zeros_like(self.optodrift_info[f'Grat_{stim}']['ArrayFinal'])
            downsampledarrayoff=np.zeros_like(self.optodrift_info[f'Grat_{stim}']['ArrayFinalOffset'])

            for trial in range(20):
                for rep in range(2):
                    # self.optodrift_info[f'Grat_{stim}']['ArrayFinal'][trial,rep]
                    downsampledarray[trial,rep]=np.abs( np.array(timestamps_full_video_milisecond - self.optodrift_info[f'Grat_{stim}']['ArrayFinal'][trial,rep])).argmin()
                    downsampledarrayoff[trial,rep]=np.abs( np.array(timestamps_full_video_milisecond - self.optodrift_info[f'Grat_{stim}']['ArrayFinalOffset'][trial,rep])).argmin()

        
        
            self.optodrift_info[f'Grat_{stim}']['ArrayFinal_downsampled_LED_clipped']=downsampledarray
            self.optodrift_info[f'Grat_{stim}']['ArrayFinalOffset_downsampled_LED_clipped']=downsampledarrayoff

            
            
        downsampledarray=np.zeros_like(self.optodrift_info['Blank']['ArrayFinal'])
        downsampledarrayoff=np.zeros_like(self.optodrift_info['Blank']['ArrayFinal'])

        for sweep in range(4):
            for trial in range(20):
                self.optodrift_info['Blank']['ArrayFinal'][trial,sweep]
                downsampledarray[trial,sweep]=np.abs( np.array(timestamps_full_video_milisecond - self.optodrift_info['Blank']['ArrayFinal'][trial,sweep])).argmin()
                downsampledarrayoff[trial,sweep]=np.abs( np.array(timestamps_full_video_milisecond - self.optodrift_info['Blank']['ArrayFinalOffset'][trial,sweep])).argmin()

        
        
            self.optodrift_info['Blank']['ArrayFinal_downsampled_LED_clipped']=downsampledarray
            self.optodrift_info['Blank']['ArrayFinalOffset_downsampled_LED_clipped']=downsampledarrayoff

            
            
        self.save_optodrift_info()           

    def load_optodrift_info(self):
        
        if os.path.isfile(self.optodrift_info_datapath):
            with open(self.optodrift_info_datapath, 'rb') as fp:
                self.optodrift_info = pickle.load(fp)
                
                
    def save_optodrift_info(self):
         
        if not os.path.isfile(self.optodrift_info_datapath):
            with open(self.optodrift_info_datapath, 'wb') as fp:
                pickle.dump(self.optodrift_info, fp)
                

    def extract_transitions_optodrift(self,signal_name,diff_thres=0.1, aligned=False, led_clipped=False, plot=True):
        #this processes the vis stim optodrift protocl, it is run once on the unaligne signals and save prairie but because i will remove the preled frames is not guseful so it has to be run gaian after finsihnin  alsignklas
        # only after led clipped should be saved
        plt.close('all')
        optoinfo=self.voltage_signals_object.acquisition_object.visstimdict['opto']
        optoinfo['allrandperm']
        optoinfo['all_grating_indexes']
        stimnum=8
        oris=np.linspace(0,360,stimnum+1)[:-1]
        trials=optoinfo['allrandperm'].shape[0]
        trial_voltage=10
        blank_volt=9.5
        blank_volt_a=blank_volt-2
        blank_volt_b=blank_volt-1
        blank_volt_c=blank_volt-3
        orient_volt=np.round(np.linspace(4.6,9,stimnum),1)
        orient_volt_a=orient_volt-2
        orient_volt_b=orient_volt-1
        orient_volt_c=orient_volt-3

        
        self.optodrift_info={}
        for num in range(stimnum):
            self.optodrift_info[f'Grat_{num+1}']={'Ori':oris[num],
                                           'Stimnum':num+1,
                                            'StimCodes':np.array((num+1,num+1+stimnum)),
                                            'FulltrialStart':np.zeros([trials]).astype(int),
                                            'FulltrialEnd':np.zeros([trials]).astype(int),
                                            'ReplStart':np.zeros([trials,2]).astype(int),
                                            'ReplEnd':np.zeros([trials,2]).astype(int),
                                            'ArrayInRep':np.zeros([trials,2]).astype(int),
                                            'ArrayInTrial':np.zeros([trials,2]).astype(int),
                                            'ArrayFinal':np.zeros([trials,2]).astype(int),
                                            'ArrayFinalOffset':np.zeros([trials,2]).astype(int),
                                            'ArrayInRepOffset':np.zeros([trials,2]).astype(int),
                                            'ArrayInTrialOffset':np.zeros([trials,2]).astype(int),
                                                 }
            self.optodrift_info['Blank']={'Stimnum':0,
                                   'StimCodes':np.arange(0+2*stimnum+1,4+2*stimnum+1).astype(int),
                                   'FulltrialStart':np.zeros([trials]).astype(int),
                                   'FulltrialEnd':np.zeros([trials]).astype(int),
                                   'ReplStart':np.zeros([trials,4]).astype(int),
                                   'ReplEnd':np.zeros([trials,4]).astype(int),
                                   'ArrayInRep':np.zeros([trials,4]).astype(int),
                                   'ArrayInTrial':np.zeros([trials,4]).astype(int),
                                   'ArrayFinal':np.zeros([trials,4]).astype(int),
                                   'ArrayFinalOffset':np.zeros([trials,4]).astype(int),
                                   'ArrayInRepOffset':np.zeros([trials,4]).astype(int),
                                   'ArrayInTrialOffset':np.zeros([trials,4]).astype(int),
                                      }
        
        if self.all_final_signals:
            if aligned:
                sig_daq=self.all_final_signals['daq']['LED_aligned']['traces'][signal_name].values.flatten()
                sig_prairie=self.all_final_signals['Prairie']['LED_aligned']['traces'][signal_name].values.flatten()
                signal_name=signal_name+'_aligned'
            elif led_clipped:
                sig_daq=self.all_final_signals['daq']['LED_clipped']['traces'][signal_name].values.flatten()
                sig_prairie=self.all_final_signals['Prairie']['LED_clipped']['traces'][signal_name].values.flatten()
                signal_name=signal_name+'_aligned_LEDshifted'

                
            else:
                sig_daq=self.all_final_signals['daq']['Movie_length_clipped']['traces'][signal_name].values.flatten()
                sig_prairie=self.all_final_signals['Prairie']['Movie_length_clipped']['traces'][signal_name].values.flatten()
            
        else:
            if aligned: # this is for LED aligned moves
                sig_daq=self.all_signals['Corrected_daq_movie_length_clipped_aligned'][signal_name].values.flatten()
                sig_prairie=self.all_signals['Prairie_movie_length_clipped_aligned'][signal_name].values.flatten()
                signal_name=signal_name+'_aligned'
            elif led_clipped: # this is for LED clipped moves
                sig_daq=self.all_signals['Corrected_daq_movie_length_clipped_aligned_LEDshifted'][signal_name].values.flatten()
                sig_prairie=self.all_signals['Prairie_movie_length_clipped_aligned_LEDshifted'][signal_name].values.flatten()
                signal_name=signal_name+'_aligned_LEDshifted'

                
            else: # this is for daq_prairie aligned and tail clipped
                sig_daq=self.all_signals['Corrected_daq_movie_length_clipped'][signal_name].values.flatten()
                sig_prairie=self.all_signals['Prairie_movie_length_clipped'][signal_name].values.flatten()
        
        
        
        
        
        signal_prairie_filtered_rounded_corrected,\
        signal_prairie_diff_filtered_rounded_corrected,\
        signal_prairie_diff_filtered_rounded_corrected_rerounded,\
        signal_prairie_errors_pairs = self.correct_voltage_split_transitions(sig_prairie,kernel_size=1)
        
        signal_daq_filtered_rounded_corrected,\
        signal_daq_diff_filtered_rounded_corrected,\
        signal_daq_diff_filtered_rounded_corrected_rerounded,\
        signal_daq_errors_pairs = self.correct_voltage_split_transitions(sig_daq,kernel_size=1)
 
        if plot:
            f,ax=plt.subplots(2)
            ax[0].plot(sig_prairie,'r')
            ax[0].plot(sig_daq,'c')
            ax[1].plot(signal_prairie_diff_filtered_rounded_corrected,'r')
            ax[1].plot(signal_daq_diff_filtered_rounded_corrected,'c')
        

 
        
        sigup_pra=np.argwhere( abs(signal_prairie_diff_filtered_rounded_corrected-trial_voltage)<=0.6).flatten()               
        sigup_daq=np.argwhere( abs(signal_daq_diff_filtered_rounded_corrected-trial_voltage)<=0.6).flatten() 
        sigdown_pra=np.argwhere( abs(signal_prairie_diff_filtered_rounded_corrected+trial_voltage)<=0.6).flatten()             
        sigdown_daq=np.argwhere( abs(signal_daq_diff_filtered_rounded_corrected+trial_voltage)<=0.6 ).flatten() 
        
        sigup_pra_corrected=np.array([tran for tran in sigup_pra if abs(signal_prairie_filtered_rounded_corrected[tran+1]-trial_voltage)<=0.1])
        sigup_daq_corrected=np.array([tran for tran in sigup_daq if abs(signal_prairie_filtered_rounded_corrected[tran+1]-trial_voltage)<=0.1])
        sigdown_pra_corrected=np.array([tran for tran in sigdown_pra if abs(signal_prairie_filtered_rounded_corrected[tran-1]-trial_voltage)<=0.1])
        sigdown_daq_corrected=np.array([tran for tran in sigdown_daq if abs(signal_prairie_filtered_rounded_corrected[tran-1]-trial_voltage)<=0.1])


        
        
        if plot:
    
            f,ax=plt.subplots()
            ax.plot(signal_prairie_diff_filtered_rounded_corrected,'r')
            ax.plot(signal_daq_diff_filtered_rounded_corrected,'k',alpha=0.5)
            ax.plot(sigup_pra_corrected,signal_prairie_diff_filtered_rounded_corrected[sigup_pra_corrected],'ro')
            ax.plot(sigup_daq_corrected,signal_daq_diff_filtered_rounded_corrected[sigup_daq_corrected],'ko',alpha=0.5)
            ax.plot(sigdown_pra_corrected,signal_prairie_diff_filtered_rounded_corrected[sigdown_pra_corrected],'ro')
            ax.plot(sigdown_daq_corrected,signal_daq_diff_filtered_rounded_corrected[sigdown_daq_corrected],'ko',alpha=0.5)
            
        #extract trialfragemnts
        trialfragments=[]
       
        trans=list(zip(sigup_pra_corrected-1,sigdown_pra_corrected+1))
 
        
        for tri in range(20):
            trialfragments.append(sig_prairie[trans[tri][1]:trans[tri+1][0]])
            signal_prairie_filtered_rounded_corrected,\
            signal_prairie_diff_filtered_rounded_corrected,\
            signal_prairie_diff_filtered_rounded_corrected_rerounded,\
            signal_prairie_errors_pairs = self.correct_voltage_split_transitions(trialfragments[tri],kernel_size=3)
            
         
            if plot:

                f,ax=plt.subplots(2,sharex=True)
                ax[0].plot(trialfragments[tri])
                ax[1].plot(signal_prairie_diff_filtered_rounded_corrected_rerounded,'r')
            
 
         

 
            blank_sigup_pra=np.argwhere(np.logical_or(np.logical_or(np.logical_or(signal_prairie_diff_filtered_rounded_corrected==blank_volt_a ,
                                                                    signal_prairie_diff_filtered_rounded_corrected==blank_volt_b),
                                                                    signal_prairie_diff_filtered_rounded_corrected==blank_volt),
                                                                   signal_prairie_diff_filtered_rounded_corrected==blank_volt_c)).flatten()   
            
            blank_sigup_pra_corrected=np.array([tran for tran in blank_sigup_pra if np.round(signal_prairie_filtered_rounded_corrected,1)[tran+1]==blank_volt])

                                        
            blank_sigdown_pra=np.argwhere(np.logical_or(np.logical_or(np.logical_or(signal_prairie_diff_filtered_rounded_corrected==-blank_volt_a ,
                                                                    signal_prairie_diff_filtered_rounded_corrected==-blank_volt_b),
                                                                    signal_prairie_diff_filtered_rounded_corrected==-blank_volt),
                                                                   signal_prairie_diff_filtered_rounded_corrected==-blank_volt_c)).flatten()   
            
               
            blank_sigdown_pra_corrected=np.array([tran for tran in blank_sigdown_pra if np.round(signal_prairie_filtered_rounded_corrected,1)[tran-1]==blank_volt])

            
            blank_reps=list(zip(blank_sigup_pra_corrected+1,blank_sigdown_pra_corrected+1))
            
            grat_reps=[]
            
            for i in range(len(orient_volt)):
                
                up=np.argwhere(np.logical_or(np.logical_or(np.logical_or( abs(signal_prairie_diff_filtered_rounded_corrected-orient_volt[i])<=0.1 ,
                                                                 abs(signal_prairie_diff_filtered_rounded_corrected-orient_volt_a[i])<=0.1),
                                                               abs(signal_prairie_diff_filtered_rounded_corrected-orient_volt_b[i])<=0.1),
                                                              abs(signal_prairie_diff_filtered_rounded_corrected-orient_volt_c[i])<=0.1)).flatten() 
                
                up_corrected=np.array([tran for tran in up if abs(signal_prairie_filtered_rounded_corrected[tran+1]-orient_volt[i])<=0.1])

                
                
                
                down=np.argwhere(np.logical_or( abs(signal_prairie_diff_filtered_rounded_corrected+orient_volt_a[i])<=0.1 ,  abs(signal_prairie_diff_filtered_rounded_corrected+orient_volt_b[i])<=0.1)).flatten()
                down_corrected=np.array([tran for tran in down if abs(signal_prairie_filtered_rounded_corrected[tran-1]-orient_volt[i])<=0.1])


 
                grat_reps.append(list(zip(up_corrected+1,down_corrected+1)))
                
            all_reps=sorted(blank_reps+[a for i in grat_reps for a in  i])
            
            #process blamnnk trials 
           
                
                
            #get vis stim starting index    
            for i in range(len(orient_volt)):
                self.optodrift_info[f'Grat_{i+1}']['FulltrialStart'][tri]=trans[tri][1]
                self.optodrift_info[f'Grat_{i+1}']['FulltrialEnd'][tri]=trans[tri+1][0]
                stimonsets=np.zeros(2).astype(int)
                stimoffsets=np.zeros(2).astype(int)
                for j in range(2):
                    start=grat_reps[i][j][1]
                    if all_reps.index(grat_reps[i][j])+1!=20:
                        fin=all_reps[all_reps.index(grat_reps[i][j])+1][0]
                    else:
                        fin=trans[tri+1][0]
                        
                        
                    self.optodrift_info[f'Grat_{i+1}']['ReplStart'][tri,j]=start
                        
                    sing_trial=trialfragments[tri][start:fin]
                    stimoffsets[j]=len(sing_trial)
                
                        
                    signal_prairie_filtered_rounded_corrected,\
                    signal_prairie_diff_filtered_rounded_corrected,\
                    signal_prairie_diff_filtered_rounded_corrected_rerounded,\
                    signal_prairie_errors_pairs = self.correct_voltage_split_transitions(sing_trial,kernel_size=3)

                    if plot:
                        f,ax=plt.subplots(2)
                        ax[0].plot(sing_trial,'r')
                        ax[1].plot(signal_prairie_diff_filtered_rounded_corrected,'r')
                        plt.show()
                    
                    if all_reps.index(grat_reps[i][j])+1!=20:
                        buggy=np.argwhere(np.logical_or(signal_prairie_diff_filtered_rounded_corrected==1, signal_prairie_diff_filtered_rounded_corrected==-1)).flatten()
                        stimonsets[j]=buggy[0]
                        
                        self.optodrift_info[f'Grat_{i+1}']['ReplEnd'][tri,j]=fin
 
                        
                    else:
                        tr=np.argwhere(np.logical_or(np.logical_or(signal_prairie_diff_filtered_rounded_corrected==1 ,
                                                     signal_prairie_diff_filtered_rounded_corrected==-1),
                                                     signal_prairie_diff_filtered_rounded_corrected==-3)).flatten()
                        stimonsets[j]=tr[0]
                        
                        self.optodrift_info[f'Grat_{i+1}']['ReplEnd'][tri,j]=start+tr[1]
               
                    self.optodrift_info[f'Grat_{i+1}']['ArrayFinal'][tri,j]=stimonsets[j]+self.optodrift_info[f'Grat_{i+1}']['ReplStart'][tri,j]+self.optodrift_info[f'Grat_{i+1}']['FulltrialStart'][tri]
                    self.optodrift_info[f'Grat_{i+1}']['ArrayInTrial'][tri,j]=stimonsets[j]+self.optodrift_info[f'Grat_{i+1}']['ReplStart'][tri,j]
 
    
                    self.optodrift_info[f'Grat_{i+1}']['ArrayFinalOffset'][tri,j]=stimoffsets[j]+self.optodrift_info[f'Grat_{i+1}']['ReplStart'][tri,j]+self.optodrift_info[f'Grat_{i+1}']['FulltrialStart'][tri]
                    self.optodrift_info[f'Grat_{i+1}']['ArrayInTrialOffset'][tri,j]=stimoffsets[j]+self.optodrift_info[f'Grat_{i+1}']['ReplStart'][tri,j]
 

                        
                self.optodrift_info[f'Grat_{i+1}']['ArrayInRep'][tri,:]=stimonsets
                self.optodrift_info[f'Grat_{i+1}']['ArrayInRepOffset'][tri,:]=stimoffsets

                
            optostimonsets=np.zeros(2).astype(int)
            blankstimoffsets=np.zeros(4).astype(int)
            for k in range(4):
                self.optodrift_info['Blank']['FulltrialStart'][tri]=trans[tri][1]
                self.optodrift_info['Blank']['FulltrialEnd'][tri]=trans[tri+1][0]
                
                
                start=blank_reps[k][1]
                if all_reps.index(blank_reps[k])+1!=20:
                    fin=all_reps[all_reps.index(blank_reps[k])+1][0]
                else:
                    fin=trans[tri+1][0]
                    
                       
                self.optodrift_info['Blank']['ReplStart'][tri,k]=start
                    
                sing_trial=trialfragments[tri][start:fin]
                blankstimoffsets[k]=len(sing_trial)            
                    
                signal_prairie_filtered_rounded_corrected,\
                signal_prairie_diff_filtered_rounded_corrected,\
                signal_prairie_diff_filtered_rounded_corrected_rerounded,\
                signal_prairie_errors_pairs = self.correct_voltage_split_transitions(sing_trial,kernel_size=3)
                if plot:
                    f,ax=plt.subplots(2)
                    ax[0].plot(sing_trial,'r')
                    ax[1].plot(signal_prairie_diff_filtered_rounded_corrected,'r')
                    plt.show()
                

                if k<2:
                    buggy=np.argwhere(np.logical_or(signal_prairie_diff_filtered_rounded_corrected==1, signal_prairie_diff_filtered_rounded_corrected==-1)).flatten()
                    if buggy.any():
                        optostimonsets[k]=buggy[0]
                        print('There are opto blank sweeps')
                    else:
                        print('There are not opto blank sweeps, using first grating onset')
                        optostimonsets[k]= self.optodrift_info['Grat_1']['ArrayInRep'][tri,0]

                    self.optodrift_info['Blank']['ReplEnd'][tri,k]=fin
                    self.optodrift_info['Blank']['ArrayFinal'][tri,k]=optostimonsets[k]+self.optodrift_info['Blank']['ReplStart'][tri,k]+self.optodrift_info['Blank']['FulltrialStart'][tri]
                    self.optodrift_info['Blank']['ArrayInTrial'][tri,k]=optostimonsets[k]+self.optodrift_info['Blank']['ReplStart'][tri,k]
                    self.optodrift_info['Blank']['ArrayInRep'][tri,:2]=optostimonsets
                    
                else: 
                       
                    self.optodrift_info['Blank']['ArrayFinal'][tri,k]=self.optodrift_info['Blank']['ArrayInRep'][tri,0]+self.optodrift_info['Blank']['ReplStart'][tri,k]+self.optodrift_info['Blank']['FulltrialStart'][tri]
                    self.optodrift_info['Blank']['ArrayInTrial'][tri,k]=self.optodrift_info['Blank']['ArrayInRep'][tri,0]+self.optodrift_info['Blank']['ReplStart'][tri,k]
                    self.optodrift_info['Blank']['ArrayInRep'][tri,2:5]=self.optodrift_info['Blank']['ArrayInRep'][tri,:2]
                    
                    
                self.optodrift_info['Blank']['ArrayFinalOffset'][tri,k]=blankstimoffsets[k]+self.optodrift_info['Blank']['ReplStart'][tri,k]+self.optodrift_info['Blank']['FulltrialStart'][tri]
                self.optodrift_info['Blank']['ArrayInTrialOffset'][tri,k]=blankstimoffsets[k]+self.optodrift_info['Blank']['ReplStart'][tri,k]
                self.optodrift_info['Blank']['ArrayInRepOffset'][tri,k]=blankstimoffsets[k]

              
                    

         
        f,ax=plt.subplots()
        ax.plot(sig_prairie,'k')
        colors= ['tab:blue', 'tab:orange', 'tab:green', 'tab:red','tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
        for stim in range(1,9):
            grating=self.optodrift_info[f'Grat_{stim}']
            ax.plot(grating['ArrayFinal'],sig_prairie[grating['ArrayFinal']],'o', color=colors[stim-1])
        for sweep in range(4):
            blank=self.optodrift_info['Blank']
            ax.plot(blank['ArrayFinal'],sig_prairie[blank['ArrayFinal']],'o', color=colors[sweep+2])


            


#%% Allen

    def process_allen_paradigms(self):
        
        self.get_paradigm_indexes()
        self.slice_visstim_by_paradigm()
        self.slice_locomotion_by_paradigm()
        

    def process_allenA_signals(self):   
        module_logger.info('Analysing AllenA Gratings')
        self.get_drifting_gratings_indexes()
        self.get_movie_one_trial_structure()
        self.get_movie_three_trial_structure()
        module_logger.info('Finished Analysing AllenA Gratings')


        
    def process_allenB_signals(self):   
        module_logger.info('Analysing AllenB Gratings')
        self.get_static_gratings_trial_structure()
        self.get_natural_images_trial_structure()
        self.get_movie_one_trial_structure()
        module_logger.info('Finished Analysing AllenB Gratings')

        
    
    def process_allenC_signals(self):  
        module_logger.info('Analysing AllenC Gratings')
        self.get_movie_one_trial_structure()
        self.get_movie_two_trial_structure()
        self.get_sparse_noise_trial_structure()
        module_logger.info('Finished Analysing AllenC Gratings')




        
        # fist get transitions between stimulaton paradigms
        pass
        
    def get_paradigm_indexes(self):   
        '''
        # this has to be done semimanually 
        Acquisitions already done
            211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_transitions_indexes
            
        '''
        # %matplotlib qt

        """
        summary of transtions
        paradigm_start <-9.5
        paradigm_end  >9.5
        
        noiseframes_start > rectified 0.5

        movietrialfirst start>2.5
        firstframemovie==movietrialatrat
        nextrframes > rectifed 0.5
        
        movietrail2problematic=
        movietrielaftesecond>2.3
        
        
        """
        

        if not self.transitions_dictionary:
            
            if self.acquisition_name=='220217_SPJZ_FOV1_AllenA_20x_920_52570_narrow_with-000':
                self.start_transitions=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']<-8).flatten()
                self.end_transitions=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']>9.6).flatten()  
                self.end_transitions=np.delete(self.end_transitions, 0)
                self.start_transitions=np.delete(self.start_transitions, -1)


                
                self.transitions_dictionary={'first_drifting_set_first':self.start_transitions[0],
                                            'first_drifting_set_last': self.end_transitions[0],
                                            
                                            'natural_movie_three_first_set_first':self.start_transitions[2],
                                            'natural_movie_three_first_set_last':self.end_transitions[2],
                                                                                       
                                            'natural_movie_one_set_first':self.start_transitions[4],                                   
                                            'natural_movie_one_set_last': self.end_transitions[4],
                                                                                 
                                            'second_drifting_set_first':self.start_transitions[6],
                                            'second_drifting_set_last':self.end_transitions[6],
                                            
                                            'spont_first':self.start_transitions[7],
                                            'spont_last':self.end_transitions[7],
                                            
                                            'natural_movie_three_second_set_first':self.start_transitions[8],
                                            'natural_movie_three_second_set_last': self.end_transitions[8],
                                            
                                            'third_drifting_set_first':self.start_transitions[10],
                                            'third_drifting_set_last':self.end_transitions[10],
                                            }

                
            else:          
                self.start_transitions=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']<-6.8).flatten()
                # self.start_transitions=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']<-9).flatten()+1
                self.start_transitions=np.delete(self.start_transitions, [9,10,11])
                self.end_transitions=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']>6.8).flatten()  
                # self.end_transitions=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']>9.5).flatten()  +1
    
                self.end_transitions=np.delete(self.end_transitions, 0)
                #eliminate first 
            
                # self.spont_start_transitions=np.argwhere(np.logical_and(self.dfdt_rounded_vis_stim['Prairie']['VisStim']>5, self.dfdt_rounded_vis_stim['Prairie']['VisStim']<6.5)).flatten()+1
                self.last_down_transition=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']<-0.3).flatten()[-1]+1
                
                fig, ax = plt.subplots(2, sharex=True)
                line, = ax[0].plot(self.visualstim_array['Prairie']['VisStim']) 
                line, = ax[1].plot(self.dfdt_rounded_vis_stim['Prairie']['VisStim']) 
                # line, = ax[0].plot(self.process_signal(self.dfdt_rounded_vis_stim,'rectified')['Prairie']['VisStim'],'r') 
                # line, = ax[2].plot(self.visualstim_array['Daq']['VisStim']) 
                # line, = ax[3].plot(self.dfdt_rounded_vis_stim['Daq']['VisStim']) 
                
                ax[0].plot(self.time_scale['Prairie'][self.start_transitions],self.visualstim_array['Prairie']['VisStim'][self.start_transitions],'rx') 
                ax[1].plot(self.time_scale['Prairie'][self.start_transitions],self.dfdt_rounded_vis_stim['Prairie']['VisStim'][self.start_transitions],'rx') 
                ax[0].plot(self.time_scale['Prairie'][self.end_transitions],self.visualstim_array['Prairie']['VisStim'][self.end_transitions],'go') 
                ax[1].plot(self.time_scale['Prairie'][self.end_transitions],self.dfdt_rounded_vis_stim['Prairie']['VisStim'][self.end_transitions],'go') 
                          
            
            
            
            """
            for the newest protocols is differnet
            """
            if self.vis_stim_protocol =='AllenA':
                self.transitions_dictionary={'first_drifting_set_first':self.start_transitions[0],
                                            'first_drifting_set_last': self.end_transitions[0],
                                            
                                            'natural_movie_three_first_set_first':self.start_transitions[2],
                                            'natural_movie_three_first_set_last':self.end_transitions[2],
                                                                                       
                                            'natural_movie_one_set_first':self.start_transitions[4],                                   
                                            'natural_movie_one_set_last': self.end_transitions[4],
                                                                                 
                                            'second_drifting_set_first':self.start_transitions[6],
                                            'second_drifting_set_last':self.end_transitions[6],
                                            
                                            'spont_first':self.start_transitions[7],
                                            'spont_last':self.end_transitions[7],
                                            
                                            'natural_movie_three_second_set_first':self.start_transitions[8],
                                            'natural_movie_three_second_set_last': self.end_transitions[8],
                                            
                                            'third_drifting_set_first':self.start_transitions[10],
                                            'third_drifting_set_last':self.last_down_transition+1,
                                            }
                pass
            
            elif self.vis_stim_protocol =='AllenB':
                selected_start_indexes=[0,2,3,4,6,8,10,12]
                selected_end_indexes=[0,2,3,4,6,7,9,11]
                fig, ax = plt.subplots(2, sharex=True)
                line, = ax[0].plot(self.visualstim_array['Prairie']['VisStim']) 
                line, = ax[1].plot(self.dfdt_rounded_vis_stim['Prairie']['VisStim']) 
                # line, = ax[0].plot(self.process_signal(self.dfdt_rounded_vis_stim,'rectified')['Prairie']['VisStim'],'r') 
                # line, = ax[2].plot(self.visualstim_array['Daq']['VisStim']) 
                # line, = ax[3].plot(self.dfdt_rounded_vis_stim['Daq']['VisStim']) 
                
                ax[0].plot(self.time_scale['Prairie'][self.start_transitions[selected_start_indexes]],self.visualstim_array['Prairie']['VisStim'][self.start_transitions[selected_start_indexes]],'rx') 
                ax[1].plot(self.time_scale['Prairie'][self.start_transitions[selected_start_indexes]],self.dfdt_rounded_vis_stim['Prairie']['VisStim'][self.start_transitions[selected_start_indexes]],'rx') 
                ax[0].plot(self.time_scale['Prairie'][self.end_transitions[selected_end_indexes]],self.visualstim_array['Prairie']['VisStim'][self.end_transitions[selected_end_indexes]],'go') 
                ax[1].plot(self.time_scale['Prairie'][self.end_transitions[selected_end_indexes]],self.dfdt_rounded_vis_stim['Prairie']['VisStim'][self.end_transitions[selected_end_indexes]],'go') 
                
                
                
                self.transitions_dictionary={'first_static_set_first':self.start_transitions[0],
                                            'first_static_set_last': self.end_transitions[0],
                                            'first_images_set_first':self.start_transitions[2],
                                            'first_images_set_last': self.end_transitions[2],
                                            'spont_first':self.start_transitions[3],
                                            'spont_last':self.end_transitions[3],
                                            'second_images_set_first':self.start_transitions[4],
                                            'second_images_set_last':self.end_transitions[4],
                                            'second_static_set_first':self.start_transitions[6],
                                            'second_static_set_last':self.end_transitions[6],
                                            'natural_movie_one_set_first':self.start_transitions[8],
                                            'natural_movie_one_set_last': self.end_transitions[7],                                             
                                            'third_images_set_first':self.start_transitions[10],
                                            'third_images_set_last':self.end_transitions[9],
                                            'third_static_set_first':self.start_transitions[12],
                                            'third_static_set_last':self.end_transitions[11],
                                            
                                          
                                            
                                            }
            
            elif self.vis_stim_protocol =='AllenC':
                
                self.transitions_dictionary={'first_noise_set_first':self.start_transitions[0],
                                            'first_noise_set_last': self.end_transitions[0],
                                            'second_noise_set_first':self.start_transitions[4],
                                            'second_noise_set_last':self.end_transitions[4],
                                            'third_noise_set_first':self.start_transitions[8],
                                            'third_noise_set_last':self.end_transitions[8],
                                            'natural_movie_one_set_first':self.start_transitions[2],
                                            'natural_movie_one_set_last': self.end_transitions[2],
                                            'natural_movie_two_set_first':self.start_transitions[6],
                                            'natural_movie_two_set_last': self.end_transitions[6],
                                            'spont1_first':self.start_transitions[1],
                                            'spont1_last':self.end_transitions[1],
                                            'spont2_first':self.start_transitions[7],
                                            'spont2_last':self.end_transitions[7],
                                            }
                
                
                
                
            self.save_transition_indexes()
        else:
            # correct renaming of move keys
            if self.vis_stim_protocol=='AllenA':
                
                if 'natural_movie_one_set_first' not in self.transitions_dictionary.keys():
                    
                    self.transitions_dictionary['natural_movie_one_set_first']= self.transitions_dictionary['short_movie_set_first']
                    self.transitions_dictionary.pop('short_movie_set_first', None)
                    
                if 'natural_movie_one_set_last' not in self.transitions_dictionary.keys():
                    
                    self.transitions_dictionary['natural_movie_one_set_last']= self.transitions_dictionary['short_movie_set_last']
                    self.transitions_dictionary.pop('short_movie_set_last', None)
                
                  
                if 'natural_movie_three_first_set_first' not in self.transitions_dictionary.keys():
                    
                    self.transitions_dictionary['natural_movie_three_first_set_first']= self.transitions_dictionary['first_movie_set_first']
                    self.transitions_dictionary.pop('first_movie_set_first', None)
                    
                if 'natural_movie_three_first_set_last' not in self.transitions_dictionary.keys():
                    
                    self.transitions_dictionary['natural_movie_three_first_set_last']= self.transitions_dictionary['first_movie_set_last']
                    self.transitions_dictionary.pop('first_movie_set_last', None)
                    
                      
                if 'natural_movie_three_second_set_first' not in self.transitions_dictionary.keys():
                    
                    self.transitions_dictionary['natural_movie_three_second_set_first']= self.transitions_dictionary['second_movie_set_first']
                    self.transitions_dictionary.pop('second_movie_set_first', None)
                    
                if 'natural_movie_three_second_set_last' not in self.transitions_dictionary.keys():
                    
                    self.transitions_dictionary['natural_movie_three_second_set_last']= self.transitions_dictionary['second_movie_set_last']
                    self.transitions_dictionary.pop('second_movie_set_last', None)
                
                
            elif self.vis_stim_protocol=='AllenB':
                
                if 'natural_movie_one_set_first' not in self.transitions_dictionary.keys():
                    
                    self.transitions_dictionary['natural_movie_one_set_first']= self.transitions_dictionary['movie_set_first']
                    self.transitions_dictionary.pop('movie_set_first', None)
                    
                if 'natural_movie_one_set_last' not in self.transitions_dictionary.keys():
                    
                    self.transitions_dictionary['natural_movie_one_set_last']= self.transitions_dictionary['movie_set_last']
                    self.transitions_dictionary.pop('movie_set_last', None)
                    
            elif self.vis_stim_protocol=='AllenC':
                
                if 'natural_movie_one_set_first' not in self.transitions_dictionary.keys():
                    
                    self.transitions_dictionary['natural_movie_one_set_first']= self.transitions_dictionary['first_movie_set_first']
                    self.transitions_dictionary.pop('first_movie_set_first', None)
                    
                if 'natural_movie_one_set_last' not in self.transitions_dictionary.keys():
                    
                    self.transitions_dictionary['natural_movie_one_set_last']= self.transitions_dictionary['first_movie_set_last']
                    self.transitions_dictionary.pop('first_movie_set_last', None)
                    
                if 'natural_movie_two_set_first' not in self.transitions_dictionary.keys():
                    
                    self.transitions_dictionary['natural_movie_two_set_first']= self.transitions_dictionary['second_movie_set_first']
                    self.transitions_dictionary.pop('second_movie_set_first', None)
                    
                if 'natural_movie_two_set_last' not in self.transitions_dictionary.keys():
                    
                    self.transitions_dictionary['natural_movie_two_set_last']= self.transitions_dictionary['second_movie_set_last']
                    self.transitions_dictionary.pop('second_movie_set_last', None)

      
            
            
                pass
            
            os.remove(self.indexes_full_file_paths_to_save[0])
            self.save_transition_indexes()

 
            # newest signals thre allen sessions
            # self.transitions_dictionary={'first_noise_set_first':self.start_transitions[0],
            #                             'first_noise_set_last': self.end_transitions[0],
            #                             'second_noise_set_first':self.start_transitions[4],
            #                             'second_noise_set_last':self.end_transitions[4],
            #                             'third_noise_set_first':self.start_transitions[8],
            #                             'third_noise_set_last':self.end_transitions[8],
            #                             'natural_movie_one_set_first':self.start_transitions[2],
            #                             'natural_movie_one_set_last': self.end_transitions[2],
            #                             'natural_movie_two_set_first':self.start_transitions[6],
            #                             'natural_movie_two_set_last': self.end_transitions[6],
            #                             'spont1_first':self.start_transitions[1],
            #                             'spont1_last':self.end_transitions[1],
            #                             'spont2_first':self.start_transitions[7],
            #                             'spont2_last':self.end_transitions[7],
            #                             }
            
            
            # newest signals thre allen sessions
            # self.transitions_dictionary={'first_noise_set_first':self.start_transitions[0],
            #                             'first_noise_set_last': self.end_transitions[0],
            #                             'second_noise_set_first':self.start_transitions[4],
            #                             'second_noise_set_last':self.end_transitions[4],
            #                             'third_noise_set_first':self.start_transitions[8],
            #                             'third_noise_set_last':self.end_transitions[8],
            #                             'natural_movie_one_first_set_first':self.start_transitions[2],
            #                             'natural_movie_one_first_set_last': self.end_transitions[2],
            #                             'natural_movie_one_second_set_first':self.start_transitions[6],
            #                             'natural_movie_one_second_set_last': self.end_transitions[6],
            #                             'spont1_first':self.start_transitions[1],
            #                             'spont1_last':self.end_transitions[1],
            #                             'spont2_first':self.start_transitions[7],
            #                             'spont2_last':self.end_transitions[7],
            #                             }
            

            # for  211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_transitions_indexes
            # self.start_transitions=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']<-6.5).flatten()+1
            # self.end_transitions=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']>6.5).flatten()+1   
            # self.end_transitions=np.delete(self.end_transitions, 0)
            # #eliminate first 
        
            # self.spont_start_transitions=np.argwhere(np.logical_and(self.dfdt_rounded_vis_stim['Prairie']['VisStim']>5, self.dfdt_rounded_vis_stim['Prairie']['VisStim']<6.5)).flatten()+1
            # self.last_down_transition=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']<-0.5).flatten()[-1]+1
            # self.transitions_dictionary={'first_drifting_set_first':self.start_transitions[0],
            #                             'first_drifting_set_last': self.end_transitions[0],
            #                             'second_drifting_set_first':self.start_transitions[6],
            #                             'second_drifting_set_last':self.end_transitions[5],
            #                             'third_drifting_set_first':self.start_transitions[10],
            #                             'third_drifting_set_last':self.last_down_transition+1,
            #                             'natural_movie_one_first_set_first':self.start_transitions[2],
            #                             'natural_movie_one_first_set_last': self.end_transitions[2],
            #                             'natural_movie_one_second_set_first':self.start_transitions[8],
            #                             'natural_movie_one_second_set_last': self.end_transitions[7],
            #                             'short_movie_set_first':self.start_transitions[4],
            #                             'short_movie_set_last':self.end_transitions[4],
            #                             'spont_first':self.spont_start_transitions[3],
            #                             'spont_last':self.end_transitions[6]-1,
            #                             }
 
            # for spja only
            # self.end_transitions=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']<-6.5).flatten()+1 
            # self.start_transitions=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']>5).flatten()+1   
            # self.last_down_transition=np.argwhere(self.dfdt_rounded_vis_stim['Prairie']['VisStim']<-0.5).flatten()[-1]+1
            # self.transitions_dictionary={'first_drifting_set_first':self.end_transitions[0],
            #                             'first_drifting_set_last': self.start_transitions[1],
            #                             'second_drifting_set_first':self.end_transitions[5],
            #                             'second_drifting_set_last':self.start_transitions[8],
            #                             'third_drifting_set_first':self.end_transitions[8],
            #                             'third_drifting_set_last':self.last_down_transition+1000,
            #                             'natural_movie_one_first_set_first':self.end_transitions[2],
            #                             'natural_movie_one_first_set_last': self.start_transitions[4],
            #                             'natural_movie_one_second_set_first':self.end_transitions[6],
            #                             'natural_movie_one_second_set_last': self.start_transitions[9],
            #                             'short_movie_set_first':self.end_transitions[4],
            #                             'short_movie_set_last':self.start_transitions[7],
            #                             'spont_first':self.start_transitions[8],
            #                             'spont_last':self.end_transitions[6]-3,
            #                             }
            

    def slice_visstim_by_paradigm (self): 
        
        self.paradigm_sliced_vis_stim={}
        
        if self.vis_stim_protocol =='AllenA':
            
            self.first_drifting_set=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['first_drifting_set_first']:self.transitions_dictionary['first_drifting_set_last']]
            self.second_drifting_set=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['second_drifting_set_first']:self.transitions_dictionary['second_drifting_set_last']]
            self.third_drifting_set=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['third_drifting_set_first']:self.transitions_dictionary['third_drifting_set_last']]
            self.natural_movie_three_first_set=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['natural_movie_three_first_set_first']:self.transitions_dictionary['natural_movie_three_first_set_last']]
            self.natural_movie_three_second_set=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['natural_movie_three_second_set_first']:self.transitions_dictionary['natural_movie_three_second_set_last']]
            self.natural_movie_one_set=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['natural_movie_one_set_first']:self.transitions_dictionary['natural_movie_one_set_last']]     
            self.spont=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['spont_first']:self.transitions_dictionary['spont_last']]
        
            self.first_drifting_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['first_drifting_set_first']:self.transitions_dictionary['first_drifting_set_last']]
            self.second_drifting_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['second_drifting_set_first']:self.transitions_dictionary['second_drifting_set_last']]
            self.third_drifting_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['third_drifting_set_first']:self.transitions_dictionary['third_drifting_set_last']]
            self.natural_movie_three_first_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['natural_movie_three_first_set_first']:self.transitions_dictionary['natural_movie_three_first_set_last']]
            self.natural_movie_three_second_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['natural_movie_three_second_set_first']:self.transitions_dictionary['natural_movie_three_second_set_last']]
            self.natural_movie_one_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['natural_movie_one_set_first']:self.transitions_dictionary['natural_movie_one_set_last']]     
            self.spont=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['spont_first']:self.transitions_dictionary['spont_last']]
            
        elif self.vis_stim_protocol =='AllenB':
            
            self.first_static_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['first_static_set_first']:self.transitions_dictionary['first_static_set_last']]
            self.second_static_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['second_static_set_first']:self.transitions_dictionary['second_static_set_last']]
            self.third_static_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['third_static_set_first']:self.transitions_dictionary['third_static_set_last']]
            self.first_images_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['first_images_set_first']:self.transitions_dictionary['first_images_set_last']]
            self.second_images_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['second_images_set_first']:self.transitions_dictionary['second_images_set_last']]
            self.third_images_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['third_images_set_first']:self.transitions_dictionary['third_images_set_last']]
            self.natural_movie_one_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['natural_movie_one_set_first']:self.transitions_dictionary['natural_movie_one_set_last']]
            self.spont=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['spont_first']:self.transitions_dictionary['spont_last']]
            
        elif self.vis_stim_protocol =='AllenC':
            
            self.first_noise_set=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['first_noise_set_first']:self.transitions_dictionary['first_noise_set_last']]
            self.second_noise_set=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['second_noise_set_first']:self.transitions_dictionary['second_noise_set_last']]
            self.third_noise_set=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['third_noise_set_first']:self.transitions_dictionary['third_noise_set_last']]
            self.natural_movie_one_set=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['natural_movie_one_set_first']:self.transitions_dictionary['natural_movie_one_set_last']]
            self.natural_movie_two_set=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['natural_movie_two_set_first']:self.transitions_dictionary['natural_movie_two_set_last']]
            self.spont1=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['spont1_first']:self.transitions_dictionary['spont1_last']]
            self.spont2=self.rounded_vis_stim['Prairie']['VisStim'][self.transitions_dictionary['spont2_first']:self.transitions_dictionary['spont2_last']]     
            
                 
            self.first_noise_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['first_noise_set_first']:self.transitions_dictionary['first_noise_set_last']]
            self.second_noise_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['second_noise_set_first']:self.transitions_dictionary['second_noise_set_last']]
            self.third_noise_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['third_noise_set_first']:self.transitions_dictionary['third_noise_set_last']]
            self.natural_movie_one_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['natural_movie_one_set_first']:self.transitions_dictionary['natural_movie_one_set_last']]
            self.natural_movie_two_set=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['natural_movie_two_set_first']:self.transitions_dictionary['natural_movie_two_set_last']]
            self.spont1=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['spont1_first']:self.transitions_dictionary['spont1_last']]
            self.spont2=self.visualstim_array['Prairie']['VisStim'][self.transitions_dictionary['spont2_first']:self.transitions_dictionary['spont2_last']]     
               
    def slice_locomotion_by_paradigm (self):  
         
        if self.vis_stim_protocol =='AllenA':
        
             self.first_drifting_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['first_drifting_set_first']:self.transitions_dictionary['first_drifting_set_last']]
             self.second_drifting_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['second_drifting_set_first']:self.transitions_dictionary['second_drifting_set_last']]
             self.third_drifting_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['third_drifting_set_first']:self.transitions_dictionary['third_drifting_set_last']]
             self.natural_movie_three_first_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['natural_movie_three_first_set_first']:self.transitions_dictionary['natural_movie_three_first_set_last']]
             self.natural_movie_three_second_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['natural_movie_three_second_set_first']:self.transitions_dictionary['natural_movie_three_second_set_last']]
             self.natural_movie_one_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['natural_movie_one_set_first']:self.transitions_dictionary['natural_movie_one_set_last']]     
             self.spont_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['spont_first']:self.transitions_dictionary['spont_last']]    
        
        elif self.vis_stim_protocol =='AllenB':
            self.first_static_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['first_static_set_first']:self.transitions_dictionary['first_static_set_last']]
            self.second_static_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['second_static_set_first']:self.transitions_dictionary['second_static_set_last']]
            self.third_static_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['third_static_set_first']:self.transitions_dictionary['third_static_set_last']]
            self.first_images_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['first_images_set_first']:self.transitions_dictionary['first_images_set_last']]
            self.second_images_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['second_images_set_first']:self.transitions_dictionary['second_images_set_last']]
            self.third_images_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['third_images_set_first']:self.transitions_dictionary['third_images_set_last']]
            self.natural_movie_one_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['natural_movie_one_set_first']:self.transitions_dictionary['natural_movie_one_set_last']]
            self.spont_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['spont_first']:self.transitions_dictionary['spont_last']]
            
        elif self.vis_stim_protocol =='AllenC':
        
            self.first_noise_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['first_noise_set_first']:self.transitions_dictionary['first_noise_set_last']]
            self.second_noise_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['second_noise_set_first']:self.transitions_dictionary['second_noise_set_last']]
            self.third_noise_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['third_noise_set_first']:self.transitions_dictionary['third_noise_set_last']]
            self.natural_movie_one_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['natural_movie_one_set_first']:self.transitions_dictionary['natural_movie_one_set_last']]
            self.natural_movie_two_set_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['natural_movie_two_set_first']:self.transitions_dictionary['natural_movie_two_set_last']]
            self.spont2_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['spont2_first']:self.transitions_dictionary['spont2_last']]     
            self.spont1_speed=self.rectified_speed_array['Prairie']['Locomotion'][self.transitions_dictionary['spont1_first']:self.transitions_dictionary['spont1_last']]



#%% common processing

    def get_movie_one_trial_structure(self):
         
        self.movie_one_frame_index_full_recording =np.zeros((1))
        self.movie_one_frame_indexes_by_trial =np.zeros((1))
        self.movie_one_trial_indexes_by_paradigm=np.zeros((1))

        self.load_movie_one_indexes()
        
        if not any( [self.movie_one_trial_indexes_by_paradigm.any(),self.movie_one_frame_index_full_recording.any(),self.movie_one_frame_indexes_by_trial.any()]):
            to_process=self.natural_movie_one_set
            if self.acquisition_name=='220213_SPJZ_FOV1_AllenC_20x_940_52570_narrow_with-000':
                #correct the bad voltage
                rounded_trace,_,_,_ = self.correct_voltage_split_transitions(self.natural_movie_one_set,plot=True)


                oldv=np.round(np.flip(np.arange(0,8.8,0.8)),3)
                newv=np.append(np.round(np.flip(np.arange(5.5,10.5,0.5)),3),3)
                

                for i,j in zip(oldv,newv):
                    fix, ax=plt.subplots(1)
                    fix.suptitle(f'Natural movie 1 correcting {i} to {j}')
                    ax.plot( rounded_trace,label='Volatage')
                    rounded_trace[rounded_trace==i]=j
                    print(i)
                    print(j)

                    ax.plot( rounded_trace,label='New Volatage')
                    ax.legend()
                    plt.show()


  
                temp=np.diff(rounded_trace)
                temp2=np.around(temp,3)
            
                fix, ax=plt.subplots(1)
                fix.suptitle('Natural movie 1')
                ax.plot( rounded_trace,label='Volatage')
                # ax.plot(temp,label='Diff')
                ax.plot(temp2,label='Rounded DIff')
                
                ax.legend()
                plt.show()
                to_process=rounded_trace
                startprocess=copy.deepcopy(to_process)
                startprocess[to_process<5.1]=0
            
                self.movie_one_trial_indexes_by_paradigm=sg.find_peaks(np.diff(startprocess),height=5)[0]
                self.movie_one_frame_indexes_by_trial=np.zeros([10,900,2])

                self.correct_movie_one_indexes_for_full_movie()

                self.save_movie_one_indexes()  
                
            elif self.acquisition_name=='220214_SPJZ_FOV1_AllenB_20x_980_52570_narrow_with-000':
                
                #correct the bad voltage
                rounded_trace,_,_,_ = self.correct_voltage_split_transitions(self.natural_movie_one_set,plot=True)


                oldv=[0]
                newv=[1]
                

                for i,j in zip(oldv,newv):
                    fix, ax=plt.subplots(1)
                    fix.suptitle(f'Natural movie 1 correcting {i} to {j}')
                    ax.plot( rounded_trace,label='Volatage')
                    rounded_trace[rounded_trace==i]=j
                    print(i)
                    print(j)

                    ax.plot( rounded_trace,label='New Volatage')
                    ax.legend()
                    plt.show()


  
                temp=np.diff(rounded_trace)
                temp2=np.around(temp,3)
            
                fix, ax=plt.subplots(1)
                fix.suptitle('Natural movie 1')
                ax.plot( rounded_trace,label='Volatage')
                # ax.plot(temp,label='Diff')
                ax.plot(temp2,label='Rounded DIff')
                
                ax.legend()
                plt.show()
                to_process=rounded_trace
                startprocess=copy.deepcopy(to_process)
                startprocess[to_process<3.5]=0
            
                self.movie_one_trial_indexes_by_paradigm=sg.find_peaks(np.diff(startprocess),height=3.5)[0]
                self.movie_one_frame_indexes_by_trial=np.zeros([10,900,2])

                self.correct_movie_one_indexes_for_full_movie()

                self.save_movie_one_indexes()  
                
            elif self.acquisition_name=='220217_SPJZ_FOV1_AllenA_20x_920_52570_narrow_with-000':
                 
                 #correct the bad voltage
                 rounded_trace,_,_,_ = self.correct_voltage_split_transitions(self.natural_movie_one_set,plot=True)
            
            
                 oldv=[0]
                 newv=[1]
                 
            
                 for i,j in zip(oldv,newv):
                     fix, ax=plt.subplots(1)
                     fix.suptitle(f'Natural movie 1 correcting {i} to {j}')
                     ax.plot( rounded_trace,label='Volatage')
                     rounded_trace[rounded_trace==i]=j
                     print(i)
                     print(j)
            
                     ax.plot( rounded_trace,label='New Volatage')
                     ax.legend()
                     plt.show()
            
            
            
                 temp=np.diff(rounded_trace)
                 temp2=np.around(temp,3)
             
                 fix, ax=plt.subplots(1)
                 fix.suptitle('Natural movie 1')
                 ax.plot( rounded_trace,label='Volatage')
                 # ax.plot(temp,label='Diff')
                 ax.plot(temp2,label='Rounded DIff')
                 
                 ax.legend()
                 plt.show()
                 to_process=rounded_trace
                 startprocess=copy.deepcopy(to_process)
                 startprocess[to_process<3.5]=0
             
                 self.movie_one_trial_indexes_by_paradigm=sg.find_peaks(np.diff(startprocess),height=3.5)[0]
                 self.movie_one_frame_indexes_by_trial=np.zeros([10,900,2])
            
                 self.correct_movie_one_indexes_for_full_movie()
            
                 self.save_movie_one_indexes()  
            
                                

            else:
                temp=np.diff(np.around(sg.medfilt(self.natural_movie_one_set, kernel_size=29),1))
                temp2=np.around(temp,3)
                temp2[np.abs(temp2)==2]=0
    
                fix, ax=plt.subplots(1)
                fix.suptitle('Natural movie 1')
                ax.plot( self.natural_movie_one_set,label='Volatage')
                # ax.plot(temp,label='Diff')
                ax.plot(temp2,label='Rounded DIff')
              
    
                ax.legend()
                plt.show()
                self.movie_one_voltage_slice_filtered_rounded_corrected,self.movie_one_diff_voltage_slice_filtered_rounded_corrected,self.movie_one_diff_voltage_slice_filtered_rounded_corrected_rerounded,self.movie_one_errors_pairs = self.correct_voltage_split_transitions(to_process,plot=True)
    
                initial_transitions_odd=np.argwhere(np.logical_and(self.movie_one_diff_voltage_slice_filtered_rounded_corrected!=2, self.movie_one_diff_voltage_slice_filtered_rounded_corrected<2))
                
                
                movie_trial_starts=np.argwhere(np.logical_and(self.movie_one_diff_voltage_slice_filtered_rounded_corrected!=2 , self.movie_one_diff_voltage_slice_filtered_rounded_corrected>1)).flatten()
                movie_trial_ends=np.argwhere(self.movie_one_diff_voltage_slice_filtered_rounded_corrected<-2).flatten()
        
                self.movie_one_frame_indexes_by_trial=np.zeros([10,900,2])
                
                fix, ax=plt.subplots(1)
                fix.suptitle('Natural movie 1')
                ax.plot( self.movie_one_diff_voltage_slice_filtered_rounded_corrected)
                for i, start in enumerate(movie_trial_starts):
                    if i==9:
                       movie_trial_starts=np.insert(movie_trial_starts,10,len(self.movie_one_diff_voltage_slice_filtered_rounded_corrected))
                    
                    movietrial=self.movie_one_diff_voltage_slice_filtered_rounded_corrected[start:movie_trial_starts[i+1]]
                    ups=np.argwhere(movietrial==2).flatten()+start
                    down=np.argwhere(movietrial==-2).flatten()+start
            
                    
                    ups=np.insert(ups, 0, start)
                    down=np.insert(down, 0, movie_trial_ends[i])
                    if i!=9:
                        down=np.append(down, movie_trial_starts[i+1])
                    else:
                        down=np.append(down, movie_trial_starts[-1]-1)
                        
                       
                    self.movie_one_frame_indexes_by_trial[i,:,0]=ups
                    self.movie_one_frame_indexes_by_trial[i,:,1]=down
        
          
                    # ax.plot(np.arange(start,movie_trial_starts[i+1]),self.movie_one_diff_voltage_slice_filtered_rounded_corrected[start:movie_trial_starts[i+1]])
                    ax.plot( ups, self.movie_one_diff_voltage_slice_filtered_rounded_corrected[ups],'bo',label='ups')
                    ax.plot( down, self.movie_one_diff_voltage_slice_filtered_rounded_corrected[down],'ko',label='downs')
                    ax.plot( movie_trial_starts, self.movie_one_diff_voltage_slice_filtered_rounded_corrected[movie_trial_starts],'ko',label='Movie Starts')
    
                    ax.legend()
        
                    framelengths= self.movie_one_frame_indexes_by_trial[i,:,1]- self.movie_one_frame_indexes_by_trial[i,:,0]
    
    
                plt.show()
    
                self.correct_movie_one_indexes_for_full_movie()
        
                self.save_movie_one_indexes()  
            
        else:
            fig,axo=plt.subplots()
            fig.suptitle('Movie 1')
            axo.plot(self.rounded_vis_stim['Prairie']['VisStim'])
            axo.plot(self.movie_one_frame_index_full_recording[:,0,0], self.rounded_vis_stim['Prairie']['VisStim'][self.movie_one_frame_index_full_recording[:,0,0]],'rx',label='Ups')
            axo.plot(self.movie_one_frame_index_full_recording[:,0,1], self.rounded_vis_stim['Prairie']['VisStim'][self.movie_one_frame_index_full_recording[:,0,1]],'bo',label='Downs')

            plt.show()
           
            
        # import scipy.io as spio
        # import caiman as cm
        # movie=spio.loadmat(r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\visual_stim\BehaviourCode\AllenStimuli\Smalles\natural_movie_one.mat')
        # movie=movie['natural_movie_one_all_warped_frames']
        
        # mov=np.moveaxis(movie, -1, 0)
        # mov=cm.movie(mov)
        # mov.play(fr=300)

        
        # fix, ax=plt.subplots(1)
        # ax.imshow(mov[1,:,:])
        
    def correct_movie_one_indexes_for_full_movie(self):
        
        self.movie_one_frame_index_full_recording = self.movie_one_frame_indexes_by_trial+self.transitions_dictionary['natural_movie_one_set_first']
        self.movie_one_frame_index_full_recording=self.movie_one_frame_index_full_recording.astype('uint32')
        self.movie_one_trial_full_recording= self.movie_one_trial_indexes_by_paradigm+self.transitions_dictionary['natural_movie_one_set_first']
        self.movie_one_trial_full_recording=self.movie_one_trial_full_recording.astype('uint32')


        fig,axo=plt.subplots()
        fig.suptitle('Natural movie 1 corrected')

        axo.plot(self.rounded_vis_stim['Prairie']['VisStim'])
        axo.plot(self.movie_one_frame_index_full_recording[:,:,0].flatten(), self.rounded_vis_stim['Prairie']['VisStim'][self.movie_one_frame_index_full_recording[:,:,0].flatten()],'rx',label='ups')
        axo.plot(self.movie_one_frame_index_full_recording[:,:,1].flatten(), self.rounded_vis_stim['Prairie']['VisStim'][self.movie_one_frame_index_full_recording[:,:,1].flatten()],'bo',label='downs')
        axo.plot(self.movie_one_trial_full_recording, self.rounded_vis_stim['Prairie']['VisStim'][self.movie_one_trial_full_recording],'co',label='trials')

        plt.show()


#%% ALLEN C PROCESSING                      
    def get_sparse_noise_trial_structure(self):
        
        self.noise_on_index_full_recording =np.zeros((1))
        self.noise_off_index_full_recording=np.zeros((1))
        self.load_noise_indexes()
        
        self.combined_noise_raw=np.concatenate((self.first_noise_set, self.second_noise_set, self.third_noise_set))
        

    def get_movie_two_trial_structure(self):
         
        self.movie_two_frame_index_full_recording =np.zeros((1))
        self.movie_two_frame_indexes_by_trial =np.zeros((1))
        self.movie_two_trial_indexes_by_paradigm=np.zeros((1))

        self.load_movie_two_indexes()
        
        if not any( [self.movie_two_trial_indexes_by_paradigm.any(),self.movie_two_frame_index_full_recording.any(),self.movie_two_frame_indexes_by_trial.any()]):
            to_process=self.natural_movie_two_set
            if self.acquisition_name=='220213_SPJZ_FOV1_AllenC_20x_940_52570_narrow_with-000':
                #correct the bad voltage
                rounded_trace,_,_,_ = self.correct_voltage_split_transitions(self.natural_movie_two_set,plot=True)
    
               
                oldv=np.round(np.flip(np.arange(0,8.8,0.8)),3)
                newv=np.append(np.round(np.flip(np.arange(5.5,10.5,0.5)),3),3)
                
               
                for i,j in zip(oldv,newv):
                    fix, ax=plt.subplots(1)
                    fix.suptitle(f'Natural movie 1 correcting {i} to {j}')
                    ax.plot( rounded_trace,label='Volatage')
                    rounded_trace[rounded_trace==i]=j
                    print(i)
                    print(j)
               
                    ax.plot( rounded_trace,label='New Volatage')
                    ax.legend()
                    plt.show()
               
               
               
                temp=np.diff(rounded_trace)
                temp2=np.around(temp,3)
               
                fix, ax=plt.subplots(1)
                fix.suptitle('Natural movie 1')
                ax.plot( rounded_trace,label='Volatage')
                # ax.plot(temp,label='Diff')
                ax.plot(temp2,label='Rounded DIff')
                
                ax.legend()
                plt.show()
                to_process=rounded_trace
                startprocess=copy.deepcopy(to_process)
                startprocess[to_process<5.1]=0
               
                self.movie_two_trial_indexes_by_paradigm=sg.find_peaks(np.diff(startprocess),height=5)[0]
                self.movie_two_frame_indexes_by_trial=np.zeros([10,900,2])
               
                self.correct_movie_two_indexes_for_full_movie()
               
                self.save_movie_two_indexes()  
   
    
    def correct_movie_two_indexes_for_full_movie(self):
        
        self.movie_two_frame_index_full_recording = self.movie_two_frame_indexes_by_trial+self.transitions_dictionary['natural_movie_two_set_first']
        self.movie_two_frame_index_full_recording=self.movie_two_frame_index_full_recording.astype('uint32')
        self.movie_two_trial_full_recording= self.movie_two_trial_indexes_by_paradigm+self.transitions_dictionary['natural_movie_two_set_first']
        self.movie_two_trial_full_recording=self.movie_two_trial_full_recording.astype('uint32')
    
    
        fig,axo=plt.subplots()
        fig.suptitle('Natural movie 1 corrected')
    
        axo.plot(self.rounded_vis_stim['Prairie']['VisStim'])
        axo.plot(self.movie_two_frame_index_full_recording[:,:,0].flatten(), self.rounded_vis_stim['Prairie']['VisStim'][self.movie_two_frame_index_full_recording[:,:,0].flatten()],'rx',label='ups')
        axo.plot(self.movie_two_frame_index_full_recording[:,:,1].flatten(), self.rounded_vis_stim['Prairie']['VisStim'][self.movie_two_frame_index_full_recording[:,:,1].flatten()],'bo',label='downs')
        axo.plot(self.movie_two_trial_full_recording, self.rounded_vis_stim['Prairie']['VisStim'][self.movie_two_trial_full_recording],'co',label='trials')
    
        plt.show()

#%% ALLEN B PROCESSING
        
    def get_static_gratings_trial_structure(self):
        trial_voltages=[2,-2]
        voltage=2
  
        self.static_grat_even_index_full_recording =np.zeros((1))
        self.static_grat_odd_index_full_recording=np.zeros((1))
        self.load_static_indexes()
        self.combined_static_raw=np.concatenate((self.first_static_set, self.second_static_set, self.third_static_set))
        
        if not (self.static_grat_even_index_full_recording.any() and self.static_grat_odd_index_full_recording.any()):
            fix, ax=plt.subplots(1)
            fix.suptitle('Static gratings sliced together')
            ax.plot(self.combined_static_raw)
            plt.show()

            temp=np.diff(np.around(sg.medfilt(self.combined_static_raw, kernel_size=29),1))
            temp2=np.around(temp,3)
            static_transition_indexes1=[np.argwhere(temp== voltage) for voltage in trial_voltages]
            static_transition_indexes2=[np.argwhere(temp2== voltage) for voltage in trial_voltages]
            fix, ax=plt.subplots(1)
            fix.suptitle('Static gratings dff')

            ax.plot(temp)
            ax.plot(temp2)
            plt.show()
            self.static_voltage_slice_filtered_rounded_corrected,self.static_diff_voltage_slice_filtered_rounded_corrected,self.static_diff_voltage_slice_filtered_rounded_corrected_rerounded,self.static_errors_pairs = self.correct_voltage_split_transitions(self.combined_static_raw)
            
            initial_transitions_odd=np.argwhere(np.logical_and(self.static_diff_voltage_slice_filtered_rounded_corrected>0.8 , self.static_diff_voltage_slice_filtered_rounded_corrected<2))
    
            
            all_even_transition_indexes=np.argwhere(self.static_diff_voltage_slice_filtered_rounded_corrected== voltage).squeeze() 
            all_even_transition_indexes2=np.argwhere(self.static_diff_voltage_slice_filtered_rounded_corrected_rerounded== voltage).squeeze()  
            all_odd_transition_indexes=np.argwhere(self.static_diff_voltage_slice_filtered_rounded_corrected== -voltage).squeeze()  
            all_odd_transition_indexes2=np.argwhere(self.static_diff_voltage_slice_filtered_rounded_corrected_rerounded== -voltage).squeeze()  
            self.static_even_transition_indexes=all_even_transition_indexes
            self.static_odd_transition_indexes=np.sort(np.append(all_odd_transition_indexes,initial_transitions_odd))
            # self.static_odd_transition_indexes[960]=self.static_odd_transition_indexes[960]+3 #SPJZ allenb
            # self.static_odd_transition_indexes[1920]=self.static_odd_transition_indexes[1920]+1#SPJZ allenb
     

            
            fig,ax=plt.subplots()
            fix.suptitle('Static gratings')

            ax.plot(self.static_voltage_slice_filtered_rounded_corrected)  
            ax.plot(self.static_even_transition_indexes,self.static_voltage_slice_filtered_rounded_corrected[self.static_even_transition_indexes],'ro')
            ax.plot(self.static_odd_transition_indexes,self.static_voltage_slice_filtered_rounded_corrected[self.static_odd_transition_indexes],'gx')
            ax.plot(self.static_odd_transition_indexes[960],self.static_voltage_slice_filtered_rounded_corrected[self.static_odd_transition_indexes[960]],'yo')
            ax.plot(self.static_odd_transition_indexes[1920],self.static_voltage_slice_filtered_rounded_corrected[self.static_odd_transition_indexes[1920]],'yo')
            plt.show()

    
    
            self.correct_static_indexes_for_full_movie()
            # self.static_grat_odd_index_full_recording[960]=self.static_grat_odd_index_full_recording[960]-1#SPJZ allenb
            # self.static_grat_odd_index_full_recording[1920]=self.static_grat_odd_index_full_recording[1920]-1#SPJZ allenb
            # name = input("Say ok to follow natural image indexing: ")

            self.save_static_indexes()
        else:
            fig,axo=plt.subplots()
            fig.suptitle('Static gratings')
            axo.plot(self.rounded_vis_stim['Prairie']['VisStim'])
            axo.plot(self.static_grat_even_index_full_recording, self.rounded_vis_stim['Prairie']['VisStim'][self.static_grat_even_index_full_recording],'rx')
            axo.plot(self.static_grat_odd_index_full_recording, self.rounded_vis_stim['Prairie']['VisStim'][self.static_grat_odd_index_full_recording],'bo')
            plt.show()

    def get_natural_images_trial_structure(self):
        trial_voltages=[2,-2]
        voltage=2
        self.natural_image_even_index_full_recording =np.zeros((1))
        self.natural_image_odd_index_full_recording=np.zeros((1))
        self.load_images_indexes()
        self.combined_images_raw=np.concatenate((self.first_images_set, self.second_images_set, self.third_images_set))
        
        
        
        if not (self.natural_image_even_index_full_recording.any() and self.natural_image_odd_index_full_recording.any()):
            fix, ax=plt.subplots(1)
            fix.suptitle('Natural Images')

            ax.plot(self.combined_images_raw)
            plr.show()
            temp=np.diff(np.around(sg.medfilt(self.combined_images_raw, kernel_size=29),1))
            temp2=np.around(temp,3)
            static_transition_indexes1=[np.argwhere(temp== voltage) for voltage in trial_voltages]
            static_transition_indexes2=[np.argwhere(temp2== voltage) for voltage in trial_voltages]
            fix, ax=plt.subplots(1)
            fix.suptitle('Natural Images dff')

            ax.plot(temp)
            ax.plot(temp2)
            plt.show()
            self.images_voltage_slice_filtered_rounded_corrected,self.images_diff_voltage_slice_filtered_rounded_corrected,self.images_diff_voltage_slice_filtered_rounded_corrected_rerounded,self.images_errors_pairs = self.correct_voltage_split_transitions(self.combined_images_raw)
            
            initial_transitions_odd=np.argwhere(np.logical_and(self.images_diff_voltage_slice_filtered_rounded_corrected>0.8 , self.images_diff_voltage_slice_filtered_rounded_corrected<2))
    
            
            all_even_transition_indexes=np.argwhere(self.images_diff_voltage_slice_filtered_rounded_corrected== voltage).squeeze() 
            all_even_transition_indexes2=np.argwhere(self.images_diff_voltage_slice_filtered_rounded_corrected_rerounded== voltage).squeeze()  
            all_odd_transition_indexes=np.argwhere(self.images_diff_voltage_slice_filtered_rounded_corrected== -voltage).squeeze()  
            all_odd_transition_indexes2=np.argwhere(self.images_diff_voltage_slice_filtered_rounded_corrected_rerounded== -voltage).squeeze()  
            self.natural_image_even_transition_indexes=all_even_transition_indexes
            self.natural_image_odd_transition_indexes=np.sort(np.append(all_odd_transition_indexes,initial_transitions_odd))
            # self.natural_image_odd_transition_indexes[960]=self.natural_image_odd_transition_indexes[960]+2 #SPJZ allenb
            # self.natural_image_odd_transition_indexes[1920]=self.natural_image_odd_transition_indexes[1920]+2#SPJZ allenb
     
            
            
            fig,ax=plt.subplots()
            fix.suptitle('Natural Images ')

            ax.plot(self.images_voltage_slice_filtered_rounded_corrected)  
            ax.plot(self.natural_image_even_transition_indexes,self.images_voltage_slice_filtered_rounded_corrected[self.natural_image_even_transition_indexes],'ro')
            ax.plot(self.natural_image_odd_transition_indexes,self.images_voltage_slice_filtered_rounded_corrected[self.natural_image_odd_transition_indexes],'gx')
            plt.show()
    
    
            self.correct_image_indexes_for_full_movie()
            # self.natural_image_odd_index_full_recording[960]=1298187#SPJZ allenb
            # self.natural_image_odd_index_full_recording[1920]=self.natural_image_odd_index_full_recording[1920]#SPJZ allenb
            # name = input("Say ok to follow natural image indexing: ")
            self.save_images_indexes()
        else:
            fig,axo=plt.subplots()
            fig.suptitle('Natural Images ')

            axo.plot(self.rounded_vis_stim['Prairie']['VisStim'])
            axo.plot(self.natural_image_even_index_full_recording, self.rounded_vis_stim['Prairie']['VisStim'][self.natural_image_even_index_full_recording],'rx')
            axo.plot(self.natural_image_odd_index_full_recording, self.rounded_vis_stim['Prairie']['VisStim'][self.natural_image_odd_index_full_recording],'bo')
            plt.show()

    def correct_static_indexes_for_full_movie(self):
        first_length=self.first_static_set.shape[0]
        second_length=self.second_static_set.shape[0]
        mivies_indexes=[self.transitions_dictionary['first_static_set_first'],
                        self.transitions_dictionary['first_static_set_last'],
                        self.transitions_dictionary['second_static_set_first'],
                        self.transitions_dictionary['second_static_set_last'],
                        self.transitions_dictionary['third_static_set_first'],
                        self.transitions_dictionary['third_static_set_last']]
        
        def correctindex(indx, first_length, second_length, mivies_indexes0, mivies_indexes2, mivies_indexes4,):
            if indx < first_length:
                return indx + mivies_indexes0
            elif np.logical_and(indx<first_length+second_length,indx>first_length) :
                return indx + mivies_indexes2-first_length
            elif indx > second_length:
                return indx + mivies_indexes4-first_length-second_length
            
        vfunc = np.vectorize(correctindex)
       
        self.static_grat_even_index_full_recording = vfunc(self.static_even_transition_indexes, first_length, second_length,  mivies_indexes[0], mivies_indexes[2], mivies_indexes[4])
        self.static_grat_odd_index_full_recording = vfunc(self.static_odd_transition_indexes, first_length, second_length,  mivies_indexes[0], mivies_indexes[2], mivies_indexes[4])
        
        

        fig,axo=plt.subplots()
        fig.suptitle('Static gratings corrected')

        axo.plot(self.rounded_vis_stim['Prairie']['VisStim'])
        axo.plot(self.static_grat_even_index_full_recording, self.rounded_vis_stim['Prairie']['VisStim'][self.static_grat_even_index_full_recording],'rx')
        axo.plot(self.static_grat_odd_index_full_recording, self.rounded_vis_stim['Prairie']['VisStim'][self.static_grat_odd_index_full_recording],'bo')
        plt.show()

    def correct_image_indexes_for_full_movie(self):
        first_length=self.first_images_set.shape[0]
        second_length=self.second_images_set.shape[0]
        mivies_indexes=[self.transitions_dictionary['first_images_set_first'],
                        self.transitions_dictionary['first_images_set_last'],
                        self.transitions_dictionary['second_images_set_first'],
                        self.transitions_dictionary['second_images_set_last'],
                        self.transitions_dictionary['third_images_set_first'],
                        self.transitions_dictionary['third_images_set_last']]
        
        def correctindex(indx, first_length, second_length, mivies_indexes0, mivies_indexes2, mivies_indexes4,):
            if indx < first_length:
                return indx + mivies_indexes0
            elif np.logical_and(indx<first_length+second_length,indx>first_length) :
                return indx + mivies_indexes2-first_length
            elif indx > second_length:
                return indx + mivies_indexes4-first_length-second_length
            
        vfunc = np.vectorize(correctindex)
       
        self.natural_image_even_index_full_recording = vfunc(self.natural_image_even_transition_indexes, first_length, second_length,  mivies_indexes[0], mivies_indexes[2], mivies_indexes[4])
        self.natural_image_odd_index_full_recording = vfunc(self.natural_image_odd_transition_indexes, first_length, second_length,  mivies_indexes[0], mivies_indexes[2], mivies_indexes[4])
        
        
    
        fig,axo=plt.subplots()
        fig.suptitle('Natural Images corrected ')

        axo.plot(self.rounded_vis_stim['Prairie']['VisStim'])
        axo.plot(self.natural_image_even_index_full_recording, self.rounded_vis_stim['Prairie']['VisStim'][self.natural_image_even_index_full_recording],'rx')
        axo.plot(self.natural_image_odd_index_full_recording, self.rounded_vis_stim['Prairie']['VisStim'][self.natural_image_odd_index_full_recording],'bo')
        plt.show()
            
        
#%% ALLEN A PROCESSING
    def get_movie_three_trial_structure(self):

        self.movie_three_frame_index_full_recording =np.zeros((2,1)).astype('int')
        self.movie_three_frame_indexes_by_trial =np.zeros((2,1)).astype('int')
        self.movie_three_trial_indexes_by_paradigm=np.zeros((2,5)).astype('int')

        self.load_movie_three_indexes()
        
        self.natural_movie_three_first_set
        self.natural_movie_three_second_set
        
        if not any( [self.movie_three_trial_indexes_by_paradigm.any(),self.movie_three_frame_index_full_recording.any(),self.movie_three_frame_indexes_by_trial.any()]):
            
            all_startprocess=[] 
            for to_process in [ self.natural_movie_three_first_set, self.natural_movie_three_second_set]:
                if self.acquisition_name=='220217_SPJZ_FOV1_AllenA_20x_920_52570_narrow_with-000':
                    #correct the bad voltage
                    rounded_trace,_,_,_ = self.correct_voltage_split_transitions(to_process,plot=False)
        
                   
                    oldv=[0]
                    newv=[1]
                    
                  
                    for i,j in zip(oldv,newv):
                        fix, ax=plt.subplots(1)
                        fix.suptitle(f'Natural movie 1 correcting {i} to {j}')
                        ax.plot( rounded_trace,label='Volatage')
                        rounded_trace[rounded_trace==i]=j
                        print(i)
                        print(j)
                   
                        ax.plot( rounded_trace,label='New Volatage')
                        ax.legend()
                        plt.show()
                   
                   
                   
                    temp=np.diff(rounded_trace)
                    temp2=np.around(temp,3)
                   
                    fix, ax=plt.subplots(1)
                    fix.suptitle('Natural movie 1')
                    ax.plot( rounded_trace,label='Volatage')
                    # ax.plot(temp,label='Diff')
                    ax.plot(temp2,label='Rounded DIff')
                    
                    ax.legend()
                    plt.show()
                    to_process=rounded_trace
                    startprocess=copy.deepcopy(to_process)
                    startprocess[to_process<3.5]=0
                    
                    fix, ax=plt.subplots(1)
                    fix.suptitle('Natural movie 1')
                    ax.plot( rounded_trace,label='Volatage')
                    ax.plot(startprocess,label='Cleaned')
                    ax.legend()
                    plt.show()
                    all_startprocess.append(startprocess)
                    
            for i,para in enumerate(all_startprocess):
                self.movie_three_trial_indexes_by_paradigm[i,:]=sg.find_peaks(np.diff(para),height=3.1)[0]
            self.movie_three_frame_indexes_by_trial=np.zeros([2,10,900,2])
           
            self.correct_movie_three_indexes_for_full_movie()
           
            self.save_movie_three_indexes()  
   
            
    def correct_movie_three_indexes_for_full_movie(self):
        self.movie_three_frame_index_full_recording = np.zeros_like(self.movie_three_frame_indexes_by_trial).astype('uint32')
        self.movie_three_trial_full_recording= np.zeros_like(self.movie_three_trial_indexes_by_paradigm).astype('uint32')
        
        
        for st, na in enumerate(   ['natural_movie_three_first_set_first','natural_movie_three_second_set_first']):
        
            self.movie_three_frame_index_full_recording[st,:,:,:] = self.movie_three_frame_indexes_by_trial[st,:,:,:] +self.transitions_dictionary[na]
            self.movie_three_trial_full_recording[st,:]= self.movie_three_trial_indexes_by_paradigm[st,:]+self.transitions_dictionary[na]


        fig,axo=plt.subplots()
        fig.suptitle('Natural movie 3 corrected')

        axo.plot(self.rounded_vis_stim['Prairie']['VisStim'])
        # axo.plot(self.movie_three_frame_index_full_recording[:,:,0].flatten(), self.rounded_vis_stim['Prairie']['VisStim'][self.movie_three_frame_index_full_recording[:,:,0].flatten()],'rx',label='ups')
        # axo.plot(self.movie_three_frame_index_full_recording[:,:,1].flatten(), self.rounded_vis_stim['Prairie']['VisStim'][self.movie_three_frame_index_full_recording[:,:,1].flatten()],'bo',label='downs')
        axo.plot(self.movie_three_trial_full_recording[0,:], self.rounded_vis_stim['Prairie']['VisStim'][self.movie_three_trial_full_recording][0,:],'co',label='trials')
        axo.plot(self.movie_three_trial_full_recording[1,:], self.rounded_vis_stim['Prairie']['VisStim'][self.movie_three_trial_full_recording][0,:],'co',label='trials')
        plt.show()


    def correct_voltage_split_transitions(self, voltage_slice, kernel_size=29,plot=False, decimals=1):
        # this is for transition that were split betwen 2 samples, I always get the inital transition to the sample with at tleast some 
        # voltage as voltages is send after the image and for the end transition i get the also the first as the image has chnaged before voltage change
        voltage_slice_filtered=sg.medfilt(voltage_slice, kernel_size=kernel_size)
        voltage_slice_filtered_rounded=np.around(voltage_slice_filtered, decimals)
        voltage_slice_filtered_rounded_corrected=np.copy(voltage_slice_filtered_rounded)

        diff_voltage_slice_filtered_rounded= np.diff(voltage_slice_filtered_rounded)
        diff_voltage_slice_filtered_rounded_rerounded =np.around(diff_voltage_slice_filtered_rounded, decimals)

        #correcting  voltage transitions betwen samples
        errors_pairs=[]
        for i in range(0, diff_voltage_slice_filtered_rounded_rerounded.size-2):
            if diff_voltage_slice_filtered_rounded_rerounded[i+1]!=0 and diff_voltage_slice_filtered_rounded_rerounded[i]!=0 and diff_voltage_slice_filtered_rounded_rerounded[i+1]*diff_voltage_slice_filtered_rounded_rerounded[i]>0:
                errors_pairs.append((i+1, i+2))
                voltage_slice_filtered_rounded_corrected[i+1]=voltage_slice_filtered_rounded_corrected[i+2]
                voltage_slice_filtered_rounded_corrected[i+2]=voltage_slice_filtered_rounded_corrected[i+3]
                voltage_slice[i+1]=voltage_slice[i+2]
                voltage_slice[i+2]=voltage_slice[i+3]
                
                
        diff_voltage_slice_filtered_rounded_corrected = np.diff(voltage_slice_filtered_rounded_corrected)
        diff_voltage_slice_filtered_rounded_corrected_rerounded =np.around(diff_voltage_slice_filtered_rounded_corrected, decimals)    
        transitions_list=[voltage_slice_filtered_rounded,diff_voltage_slice_filtered_rounded_corrected, diff_voltage_slice_filtered_rounded_corrected_rerounded, errors_pairs]
        if plot:
            self.check_split_transitions(voltage_slice,transitions_list)
        return  voltage_slice_filtered_rounded_corrected,\
                diff_voltage_slice_filtered_rounded_corrected,\
                diff_voltage_slice_filtered_rounded_corrected_rerounded, \
                errors_pairs

        
    def check_split_transitions(self,voltage_slice,transitions_list,decimals=1): 
        voltage_slice_filtered_rounded_corrected, diff_voltage_slice_filtered_rounded_corrected, diff_voltage_slice_filtered_rounded_corrected_rerounded, errors_pairs=transitions_list
        voltage_slice_filtered=sg.medfilt(voltage_slice, kernel_size=1)
        voltage_slice_filtered_rounded=np.around(voltage_slice_filtered, decimals)
        voltage_slice_filtered_rounded_corrected=np.copy(voltage_slice_filtered_rounded)
        diff_voltage_slice_filtered_rounded= np.diff(voltage_slice_filtered_rounded)
        diff_voltage_slice_filtered_rounded_rerounded =np.around(diff_voltage_slice_filtered_rounded, decimals)
        
        f,ax=plt.subplots(2,sharex=True)
        ax[0].plot(voltage_slice,label='Raw')
        ax[0].plot(voltage_slice_filtered_rounded,label='Filtered rounded')
        ax[0].plot(voltage_slice_filtered_rounded_corrected,label='Filtered rounded Corrected')
        ax[0].legend()
        ax[1].plot(diff_voltage_slice_filtered_rounded,label='Raw')
        ax[1].legend()

        maxplots=8
        figures=int(np.ceil(len(errors_pairs)/maxplots))
        
        for n in range(figures):
            indexes=np.arange(n*maxplots,(n+1)*maxplots,1, 'int')
            if n==figures-1:
                indexes=np.arange(n*maxplots,len(errors_pairs),1, 'int')
            fig,axa=plt.subplots(len(indexes), sharex=True)
            if not isinstance(axa,np.ndarray):
                axa=[axa]
            for i in  range(len(indexes)) :  
                l1=axa[i].plot(diff_voltage_slice_filtered_rounded_rerounded[errors_pairs[indexes[i]][0]-10:errors_pairs[indexes[i]][0]+10])
                l2=axa[i].plot(diff_voltage_slice_filtered_rounded_corrected_rerounded[errors_pairs[indexes[i]][0]-10:errors_pairs[indexes[i]][0]+10])
    
            fig,axo=plt.subplots(len(indexes), sharex=True)
            if not isinstance(axa,np.ndarray):
                axo=[axo]
            for i  in range(len(indexes)) :  
                axo[i].plot(voltage_slice_filtered_rounded[errors_pairs[indexes[i]][0]-10:errors_pairs[indexes[i]][0]+10])
                axo[i].plot(voltage_slice_filtered_rounded_corrected[errors_pairs[indexes[i]][0]-10:errors_pairs[indexes[i]][0]+10])
          
    
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
        
        # concatenati drifitng gratings
        self.combined_gratings_raw=np.concatenate((self.first_drifting_set, self.second_drifting_set, self.third_drifting_set))

        
        if not (self.tuning_stim_on_index_full_recording.any() and self.tuning_stim_off_index_full_recording.any()):

            
            if self.acquisition_name=='220217_SPJZ_FOV1_AllenA_20x_920_52570_narrow_with-000':
                             
                
                voltages=np.append(np.arange(4,5,0.025),9.5)
                filtered_and_round,_,_,_=self.correct_voltage_split_transitions(self.combined_gratings_raw,kernel_size=29,decimals=3)
                onsetsonly=copy.deepcopy(filtered_and_round)
                onsetsonly[onsetsonly<3.1]=0

                dif=np.diff(onsetsonly)
                starts=sg.find_peaks(-dif,height=3)[0]
                             
                
                fix, ax=plt.subplots(1)
                ax.plot(filtered_and_round)
                # ax.plot(self.combined_gratings_raw)
                # ax.plot(dif)
                # ax.plot(onsetsonly)
                for y in starts:
                    # ax.plot(y,dif[y],'cx')
                    ax.plot(y,filtered_and_round[y],'co')


                for y in voltages:
                    ax.axhline(y = y,color='cyan') 
                    
                    
                onsets=np.zeros_like(starts)   
                offsets=np.zeros_like(starts)   
                identities=np.zeros_like(starts)   
                plt.close('all')
                for i,trial in  enumerate(starts):
                    # f,ax=plt.subplots()
                    # ax.plot(np.abs(np.diff(filtered_and_round[trial+1:trial+3200])))
                    peaks=sg.find_peaks(np.abs(np.diff(filtered_and_round[trial+1:trial+1200])),height=0.25)[0]+trial+1
                    offpeaks=sg.find_peaks(np.abs(np.diff(filtered_and_round[trial+1:trial+3200])),height=0.25)[0]+trial+1

                    if peaks.any():
                        onsets[i]=peaks[0]
                        if np.abs(np.abs(np.diff(filtered_and_round[trial+1:trial+3200]))[offpeaks[-3]-trial-1]-np.abs(np.diff(filtered_and_round[trial+1:trial+3200]))[offpeaks[-4]-trial-1])<0.1:
                            offsets[i]=offpeaks[-2]
                        else:
                            offsets[i]=offpeaks[-3]
                            
                        identities[i]= np.argmin(np.abs([onsetsonly[trial]-vol for vol in voltages]))+1
                    else:
                        onsets[i]=  trial+onsets[i-1]-starts[i-1]
                        offsets[i]= trial+offsets[i-1]-starts[i-1]

                        identities[i]=0
                        
                   
                fix, ax=plt.subplots(1)
                ax.plot(filtered_and_round)

                for y in starts:
                    ax.plot(y,filtered_and_round[y],'co')
                    
                for y in onsets:
                    ax.plot(y,filtered_and_round[y],'cx')
                
                for y in offsets:
                    ax.plot(y,filtered_and_round[y],'mx')
                
                
                self.drifting_on_transition_indexes=np.zeros([40,15]).astype('int')
                self.drifting_off_transition_indexes=np.zeros([40,15]).astype('int')
                self.blank_on_transition_indexes=np.empty([0]).astype('int')
                self.blank_off_transition_indexes=np.empty([0]).astype('int')

                
                reps=np.zeros([40]).astype('int')
                for i,j in enumerate(identities):
                    if j!=0:
                        self.drifting_on_transition_indexes[j-1, reps[j-1]]=onsets[i]
                        self.drifting_off_transition_indexes[j-1, reps[j-1]]=offsets[i]
                        reps[j-1]=reps[j-1]+1

                    else:
                        self.blank_on_transition_indexes=np.append(self.blank_on_transition_indexes,onsets[i])
                        self.blank_off_transition_indexes=np.append(self.blank_off_transition_indexes,offsets[i])


                self.blank_on_transition_indexes=np.expand_dims(self.blank_on_transition_indexes,1)
                self.blank_off_transition_indexes=np.expand_dims(self.blank_off_transition_indexes,1)

   

                

            
            else:
                
                # round and first derivative two methods
               
                temp=np.diff(np.around(sg.medfilt(self.combined_gratings_raw, kernel_size=29),1))

                temp2=np.around(temp,3)
                drifting_transition_indexes1=[np.argwhere(temp== voltage) for voltage in self.orientations_and_blank_sweep]
                drifting_transition_indexes2=[np.argwhere(temp2== voltage) for voltage in self.orientations_and_blank_sweep]
                fix, ax=plt.subplots(1)
                ax.plot(temp)
                # ax.plot(temp2)
                for y in np.arange(4,5.1,0.1):
                    ax.axhline(y = y,color='cyan') 
    
    
                
                # this will correct transtions over multiple frame to the corresponding first frame that has a change of voltage(all voltage signals appear after screen flip)
                
                self.drifting_voltage_slice_filtered_rounded_corrected,\
                self.drifting_diff_voltage_slice_filtered_rounded_corrected,\
                self.drifting_diff_voltage_slice_filtered_rounded_corrected_rerounded,\
                self.drifting_errors_pairs=self.correct_voltage_split_transitions(self.combined_gratings_raw)
    
                all_on_transition_indexes=[np.argwhere(self.drifting_diff_voltage_slice_filtered_rounded_corrected== voltage).squeeze() for voltage in self.orientations_and_blank_sweep]
                all_on_transition_indexes2=[np.argwhere(self.drifting_diff_voltage_slice_filtered_rounded_corrected_rerounded== voltage) for voltage in self.orientations_and_blank_sweep]
                all_off_transition_indexes=[np.argwhere(self.drifting_diff_voltage_slice_filtered_rounded_corrected== -voltage).squeeze()  for voltage in self.orientations_and_blank_sweep]
                all_off_transition_indexes2=[np.argwhere(self.drifting_diff_voltage_slice_filtered_rounded_corrected_rerounded== -voltage) for voltage in self.orientations_and_blank_sweep]
                self.drifting_on_transition_indexes=np.vstack( all_on_transition_indexes[0:len(self.orientations)])+1
                self.drifting_off_transition_indexes=np.vstack( all_off_transition_indexes[0:len(self.orientations)])+1
                self.blank_on_transition_indexes=np.vstack( all_on_transition_indexes[-1])+1
                self.blank_off_transition_indexes=np.vstack( all_off_transition_indexes[-1])+1
                
            oritoplot=40
            maxplots=8
            figures=5
            for n in range(figures):
                indexes=np.arange(n*maxplots,(n+1)*maxplots,1, 'int')
                fig,axa=plt.subplots(len(indexes), sharex=True)
                for i in  range(len(indexes)) :  
                    for j  in  range(15) :  
                        axa[i].plot(self.drifting_voltage_slice_filtered_rounded_corrected[self.drifting_on_transition_indexes[indexes[i],j]-600:self.drifting_off_transition_indexes[indexes[i],j]+600],'r')
                        
         
                fig,axo=plt.subplots(len(indexes), sharex=True)
                for i in  range(len(indexes)) :  
                    for j  in  range(15) :  
                        axo[i].plot(self.drifting_voltage_slice_filtered_rounded_corrected[self.drifting_off_transition_indexes[indexes[i],j]-2000:self.drifting_off_transition_indexes[indexes[i],j]+2000],'b')




            fig,ax=plt.subplots(2)
            ax[0].plot(self.drifting_diff_voltage_slice_filtered_rounded_corrected)
            ax[0].plot(self.drifting_diff_voltage_slice_filtered_rounded_corrected_rerounded)


            stim_period_lengths=self.drifting_off_transition_indexes-self.drifting_on_transition_indexes
            # plt.hist(stim_period_lengths.flatten(), bins=np.arange(2047, 2051), range=(2047,2051))
            plt.hist(stim_period_lengths.flatten())

            plt.show()
       
            index_start=  self.drifting_on_transition_indexes[-1]
            index_end= self.drifting_off_transition_indexes[-1]
            
            fig,axa=plt.subplots(1)
            for i ,j in  enumerate(index_start) : 
                axa.plot(self.drifting_voltage_slice_filtered_rounded_corrected[j-10:j+2500])
                
            fig,axo=plt.subplots(1)
            for i ,j in  enumerate(index_end) :  
                axo.plot(self.drifting_voltage_slice_filtered_rounded_corrected[j-2500:j+10])

            self.correct_grating_indexes_for_full_movie()

            self.save_drifting_grating_indexes()
                
            # self.create_full_recording_grating_binary_matrix()



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
        
        self.blank_sweep_on_index_full_recording = vfunc(self.blank_on_transition_indexes, first_length, second_length,  mivies_indexes[0], mivies_indexes[2], mivies_indexes[4])
        self.blank_sweep_off_index_full_recording = vfunc(self.blank_off_transition_indexes, first_length, second_length,  mivies_indexes[0], mivies_indexes[2], mivies_indexes[4])
        

        oritoplot=40
        fig,axo=plt.subplots(oritoplot, sharex=(True))
        for k, j in enumerate(range(oritoplot)):
            for i  in  range(10) :  
                axo[k].plot(self.rounded_vis_stim['Prairie']['VisStim'][self.tuning_stim_on_index_full_recording[j,i]-600:self.tuning_stim_off_index_full_recording[j,i]+600])
                
     
    def create_full_recording_grating_binary_matrix(self):

        self.full_stimuli_binary_matrix=np.zeros((self.orientations.size,self.visualstim_array['Prairie']['VisStim'].shape[0] ))
        for i, row in enumerate(self.tuning_stim_on_index_full_recording):
            for j, trial in enumerate(row):
                self.full_stimuli_binary_matrix[i, self.tuning_stim_on_index_full_recording[i,j]:self.tuning_stim_off_index_full_recording[i,j]]=1


        

#%% TO DO MISC
    def confirm_grating_indexes(self):
        module_logger.info('todo')
        # confirm number of trial
        # confirm length of ranges
        
    def check_csv_in_folder(self):
     if self.acquisition_directory_raw:
            csvfiles=glob.glob(self.acquisition_directory_raw+os.sep+'**.csv')
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
        if  not os.path.isfile(self.indexes_full_file_paths_to_save[0]):
            with open(  self.indexes_full_file_paths_to_save[0], 'wb') as f:
                pickle.dump(self.transitions_dictionary, f, pickle.HIGHEST_PROTOCOL)

    def load_indexes_from_file(self):
        if os.path.isfile(self.indexes_full_file_paths_to_save[0]):
            with open(self.indexes_full_file_paths_to_save[0], 'rb') as f:
                self.transitions_dictionary= pickle.load(f)
                
                
                
                
    def save_drifting_grating_indexes(self):
   
        with open( self.indexes_full_file_paths_to_save[1], 'wb') as f:
            pickle.dump(self.tuning_stim_on_index_full_recording, f, pickle.HIGHEST_PROTOCOL)

        with open(self.indexes_full_file_paths_to_save[2], 'wb') as f:
            pickle.dump(self.tuning_stim_off_index_full_recording, f, pickle.HIGHEST_PROTOCOL)
            
        with open( self.indexes_full_file_paths_to_save[3], 'wb') as f:
            pickle.dump(self.drifting_on_transition_indexes, f, pickle.HIGHEST_PROTOCOL)

        with open(self.indexes_full_file_paths_to_save[4], 'wb') as f:
            pickle.dump(self.drifting_off_transition_indexes, f, pickle.HIGHEST_PROTOCOL)
            
            
            
        with open( self.indexes_full_file_paths_to_save[5], 'wb') as f:
            pickle.dump(self.blank_sweep_on_index_full_recording, f, pickle.HIGHEST_PROTOCOL)

        with open(self.indexes_full_file_paths_to_save[6], 'wb') as f:
            pickle.dump(self.blank_sweep_off_index_full_recording, f, pickle.HIGHEST_PROTOCOL)
        
        with open( self.indexes_full_file_paths_to_save[7], 'wb') as f:
            pickle.dump(self.blank_on_transition_indexes, f, pickle.HIGHEST_PROTOCOL)

        with open(self.indexes_full_file_paths_to_save[8], 'wb') as f:
            pickle.dump(self.blank_off_transition_indexes, f, pickle.HIGHEST_PROTOCOL)
            
   

    def load_drifting_grating_indexes(self):
        if os.path.isfile(self.indexes_full_file_paths_to_save[1]):
            with open(self.indexes_full_file_paths_to_save[1], 'rb') as f:
                self.tuning_stim_on_index_full_recording=pickle.load(f)

        if os.path.isfile(self.indexes_full_file_paths_to_save[2]):
            with open( self.indexes_full_file_paths_to_save[2], 'rb') as f:
                self.tuning_stim_off_index_full_recording=pickle.load(f)
                
                
        if os.path.isfile(self.indexes_full_file_paths_to_save[3]):
            with open(self.indexes_full_file_paths_to_save[3], 'rb') as f:
                self.drifting_on_transition_indexes=pickle.load(f)

        if os.path.isfile(self.indexes_full_file_paths_to_save[4]):
            with open( self.indexes_full_file_paths_to_save[4], 'rb') as f:
                self.drifting_off_transition_indexes=pickle.load(f)
   
    
   
        if os.path.isfile(self.indexes_full_file_paths_to_save[5]):
            with open(self.indexes_full_file_paths_to_save[5], 'rb') as f:
                self.blank_sweep_on_index_full_recording=pickle.load(f)
    
        if os.path.isfile(self.indexes_full_file_paths_to_save[6]):
            with open( self.indexes_full_file_paths_to_save[6], 'rb') as f:
                self.blank_sweep_off_index_full_recording=pickle.load(f)
                
                
        if os.path.isfile(self.indexes_full_file_paths_to_save[7]):
            with open(self.indexes_full_file_paths_to_save[7], 'rb') as f:
                self.blank_on_transition_indexes=pickle.load(f)
    
        if os.path.isfile(self.indexes_full_file_paths_to_save[8]):
            with open( self.indexes_full_file_paths_to_save[8], 'rb') as f:
                self.blank_off_transition_indexes=pickle.load(f)

   
           
        
    def load_noise_indexes(self):
        pass
    
    def save_noise_indexes(self):
        pass
    
    def load_movie_one_indexes(self):
        
        
        if os.path.isfile(self.indexes_full_file_paths_to_save[17]):
            with open(self.indexes_full_file_paths_to_save[17], 'rb') as f:
                self.movie_one_frame_index_full_recording=pickle.load(f)
                
                
                                
        if os.path.isfile(self.indexes_full_file_paths_to_save[18]):
            with open(self.indexes_full_file_paths_to_save[18], 'rb') as f:
                self.movie_one_frame_indexes_by_trial=pickle.load(f)
                
                             
        if os.path.isfile(self.indexes_full_file_paths_to_save[26]):
            with open(self.indexes_full_file_paths_to_save[26], 'rb') as f:
                self.movie_one_trial_full_recording=pickle.load(f)
        
    
    def save_movie_one_indexes(self):
        
        
        with open( self.indexes_full_file_paths_to_save[17], 'wb') as f:
            pickle.dump(self.movie_one_frame_index_full_recording, f, pickle.HIGHEST_PROTOCOL)

        with open(self.indexes_full_file_paths_to_save[18], 'wb') as f:
            pickle.dump(self.movie_one_frame_indexes_by_trial, f, pickle.HIGHEST_PROTOCOL)
            
        with open(self.indexes_full_file_paths_to_save[26], 'wb') as f:
            pickle.dump(self.movie_one_trial_full_recording, f, pickle.HIGHEST_PROTOCOL)
            
            
        
    def load_movie_two_indexes(self):
        
        
        if os.path.isfile(self.indexes_full_file_paths_to_save[19]):
            with open(self.indexes_full_file_paths_to_save[19], 'rb') as f:
                self.movie_two_frame_index_full_recording=pickle.load(f)
                
                
                                
        if os.path.isfile(self.indexes_full_file_paths_to_save[20]):
            with open(self.indexes_full_file_paths_to_save[20], 'rb') as f:
                self.movie_two_frame_indexes_by_trial=pickle.load(f)
                
                             
        if os.path.isfile(self.indexes_full_file_paths_to_save[27]):
            with open(self.indexes_full_file_paths_to_save[27], 'rb') as f:
                self.movie_two_trial_full_recording=pickle.load(f)
        
    
    def save_movie_two_indexes(self):
        
        
        with open( self.indexes_full_file_paths_to_save[19], 'wb') as f:
            pickle.dump(self.movie_two_frame_index_full_recording, f, pickle.HIGHEST_PROTOCOL)

        with open(self.indexes_full_file_paths_to_save[20], 'wb') as f:
            pickle.dump(self.movie_two_frame_indexes_by_trial, f, pickle.HIGHEST_PROTOCOL)
            
        with open(self.indexes_full_file_paths_to_save[27], 'wb') as f:
            pickle.dump(self.movie_two_trial_full_recording, f, pickle.HIGHEST_PROTOCOL)
            
    def load_movie_three_indexes(self):
        
        
        if os.path.isfile(self.indexes_full_file_paths_to_save[21]):
            with open(self.indexes_full_file_paths_to_save[21], 'rb') as f:
                self.movie_three_frame_index_full_recording=pickle.load(f)
                
                
                                
        if os.path.isfile(self.indexes_full_file_paths_to_save[22]):
            with open(self.indexes_full_file_paths_to_save[22], 'rb') as f:
                self.movie_three_frame_indexes_by_trial=pickle.load(f)
                
                             
                             
        if os.path.isfile(self.indexes_full_file_paths_to_save[28]):
            with open(self.indexes_full_file_paths_to_save[28], 'rb') as f:
                self.movie_three_trial_full_recording=pickle.load(f)
   
        
    
    def save_movie_three_indexes(self):
        
        
        with open( self.indexes_full_file_paths_to_save[21], 'wb') as f:
            pickle.dump(self.movie_three_frame_index_full_recording, f, pickle.HIGHEST_PROTOCOL)

        with open(self.indexes_full_file_paths_to_save[22], 'wb') as f:
            pickle.dump(self.movie_three_frame_indexes_by_trial, f, pickle.HIGHEST_PROTOCOL)
            
        with open(self.indexes_full_file_paths_to_save[28], 'wb') as f:
            pickle.dump(self.movie_three_trial_full_recording, f, pickle.HIGHEST_PROTOCOL)

  
    
    def load_static_indexes(self):
        if os.path.isfile(self.indexes_full_file_paths_to_save[9]):
            with open(self.indexes_full_file_paths_to_save[9], 'rb') as f:
                self.static_grat_even_index_full_recording=pickle.load(f)

        if os.path.isfile(self.indexes_full_file_paths_to_save[10]):
            with open( self.indexes_full_file_paths_to_save[10], 'rb') as f:
                self.static_grat_odd_index_full_recording=pickle.load(f)
                
                
        if os.path.isfile(self.indexes_full_file_paths_to_save[11]):
            with open(self.indexes_full_file_paths_to_save[11], 'rb') as f:
                self.static_even_transition_indexes=pickle.load(f)

        if os.path.isfile(self.indexes_full_file_paths_to_save[12]):
            with open( self.indexes_full_file_paths_to_save[12], 'rb') as f:
                self.static_odd_transition_indexes=pickle.load(f)
   
    
    def save_static_indexes(self):
     
        
        with open( self.indexes_full_file_paths_to_save[9], 'wb') as f:
            pickle.dump(self.static_grat_even_index_full_recording, f, pickle.HIGHEST_PROTOCOL)

        with open(self.indexes_full_file_paths_to_save[10], 'wb') as f:
            pickle.dump(self.static_grat_odd_index_full_recording, f, pickle.HIGHEST_PROTOCOL)
            
        with open( self.indexes_full_file_paths_to_save[11], 'wb') as f:
            pickle.dump(self.static_even_transition_indexes, f, pickle.HIGHEST_PROTOCOL)

        with open(self.indexes_full_file_paths_to_save[12], 'wb') as f:
            pickle.dump(self.static_odd_transition_indexes, f, pickle.HIGHEST_PROTOCOL)
    
    def load_images_indexes(self):
        if os.path.isfile(self.indexes_full_file_paths_to_save[13]):
            with open(self.indexes_full_file_paths_to_save[13], 'rb') as f:
                self.natural_image_even_index_full_recording=pickle.load(f)

        if os.path.isfile(self.indexes_full_file_paths_to_save[14]):
            with open( self.indexes_full_file_paths_to_save[14], 'rb') as f:
                self.natural_image_odd_index_full_recording=pickle.load(f)

        if os.path.isfile(self.indexes_full_file_paths_to_save[15]):
            with open(self.indexes_full_file_paths_to_save[15], 'rb') as f:
                self.natural_image_even_transition_indexes=pickle.load(f)

        if os.path.isfile(self.indexes_full_file_paths_to_save[16]):
            with open( self.indexes_full_file_paths_to_save[16], 'rb') as f:
                self.natural_image_odd_transition_indexes=pickle.load(f)
    
    def save_images_indexes(self):
        
        with open( self.indexes_full_file_paths_to_save[13], 'wb') as f:
            pickle.dump(self.natural_image_even_index_full_recording, f, pickle.HIGHEST_PROTOCOL)

        with open(self.indexes_full_file_paths_to_save[14], 'wb') as f:
            pickle.dump(self.natural_image_odd_index_full_recording, f, pickle.HIGHEST_PROTOCOL)
            
        with open( self.indexes_full_file_paths_to_save[15], 'wb') as f:
            pickle.dump(self.natural_image_even_transition_indexes, f, pickle.HIGHEST_PROTOCOL)

        with open(self.indexes_full_file_paths_to_save[16], 'wb') as f:
            pickle.dump(self.natural_image_odd_transition_indexes, f, pickle.HIGHEST_PROTOCOL)
         
         
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
        line, = ax.plot(self.time_scale['Prairie'],self.visualstim_array['Prairie']['VisStim']) 
        
    def plot_stim_and_speed(self):
        fig, ax = plt.subplots(2)
        line, = ax[0].plot(self.time_scale['Prairie'],self.visualstim_array['Prairie']['VisStim']) 
        line, = ax[1].plot(self.time_scale['Prairie'],self.rectified_speed_array['Prairie']['Locomotion']) 
                
    def plot_full_locomotion(self):
  
        fig, ax = plt.subplots(nrows=3,sharey=True)
        # fig.set_title('Snapping cursor')
        for i in range(0,3):
            if i==0:
                line, = ax[i].plot(self.time_scale['Prairie'],self.locomotion_array['Prairie']['Locomotion']) 
            elif i==1:
                line, = ax[i].plot(self.time_scale['Prairie'],self.rectified_speed_array['Prairie']['Locomotion'])  
            elif i==2:
                line, = ax[i].plot(self.time_scale['Prairie'],self.rectified_acceleration_array['Prairie']['Locomotion'])  
         
        # snap_cursor = SnappingCursor(ax[0], line)  
        # fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
        mplcursors.cursor(line) # or just mplcursors.cursor()
        
    def plot_speed(self):
        fig, ax = plt.subplots(1)
        
        line, = ax.plot(self.time_scale['Prairie'],self.rectified_speed_array['Prairie']['Locomotion']) 
        
    def plotting_paradigm_transitions(self):   
        
        if self.vis_stim_protocol=='AllenA':
            datasets_to_plot=[self.first_drifting_set, self.second_drifting_set, self.third_drifting_set, self.natural_movie_one_set,self.natural_movie_three_first_set_set,self.natural_movie_three_second_set_set,self.spont]
        elif self.vis_stim_protocol=='AllenB':
            datasets_to_plot=[self.first_static_set, self.second_static_set, self.third_static_set, self.first_images_set,self.second_images_set,self.third_images_set,self.natural_movie_one_set, self.spont]
        elif self.vis_stim_protocol=='AllenC':
            datasets_to_plot=[self.first_noise_set, self.second_noise_set, self.third_noise_set, self.natural_movie_one_set,self.natural_movie_two_set,self.spont1,self.spont2]

        
        datasets_to_plot.append(self.rounded_vis_stim['Prairie']['VisStim'])
         
        fig, axs = plt.subplots(1)
        fig.suptitle('VisStim Paradigm Transitions')
        
        axs.plot(self.rounded_vis_stim['Prairie']['VisStim'])     
        symbol_list=['x','o','<','^','v','s','>','+','d',]
        color_list=['r', 'g']
        n=2        
        indexes=list(self.transitions_dictionary.values())
        for i in range(0, len(indexes)-n+1, n):
            axs.plot(indexes[i], self.rounded_vis_stim['Prairie']['VisStim'][indexes[i]],symbol_list[i-int(i/2)],  color=color_list[i%2],label='Start')
            axs.plot(indexes[i+1], self.rounded_vis_stim['Prairie']['VisStim'][indexes[i+1]],symbol_list[i-int(i/2)],  color=color_list[(i+1)%2],label='End')
        axs.legend()

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
        axs[0].plot(self.second_scale['Prairie'], self.rounded_vis_stim['Prairie']['VisStim'])
        axs[1].plot(self.second_scale['Prairie'][0:-1], self.dfdt_rounded_vis_stim['Prairie']['VisStim'])
        # axs[2].plot(second_scale,photodiode_aray)
        mplcursors.cursor(axs) # or just mplcursors.cursor()
        
        fig, axs = plt.subplots(1)
        axs.plot(self.rounded_vis_stim['Prairie']['VisStim'])
        axs.plot(self.tuning_stim_on_index_full_recording,  self.rounded_vis_stim['Prairie']['VisStim'][self.tuning_stim_on_index_full_recording],'o')
        axs.plot(self.tuning_stim_off_index_full_recording, self.rounded_vis_stim['Prairie']['VisStim'][self.tuning_stim_off_index_full_recording],'x')
  
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
        ax.plot(self.rounded_vis_stim['Prairie']['VisStim'])
        ax.plot(self.tuning_stim_off_index_full_recording[0,:]-1,self.rounded_vis_stim['Prairie']['VisStim'][self.tuning_stim_off_index_full_recording[0,:]-1],'o')
        # snap_cursor = SnappingCursor(ax, line)
        # # fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
        # mplcursors.cursor(line) # or just mplcursors.cursor()

        fig, axs=plt.subplots(1)
        axs.plot(self.rounded_vis_stim['Prairie']['VisStim'])
        
        for i in range(40):
            color = tuple(np.random.choice(range(256), size=3)/256)
            axs.plot(np.argwhere(self.full_stimuli_binary_matrix[i,:]) ,self.rounded_vis_stim['Prairie']['VisStim'][np.argwhere(self.full_stimuli_binary_matrix[i,:])],'x', color=color)
            
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
  