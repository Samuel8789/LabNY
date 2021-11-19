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
# from TestPLot import SnappingCursor
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b"]) 



class VoltageSignalsExtractions():
    
    def __init__(self, voltage_excel_path=False, temporary_path=False, just_copy=False, acquisition_directory_raw=False):
        print('Processing Voltage Signals')
        self.voltage_excel_path=voltage_excel_path
        self.temporary_path=temporary_path
        self.acquisition_directory_raw=acquisition_directory_raw
        self.check_csv_in_folder()

        if just_copy:
    
            shutil.copy(self.voltage_excel_path, self.temporary_path)
            
  
        else:
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
                
            # self.slice_gratings_by_paradigm()
            # self.get_drifting_gratings_indexes()
            self.process_locomotion()
            # self.slice_locomotion_by_paradigm()
            
            
            # self.plotting_paradigm_transitions()
            # self.plotting_grating_transitions()
        
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
        
    def process_locomotion(self):
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
        
    def process_allenA_signals(self):   
        print('Analysing AllenA Gratings')
        
        # fist get transitions between stimulaton paradigms
        self.rounded_vis_stim=np.around(self.visualstim_array, 1)
        self.dfdt_rounded_vis_stim = np.diff(self.rounded_vis_stim)
        
    def get_paradign_indexes(self):   
        
        self.start_transitions=np.argwhere(self.dfdt_rounded_vis_stim<-6.5).flatten()+1
        self.spont_start_transitions=np.argwhere(np.logical_and(self.dfdt_rounded_vis_stim>5, self.dfdt_rounded_vis_stim<6.5)).flatten()+1
        self.end_transitions=np.argwhere(self.dfdt_rounded_vis_stim>6.5).flatten()+1    
        #eliminate first 
        self.end_transitions=np.delete(self.end_transitions, 0)        
        self.last_down_transition=np.argwhere(self.dfdt_rounded_vis_stim<-0.5).flatten()[-1]+1

        self.transitions_dictionary={'first_drifting_set_first':self.start_transitions[0],
                                    'first_drifting_set_last': self.end_transitions[0],
                                    'second_drifting_set_first':self.start_transitions[6],
                                    'second_drifting_set_last':self.end_transitions[5],
                                    'third_drifting_set_first':self.start_transitions[10],
                                    'third_drifting_set_last':self.last_down_transition+1,
                                    'first_movie_set_first':self.start_transitions[2],
                                    'first_movie_set_last': self.end_transitions[2],
                                    'second_movie_set_first':self.start_transitions[8],
                                    'second_movie_set_last': self.end_transitions[7],
                                    'short_movie_set_first':self.start_transitions[4],
                                    'short_movie_set_last':self.end_transitions[4],
                                    'spont_first':self.spont_start_transitions[3],
                                    'spont_last':self.end_transitions[6]-1,
                                    }

   
       
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

        
    def slice_gratings_by_paradigm (self):  
        
        self.first_drifting_set=self.rounded_vis_stim[self.transitions_dictionary['first_drifting_set_first']:self.transitions_dictionary['first_drifting_set_last']]
        self.second_drifting_set=self.rounded_vis_stim[self.transitions_dictionary['second_drifting_set_first']:self.transitions_dictionary['second_drifting_set_last']]
        self.third_drifting_set=self.rounded_vis_stim[self.transitions_dictionary['third_drifting_set_first']:self.transitions_dictionary['third_drifting_set_last']]
        self.first_movie_set=self.rounded_vis_stim[self.transitions_dictionary['first_movie_set_first']:self.transitions_dictionary['first_movie_set_last']]
        self.second_movie_set=self.rounded_vis_stim[self.transitions_dictionary['second_movie_set_first']:self.transitions_dictionary['second_movie_set_last']]
        self.short_movie_set=self.rounded_vis_stim[self.transitions_dictionary['short_movie_set_first']:self.transitions_dictionary['short_movie_set_last']]     
        self.spont=self.rounded_vis_stim[self.transitions_dictionary['spont_first']:self.transitions_dictionary['spont_last']]
        
        
    def slice_locomotion_by_paradigm (self):  
         
         self.first_drifting_set_speed=self.rectified_speed_array[self.transitions_dictionary['first_drifting_set_first']:self.transitions_dictionary['first_drifting_set_last']]
         self.second_drifting_set_speed=self.rectified_speed_array[self.transitions_dictionary['second_drifting_set_first']:self.transitions_dictionary['second_drifting_set_last']]
         self.third_drifting_set_speed=self.rectified_speed_array[self.transitions_dictionary['third_drifting_set_first']:self.transitions_dictionary['third_drifting_set_last']]
         self.first_movie_set_speed=self.rectified_speed_array[self.transitions_dictionary['first_movie_set_first']:self.transitions_dictionary['first_movie_set_last']]
         self.second_movie_set_speed=self.rectified_speed_array[self.transitions_dictionary['second_movie_set_first']:self.transitions_dictionary['second_movie_set_last']]
         self.short_movie_set_speed=self.rectified_speed_array[self.transitions_dictionary['short_movie_set_first']:self.transitions_dictionary['short_movie_set_last']]     
         self.spont_speed=self.rectified_speed_array[self.transitions_dictionary['spont_first']:self.transitions_dictionary['spont_last']]    
        

    def get_drifting_gratings_indexes(self):
       
        self.combined_gratings=np.concatenate((self.first_drifting_set, self.second_drifting_set, self.third_drifting_set))
        self.df_combined_gratings = np.diff(self.combined_gratings)
        self.orientations=np.round(np.linspace(0.1,4,40),1)
        blanksweepvoltage=4.5
        self.repetitions=15
        self.orientations_and_blank_sweep= np.append(self.orientations, blanksweepvoltage)
        self.total_grating_trials=self.orientations.shape[0]*self.repetitions
        self.df_combined_gratings = np.diff(self.combined_gratings)
        self.df_combined_gratings_rounded=np.around(self.df_combined_gratings, 1)
        
        #correcting slower voltage transitions
        for i in range(2, self.df_combined_gratings_rounded.size-2):
            if self.df_combined_gratings_rounded[i+1]!=0 and self.df_combined_gratings_rounded[i]!=0:
               self.df_combined_gratings_rounded[i+1]= self.df_combined_gratings_rounded[i+1]+ self.df_combined_gratings_rounded[i];
               self.df_combined_gratings_rounded[i]=0;
        
        
        
        #testing the indexes
        idxx=8
        self.test_down=np.argwhere(self.df_combined_gratings==self.orientations[idxx]).flatten()
        self.test_upn=np.argwhere(np.logical_and(self.df_combined_gratings>0.05, self.df_combined_gratings<4.1)).flatten()
        #%% create binary atrix of starts and end for full movie, it includes the correction for the diff
        #getting bolean indexing vectors

        self.orientations_boolean_starts=np.zeros(( self.orientations.shape[0],self.combined_gratings.size))
        self.orientations_boolean_end=np.zeros(( self.orientations.shape[0],self.combined_gratings.size))
        
        for i in range(self.df_combined_gratings_rounded.size):
            for j, voltage in enumerate( self.orientations):
                if self.df_combined_gratings_rounded[i] == voltage:
                    self.orientations_boolean_starts[j, i+1] = 1;
                if self.df_combined_gratings_rounded[i] == -voltage:
                    self.orientations_boolean_end[j, i+1] = 1;
 
    
  #%%    # get indexes from full movie
  
        self.stim_on_indexes_tuning_period=np.zeros(( self.orientations.size,  self.repetitions)).astype('int64')
        self.stim_off_indexes_tuning_period=np.zeros(( self.orientations.size, self.repetitions)).astype('int64')

        for row in range(self.orientations.size):
            
            self.stim_on_indexes_tuning_period[row,:]=np.argwhere(self.orientations_boolean_starts[row,:]).flatten()
            # i dont know why this is here. some partciluar correctio
            if np.argwhere( self.orientations_boolean_end[row,:]).flatten().size==9:
                tosave= np.argwhere( self.orientations_boolean_end[row,:]).flatten() 
                tosave=np.append(tosave,self.period_tuning.size)
                self.stim_off_indexes_tuning_period[row,:]=tosave-1
            else:
                self.stim_off_indexes_tuning_period[row,:]=np.argwhere(self.orientations_boolean_end[row,:]).flatten()-1
                
                
#%% shorter indexing     
        self.short_indexes_first=[]  
        self.short_indexes_last=[]                     
        for row in range(0, self.orientations.size):
            self.short_indexes_first.append(np.argwhere(self.df_combined_gratings==self.orientations[row]).flatten()+1)
            self.short_indexes_last.append(np.argwhere(self.df_combined_gratings== - self.orientations[row]).flatten()+1)
        self.short_indexes_first=np.vstack( self.short_indexes_first)
        self.short_indexes_last=np.vstack( self.short_indexes_first)

        
  #%%
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

        self.tuning_stim_on_index_full_recording = vfunc(self.stim_on_indexes_tuning_period, first_length, second_length,  mivies_indexes[0], mivies_indexes[2], mivies_indexes[4])
        self.tuning_stim_off_index_full_recording = vfunc(self.stim_off_indexes_tuning_period, first_length, second_length,  mivies_indexes[0], mivies_indexes[2], mivies_indexes[4])
        
        
        
        
#%% fill in the full rnage with mnumbers  

        self.full_stimuli_binary_matrix=np.zeros((self.orientations.size,self.visualstim_array.shape[0] ))
        for i, row in enumerate(self.tuning_stim_on_index_full_recording):
            for j, trial in enumerate(row):
                self.full_stimuli_binary_matrix[i, self.tuning_stim_on_index_full_recording[i,j]:self.tuning_stim_off_index_full_recording[i,j]]=1



#%%
        # file_path=dataset_path+"\\diftingindexon.mat"
        # scipy.io.savemat(file_path, {'diftingindexon': self.tuning_stim_on_index_full_recording})
        # file_path=dataset_path+"\\diftingindexoff.mat"
        # scipy.io.savemat(file_path, {'diftingindexoff': self.tuning_stim_off_index_full_recording})

#%%
    def confirm_grating_indexes(self):
        
        print('todo')
        # confirm number of trial
        # confirm length of ranges
        
        
    def check_csv_in_folder(self):
     if self.acquisition_directory_raw:
            csvfiles=glob.glob(self.acquisition_directory_raw+'\\**.csv')
            for csv in csvfiles:
                if 'VoltageRecording'  in csv:
                    self.voltage_excel_path=csv
     


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
        print('todo')


    def save_transition_indexes(self):
        with open(  self.transition_index_to_save_path, 'wb') as f:
            pickle.dump(self.transitions_dictionary, f, pickle.HIGHEST_PROTOCOL)

    def load_indexes_from_file(self):
        with open(self.transition_index_to_save_path, 'rb') as f:
            self.transitions_dictionary= pickle.load(f)
    

    def plotting_paradigm_transitions(self):   
         
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

        fig, axs = plt.subplots(8)
        fig.suptitle('VisStim Paradigm Transitions')
        axs[0].plot(self.first_drifting_set)
        axs[1].plot(self.second_drifting_set)
        axs[2].plot(self.third_drifting_set)
        axs[3].plot(self.first_movie_set)
        axs[4].plot(self.second_movie_set)
        axs[5].plot(self.short_movie_set)
        axs[6].plot(self.spont)
        axs[7].plot(self.rounded_vis_stim)
        mplcursors.cursor(axs) # 
        
         
    def plotting_grating_transitions(self): 
        
        pixel_per_bar = 10
        dpi = 200
        fig = plt.figure(figsize=(200 * pixel_per_bar / dpi, 2), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
        ax.set_axis_off()
        ax.imshow(self.full_stimuli_binary_matrix,vmax=0.0001, cmap='binary', aspect='auto')

        
         
        fig, axs = plt.subplots(4, sharex=True)
        axs[0].plot(self.combined_gratings)
        axs[0].plot(self.test_down+1, self.combined_gratings[self.test_down+1],'bx')
        axs[1].plot(self.df_combined_gratings)
        axs[1].plot(self.test_down, self.df_combined_gratings[self.test_down],'bo')
        axs[2].plot(self.combined_gratings)
        axs[2].plot(self.test_upn+1, self.combined_gratings[self.test_upn+1],'rx')
        axs[3].plot(self.df_combined_gratings)
        axs[3].plot(self.test_upn, self.df_combined_gratings[self.test_upn],'ro')
        mplcursors.cursor(axs) # or just mplcursors.cursor()

        fig, axs = plt.subplots(1)
        axs.plot(self.combined_gratings)
        axs.plot(self.stim_on_indexes_tuning_period,  self.combined_gratings[self.stim_on_indexes_tuning_period],  'o')
        axs.plot(self.stim_off_indexes_tuning_period, self.combined_gratings[self.stim_off_indexes_tuning_period], 'x')

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

        fig, ax = plt.subplots(nrows=8,sharey=True)
        # fig.set_title('Snapping cursor')
        for i in range(0,8):
            line, = ax[i].plot(self.orientations_boolean_starts[i]) 
        # snap_cursor = SnappingCursor(ax[0], line)  
        # fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
        mplcursors.cursor(line) # or just mplcursors.cursor()

        fig, ax = plt.subplots(nrows=1,sharey=True)            
        ax.plot(self.rounded_vis_stim)
        ax.plot(self.tuning_stim_off_index_full_recording[0,:]-1,self.rounded_vis_stim[self.tuning_stim_off_index_full_recording[0,:]-1],'o')
        # snap_cursor = SnappingCursor(ax, line)
        # fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
        mplcursors.cursor(line) # or just mplcursors.cursor()

        fig, axs=plt.subplots(1)
        axs.plot(self.rounded_vis_stim)
        
        for i in range(40):
            color = tuple(np.random.choice(range(256), size=3)/256)
            axs.plot(np.argwhere(self.full_stimuli_binary_matrix[i,:]) ,self.rounded_vis_stim[np.argwhere(self.full_stimuli_binary_matrix[i,:])],'x', color=color)


if __name__ == "__main__":
    
    # temporary_path1='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'
    # temporary_path1='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'
    temporary_path1='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211113_SPKQ_FOV1_2planeAllenA_20x_920_50024_narrow_without-000\Plane1\211113_SPKQ_FOV1_2planeAllenA_20x_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'
    # temporary_path1='/home/samuel/Desktop/SPJAFUllAllen/210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'
    
    voltagesignals=VoltageSignalsExtractions(temporary_path1)

    plt.plot(voltagesignals.rectified_speed_array)
