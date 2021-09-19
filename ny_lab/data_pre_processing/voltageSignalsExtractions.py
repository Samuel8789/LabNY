# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:32:35 2021

@author: sp3660
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
# from TestPLot import SnappingCursor



class VoltageSignalsExtractions():
    
    def __init__(self, aquisition_object):
        self.associated_acquisition=aquisition_object
        self.voltage_signals=self.associated_acquisition.voltage_signals_dictionary
        
        #%% loading and general settings
        
        locomotion_df=self.voltage_signals['Locomotion']
        self.locomotion_aray=locomotion_df.to_numpy()
        visualstim_df=self.voltage_signals['VisStim']
        self.visualstim_aray=visualstim_df.to_numpy()
        self.voltage_rate=1000;
        self.milisecondscale=np.arange(1,locomotion_df.size+1)
        self.second_scale=self.milisecondscale/self.voltage_rate
        self.minutes_scale=self.second_scale/60
        
        
 #%% process locomotion data       
        # calculate movement speed in transitions per milisecond
        first_derivative=np.diff(np.around(self.locomotion_aray,4))
        # recify(get all signals as positive)
        rectified_speed_array=np.absolute(first_derivative)
        # add the datapoint lost during the diff
        rectified_speed_array=np.insert(rectified_speed_array,0,0)
        
#%%   ALLEN A 
    def process_allenA_signals(self):   
        print('doing')
        
        # fist get transitions between stimulaton paradigms
        self.rounded_vis_stim=np.around(self.visualstim_aray, 1)
        self.dfdt_rounded_vis_stim = np.diff(self.rounded_vis_stim)
        
        
        
        
        
    def get_paradign_indexes(self):    
        self.end_transitions=np.argwhere(self.df_rounded_vis_stim<-6.5).flatten()
        
        self.start_transitions=np.argwhere(self.df_rounded_vis_stim>5).flatten()    
        
        self.last_down_transition=np.argwhere(self.df_rounded_vis_stim<-0.5).flatten()[-1]
        
 
        
        self.first_drifitng_set_first=self.end_transitions[0]+1
        self.first_drifitng_set_last= self.start_transitions[1]+1
        self.second_drifitng_set_first=self.end_transitions[5]+1
        self.second_drifitng_set_last= self.start_transitions[8]+1
        self.third_drifitng_set_first=self.end_transitions[8]+1
        self.third_drifitng_set_last=self.all_down_transition+2
        
        self.first_movie_set_first=self.end_transitions[2]+1
        self.first_movie_set_last= self.start_transitions[4]+1
        self.second_movie_set_first=self.end_transitions[6]+1
        self.second_movie_set_last= self.start_transitions[9]+1
        
        self.short_movie_set_first=self.end_transitions[4]+1
        self.short_movie_set_last= self.start_transitions[7]+1
        
        self.spont_first= self.start_transitions[8]+1
        self.spont_last=self.end_transitions[6]-2
        
        
        fig, axs = plt.subplots(8)
        fig.suptitle('VisStim Paradigm Transitions')
        self.rounded_vis_stim
        axs[0].plot(self.rounded_vis_stim)
        
        axs[0].plot(self.rounded_vis_stim,)
        axs[0].plot(self.rounded_vis_stim,)
        
        axs[0].plot(self.rounded_vis_stim[self.first_drifitng_set_first],'x', 'r')
        axs[0].plot(self.rounded_vis_stim[self.first_drifitng_set_last],'x', 'b')
        axs[0].plot(self.rounded_vis_stim[self.second_drifitng_set_first],'o', 'r')
        axs[0].plot(self.rounded_vis_stim[self.second_drifitng_set_last],'o', 'b')
        axs[0].plot(self.rounded_vis_stim[self.third_drifitng_set_first],'.', 'r')
        axs[0].plot(self.rounded_vis_stim[self.third_drifitng_set_last],'.', 'b')
        
        axs[0].plot(self.rounded_vis_stim[self.first_movie_set_first],'^', 'g')
        axs[0].plot(self.rounded_vis_stim[self.first_movie_set_last],'^', 'y')
        axs[0].plot(self.rounded_vis_stim[self.second_movie_set_first],'v', 'g')
        axs[0].plot(self.rounded_vis_stim[self.second_movie_set_last],'v', 'y')
        
        axs[0].plot(self.rounded_vis_stim[self.short_movie_set_first],'-', 'k')
        axs[0].plot(self.rounded_vis_stim[self.short_movie_set_last],'-', 'c') 
        
        axs[0].plot(self.rounded_vis_stim[self.spont_first],':', 'k')
        axs[0].plot(self.rounded_vis_stim[self.spont_last],':', 'c') 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    def slice_gratings_by_paradigm (self):   
        self.first_drifiting_set=self.rounded_vis_stim[self.first_drifitng_set_first:self.first_drifitng_set_last]
        self.second_drifiting_set=self.rounded_vis_stim[self.second_drifitng_set_first:self.second_drifitng_set_last]
        self.third_drifiting_set=self.rounded_vis_stim[self.third_drifitng_set_first:self.third_drifitng_set_last]
        
        
        self.first_movie_set=self.rounded_vis_stim[self.first_movie_set_first:self.first_movie_set_last]
        self.second_movie_set=self.rounded_vis_stim[self.second_movie_set_first:self.second_movie_set_last]
        
        self.short_movie_set=self.rounded_vis_stim[self.short_movie_set_first:self.short_movie_set_last]
        
        self.spont=self.rounded_vis_stim[self.spont_first:self.spont_last]

        fig, axs = plt.subplots(8)
        fig.suptitle('VisStim Paradigm Transitions')
        axs[0].plot(self.first_drifitng_set)
        axs[1].plot(self.second_drifitng_set)
        axs[2].plot(self.third_drifitng_set)
        axs[3].plot(self.first_movie_set)
        axs[4].plot(self.second_movie_set)
        axs[5].plot(self.short_movie_set)
        axs[6].plot(self.spont)
        axs[7].plot(self.rounded_vis_stim)
        mplcursors.cursor(axs) # 
        
        
        
#%% HABITUATION 
    def process_habituation_signals(self):   
        print('TO DO')

