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
    
    def __init__(self, voltage_excel_path):
        self.voltage_excel_path=voltage_excel_path
        # try:
        self.voltage_signals_raw = pd.read_csv(self.voltage_excel_path)    
        self.voltage_signals={signal:self.voltage_signals_raw[signal].to_frame() for signal in self.voltage_signals_raw.columns.tolist()[1:]}
        
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
            

        
        #%% loading and general settings
        
        locomotion_df=self.voltage_signals['Locomotion'].T
        self.locomotion_aray=locomotion_df.to_numpy()
        visualstim_df=self.voltage_signals['VisStim'].T
        self.visualstim_array=visualstim_df.to_numpy().squeeze()
        self.voltage_rate=1000;
        self.milisecondscale=np.arange(1,locomotion_df.size+1)
        self.second_scale=self.milisecondscale/self.voltage_rate
        self.minutes_scale=self.second_scale/60
        self.process_allenA_signals()
        self.get_paradign_indexes()
        self.slice_gratings_by_paradigm()
        
 #%% process locomotion data       
        # calculate movement speed in transitions per milisecond
        self.first_derivative_locomotion=np.diff(np.around(self.locomotion_aray,4))
        # recify(get all signals as positive)
        self.rectified_speed_array=np.absolute(self.first_derivative_locomotion)
        # add the datapoint lost during the diff
        self.rectified_speed_array=np.insert(self.rectified_speed_array,0,0)
        self.plotting()
#%%   ALLEN A 
    def process_allenA_signals(self):   
        print('doing')
        
        # fist get transitions between stimulaton paradigms
        self.rounded_vis_stim=np.around(self.visualstim_array, 1)
        self.dfdt_rounded_vis_stim = np.diff(self.rounded_vis_stim)
        
        

    def get_paradign_indexes(self):   
        
        self.end_transitions=np.argwhere(self.dfdt_rounded_vis_stim<-6.5).flatten()
        
        self.start_transitions=np.argwhere(self.dfdt_rounded_vis_stim>5).flatten()    
        
        self.last_down_transition=np.argwhere(self.dfdt_rounded_vis_stim<-0.5).flatten()[-1] 

        self.first_drifting_set_first=self.end_transitions[0]+1
        self.first_drifting_set_last= self.start_transitions[1]+1
        self.second_drifiting_set_first=self.end_transitions[5]+1
        self.second_drifting_set_last= self.start_transitions[8]+1
        self.third_drifiting_set_first=self.end_transitions[8]+1        
        self.third_drifting_set_last=self.last_down_transition+2        
        self.first_movie_set_first=self.end_transitions[2]+1
        self.first_movie_set_last= self.start_transitions[4]+1
        self.second_movie_set_first=self.end_transitions[6]+1
        self.second_movie_set_last= self.start_transitions[9]+1
        self.short_movie_set_first=self.end_transitions[4]+1
        self.short_movie_set_last= self.start_transitions[7]+1
        self.spont_first= self.start_transitions[8]+1
        self.spont_last=self.end_transitions[6]-2
        

        
    def slice_gratings_by_paradigm (self):   
        self.first_drifting_set=self.rounded_vis_stim[self.first_drifting_set_first:self.first_drifting_set_last]
        self.second_drifting_set=self.rounded_vis_stim[self.second_drifiting_set_first:self.second_drifting_set_last]
        self.third_drifting_set=self.rounded_vis_stim[self.third_drifiting_set_first:self.third_drifting_set_last]
        self.first_movie_set=self.rounded_vis_stim[self.first_movie_set_first:self.first_movie_set_last]
        self.second_movie_set=self.rounded_vis_stim[self.second_movie_set_first:self.second_movie_set_last]
        self.short_movie_set=self.rounded_vis_stim[self.short_movie_set_first:self.short_movie_set_last]      
        self.spont=self.rounded_vis_stim[self.spont_first:self.spont_last]


    def plotting(self):   
        
         fig, axs = plt.subplots(1)
         fig.suptitle('VisStim Paradigm Transitions')
         
         axs.plot(self.rounded_vis_stim)         
         axs.plot(self.first_drifting_set_first, self.rounded_vis_stim[self.first_drifting_set_first],'x', 'r')
         axs.plot(self.first_drifting_set_last, self.rounded_vis_stim[self.first_drifting_set_last],'x', 'b')
         axs.plot(self.second_drifiting_set_first, self.rounded_vis_stim[self.second_drifiting_set_first],'o', 'r')
         axs.plot(self.second_drifting_set_last, self.rounded_vis_stim[self.second_drifting_set_last],'o', 'b')
         axs.plot(self.third_drifiting_set_first, self.rounded_vis_stim[self.third_drifiting_set_first],'.', 'r')
         axs.plot(self.third_drifting_set_last, self.rounded_vis_stim[self.third_drifting_set_last],'.', 'b')
         
         axs.plot(self.first_movie_set_first, self.rounded_vis_stim[self.first_movie_set_first],'^', 'g')
         axs.plot(self.first_movie_set_last, self.rounded_vis_stim[self.first_movie_set_last],'^', 'y')
         axs.plot(self.second_movie_set_first, self.rounded_vis_stim[self.second_movie_set_first],'v', 'g')
         axs.plot(self.second_movie_set_last, self.rounded_vis_stim[self.second_movie_set_last],'v', 'y')
         
         axs.plot(self.short_movie_set_first, self.rounded_vis_stim[self.short_movie_set_first],'-', 'g')
         axs.plot(self.short_movie_set_last, self.rounded_vis_stim[self.short_movie_set_last],'-', 'g') 
         
         axs.plot(self.spont_first,self.rounded_vis_stim[self.spont_first],'o', 'b')
         axs.plot(self.spont_last,self.rounded_vis_stim[self.spont_last],'o', 'b') 
         
         
         
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
#%% HABITUATION 
    def process_habituation_signals(self):   
        print('TO DO')

if __name__ == "__main__":
    
    # temporary_path1='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'

    temporary_path1='/home/samuel/Desktop/SPJAFUllAllen/210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'
    voltagesignals=VoltageSignalsExtractions(temporary_path1)

    
    

