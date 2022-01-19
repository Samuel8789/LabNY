# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:12:17 2022

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
import gc
# from TestPLot import SnappingCursor
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b"]) 
import logging 
module_logger = logging.getLogger(__name__)


class VoltageSignals():
    
    def __init__(self,acquisition_object=None, voltage_excel_path=False, temporary_path=False, just_copy=False, acquisition_directory_raw=False ):
        module_logger.info('Processing Voltage Signals')
        self.voltage_excel_path=voltage_excel_path
        self.temporary_path=temporary_path
        self.acquisition_directory_raw=acquisition_directory_raw
        self.acquisition_object=acquisition_object
        self.voltage_signals_dictionary={}
        self.voltage_signals_path_dictionary={}

        self.create_slow_storage_paths()
        self.check_if_files_in_slow_storage()

        
        
        self.voltage_signals_dictionary={'Locomotion': pd.DataFrame({'Locomotion' : []}),
                                         'VisStim':pd.DataFrame({'VisStim' : []}),
                                         'LED':pd.DataFrame({'LED' : []}),
                                         'PhotoDiode':pd.DataFrame({'PhotoDiode' : []}),
                                         'Frames':pd.DataFrame({'LED' : []}),
                                         'Input2':pd.DataFrame({'PhotoDiode' : []})
                                         }

        if self.acquisition_object:
            
            if self.acquisition_object.raw_input_path:

                self.transfer_raw_csv()
                self.process_raw_voltage_recording()
                self.check_if_files_in_slow_storage()

                
            else:
                self.check_if_files_in_slow_storage()
            
#%% methods

    def transfer_raw_csv(self):
        self.voltage_recording_raw_file_transfered_path=self.acquisition_object.slow_storage_all_paths['raw_volatge_csv']    
        self.voltage_recording_raw_file_full_path=[file for file in glob.glob( os.path.join(self.acquisition_object.aquisition_path,self.acquisition_object.aquisition_name)+'\\**', recursive=False) if 'VoltageRecording' in file and '.csv' in file   ]
        if self.voltage_recording_raw_file_full_path and not os.path.isfile(self.voltage_recording_raw_file_transfered_path):
            self.voltage_recording_raw_file_full_path=self.voltage_recording_raw_file_full_path[0]
            final_file_path=os.path.join(self.voltage_recording_raw_file_transfered_path,os.path.split(self.voltage_recording_raw_file_full_path)[1])
            if not os.path.isfile(final_file_path):
                shutil.copy(self.voltage_recording_raw_file_full_path, self.voltage_recording_raw_file_transfered_path)
            
    def process_raw_voltage_recording(self):
       
        self.voltage_recording_raw_file_full_path=[file for file in glob.glob( os.path.join(self.acquisition_object.aquisition_path,self.acquisition_object.aquisition_name)+'\\**', recursive=False) if 'VoltageRecording' in file and '.csv' in file   ]
        if self.voltage_recording_raw_file_full_path:
            self.voltage_recording_raw_file_full_path=self.voltage_recording_raw_file_full_path[0]
            
            try:
                self.signals_recorded=pd.read_csv(self.voltage_recording_raw_file_full_path, index_col=0, nrows=0).columns.tolist()
                self.voltage_signals_dictionary={key:pd.DataFrame({key : []}) for key in pd.read_csv(self.voltage_recording_raw_file_full_path, index_col=0, nrows=0).columns.tolist()}
                self.correct_csv_signals_names()
                
                
                if len(self.voltage_signals_dictionary.keys())>len(self.is_file_dictionary.keys()) or not all([signal_name for signal_name in  self.is_file_dictionary.keys() if  self.is_file_dictionary[signal_name]==1]):

                    self.voltage_signals = pd.read_csv(self.voltage_recording_raw_file_full_path)
                    self.voltage_signals_dictionary={signal:self.voltage_signals[signal].to_frame() for signal in self.voltage_signals.columns.tolist()[1:]}
                    self.correct_csv_signals_names()
                # except:
                #     self.voltage_signals_dictionary={'Locomotion': pd.DataFrame({'Locomotion' : []}),
                #                                      'VisStim':pd.DataFrame({'VisStim' : []}),
                #                                      'LED':pd.DataFrame({'LED' : []}),
                #                                      'PhotoDiode':pd.DataFrame({'PhotoDiode' : []})}
                #     module_logger.exception('cretaing dataframes from scratch')
                
                    for signal, dataf in self.voltage_signals_dictionary.items():
                            self.voltage_signals_dictionary[signal].to_feather(self.voltage_signals_path_dictionary[signal])
                else:
                    module_logger.info('all signals already there')


            except:
                module_logger.exception('couldn\'t read voltage signals')


 
        
    def create_slow_storage_paths(self): 
        
        self.voltage_signals_path_dictionary={'Locomotion':os.path.join(self.acquisition_object.slow_storage_all_paths['locomotion'], 'locomotion.ftr') ,
                                         'VisStim':os.path.join(self.acquisition_object.slow_storage_all_paths['visual stim'], 'vis_stim_voltage.ftr'),
                                         'LED':os.path.join(self.acquisition_object.slow_storage_all_paths['eye camera'], 'led.ftr'),
                                         'PhotoDiode':os.path.join(self.acquisition_object.slow_storage_all_paths['photodiode'], 'screen_photodiode.ftr'),
                                         'Frames':os.path.join(self.acquisition_object.slow_storage_all_paths['planes'], 'frames.ftr'),
                                         'Input2':os.path.join(self.acquisition_object.slow_storage_all_paths['planes'], 'inpu2.ftr')

                                             }
           
    def correct_csv_signals_names(self):       
        signals=list(self.voltage_signals_dictionary.keys())
        for signal in signals:
            if 'Locomotion' in signal:     
                self.voltage_signals_dictionary['Locomotion']= self.voltage_signals_dictionary[signal]
                del self.voltage_signals_dictionary[signal]                
            if 'VisStim' in signal:           
                self.voltage_signals_dictionary['VisStim']= self.voltage_signals_dictionary[signal]
                del self.voltage_signals_dictionary[signal]                
            if 'LED' in signal:               
                self.voltage_signals_dictionary['LED']= self.voltage_signals_dictionary[signal]
                del self.voltage_signals_dictionary[signal]                
            if 'PhotoDiode' in signal:                    
                self.voltage_signals_dictionary['PhotoDiode']= self.voltage_signals_dictionary[signal]
                del self.voltage_signals_dictionary[signal]      
            if 'Frames' in signal:                    
                self.voltage_signals_dictionary['Frames']= self.voltage_signals_dictionary[signal]
                del self.voltage_signals_dictionary[signal]        
            if 'Input2' in signal:                    
                self.voltage_signals_dictionary['Input2']= self.voltage_signals_dictionary[signal]
                del self.voltage_signals_dictionary[signal]        
                
        
        
    def check_if_files_in_slow_storage(self):
        self.is_file_dictionary={}
        
        for name, path in self.voltage_signals_path_dictionary.items() :
            if os.path.isfile(path) :
                if os.path.getsize(path)>0:
                        self.is_file_dictionary[name]=1
                        module_logger.info('signal is there' + path)                   
                else:       
                        self.is_file_dictionary[name]='Bad File'
                        module_logger.info('something wrong with file' + path)
            else:
                self.is_file_dictionary[name]=0

                module_logger.info('file not there' + path)

        
    def load_slow_storage_voltage_signals(self):   
    
        self.voltage_signals_dictionary={}
        for name, path in self.voltage_signals_path_dictionary.items():
            if os.path.isfile(path) :
                if os.path.getsize(path)>0:  
                    self.voltage_signals_dictionary[name]=pd.read_feather(path)
            else:
   
                module_logger.info('no voltage signal '+ path)

            
    def unload_voltage_signals(self):
        module_logger.info('Unloading voltage signals')

        if self.voltage_signals_dictionary:
            del self.voltage_signals_dictionary
            gc.collect()
            # sys.stdout.flush()
