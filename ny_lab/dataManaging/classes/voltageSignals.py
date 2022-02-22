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
import scipy.io as sio
import copy
# from TestPLot import SnappingCursor
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b"]) 
import logging 
module_logger = logging.getLogger(__name__)
try :
    from .standalone.voltageSignalsExtractions import VoltageSignalsExtractions
except:
    from standalone.voltageSignalsExtractions import VoltageSignalsExtractions


class VoltageSignals():
    
    def __init__(self, acquisition_object=None, voltage_excel_path=False, temporary_path=False, just_copy=False,slow_voltage_raw_csv_path=None, acquisition_directory_raw=False, extra_daq_path=None ):
        module_logger.info('Processing Voltage Signals')
        self.voltage_signals_dictionary={'Locomotion': pd.DataFrame({'Locomotion' : []}),
                                         'VisStim':pd.DataFrame({'VisStim' : []}),
                                         'LED':pd.DataFrame({'LED' : []}),
                                         'PhotoDiode':pd.DataFrame({'PhotoDiode' : []}),
                                         'Frames':pd.DataFrame({'Frames' : []}),
                                         'Input2':pd.DataFrame({'Input2' : []}),
                                         'Time':pd.DataFrame({'Time' : []}),
                                         }
        self.voltage_signals_dictionary_daq=copy.copy(self.voltage_signals_dictionary)

        self.slow_voltage_raw_csv_path=slow_voltage_raw_csv_path
        self.extra_daq_path=extra_daq_path
        self.voltage_excel_path=voltage_excel_path
        self.temporary_path=temporary_path
        if extra_daq_path and not temporary_path:
            self.temporary_path=os.path.split(extra_daq_path)[0]
        self.acquisition_directory_raw=acquisition_directory_raw
        self.acquisition_object=acquisition_object
        
        self.voltage_signals_dictionary={}
        self.voltage_signals_path_dictionary={}
        
        self.create_slow_storage_paths()
        self.load_from_second_matlab()
        self.check_if_files_in_slow_storage()
        self.convert_slow_raw_csv_to_feather()


        if self.acquisition_object: 
            if self.acquisition_object.raw_input_path:
                self.transfer_raw_csv()
                self.voltage_recording_raw_file_full_path=[file for file in glob.glob( os.path.join(self.acquisition_object.aquisition_path, self.acquisition_object.aquisition_name)+'\\**', recursive=False) if 'VoltageRecording' in file and '.csv' in file   ]
                if self.voltage_recording_raw_file_full_path:
                    self.voltage_recording_raw_file_full_path=self.voltage_recording_raw_file_full_path[0]
                self.process_raw_voltage_recording()
                self.check_if_files_in_slow_storage()
            else:
                self.check_if_files_in_slow_storage()
            
#%% methods
    def signal_extraction_object(self):
        extraction_object=VoltageSignalsExtractions(voltage_signals_object=self)

    def load_from_second_matlab(self):
        if  self.extra_daq_path:
            self.mat=sio.loadmat(self.extra_daq_path)# load mat-file           
            mdata = self.mat['daq_data']  # variable in mat file 
            ndata = {n: mdata[n][0,0] for n in mdata.dtype.names}
            Columns = [n for n, v in ndata.items()]
            # data_dic=dict((c, ndata[c]) for c in Columns)
            time_array=ndata['time']
            volt_array=ndata['voltage']
            self.fullsignals_extra_daq=np.hstack((time_array,volt_array)).T
            self.process_extra_daq_recording()
        else:
            self.fullsignals_extra_daq=None
    
    def process_extra_daq_recording(self):
        self.voltage_signals_dictionary_daq['Locomotion']=self.voltage_signals_dictionary_daq['Locomotion'].assign(Locomotion=self.fullsignals_extra_daq[3,:].T.tolist())
        self.voltage_signals_dictionary_daq['VisStim']=self.voltage_signals_dictionary_daq['VisStim'].assign(VisStim=self.fullsignals_extra_daq[1,:].T.tolist())
        self.voltage_signals_dictionary_daq['LED']=self.voltage_signals_dictionary_daq['LED'].assign(LED=self.fullsignals_extra_daq[4,:].T.tolist())
        self.voltage_signals_dictionary_daq['PhotoDiode']=self.voltage_signals_dictionary_daq['PhotoDiode'].assign(PhotoDiode=self.fullsignals_extra_daq[2,:].T.tolist())
        self.voltage_signals_dictionary_daq['Frames']=self.voltage_signals_dictionary_daq['Frames'].assign(Frames=self.fullsignals_extra_daq[5,:].T.tolist())
        self.voltage_signals_dictionary_daq['Time']=self.voltage_signals_dictionary_daq['Time'].assign(Time=self.fullsignals_extra_daq[0,:].T.tolist())

    def transfer_second_daq_mat(self):
        pass
    def look_for_second_daq_mat(self):
        pass

    def transfer_raw_csv(self):
        self.voltage_recording_raw_file_transfered_path=self.acquisition_object.slow_storage_all_paths['raw_volatge_csv']    
        self.voltage_recording_raw_file_full_path=[file for file in glob.glob( os.path.join(self.acquisition_object.aquisition_path,self.acquisition_object.aquisition_name)+'\\**', recursive=False) if 'VoltageRecording' in file and '.csv' in file   ]
        if self.voltage_recording_raw_file_full_path and not os.path.isfile(self.voltage_recording_raw_file_transfered_path):
            self.voltage_recording_raw_file_full_path=self.voltage_recording_raw_file_full_path[0]
            final_file_path=os.path.join(self.voltage_recording_raw_file_transfered_path,os.path.split(self.voltage_recording_raw_file_full_path)[1])
            if not os.path.isfile(final_file_path):
                shutil.copy(self.voltage_recording_raw_file_full_path, self.voltage_recording_raw_file_transfered_path)
            
    def process_raw_voltage_recording(self):
        try:
            self.signals_recorded=pd.read_csv(self.voltage_recording_raw_file_full_path, index_col=0, nrows=0).columns.tolist()
            self.voltage_signals_dictionary={key:pd.DataFrame({key : []}) for key in pd.read_csv(self.voltage_recording_raw_file_full_path, index_col=0, nrows=0).columns.tolist()}
            self.correct_csv_signals_names()
            
            
            if len(self.voltage_signals_dictionary.keys())>len(self.is_file_dictionary.keys()) or not all([signal_name for signal_name in  self.is_file_dictionary.values()]):

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



    def convert_slow_raw_csv_to_feather(self):
        if self.slow_voltage_raw_csv_path:
            if not any(value for value in self.is_file_dictionary.values() if value==0):
                self.voltage_recording_raw_file_full_path= self.slow_voltage_raw_csv_path
                self.process_raw_voltage_recording()
                self.check_if_files_in_slow_storage()
       

    def create_slow_storage_paths(self): 
        keys=['locomotion', 'visual stim', 'eye camera', 'photodiode', 'planes']

        if self.acquisition_object:
           tosavepath=self.acquisition_object.slow_storage_all_paths     
        elif self.temporary_path:
            tosavepath={key:self.temporary_path for key in keys}
        elif  self.slow_voltage_raw_csv_path:
            aqpath=os.path.split(os.path.split(self.slow_voltage_raw_csv_path)[0])[0]
            tosavepath={key:os.path.join(aqpath,key) for key in keys}
  
        self.voltage_signals_path_dictionary={'Locomotion':os.path.join(tosavepath['locomotion'], 'locomotion.ftr') ,
                                         'VisStim':os.path.join(tosavepath['visual stim'], 'vis_stim_voltage.ftr'),
                                         'LED':os.path.join(tosavepath['eye camera'], 'led.ftr'),
                                         'PhotoDiode':os.path.join(tosavepath['photodiode'], 'screen_photodiode.ftr'),
                                         'Frames':os.path.join(tosavepath['planes'], 'frames.ftr'),
                                         'Input2':os.path.join(tosavepath['planes'], 'inpu2.ftr')
                                             }
      
    def plot_all_signals(self):
        for key in self.voltage_signals_dictionary.keys():
            self.voltage_signals_dictionary[key].plot()
    
        pass
           
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
            
            
if __name__ == "__main__":
    extra=r'I:\Projects\LabNY\20220209\VisStimTests\SessioonCprairirev1test_1\TestSessionC_prairirev1_2_9_2022_17_36.mat'
    extra=r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\visual_stim\BehaviourCode\YuryScripts\behavior\NI_DAQ\output_data\test_2_12_2022_16_5.mat'
    voltagesignals=VoltageSignals(extra_daq_path=extra)
    test=voltagesignals.voltage_signals_dictionary_daq
    # temporary_path1='\\\\?\\'+r'K:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Imaging\G2C\Ai14\SPJA\imaging\20210702\data aquisitions\FOV_1\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\raw_volatge_csv\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'
    # voltagesignals=VoltageSignals(slow_voltage_raw_csv_path=temporary_path1)
    # voltagesignals.plot_all_signals()
    pass