# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:12:17 2022

@author: sp3660
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import glob
# lazy_import
import mplcursors
import math
import shutil
import gc
import scipy.io as sio
import copy
import matplotlib as mpl
# from TestPLot import SnappingCursor
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b"]) 
import logging 
module_logger = logging.getLogger(__name__)

try:
   from .standalone.voltageSignalsExtractions import VoltageSignalsExtractions
except:
   from standalone.voltageSignalsExtractions import VoltageSignalsExtractions


class VoltageSignals():
    
    def __init__(self, acquisition_object=None, voltage_excel_path=False, temporary_path=False, just_copy=False, slow_voltage_raw_csv_path=None, acquisition_directory_raw=False, extra_daq_path=None, scanimage_voltage_csv_path=None ):
        module_logger.info('Processing Voltage Signals')

        # create empty dictionarys
        self.no_voltage_signals=True
        self.voltage_signals_dictionary_daq={'Locomotion': pd.DataFrame({'Locomotion' : []}),
                                         'VisStim':pd.DataFrame({'VisStim' : []}),
                                         'LED':pd.DataFrame({'LED' : []}),
                                         'PhotoDiode':pd.DataFrame({'PhotoDiode' : []}),
                                         'Frames':pd.DataFrame({'Frames' : []}),
                                         'Input2':pd.DataFrame({'Input2' : []}),
                                         'Time':pd.DataFrame({'Time' : []}),
                                         'Input7':pd.DataFrame({'Input7' : []}),
                                         'PhotoStim':pd.DataFrame({'PhotoStim' : []}),
                                         'PhotoTrig':pd.DataFrame({'PhotoTrig' : []}),   
                                         'AcqTrig':pd.DataFrame({'AcqTrig' : []})
                                         }
        self.voltage_signals_dictionary=copy.copy(self.voltage_signals_dictionary_daq)
        del  self.voltage_signals_dictionary['Time']
        self.voltage_signals_ftr_path_dictionary={}
        self.scanimage_voltage_csv_path=scanimage_voltage_csv_path
        

        self.slow_voltage_raw_csv_path=slow_voltage_raw_csv_path
        self.slow_extra_daq_path=extra_daq_path

        self.extra_daq_path=extra_daq_path
        self.voltage_excel_path=voltage_excel_path
        
        
        self.temporary_path=temporary_path
        if extra_daq_path and not temporary_path:
            self.temporary_path=os.path.split(extra_daq_path)[0]
            
            
        self.acquisition_directory_raw=acquisition_directory_raw
        self.acquisition_object=acquisition_object
        

        if self.acquisition_object: 
           
            self.check_slow_files()  
            
            
            
            if self.acquisition_object.raw_input_path:
                self.check_raw_files()
                self.transfer_to_slow()
                self.load_full_daq()
                self.load_full_csv()
                # self.correct_voltage_signals_dictionary_names()
                self.convert_signals_dictionaries_to_feather()
                self.unload_voltage_signals()
                self.unload_voltage_signals_daq()
        else:
            if  self.extra_daq_path:
                self.voltage_recording_extra_daq_slow_full_file_path= self.extra_daq_path
                self.load_full_daq()
                
            if self.voltage_excel_path:
                self.acq_temp_name=os.path.split(os.path.split(self.voltage_excel_path)[0])[1]
                self.temporary_folder=os.path.join(r'C:\Users\sp3660\Desktop\TemporaryProcessing',self.acq_temp_name)
                if not os.path.isdir( self.temporary_folder):
                    os.mkdir(self.temporary_folder)

                self.voltage_recording_raw_file_slow_full_file_path=self.voltage_excel_path
                self.load_full_csv()
            if  self.scanimage_voltage_csv_path:
                self.acq_temp_name=os.path.split(os.path.split(self.scanimage_voltage_csv_path)[0])[1]
                self.temporary_folder=os.path.join(r'C:\Users\sp3660\Desktop\TemporaryProcessing',self.acq_temp_name)
                if not os.path.isdir( self.temporary_folder):
                    os.mkdir(self.temporary_folder)
                self.voltage_recording_raw_file_slow_full_file_path=self.scanimage_voltage_csv_path




                self.load_scainimages_csv()
                


                

                
            
#%% methods

    def load_scainimages_csv(self):
        voltage_signals = pd.read_csv(self.scanimage_voltage_csv_path,names=['VisStim','Locomotion','Nothing'], sep="\t")
        fig, axes = plt.subplots(nrows=3, ncols=1)
        voltage_signals.iloc[:,0].plot(ax=axes[0])
        voltage_signals.iloc[:,1].plot(ax=axes[1])
        voltage_signals.iloc[:,2].plot(ax=axes[2])

        
        for signal in voltage_signals.columns.tolist():
            if 'Locomotion' in signal or  ' Locomotion' in signal:
                self.voltage_signals_dictionary['Locomotion']=voltage_signals[signal].to_frame()
            if 'VisStim' in signal or ' VisStim' in signal:
                self.voltage_signals_dictionary['VisStim']=voltage_signals[signal].to_frame()
          
        self.voltage_signals_dictionary['Time']=pd.DataFrame(np.arange(voltage_signals.shape[0]), columns=['Time'])
        

    def add_time_to_voltage_signals_dic(self):
        
        if self.voltage_signals_dictionary and not set(self.voltage_signals_dictionary).issuperset(['Time']):
            self.voltage_signals_dictionary['Time']=pd.DataFrame(np.arange(self.voltage_signals_dictionary['Locomotion'].shape[0]), columns=['Time'])
            
            


    
    def check_slow_files(self):
        self.create_slow_storage_ftr_paths()
        self.check_if_ftr_files_in_slow_storage()
        self.check_for_csv_slow()
        self.check_for_daq_slow()
        
    def check_raw_files(self):
        
        self.check_for_csv_raw()
        self.check_for_daq_raw()

    def create_slow_storage_ftr_paths(self): 
        keys=['locomotion', 'visual stim', 'eye camera', 'photodiode', 'planes', 'photostim']
  
        if self.acquisition_object:
           tosavepath=self.acquisition_object.slow_storage_all_paths     
        elif self.temporary_path:
            tosavepath={key:self.temporary_path for key in keys}
        elif  self.slow_voltage_raw_csv_path:
            aqpath=os.path.split(os.path.split(self.slow_voltage_raw_csv_path)[0])[0]
            tosavepath={key:os.path.join(aqpath,key) for key in keys}
  
        self.voltage_signals_ftr_path_dictionary={'Locomotion':os.path.join(tosavepath['locomotion'], 'locomotion.ftr') ,
                                         'VisStim':os.path.join(tosavepath['visual stim'], 'vis_stim_voltage.ftr'),
                                         'LED':os.path.join(tosavepath['eye camera'], 'led.ftr'),
                                         'PhotoDiode':os.path.join(tosavepath['photodiode'], 'screen_photodiode.ftr'),
                                         'Frames':os.path.join(tosavepath['planes'], 'frames.ftr'),
                                         'Input2':os.path.join(tosavepath['planes'], 'input2.ftr'),
                                         'Input7':os.path.join(tosavepath['planes'], 'input7.ftr'),
                                         'PhotoStim':os.path.join(tosavepath['photostim'], 'photostim.ftr'),
                                         'PhotoTrig':os.path.join(tosavepath['photostim'], 'phototrigg.ftr'),
                                         'AcqTrig':os.path.join(tosavepath['planes'], 'acqtrig.ftr'),
                                         'Time':os.path.join(tosavepath['planes'], 'time.ftr')

                                             }
        
        self.voltage_signals_ftr_path_daq_dictionary={'Locomotion':os.path.join(tosavepath['locomotion'], 'locomotion_daq.ftr') ,
                                         'VisStim':os.path.join(tosavepath['visual stim'], 'vis_stim_voltage_daq.ftr'),
                                         'LED':os.path.join(tosavepath['eye camera'], 'led_daq.ftr'),
                                         'PhotoDiode':os.path.join(tosavepath['photodiode'], 'screen_photodiode_daq.ftr'),
                                         'Frames':os.path.join(tosavepath['planes'], 'frames_daq.ftr'),
                                         'PhotoStim':os.path.join(tosavepath['photostim'], 'photostim_daq.ftr'),
                                         'PhotoTrig':os.path.join(tosavepath['photostim'], 'phototrigg_daq.ftr'),
                                         'AcqTrig':os.path.join(tosavepath['planes'], 'acqtrig_daq.ftr'),
                                         'Time':os.path.join(tosavepath['planes'], 'time_daq.ftr')
                                             }
        
    def check_if_ftr_files_in_slow_storage(self):
        self.is_file_dictionary={}
        self.is_daq_file_dictionary={}
        
        for name, path in self.voltage_signals_ftr_path_dictionary.items() :
            if os.path.isfile(path) :
                self.no_voltage_signals=False
                if os.path.getsize(path)>0:
                        self.is_file_dictionary[name]=1
                        module_logger.info('signal is there /n' + path)                   
                else:       
                        self.is_file_dictionary[name]='Bad File'
                        module_logger.info('something wrong with file' + path)
            else:
                self.is_file_dictionary[name]=0

                module_logger.info('file not there' + path)
                
        for name, path in self.voltage_signals_ftr_path_daq_dictionary.items() :
            if os.path.isfile(path) :
                self.no_voltage_signals=False
                if os.path.getsize(path)>0:
                        self.is_daq_file_dictionary[name]=1
                        module_logger.info('daq signal is there' + path)                   
                else:       
                        self.is_daq_file_dictionary[name]='Bad File'
                        module_logger.info('something wrong with file' + path)
            else:
                self.is_daq_file_dictionary[name]=0
                module_logger.info('file not there' + path)
                            

    def check_for_csv_raw(self):
            self.voltage_recording_raw_file_full_paths=[file for file in glob.glob(os.path.join(self.acquisition_object.aquisition_path, self.acquisition_object.aquisition_name)+'\\**', recursive=False) if 'VoltageRecording' in file and '.csv' in file   ]
            if  self.voltage_recording_raw_file_full_paths:
                self.voltage_recording_raw_file_full_path= self.voltage_recording_raw_file_full_paths[0]
            else:
                self.voltage_recording_raw_file_full_path=None
                
    def check_for_daq_raw(self):
            self.voltage_recording_extra_daq_full_paths= glob.glob(os.path.join(self.acquisition_object.aquisition_path, 'ExtraDaq')+'\\**.mat', recursive=False) 
            if  self.voltage_recording_extra_daq_full_paths:
                self.voltage_recording_extra_daq_full_path= self.voltage_recording_extra_daq_full_paths[0]
            else:
                self.voltage_recording_extra_daq_full_path=None
                
    def check_for_csv_slow(self):
            
            self.voltage_recording_raw_file_slow_path=self.acquisition_object.slow_storage_all_paths['raw_volatge_csv']    
            self.voltage_recording_raw_file_slow_full_file_paths=glob.glob(self.voltage_recording_raw_file_slow_path+'\\**.csv', recursive=False)           
            if  self.voltage_recording_raw_file_slow_full_file_paths:
                self.voltage_recording_raw_file_slow_full_file_path= self.voltage_recording_raw_file_slow_full_file_paths[0]
            else:
                self.voltage_recording_raw_file_slow_full_file_path=None

    def check_for_daq_slow(self):
            
            self.voltage_recording_extra_daq_slow_path=self.acquisition_object.slow_storage_all_paths['voltage_signals_daq']   
            self.voltage_recording_extra_daq_slow_full_file_paths=glob.glob( self.voltage_recording_extra_daq_slow_path+'\\**.mat', recursive=False) 

            if  self.voltage_recording_extra_daq_slow_full_file_paths:
                self.voltage_recording_extra_daq_slow_full_file_path= self.voltage_recording_extra_daq_slow_full_file_paths[0]    
            else:
                self.voltage_recording_extra_daq_slow_full_file_path=None

    def transfer_to_slow(self):
        
        if  self.voltage_recording_raw_file_full_path and not self.voltage_recording_raw_file_slow_full_file_path:
            self.transfer_raw_csv()     
            self.check_for_csv_slow()
            
        if  self.voltage_recording_extra_daq_full_path and not self.voltage_recording_extra_daq_slow_full_file_path:
            self.transfer_raw_daq()
            self.check_for_daq_slow()

    def transfer_raw_daq(self):       

        final_file_path=os.path.join(self.voltage_recording_extra_daq_slow_path, os.path.split(self.voltage_recording_extra_daq_full_path)[1])
        if not os.path.isfile(final_file_path):
            shutil.copy(self.voltage_recording_extra_daq_full_path, self.voltage_recording_extra_daq_slow_path)

    def transfer_raw_csv(self):
 
        final_file_path=os.path.join(self.voltage_recording_raw_file_slow_path, os.path.split(self.voltage_recording_raw_file_full_path)[1])
        if not os.path.isfile(final_file_path):
            shutil.copy(self.voltage_recording_raw_file_full_path, self.voltage_recording_raw_file_slow_path)
   
  
    def load_full_csv(self):
        csv_path=None
        
        if self.voltage_recording_raw_file_slow_full_file_path:
            csv_path=self.voltage_recording_raw_file_slow_full_file_path 
        elif self.voltage_recording_raw_file_full_path:
            csv_path=self.voltage_recording_raw_file_full_path
        
        if csv_path:
            voltage_signals = pd.read_csv(csv_path)            
            for signal in voltage_signals.columns.tolist()[1:]:
                if 'Locomotion' in signal or  ' Locomotion' in signal:
                    self.voltage_signals_dictionary['Locomotion']=voltage_signals[signal].to_frame()
                if 'VisStim' in signal or ' VisStim' in signal:
                    self.voltage_signals_dictionary['VisStim']=voltage_signals[signal].to_frame()
                if 'LED' in signal or ' LED' in signal:
                    self.voltage_signals_dictionary['LED']=voltage_signals[signal].to_frame()
                if 'PhotoDiode' in signal or 'Photo Diode' in signal :   
                    self.voltage_signals_dictionary['PhotoDiode']=voltage_signals[signal].to_frame()
                if 'PhotoStim' in signal or ' Photostim' in signal or ' UncagiingPokels' in signal:   
                    self.voltage_signals_dictionary['PhotoStim']=voltage_signals[signal].to_frame()
                if 'Optotrigger' in signal or ' Optotrigger' in signal:  
                    self.voltage_signals_dictionary['PhotoTrig']=voltage_signals[signal].to_frame()
                if 'StartEnd' in signal or ' StartEnd' in signal or ' Start/End' in signal: 
                    self.voltage_signals_dictionary['AcqTrig']=voltage_signals[signal].to_frame()
                if 'Frames' in signal:
                    self.voltage_signals_dictionary['Frames']=voltage_signals[signal].to_frame()
                if 'Input2' in signal or 'Input 2' in signal:       
                    self.voltage_signals_dictionary['Input2']=voltage_signals[signal].to_frame()
                if 'Input7' in signal or 'Input 7' in signal:
                    self.voltage_signals_dictionary['Input7']=voltage_signals[signal].to_frame()
                    
                    
            self.voltage_signals_dictionary['Time']=pd.DataFrame(np.arange(voltage_signals.shape[0]), columns=['Time'])

            
   
    def load_full_daq(self):
        extra_daq_path=None
        if self.voltage_recording_extra_daq_slow_full_file_path:
            extra_daq_path=self.voltage_recording_extra_daq_slow_full_file_path
        elif self.voltage_recording_extra_daq_full_path:
            extra_daq_path=self.voltage_recording_extra_daq_full_path
            
        if extra_daq_path:
            mat=sio.loadmat(extra_daq_path)# load mat-file           
            mdata = mat['daq_data']  # variable in mat file 
            ndata = {n: mdata[n][0,0] for n in mdata.dtype.names}
            time_array=ndata['time']
            volt_array=ndata['voltage']
            fullsignals_extra_daq=np.hstack((time_array,volt_array)).T
            
            
            self.new_daq_keys=['Time', 'VisStim', 'Optopockels', 'Start/End', 'LED', 'PhotoTrigger', 'Locomotion']
            self.old_daq_keys=['Time', 'VisStim', 'Photodiode', 'Locomotion', 'LED/Frames', 'Optopockels',]
            if volt_array.shape[1]==5:
                self.daq_keys=self.old_daq_keys
            elif volt_array.shape[1]>=6:
                self.daq_keys=self.new_daq_keys
            

            for i, key in enumerate(self.daq_keys):
                if 'Locomotion' in key:
                    self.voltage_signals_dictionary_daq['Locomotion']=self.voltage_signals_dictionary_daq['Locomotion'].assign(Locomotion=fullsignals_extra_daq[i,:].T.tolist())
                if 'VisStim' in key:
                    self.voltage_signals_dictionary_daq['VisStim']=self.voltage_signals_dictionary_daq['VisStim'].assign(VisStim=fullsignals_extra_daq[i,:].T.tolist())
                if 'LED' in key:
                    self.voltage_signals_dictionary_daq['LED']=self.voltage_signals_dictionary_daq['LED'].assign(LED=fullsignals_extra_daq[i,:].T.tolist()) 
                if 'Photodiode' in key:
                    self.voltage_signals_dictionary_daq['PhotoDiode']=self.voltage_signals_dictionary_daq['PhotoDiode'].assign(PhotoDiode=fullsignals_extra_daq[i,:].T.tolist()) 
                if  'Optopockels' in key:
                    self.voltage_signals_dictionary_daq['PhotoStim']=self.voltage_signals_dictionary_daq['PhotoStim'].assign(PhotoStim=fullsignals_extra_daq[i,:].T.tolist())
                if 'PhotoTrigger' in key:
                    self.voltage_signals_dictionary_daq['PhotoTrig']=self.voltage_signals_dictionary_daq['PhotoTrig'].assign(PhotoTrig=fullsignals_extra_daq[i,:].T.tolist())
                if 'Start/End' in key:
                    self.voltage_signals_dictionary_daq['AcqTrig']=self.voltage_signals_dictionary_daq['AcqTrig'].assign(AcqTrig=fullsignals_extra_daq[i,:].T.tolist()) 
                if 'Time' in key:
                    self.voltage_signals_dictionary_daq['Time']=self.voltage_signals_dictionary_daq['Time'].assign(Time=fullsignals_extra_daq[i,:].T.tolist()) 
                if 'Frames' in key:
                    self.voltage_signals_dictionary_daq['Frames']=self.voltage_signals_dictionary_daq['Frames'].assign(Frames=fullsignals_extra_daq[i,:].T.tolist()) 
             
    def convert_signals_dictionaries_to_feather(self):
        try:            
            recorded_signals=[ signal for signal, df in self.voltage_signals_dictionary.items() if len(df)>0]
            len(recorded_signals)
            saved_signals=[signal_name for signal_name, value in  self.is_file_dictionary.items() if value==1]
            len(saved_signals)

            if  len(recorded_signals) >  len(saved_signals):
                for signal, dataf in self.voltage_signals_dictionary.items():
                    if len(dataf)>0:
                        if not os.path.isfile(self.voltage_signals_ftr_path_dictionary[signal]):
                            self.voltage_signals_dictionary[signal].to_feather(self.voltage_signals_ftr_path_dictionary[signal])  
            else:
                module_logger.info('all csv signals already feathered or not there')
        except:
             module_logger.exception('couldn\'t read csv voltage signals')
        
        try:
            recorded_signals=[ signal for signal, df in self.voltage_signals_dictionary_daq.items() if len(df)>0]
            len(recorded_signals)
            saved_signals=[signal_name for signal_name, value in  self.is_daq_file_dictionary.items() if value==1]
            len(saved_signals)
            if  len(recorded_signals) >  len(saved_signals):
                for signal, dataf in self.voltage_signals_dictionary_daq.items():
                    if len(dataf)>0:
                        if not os.path.isfile(self.voltage_signals_ftr_path_daq_dictionary[signal]):
                            self.voltage_signals_dictionary_daq[signal].to_feather(self.voltage_signals_ftr_path_daq_dictionary[signal])
            else:
                module_logger.info('all daq signals already feathered or not there')
        except:
             module_logger.exception('couldn\'t read daq voltage signals')

   
    def plot_all_signals(self):
        for key in self.voltage_signals_dictionary.keys():
            self.voltage_signals_dictionary[key].plot()
        pass
    
    def plot_all_signals_daq(self):

        for key in self.voltage_signals_dictionary_daq.keys():
            if not self.voltage_signals_dictionary_daq[key].empty:
                self.voltage_signals_dictionary_daq[key].plot()
        pass


    def load_slow_storage_voltage_signals(self):   
    
        self.voltage_signals_dictionary={}
        for name, path in self.voltage_signals_ftr_path_dictionary.items():
            if os.path.isfile(path) :
                if os.path.getsize(path)>0:  
                    self.voltage_signals_dictionary[name]=pd.read_feather(path)
                self.no_voltage_signals=False
            else:
   
                module_logger.info('no voltage signal '+ path)
                
        self.add_time_to_voltage_signals_dic()

                
                
    def load_slow_storage_voltage_signals_daq(self):   
    
        for name, path in self.voltage_signals_ftr_path_daq_dictionary.items():
            if os.path.isfile(path) :
                if os.path.getsize(path)>0:  
                    self.voltage_signals_dictionary_daq[name]=pd.read_feather(path)
                self.no_daq_voltage_signals=False
            else:
   
                module_logger.info('no daq voltage signal '+ path)
        self.add_time_to_voltage_signals_dic()


            
    def unload_voltage_signals(self):
        module_logger.info('Unloading voltage signals')

        if self.voltage_signals_dictionary:
            del self.voltage_signals_dictionary
            gc.collect()
            # sys.stdout.flush()

            
    def unload_voltage_signals_daq(self):
        module_logger.info('Unloading daq voltage signals')

        if self.voltage_signals_dictionary_daq:
            del self.voltage_signals_dictionary_daq
            gc.collect()
            # sys.stdout.flush()
                    
    def signal_extraction_object(self):
        self.extraction_object=VoltageSignalsExtractions(voltage_signals_object=self)

              
          #%%  
if __name__ == "__main__":
    extra=r'G:\Projects\TemPrairireSSH\20220422\Calibrations\SensoryStimulation\UnprocessedDaq\220422_StimTest_0_AllenC_10minspont10reps_4_22_2022_18_42.mat' 
    prairiresignals=r'K:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Chandelier_Imaging\VRC\SLF\Ai65\SPKQ\imaging\20211113\data aquisitions\FOV_1\211113_SPKQ_FOV1_2planeAllenA_20x_920_50024_narrow_without-000\raw_volatge_csv\211113_SPKQ_FOV1_2planeAllenA_20x_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'
    # voltagesignals=VoltageSignals(voltage_excel_path=prairiresignals,extra_daq_path=extra)
    voltagesignals=VoltageSignals(voltage_excel_path=prairiresignals)

    # voltagesignals.plot_all_signals_daq()
    # voltagesignals.plot_all_signals()
    # voltagesignals.signal_extraction_object()
    # extraction=voltagesignals.extraction_object
    


    pass