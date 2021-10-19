# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:09:58 2021

@author: sp3660
"""

from caimanSorterYSResults import CaimanSorterYSResults
from cRFsResults import CRFsResults
from voltageSignalsExtractions import VoltageSignalsExtractions
import matplotlib.pyplot as plt
import os

import numpy as np
import matplotlib as mlp


class ResultsAnalysis():
    
    def __init__(self,  plane1_caiman_sorter_results_object=None, plane2_caiman_sorter_results_object=None,plane3_caiman_sorter_results_object=None, crf_results_object=None, acquisition_voltage_signals_object=None ):
        
        
        self.plane1_results=plane1_caiman_sorter_results_object
        self.plane2_results=plane2_caiman_sorter_results_object
        self.plane3_results=plane3_caiman_sorter_results_object
        self.crf_results=crf_results_object
        self.acquisition_voltage_signals=acquisition_voltage_signals_object
        
     
        
        self.allplanesC=np.concatenate((self.plane1_results.C_matrix,
                                    self.plane2_results.C_matrix,
                                    self.plane3_results.C_matrix), axis=0)

        self.allplanesdfdt=np.concatenate((self.plane1_results.dfdt_matrix,
                                      self.plane2_results.dfdt_matrix,
                                      self.plane3_results.dfdt_matrix), axis=0)

        self.allplanesfoopsi=np.concatenate((self.plane1_results.foopsi_matrix,
                                        self.plane2_results.foopsi_matrix,
                                        self.plane3_results.foopsi_matrix))

        self.analyze_drifitng_gratings()
        
    def analyze_drifitng_gratings(self):
        
        self.activity_matrix=self.allplanesfoopsi

        self.gratin1_periods_only=self.activity_matrix[:,self.acquisition_voltage_signals.first_drifitng_set_first:self.acquisition_voltage_signals.first_drifitng_set_last]
        self.gratin2_periods_only=self.activity_matrix[:,self.acquisition_voltage_signals.second_drifitng_set_first:self.acquisition_voltage_signals.second_drifitng_set_last]
        self.gratin3_periods_only=self.activity_matrix[:,self.acquisition_voltage_signals.third_drifitng_set_first:self.acquisition_voltage_signals.third_drifitng_set_last]
              
        self.allplanesfoopsigrating=np.concatenate((self.gratin1_periods_only, self.gratin2_periods_only, self.gratin3_periods_only), axis=1)

        
        # grating_start_indexes=movie_frames_tuning_on
        # grating_end_indexes=movie_frames_tuning_off
        # isi_time=1
        # stim_time=2
        # movie_rate=16.10383676648614 #hz
        # milisecond_period=1000/movie_rate

        # pre_frames=16
        # post_frames=16




        # grating_number=grating_start_indexes.shape[0]
        # repetitions=grating_start_indexes.shape[1]
        # frame_number=np.arange(grating_start_indexes[0,0]-pre_frames,grating_end_indexes[0,0]+post_frames).size

        
        
    def plotting(self):
        fig, ax = plt.subplots(3, 1, sharex=True)
        norm1=mlp.colors.Normalize(0, 1)
        ax[0].imshow(self.allplanesC,aspect='auto',cmap='binary', norm=norm1)
        ax[1].imshow(self.allplanesdfdt,aspect='auto',cmap='binary', norm=norm1)
        ax[2].imshow(self.allplanesfoopsi,aspect='auto',cmap='binary', norm=norm1)
             
        fig, axs = plt.subplots(2,1)
        fig.suptitle('Locomotion')
        
        axs[0].plot(self.acquisition_voltage_signals.second_scale, self.acquisition_voltage_signals.rectified_speed_array)
        axs[0].set_ylim(0, max(self.acquisition_voltage_signals.rectified_speed_array))
        axs[1].imshow(self.allplanesfoopsi,vmin=0, vmax=0.1,aspect='auto',cmap='inferno')


        fig, axs = plt.subplots(1,1)
        axs.imshow(self.gratin1_periods_only.T,vmin=0, vmax=0.1,aspect='auto',cmap='inferno')
        fig.colorbar()  
        
#%%
if __name__ == "__main__":
    # sorter results
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIK3planeallen\Plane1'
    
    linux_temp='/home/samuel/Desktop/SPJAFUllAllen/'
    # windowstemp='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\'
    plane1='Plane1'+os.sep
    plane2='Plane2'+os.sep
    plane3='Plane3'+os.sep
    # temporary_path1=windowstemp+plane1
    # temporary_path2=windowstemp+plane2
    # temporary_path3=windowstemp+plane3
    temporary_path1=linux_temp+plane1
    temporary_path2=linux_temp+plane2
    temporary_path3=linux_temp+plane3

    SPJA_0702_allen_plane1=CaimanSorterYSResults(temporary_path1+ '210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_d1_256_d2_256_d3_1_order_F_frames_64416_cnmf_results_sort.mat')
    SPJA_0702_allen_plane2=CaimanSorterYSResults(temporary_path2+ '210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_d1_256_d2_256_d3_1_order_F_frames_64416_cnmf_results_sort.mat')
    SPJA_0702_allen_plane3=CaimanSorterYSResults(temporary_path3+ '210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_d1_256_d2_256_d3_1_order_F_frames_64416_cnmf_results_sort.mat')
#%% crf results

    temporary_path=linux_temp +os.sep+'SAMUEL_BIG_RUN_2Hz'
    # temporary_path=windowstemp +os.sep+'SAMUEL_BIG_RUN_2Hz'
    # this results are for the 2hz frequency only
    SPJA_0702_allen_CRFS=CRFsResults(temporary_path+ os.sep+'results.mat',
                                     plane1_cell_number=SPJA_0702_allen_plane1.accepted_cells,
                                     plane2_cell_number=SPJA_0702_allen_plane2.accepted_cells ,
                                     plane3_cell_number=SPJA_0702_allen_plane3.accepted_cells
                                     )


#%% voltage signals
     

    temporary_path1=linux_temp +os.sep+'210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'
    voltagesignals=VoltageSignalsExtractions(temporary_path1)

#%%

    analysis=ResultsAnalysis(SPJA_0702_allen_plane1, SPJA_0702_allen_plane2, SPJA_0702_allen_plane3, SPJA_0702_allen_CRFS, voltagesignals)
    analysis.plotting()