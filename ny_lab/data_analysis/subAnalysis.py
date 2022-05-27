# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:37:18 2022

@author: sp3660
"""
"""
base object for analysis to integrae different analysis pipeline so they have same input data form main analys onbject
it will also deal with the stimuls selection
and the slicing and activity matrix selection
so in th analysis the input is:
    Activity matrix selection
    Cell slicing
    Stimulus slicing
    Stimulus table definition
    Saving and loading paths
    Saving and loading selections
    Create sweep response matrix
    
"""

import numpy as np
import scipy.stats as st
import pandas as pd

class SubAnalysis():
    
    def __init__(self, results_analysis_object=None):
        
        
        self.selected_plane
        self.selected_activity
        self.selected_cells
        self.selected_stimulus
        self.running_speed
        
        self.sweep_response, self.mean_sweep_response,  self.pval = self.get_sweep_response(self.selected_plane, self.selected_activity, self.selected_stimulus, self.selected_cells)
        
        

    def check_stimulus(self):   
        pass

    def select_planes(self):
        pass

    def filter_stimulus(self):  
        pass
        # spont
        # allena
        #     drift grat
        #     movies
        #     spont
        # allenb
        # allenc
        # mistmatch
        # habituation

    def filter_cells(self):
        pass
        # good cells
        # interneuron
        # pyramidal
        # combine multiple planes indexes in one indexing array
        
    def select_data_matrix(self):
        pass 
 
    def get_sweep_response(self, plane, matrix, stimulus, cells):
        def do_mean(x):
            # +1])
            return np.mean(
                x[self.interlength:
                  self.interlength + self.sweeplength + self.extralength])

        def do_p_value(x):
            (_, p) = \
                st.f_oneway(
                    x[:self.interlength],
                    x[self.interlength:
                      self.interlength + self.sweeplength + self.extralength])
            return p

        C_mat=self.analysis_object.full_data['imaging_data'][plane]['Traces'][matrix]
        self.numbercells=len(C_mat)
        self.celltraces=C_mat
        
        self.dxcm=self.mock_allen_dataset.get_running_speed()
        sweep_response = pd.DataFrame(index=self.stimulus_table['drifting_gratings'].index.values,
                                      columns=list(map(str, range(
                                          self.numbercells + 1))))
        
        sweep_response.rename(
            columns={str(self.numbercells): 'dx'}, inplace=True)
        
        
        for index, row in self.stimulus_table['drifting_gratings'].iterrows():
            start = int(row['start'] - self.interlength)
            end = int(row['start'] + self.sweeplength + self.interlength)

            for nc in range(self.numbercells):
                temp = self.celltraces[int(nc), start:end]
                sweep_response[str(nc)][index] =temp 
            sweep_response['dx'][index] = self.dxcm[start:end]
            
        mean_sweep_response = sweep_response.applymap(do_mean)

        pval = sweep_response.applymap(do_p_value)
        return sweep_response, mean_sweep_response, pval
    
   

