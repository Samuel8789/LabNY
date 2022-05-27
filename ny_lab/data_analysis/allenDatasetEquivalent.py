# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:07:43 2022

@author: sp3660
"""




class AllenDatasetEquivalent():    
    def __init__(self,  full_data , plane):
  
        self.full_data=full_data
        self.plane=plane

    def get_corrected_fluorescence_traces(self, cell_specimen_ids=None):
        
        return (self.full_data['imaging_data'][self.plane]['Timestamps'][0],  self.full_data['imaging_data'][self.plane]['Traces']['denoised']+1)
        # return (self.full_data['imaging_data'][self.plane]['Timestamps'][0],  self.full_data['imaging_data'][self.plane]['Traces']['denoised'])

    
    def get_roi_ids(self):
      
       return self.full_data['imaging_data'][self.plane]['CellIds']
   
    
   
    def get_cell_specimen_ids(self):
        
        return self.full_data['imaging_data'][self.plane]['CellIds']
    
    
    def get_dff_traces(self, cell_specimen_ids=None):
      

        return (self.full_data['imaging_data'][self.plane]['Timestamps'][0], self.full_data['imaging_data'][self.plane]['Traces']['df/f_denoised'])
    
    def get_running_speed(self):
        
        if 'rough' in self.plane:
            self.plane='Plane1'
            
        return (self.full_data['voltage_traces']['Speed'], self.full_data['imaging_data'][self.plane]['Timestamps'][0])
    
    
    def get_stimulus_table(self, stimulus_name):
     


          if stimulus_name == 'drifting_gratings':
             
              return   self.full_data['visstim_info']['Drifting_Gratings']['stimulus_table']

          if stimulus_name == 'spontaneous':
             
              return    self.full_data['visstim_info']['Spontaneous']['stimulus_table']


