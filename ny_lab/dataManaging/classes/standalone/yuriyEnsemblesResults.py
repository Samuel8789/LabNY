# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:38:06 2021

@author: sp3660
"""


import matplotlib.pyplot as plt

# import h5py
# import scipy.io
import mat73
import numpy as np

class YuriyEnsemblesResults():
    
    def __init__(self, mat_results_file, best_model_file=None, model_collection_file=None, model_parameters_file=None, structures_file=None):
             
        frequencies=[0,45,90,135,180,225,270,315]
        self.mat_results_file = mat_results_file
        self.results = mat73.loadmat(self.mat_results_file)
        self.core_crf=self.results['results']['core_crf']
        self.PCNs=self.results['PCNs']
        self.ensembles={ frequencies[i]:ensemble.astype(int)        for i, ensemble in enumerate(self.core_crf)}
        self.PatternCompletionCells={ frequencies[i]:PCN.astype(int)    for i, PCN in enumerate(self.PCNs)}
        self.stimulus_matrix=self.results['params']['data']
        self.activity_matrix=self.results['params']['UDF']

if __name__ == "__main__":
    
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIK3planeallen\Plane1'
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\SAMUEL_BIG_RUN_2Hz'
    SPJA_0702_allen_CRFS=CRFsResults(temporary_path+ r'\results.mat')
    
    