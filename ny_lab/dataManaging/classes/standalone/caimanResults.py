# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:19:14 2021

@author: sp3660
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile
try:
    if __IPYTHON__:
        # this is used for debugging purposes only.
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
import time
import matlab.engine
import scipy.io as spio
import glob
from .caimanSorterYSResults import CaimanSorterYSResults
import keyboard
from ...functions.transform_path import transform_path


class CaimanResults():
    
    def __init__(self, hdf5_file_path, movie_path=None, dataset_object=None, caiman_object=None ):
        
        self.caiman_object=caiman_object
        self.dataset_object=dataset_object
        self.movie_path=movie_path
        self.hdf5_file_path=hdf5_file_path

        if dataset_object and caiman_object:
            self.movie_path=self.dataset_object.kalman_object.kalman_path
            self.hdf5_file_path=self.caiman_object.caiman_full_path
         
        self.check_if_mat_file()    
         
    def load_YsSorter_results(self):
        if self.mat_results_path:
            self.ys_sorter_object=CaimanSorterYSResults(self.mat_results_path)
         
    def load_pyhton_cnm_object(self) :      
        self.cnm = cnmf.online_cnmf.OnACID(path=self.hdf5_file_path)


    def check_if_mat_file(self):
        
        self.dataset_dir=os.path.split(self.hdf5_file_path)[0]
        self.mat_results_path=os.path.splitext(self.hdf5_file_path)[0] + '_sort.mat'

        if os.path.isfile(self.mat_results_path):
            self.mat_results_path=self.mat_results_path
        else:
            self.mat_results_path=[]


    def open_caiman_sorter(self):
        self.check_if_mat_file()
        self.options_path=r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\data_pre_processing\caiman_sorter-master\caiman_sorter_options.mat'
        self.options_mat=self.loadmat(self.options_path)
    
        self.options_mat['ops']['file_path_from_python']='\\\\?\\'+transform_path(self.hdf5_file_path, fast_output=True)
        if  self.mat_results_path:
            self.options_mat['ops']['file_path_from_python']='\\\\?\\'+transform_path(self.mat_results_path, fast_output=True)            
        spio.savemat(self.options_path, self.options_mat)
                
        eng = matlab.engine.start_matlab()
        eng.addpath(r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\data_pre_processing\caiman_sorter-master',nargout=0)
        self.dataset_object.open_dataset_directory()
        eng.caiman_sorter(nargout=0)
        print('Press Home To Finish')
        while True:
            if keyboard.is_pressed('home'):  # The same. you can put any key you like instead of 'space'
                break
        self.check_if_mat_file() 

    def load_movie(self):
 
        if  self.dataset_object:
        
            self.dataset_object.kalman_object.load_mc_kalman_tiff()
            self.image_sequence=self.dataset_object.kalman_object.dataset_kalman_caiman_movie

        else:
            if '.tiff' in self.movie_path:
                with tifffile.TiffFile(self.movie_path) as tffl:
                     input_arr = tffl.asarray()
                     self.image_sequence=cm.movie(input_arr.astype(np.uint16))
                del(input_arr)   
            elif '.mmap' in self.movie_path:
                self.image_sequence=cm.load(self.movie_path)
                    


    def loadmat(self,filename):
        '''
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        '''
        def _check_keys(d):
            '''
            checks if entries in dictionary are mat-objects. If yes
            todict is called to change them to nested dictionaries
            '''
            for key in d:
                if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                    d[key] = _todict(d[key])
            return d
    
        def _todict(matobj):
            '''
            A recursive function which constructs from matobjects nested dictionaries
            '''
            d = {}
            for strg in matobj._fieldnames:
                elem = matobj.__dict__[strg]
                if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                    d[strg] = _todict(elem)
                elif isinstance(elem, np.ndarray):
                    d[strg] = _tolist(elem)
                else:
                    d[strg] = elem
            return d
    
        def _tolist(ndarray):
            '''
            A recursive function which constructs lists from cellarrays
            (which are loaded as numpy ndarrays), recursing into the elements
            if they contain matobjects.
            '''
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_tolist(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return elem_list
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)
                
if __name__ == "__main__":
    
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211118_SPKA_FOVGood_optotest2cells_25x_920_50024_narrow_with_cell1_2_210_1-000\Plane1'
    hdf5path=temporary_path+ r'\211118_SPKA_FOVGood_optotest2cells_25x_920_50024_narrow_with_cell1_2_210_1-000_Shifted_Movie_MC_OnACID_20211204-125615_cnmf_results.hdf5'
    movie_path=temporary_path+ r'\211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000_Shifted_Movie_MC_kalman.tiff'
    movie_path2=temporary_path+ r'\211118_SPKA_FOVGood_optotest2cells_25x_920_50024_narrow_with_cell1_2_210_1-000_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_33901_.mmap'
    
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211117_SPKA_Cell1Opto_25x_920_50024_narrow_with_10x_10_ms_16sp_21la_0.6di-000\Plane1'
    hdf5path=temporary_path+ r'\211117_SPKA_Cell1Opto_25x_920_50024_narrow_with_10x_10_ms_16sp_21la_0.6di-000_Shifted_Movie_MC_OnACID_20211204-103542_cnmf_results.hdf5'
    movie_path=temporary_path+ r'\211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000_Shifted_Movie_MC_kalman.tiff'
    movie_path2=temporary_path+ r'\211117_SPKA_Cell1Opto_25x_920_50024_narrow_with_10x_10_ms_16sp_21la_0.6di-000_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_1413_.mmap'
    results=CaimanResults(hdf5path, movie_path2)
    cnmest=results.cnm.estimates
    cnmest.plot_contours()
    activity=cnmest.C
    cnmest.view_components()
   

