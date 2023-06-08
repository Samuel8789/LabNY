# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:19:14 2021

@author: sp3660
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import glob
import scipy.signal as sg
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys

import tifffile
from PIL import Image
from pydoc import importfile
try:
    if __IPYTHON__:
        # this is used for debugging purposes only.
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass
import mat73
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
import time
try:
    import matlab.engine
except:
        print('Not able to install matlabengine')
import scipy.io as spio
import keyboard
import copy
try:
    from ...functions.transform_path import transform_path
except:
    module = importfile(r'C:/Users/sp3660/Documents/Github/LabNY/ny_lab/dataManaging/functions/transform_path.py')
    transform_path = module.transform_path
    

    
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b",'g','y','c','m', 'tab:brown']) 

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'



class CaimanResults():
    
    def __init__(self, hdf5_file_path, movie_path=None, dataset_object=None, caiman_object=None ):
        
        self.caiman_object=caiman_object
        self.dataset_object=dataset_object
        self.movie_path=movie_path
        self.hdf5_file_path=hdf5_file_path
        
        
        self.MCMC_sigma=50#ms
        self.dfdt_sigma=50#ms
        
        self.binarized_dfdt=np.empty((0, 0))
        self.C_matrix=np.empty((0, 0))
        self.raw=np.empty((0, 0))
        self.dfdt_unsmoothed=np.empty((0, 0))
        self.dfdt_accepted_matrix=np.empty((0, 0))
        self.binarized_dfdt=np.empty((0, 0))
        self.MCMC_matrix=np.empty((0, 0))
        self.convolved_MCMC=np.empty((0, 0))
        self.binarized_MCMC=np.empty((0, 0))
        self.foopsi_matrix=np.empty((0, 0))
        self.convolved_foopsi=np.empty((0, 0))
        self.binarized_foospi=np.empty((0, 0))
        

        if dataset_object and caiman_object:
            self.movie_path=self.dataset_object.kalman_object.kalman_path
            self.hdf5_file_path=self.caiman_object.caiman_full_path
         
            #%%
            self.caiman_object.check_caiman_files()
         #%%
        self.check_if_mat_file()    
                 
    def load_pyhton_cnm_object(self) :      
        self.cnm = cnmf.online_cnmf.OnACID(path=self.hdf5_file_path)


    def check_if_mat_file(self):
        # matlab doesn save full path so change to glob mat files finsihed with sort
        self.dataset_dir=os.path.split(self.hdf5_file_path)[0]
        # self.mat_results_path=os.path.splitext(self.hdf5_file_path)[0] + '_sort.mat'

        # if os.path.isfile(self.mat_results_path):
        #     self.mat_results_path=self.mat_results_path
        # else:
        #     self.mat_results_path=[]


        self.mat_results_paths=glob.glob(self.dataset_dir+'\\**_sort.mat')
        if self.mat_results_paths:
            self.mat_results_path=self.mat_results_paths[0]
        else:
            self.mat_results_path=[]


    def load_caiman_hdf5_results(self):
        self.cnm = cnmf.online_cnmf.OnACID(path=self.hdf5_file_path)
        self.data={}
        self.data['est']={}
        self.data['est']['C']=self.cnm.estimates.C
        self.data['est']['YrA']=self.cnm.estimates.YrA
        self.accepted_indexes_sorter=self.cnm.estimates.idx_components
        self.accepted_cells_number=len( self.accepted_indexes_sorter)
        
        self.dfdt_unsmoothed =np.empty((0))
        self.dfdt_accepted_matrix=np.empty((0))
        self.binarized_dfdt=np.empty((0))
        self.foopsi_matrix=np.empty((0))
        self.convolved_foopsi=np.empty((0))
        self.binarized_foospi=np.empty((0))
        self.MCMC_matrix=np.empty((0))
        self.convolved_MCMC=np.empty((0))
        self.binarized_MCMC=np.empty((0))
        self.accepted_cells_number=np.empty((0))
        self.final_accepted_cells_matlabcorrected_indexes=np.empty((0))
        
    def get_rois_center_of_mass(self):
          
        contours= self.data['est']['contours']
        self.centers_of_mass=[]
        for i, a_cell in enumerate(contours):
            coordinates=a_cell[0]
            cmas=np.mean(coordinates,0)
            self.centers_of_mass.append(cmas)

        accepted_center_of_mass=np.array(self.centers_of_mass)[self.accepted_indexes_sorter]
        self.accepted_center_of_mass=np.around(accepted_center_of_mass).astype('uint16')



    def load_YSmat_results(self):
        if self.mat_results_path:
            self.data = mat73.loadmat(self.mat_results_path)
            self.get_all_sorting_indexes()
        else:
            self.load_caiman_hdf5_results()
            
    def load_denoised_traces(self):  
        # self.raw=self.data['est']['C']+self.data['est']['YrA']
        self.C_matrix=self.data['est']['C'][self.accepted_indexes_sorter,:]
        self.YrA_matrix=self.data['est']['YrA'][self.accepted_indexes_sorter,:]
        self.raw=self.C_matrix + self.YrA_matrix
        
    def z_score(self, X):
       # X: ndarray, shape (n_features, n_samples)
       ss = StandardScaler(with_mean=True, with_std=True)
       Xz = ss.fit_transform(X.T).T
       return Xz
    

    def gaussian_smooth_kernel_convolution(self, signal, sigma):
        dt = 1000/np.double(self.data['ops']['init_params_caiman']['data']['fr'])
        sigma_frames = sigma/dt
        # make kernel
        kernel_half_size = int(np.ceil(np.sqrt(-np.log(0.05)*2*sigma_frames**2)))
        gaus_win =list(range( -kernel_half_size,kernel_half_size+1))
        gaus_kernel = [np.exp(-(i**2)/(2*sigma_frames**2)) for i in gaus_win]
        gaus_kernel = gaus_kernel/sum(gaus_kernel)
        conv_trace = sg.convolve2d(np.expand_dims(signal,1), np.expand_dims(gaus_kernel,1), mode='same')
        return conv_trace

    def load_dfdt_traces(self):
        # dfdt traces are saved as smotthe/rectified/normalized acoording to checkboxes form yuriy sirter, dont need to do
        # default threshold of 2 stds 
    
        self.dfdt_std_threshold=2
        
        self.dfdt= self.data['proc']['deconv']['smooth_dfdt']
        self.dfdt_options=self.dfdt['params']
        self.dfdt_spikes= self.dfdt['S']
        self.dfdt_std= self.dfdt['S_std']
        self.dfdt_accepted_matrix= self.dfdt['S'][ self.accepted_indexes_sorter,:]
        self.dfdt_accepted_std= self.dfdt['S_std'][ self.accepted_indexes_sorter]
        self.thresholdline= self.dfdt_std_threshold* self.dfdt_accepted_std
        self.std_filter=np.tile( np.expand_dims( self.thresholdline, 1), [1, self.dfdt_spikes.shape[1]])
        
        
        self.dfdt_thesholded_accepted_matrix=copy.copy( self.dfdt_accepted_matrix)
        self.dfdt_thesholded_accepted_matrix[ self.dfdt_thesholded_accepted_matrix< self.std_filter]= self.std_filter[ self.dfdt_thesholded_accepted_matrix< self.std_filter]
        self.dfdt_thesholded_accepted_matrix=  self.dfdt_thesholded_accepted_matrix- self.std_filter
        self.binarized_dfdt=np.where( self.dfdt_thesholded_accepted_matrix > 0, 1, 0)

    def load_foopsi_traces(self):
        # self.foopsi=self.data['proc']['deconv']['c_foopsi']
        # self.foopsi_good_components=[cell[0] for cell in self.foopsi['S'] if cell[0] is not None]
        # self.foopsi_matrix=np.zeros(1)
        # self.foopsi_matrix=np.array(self.foopsi['S']).squeeze().astype('float64')[self.accepted_indexes_sorter,:]
        # self.binarized_foopsi=(self.foopsi_good_components > 0.0001).astype(np.int_)
        pass

    def load_mcmc_traces(self):
        self.z_scored_binarized_mcmc_binarized=np.zeros(1)
        self.convolved_MCMC=np.zeros(1)
        self.binarized_MCMC=np.zeros(1)
        self.mcmc= self.data['proc']['deconv']['MCMC']
        self.mcmc_good_components=[cell[0] for cell in np.array(self.mcmc['S'])[self.accepted_indexes_sorter] if cell[0] is not None]
        if self.mcmc_good_components:
            self.MCMC_matrix=np.zeros(1)
            self.MCMC_matrix=np.array(self.mcmc_good_components).squeeze().astype('float64')
            self.convolved_MCMC=np.apply_along_axis(self.gaussian_smooth_kernel_convolution, axis=1, arr=self.MCMC_matrix,  sigma=self.MCMC_sigma)
            self.convolved_MCMC=np.squeeze(self.convolved_MCMC)
            self.binarized_MCMC=np.where( self.convolved_MCMC > 0, 1, 0)
            # binarized_MCMC=(mcmc_good_components > 0.0001).astype(np.int_)
            self.z_scored_binarized_mcmc=self.z_score(self.convolved_MCMC)
            self.mcmcscoredsigma=3
            self.z_scored_binarized_mcmc_binarized=np.where( self.z_scored_binarized_mcmc > self.mcmcscoredsigma, 1, 0)
        

    def get_all_sorting_indexes(self):
        self.accepted_list_sorter=self.data['proc']['comp_accepted'].astype(int)
        self.accepted_list_sorter_core=self.data['proc']['comp_accepted_core'].astype(int)
        # substract 1 from matlab indexes
        self.accepted_indexes_sorter=self.data['proc']['idx_components'].astype(int)-1
        self.rejected_indexes_sorter=self.data['proc']['idx_components_bad'].astype(int)
        self.accepted_indexes_sorter_manual=self.data['proc']['idx_manual'].astype(int)
        self.rejected_indexes_sorter_manual=self.data['proc']['idx_manual_bad'].astype(int)
        self.accepted_indexes_caiman=self.data['est']['idx_components']
        self.rejected_indexes_caiman=self.data['est']['idx_components_bad']    
        self.accepted_cells_number=len( self.accepted_indexes_sorter)
        self.final_accepted_cells_matlabcorrected_indexes= self.accepted_indexes_sorter
       

    def open_caiman_sorter(self):
        self.check_if_mat_file()
        self.options_path=r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\data_pre_processing\caiman_sorter-master\caiman_sorter_options.mat'
        #testing new versions
        # self.options_path=r'C:\Users\sp3660\Downloads\caiman_sorter-master\caiman_sorter_options.mat'

        self.options_mat=self.loadmat(self.options_path)
        self.options_mat['ops']['file_path_from_python']='\\\\?\\'+transform_path(self.hdf5_file_path, fast_output=False)
        if  self.mat_results_path:
            self.options_mat['ops']['file_path_from_python']='\\\\?\\'+transform_path(self.mat_results_path, fast_output=False)            
        spio.savemat(self.options_path, self.options_mat)
        self.dataset_object.open_dataset_directory()

        if 'matlab.engine' in sys.modules:        
            eng = matlab.engine.start_matlab()
            eng.addpath(r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\data_pre_processing\caiman_sorter-master',nargout=0)
        #testing new versions
        #         eng.addpath(r'C:\Users\sp3660\Downloads\caiman_sorter-master',nargout=0)

            eng.caiman_sorter(nargout=0)
            print('Press Home To Finish')
            while True:
                if keyboard.is_pressed('home'):  # The same. you can put any key you like instead of 'space'
                    break
        else:
            print('Cant connect to matlab')
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
                    


        
    def plot_final_rasters(self):    
        self.accepted_cells_number
        self.accepted_indexes_sorter
        
        pixel_per_bar = 4
        dpi = 100
        
               
        fig,ax = plt.subplots(2,  figsize=(16,9), dpi=dpi, sharex=True)
        ax[0].imshow(self.dfdt_thesholded_accepted_matrix, cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        ax[0].title.set_text('Smoothed_thresholded_dfdt_{}_{}'.format(self.dfdt_sigma, self.dfdt_std_threshold))
        ax[1].imshow( self.binarized_dfdt, cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        ax[1].title.set_text('Binarized_Smoothed_thresholded_dfdt_{}_{}_{}'.format(0, self.dfdt_sigma, self.dfdt_std_threshold))
        ax[1].set_xlabel('Time(s)')
        fig.supylabel('Cell Number')
        fig.suptitle('Raster_df/dt')
        
        fig,ax = plt.subplots(3,  figsize=(16,9), dpi=dpi, sharex=True)
        ax[0].imshow( self.MCMC_matrix, cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        ax[0].title.set_text('Raw_MCMC')
        ax[1].imshow( self.convolved_MCMC, cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        ax[1].title.set_text('Smoothed_MCMC_{}'.format(self.MCMC_sigma))
        ax[2].imshow( self.binarized_MCMC, cmap='binary', aspect='auto',
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        ax[2].title.set_text('Binarized_smoothed_MCMC_{}_{}'.format(0, self.MCMC_sigma ))
        ax[-1].set_xlabel('Time(s)')
        fig.supylabel('Cell Number')
        fig.suptitle('Raster_MCMC')
    
        

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
    
    def load_all(self):
        self.load_YSmat_results()
        self.load_denoised_traces()
        if self.mat_results_path:
            self.load_dfdt_traces()
            self.load_mcmc_traces()
            self.get_rois_center_of_mass()

                
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
   

