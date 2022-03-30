# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:19:14 2021

@author: sp3660
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile
from PIL import Image

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
import matlab.engine
import scipy.io as spio
import glob
from .caimanSorterYSResults_legacy import CaimanSorterYSResults
import keyboard
from ...functions.transform_path import transform_path
from scipy.signal import convolve2d
import copy
from ....data_analysis.resultsAnalysis import ResultsAnalysis
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

        if dataset_object and caiman_object:
            self.movie_path=self.dataset_object.kalman_object.kalman_path
            self.hdf5_file_path=self.caiman_object.caiman_full_path
         
        self.check_if_mat_file()    
         
    def load_YsSorter_results_legacy(self):
        if self.mat_results_path:
            self.ys_sorter_object=CaimanSorterYSResults(self.mat_results_path)
         
    def load_pyhton_cnm_object(self) :      
        self.cnm = cnmf.online_cnmf.OnACID(path=self.hdf5_file_path)

    def load_analysis_object(self):
        self.analysis_object=ResultsAnalysis(caiman_results_object=self)

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
        pass

    def load_YSmat_results(self):
        self.data = mat73.loadmat(self.mat_results_path)
        self.get_all_sorting_indexes()
    
    def load_denoised_traces(self):  
        # self.raw=self.data['est']['C']+self.data['est']['YrA']
        self.C_matrix=self.data['est']['C'][self.accepted_indexes_sorter,:]
        self.YrA_matrix=self.data['est']['C'][self.accepted_indexes_sorter,:]
        self.raw=self.C_matrix + self.YrA_matrix

    def gaussian_smooth_kernel_convolution(self, signal):
        dt = 1000/np.double(self.data['ops']['init_params_caiman']['data']['fr'])
        sigma=50#ms
        sigma_frames = sigma/dt
        # make kernel
        kernel_half_size = int(np.ceil(np.sqrt(-np.log(0.05)*2*sigma_frames**2)))
        gaus_win =list(range( -kernel_half_size,kernel_half_size+1))
        gaus_kernel = [np.exp(-(i**2)/(2*sigma_frames**2)) for i in gaus_win]
        gaus_kernel = gaus_kernel/sum(gaus_kernel)
        conv_trace = convolve2d(np.expand_dims(signal,1), np.expand_dims(gaus_kernel,1), mode='same')
        return conv_trace

    def load_dfdt_traces(self):
        # dfdt traces are saved as smotthe/rectified/normalized acoording to checkboxes form yuriy sirter, dont need to do
        # default threshold of 2 stds 
        self.dfdt=self.data['proc']['deconv']['smooth_dfdt']
        self.dfdt_spikes=self.dfdt['S']
        self.dfdt_std=self.dfdt['S_std']
        self.dfdt_std_threshold=2
        self.dfdt_accepted_matrix=self.dfdt['S'][self.accepted_indexes_sorter,:]
        
        self.dfdt= self.data['proc']['deconv']['smooth_dfdt']
        self.dfdt_spikes= self.dfdt['S']
        self.dfdt_std= self.dfdt['S_std']
        self.dfdt_accepted_matrix= self.dfdt['S'][ self.accepted_indexes_sorter,:]
        self.dfdt_accepted_std= self.dfdt['S_std'][ self.accepted_indexes_sorter]
        self.thresholdline= self.dfdt_std_threshold* self.dfdt_accepted_std
        self.std_filter=np.tile( np.expand_dims( self.thresholdline, 1), [1, self.dfdt_spikes.shape[1]])
        
        
        self.dfdt_thesholded_accepted_matrix=copy.copy( self.dfdt_accepted_matrix)
        self.dfdt_thesholded_accepted_matrix[ self.dfdt_thesholded_accepted_matrix< self.std_filter]= self.std_filter[ self.dfdt_thesholded_accepted_matrix< self.std_filter]
        self.dfdt_thesholded_accepted_matrix=  self.dfdt_thesholded_accepted_matrix- self.std_filter
        plt.plot(self.dfdt_thesholded_accepted_matrix[:,0])
        self.binarized_dfdt=np.where( self.dfdt_thesholded_accepted_matrix > 0, 1, 0)

    def load_foopsi_traces(self):
        # self.foopsi=self.data['proc']['deconv']['c_foopsi']
        # self.foopsi_good_components=[cell[0] for cell in self.foopsi['S'] if cell[0] is not None]
        # self.foopsi_matrix=np.zeros(1)
        # self.foopsi_matrix=np.array(self.foopsi['S']).squeeze().astype('float64')[self.accepted_indexes_sorter,:]
        # self.binarized_foopsi=(self.foopsi_good_components > 0.0001).astype(np.int_)
        pass

    def load_mcmc_traces(self):

        self.mcmc= self.data['proc']['deconv']['MCMC']
        self.mcmc_good_components=[cell[0] for cell in np.array(self.mcmc['S'])[self.accepted_indexes_sorter] if cell[0] is not None]
        self.MCMC_matrix=np.zeros(1)
        self.MCMC_matrix=np.array(self.mcmc_good_components).squeeze().astype('float64')
        self.convolved_MCMC=np.apply_along_axis(self.gaussian_smooth_kernel_convolution, axis=1, arr=self.MCMC_matrix)
        self.convolved_MCMC=np.squeeze(self.convolved_MCMC)
        self.binarized_MCMC=np.where( self.convolved_MCMC > 0, 1, 0)
        # binarized_MCMC=(mcmc_good_components > 0.0001).astype(np.int_)
        

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
                    
    def get_exciatotry_indices(self):
        
        matlabindexespyr=[9,55,88,99,102,142]
        pythoncorrectedpyr=[i-1 for i in matlabindexespyr]
        pyrcellindexactivity=[self.final_accepted_cells_matlabcorrected_indexes.tolist().index(i) for i in pythoncorrectedpyr]
            
    def get_inhibitory_indices(self):
        matlabindexesinter=[2,5,17,30,44,45,66,71]
        pythoncorrectedinter=[i-1 for i in matlabindexesinter]
        intercellindexactivity=[self.final_accepted_cells_matlabcorrected_indexes.tolist().index(i) for i in pythoncorrectedinter]

    def plot_registered_two_color_projections(self):

        self.caiman_object.dataset_object.associated_tomato_dataset.load_dataset(kalman=False)
        
        calcium_image=self.caiman_object.dataset_object.summary_images_object.projection_dic['std_projection_path']
        tomato_image=self.caiman_object.dataset_object.associated_tomato_dataset.summary_images_object.projection_dic['average_projection_path']

  

        masks=self.data['est']['contours']
        selected_masks=[ masks[i] for i in self.accepted_indexes_sorter]
        self.load_pyhton_cnm_object()
        Amat=self.cnm.estimates.A

        maqsks=np.reshape(Amat, (256,256,227))
        
        accepted=Amat[:,self.accepted_indexes_sorter]
     
        
        masks=''
        #overlay accepted mask with tomato and green image
        #measure red intensity in masks
        # make a list of maska and a index selecting tomato /calcium
        
        self.cnm.estimates.plot_contours( tomato_image, self.accepted_indexes_sorter[21:30])
        pyrindexes=[2,5,6,7,8,11,17]
        inteindexes=[1,3,4,9,10,12,13,14,15,16,18,19]
        
        
        
        
    
        
    

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
        self.load_dfdt_traces()
        self.load_mcmc_traces()
                
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
   

