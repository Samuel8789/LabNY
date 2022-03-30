# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:19:36 2021

@author: sp3660
"""
import matplotlib.pyplot as plt

# import h5py
# import scipy.io
import mat73
import numpy as np
import matplotlib as mlp

class CaimanSorterYSResults():
    
    def __init__(self, mat_file):
        print('Processing Metadata')

             
        self.mat_file = mat_file
        self.data = mat73.loadmat(self.mat_file)

        self.raw=self.data['est']['C']+self.data['est']['YrA']
        self.dfdt=self.data['proc']['deconv']['smooth_dfdt']
        self.foopsi=self.data['proc']['deconv']['c_foopsi']
        self.mcmc= self.data['proc']['deconv']['MCMC']


        self.dfdt_spikes=self.dfdt['S']
        self.dfdt_std=self.dfdt['S_std']
        # self.std_filter=np.tile( self.dfdt_std, [self.dfdt_std.shape[0],self.dfdt_spikes.shape[1]])

        self.foopsi_good_components=[cell[0] for cell in self.foopsi['S'] if cell[0] is not None]
        self.mcmc_good_components=[cell[0] for cell in self.mcmc['S'] if cell[0] is not None]

        self.accepted_list_sorter=self.data['proc']['comp_accepted'].astype(int)
        self.accepted_list_sorter_core=self.data['proc']['comp_accepted_core'].astype(int)
        self.accepted_indexes_sorter=self.data['proc']['idx_components'].astype(int)-1
        self.rejected_indexes_sorter=self.data['proc']['idx_components_bad'].astype(int)
        self.accepted_indexes_sorter_manual=self.data['proc']['idx_manual'].astype(int)
        self.rejected_indexes_sorter_manual=self.data['proc']['idx_manual_bad'].astype(int)
        self.accepted_indexes_caiman=self.data['est']['idx_components']
        self.rejected_indexes_caiman=self.data['est']['idx_components_bad']
        self.accepted_cells=len( self.accepted_indexes_sorter)
        
        self.C_matrix=self.data['est']['C'][self.accepted_indexes_sorter,:]
        self.dfdt_matrix=self.dfdt['S'][self.accepted_indexes_sorter,:]
        
        self.foopsi_matrix=np.zeros(1)
        self.MCMC_matrix=np.zeros(1)
        # self.foopsi_matrix=np.array(self.foopsi['S']).squeeze().astype('float64')[self.accepted_indexes_sorter,:]
        # correctedmcmc=[x if isinstance(x[0],(np.ndarray)) else [np.zeros((self.foopsi['S'][0][0].shape))] for x in self.mcmc['S']]
        # self.MCMC_matrix=np.array(correctedmcmc).squeeze().astype('float64')[self.accepted_indexes_sorter,:]
        

        # self.binarized_dfdt=np.where(self.dfdt_matrix > 2* self.dfdt_std[:,None], 1, 0)
        # self.binarized_MCMC=(self.mcmc_good_components > 0.0001).astype(np.int_)
        # self.binarized_foopsi=(self.foopsi_good_components > 0.0001).astype(np.int_)
        

        # self.plotting()
 
     
    def plotting(self, cell_range=None):
        if not cell_range:
            cell_range=range(0,self.binarized_dfdt.shape[0])
        plt.figure()    
        plt.imshow(self.binarized_MCMC[cell_range,], cmap='inferno', aspect='auto')   
        plt.figure()
        plt.imshow(self.binarized_foopsi[cell_range,:], cmap='inferno', aspect='auto', vmax=0.2)    
        plt.figure()
        plt.imshow(self.binarized_dfdt, cmap='inferno', aspect='auto', vmax=0.2)    

if __name__ == "__main__":
    
    # # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIK3planeallen\Plane1'
    temporary_path1='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\Plane1'
    temporary_path2='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\Plane2'
    temporary_path3='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\Plane3'
    SPJA_0702_allen_plane1=CaimanSorterYSResults(temporary_path1+ r'\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_d1_256_d2_256_d3_1_order_F_frames_64416_cnmf_results_sort.mat')
    SPJA_0702_allen_plane2=CaimanSorterYSResults(temporary_path2+ r'\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_d1_256_d2_256_d3_1_order_F_frames_64416_cnmf_results_sort.mat')
    SPJA_0702_allen_plane3=CaimanSorterYSResults(temporary_path3+ r'\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_d1_256_d2_256_d3_1_order_F_frames_64416_cnmf_results_sort.mat')




    # temporary_path1='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\Plane1'
    # temporary_path1='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\Plane2'

    # SPKG=CaimanSorterYSResults(temporary_path1+ r'\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_MC_OnACID_20211020-013317_cnmf_results_sort.mat')
    # SPKG=CaimanSorterYSResults(temporary_path1+ r'\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_MC_OnACID_20211020-013317_cnmf_results_sort.mat')

    totalcells=SPJA_0702_allen_plane1.accepted_cells + SPJA_0702_allen_plane2.accepted_cells + SPJA_0702_allen_plane3.accepted_cells



    plt.imshow(SPJA_0702_allen_plane1.raw, aspect='auto')


