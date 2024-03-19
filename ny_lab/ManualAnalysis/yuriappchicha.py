#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:03:28 2024

@author: sp3660


loading the cnm object to p[yhton gui app
                            
thing sI need
    images to plot
    accpeted roi list
    rejecvted roi list
    run parameter
    selection parameter
    traces matrixes
                            
                            
"""
import os
from pathlib import Path
import caiman as cm
import numpy as np
import matplotlib.pyplot as plt
from caiman.source_extraction import cnmf
import scipy

def plot_side_by_side(image1, image2, cmap='gray', figsize=(8, 4)):
    """
    Plot two images side by side.

    Parameters:
        image1 (numpy.ndarray): The first image as a NumPy array.
        image2 (numpy.ndarray): The second image as a NumPy array.
        cmap (str, optional): The color map to use for displaying grayscale images. Default is 'gray'.
        figsize (tuple, optional): The figure size. Default is (8, 4).
    """
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot the first image
    axes[0].imshow(image1, cmap=cmap)
    axes[0].set_title('Image 1')
    axes[0].axis('off')

    # Plot the second image
    axes[1].imshow(image2, cmap=cmap)
    axes[1].set_title('Image 2')
    axes[1].axis('off')

    # Show the plot
    plt.show()
#%%

dtsetpath=Path(r'/media/sp3660/Data Slow/Projects/LabNY/Full_Mice_Pre_Processed_Data/Mice_Projects/Chandelier_Imaging/VRC/PVF/SPSM/imaging/20240118/data aquisitions/FOV_1/240118_SPSM_FOV1_2z_ShortDrift_AllCell_617optoFulp_25x_920_51020_60745_with-000/planes/Plane1/Green/')
mmappname='240118_SPSM_FOV1_2z_ShortDrift_AllCell_617optoFulp_25x_920_51020_60745_with-000_plane1_Shifted_Movie_MC_OnACID_d1_256_d2_256_d3_1_order_F_frames_20856.mmap'
hdf5_file_path='240118_SPSM_FOV1_2z_ShortDrift_AllCell_617optoFulp_25x_920_51020_60745_with-000_plane1_Shifted_Movie_MC_OnACID_20240123-000000_cnmf_results.hdf5'
cnm=cnmf.online_cnmf.OnACID(path=dtsetpath / hdf5_file_path)

par=cnm.params
est=cnm.estimates
Yr, dims, T = cm.load_memmap(dtsetpath / mmappname)
images = np.reshape(Yr.T, [T] + list(dims), order='F') 
Cn = cm.local_correlations(images.transpose(1,2,0))
Cn[np.isnan(Cn)] = 0
cnm.estimates.plot_contours(img=Cn)

original=cm.movie(images)



#%% methods
est.view_components(img=Cn) # gui to explore the data find here how they make teh rois 
est.A_thr=None
est.threshold_spatial_components(maxthr=0.4) # threshold A matrix creatin A_th
#%% data
est.vr # no ides
est.sv # no ides

est.time_new_comp # empty
allcomponents, T = est.C.shape

est.A # spatial components as sparse matrix
est.A_thr
est.b #bakground spatial component
est.f #bakground temporal component
est.C #temp components, is the cleaned trace
est.YrA # residual signal of denoised Cq
spatiotemporalnoise=est.b*est.f
tempcompnoised=est.C+est.YrA 

reshapesA=np.reshape(est.A,  list(dims)+[allcomponents] , order='F')

Ycolumn=est.A*(est.C+est.YrA) + est.b*est.f
Ycolumn=est.A*(est.C+est.YrA) + est.b*est.f
reconstructed=cm.movie(np.reshape(Ycolumn.T, [T] + list(dims), order='F') )
reconstructed.play(fr=300)

img = np.reshape(np.array(est.A.mean(axis=1)), est.dims, order='F')
img_thr = np.reshape(np.array(est.A_thr.mean(axis=1)), est.dims, order='F')
img_b = np.reshape(np.array(est.b.mean(axis=1)), est.dims, order='F')

bkgdcomponentsweight= est.C.mean(axis=1)
img_comp=np.reshape(np.array(est.A.sum(axis=1)), est.dims, order='F')
img_weighted = np.reshape(np.array(est.A*bkgdcomponentsweight), est.dims, order='F')




# Plot the images
plot_side_by_side(img, img_b)

#%%

yuribackgrnoise=np.reshape(np.array(est.b*est.f.mean()), est.dims, order='F')
bakground=spatiotemporalnoise
componetimages=img_comp
cweightecomponents=img_weighted
weightedbackground=img_weighted+yuribackgrnoise

c=est.C
raw=tempcompnoised

plot_side_by_side(componetimages,weightedbackground)

# contours to plot

#cnm object has coordinates 
tt=est.coordinates

cellcoord=tt[0]
nA = np.sqrt(np.ravel(est.A.power(2).sum(axis=0)))
nA_inv_mat = scipy.sparse.spdiags(1. / nA, 0, nA.shape[0], nA.shape[0])
A = est.A * nA_inv_mat
A=A.toarray().reshape([256,256,allcomponents])
   
 

