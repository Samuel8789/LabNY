# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:05:00 2023

@author: sp3660
"""

#testing normal cnmf caiman
import os

from pathlib import Path
import caiman as cm
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.summary_images import local_correlations_movie_offline


logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.WARNING)


# analysis.full_data['imaging_data']['Frame_rate']   


#%%
def main():
    channel='Ch2Green'
    plane='plane1'
    mov_strat=0
    slizesice=None
    filename=r'231105_SPRZ_FOV1_2z_ShortDrift6171p_25x_920_51020_63075_with-000_plane1_Shifted_Movie_MC_OnACID_d1_256_d2_256_d3_1_order_F_frames_20875.mmap'
    orderFmmappath=Path(r'K:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Chandelier_Imaging\VRC\PVF\SPRZ\imaging\20231105\data aquisitions\FOV_1\231105_SPRZ_FOV1_2z_ShortDrift6171p_25x_920_51020_63075_with-000\planes\Plane1\Green',filename)
    fnames =[orderFmmappath] # filename to be processed
    
    
    #%% First setup some parameters for data and motion correction
    
    # dataset dependent parameters
    fr = 24.5             # imaging rate in frames per second
    decay_time = 0.2  # length of a typical transient in seconds
    dxy = (2., 2.)      # spatial resolution in x and y in (um per pixel)
    # note the lower than usual spatial resolution here
    max_shift_um = (40., 40.)       # maximum shift in um
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
    
    # motion correction parameters
    pw_rigid = False       # flag to select rigid vs pw_rigid motion correction
    # maximum allowed rigid shift in pixels
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between pathes (size of patch in pixels: strides+overlaps)
    overlaps = (24, 24)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3
    mot_corr=False
    
    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'dxy': dxy,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': 'copy',
        'mot_corr':mot_corr,
        # 'dist':1,
        # 'maxthr': 0.05,   
    }
    
    opts = params.CNMFParams(params_dict=mc_dict)
    
    # %% play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q
    display_images = False
    
    if display_images:
        m_orig = cm.load_movie_chain(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=120, fr=60, magnification=3)
    
    # %% start a cluster for parallel processing
    # c, dview, n_processes = cm.cluster.setup_cluster(
    #     backend='local', n_processes=10, single_thread=False)
    dview=None
    n_processes=1
    # %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    # mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # note that the file is not loaded in memory
    
    # %% Run (piecewise-rigid motion) correction using NoRMCorre
    # mc.motion_correct(save_movie=True)
    
    # %% compare with original movie
    # if display_images:
    #     m_orig = cm.load_movie_chain(fnames)
    #     m_els = cm.load(mc.mmap_file)
    #     ds_ratio = 0.2
    #     moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
    #                                   m_els.resize(1, 1, ds_ratio)], axis=2)
    #     moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit
    
    # %% MEMORY MAPPING
    border_to_0 = 0 #if mc.border_nan == 'copy' else mc.border_to_0
    # you can include the boundaries of the FOV if you used the 'copy' option
    # during motion correction, although be careful about the components near
    # the boundaries
    
    # memory map the file in order 'C'
    # fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
    #                             border_to_0=border_to_0)  # exclude borders
    
    # now load the file
    fil = cm.load(fnames[0])
    if not slizesice:
        slizesice=len(fil)
    fname_new=fil[mov_strat:mov_strat+slizesice,:,:].save(file_name='memmap.mmap',order='C')
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)
    
    # %% restart cluster to clean up memory
    # cm.stop_server(dview=dview)
    # c, dview, n_processes = cm.cluster.setup_cluster(
    #     backend='local', n_processes=10, single_thread=False)
    
    # %%  parameters for source extraction and deconvolution
    p = 1                    # order of the autoregressive system
    gnb = 2                  # number of global background components
    merge_thr = 0.6      # merging threshold, max correlation allowed
    rf = 25
    # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 6          # amount of overlap between the patches in pixels
    K =30                # number of components per patch
    gSig = [4, 4]            # expected half size of neurons in pixels
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    method_init = 'greedy_roi'
    ssub = 1                     # spatial subsampling during initialization
    tsub = 1                    # temporal subsampling during intialization
    epochs=2
    ds_factor = 1
    fudge_factor=0.99
    # parameters for component evaluation
    opts_dict = {'fnames': fnames,
                 'p': p,
                 'fr': fr,
                 'nb': gnb,
                 'rf': rf,
                 'K': K,
                 'gSig': gSig,
                 'gSiz':[i+0 for i in gSig],
                 'stride': stride_cnmf,
                 'method_init': method_init,
                 'rolling_sum': True,
                 'merge_thr': merge_thr,
                 'n_processes': n_processes,
                 'only_init': True,
                 'ssub': ssub,
                 'tsub': tsub,
                 'epochs':epochs,
                 'ds_factor': ds_factor,
                 'fudge_factor':fudge_factor,
                 'maxthr':0.2
                 }
    
    opts.change_params(params_dict=opts_dict);
    # %% RUN CNMF ON PATCHES
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0). If you want to have
    # deconvolution within each patch change params.patch['p_patch'] to a
    # nonzero value
    
    #opts.change_params({'p': 0})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)
    
    # %% ALTERNATE WAY TO RUN THE PIPELINE AT ONCE
    #   you can also perform the motion correction plus cnmf fitting steps
    #   simultaneously after defining your parameters object using
    #  cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
    #  cnm1.fit_file(motion_correct=True)
    
    # %% plot contours of found components
    Cns = local_correlations_movie_offline(fname_new,
                                           remove_baseline=True, window=1000, stride=1000,
                                           winSize_baseline=100, quantil_min_baseline=10,
                                           dview=dview)
    Cn = Cns.max(axis=0)
    Cn[np.isnan(Cn)] = 0
    cnm.estimates.plot_contours(img=Cn)
    plt.title('Contour plots of found components')
    #%% save results
    cnm.estimates.Cn = Cn
    cnm.save(fname_new[:-5]+'_init.hdf5')
    
    # %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    # cnm3 = cnm.refit(images, dview=dview)
    # cnm4 = cnm3.refit(images, dview=dview)

    cnm2 = cnm.refit(images, dview=dview)
    # %% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    min_SNR = 2  # signal to noise ratio for accepting a component
    rval_thr = 0.85  # space correlation threshold for accepting a component
    cnn_thr = 0.99  # threshold for CNN based classifier
    cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected
    
    cnm2.params.set('quality', {'decay_time': decay_time,
                               'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'use_cnn': True,
                               'min_cnn_thr': cnn_thr,
                               'cnn_lowest': cnn_lowest})
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
    # %% PLOT COMPONENTS
    cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
    
    # %% VIEW TRACES (accepted and rejected)
    
    if display_images:
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components)
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components_bad)
    #%% update object with selected components
    cnm2.estimates.select_components(use_object=True)
    #%% Extract DF/F values
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
    
    #%% Show final traces
    cnm2.estimates.view_components(img=Cn)
    #%%
    cnm2.estimates.Cn = Cn
    cnm2.save(cnm2.mmap_file[:-4] + 'hdf5')
    #%% reconstruct denoised movie (press q to exit)
    if display_images:
        cnm2.estimates.play_movie(images, q_max=99.9, gain_res=2,
                                  magnification=2,
                                  bpx=border_to_0,
                                  include_bck=False)  # background not shown
    
    #%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
        
    return fname_new, cnm2
        
    
    
# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    fname_new,cnm2=main()
    os.remove(fname_new)
    399
    279
    
    cnm2.estimates.idx_components[399]
    # components = [[self.idx_components[0], self.idx_components_bad[9]]]
    cnm2.estimates.manual_merge([[399,279]],cnm2.params)
    
