# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:42:00 2023

@author: sp3660
"""
import bokeh.plotting as bpl
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except():
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
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from caiman.summary_images import local_correlations_movie_offline

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.WARNING)


def run_caiman(motion_corrected_movie_path):

    fnames = motion_corrected_movie_path  # filename to be processed
    
    fr = 20                             # imaging rate in frames per second
    decay_time = 0.3                    # length of a typical transient in seconds
    
    # motion correction parameters
    pw_rigid = False             # flag for performing non-rigid motion correction
    
    # parameters for source extraction and deconvolution
    p = 2                      # order of the autoregressive system
    gnb = 2                     # number of global background components
    merge_thr = 0.8           # merging threshold, max correlation allowed
    rf = 20                    # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 10           # amount of overlap between the patches in pixels
    K = 1                       # number of components per patch
    gSig = [2, 2]               # expected half size of neurons in pixels
    method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
    ssub = 1                    # spatial subsampling during initialization
    tsub = 1                    # temporal subsampling during intialization
    
    # parameters for component evaluation
    min_SNR = 1.0               # signal to noise ratio for accepting a component
    rval_thr = 0.85              # space correlation threshold for accepting a component
    cnn_thr = 0.5              # threshold for CNN based classifier
    cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected
    fudge_factor = .99
    opts_dict = {'fnames': fnames,
            'fr': fr,
            'decay_time': decay_time,
            'pw_rigid': pw_rigid,
            'p': p,
            'nb': gnb,
            'fudge_factor': fudge_factor,
            'rf': rf,
            'K': K, 
            'stride': stride_cnmf,
            'method_init': method_init,
            'rolling_sum': True,
            'only_init': True,
            'ssub': ssub,
            'tsub': tsub,
            'merge_thr': merge_thr, 
            'min_SNR': min_SNR,
            'rval_thr': rval_thr,
            'use_cnn': True,
            'min_cnn_thr': cnn_thr,
            'cnn_lowest': cnn_lowest}

    opts = params.CNMFParams(params_dict=opts_dict)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    fname_new=None
    if glob.glob(os.path.split(fnames)[0]+'\memmap_**.mmap'):
        fname_new=glob.glob(os.path.split(fnames)[0]+'\memmap_**.mmap')[0]
    if not fname_new:
        fname_new = cm.save_memmap([fnames], base_name='memmap_', order='C')  # exclude borders


    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    opts.change_params({'p': 0})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)
    # %% plot contours of found components
    Cns = local_correlations_movie_offline(fnames,
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
    cnm.params.change_params({'p': p})
    cnm2 = cnm.refit(images, dview=dview)
    
    print('First refit: %d comp total' % (cnm2.estimates.C.shape[0]))     
    # %% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    min_SNR = 2  # signal to noise ratio for accepting a component
    rval_thr = 0.95  # space correlation threshold for accepting a component
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

    # %% VIEW TRACES (accepted and rejected)
     
    cnm3 = cnm2.refit(images, dview=dview)
    cnm3.estimates.dims = cnm3.dims
    print('Second refit: %d comp total' % (cnm3.estimates.C.shape[0]))
                
    #%% update object with selected components
    #%% Extract DF/F values
    cnm3.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    #%% Show final traces
    #%%
    cnm3.estimates.Cn = Cn
    cnm3.save(cnm2.mmap_file[:-4] + 'hdf5')
    #%% reconstruct denoised movie (press q to exit)
    
    #%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    
    
if __name__ == "__main__":
    motion_corrected_movie_path=    r'K:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Chandelier_Optogenetics\VRC\SLF\Ai65\SPRB\imaging\20230418\data aquisitions\FOV_1\230418_SPRB_FOV1_1z_10min_CellOptoScreen_3x_opto_1_25x_920_51020_63075_with-000\planes\Plane1\Green\230418_SPRB_FOV1_1z_10min_CellOptoScreen_3x_opto_1_25x_920_51020_63075_with-000_plane1_Shifted_Movie_MC_OnACID_d1_256_d2_256_d3_1_order_F_frames_17240_.mmap'
    run_caiman(motion_corrected_movie_path)
