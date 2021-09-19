#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete pipeline for online processing using CaImAn Online (OnACID).
The demo demonstates the analysis of a sequence of files using the CaImAn online
algorithm. The steps include i) motion correction, ii) tracking current 
components, iii) detecting new components, iv) updating of spatial footprints.
The script demonstrates how to construct and use the params and online_cnmf
objects required for the analysis, and presents the various parameters that
can be passed as options. A plot of the processing time for the various steps
of the algorithm is also included.
@author: Eftychios Pnevmatikakis @epnev
Special thanks to Andreas Tolias and his lab at Baylor College of Medicine
for sharing the data used in this demo.
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
bpl.output_notebook()

# %%
logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.WARNING)

    # %%  download and list all files to be processed
    #
    ## folder inside ./example_movies where files will be saved
    #fld_name = 'Mesoscope'
    #download_demo('Tolias_mesoscope_1.hdf5', fld_name)
    #download_demo('Tolias_mesoscope_2.hdf5', fld_name)
    #download_demo('Tolias_mesoscope_3.hdf5', fld_name)
    #
    ## folder where files are located
    #folder_name = os.path.join(caiman_datadir(), 'example_movies', fld_name)
    #extension = 'hdf5'                                  # extension of files
    ## read all files to be processed
    #fnames = glob.glob(folder_name + '/*' + extension)
    #
    ## your list of files should look something like this
    #logging.info(fnames)
    #
    #
      
    
    start_t = time.time()
    
    #f_dir = 'E:\data\V1\proc_data\\'
    #f_dir = 'E:\\data\\Auditory\\caiman_out\\'
    #f_name = 'A2_freq_grating1_10_2_18_OA_cut'
    f_dir = 'E:\\Samuel\\2020\\01_January\\17\\SPDW\\'
    f_name = 'SPDW_GreenbOnly_00001_Plane1'
    #f_name = 'rest1_5_9_19_2_cut_ca';
    f_ext = 'tif'
    fnames = [f_dir + f_name + '.' + f_ext]
    #%%
display_movie = True
if display_movie:
    m_orig = cm.load_movie_chain(fnames)
    ds_ratio = 0.2
    m_orig.resize(1, 1, ds_ratio).play(
        q_max=99.5, fr=30, magnification=2)

    
    # %%   Set up some parameters
    
    fr = 7.61  # frame rate (Hz)
    decay_time = 1  # approximate length of transient event in seconds
    gSig = (5,5)  # expected half size of neurons 5 for 256*256 movies and 10 for 512*512
    p = 2  # order of AR indicator dynamics
    min_SNR = 1   # minimum SNR for accepting new components
    ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
    gnb = 2  # number of background components
    gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int')) # recompute gSig if downsampling is involved
    mot_corr = True  # flag for online motion correction
    pw_rigid = True  # flag for pw-rigid motion correction (slower but potentially more accurate)
    max_shifts_online = 6  # maximum allowed shift during motion correction
    sniper_mode = True  # use a CNN to detect new neurons (o/w space correlation)
    rval_thr = 0.9  # soace correlation threshold for candidate components
    # set up some additional supporting parameters needed for the algorithm
    # (these are default values but can change depending on dataset properties)
    init_batch = 1000  # number of frames for initialization (presumably from the first file)
    K = 2  # initial number of components
    epochs = 1  # number of passes over the data
    show_movie = False # show the movie as the data gets processed
    merge_thr = 0.8
    
    params_dict = {'fnames': fnames,
                   'fr': fr,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': p,
                   'min_SNR': min_SNR,
                   'rval_thr': rval_thr,
                   'merge_thr': merge_thr,
                   'ds_factor': ds_factor,
                   'nb': gnb,
                   'motion_correct': mot_corr,
                   'init_batch': init_batch,
                   'init_method': 'bare',
                   'normalize': True,
                   'sniper_mode': sniper_mode,
                   'K': K,
                   'epochs': epochs,
                   'max_shifts_online': max_shifts_online,
                   'pw_rigid': pw_rigid,
                   'dist_shape_update': True,
                   'min_num_trial': 10,
                   'show_movie': show_movie}
    opts =params.CNMFParams(params_dict=params_dict)
    
   #%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

#%%
# first we create a motion correction object with the parameters specified
mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
# note that the file is not loaded in memory
#%%

#Run piecewise-rigid motion correction using NoRMCorre
mc.motion_correct(save_movie=True)
m_els = cm.load(mc.fname_tot_els)
border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0 
    # maximum shift to be used for trimming against NaNs
    
    
 #%% compare with original movie
display_movie = True
if display_movie:
    m_orig = cm.load_movie_chain(fnames)
    ds_ratio = 0.2
    cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                    m_els.resize(1, 1, ds_ratio)], 
                   axis=2).play(fr=60, gain=5, magnification=2, offset=0)  # press q to exit   
    
#%% MEMORY MAPPING
# memory map the file in order 'C'
fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                           border_to_0=border_to_0, dview=dview) # exclude borders

# now load the file
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F') 
    #load frames in python format (T x X x Y)    
     
    
#%% restart cluster to clean up memory
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)    
    
    

#%%    
    
    

    # %% fit online
    
    
    
