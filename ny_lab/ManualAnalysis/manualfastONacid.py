# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:49:34 2023

@author: sp3660
"""

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
import os 
import numpy as np 
import time 

fr = imagingrate  # frame rate (Hz) 3pl + 4ms = 15.5455
decay_time = 0.2# 2 for s 0.5 for f # approximate length of transient event in seconds
gSig = (10,10)  # expected half size of neurons
p = 2  # order of AR indicator dynamics
min_SNR = 1   # minimum SNR for accepting new components
ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
gnb = 2  # number of background components
gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int')) # recompute gSig if downsampling is involved
mot_corr = False  # flag for online motion correction
pw_rigid = False  # flag for pw-rigid motion correction (slower but potentially more accurate)
max_shifts_online = 10  # maximum allowed shift during motion correction
sniper_mode = True  # use a CNN to detect new neurons (o/w space correlation)
rval_thr = 0.8  # soace correlation threshold for candidate components
# set up some additional supporting parameters needed for the algorithm
# (these are default values but can change depending on dataset properties)
init_batch = 500 # number of frames for initialization (presumably from the first file)
K = 1  # initial number of components
epochs = 2 # number of passes over the data
show_movie = False # show the movie as the data gets processed
merge_thr = 0.2
use_cnn = True  # use the CNN classifier
min_cnn_thr = 0.90  # if cnn classifier predicts below this value, reject
cnn_lowest = 0.3  # neurons with cnn probability lowe
fudge_factor=0.99 #defqault is 0.96

parameter_dict = {'fnames': moviepath,
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
                                'max_shifts_online': max_shifts_online,
                                'pw_rigid': pw_rigid,
                                'dist_shape_update': True,
                                'min_num_trial': 10,
                                'show_movie': show_movie,
                                'epochs':epochs,
                                'use_cnn': use_cnn,
                                'min_cnn_thr': min_cnn_thr,
                                'cnn_lowest': cnn_lowest,
                                'fudge_factor':fudge_factor
                                 }


fnamestemp=parameter_dict['fnames']
opts = cnmf.params.CNMFParams(params_dict=parameter_dict)
cnm = cnmf.online_cnmf.OnACID(params=opts)
cnm.fit_online()




images = cm.load(fnamestemp)
# module_logger.info('calculating dff')
# cnm.estimates.detrend_df_f()

mmap_directory, caiman_filename=os.path.split(fnamestemp)
good_filename=caiman_filename[:caiman_filename.find('.t')]   
MC_onacid_file_path='_'.join([os.path.join(mmap_directory, good_filename)])

MC_movie = images  
Cn = MC_movie.local_correlations(swap_dim=False, frames_per_chunk=500)
cnm.estimates.Cn = Cn

# Cn = MC_movie.local_correlations(swap_dim=False, frames_per_chunk=frames_per_chunk)


timestr = time.strftime("%Y%m%d-%H%M%S")
caiman_results_path='_'.join([MC_onacid_file_path, timestr+'.hdf5'])  
cnm.save(caiman_results_path)


#%%


cnm = cnmf.online_cnmf.OnACID(path=r'C:\Users\sp3660\Desktop\Chandelier_ Calcium&Volatage\Chandelie_AS-003_20230610-102153.hdf5')
cnm.estimates.evaluate_components(images, cnm.params)


components = [[cnm.estimates.idx_components[0], cnm.estimates.idx_components_bad[9]]]
components = [[4,43]]

tt=cnm.estimates.manual_merge( components,cnm.params)