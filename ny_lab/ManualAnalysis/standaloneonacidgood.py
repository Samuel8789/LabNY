#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:22:59 2024

@author: sp3660
"""



from IPython import get_ipython
import logging
import matplotlib.pyplot as plt
import numpy as np

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
import time 

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.INFO)

try:
    if __IPYTHON__:
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass
import os
import bokeh.plotting as bpl
from pathlib import Path
datasetpath=Path(os.path.join(os.path.expanduser('~'),r'Desktop/CaimanTemp/'))
fname=datasetpath / '240308_SPSX_FOV1_2z_30m_ShortDrift_LED_opto_1st_25x_920_51020_60745_with-000_plane2_Shifted_Movie_MC_OnACID_d1_256_d2_256_d3_1_order_F_frames_41763.mmap'

volume_period= 0.082796004/2
halfsize=5


fr = 1/ volume_period                                                         # frame rate (Hz)
decay_time = 1                                                    # approximate length of transient event in seconds
gSig = (halfsize,halfsize)                                                        # expected half size of neurons
p = 2                                                             # order of AR indicator dynamics
min_SNR = 1                                                         # minimum SNR for accepting new components
rval_thr = 0.90                                                     # correlation threshold for new component inclusion
ds_factor = 1                                                       # spatial downsampling factor (increases speed but may lose some fine structure)
gnb = 1                                                             # number of background components
gSig = tuple(np.ceil(np.array(gSig)/ds_factor).astype('int'))       # recompute gSig if downsampling is involved
mot_corr = False                                                     # flag for online motion correction 
pw_rigid = False                                                    # flag for pw-rigid motion correction (slower but potentially more accurate)
max_shifts_online = np.ceil(10./ds_factor).astype('int')            # maximum allowed shift during motion correction
sniper_mode = True                                                  # flag using a CNN to detect new neurons (o/w space correlation is used)
init_batch = 200                                                    # number of frames for initialization (presumably from the first file)
expected_comps = 500                                              # maximum number of expected components used for memory pre-allocation (exaggerate here)
dist_shape_update = True                                            # flag for updating shapes in a distributed way
min_num_trial = 10                                                  # number of candidate components per frame     
K = 2                                                           # initial number of components
epochs = 3                                                  # number of passes over the data
show_movie = False                                                  # show the movie with the results as the data gets processed

params_dict = {'fnames': str(fname),
               'fr': fr,
               'decay_time': decay_time,
               'gSig': gSig,
               'p': p,
               'min_SNR': min_SNR,
               'rval_thr': rval_thr,
               'ds_factor': ds_factor,
               'nb': gnb,
               'motion_correct': mot_corr,
               'init_batch': init_batch,
               'init_method': 'bare',
               'normalize': True,
               'expected_comps': expected_comps,
               'sniper_mode': sniper_mode,
               'dist_shape_update' : dist_shape_update,
               'min_num_trial': min_num_trial,
               'K': K,
               'epochs': epochs,
               'max_shifts_online': max_shifts_online,
               'pw_rigid': pw_rigid,
               'show_movie': show_movie}
opts = cnmf.params.CNMFParams(params_dict=params_dict)



cnm = cnmf.online_cnmf.OnACID(params=opts)
cnm.fit_online()


Cn = cm.load(fname).local_correlations(swap_dim=False)
cnm.estimates.Cn = Cn
cnm.estimates.plot_contours(img=Cn) 

cnm.mmap_file = str(fname)
Yr, dims, T = cm.load_memmap(fname)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
cnm.estimates.evaluate_components(images, cnm.params, dview=None)
timestr = time.strftime("%Y%m%d-%H%M%S")
caiman_results_path=str(fname.parent / (fname.stem+ timestr+'_cnmf_results.hdf5'))
cnm.save(caiman_results_path)

