#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:23:55 2024

@author: sp3660
"""


from IPython import get_ipython
import logging
import matplotlib.pyplot as plt
import numpy as np
import glob
import caiman as cm
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

import bokeh.plotting as bpl
from pathlib import Path
datasetpath=Path(r'/home/sp3660/Desktop/CaimanTemp/')
fname=datasetpath / '230813_SPRM_FOV1_1z_ShortDrift_LED_opto_25x_920_51020_63075_without-000-000_plane1_Shifted_Movie_MC_OnACID_d1_256_d2_256_d3_1_order_F_frames_48841.mmap'


mmapilename=f'{fname.stem[fname.stem.find("_SP")+1:fname.stem.find("_SP")+5]}_{fname.stem[fname.stem.find("_plane")+1:fname.stem.find("_plane")+7]}_load_test.hdf5'
# mmapilename='load_orderF_test.hdf5'
# mmapilename='load_orderC_test.hdf5'
#%%

mov = cm.load(fname)
# mov = cm.movie(np.array(cm.load(fname),order='C'))
# mov = cm.movie(np.array(cm.load(fname),order='F'))




# mmapilename='load_memmap_test.hdf5'

# mmapilename='load_memmap_non_reshaped_orderF_test.hdf5'
# mmapilename='load_memmap_non_reshaped_orderC_test.hdf5'
# mmapilename='load_memmap_reshapoedtodimT_orderF_test.hdf5'
# mmapilename='load_memmap_reshapoedtodimT_orderC_test.hdf5'



# Yr, dims, T = cm.load_memmap(fname)
# images = np.reshape(Yr.T,  list(dims)+[T], order='F')
# images = np.reshape(Yr.T,  Yr.shape, order='C')

# images=Yr
# mov=cm.movie(images)


mov.save(Path(datasetpath, mmapilename))

#%% load artifact corrected movie

fnamedenoised=datasetpath / Path(mmapilename[:-5]+'_denoised.hdf5')
mov = cm.load(fnamedenoised)
mov.save(str(datasetpath / Path(fname.stem+'_denoised.mmap')))

mov=cm.load(glob.glob(str(datasetpath / Path('**denoised**.mmap') ))[0])



