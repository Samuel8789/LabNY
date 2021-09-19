# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:41:18 2020

@author: sp3660
"""

# Understanding the caimna python software working on the demos

#FIrst do all the importing and why
#%% Create varibale siwth my directory structure
from PathStructure import PathStructure
ProjectCodePath, ProjectName, ProjectDataPath, ProjectRAWPath, ProjectTempRAWPath =PathStructure()



#%% GENERAL IMPORTINGS

import cv2 # Why? Seems to be for pararelization
import glob # Why? Path name manager
import logging # Why? Logging utility
import numpy as np # Why? Basic array managing
import os # Why? Path builder also

try:
    cv2.setNumThreads(0) # Why? I guess is to depararelize
except:
    pass

try:
    if __IPYTHON__: # tTo check if i am usingh the ipython shell
        get_ipython().magic('load_ext autoreload') # Why TO reload all  modules before very run
        get_ipython().magic('autoreload 2') # Why?
        #%aimport -PathStructure # Ignore the path structure to avoid breaking folder structure
except NameError:
    pass


import caiman as cm # Basic caiman functions. Used twice. To start clusters and to load nmap
from caiman.paths import caiman_datadir # To save the examples movies and the data. Do not use this as it interferes with my own path
from caiman.source_extraction.cnmf import cnmf as cnmf # Why? RUns the main caiman code
from caiman.source_extraction.cnmf import params as params # Why? TO generate the parameter file
from caiman.summary_images import local_correlations_movie_offline # Why?


# %% LOGGER SETTING
# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(filename=ProjectDataPath+'\\caimanLog.log', format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.DEBUG)
                     
    
#%%
def main():
    pass # For compatibility between running under Spyder and the CLI
    
# %% CLUSTER SETING

c, dview, n_processes =\
    cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
#to stop the cluster cm.cluster.stop_server()

# %% LOOKING OFR THE FILES. I HAVE ALSO TO DO A FILE MANAGER FORM PRAIRIE FOR TGHE MOMENT USE THE EXAMPLES A COPIED TO MY DATA FOLDER
fnames = [os.path.join(ProjectDataPath, 'demoMovie.tif')]




# %% PARAMETER SETTING THIS IS WHERE THE MAGIC HAPPENS

is_patches = True       # flag for processing in patches or not
fr = 10                 # approximate frame rate of data
decay_time = 5.0        # length of transient

if is_patches:          # PROCESS IN PATCHES AND THEN COMBINE
    rf = 10             # half size of each patch
    stride = 4          # overlap between patches
    K = 4               # number of components in each patch
else:                   # PROCESS THE WHOLE FOV AT ONCE
    rf = None           # setting these parameters to None
    stride = None       # will run CNMF on the whole FOV
    K = 30              # number of neurons expected (in the whole FOV)

gSig = [6, 6]           # expected half size of neurons
merge_thresh = 0.80     # merging threshold, max correlation allowed
p = 2                   # order of the autoregressive system
gnb = 2                 # global background order

params_dict = {'fnames': fnames,
               'fr': fr,
               'decay_time': decay_time,
               'rf': rf,
               'stride': stride,
               'K': K,
               'gSig': gSig,
               'merge_thr': merge_thresh,
               'p': p,
               'nb': gnb}

opts = params.CNMFParams(params_dict=params_dict)
#%%  Now RUN CaImAn Batch (CNMF)

cnm = cnmf.CNMF(n_processes, params=opts, dview=dview) # LOAD PARAMETER ONLY
cnm = cnm.fit_file()  # RUNS motion correction memory mapping patch cnmf and component evaluation 

# %% plot contour plots of components
Cns = local_correlations_movie_offline(fnames[0],
                                       remove_baseline=True,
                                       swap_dim=False, window=1000, stride=1000,
                                       winSize_baseline=100, quantil_min_baseline=10,
                                       dview=dview)
Cn = Cns.max(axis=0)
cnm.estimates.plot_contours(img=Cn)


# %% load memory mapped file
    Yr, dims, T = cm.load_memmap(cnm.mmap_file)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')


# %% refit
    cnm2 = cnm.refit(images, dview=dview)
    cnm2.estimates.plot_contours(img=Cn)

# %% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier (this will pick up only neurons
    #           and filter out active processes)

    min_SNR = 2      # peak SNR for accepted components (if above this, acept)
    rval_thr = 0.85     # space correlation threshold (if above this, accept)
    use_cnn = True      # use the CNN classifier
    min_cnn_thr = 0.99  # if cnn classifier predicts below this value, reject
    cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

    cnm2.params.set('quality', {'min_SNR': min_SNR,
                                'rval_thr': rval_thr,
                                'use_cnn': use_cnn,
                                'min_cnn_thr': min_cnn_thr,
                                'cnn_lowest': cnn_lowest})

    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)










# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()

