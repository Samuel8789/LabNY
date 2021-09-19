# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:03:57 2020

@author: sp3660
"""

#%% Create varibale siwth my directory structure
from PathStructure import PathStructure
ProjectCodePath, ProjectName, ProjectDataPath, ProjectRAWPath, ProjectTempRAWPath =PathStructure()

#%%




import cv2 # Why? Seems to be for pararelization
import glob # Why? Path name manager
import logging # Why? Logging utility
import numpy as np # Why? Basic array managing
import os # Why? Path builder also
import matplotlib.pyplot as plt


try:
    cv2.setNumThreads(0) # Why? I guess is to depararelize
except:
    pass

try:
    if __IPYTHON__: # tTo check if i am usingh the ipython shell
        get_ipython().magic('load_ext autoreload') # Why TO reload all  modules before very run
        get_ipython().magic('autoreload 2') # Why?
except NameError:
    pass
#%aimport -PathStructure # Ignore the path structure to avoid breaking folder structure


import caiman as cm # Basic caiman functions. Used twice. To start clusters and to load nmap
#from caiman.paths import caiman_datadir # To save the examples movies and the data. Do not use this as it interferes with my own path
from caiman.source_extraction import cnmf as cnmf # Why? RUns the main caiman code
#from caiman.source_extraction.cnmf import params as params # Why? TO generate the parameter file
#from caiman.summary_images import local_correlations_movie_offline # Why?
#from caiman.motion_correction import MotionCorrect
#from caiman.utils.utils import download_demo

#%%

logging.basicConfig(filename=ProjectDataPath+'\\caimanLog.log', format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.DEBUG)
                     
#%%    Select Files
# fnames = [os.path.join(ProjectDataPath, 'demoMovie.tif')]

# Movies=os.listdir(ProjectTempRAWPath)
# files=os.listdir(ProjectTempRAWPath+'\\'+Movies[0])

# matching = [s for s in files if "Ch1" in s]
    
# matching=[ProjectTempRAWPath+'\\'+Movies[0]+'\\' + s for s in matching]

# mv=cm.load(matching[0])
# ch1=mv[:,0,:,:]
# ch2=mv[:,1,:,:]
# ch2.save(ProjectTempRAWPath+'\\Ch2.tiff')
#fname=ProjectTempRAWPath+'\\Ch2.tiff'

# %% CLUSTER SETING

c, dview, n_processes =\
    cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    
# %% set up some parameters

fr = 30  # frame rate (Hz)
decay_time = 0.4  # approximate length of transient event in seconds
gSig = (4,4)  # expected half size of neurons 5 for 256*256 movies and 10 for 512*512
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
K = 20  # initial number of components
epochs = 3  # number of passes over the data
show_movie = False # show the movie as the data gets processed
merge_thr = 0.8
expected_c=5

params_dict = {'fnames': fname,
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
               'show_movie': show_movie,
               'expected_components': expected_c}
opts = cnmf.params.CNMFParams(params_dict=params_dict)

   # %% fit online
    
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    
    cnm.fit_online()
    
    # %% plot contours (this may take time)
    %matplotlib inline
    logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
    images = cm.load(fname)
    Cn = images.local_correlations(swap_dim=False, frames_per_chunk=500)
    cnm.estimates.plot_contours(img=Cn, display_numbers=False)
    
    # %% view components
    cnm.estimates.view_components(img=Cn)
    
    # %% plot timing performance (if a movie is generated during processing, timing
    # will be severely over-estimated)
    
    T_motion = 1e3*np.array(cnm.t_motion)
    T_detect = 1e3*np.array(cnm.t_detect)
    T_shapes = 1e3*np.array(cnm.t_shapes)
    T_track = 1e3*np.array(cnm.t_online) - T_motion - T_detect - T_shapes
    plt.figure()
    plt.stackplot(np.arange(len(T_motion)), T_motion, T_track, T_detect, T_shapes)
    plt.legend(labels=['motion', 'tracking', 'detect', 'shapes'], loc=2)
    plt.title('Processing time allocation')
    plt.xlabel('Frame #')
    plt.ylabel('Processing time [ms]')
    #%% RUN IF YOU WANT TO VISUALIZE THE RESULTS (might take time)
    # c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None,
    #                                  single_thread=False)
    
    
    if opts.online['motion_correct']:
        shifts = cnm.estimates.shifts[-cnm.estimates.C.shape[-1]:]
        if not opts.motion['pw_rigid']:
            memmap_file = cm.motion_correction.apply_shift_online(images, shifts,
                                                        save_base_name=(fname+'MC'))
        else:
            mc = cm.motion_correction.MotionCorrect(fname, dview=dview,
                                                    **opts.get_group('motion'))
    
            mc.y_shifts_els = [[sx[0] for sx in sh] for sh in shifts]
            mc.x_shifts_els = [[sx[1] for sx in sh] for sh in shifts]
            memmap_file = mc.apply_shifts_movie(fname, rigid_shifts=False,
                                                save_memmap=True,
                                                save_base_name=(fname+'MC'))
    else:  # To do: apply non-rigid shifts on the fl
        memmap_file = images.save(fname[0][:-4] + 'mmap')
        
        
    cnm.mmap_file = memmap_file
    Yr, dims, T = cm.load_memmap(memmap_file)
    
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    min_SNR = 2  # peak SNR for accepted components (if above this, acept)
    rval_thr = 0.85  # space correlation threshold (if above this, accept)
    use_cnn = True  # use the CNN classifier
    min_cnn_thr = 0.99  # if cnn classifier predicts below this value, reject
    cnn_lowest = 0.1  # neurons with cnn probability lower than this value are rejected
    
    cnm.params.set('quality',   {'min_SNR': min_SNR,
                                'rval_thr': rval_thr,
                                'use_cnn': use_cnn,
                                'min_cnn_thr': min_cnn_thr,
                                'cnn_lowest': cnn_lowest})
    
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    cnm.estimates.Cn = Cn
    cnm.save(os.path.splitext(fname)[0]+'_ar'+str(p)+'_'+str(gSig[0])+'gsig_results.hdf5')
    
    
    dview.terminate()
    
    duration = time.time() - start_t
    print(duration/60)
#%%










# %% STOP CLUSTER and clean up log files
cm.stop_server(dview=dview)

log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
