# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:49:09 2021

@author: sp3660
"""

#%%    temporary placement of files

from PathStructure import PathStructure
projectCodePath, projectName, projectDataPath, projectRAWPath, projectTempRAWPath =PathStructure()   

ouput_movie_path='G:\\CodeTempRawData\\Caiman-Python Testing\\GreyCalibration-001\\GreyCalibration-001_Full_GCaMP.tiff'
#%%
import cv2 # Why? Seems to be for pararelization


try:
    cv2.setNumThreads(0) # Why? I guess is to depararelize
except:
    pass

#%aimport -PathStructure # Ignore the path structure to avoid breaking folder structure


 params # Why? TO generate the parameter file
#from caiman.summary_images import local_correlations_movie_offline # Why?
#from caiman.motion_correction import MotionCorrect
#from caiman.utils.utils import download_demo



#%%


logging.basicConfig(filename=projectDataPath+'\\caimanLog.log', format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.DEBUG)
                     

# %% CLUSTER SETING

c, dview, n_processes =\
    cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    

    
# %% set up some parameters
rastersperframe=2
frameperiod=0.01744321
fr = 1/(rastersperframe*frameperiod)  # frame rate (Hz)
decay_time = 0.4  # approximate length of transient event in seconds
gSig = (4,4)  # expected half size of neurons 5 for 256*256 movies and 10 for 512*512
p = 2  # order of AR indicator dynamics
min_SNR = 1   # minimum SNR for accepting new components
ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
gnb = 2  # number of background components
gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int')) # recompute gSig if downsampling is involved
mot_corr = False  # flag for online motion correction
pw_rigid = False  # flag for pw-rigid motion correction (slower but potentially more accurate)
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
expected_c=20

params_dict = {'fnames': ouput_movie_path,
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
               'expected_comps': expected_c}
opts = cnmf.params.CNMFParams(params_dict=params_dict)

   # %% fit online

cnm = cnmf.online_cnmf.OnACID(params=opts)

cnm.fit_online()

# %% plot contours (this may take time)
#%matplotlib inline
logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
images = cm.load(ouput_movie_path)
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
                                                    save_base_name=(ouput_movie_path+'MC'))
    else:
        mc = cm.motion_correction.MotionCorrect(ouput_movie_path, dview=dview,
                                                **opts.get_group('motion'))

        mc.y_shifts_els = [[sx[0] for sx in sh] for sh in shifts]
        mc.x_shifts_els = [[sx[1] for sx in sh] for sh in shifts]
        memmap_file = mc.apply_shifts_movie(ouput_movie_path, rigid_shifts=False,
                                            save_memmap=True,
                                            save_base_name=(ouput_movie_path+'MC'))
else:  # To do: apply non-rigid shifts on the fl
    memmap_file = images.save(ouput_movie_path[:-4] + '.mmap')
    
    
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
cnm.save(os.path.splitext(ouput_movie_path)[0]+'_ar'+str(p)+'_'+str(gSig[0])+'gsig_results.hdf5')


dview.terminate()

duration = time.time() - start_t
print(duration/60)
#%%










# %% STOP CLUSTER and clean up log files
cm.stop_server(dview=dview)

log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)







            
            
    
            
            