# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 09:33:56 2022

@author: sp3660
"""
import caiman as cm
import sys
import os
import glob
import imagej
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
import scipy.stats as st
from numpy import fft
from typing import Tuple
import numpy as np
from numpy.fft import ifftshift#, fft2, ifft2
from scipy.fft import next_fast_len#, fft2, ifft2
from scipy.ndimage import gaussian_filter1d
from suite2p.registration import register
import time
from suite2p import run_s2p
from suite2p import default_ops
from tempfile import TemporaryDirectory
import suite2p
import tifffile
from ScanImageTiffReader import ScanImageTiffReader
sys.path.insert(0, r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\dataManaging\classes\standalone')
from bidicorrect_image import shiftBiDi, biDiPhaseOffsets
from metadata import Metadata

import scipy as sp
import pickle
from operator import itemgetter
from caiman.source_extraction.cnmf import params as params
from caiman import summary_images
from caiman.source_extraction import cnmf as cnmf

tosavedir=r'C:\Users\sp3660\Desktop\TemporaryProcessing\New folder'
prairireseqdir=r'F:\Projects\LabNY\Imaging\2022\20220922\Mice\SPOU\FOV_1\Aq_1\220022_SPOU_FOV1_3z_AllenA_25x_940_51020_60745_with-000\Ch2Green\plane1'
aqname=os.path.split(os.path.split(os.path.split(prairireseqdir)[0])[0])[1]+'_'+os.path.split(prairireseqdir)[1]
led_start_end_file=aqname+'_LED_start_end.txt'
#%%
initial=1
if initial:
    movie=cm.load(glob.glob(os.path.join(prairireseqdir,'**')))
    # easier loading raw files
    movie.save(os.path.join(tosavedir,aqname)+'.mmap',to32=False)
    movie.save(os.path.join(tosavedir,aqname)+'.tiff',to32=False)
    m_mean = movie.mean(axis=(1, 2))
    scored=st.zscore(m_mean)
    # plt.plot(m_mean)
    dif=np.diff(scored)
    median=sg.medfilt(dif, kernel_size=1)
    rounded=np.round(median)
    plt.plot(rounded)
    # finsg start transition
    if max(abs(dif)) >10*np.mean(abs(dif)):
    
        transitions=np.where(abs(dif)>max(abs(dif))/2)[0]
        transitions_median=np.where(abs(median)>max(abs(median))/2)[0]
        transitions_medina_rounded=np.where(abs(rounded)>max(abs(rounded))/2)[0]
        
        star_led=transitions_medina_rounded[transitions_medina_rounded<int(len(m_mean)/2)]
        end_led=transitions_medina_rounded[transitions_medina_rounded>int(len(m_mean)/2)]
    else:
        star_led=[]
        end_led=[]
        
    
    if len(star_led)>0:
        led_frame_start_end=star_led[-1]+1
        pad1=5
    else:
        led_frame_start_end=0
        pad1=0
    
    if len(end_led)>0:
        led_frame_end_start=end_led[0]
        pad2=4
    else:     
        led_frame_end_start=len(m_mean)+1
        pad2=0
        
    if not os.path.isfile(os.path.join(tosavedir,led_start_end_file)):
         with open(os.path.join(tosavedir,led_start_end_file), 'w') as f:
             f.writelines((str( led_frame_start_end+pad1),'\n', str( led_frame_end_start-pad2)))

    if os.path.isfile(os.path.join(tosavedir,led_start_end_file)):
        with open(os.path.join(tosavedir,led_start_end_file)) as f:
            lines = f.readlines()
        led_start_end=[int(x) for x in lines]
        led_corrected_frame_start= led_start_end[0]
        led_corrected_frame_end=led_start_end[1] 
        
        
        
        
    movie[led_corrected_frame_start:led_corrected_frame_end,:,:].save(os.path.join(tosavedir,aqname)+'_LED_clipped.mmap',to32=False)
    # movie[led_corrected_frame_start:led_corrected_frame_end,:,:].save(os.path.join(tosavedir,aqname)+'_LED_clipped.tiff',to32=False)



    


 #%%

raw_mov=cm.load(glob.glob(os.path.join(tosavedir,aqname)+'_d1**.mmap')[0])
clipped_mov=cm.load(glob.glob(os.path.join(tosavedir,aqname)+'_LED_clipped_d1**.mmap')[0])



bidiphases=[]
for i in range(clipped_mov.shape[0]):
    BiDiPhase=biDiPhaseOffsets(clipped_mov[i,:,:])
    bidiphases.append(BiDiPhase)

plt.plot([tup[0] for tup in bidiphases])

bidiphase_file_path=os.path.join(tosavedir,aqname)+f'_bidiphases_maxshift-{bidiphases[0][1]}_2d-sigma-{bidiphases[0][2]}.pkl'
if bidiphases and bidiphase_file_path:
    if not os.path.isfile(bidiphase_file_path):
        with open(bidiphase_file_path, "wb") as fp:   #Pickling
            pickle.dump(bidiphases, fp)
       

clipped_bidishifted_mov=np.zeros(clipped_mov.shape).astype('float32')
    
for j in range(clipped_mov.shape[0]):
    clipped_bidishifted_mov[j,:,:] =shiftBiDi(bidiphases[j][0], clipped_mov[j,:,:])

sigma=3
smoothed_clipped_bidishifted_mov=cm.movie(sp.ndimage.gaussian_filter1d(clipped_bidishifted_mov, sigma, axis=0))  
smoothed_clipped_mov=cm.movie(sp.ndimage.gaussian_filter1d(clipped_mov, sigma, axis=0))


cm.movie(clipped_bidishifted_mov).save(os.path.join(tosavedir,aqname)+'_LED_clipped_bidicorrected.mmap',to32=False)
# cm.movie(clipped_bidishifted_mov).save(os.path.join(tosavedir,aqname)+'_LED_clipped_bidicorrected.tiff',to32=False)

# smoothed_clipped_mov.save(os.path.join(tosavedir,aqname)+f'_LED_clipped_smoothed_{sigma}_sigma.mmap',to32=False)
smoothed_clipped_mov.save(os.path.join(tosavedir,aqname)+f'_LED_clipped_smoothed_{sigma}_sigma.tiff',to32=False)

# smoothed_clipped_bidishifted_mov.save(os.path.join(tosavedir,aqname)+f'_LED_clipped_bidicorrected_smoothed-{sigma}-sigma.mmap',to32=False)
smoothed_clipped_bidishifted_mov.save(os.path.join(tosavedir,aqname)+f'_LED_clipped_bidicorrected_smoothed-{sigma}-sigma.tiff',to32=False)



  
  
    
#%%
# filtetoregister=glob.glob(os.path.join(tosavedir,aqname)+'**_LED_clipped_bidicorrected.tiff')[0]

# ops = default_ops()
# ops['data_path']=[tosavedir]

# ops['tiff_list']=[os.path.split(filtetoregister)[1]]
# ops['roidetect']=False
# ops['smooth_sigma']=0.1
# db = {
#     'save_path0': os.path.join(tosavedir,'unsmoothed')
# }

# # ops['smooth_sigma_time']=3 ,
# # ops['smooth_sigma']=1.15,

# ops['nonrigid']=False
# run_s2p(ops, db)





# dirpath=os.path.join(tosavedir,'suite2p','plane0')
# ops =  np.load(os.path.join(dirpath,'ops.npy'), allow_pickle=True)
# ops = ops.item()
# # fname = ops['reg_file'] # Let's say input is of shape (4200, 325, 556)
# # Lx, Ly = ops['refImg'].shape # Lx and Ly are the x and y dimensions of the imaging input
# # # Read in our input tif and convert it to a BinaryRWFile
# # f_input = suite2p.io.BinaryRWFile(Ly=Ly, Lx=Lx, filename=fname)







fname=glob.glob(os.path.join(tosavedir,aqname)+'**_LED_clipped_bidicorrected_d1**.mmap')[0]
images=cm.load(glob.glob(os.path.join(tosavedir,aqname)+'**_LED_clipped_bidicorrected_d1**.mmap')[0])

#%% rigid
mc=cm.motion_correction.MotionCorrect(fname, pw_rigid = False)
start=time.time()
mc.motion_correct(save_movie=True)
end=time.time()-start
print(end)
rigid_mc_file=max(glob.glob(os.path.join(tosavedir)+'\**'), key=os.path.getctime)
to_remove=rigid_mc_file.find('_d1')
to_keep=rigid_mc_file.find('_rig__d1')
rigid_mc_file=os.rename(rigid_mc_file, rigid_mc_file[:to_remove]+'_caiman_mc_rigid'+rigid_mc_file[to_keep+5:])
rigid_registered_mov=cm.load(glob.glob(os.path.join(tosavedir,aqname)+'_LED_clipped_bidicorrected_caiman_mc_rigid_d1**.mmap')[0])
cm.movie(sp.ndimage.gaussian_filter1d(rigid_registered_mov, sigma, axis=0)).save(os.path.join(tosavedir,aqname)+f'_LED_clipped_bidicorrected_caiman_mc_rigid_smoothed-{sigma}-sigma.tiff')


#%% non rigid

mc=cm.motion_correction.MotionCorrect(fname, pw_rigid = True)
start=time.time()
mc.motion_correct(save_movie=True)
end=time.time()-start
print(end)

elastic_mc_file=max(glob.glob(os.path.join(tosavedir)+'\**'), key=os.path.getctime)
to_remove=elastic_mc_file.find('_d1')
to_keep=elastic_mc_file.find('_els__d1')
rigid_mc_file=os.rename(elastic_mc_file, elastic_mc_file[:to_remove]+'_caiman_mc_elastic'+elastic_mc_file[to_keep+5:])
elastic_registered_mov=cm.load(glob.glob(os.path.join(tosavedir,aqname)+'_LED_clipped_bidicorrected_caiman_mc_elastic_d1**.mmap')[0])
cm.movie(sp.ndimage.gaussian_filter1d(elastic_registered_mov, sigma, axis=0)).save(os.path.join(tosavedir,aqname)+f'_LED_clipped_bidicorrected_caiman_mc_elastic_smoothed-{sigma}-sigma.tiff')


mc.x_shifts_els
mc.y_shifts_els
#%%
ds_ratio = 0.2
moviehandle = cm.concatenate([images.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                              elastic_registered_mov.resize(1, 1, ds_ratio)], axis=2)
moviehandle.play(fr=60, q_max=99.5, magnification=2)  #

#%%
rigid_registered_mov=cm.load(glob.glob(os.path.join(tosavedir,aqname)+'_LED_clipped_bidicorrected_caiman_mc_rigid_d1**.mmap')[0])
elastic_registered_mov=cm.load(glob.glob(os.path.join(tosavedir,aqname)+'_LED_clipped_bidicorrected_caiman_mc_elastic_d1**.mmap')[0])

raw_mov=cm.load(glob.glob(os.path.join(tosavedir,aqname)+'_d1**.mmap')[0])
clipped_mov=cm.load(glob.glob(os.path.join(tosavedir,aqname)+'_LED_clipped_d1**.mmap')[0])
bidishifted_mov=cm.load(glob.glob(os.path.join(tosavedir,aqname)+'_LED_clipped_bidicorrected_d1**.mmap')[0])

movies=[raw_mov,clipped_mov,bidishifted_mov,rigid_registered_mov,elastic_registered_mov]



#%%
temporary_path1=r'F:\Projects\LabNY\Imaging\2022\20220901\Mice\SPON\FOV_1\Aq_1\220901_SPOL_FOV1_10MinSpont_2z_25x_617_LED_940_51020_60745_with-000'
meta =Metadata(acquisition_directory_raw=temporary_path1)



#%%
rigid=0
elastic=1
if rigid:
    movie_to_process=glob.glob(os.path.join(tosavedir,aqname)+'_LED_clipped_bidicorrected_caiman_mc_rigid_d1**.mmap')[0]
    loaded_movie=rigid_registered_mov
if elastic:
    movie_to_process=glob.glob(os.path.join(tosavedir,aqname)+'_LED_clipped_bidicorrected_caiman_mc_elastic_d1**.mmap')[0]
    loaded_movie=elastic_registered_mov

x=np.ascontiguousarray(loaded_movie)


fname_new = cm.save_memmap([movie_to_process], base_name=os.path.join(tosavedir,aqname)+'memmap_', order='C') 


#e borders
movie_to_process=fname_new


#%%
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
# groups=['data','spatial_params','temporal_params','init_params','preprocess_params','patch_params','online','quality','merging','motion','ring_CNN']

opts = params.CNMFParams()

# dataset dependent parameters
fr = meta.translated_imaging_metadata['FinalFrequency']             # imaging rate in frames per second
decay_time = 0.06    # length of a typical transient in seconds

p = 1                    # order of the autoregressive system
gnb = 2                 # number of global background components
merge_thr = 0.85         # merging threshold, max correlation allowed
rf = 15
# half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 6          # amount of overlap between the patches in pixels
K = 6                   # number of components per patch
gSig = [2.5, 2.5]            # expected half size of neurons in pixels
# initialization method (if analyzing dendritic data using 'sparse_nmf')
method_init = 'greedy_roi'


# parameters for component evaluation
opts_dict = {'fnames': movie_to_process,
             'p': p,
             'fr': fr,
             'nb': gnb,
             'rf': rf,
             'K': K,
             'gSig': gSig,
             'stride': stride_cnmf,
             'method_init': method_init,
             'rolling_sum': True,
             'merge_thr': merge_thr,
             'only_init': True,
             'motion_correct':False}

opts.change_params(params_dict=opts_dict);
# %% RUN CNMF ON PATCHES
# First extract spatial and temporal components on patches and combine them
# for this step deconvolution is turned off (p=0). If you want to have
# deconvolution within each patch change params.patch['p_patch'] to a
# nonzero value

#opts.change_params({'p': 0})
start=time.time()
cnm = cnmf.CNMF(1,params=opts)
cnm.fit_file(motion_correct=False)
end=time.time()-start


# %% plot contours of found components
from caiman.summary_images import local_correlations_movie_offline

Cn = loaded_movie.local_correlations()
Cn[np.isnan(Cn)] = 0
cnm.estimates.plot_contours(img=Cn)
plt.title('Contour plots of found components')
#%% save results
cnm.estimates.Cn = Cn
cnm.save(fname_new[:-5]+'_init.hdf5')

# %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
cnm2 = cnm.refit(cm.load(fname_new))
cnm3 = cnm2.refit(cm.load(fname_new))
cnm4 = cnm3.refit(cm.load(fname_new))
cnm5 = cnm4.refit(cm.load(fname_new))

#%% ONACID

fr = meta.translated_imaging_metadata['FinalFrequency']  # frame rate (Hz) 3pl + 4ms = 15.5455
decay_time = 0.06# 2 for s 0.5 for f # approximate length of transient event in seconds
gSig = (3,3)  # expected half size of neurons
p = 2  # order of AR indicator dynamics
min_SNR = 1.5   # minimum SNR for accepting new components
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
init_batch = 100 # number of frames for initialization (presumably from the first file)
K = 2  # initial number of components
epochs = 3 # number of passes over the data
show_movie = False # show the movie as the data gets processed
merge_thr = 0.8
use_cnn = True  # use the CNN classifier
min_cnn_thr = 0.90  # if cnn classifier predicts below this value, reject
cnn_lowest = 0.3  # neurons with cnn probability lowe
fudge_factor=0.99 #defqault is 0.96


dataset_caiman_parameters = {'fnames': movie_to_process,
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

opts = cnmf.params.CNMFParams(params_dict=dataset_caiman_parameters)
opts.set('temporal', {'fudge_factor':0.99})

cnmo = cnmf.online_cnmf.OnACID(params=opts)
preprocetime=time.time()
cnmo.fit_online()
postprocetime=time.time()


Cn = loaded_movie.local_correlations(swap_dim=False, frames_per_chunk=500)
cnmo.estimates.Cn = Cn


# Cn = MC_movie.local_correlations(swap_dim=False, frames_per_chunk=frames_per_chunk)

cnmo.mmap_file = movie_to_process
Yr, dims, T = cm.load_memmap(movie_to_process)

images = np.reshape(Yr.T, [T] + list(dims), order='F')
cnmo.estimates.evaluate_components(images, cnmo.params, dview=None)
timestr = time.strftime("%Y%m%d-%H%M%S")
cnmo.save(cnm.mmap_file[:-5] + 'online.hdf5')


# %% COMPONENT EVALUATION
# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier
# min_SNR = 2  # signal to noise ratio for accepting a component
# rval_thr = 0.85  # space correlation threshold for accepting a component
# cnn_thr = 0.99  # threshold for CNN based classifier
# cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

# cnm2.params.set('quality', {'decay_time': decay_time,
#                            'min_SNR': min_SNR,
#                            'rval_thr': rval_thr,
#                            'use_cnn': True,
#                            'min_cnn_thr': cnn_thr,
#                            'cnn_lowest': cnn_lowest})
# cnm2.estimates.evaluate_components(images, cnm2.params)
# %% PLOT COMPONENTS
cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
cnm3.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
cnm4.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
cnm5.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

# %% VIEW TRACES (accepted and rejected)

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
cnm2.save(cnm2.mmap_file[:-5] + '2.hdf5')
cnm3.save(cnm3.mmap_file[:-5] + '3.hdf5')
cnm4.save(cnm4.mmap_file[:-5] + '4.hdf5')
cnm5.save(cnm5.mmap_file[:-5] + '5.hdf5')




#%% reconstruct denoised movie (press q to exit)
# cnm2.estimates.play_movie(images, q_max=99.9, gain_res=2,
#                           magnification=2,
#                           include_bck=False)  # background not shown