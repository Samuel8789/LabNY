# -*- coding: utf-8 -*-
"""
Created on Wed May 11 20:49:58 2022

@author: sp3660
"""


import os.path
import paramiko
import numpy as np
import pickle
import time
from datetime import datetime

galoistempdir='/home/sp3660/Desktop/caiman_temp'
windowstempdir=r'C:\Users\sp3660\Desktop\CaimanTemp'

filetocopy=r'C:\Users\sp3660\Desktop\CaimanTemp\211113_SPKG_FOV1_3PlaneAllenA_25x_920_50024_narrow_without-000_Shifted_Movie_MC_OnACID_d1_256_d2_256_d3_1_order_F_frames_64415_.mmap'
scripttocoy=r'C:/Users/sp3660/Documents/Github/LabNY/ny_lab/data_pre_processing/galois_caiman.py'
paramstocopy=r'C:\Users\sp3660\Desktop\CaimanTemp\parameter_dict.pkl'

destination='/home/sp3660/Desktop/caiman_temp/211113_SPKG_FOV1_3PlaneAllenA_25x_920_50024_narrow_without-000_Shifted_Movie_MC_OnACID_d1_256_d2_256_d3_1_order_F_frames_64415_.mmap'
scriptdest='/home/sp3660/Desktop/caiman_temp/galois_caiman.py'
paramsdest='/home/sp3660/Desktop/caiman_temp/parameter_dict.pkl'

fr = 16 # frame rate (Hz) 3pl + 4ms = 15.5455
decay_time = 0.5# 2 for s 0.5 for f # approximate length of transient event in seconds
gSig = (5,5)  # expected half size of neurons
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
pipeline='onacid'#'cnmf' 'onacid'
refit=False

# rf = 15                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
# stride_cnmf = 6 

dataset_caiman_parameters = {'fnames': destination,
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
                                   'pipeline':pipeline,
                                   'refit':refit,
                                   # 'rf':rf,
                                   # 'stride':stride_cnmf
                                    }


with open(paramstocopy, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dataset_caiman_parameters, f, pickle.HIGHEST_PROTOCOL)
       



with paramiko.SSHClient() as ssh:
    
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("192.168.0.244", username="sp3660",
                key_filename=os.path.join(os.path.expanduser('~'), ".ssh", "galois"))

    start_transfer=time.time()
    sftp = ssh.open_sftp() 
    # sftp.put(filetocopy, destination)
    sftp.put(scripttocoy, scriptdest)
    sftp.put(paramstocopy, paramsdest)
    sftp.close()
    transfer_duration = time.time() - start_transfer
    print('Transfer Finsihed: {}'.format(transfer_duration/60))
    
    start_processing=time.time()
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print('Start Processing: {}'.format(current_time))
    cmd_run_caiman="source ~/miniconda3/bin/activate caiman; cd Desktop/caiman_temp ;python galois_caiman.py" 
    stdin, stdout, stderr = ssh.exec_command(cmd_run_caiman)
    channel = stdout.channel
    status = channel.recv_exit_status()
    processing_duration = time.time() - start_processing
    print('Processing Finsihed: {}'.format(processing_duration/60))
    

    cmd="ls  '/home/sp3660/Desktop/caiman_temp' | grep \.hdf5" 
    stdin, stdout, stderr = ssh.exec_command(cmd)
    file=stdout.readlines()
    fullfilepath=galoistempdir+'/'+ file[0][:-1]
    hdf5destination=os.path.join(destination, os.path.split(fullfilepath)[1])
    
    sftp = ssh.open_sftp() 
    sftp.get(fullfilepath, hdf5destination)
    sftp.close()






