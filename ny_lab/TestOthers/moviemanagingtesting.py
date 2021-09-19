# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 08:59:09 2021

@author: sp3660
"""
import sys
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/AllFunctions')
import os
import caiman as cm
import numpy as np
import tifffile
from kalman_stack_filter import kalman_stack_filter
from save_imagej_hdf5_tiff import save_imagej_hdf5

#%%
selected_dataset_raw_path=r'F:\Projects\LabNY\Imaging\2021\20210522\Mice\SPJO\FOV_1\Aq_1\FaceCamera'
selected_dataset_raw_path=r'F:\Projects\LabNY\Imaging\2021\20210522\Mice\SPJO\FOV_1\Aq_1\FaceCamera\210522_SPJO_HabituationDay2_1_MMStack_Default.ome.tif'
image_sequence_files=os.listdir(selected_dataset_raw_path)
image_sequence_paths= [os.path.join(selected_dataset_raw_path, image) for image in image_sequence_files if '.tif' in image]
image_sequence=cm.load(image_sequence_paths)
image_sequence=cm.load(selected_dataset_raw_path)

image_sequence.play()
image_sequence_changed_type=image_sequence.astype('uint16')
        #%%
selected_dataset_raw_path=r'F:\Projects\LabNY\Imaging\2021\20210522\Mice\SPJO\FOV_1\Aq_1\FaceCamera\210522_SPJO_HabituationDay2_1_MMStack_Default.ome.tif'

file_name=selected_dataset_raw_path
with tifffile.TiffFile(file_name) as tffl:
    multi_page = True if tffl.series[0].shape[0] > 1 else False


        # input_arr = tffl.asarray()
      # for page in tffl.pages:
    input_arr = tffl.asarray()
    # input_arr = np.squeeze(input_arr)
    zzz=cm.movie(input_arr.astype(np.float32),
                      fr= 30,
                      start_time=0,                   
                      file_name=os.path.split(file_name)[-1],
                      meta_data=None,
                      )
zzz.play()
zzzz=zzz.computeDFF()

#%%
movies=[]
selected_dataset_raw_path=r'F:\Projects\LabNY\Imaging\2021\20210522\Mice\SPJO\FOV_1\Aq_1\FaceCamera'
image_sequence_files=os.listdir(selected_dataset_raw_path)
image_sequence_paths= [os.path.join(selected_dataset_raw_path, image) for image in image_sequence_files if '.tif' in image]
for file_name in image_sequence_paths:
    with tifffile.TiffFile(file_name) as tffl:
        input_arr = tffl.asarray()
        zzz=cm.movie(input_arr.astype(np.float32),
                          fr= 30,
                          start_time=0,                   
                          file_name=os.path.split(file_name)[-1],
                          meta_data=None,
                          )
        movies.append(zzz)
        


test=cm.base.timeseries.concatenate(movies[0],movies[1],movies[2])


#%%'
rawmovpath=r'C:\Users\sp3660\Documents\Projects\LabNY\Working_Mice_Data\Mice_Projects\Interneuron_Imaging\G2C\Ai14\SPHV\imaging\20210615\data aquisitions\FOV_1\210615_SPHV_FOV1_10minspont_50024_narrored_920_with-000\planes\Plane1\Green\210615_SPHV_FOV1_10minspont_50024_narrored_920_with-000_d1_256_d2_256_d3_1_order_F_frames_33901_.mmap'
rawmovpath='\\\?\\'+rawmovpath
rawmov=cm.load(rawmovpath)
# rawmov.play(fr=500,gain=0.2)

if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None,
                                    single_thread=False)
  
rawmovMC=cm.motion_correction.MotionCorrect(rawmovpath, dview=dview)
rawmovMC.motion_correct(save_movie=True)

dview.terminate()

rawmovMCpath=os.path.splitext(rawmovpath)[0]+'._rig__d1_256_d2_256_d3_1_order_F_frames_3391_.mmap'
rawmovMCpath=cm.load(rawmovpath, outtype='uint16')

dataset_kalman_array=kalman_stack_filter(rawmovMCpath)
dataset_kalman_caiman_movie=cm.movie(dataset_kalman_array, fr=300,start_time=0,file_name=None, meta_data=None)
save_imagej_hdf5(dataset_kalman_caiman_movie, os.path.splitext(rawmovMCpath)[0]+'_kalman', '.tiff', to32=False)
#%%
# pth=r'F:\Projects\LabNY\Imaging\2021\20210615\Mice\SPHV\FOV_1\Aq_1\210615_SPHV_FOV1_10minspont_50024_narrored_920_with-000\Ch2Green\plane1'
# image_sequence_files=os.listdir(pth)
# image_sequence_paths= [os.path.join(pth, image) for image in image_sequence_files]
# image_sequence=cm.load(image_sequence_paths)
# image_sequence2=cm.load(image_sequence_paths, outtype='uint16')
# image_sequence_changed_type=image_sequence.astype(np.uint16)
# image_sequence2.save(os.path.join(os.path.split(pth)[0], 'test') + '.mmap' , to32=False)

