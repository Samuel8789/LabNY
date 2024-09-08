#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 09:04:55 2024

@author: sp3660

mnual caiman becaus eoif broken params
"""
import os
import matplotlib.pyplot as plt
from pathlib import Path
import caiman as cm
import logging 
module_logger = logging.getLogger(__name__)
import numpy as np
try:
    if __IPYTHON__:
        # this is used for debugging purposes only.
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass


from caiman.motion_correction import MotionCorrect
datasetpath=Path(os.path.join(os.path.expanduser('~'),r'Desktop/CaimanTemp/'))
fnames=str(datasetpath / '240308_SPSX_FOV1_2z_30m_ShortDrift_LED_opto_1st_25x_920_51020_60745_with-000_plane1_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_41763.mmap')


m_orig = cm.load(fnames)
downsample_ratio = .2  # motion can be perceived better when downsampling in time
m_orig.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=30, magnification=2) 


max_shifts = (6, 6)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)

mc = MotionCorrect(fnames, dview=None, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps,
                  max_deviation_rigid=max_deviation_rigid, 
                  shifts_opencv=shifts_opencv, nonneg_movie=True,
                  border_nan=border_nan)

mc.motion_correct(save_movie=True)
#%%
# load motion corrected movie
m_rig = cm.load(mc.mmap_file)
bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(int)
#%% visualize templates
plt.figure(figsize = (20,10))
plt.imshow(mc.total_template_rig, cmap = 'gray');

#%% inspect moviec
m_rig.resize(1, 1, downsample_ratio).play(
    q_max=99.5, fr=30, magnification=2, bord_px = 0*bord_px_rig) # press q to exit

#%% plot rigid shifts
plt.close()
plt.figure(figsize = (20,10))
plt.plot(mc.shifts_rig)
plt.legend(['x shifts','y shifts'])
plt.xlabel('frames')
plt.ylabel('pixels');


#%% motion correct piecewise rigid
# mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
# mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)

# mc.motion_correct(save_movie=True, template=mc.total_template_rig)
# m_els = cm.load(mc.fname_tot_els)
# m_els.resize(1, 1, downsample_ratio).play(
#     q_max=99.5, fr=30, magnification=2,bord_px = bord_px_rig)

# cm.concatenate([m_orig.resize(1, 1, downsample_ratio) - mc.min_mov*mc.nonneg_movie,
#                 m_rig.resize(1, 1, downsample_ratio), m_els.resize(
#             1, 1, downsample_ratio)], axis=2).play(fr=60, q_max=99.5, magnification=2, bord_px=bord_px_rig)

# #%% visualize elastic shifts
# plt.close()
# plt.figure(figsize = (20,10))
# plt.subplot(2, 1, 1)
# plt.plot(mc.x_shifts_els)
# plt.ylabel('x shifts (pixels)')
# plt.subplot(2, 1, 2)
# plt.plot(mc.y_shifts_els)
# plt.ylabel('y_shifts (pixels)')
# plt.xlabel('frames')
#%% compute borders to exclude
bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                  np.max(np.abs(mc.y_shifts_els)))).astype(int)



# plt.figure(figsize = (20,10))
# plt.subplot(1,3,1); plt.imshow(m_orig.local_correlations(eight_neighbours=True, swap_dim=False))
# plt.subplot(1,3,2); plt.imshow(m_rig.local_correlations(eight_neighbours=True, swap_dim=False))
# plt.subplot(1,3,3); plt.imshow(m_els.local_correlations(eight_neighbours=True, swap_dim=False))

# #%%
# final_size = np.subtract(mc.total_template_els.shape, 2 * bord_px_els) # remove pixels in the boundaries
# winsize = 100
# swap_dim = False
# resize_fact_flow = .2    # downsample for computing ROF

# tmpl_orig, correlations_orig, flows_orig, norms_orig, crispness_orig = cm.motion_correction.compute_metrics_motion_correction(
#     fnames, final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

# tmpl_rig, correlations_rig, flows_rig, norms_rig, crispness_rig = cm.motion_correction.compute_metrics_motion_correction(
#     mc.fname_tot_rig[0], final_size[0], final_size[1],
#     swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

# tmpl_els, correlations_els, flows_els, norms_els, crispness_els = cm.motion_correction.compute_metrics_motion_correction(
#     mc.fname_tot_els[0], final_size[0], final_size[1],
#     swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

# plt.figure(figsize = (20,10))
# plt.subplot(211); plt.plot(correlations_orig); plt.plot(correlations_rig); plt.plot(correlations_els)
# plt.legend(['Original','Rigid','PW-Rigid'])
# plt.subplot(223); plt.scatter(correlations_orig, correlations_rig); plt.xlabel('Original'); 
# plt.ylabel('Rigid'); plt.plot([0.3,0.7],[0.3,0.7],'r--')
# axes = plt.gca(); axes.set_xlim([0.3,0.7]); axes.set_ylim([0.3,0.7]); plt.axis('square');
# plt.subplot(224); plt.scatter(correlations_rig, correlations_els); plt.xlabel('Rigid'); 
# plt.ylabel('PW-Rigid'); plt.plot([0.3,0.7],[0.3,0.7],'r--')
# axes = plt.gca(); axes.set_xlim([0.3,0.7]); axes.set_ylim([0.3,0.7]); plt.axis('square');

# # print crispness values
# print('Crispness original: ' + str(int(crispness_orig)))
# print('Crispness rigid: ' + str(int(crispness_rig)))
# print('Crispness elastic: ' + str(int(crispness_els)))

# #%% plot the results of Residual Optical Flow
# fls = [cm.paths.fname_derived_presuffix(mc.fname_tot_els[0], 'metrics', swapsuffix='npz'),
#        cm.paths.fname_derived_presuffix(mc.fname_tot_rig[0], 'metrics', swapsuffix='npz'),
#        cm.paths.fname_derived_presuffix(mc.fname[0],         'metrics', swapsuffix='npz'),
#       ]

# plt.figure(figsize = (20,10))
# for cnt, fl, metr in zip(range(len(fls)), fls, ['pw_rigid','rigid','raw']):
#     with np.load(fl) as ld:
#         print(ld.keys())
#         print(fl)
#         print(str(np.mean(ld['norms'])) + '+/-' + str(np.std(ld['norms'])) +
#               ' ; ' + str(ld['smoothness']) + ' ; ' + str(ld['smoothness_corr']))
        
#         plt.subplot(len(fls), 3, 1 + 3 * cnt)
#         plt.ylabel(metr)
#         print(f"Loading data with base {fl[:-12]}")
#         try:
#             mean_img = np.mean(
#             cm.load(fl[:-12] + '.mmap'), 0)[12:-12, 12:-12]
#         except:
#             try:
#                 mean_img = np.mean(
#                     cm.load(fl[:-12] + '.tif'), 0)[12:-12, 12:-12]
#             except:
#                 mean_img = np.mean(
#                     cm.load(fl[:-12] + 'hdf5'), 0)[12:-12, 12:-12]
                    
#         lq, hq = np.nanpercentile(mean_img, [.5, 99.5])
#         plt.imshow(mean_img, vmin=lq, vmax=hq)
#         plt.title('Mean')
#         plt.subplot(len(fls), 3, 3 * cnt + 2)
#         plt.imshow(ld['img_corr'], vmin=0, vmax=.35)
#         plt.title('Corr image')
#         plt.subplot(len(fls), 3, 3 * cnt + 3)
#         flows = ld['flows']
#         plt.imshow(np.mean(
#         np.sqrt(flows[:, :, :, 0]**2 + flows[:, :, :, 1]**2), 0), vmin=0, vmax=0.3)
#         plt.colorbar()
#         plt.title('Mean optical flow');  