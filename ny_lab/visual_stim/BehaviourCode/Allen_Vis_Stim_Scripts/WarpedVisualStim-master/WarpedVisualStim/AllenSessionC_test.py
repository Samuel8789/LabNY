# -*- coding: utf-8 -*-
"""
Example script to test StimulusRoutines.CombinedStimuli class
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import WarpedVisualStim as rm
import WarpedVisualStim.StimulusRoutines as stim
from WarpedVisualStim.MonitorSetup import Monitor, Indicator
from WarpedVisualStim.DisplayStimulus import DisplaySequence

# ============================ monitor setup ======================================
mon_resolution = (1440, 2560)  # enter your monitors resolution
mon_width_cm = 60.  # enter your monitors width in cm
mon_height_cm = 33.6  # enter your monitors height in cm
mon_refresh_rate = 60  # enter your monitors height in Hz
mon_C2T_cm = mon_height_cm / 2.
mon_C2A_cm = mon_width_cm / 2.
mon_center_coordinates = (0., 60.)
mon_dis = 15.
mon_downsample_rate = 5
# =================================================================================

# ============================ indicator setup ====================================
ind_width_cm = 1.
ind_height_cm = 1.
ind_position = 'northeast'
ind_is_sync = True
ind_freq = 1.
# =================================================================================

# ============================ DisplaySequence ====================================
ds_log_dir = r'C:\data'
# ds_log_dir = '/home/zhuangjun1981'
ds_backupdir = None
ds_identifier = 'TEST'
ds_display_iter = 1
ds_mouse_id = 'MOUSE'
ds_user_id = 'USER'
ds_psychopy_mon = 'testMonitor'
ds_is_by_index = True
ds_is_interpolate = False
ds_is_triggered = False
ds_is_save_sequence = False
ds_trigger_event = "negative_edge"
ds_trigger_NI_dev = 'Dev1'
ds_trigger_NI_port = 1
ds_trigger_NI_line = 0
ds_is_sync_pulse = False
ds_sync_pulse_NI_dev = 'Dev1'
ds_sync_pulse_NI_port = 1
ds_sync_pulse_NI_line = 1
ds_display_screen = 0
ds_initial_background_color = 0.
ds_color_weights = (1., 1., 1.)
# =================================================================================

# ============================ generic stimulus parameters ========================
pregap_dur = 2.
postgap_dur = 3.
background = 0.
coordinate = 'degree'
# =================================================================================

# ============================ UniformContrast ====================================
uc_duration = 10.
uc_color = 0
# =================================================================================

# ============================ FlashingCircle =====================================
fc_center = (15, 60)
fc_radius = 30.
fc_color = -1.
fc_flash_frame_num = 30
fc_midgap_dur = 5.
fc_iteration = 5.
fc_is_smooth_edge = True
fc_smooth_width_ratio = 0.2
fc_smooth_func = stim.blur_cos
# =================================================================================

# ============================ SinusoidalLuminance ================================
sl_max_level = 1.
sl_min_level = -1.
sl_frequency = 1.
sl_cycle_num = 10
sl_start_phase = 0.
sl_midgap_dur = 0.
# =================================================================================

# ============================ SparseNoise ========================================
sn_subregion = (-40., 60., 30., 90.)
sn_grid_space = (20., 20.)
sn_probe_size = (20., 10.)
sn_probe_orientation = 30.
sn_probe_frame_num = 15
sn_sign = 'ON-OFF'
sn_iteration = 2
sn_is_include_edge = True
# =================================================================================

# ============================ LocallySparseNoise =================================
lsn_subregion = (-10., 20., 0., 60.)
lsn_min_distance = 40.
lsn_grid_space = (10., 10.)
lsn_probe_size = (10., 10.)
lsn_probe_orientation = 0.
lsn_probe_frame_num = 4
lsn_sign = 'OFF'
lsn_iteration = 2
lsn_repeat = 3
lsn_is_include_edge = True
# =================================================================================

# ============================ DriftingGratingCircle ==============================
dgc_center = (10., 90.)
dgc_sf_list = (0.01, 0.16)
dgc_tf_list = (2., 8.,)
dgc_dire_list = np.arange(0., 360., 180.)
dgc_con_list = (0.8,)
dgc_radius_list = (30.,)
dgc_block_dur = 1.
dgc_midgap_dur = 1.
dgc_iteration = 2
dgc_is_smooth_edge = True
dgc_smooth_width_ratio = 0.2
dgc_smooth_func = stim.blur_cos
dgc_is_blank_block = True
dgc_is_random_start_phase = False
# =================================================================================

# ============================ StaticGratingCirlce ================================
sgc_center = (0., 40.)
sgc_sf_list = (0.08, 0.16)
sgc_ori_list = (0., 90.)
sgc_con_list = (0.5,)
sgc_radius_list = (25.,)
sgc_phase_list = (0., 90., 180., 270.)
sgc_display_dur = 0.25
sgc_midgap_dur = 0.
sgc_iteration = 10
sgc_is_smooth_edge = True
sgc_smooth_width_ratio = 0.2
sgc_smooth_func = stim.blur_cos
sgc_is_blank_block = True
# =================================================================================

# ============================ StaticImages =======================================
si_img_center = (0., 60.)
si_deg_per_pixel = (0.5, 0.5)
si_display_dur = 0.25
si_midgap_dur = 0.
si_iteration = 10
si_is_blank_block = True
si_images_folder = os.path.join(os.path.dirname(rm.__file__), 'test', 'test_data')
# =================================================================================

# ============================ StimulusSeparator ==================================
ss_indicator_on_frame_num = 4
ss_indicator_off_frame_num = 4
ss_cycle_num = 10
# =================================================================================

# ============================ CombinedStimuli ====================================
cs_stim_ind_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# =================================================================================



# ================ Initialize the monitor object ==================================
mon = Monitor(resolution=mon_resolution, dis=mon_dis, mon_width_cm=mon_width_cm,
              mon_height_cm=mon_height_cm, C2T_cm=mon_C2T_cm, C2A_cm=mon_C2A_cm,
              center_coordinates=mon_center_coordinates,
              downsample_rate=mon_downsample_rate)
# mon.plot_map()
# plt.show()
# =================================================================================

# ================ Initialize the indicator object ================================
ind = Indicator(mon, width_cm=ind_width_cm, height_cm=ind_height_cm,
                position=ind_position, is_sync=ind_is_sync, freq=ind_freq)
# =================================================================================

# ================ Initialize the DisplaySequence object ==========================
ds = DisplaySequence(log_dir=ds_log_dir, backupdir=ds_backupdir,
                     identifier=ds_identifier, display_iter=ds_display_iter,
                     mouse_id=ds_mouse_id, user_id=ds_user_id,
                     psychopy_mon=ds_psychopy_mon, is_by_index=ds_is_by_index,
                     is_interpolate=ds_is_interpolate, is_triggered=ds_is_triggered,
                     trigger_event=ds_trigger_event, trigger_NI_dev=ds_trigger_NI_dev,
                     trigger_NI_port=ds_trigger_NI_port, trigger_NI_line=ds_trigger_NI_line,
                     is_sync_pulse=ds_is_sync_pulse, sync_pulse_NI_dev=ds_sync_pulse_NI_dev,
                     sync_pulse_NI_port=ds_sync_pulse_NI_port,
                     sync_pulse_NI_line=ds_sync_pulse_NI_line,
                     display_screen=ds_display_screen, is_save_sequence=ds_is_save_sequence,
                     initial_background_color=ds_initial_background_color,
                     color_weights=ds_color_weights)
# =================================================================================

# ========================== Uniform Contrast =====================================
uc = stim.UniformContrast(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                          postgap_dur=postgap_dur, coordinate=coordinate,
                          background=background, duration=uc_duration,
                          color=uc_color)
# =================================================================================

# ======================= Flashing Circle =========================================
fc = stim.FlashingCircle(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                         postgap_dur=postgap_dur, coordinate=coordinate,
                         background=background, center=fc_center, radius=fc_radius,
                         color=fc_color, flash_frame_num=fc_flash_frame_num,
                         midgap_dur=fc_midgap_dur, iteration=fc_iteration,
                         is_smooth_edge=fc_is_smooth_edge,
                         smooth_width_ratio=fc_smooth_width_ratio,
                         smooth_func=fc_smooth_func)
# =================================================================================

# ============================ SinusoidalLuminance ================================
sl = stim.SinusoidalLuminance(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                              postgap_dur=postgap_dur, coordinate=coordinate,
                              background=background, max_level=sl_max_level,
                              min_level=sl_min_level, frequency=sl_frequency,
                              cycle_num=sl_cycle_num, start_phase=sl_start_phase,
                              midgap_dur=sl_midgap_dur)
# =================================================================================

# ======================== Sparse Noise ===========================================
sn = stim.SparseNoise(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                      postgap_dur=postgap_dur, coordinate=coordinate,
                      background=background, subregion=sn_subregion,
                      grid_space=sn_grid_space, sign=sn_sign,
                      probe_size=sn_probe_size, probe_orientation=sn_probe_orientation,
                      probe_frame_num=sn_probe_frame_num, iteration=sn_iteration,
                      is_include_edge=sn_is_include_edge)
# =================================================================================

# ======================= Locally Sparse Noise ====================================
lsn = stim.LocallySparseNoise(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                              postgap_dur=postgap_dur, coordinate=coordinate,
                              background=background, subregion=lsn_subregion,
                              grid_space=lsn_grid_space, sign=lsn_sign,
                              probe_size=lsn_probe_size, probe_orientation=lsn_probe_orientation,
                              probe_frame_num=lsn_probe_frame_num, iteration=lsn_iteration,
                              is_include_edge=lsn_is_include_edge, min_distance=lsn_min_distance,
                              repeat=lsn_repeat)
# =================================================================================

# ======================= Drifting Grating Circle =================================
dgc = stim.DriftingGratingCircle(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                                 postgap_dur=postgap_dur, coordinate=coordinate,
                                 background=background, center=dgc_center,
                                 sf_list=dgc_sf_list, tf_list=dgc_tf_list,
                                 dire_list=dgc_dire_list, con_list=dgc_con_list,
                                 radius_list=dgc_radius_list, block_dur=dgc_block_dur,
                                 midgap_dur=dgc_midgap_dur, iteration=dgc_iteration,
                                 is_smooth_edge=dgc_is_smooth_edge,
                                 smooth_width_ratio=dgc_smooth_width_ratio,
                                 smooth_func=dgc_smooth_func, is_blank_block=dgc_is_blank_block,
                                 is_random_start_phase=dgc_is_random_start_phase)
# =================================================================================

# ======================= Static Grating Cricle ===================================
sgc = stim.StaticGratingCircle(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                               postgap_dur=postgap_dur, coordinate=coordinate,
                               background=background, center=sgc_center,
                               sf_list=sgc_sf_list, ori_list=sgc_ori_list,
                               con_list=sgc_con_list, radius_list=sgc_radius_list,
                               phase_list=sgc_phase_list, display_dur=sgc_display_dur,
                               midgap_dur=sgc_midgap_dur, iteration=sgc_iteration,
                               is_smooth_edge=sgc_is_smooth_edge,
                               smooth_width_ratio=sgc_smooth_width_ratio,
                               smooth_func=sgc_smooth_func, is_blank_block=sgc_is_blank_block)
# =================================================================================

# =============================== Static Images ===================================
si = stim.StaticImages(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                       postgap_dur=postgap_dur, coordinate=coordinate,
                       background=background, img_center=si_img_center,
                       deg_per_pixel=si_deg_per_pixel, display_dur=si_display_dur,
                       midgap_dur=si_midgap_dur, iteration=si_iteration,
                       is_blank_block=si_is_blank_block)
# =================================================================================

# ============================ wrape images =======================================
print ('wrapping images ...')
static_images_path = os.path.join(si_images_folder, 'wrapped_images_for_display.hdf5')
if os.path.isfile(static_images_path):
    os.remove(static_images_path)
si.wrap_images(si_images_folder)
# =================================================================================

# ======================= Stimulus Separator ======================================
ss = stim.StimulusSeparator(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                            postgap_dur=postgap_dur, coordinate=coordinate,
                            background=background,
                            indicator_on_frame_num=ss_indicator_on_frame_num,
                            indicator_off_frame_num=ss_indicator_off_frame_num,
                            cycle_num=ss_cycle_num)
# =================================================================================

# ======================= Combined Stimuli ========================================
cs = stim.CombinedStimuli(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                          postgap_dur=postgap_dur, coordinate=coordinate,
                          background=background)
# =================================================================================

# ======================= Set Stimuli Sequence ====================================
all_stim = [uc, fc, sl, sn, lsn, dgc, sgc, si, ss]
stim_seq = [all_stim[stim_ind] for stim_ind in cs_stim_ind_sequence]
cs.set_stimuli(stimuli=stim_seq, static_images_path=static_images_path)
# =================================================================================

# =============================== display =========================================
ds.set_stim(cs)
log_path, log_dict = ds.trigger_display()
# =============================== display =========================================

#%%
# =============================== convert log to .nwb =============================
import os
import WarpedVisualStim.DisplayLogAnalysis as dla
# import NeuroAnalysisTools.NwbTools as nt
log_folder, log_fn = os.path.split(log_path)
log_nwb_path = os.path.splitext(log_path)[0] + '.nwb'
# save_f = nt.RecordedFile(filename=log_nwb_path, identifier=os.path.splitext(log_fn)[0], description='')
stim_log = dla.DisplayLogAnalyzer(log_path)
# save_f.add_visual_display_log_retinotopic_mapping(stim_log=stim_log)
# save_f.close()
# =============================== convert log to .nwb =============================

# =============================== show plot========================================
plt.show()
# =================================================================================
