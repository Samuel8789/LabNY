# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:10:26 2021

@author: sp3660
"""




import os
import numpy as np
import matplotlib.pyplot as plt
import math

import sys
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/AllFunctions')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/ProcessingScripts')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/MainClasses')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/MainClasses/Visual Stimulation/Allen_Vis_Stim/WarpedVisualStim-master')
    
import WarpedVisualStim as rm
import WarpedVisualStim.StimulusRoutines as stim
from WarpedVisualStim.MonitorSetup import Monitor, Indicator
from WarpedVisualStim.DisplayStimulus import DisplaySequence


#%%
monitor='Desktop'
monitor='Prairie1'



#%%
if monitor=='Desktop':
# ============================ monitor setup ======================================
    mon_resolution = (1440, 2560)  # enter your monitors resolution
    mon_width_cm = 60.  # enter your monitors width in cm
    mon_height_cm = 33.6  # enter your monitors height in cm
    mon_refresh_rate = 60  # enter your monitors height in Hz
    mon_C2T_cm = mon_height_cm / 2.
    mon_C2A_cm = mon_width_cm / 2.
    mon_center_coordinates = (0., 60.)
    mon_dis = 15.
    mon_downsample_rate = 10
    ds_display_screen = 0

if monitor=='Prairie1':
    mon_resolution = (1024, 1280)  # enter your monitors resolution
    mon_width_cm = 38.  # enter your monitors width in cm
    mon_height_cm = 30  # enter your monitors height in cm
    mon_refresh_rate = 60  # enter your monitors height in Hz
    mon_C2T_cm = mon_height_cm / 2.
    mon_C2A_cm = mon_width_cm / 2.
    mon_center_coordinates = (0., 60.)
    mon_dis = 15.
    mon_downsample_rate = 10
    ds_display_screen = 1
    
    
    
    
    
# =================================================================================

# ============================ indicator setup ====================================
ind_width_cm = 1.
ind_height_cm = 1.
ind_position = 'northeast'
ind_is_sync = True
ind_freq = 1.
# =================================================================================

# ============================ DisplaySequence ====================================
ds_log_dir = r'C:/Users/sp3660/Documents/Github/LabNY/MainClasses/Visual Stimulation/Allen_Vis_Stim/WarpedVisualStim-master'
# ds_log_dir = '/home/zhuangjun1981'
ds_backupdir = None
ds_identifier = 'TEST'
ds_display_iter = 1
ds_mouse_id = 'MOUSE'
ds_user_id = 'USER'
ds_psychopy_mon = monitor
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
ds_color_weights = (0., 0., 1.)
# =================================================================================

# ============================ generic stimulus parameters ========================
pregap_dur = 1.5
postgap_dur = 1.5
background = 0.
coordinate = 'degree'
# =================================================================================

# ============================ UniformContrast ====================================
uc_duration = 600.
uc_color = 0
# =================================================================================
# ============================ DriftingGratingCircle ==============================
dgc_center = (0., 60.)
dgc_sf_list = (0.04,)
dgc_tf_list = (1., 2., 4., 8., 15.,)
dgc_dire_list = np.arange(0., 360., 45.)
dgc_con_list = (0.8,)
dgc_radius_list = (65.,)
dgc_block_dur = 2.
dgc_midgap_dur = 1.
dgc_iteration = 15
dgc_is_smooth_edge = True
dgc_smooth_width_ratio = 0.2
dgc_smooth_func = stim.blur_cos
dgc_is_blank_block = True
dgc_is_random_start_phase = False
# =================================================================================
# ============================ StimulusSeparator ==================================
ss_indicator_on_frame_num = 4
ss_indicator_off_frame_num = 4
ss_cycle_num = 10
# =================================================================================

# ============================ CombinedStimuli ====================================
cs_stim_ind_sequence = [0, 1]
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
all_stim = [uc, dgc]
stim_seq = [all_stim[stim_ind] for stim_ind in cs_stim_ind_sequence]
cs.set_stimuli(stimuli=stim_seq)
# =================================================================================

# =============================== display =========================================
ds.set_stim(cs)
log_path, log_dict = ds.trigger_display()

# =============================== convert log to .nwb =============================
import os
import WarpedVisualStim.DisplayLogAnalysis as dla
import NeuroAnalysisTools.NwbTools as nt
log_folder, log_fn = os.path.split(log_path)
log_nwb_path = os.path.splitext(log_path)[0] + '.nwb'
save_f = nt.RecordedFile(filename=log_nwb_path, identifier=os.path.splitext(log_fn)[0], description='')
stim_log = dla.DisplayLogAnalyzer(log_path)
save_f.add_visual_display_log_retinotopic_mapping(stim_log=stim_log)
save_f.close()
# =============================== convert log to .nwb =============================

# =============================== show plot========================================
plt.show()
# =================================================================================