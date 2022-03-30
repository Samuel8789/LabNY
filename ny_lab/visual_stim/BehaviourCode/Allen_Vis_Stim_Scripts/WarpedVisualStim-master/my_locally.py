# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:02:02 2021

@author: sp3660
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:49:52 2021

@author: sp3660
"""



import os
import numpy as np
import matplotlib.pyplot as plt
import math


script_folder=r'C:\Users\sp3660\Documents\GitHub\Allen_Vis_Stim\WarpedVisualStim-master'
    
os.chdir(script_folder)
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
mon_downsample_rate = 10
# =================================================================================

# ============================ indicator setup ====================================
ind_width_cm = 1.
ind_height_cm = 1.
ind_position = 'northeast'
ind_is_sync = True
ind_freq = 1.
# =================================================================================

# ============================ DisplaySequence ====================================
ds_log_dir = r'C:/Users/sp3660/Documents/GitHub/Allen_Vis_Stim/WarpedVisualStim-master'
# ds_log_dir = '/home/zhuangjun1981'
ds_backupdir = None
ds_identifier = 'TEST1'
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
ds_color_weights = (0., 0., 1.)
# =================================================================================

# ============================ generic stimulus parameters ========================
pregap_dur = .5
postgap_dur = .5
background = 0.
coordinate = 'degree'
# =================================================================================

# ============================ UniformContrast ====================================
uc_duration = 1.
uc_color = 0
# =================================================================================
# ============================ LocallySparseNoise =================================
lsn_subregion = None
lsn_min_distance = 46.5
lsn_grid_space = (1., 1.)
lsn_probe_size = (4.65, 4.65)
lsn_probe_orientation = 0.
lsn_probe_frame_num = 4
lsn_sign = 'ON-OFF'
lsn_iteration = 2
lsn_repeat = 3
lsn_is_include_edge = True
# =================================================================================
# ============================ StimulusSeparator ==================================
ss_indicator_on_frame_num = 4
ss_indicator_off_frame_num = 4
ss_cycle_num = 10
# =================================================================================

# ============================ CombinedStimuli ====================================
cs_stim_ind_sequence = [0, 1]
# cs_stim_ind_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8]

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
all_stim = [uc, lsn]
# all_stim = [uc, fc, sl, sn, lsn, dgc, sgc, si, ss]
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