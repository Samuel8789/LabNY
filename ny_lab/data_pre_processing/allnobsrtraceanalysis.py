# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:11:58 2022

@author: sp3660
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile
try:
    if __IPYTHON__:
        # this is used for debugging purposes only.
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass
import mat73
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
import time
import matlab.engine
import scipy.io as spio
import glob
import keyboard
from scipy.signal import convolve2d
import copy
import matplotlib as mpl
import tkinter as tk
from tkinter import END, Label, RAISED, Text, WORD, StringVar, Button, ttk, Listbox, Scrollbar
from tkinter import ttk
import  matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import shutil
import sys
from pprint import pprint

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b",'g','y','c','m', 'tab:brown']) 

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

mat_results_path=r'C:\Users\sp3660\Documents\Projects\AllenBrainObservatory\corrected_traces.mat'
data = spio.loadmat(mat_results_path)
corrected_traces=data['corrected_traces']
dff_traces=data['dff_traces']


pixel_per_bar = 4
dpi = 100
# fig = plt.figure(figsize=(6+(200*pixel_per_bar/dpi), 10), dpi=dpi)
fig = plt.figure(figsize=(16,9), dpi=dpi)

ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
# ax.set_axis_off()
ax.imshow(dff_traces, cmap='binary', aspect='auto',
    interpolation='nearest')
ax.set_xlabel('Time (s)')
fig.supylabel('Cell Number')