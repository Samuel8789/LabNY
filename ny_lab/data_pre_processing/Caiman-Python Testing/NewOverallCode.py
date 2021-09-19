# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:04:59 2020

@author: sp3660
"""
"""
#Pipeline for analysis of my own movies. Start with the demom examples. Input is a tif movie



"""
#Importing and initialization

import bokeh.plotting as bpl
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import time

try:
    cv2.setNumThreads(0)
except():
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
bpl.output_notebook()
from caiman.paths import caiman_datadir
from caiman.summary_images import local_correlations_movie_offline

import sys
#%%

# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.WARNING)
    # filename="/tmp/caiman.log"

#%%
def main():
    pass # For compatibility between running under Spyder and the CLI
#%% Select Files

if os.name=='posix'
folder_name=

'/home/sp3660/Data_To_Analyze'
'/home/sp3660/Caiman_Results''



if os.name=='nt'

'C:\Users\sp3660\caiman_data\example_movies'





























#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
