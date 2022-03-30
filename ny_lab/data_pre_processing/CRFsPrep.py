# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:56:22 2022

@author: sp3660
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# import h5py
# import scipy.io
import mat73
import numpy as np
import copy
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, RocCurveDisplay,  PrecisionRecallDisplay, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import pandas as pd

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b",'g','y','c','m', 'tab:brown']) 

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

class CRFsPrep():
    
    def __init__(self, analysis_object=None):
        pass
        print('Processing CRFResults')
