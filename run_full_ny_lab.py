# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 17:57:08 2021

@author: sp3660
"""
import os
import sys
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/ny_lab')
sys.path.insert(0, os.path.join(os.path.expanduser('~'),r'Documents/Github/LabNY'))
sys.path.insert(0, os.path.join(os.path.expanduser('~'),r'Documents/Github/LabNY/ny_lab'))
import ny_lab
gui=1
lab=ny_lab.RunNYLab(gui)



