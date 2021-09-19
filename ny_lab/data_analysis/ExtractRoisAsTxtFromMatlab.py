# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:12:25 2021

@author: sp3660
"""





import h5py
import glob
import os
import matplotlib.pyplot as plt
from scipy import signal
import caiman as cm
dataset_path='\\\?\\'+r'D:\Projects\LabNY\Full_Mice_Data\Mice_Projects\Interneuron_Imaging\G2C\Ai14\SPJA\imaging\20210720\data aquisitions\FOV_1\210720_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\planes\Plane1\Green'
moviedatapath=glob.glob(dataset_path+'\**.mmap')[0]
file_path=glob.glob(dataset_path+'\**.mat')[0]
components_path=os.path.join(dataset_path,'SpatialComponents')
if not os.path.isdir(components_path):
    os.mkdir(components_path)


f=h5py.File(file_path, 'r')
test=f['est']['contours']
check=test[0]
acell=check[0]
conts=f[acell]
coordinates=list(zip(conts[0],conts[1]))
coordinttolist=[','.join(map(str,map(int, coord)))  for coord in coordinates]

for i, acell in enumerate(check):
    conts=f[acell]
    coordinates=list(zip(conts[0],conts[1]))
    coordinttolist=[','.join(map(str,map(int, coord)))  for coord in coordinates]
    with open(os.path.join(components_path,'Cell{}.txt'.format(str(i+1).zfill(4))), 'w') as output:
        for row in coordinttolist:
            output.write(str(row) + '\n')
        
        
        


