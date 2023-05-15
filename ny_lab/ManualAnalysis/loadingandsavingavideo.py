# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 08:45:44 2023

@author: sp3660
"""
import caiman as cm
import os
import glob
from pathlib import Path
from skimage.measure import block_reduce
import numpy as np

acqdir=r'D:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Imaging\VRC\SLF\Ai65\SPRB\imaging\20230321\test aquisitions\230321_SPRB_3MinTest_25x_920_51020_63075_with-001\planes\Plane1\Green'
# mmapfilepath=Path(glob.glob(acqdir+'\**ACID**.mmap')[0])
mmapfilepath=Path(glob.glob(acqdir+'\**.mmap')[0])

desktop=r'C:\Users\sp3660\Desktop'


tiffilename=mmapfilepath.stem + acqdir[acqdir.find('planes\Plane')+14:]+'.tiff'





mov=cm.load(mmapfilepath)

mov2=mov.astype(np.uint16)

mov2.save(Path(desktop, tiffilename),compress=1, to32=False)


# downmov=block_reduce(mov,block_size=(2,1,1),func=np.mean)


# mov.st