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

acqdir=r'K:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Chandelier_Imaging\VRC\SLF\Ai65\SPRM\imaging\20230820\data aquisitions\FOV_1\230820_SPRM_FOV1_1z_30min_ShortDrift_Cell1_opto_1.2_25x_920_51020_63075_with-000\planes\Plane1\Red'
# mmapfilepath=Path(glob.glob(acqdir+'\**ACID**.mmap')[0])
mmapfilepath=Path(glob.glob(acqdir+'\**.mmap')[0])

desktop=r'C:\Users\sp3660\Desktop\MoviesToCheck'


tiffilename=mmapfilepath.stem + acqdir[acqdir.find('planes\Plane')+14:]+'.tiff'


os.startfile(desktop)


mov=cm.load(mmapfilepath)

mov2=mov.astype(np.uint16)

mov2.save(Path(desktop, tiffilename),compress=1, to32=False)


# downmov=block_reduce(mov,block_size=(2,1,1),func=np.mean)


# mov.st