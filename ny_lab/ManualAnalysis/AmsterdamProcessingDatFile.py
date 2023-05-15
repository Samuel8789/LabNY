# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:48:05 2023

@author: sp3660
"""

import os
import glob
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import caiman as cm
movpath=r'C:\Users\sp3660\Documents\2minmovietest'


imagedatafile=glob.glob(movpath+'\**.dat')[0]
frametimesdatafile=glob.glob(movpath+'\**frameTimes**.mat')[0]
apphandlesfile=glob.glob(movpath+'\**handles**.mat')[0]

imaginginfo=spio.loadmat(frametimesdatafile)
times=imaginginfo['frameTimes'].squeeze()
frametime=np.diff(times)*1e5
timestamps=times-times[0]
timestamps=timestamps*1e5
plt.plot(frametime)
plt.plot(timestamps)


recordingsize=imaginginfo['imgSize'].squeeze()
preStimFrames=imaginginfo['preStim'].squeeze()
postStimFrames=imaginginfo['postStim'].squeeze()



video=np.fromfile(imagedatafile, dtype=np.uint16)
video2=np.transpose(np.reshape(video,np.flip(recordingsize)), (3, 2, 1, 0))
movie=cm.movie(video2)
plt.imshow(movie[:,:,0,0])

#BOX FILTER
tets=np.mean(np.reshape(movie,[4,(movie.shape[0]*movie.shape[1])//4,movie.shape[2],movie.shape[3]],order="F"),axis=0)
tets2=np.reshape(tets,[movie.shape[0]//4,movie.shape[1],movie.shape[2],movie.shape[3]],order="F")
plt.imshow(tets2[:,:,0,0])
test3=np.transpose(tets2, (1, 0, 2, 3))

tets4=np.mean(np.reshape(test3,[4,(test3.shape[0]*test3.shape[1])//4,test3.shape[2],test3.shape[3]],order="F"),axis=0)
tets5=np.reshape(tets4,[test3.shape[0]//4,test3.shape[1],test3.shape[2],test3.shape[3]],order="F")
test6=np.transpose(tets5, (1, 0, 2, 3))
plt.imshow(test6[:,:,0,0])

