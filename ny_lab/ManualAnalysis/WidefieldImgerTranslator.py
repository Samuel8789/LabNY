# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:36:40 2023

@author: sp3660
"""
"""

Parameters during the recording in Matlab: Baseline 2s, FrameRate 30Hz, Binning 2. 

Parameters from one trial: 3s ITI => 3s Stim period => 3s ITI as one recording
100 repetitions in one recording.

"""
import os
import numpy as np
import glob
import scipy.io as spio
import pathlib
import numpy as np
from PIL import Image, ImageSequence
from scipy.stats import zscore
import tifffile


#%%

def loadRawData(cFile, condition, dataType, imgSize):
    
    try:
        dataType
    except NameError:
        dataType = 'uint16'
        
    data = []
    header = []
    fileType = pathlib.Path(cFile).suffix #check filetype.
    
    
    if fileType== '.tif':
       im = tifffile.imread(cFile, maxworkers=6)
       height = im.shape[1]
       width = im.shape[2]

       data=np.moveaxis(im,0,2)

        
        
    elif fileType in ['.mj2','.mp4']:
        pass
    else:
        if condition=='analog':

            with open(cFile, 'rb') as f:
                # Read the header size
                hSize = np.frombuffer(f.read(8), dtype=np.float64)[0]
                # Read the header
                header = list(np.frombuffer(f.read(int(hSize) * 8), dtype=np.float64))
                siz=len(f.read())
                # Update the last element of the header with the size of the data
                header[-1] = (siz/10 )
                # Move the file pointer back to the beginning of the data
                f.seek(int(hSize) * 8 + 8)
                # Read the data
                data = np.frombuffer(f.read(), dtype=np.uint16).reshape((int(header[-1]), int(header[-2]))).T

        elif condition=='frames':
            pass
    

    
    
    return [header,data]


def splitChannels(opts_dict,trialNr, fileType):
    """
     Code to separate blue and violet channel from widefield data. This needs
     analog data that contains a stimulus onset from which a pre and
     poststimulus dataset can be returned. Also requires trigger channels for
     blue/violet LEDs and some dark frames at the end.
     This code is the newer version of Widefield_CheckChannels.
    """

    try:
        fileType
    except NameError:
        fileType = '.dat'
        
    falseAlign = False
    
    cFile=os.path.join(opts_dict['fPath'],'Analog_' + str(trialNr+1) +'.dat' )
    [_,analog] = loadRawData(cFile,'analog') #load analog data
    analog = analog.astype(np.float64);
    
    
    
    
    meta=spio.loadmat(os.path.join(opts_dict['fPath'], f'frameTimes_{str(trialNr+1).zfill(4)}.mat'))
    
    frameTimes = meta['frameTimes'].flatten() * 86400*1e3 #convert to seconds
    imgSize=meta['imgSize']
    cFile = os.path.join(opts_dict['fPath'],  opts_dict['fName']+ '_' + str(trialNr+1).zfill(4)+ fileType) #current file to be read
    
    [_, data] = loadRawData(cFile,'Frames',[], imgSize) #load video data
    
    
    #reshape data to compute mean frame intensities
    dSize = data.shape
    temp = zscore( np.mean(data, axis=(0,1)))


    bFrame = np.argwhere(temp < min(temp)*.75).flatten() #index for black frames
    
    if bFrame(0) == 0: #if first frame is dark, remove initial frames from data until LEDs are on
        # #remove initial dark frames
        # cIdx = find(diff(bFrame) > 1, 1); 
        # data(:,:,1:cIdx) = [];
        # dSize = size(data);
        # temp(1:cIdx) = [];
        # frameTimes(1:cIdx) = [];
        pass
    
    
    #determine imaging rate - either given as input or determined from data
    if 'frameRate' in opts_dict.keys():
        sRate = opts_dict['frameRate']
    else:
        sRate = 1000/(np.mean(np.diff(frameTimes))*2)
        
        
    # check if pre- and poststim are given. use all frames if not.
    if  ('preStim' not in opts_dict.keys()) or ('postStim' not in opts_dict.keys()):      
        opts_dict['preStim'] = 0;
        opts_dict['postStim'] = np.inf
    else:
       opts_dict['preStim'] = np.ceil(opts_dict['preStim']  * sRate);
       opts_dict['postStim'] = np.ceil( opts_dict['postStim'] * sRate);
       
       
    

    

    

    return 
# [blueData,blueTimes,hemoData,hemoTimes,stimOn,falseAlign,sRate]




#%%

# function to return the file extension

dataPath=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\28-May-2023_6'

opts_dict={}
opts_dict['fPath']=dataPath #path to imaging data
opts_dict['fName']='Frames_2_487_569_uint16' #name of imaging data files.
opts_dict['stimLine']=3 #analog line that contains stimulus trigger.
opts_dict['trigLine']=[1,2] #analog lines for blue and violet light triggers.
opts_dict['preStim']=2 #pre-stimulus duration in seconds
opts_dict['postStim']=3 #post-stimulus duration in seconds
opts_dict['plotChans']=True #flag to show separate channels when loading dual-wavelength data in each trial
opts_dict['sRate']=15 #sampling rate in Hz
opts_dict['downSample']=4 #spatial downsampling factor
opts_dict['hemoCorrect']=False #hemodynamic correction is optional (this only works with dual-color data in raw datasets).
opts_dict['fileExt']='.tif' #type of video file. Use '.dat' for binary files (also works for .tif or .mj2 files)
opts_dict['preProc']=False #case if data is single channel and can be loaded directly (this is only true for the pre-processed example dataset).


rawFiles = glob.glob(os.path.join(opts_dict['fPath'],opts_dict['fName'])+'**' ) #find data files
imgSize=spio.loadmat(os.path.join(opts_dict['fPath'], 'frameTimes_0001.mat'))['imgSize'][0] #get size of imaging data.
dataSize = np.floor(imgSize / opts_dict['downSample']).astype('uint16') #adjust for downsampling

stimOn = int((opts_dict['preStim']*opts_dict['sRate'])) #frames before stimulus onset
baselineDur = np.arange(0, np.min([opts_dict['sRate'], stimOn])) #use the first second or time before stimulus as baseline

nrFrames = (opts_dict['preStim'] + opts_dict['postStim']) * opts_dict['sRate'] #frames per trial
nrTrials = len(rawFiles)#nr of trials
nrTrials = 10#nr of trials

allData = np.full([dataSize[0],dataSize[1],nrFrames, nrTrials], np.nan)

for trialNr in range(0,nrTrials):
    
    a=pathlib.Path(rawFiles[trialNr]).suffix #get data type (should be .dat or .tif for raw. also works with .mj2 for compressed data)
    
    if not opts_dict['preProc']: #no preprocessing. assuming two wavelengths for expsure light (blue and violet).
        [bData,_,vData] = splitChannels(opts_dict,trialNr,a) #split channels and get blue and violet data

    
        pass
    
    elif opts_dict['preProc']: #pre-processed data. simply load all available data and skip motion correction ect.
    
    
        pass
    else:
        print('Could not read number of channels from filename or channelnumber is >2.') #for this to work filenames should contain the channelnumber after a _ delimiter

    
    
    pass


#%% simp;lified