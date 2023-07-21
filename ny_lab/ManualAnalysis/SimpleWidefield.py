# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 09:29:17 2023

@author: sp3660
"""

import os
import numpy as np
import glob
import scipy.io as spio
import pathlib as pth
from scipy.stats import zscore
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import caiman as cm
from tifffile import tifffile

#%% SET UP YOUR OWN PATH To DATA
opts_dict={}
dirpath=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam'
foldername='29-May-2023_13'
dataPath=os.path.join(dirpath,foldername)
opts_dict['fName']=pth.Path(glob.glob(dataPath+'\**.tif')[0]).stem[0:-5]
try:
    os.remove(pth.Path(dataPath, opts_dict['fName']+'_processed.tiff'))
except:
    pass

#%% SOME SETTINGS


opts_dict['fPath']=dataPath #path to imaging data
opts_dict['stimLine']=3 #analog line that contains stimulus trigger.(RIght now this is not working)
opts_dict['trigLine']=[2,3] #analog lines for blue and violet light triggers.
opts_dict['preStim']=2 #pre-stimulus duration in seconds
opts_dict['postStim']=3 #post-stimulus duration in seconds
opts_dict['fRate']=30 #sampling rate in Hz



#%%
opts_dict['sRate']=opts_dict['fRate']/2 #sampling rate in Hz

rawFiles = glob.glob(os.path.join(opts_dict['fPath'],opts_dict['fName'])+'**' ) #find data files
imgSize=spio.loadmat(os.path.join(opts_dict['fPath'], 'frameTimes_0001.mat'))['imgSize'][0] #get size of imaging data.
stimOn = int((opts_dict['preStim']*opts_dict['sRate'])) #frames before stimulus onset
baselineDur = np.arange(0,  stimOn) #use the first second or time before stimulus as baseline
nrFrames = int((opts_dict['preStim'] + opts_dict['postStim']) * opts_dict['sRate']) #frames per trial
nrTrials = len(rawFiles)-1#nr of trials remove las trial as is the closing one if not properly rtirgerred

alldata= np.full([imgSize[0],imgSize[1],nrFrames, nrTrials], np.nan)
for trialNr in range(nrTrials):
    #%% load movie
    meta=spio.loadmat(os.path.join(opts_dict['fPath'], f'frameTimes_{str(trialNr+1).zfill(4)}.mat'))
    frameTimes = meta['frameTimes'].flatten() * 86400*1e3 #convert to seconds
    imgSize=meta['imgSize']
    vFile = os.path.join(opts_dict['fPath'],  opts_dict['fName']+ '_' + str(trialNr+1).zfill(4)+ '.tif') #current file to be read
    im = tifffile.imread(vFile, maxworkers=6)
    height = im.shape[1]
    width = im.shape[2]
    data=np.moveaxis(im,0,2)
    
    #%% load signals
    aFile=os.path.join(opts_dict['fPath'],'Analog_' + str(trialNr+1) +'.dat' )
    with open(aFile, 'rb') as f:
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
        analog = np.frombuffer(f.read(), dtype=np.uint16).reshape((int(header[-1]), int(header[-2]))).T
    analog = analog.astype(np.float64)
    
    
    #%% separate movie channels and trim to trial

    dSize = data.shape
    temp = zscore(data.mean(axis=(0,1)))
    
    bFrame = np.where(temp < min(temp)*.75)[0] #index for black frames
    
    bFrame=bFrame[bFrame > np.round(temp.shape[0]*(2/3))]
    
    
    
    bFrameidx = np.argwhere(temp < min(temp)*.75)[0][0] #index for black frames
    opts_dict['preStimfr'] = int(np.ceil(opts_dict['preStim'] * opts_dict['sRate']))
    opts_dict['postStimfr']= int(np.ceil(opts_dict['postStim'] * opts_dict['sRate']))
    
        
    trace =analog[opts_dict['trigLine'],:] #blue and violet light trigger channels  
    trace = zscore(trace[0,-1::-1] -trace[1,-1::-1]) #invert and subtract to check color of last frame    
    trace[np.round(np.diff(trace,append=trace[-1])) != 0] = 0 #don't use triggers that are only 1ms long
    
    lastBlue = np.argwhere(trace > 1)[1]
    lastHemo = np.argwhere(trace < -1)[1]
    blueLast = lastBlue < lastHemo
    blueInd = np.full(len(temp),False)
    
    if blueLast: #last frame before black is blue
        if bFrame[0]%2 == 0: #blue frames (bFrame - 1) have uneven numbers
            blueInd[1:dSize[-1]:2] = True
        else:
            blueInd[0:dSize[-1]:2] = True
        lastFrame = analog.shape[1] - lastBlue #index for end of last frame
        
    else: #last frame before black is violet
        if bFrame[0]%2 == 0: #blue frames (bFrame - 1) have uneven numbers
            blueInd[0:dSize[-1]:2] = True
        else:
            blueInd[1:dSize[-1]:2] = True
        lastFrame = analog.shape[1] - lastHemo #index for end of last frame
    
    
    hemoInd=np.invert(blueInd)
    
    #realign frameTime based on time of last non-dark frame(this is optional)
    frameTimes = (frameTimes - frameTimes[bFrame[0] - 1]) + lastFrame
    blueInd[bFrame[0] - 1:]=False  #exclude black and last non-black frame
    hemoInd[bFrame[0] - 1:]=False  #exclude black and last non-black frame
    
    
    
    blueTimes = frameTimes[blueInd]
    hemoTimes = frameTimes[hemoInd]
    blueData = data[:,:,blueInd]
    hemoData = data[:,:,hemoInd]
       
    blueData = blueData[:,:,:opts_dict['preStimfr'] + opts_dict['postStimfr']]
    hemoData = hemoData[:,:,:opts_dict['preStimfr'] + opts_dict['postStimfr']]
    blueTimes= frameTimes[1:opts_dict['preStimfr'] + opts_dict['postStimfr']:2]
    hemoTimes = frameTimes[1:opts_dict['preStimfr'] + opts_dict['postStimfr']:2]
    
    tt=blueData[:,:,0]
    ttt=hemoData[:,:,0]


    #%% hemo correct(smooth hemo data )
    smoothFact=5
    baselineAvg = np.expand_dims(blueData[:,:,baselineDur].mean(axis=2),2)
    dffdata=(blueData-baselineAvg)/baselineAvg
    tt=dffdata[:,:,0]
    baselinehemoAvg = np.expand_dims(hemoData[:,:,baselineDur].mean(axis=2),2)
    dffhemodata=(hemoData-baselinehemoAvg)/baselinehemoAvg
    
    dffdatavector=np.reshape(dffdata,[-1,dffdata.shape[-1]],order='F')
    dffhemovector=np.reshape(dffhemodata,[-1,dffhemodata.shape[-1]],order='F')
    

    kernel=np.ones((1,smoothFact))/smoothFact
    cbegin=dffhemovector[:,0:smoothFact-2].cumsum(axis=1)
    cbegin2=cbegin[:,::2]/np.arange(1,smoothFact-1,2)
    cend=dffhemovector[:,-1:-(smoothFact-1):-1].cumsum(axis=1)
    cend2=cend[:,::-2]/np.arange(smoothFact-2,0,-2)
    
    smoothedhemo=convolve2d(dffhemovector,kernel,'full')
    clipped=smoothedhemo[:,smoothFact-1:-smoothFact+1]
    finalsmoothedhemo=np.concatenate([cbegin2,clipped,cend2],axis=1)

#%% hemo correct(linear regreession)

    m=np.zeros(dffdatavector.shape[0])
    b=np.zeros(dffdatavector.shape[0])
    
    for iPix in range(dffdatavector.shape[0]):
      theta = np.linalg.lstsq(np.array([finalsmoothedhemo[iPix,:], np.ones(dffdatavector.shape[1])]).T, np.expand_dims(dffdatavector[iPix,:],axis=1))[0]
      m[iPix] = theta[0][0]
      b[iPix] = theta[1][0]
    
    
    
    predict=finalsmoothedhemo*np.expand_dims(m,1)+np.expand_dims(b,1)
    correcteddata = dffdatavector - predict
    finaldata = np.reshape(correcteddata,blueData.shape,order='F')

    finaldataAvg = finaldata[:,:,baselineDur].mean(2)
    offsetfinal=finaldata-np.expand_dims(finaldataAvg,2)
    alldata[:,:,:,trialNr]=offsetfinal
#%% saving corrected movie

mov=cm.movie(np.moveaxis(alldata.mean(axis=3), 2, 0))
mov.save(pth.Path(dataPath, opts_dict['fName']+'_processed.tiff'),compress=1, to32=True)
    
    
    
#%% plotting


#manually set some pixels to look and replot
xleft=40
xright=128
ytop=65
ybottom=152



f,ax=plt.subplots(1,2)
avgMap = alldata[:,:,stimOn+1:,:].mean(axis=(2,3))
im=ax[0].imshow(avgMap,cmap='inferno')
f.colorbar(im,ax=ax[0],shrink=0.5)
ax[0].set_title('Stimulus-triggered activity')



meanTrace = np.mean(alldata[xleft:xright,ytop:ybottom,:,:],axis=(0,1))
men=np.mean(meanTrace,1)
sem=np.std(meanTrace, ddof=1) / np.sqrt(np.size(meanTrace))
timeTrace = (np.arange(0,nrFrames) / opts_dict['sRate']) - opts_dict['preStim']
ax[1].plot(timeTrace,men,'k')
ax[1].fill_between(timeTrace, men-sem, men+sem,color='black', alpha=0.5)
ax[1].set_title('Average change over all pixels')
ax[1].vlines(0,min(men),max(men),color='black',linestyles ='dashed')