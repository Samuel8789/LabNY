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
import caiman as caim
from tifffile import tifffile
from pylab import *
from numpy import *
from scipy import *
import pickle

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

#%% SET UP YOUR OWN PATH To DATA
opts_dict={}
dirpath=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Data'
opts_dict={}
firstnondelay=''

session='20230826'
foldername='26-Aug-2023_1'
nosignal=['28-May-2023_6', '29-May-2023_3','29-May-2023_13']
check_signal=False
dataPath=os.path.join(dirpath,session,foldername)



nma=pth.Path(glob.glob(dataPath+'\**.tif')[0]).stem[0:-5]
toplot=False

alldata=np.empty(0)
goodtrials=np.empty(0)


# loadexisting data
if os.path.isfile(os.path.join(dataPath,f"{foldername}.mat")):
    matcontents=loadmat(os.path.join(dataPath,f"{foldername}.mat"))
    alldata=matcontents["correcteddata"]
    goodtrials=squeeze(matcontents["goodtrials"])
    opts_dict=matcontents["metadata"]

elif os.path.isfile(os.path.join(dataPath,f"{foldername}.pkl")):
     with open(os.path.join(dataPath,f"{foldername}.pkl"), 'rb') as f:
         transitions_dictionary= pickle.load(f)
         alldata=transitions_dictionary["correcteddata"]
         goodtrials=squeeze(transitions_dictionary["goodtrials"])
         opts_dict=transitions_dictionary["metadata"]
     
#%% SOME SETTINGS

print(dataPath)
if not opts_dict:
    opts_dict['Delay']=False
    opts_dict['fName']=nma
    if foldername in nosignal:
        opts_dict['stimONsignal']=None
    else:
        opts_dict['stimONsignal']=True
    opts_dict['fPath']=dataPath #path to imaging data
    opts_dict['stimLine']=1 #analog line that contains stimulus trigger.(RIght now this is not working)
    opts_dict['trigLine']=[2,3] #analog lines for blue and violet light triggers.
    opts_dict['preStim']=2 #pre-stimulus duration in seconds
    opts_dict['postStim']=6 #post-stimulus duration in seconds
    opts_dict['fRate']=30 #sampling rate in Hz
    opts_dict['sRate']=opts_dict['fRate']/2 #sampling rate in Hz
    opts_dict['preStimfr'] = int(np.ceil(opts_dict['preStim'] * opts_dict['sRate']))
    opts_dict['postStimfr']= int(np.ceil(opts_dict['postStim'] * opts_dict['sRate']))


#%%

rawFiles = glob.glob(os.path.join(opts_dict['fPath'],opts_dict['fName'])+'**.tif' ) #find data files
imgSize=spio.loadmat(os.path.join(opts_dict['fPath'], 'frameTimes_0001.mat'))['imgSize'][0] #get size of imaging data.
stimOn = int((opts_dict['preStim']*opts_dict['sRate'])) #frames before stimulus onset
baselineDur = np.arange(0,  stimOn) #use the first second or time before stimulus as baseline
nrFrames = int((opts_dict['preStim'] + opts_dict['postStim']) * opts_dict['sRate']) #frames per trial
opts_dict['nrTrials'] = len(rawFiles)#nr of trials remove las trial as is the closing one if not properly rtirgerred
if 'nrTrials' not in opts_dict.keys():
    if abs(opts_dict['nrTrials']-100)<10:
        opts_dict['nrTrials']=100
    elif abs(opts_dict['nrTrials']-50)<10:
        opts_dict['nrTrials']=50

trialstoremove=[]
if (not alldata.any()) or (not goodtrials.any()):
    try:
        os.remove(pth.Path(dataPath, nma+'_processed.tiff'))
    except:
        pass
    
    alldata= np.full([imgSize[0],imgSize[1],nrFrames, opts_dict['nrTrials']], np.nan)
    
    for trialNr in range(opts_dict['nrTrials']):
        try:
            #%% load movie
            meta=spio.loadmat(os.path.join(opts_dict['fPath'], f'frameTimes_{str(trialNr+1).zfill(4)}.mat'))
            frameTimes = meta['frameTimes'].flatten() * 86400*1e3 #convert to seconds
            if trialNr==0:
                if 'RecordingHZ' not in opts_dict.keys():
                    opts_dict['RecordingHZ']=mean(diff(frameTimes))
                if 'RecordingMeta' not in opts_dict.keys():
                    opts_dict['RecordingMeta']=meta


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
            
            #chek if stimon signal
            if check_signal:
                plot(analog[1,:])
                show()
            
            
            
            #%%separate movie channels and trim to trial
     
            temp = zscore(data.mean(axis=(0,1)))
            darkframes= np.where(temp < min(temp)*.6)[0] #index for black frames
            missedframes=darkframes[darkframes<len(temp)-10*(opts_dict['fRate']/30)]
            endblackframes=darkframes[darkframes>len(temp)-10*opts_dict['fRate']/30]
            goodframes=temp[:endblackframes[0]]
            blueframes=goodframes>0.2
            violetframes=np.logical_and(goodframes<-0.01 , goodframes>-1)
            
            
            
            
            trace =analog[opts_dict['trigLine'],:] #blue and violet light trigger channels  
            trace = zscore(trace[0,:] -trace[1,:]) #invert and subtract to check color of last frame    
            trace[np.round(np.diff(trace,append=trace[-1])) != 0] = 0 #don't use triggers that are only 1ms long
        
            
            ttt=np.where(diff(trace)>0.75)[0]
            bluestart=np.delete(ttt,np.where(diff(ttt)<10)[0]+1)
            ttt=np.where(diff(trace)<-0.75)[0]
            viostart=np.delete(ttt,np.where(diff(ttt)<10)[0]+1)
            if  bluestart[-1]>viostart[-1]:
                bluestart=np.delete(bluestart,-1)
            else:
                viostart=np.delete(viostart,-1)
            lastBluems = bluestart[-1]
            lastHemoms = viostart[-1]
            lastframems=max(lastBluems,lastHemoms)
            ffrm = (frameTimes - frameTimes[endblackframes[0]-1])+lastframems

        
            if opts_dict['stimONsignal']:
                stimon=np.where(diff(zscore(analog[1,:]))>1)
                stimonframe=np.argmin(abs(ffrm-stimon))
                # blueframes[np.argmin(abs(ffrm-stimon))]  
            else:
                stimonframe=stimOn*2 # assume it starts at 0
                
                
            trialslice=np.arange(stimonframe-stimOn*2,stimonframe-stimOn*2+(nrFrames*2))

                
            
            if toplot:
                close('all')
                f,ax=plt.subplots(2,1)
                f.suptitle(f'Trial {trialNr+1}')
                ax[0].plot(trace)
                ax[0].plot(zscore(analog[1,:]))
                ax[0].plot(bluestart,trace[bluestart],'xg')
                ax[0].plot(ffrm,temp)
                ax[0].plot(ffrm[darkframes],temp[darkframes],'ro')
                ax[0].plot(ffrm[stimonframe],temp[stimonframe],'ro')
                ax[0].plot(ffrm[trialslice],temp[trialslice],'c')
                ax[1].plot(darkframes,temp[darkframes],'ro')
                ax[1].plot(temp)
                show()
             
        
            dataslice=data[:,:,trialslice]
            tistampslice=ffrm[trialslice]
            blueslice=blueframes[trialslice]
            violetslice=violetframes[trialslice]
            removedframes=0
            if not all( [el for el in diff(blueslice)]):
                missedframe=np.where(np.invert(diff(blueslice)))[0]
                removedframes=len(missedframe)
                if not missedframe[0]%2:
                    torem=1
                else:
                    torem=-1
        
                if  blueslice[missedframe+torem] :
            
                    blueslice[missedframe+torem]=False
                    
                if violetslice[missedframe+torem]:
                    violetslice[missedframe+torem]=False
                    
            blueData=dataslice[:,:,blueslice]
            hemoData=dataslice[:,:,violetslice]
            blueTimes=tistampslice[blueslice]
            hemoTimes=tistampslice[violetslice]
            
            if toplot:
                
                f,ax=plt.subplots(2,1)
                f.suptitle(f'Trial {trialNr+1}')
                ax[0].plot(trace)
                ax[0].plot(zscore(analog[1,:]))
                ax[0].plot(blueTimes,zscore(blueData.mean((0,1))))
                ax[0].plot(blueTimes[baselineDur],zscore(blueData.mean((0,1)))[baselineDur],'r')
            
                ax[1].plot(trace)
                ax[1].plot(zscore(analog[1,:]))    
                ax[1].plot(hemoTimes,zscore(hemoData.mean((0,1))))
                ax[1].plot(hemoTimes[baselineDur],zscore(hemoData.mean((0,1)))[baselineDur],'r')
                show()
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
              theta = np.linalg.lstsq(np.array([finalsmoothedhemo[iPix,:], np.ones(dffdatavector.shape[1])]).T, np.expand_dims(dffdatavector[iPix,:],axis=1),rcond=None)[0]
              m[iPix] = theta[0][0]
              b[iPix] = theta[1][0]
            
            
            
            predict=finalsmoothedhemo*np.expand_dims(m,1)+np.expand_dims(b,1)
            correcteddata = dffdatavector - predict
            finaldata = np.reshape(correcteddata,blueData.shape,order='F')
        
            finaldataAvg = finaldata[:,:,baselineDur].mean(2)
            offsetfinal=finaldata-np.expand_dims(finaldataAvg,2)
            alldata[:,:,:nrFrames-removedframes,trialNr]=offsetfinal
            
            
            if toplot:
        
                f,ax=plt.subplots(3,1)
                f.suptitle(f'Trial {trialNr+1}')
                ax[0].plot(trace)
                ax[0].plot(zscore(analog[1,:]))
                ax[0].plot(blueTimes,zscore(blueData.mean((0,1))))
                ax[0].plot(blueTimes[baselineDur],zscore(blueData.mean((0,1)))[baselineDur],'r')
                
                ax[1].plot(trace)
                ax[1].plot(zscore(analog[1,:]))    
                ax[1].plot(hemoTimes,zscore(hemoData.mean((0,1))))
                ax[1].plot(hemoTimes[baselineDur],zscore(hemoData.mean((0,1)))[baselineDur],'r')
                
                ax[2].plot(trace)
                ax[2].plot(zscore(analog[1,:]))
                ax[2].plot(blueTimes,zscore(alldata.mean((0,1))[:nrFrames-removedframes,trialNr]))
                ax[2].plot(blueTimes[baselineDur],zscore((alldata.mean((0,1))[:nrFrames-removedframes,trialNr]))[baselineDur],'r')
                show()
            
            if missedframes.any():
                print(f'flag trial number-{trialNr+1} for removal because of missing frame')
                trialstoremove.append(trialNr)
            
        except:
            print(f'very problematic, nor prcessed, probably more than one frame missing trial{trialNr+1}')
            trialstoremove.append(trialNr)
    
    
    goodtrials=np.delete(np.arange(0,opts_dict['nrTrials']),trialstoremove)
    try:
        io.savemat(os.path.join(dataPath,f"{foldername}.mat"), {"correcteddata": alldata,
                                                                "goodtrials":goodtrials,
                                                                'metadata':opts_dict,
                                                                })    
    except:
        with open(  os.path.join(dataPath,f"{foldername}.pkl"), 'wb') as f:
            pickle.dump( {"correcteddata": alldata.astype('float32'),
                          "goodtrials":goodtrials,
                          'metadata':opts_dict,
                          }, f, pickle.HIGHEST_PROTOCOL)
            
            
        
#%% saving corrected movie




mov=caim.movie(np.moveaxis(alldata[:,:,:,goodtrials].mean(axis=3), 2, 0))
mov.save(pth.Path(dataPath, opts_dict['fName']+'_processed.tiff'),compress=1, to32=True)
    
    
    
#%% plotting
initstimOn = int((opts_dict['preStim']*opts_dict['sRate'])) #frames before stimulus onset

#manually set some pixels to look and replot
xleft=80
xright=120
ytop=120
ybottom=180
meanTrace = np.mean(alldata[ytop:ybottom,xleft:xright,:,goodtrials],axis=(0,1))
men=np.mean(meanTrace,1)
timeTrace = (np.arange(0,nrFrames) / opts_dict['sRate']) - opts_dict['preStim']


if opts_dict['Delay']:
    stimOn=44
    stimOn=31

else:
    stimOn=int((opts_dict['preStim']*opts_dict['sRate']))
      
deviation= stimOn-   initstimOn
    
timeTrace = (np.arange(0,nrFrames) / opts_dict['sRate']) - opts_dict['preStim']-deviation/opts_dict['sRate']
sem=np.std(meanTrace, ddof=1) / np.sqrt(np.size(meanTrace))






f,ax=plt.subplots(1,2,figsize=(24,12))
f.set_tight_layout(True)

avgMap = alldata[:,:,stimOn+1:stimOn+10,goodtrials].mean(axis=(2,3))
im=ax[0].imshow(avgMap,cmap='inferno')
f.colorbar(im,ax=ax[0],shrink=0.5, orientation='horizontal',label='ΔF/F (%)')
ax[0].set_title('Stimulus-triggered activity')






# if opts_dict['Delay']:
#     plot(diff(men))

# ax[1].plot(timeTrace,men,'k')
# ax[1].fill_between(timeTrace, men-sem, men+sem,color='black', alpha=0.5)

ax[1].plot(men,'k')
ax[1].fill_between(men-sem, men+sem,color='black', alpha=0.5)

ax[1].set_title('Average change over all pixels')
ax[1].vlines(0,min(men),max(men),color='black',linestyles ='dashed')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('ΔF/F (%)')

#%%
xleft=46
xright=xleft+143
ytop=47
ybottom=ytop+145


f,ax=plt.subplots(1,figsize=(18,12))
f.set_tight_layout(True)
avgMap = alldata[ytop:ybottom,xleft:xright,stimOn+1:stimOn+10,goodtrials].mean(axis=(2,3))
im=ax.imshow(avgMap,cmap='inferno')
cb=f.colorbar(im,ax=ax,shrink=0.5,)
cb.set_label(label='ΔF/F (%)',fontsize = 20)
ax.set_title('Stimulus-triggered activity',fontsize = 40,weight='bold')
ax.set_xticks([])
ax.set_yticks([])

f,ax=plt.subplots(1,figsize=(18,12))
f.set_tight_layout(True)
ax.plot(timeTrace,men,'k')
ax.fill_between(timeTrace, men-sem, men+sem,color='black', alpha=0.5)
ax.set_title('Average change over selected area',fontsize = 40,weight='bold')
ax.vlines(0,min(men),max(men),color='black',linestyles ='dashed')
ax.set_xlabel('Time (s)',fontsize = 30)
ax.set_ylabel('ΔF/F (%)',fontsize = 30)