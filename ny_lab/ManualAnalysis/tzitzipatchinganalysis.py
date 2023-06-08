# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:16:37 2023

@author: sp3660
"""

import os
import scipy.io as spio
from pathlib import Path
import pprint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import zscore, linregress
import random
import matplotlib.animation as animation
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import signal
import caiman as cm

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

def smooth_trace( trace,window):
    framenumber=len(trace)
    frac=window/framenumber
    filtered = lowess(trace, np.arange(framenumber), frac=frac)
    return filtered[:,1]  

def do_trial_analysis(times_seconds, mat, calcium, calcium_shuffled=np.array([])):
    
    timestamps_video=mat['time_transients']
    imagingrate=1/mat['frame_period']
 
    if calcium_shuffled.any():
        calcium=  calcium_shuffled[10,:,:]

    prespiketime=5
    postspiketime=5
    videostimpreframes=int(prespiketime*imagingrate)
    videostimpostframes=int(postspiketime*imagingrate)
    perispikerange=np.linspace(-prespiketime,postspiketime,videostimpreframes+videostimpostframes )
    spiketrialactivity=np.zeros((len(times_seconds),calcium.shape[0],len(perispikerange)))
    

    videostimpreframesvoltage=int(prespiketime*mat['FPS_V'])
    videostimpostframesvoltage=int(postspiketime*mat['FPS_V'])
    perispikerangevoltage=np.linspace(-prespiketime,postspiketime,videostimpreframesvoltage+videostimpostframesvoltage )
    spiketrialactivityvoltage=np.zeros((len(times_seconds),len(perispikerangevoltage)))
    
    
    spikeframes=np.zeros_like(times_seconds).astype(int)
    spikeframevoltage=np.zeros_like(times_seconds).astype(int)

    
    for spike, spike_time in enumerate(times_seconds):
        for cell in range(calcium.shape[0]):
            spikeframes[spike]=np.argmin(np.abs(timestamps_video-spike_time))

            if spikeframes[spike]+videostimpostframes<=len(timestamps_video) and spikeframes[spike]-videostimpreframes>0:
                
                spiketrialactivity[spike, cell,:]=calcium[cell,  spikeframes[spike]-videostimpreframes:spikeframes[spike]+videostimpostframes]
    
            elif spikeframes[spike]+videostimpostframes>len(timestamps_video) :
                
                pad=np.zeros(spikeframes[spike]+videostimpostframes-len(timestamps_video))
                spiketrialactivity[spike, cell,:]=np.concatenate((calcium[cell,  spikeframes[spike]-videostimpreframes:],pad))
                
               
            elif spikeframes[spike]-videostimpreframes<0:
                pad=np.zeros( np.abs(spikeframes[spike]-videostimpreframes))
                spiketrialactivity[spike, cell,:]=np.concatenate((pad,calcium[cell,  :spikeframes[spike]+videostimpostframes]))
            
    for spike, spike_time in enumerate(times_seconds):
            spikeframevoltage[spike]=np.argmin(np.abs(timestamps_voltage-spike_time))
            if spikeframevoltage[spike]+videostimpostframesvoltage<=len(timestamps_voltage) and spikeframevoltage[spike]-videostimpreframesvoltage>0:
                
                spiketrialactivityvoltage[spike,:]=chand_record[spikeframevoltage[spike]-videostimpreframesvoltage:spikeframevoltage[spike]+videostimpostframesvoltage]
    
            elif spikeframevoltage[spike]+videostimpostframesvoltage>len(timestamps_voltage) :
    
                pad2=np.zeros(spikeframevoltage[spike]+videostimpostframesvoltage-len(timestamps_voltage))
                spiketrialactivityvoltage[spike,:]=np.concatenate((chand_record[spikeframevoltage[spike]-videostimpreframesvoltage:],pad2))
                
            elif spikeframevoltage[spike]-videostimpreframesvoltage<0:
           
                pad2=np.zeros( np.abs(spikeframevoltage[spike]-videostimpreframesvoltage))
                spiketrialactivityvoltage[spike,:]=np.concatenate((pad2,chand_record[  :spikeframevoltage[spike]+videostimpostframesvoltage]))

    scoredactivity=zscore(spiketrialactivity,axis=2)
    scoredvoltage=zscore(spiketrialactivityvoltage,axis=1)

    return spiketrialactivity,spiketrialactivityvoltage,scoredactivity,scoredvoltage,perispikerange,perispikerangevoltage

def scale_signal(signal,method='ZeroOne'):
    import scipy.stats as stats
    if method=='ZeroOne':
        scaled_signal=(signal-np.min(signal))/(np.max(signal)-np.min(signal))
                                        
    elif method=='ZScored':
        scaled_signal=stats.zscore(signal)
                         
    return scaled_signal
#do caiman of video









#%%       compute  
#loading data and variables
dirpath=r'C:\Users\sp3660\Desktop\Chandelier_ Calcium&Volatage'
dataname='Chandelie_AS-003_data.mat'
full_data_path=os.path.join(dirpath,dataname)
voltagedat='VoltageRecording_AS_003_ctr.csv'
mat=loadmat(full_data_path)
moviname='Chandelie_AS-003.tif'
moviepath=os.path.join(dirpath,moviname)
mov=cm.load(moviepath)
noisecellsname='noise_data.csv'
noisdatapath=os.path.join(dirpath,noisecellsname)

voltagedata=pd.read_csv(os.path.join(dirpath,voltagedat))
calcium=mat['CaTransients']
timestamps_voltage=mat['time_voltage']
timestamps_video=mat['time_transients']
chandelier_activity=mat['Cell_caTransient']
chand_record=mat['recording_method']
binarizedspikes=mat['Aps_in_frames_bins']
imagingrate=1/mat['frame_period']
spiketimes_seconds=mat['spike_time']/mat['FPS_V']



#defining bursts of activity
isi=np.diff(spiketimes_seconds)
x=plt.hist(isi,1000,log=True)
# thr_isi=mat['frame_period']
thr_isi=0.2
bursts=np.argwhere(isi>thr_isi)+1
bursts=np.insert(bursts,0,0)
bursts_times_seconds=spiketimes_seconds[bursts]


# convolving spikes with gaussian
bintime=len(chand_record)/(10000*calcium.shape[1])
binframes=bintime*mat['FPS_V']
totalbinnedframes=int(np.around(len(chand_record)/binframes))
bined_acti=np.zeros(totalbinnedframes)
convperiod=np.around(binframes)/mat['FPS_V']
contimestamps=np.linspace(timestamps_video[0],totalbinnedframes*convperiod,totalbinnedframes)
spiketimes_10kz=spiketimes_seconds*10000
for n in range(totalbinnedframes):
    bined_acti[n]=len(np.where(np.logical_and(spiketimes_10kz>=n*np.around(binframes), spiketimes_10kz<=(n+1)*np.around(binframes)))[0])/bintime
sigma=1
gx = np.arange(-3*sigma, 3*sigma, thr_isi)
gaussian = np.exp(-(gx/sigma)**2/2)
gaussiankernelfiringrate = np.convolve(bined_acti, gaussian, mode='same')


#shuffling calcium activity
n=10000
randp=np.zeros((n)).astype(int)
shuffled=np.zeros((n,calcium.shape[0],calcium.shape[1]))
for n in range(n):
    for cell in range(calcium.shape[0]):
        randp[n] = np.random.randint(0,calcium.shape[1])
        shuffled[n,cell,:] = np.roll(calcium[cell,:],randp[n])

calcium_shuffled=shuffled


#disect activity by trail
burst_data=do_trial_analysis(bursts_times_seconds,mat,calcium) 
burst_data_shuffled=do_trial_analysis(bursts_times_seconds,mat,calcium,calcium_shuffled) 
spike_data=do_trial_analysis(spiketimes_seconds,mat,calcium) 
spike_data_shuffled=do_trial_analysis(spiketimes_seconds,mat,calcium,calcium_shuffled) 
alldata=[spike_data,spike_data_shuffled,burst_data,burst_data_shuffled]


#determine activity to plot
bursts_flag=False
scored_flag=True
window=10

if bursts_flag:
    data=[2,3]
    times_seconds=bursts_times_seconds
else:
    data=[0,1]
    times_seconds=spiketimes_seconds

if scored_flag:
    activ=[2 ,3]
else:
    activ=[0 ,1] 
          
activity=alldata[data[0]][activ[0]]
voltage=alldata[data[0]][activ[1]]
shuffledactivity=alldata[data[1]][activ[0]]
shuffledvoltage=alldata[data[1]][activ[1]]
perispikerange=alldata[data[0]][4]
perispikerangevoltage=alldata[data[0]][5]
meanspike=voltage.mean(0)
meancellactivitychand=activity[:,0,:].mean(0)


#crosscorrelate calcium trace and firing rate of chandelier, calculate shifted traces for ref
corr = signal.correlate(gaussiankernelfiringrate-np.mean(gaussiankernelfiringrate), calcium[0,:] - np.mean(calcium[0,:]), mode="full")
lags = signal.correlation_lags(len(gaussiankernelfiringrate), len(calcium[0,:]))
lag = lags[np.argmax(corr)]
# corr /= np.max(corr)

allcoors=np.corrcoef(gaussiankernelfiringrate, calcium )
voltagecoors=allcoors[0,1:]
sorted_by_correlation=np.flip(np.argsort(voltagecoors))

sortedcorr=np.corrcoef(gaussiankernelfiringrate, calcium[sorted_by_correlation,:])
sortedvoltagecoors=sortedcorr[0,1:]

alignedrate=gaussiankernelfiringrate[:lag]
alignedclac=calcium[:,np.abs(lag):]
alignedcorrelations=np.corrcoef(alignedrate, alignedclac )
sorted_by_alignedcorrelation=np.flip(np.argsort(alignedcorrelations[0,1:]))


alignedcorr = signal.correlate(alignedrate-np.mean(alignedrate), alignedclac[0,:] - np.mean(alignedclac[0,:]), mode="full")
alignedlags = signal.correlation_lags(len(alignedrate), len(alignedclac[0,:]))
alignedlag = alignedlags[np.argmax(alignedcorr)]
#%%
alignedcorrelationssorted=np.corrcoef(alignedrate, alignedclac[sorted_by_alignedcorrelation,:])
alignedcorrelationssortedvolt=alignedcorrelationssorted[0,1:]
f,ax=plt.subplots()
ax.plot(corr)
ax.plot(alignedcorr)

f,ax=plt.subplots(2,figsize=(20,12),sharex=True)
ax[0].plot(timestamps_video,zscore(gaussiankernelfiringrate))
ax[0].plot(timestamps_video,zscore(smooth_trace(calcium[0,:],10)))
ax[1].plot(timestamps_video[:lag],zscore(gaussiankernelfiringrate[:lag]))
ax[1].plot(timestamps_video[:lag],zscore(smooth_trace(calcium[0,np.abs(lag):],10)))



f,ax=plt.subplots(2,figsize=(20,12))
ax[0].imshow(alignedclac,aspect='auto')
ax[1].imshow(alignedclac[sorted_by_alignedcorrelation,:],aspect='auto')




f,ax=plt.subplots(figsize=(20,12))
pos=ax.imshow(alignedcorrelations)
f.colorbar(pos, ax=ax,shrink=0.4)

f,ax=plt.subplots(figsize=(20,12))
pos=ax.imshow(alignedcorrelationssorted)
f.colorbar(pos, ax=ax,shrink=0.4)
# linear regression

res=linregress(zscore(alignedrate), y=zscore(alignedclac[cell,:]))





#%% all the plotting

#plot bursts definition
f,ax=plt.subplots()
ax.plot(spiketimes_seconds,np.ones_like(spiketimes_seconds),'o')
ax.vlines(bursts_times_seconds,ymin=0.98,ymax=1.02)


#plot smoothe firing rat
f,ax=plt.subplots()
# ax.plot(contimestamps,scale_signal(bined_acti,'ZScored'))
ax.plot(contimestamps,scale_signal(gaussiankernelfiringrate,'ZScored'))
ax.plot(timestamps_video,scale_signal(calcium[0,:],'ZScored'))
ax.plot(spiketimes_seconds,np.ones_like(spiketimes_seconds)-2,'o')


#plotting chandleier activty general
f,ax=plt.subplots(5,sharex=True)
ax[0].plot(timestamps_video,calcium[0,:])
ax[1].plot(timestamps_voltage,chand_record)
ax[2].plot(timestamps_video,scale_signal(calcium[0,:],'ZScored'))
ax[2].plot(contimestamps,scale_signal(gaussiankernelfiringrate,'ZScored'))
ax[3].plot(spiketimes_seconds,np.ones_like(spiketimes_seconds),'o')
ax[4].plot(bursts_times_seconds,np.ones_like(bursts_times_seconds),'o')
# ax[5].plot(timestamps_video,calcium_shuffled[0,0,:])

#animate all spikes
# f,ax=plt.subplots(3,sharex=True)
# line1,=ax[0].plot(perispikerange,meancellactivitychand,'k')
# ax[1].plot(perispikerangevoltage,meanspike,'k')
# line2,=ax[2].plot(perispikerange,meancellactivity,'k')
# ax[0].set_ylim([-2,5])
# ax[2].set_ylim([-2,5])
# def animate(spike):
#     line1.set_ydata(activity[spike,0,:])  # update the data.
#     line2.set_ydata(activity[spike,cell,:])  # update the data.
#     return line1,line2
# ani = animation.FuncAnimation(
#     f, animate, interval=1)


sorted_cell=0
chandsortedcell=np.argwhere(sorted_by_alignedcorrelation==0)[0][0]
plt.close('all')
#%%
for sorted_cell in range(len(sorted_by_alignedcorrelation)):
    cell=sorted_by_alignedcorrelation[sorted_cell]
    f,ax=plt.subplots(4,2,figsize=(20,12))
    
    ax[0,0].sharex(ax[1,0])
    ax[0,0].plot(contimestamps,zscore(gaussiankernelfiringrate),'r')
    ax[0,0].plot(timestamps_video,zscore(smooth_trace(calcium[0,:],10)),'b')
    ax[0,0].plot(times_seconds,np.ones_like(times_seconds)-4,'ro')
    ax[0,0].plot(timestamps_voltage,chand_record+5,'r')
    
    
    ax[1,0].plot(timestamps_video, zscore(calcium[cell,:]),'b')
    ax[1,0].plot(contimestamps,zscore(gaussiankernelfiringrate),'r')
    # for a in ax:
    #     a.margins(0)
    
    ax[1,0].text(0.95, 0.95, f'pearson correlation={np.round(alignedcorrelationssortedvolt[sorted_cell],3)}',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax[1,0].transAxes)
    
    res=linregress(zscore(alignedrate), y=zscore(alignedclac[cell,:]))

    
    ax[2,0].scatter(zscore(gaussiankernelfiringrate),zscore(calcium[cell,:]))
    ax[2,0].plot(zscore(gaussiankernelfiringrate), res.intercept + res.slope*zscore(gaussiankernelfiringrate), 'r', label='fitted line')
    ax[2,0].text(0.95, 0.95, f"R-squared: {res.rvalue**2:.6f}",
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax[2,0].transAxes)
    
    
    
    
    corr = signal.correlate(gaussiankernelfiringrate-np.mean(gaussiankernelfiringrate), calcium[cell,:] - np.mean(calcium[cell,:]), mode="full")
    lags = signal.correlation_lags(len(gaussiankernelfiringrate), len(calcium[cell,:]))
    lag = lags[np.argmax(corr)]
    # corr /= np.max(corr)
    ax[3,0].plot(lags,corr)
    ax[3,0].set_ylim([-1500000,2000000])
    
    
    
    meancellactivity=activity[:,cell,:].mean(0)
    meancellactivity_shuffled=shuffledactivity[:,cell,:].mean(0)
    meanshuffledactivity=shuffledactivity[:,cell,:].mean(0)
    
    
    ax[0,1].plot(perispikerange,meancellactivitychand,'k')
    ax[1,1].plot(perispikerangevoltage,meanspike,'k')
    ax[2,1].plot(perispikerange,meancellactivity,'k')
    ax[3,1].plot(perispikerange,meancellactivity_shuffled,'k')
    ax[0,1].sharex(ax[1,1])
    ax[1,1].sharex(ax[2,1])
    ax[2,1].sharex(ax[3,1])
    
    
    # for x in [0,2,3]:
    #     ax[x,1].set_ylim(bottom=20)
    
    for spike, spike_time in enumerate(times_seconds[len(times_seconds)//40:2*len(times_seconds)//40]):
        ax[0,1].plot(perispikerange,smooth_trace(activity[spike,0,:],window),'c',alpha=0.1)
        # ax[1].plot(perispikerangevoltage,voltage[spike,:],'y',alpha=0.05)
        ax[2,1].plot(perispikerange,smooth_trace(activity[spike,cell,:],window),'c',alpha=0.1)
        ax[3,1].plot(perispikerange,smooth_trace(shuffledactivity[spike,cell,:],window),'c',alpha=0.1)
    
    plt.show()
        
