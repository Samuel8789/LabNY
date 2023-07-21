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
import matplotlib.pyplot as plt
import mat73
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from scipy.spatial.distance import cosine


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

def do_trial_analysis(times_seconds, mat, calcium, calcium_shuffled=np.array([]),prespiketime=5, postspiketime=5):
    
    timestamps_video=mat['time_transients']
    imagingrate=1/mat['frame_period']
 
    if calcium_shuffled.any():
        calcium=  calcium_shuffled[10,:,:]

    prespiketime=prespiketime
    postspiketime=postspiketime
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


def save_pdf(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
       fig.savefig(pp, format='pdf')
    pp.close()
    plt.close('all')



#%% Load on acid and movie data

# cnm = cnmf.online_cnmf.OnACID(path=hdf5_file_path)
data = mat73.loadmat(r'C:\Users\sp3660\Desktop\Chandelier_ Calcium&Volatage\Chandelie_AS-003_20230610-102153_sort.mat')
accepted_list_sorter=data['proc']['comp_accepted'].astype(int)
accepted_list_sorter_core=data['proc']['comp_accepted_core'].astype(int)
# substract 1 from matlab indexes
accepted_indexes_sorter=data['proc']['idx_components'].astype(int)-1
rejected_indexes_sorter=data['proc']['idx_components_bad'].astype(int)
accepted_indexes_sorter_manual=data['proc']['idx_manual'].astype(int)
rejected_indexes_sorter_manual=data['proc']['idx_manual_bad'].astype(int)
accepted_indexes_caiman=data['est']['idx_components']
rejected_indexes_caiman=data['est']['idx_components_bad']    
accepted_cells_number=len( accepted_indexes_sorter)
final_accepted_cells_matlabcorrected_indexes= accepted_indexes_sorter

C_matrix=data['est']['C'][accepted_indexes_sorter,:]
YrA_matrix=data['est']['YrA'][accepted_indexes_sorter,:]
raw=C_matrix + YrA_matrix
plt.figure()
chandelier=3
plt.plot(raw[chandelier,1:])

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

noisedata=pd.read_csv(noisdatapath)

noiserois=noisedata.values.T


voltagedata=pd.read_csv(os.path.join(dirpath,voltagedat))
calcium=mat['CaTransients']
timestamps_voltage=mat['time_voltage']
timestamps_video=mat['time_transients']
chandelier_activity=mat['Cell_caTransient']
chand_record=mat['recording_method']
binarizedspikes=mat['Aps_in_frames_bins']
imagingrate=1/mat['frame_period']
spiketimes_seconds=mat['spike_time']/mat['FPS_V']

calcium=np.concatenate([calcium, noiserois,np.expand_dims(raw[chandelier,1:],axis=0)])



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

# thresholding firing rate

thr=300
ttt=np.argwhere(gaussiankernelfiringrate<thr)
thresholdedfr=gaussiankernelfiringrate-thr
thresholdedfr[ttt]=0
diffthre=2
burstsfr=np.where(np.diff(thresholdedfr)>diffthre)[0]
burstsfr=np.insert(burstsfr,0,0)
firing_rate_bursts=burstsfr[np.where(np.diff(burstsfr)>1)[0]+1]
burstsfr_times_seconds=timestamps_video[firing_rate_bursts]

#shuffling calcium activity
n=10000
randp=np.zeros((n)).astype(int)
shuffled=np.zeros((n,calcium.shape[0],calcium.shape[1]))
for n in range(n):
    for cell in range(calcium.shape[0]):
        randp[n] = np.random.randint(0,calcium.shape[1])
        shuffled[n,cell,:] = np.roll(calcium[cell,:],randp[n])

calcium_shuffled=shuffled

prespiketime=5
postspiketime=5
#disect activity by trail
burst_data=do_trial_analysis(bursts_times_seconds,mat,calcium,prespiketime=prespiketime, postspiketime=postspiketime) 
burst_data_shuffled=do_trial_analysis(bursts_times_seconds,mat,calcium,calcium_shuffled,prespiketime=prespiketime, postspiketime=postspiketime) 
burstfr_data=do_trial_analysis(burstsfr_times_seconds,mat,calcium,prespiketime=prespiketime, postspiketime=postspiketime) 
burstfr_data_shuffled=do_trial_analysis(burstsfr_times_seconds,mat,calcium,calcium_shuffled,prespiketime=prespiketime, postspiketime=postspiketime) 
spike_data=do_trial_analysis(spiketimes_seconds,mat,calcium,prespiketime=prespiketime, postspiketime=postspiketime) 
spike_data_shuffled=do_trial_analysis(spiketimes_seconds,mat,calcium,calcium_shuffled,prespiketime=prespiketime, postspiketime=postspiketime) 
alldata=[spike_data,spike_data_shuffled,burst_data,burst_data_shuffled,burstfr_data,burstfr_data_shuffled]

#%% SELECT BUS AND ZSCORE
#determine activity to plot
bursts_flag=False
burstsfr_flag=True
scored_flag=True
window=10

if burstsfr_flag:
    data=[4,5]
    times_seconds=burstsfr_times_seconds
elif bursts_flag:
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
alignedcorrelationsvolt=alignedcorrelations[0,1:]


matforcosines=np.concatenate((np.expand_dims(zscore(alignedrate),axis=0), zscore(alignedclac)))
cosines=np.zeros((matforcosines.shape[0],matforcosines.shape[0]))
for i in range(matforcosines.shape[0]):
    for j in range(matforcosines.shape[0]):
        cosines[i,j]=1-cosine(zscore(matforcosines[i,:]),zscore(matforcosines[j,:]))
        
        
sorted_by_alignedcorrelation=np.flip(np.argsort(alignedcorrelations[0,1:]))
sorted_by_alignedcosines=np.flip(np.argsort(cosines[0,1:]))



matforcosines=np.concatenate((np.expand_dims(zscore(alignedrate),axis=0), zscore(alignedclac[sorted_by_alignedcosines,:])))
sortedcosines=np.zeros((matforcosines.shape[0],matforcosines.shape[0]))
for i in range(matforcosines.shape[0]):
    for j in range(matforcosines.shape[0]):
        sortedcosines[i,j]=1-cosine(zscore(matforcosines[i,:]),zscore(matforcosines[j,:]))





alignedcorr = signal.correlate(alignedrate-np.mean(alignedrate), alignedclac[0,:] - np.mean(alignedclac[0,:]), mode="full")
alignedlags = signal.correlation_lags(len(alignedrate), len(alignedclac[0,:]))
alignedlag = alignedlags[np.argmax(alignedcorr)]

alignedcorrelationssorted=np.corrcoef(alignedrate, alignedclac[sorted_by_alignedcorrelation,:])
alignedcorrelationssortedvolt=alignedcorrelationssorted[0,1:]


meancellactivity=activity[:,:,:].mean(0)
meancellactivity_shuffled=shuffledactivity[:,:,:].mean(0)



allres=[]
for cell in range(35):
    res=linregress(zscore(gaussiankernelfiringrate), y=zscore(calcium[cell,:]))
    allres.append(res.rvalue**2)
#%% PLOTTING 1
plt.plot(np.diff(thresholdedfr))
plt.plot(thresholdedfr)
plt.plot(gaussiankernelfiringrate)
plt.plot(firing_rate_bursts,thresholdedfr[firing_rate_bursts],'o')


f,ax=plt.subplots()
ax.plot(spiketimes_seconds,np.ones_like(spiketimes_seconds),'o')
bursts_times_seconds=timestamps_video[firing_rate_bursts]
ax.vlines(bursts_times_seconds,ymin=0.98,ymax=1.02)



#%% plotting 2


mm=np.zeros([36,294])+perispikerange
mmm=np.stack([mm,meancellactivity[:36,:]],axis=2)
ttt=mmm.tolist()

x = alignedcorrelationsvolt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_segments = LineCollection(ttt,array=x)

fig, ax = plt.subplots()
ax.set_xlim(perispikerange[0], perispikerange[-1])
ax.set_ylim(-1, 1)
axcb = fig.colorbar(line_segments)
ax.add_collection(line_segments)
plt.show()





f,ax=plt.subplots()
ax.plot(corr / np.max(corr))

f,ax=plt.subplots(2,figsize=(20,12),sharex=True)
ax[0].plot(contimestamps,zscore(gaussiankernelfiringrate))
ax[0].plot(timestamps_video,zscore(smooth_trace(calcium[0,:],10)))
ax[1].plot(contimestamps[:lag],zscore(gaussiankernelfiringrate[:lag]))
ax[1].plot(timestamps_video[:lag],zscore(smooth_trace(calcium[0,np.abs(lag):],10)))

#plot caiman chandlier
f,ax=plt.subplots(2,figsize=(20,12),sharex=True)
ax[0].plot(timestamps_video,zscore(calcium[0,:]))
ax[1].plot(timestamps_video,zscore(smooth_trace(calcium[-1,:],10)))




f,ax=plt.subplots(2,figsize=(20,12))
ax[0].imshow(alignedclac,aspect='auto')
ax[1].imshow(alignedclac[sorted_by_alignedcorrelation,:],aspect='auto')




f,ax=plt.subplots(figsize=(20,12))
pos=ax.imshow(alignedcorrelations)
f.colorbar(pos, ax=ax,shrink=0.6)

f,ax=plt.subplots(figsize=(20,12))
pos=ax.imshow(alignedcorrelationssorted)
f.colorbar(pos, ax=ax,shrink=0.6)
# linear regression






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
cell=np.argwhere(sorted_by_alignedcorrelation==sorted_cell)[0][0]

plt.close('all')
#%% LOOPING FULL PLOT SUMMARY (SLOW)

 # ax.set_xlabel('Time(s)')
 # ax.set_ylabel('Activity(a.u.)')

for sorted_cell, cell in enumerate(sorted_by_alignedcorrelation):
    f,ax=plt.subplots(4,2,figsize=(20,12))
    f.suptitle(f'PeriSpike Analysis Cell{cell+1}', fontsize=16)

    
    ax[0,0].sharex(ax[1,0])
    ax[0,0].plot(contimestamps,zscore(gaussiankernelfiringrate),'r')
    ax[0,0].plot(timestamps_video,zscore(smooth_trace(calcium[0,:],10)),'b')
    ax[0,0].plot(times_seconds,np.ones_like(times_seconds)-4,'ro')
    ax[0,0].plot(timestamps_voltage,chand_record+5,'r')
    ax[0,0].set_xlabel('Time(s)')
    
    ax[1,0].plot(timestamps_video, zscore(smooth_trace(calcium[cell,:],10)),'b')
    ax[1,0].plot(contimestamps,zscore(gaussiankernelfiringrate),'r')
    ax[1,0].text(0.95, 0.95, f'pearson correlation={np.round(alignedcorrelationssortedvolt[sorted_cell],3)}',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax[1,0].transAxes)
    ax[1,0].set_xlabel('Time(s)')

    
    
    res=linregress(zscore(gaussiankernelfiringrate), y=zscore(calcium[cell,:]))
    ax[2,0].scatter(zscore(gaussiankernelfiringrate),zscore(calcium[cell,:]))
    ax[2,0].plot(zscore(gaussiankernelfiringrate), res.intercept + res.slope*zscore(gaussiankernelfiringrate), 'r', label='fitted line')
    ax[2,0].text(0.95, 0.95, f"R-squared: {res.rvalue**2:.6f}",
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax[2,0].transAxes)
    ax[2,0].set_xlabel('Zscore')

    
    
    
    corr = signal.correlate(gaussiankernelfiringrate-np.mean(gaussiankernelfiringrate), calcium[cell,:] - np.mean(calcium[cell,:]), mode="full")
    lags = signal.correlation_lags(len(gaussiankernelfiringrate), len(calcium[cell,:]))
    lag = lags[np.argmax(corr)]
    # corr /= np.max(corr)
    ax[3,0].plot(lags,corr)
    ax[3,0].set_ylim([-1500000,2000000])
    ax[3,0].set_xlabel('Lags (ms)')

    

    
    ax[0,1].plot(perispikerange,meancellactivitychand,'k')
    ax[1,1].plot(perispikerangevoltage,meanspike,'k')
    ax[2,1].plot(perispikerange,meancellactivity[cell,:],'k')
    ax[3,1].plot(perispikerange,meancellactivity_shuffled[cell,:],'k')
    ax[0,1].sharex(ax[1,1])
    ax[1,1].sharex(ax[2,1])
    ax[2,1].sharex(ax[3,1])
    
    ax[1,1].set_xlabel('Time(s)')
    for x in [0,2,3]:
        ax[x,1].set_ylim([-3,3])
        ax[x,1].set_xlabel('Time(s)')
    
    
        
    
    allchand=activity[:,0,:]
    alltestcell=activity[:,cell,:]
    allshuffled=shuffledactivity[:,0,:]
    xarray=np.zeros(alltestcell.shape)+perispikerange
    
    chandarray=np.stack([xarray,allchand],axis=2).tolist()
    testcellarray=np.stack([xarray,alltestcell],axis=2).tolist()
    shuffledarray=np.stack([xarray,allshuffled],axis=2).tolist()
    
    line_segmentschand = LineCollection(chandarray,alpha=0.1)
    line_segmentstest = LineCollection(testcellarray,alpha=0.1)
    line_segmentsshuffled = LineCollection(shuffledarray,alpha=0.1)
    
    # ax[0,1].set_xlim(perispikerange[0], perispikerange[-1])
    # ax[0,1].set_ylim(-5, 5)
    ax[0,1].add_collection(line_segmentschand)
    ax[2,1].add_collection(line_segmentstest)
    ax[3,1].add_collection(line_segmentsshuffled)
    
    plt.show()
    
#%% Chandelier summary

f,ax=plt.subplots(3,sharex=True)
plt.subplots_adjust(hspace=.0)
f.tight_layout()
ax[0].plot(timestamps_video,calcium[0,:],'k')
ax[0].set_ylabel('Fluorescence (a.u.)',fontsize=16)
chandflulimits=ax[0].get_ylim()

ax[1].plot(contimestamps,gaussiankernelfiringrate/5,'k')
ax[1].set_ylabel('Firing Rate (hz)',  fontsize=18)
ax[1].get_xaxis().set_visible(False)


ax[2].plot(timestamps_voltage,chand_record+5,'k')
ax[2].plot(times_seconds,np.zeros_like(times_seconds)-3,'ko')
ax[2].set_ylabel('Voltage (mV)',fontsize=16)

ax[2].set_xlabel('Time (s)',fontsize=16)

for x in ax:
    x.margins(x=0)
    
    
#%%chandelier spike average only
    
f,ax=plt.subplots()
f.tight_layout()
ax.plot(perispikerangevoltage,meanspike,'k')
for i in range(voltage.shape[0]):
    ax.plot(perispikerangevoltage,voltage[i,:],alpha=0.01,color='grey')
ax.set_ylabel('Voltage (mV)',fontsize=16)
ax.set_xlabel('Time (s)',fontsize=16)


#%% chandelier spike triggered summary
allchand=activity[:,0,:]
allvolt=voltage
xarray=np.zeros(allchand.shape)+perispikerange
chandarray=np.stack([xarray,allchand],axis=2).tolist()
chandshuffled=shuffledactivity[:,0,:]
chandarrayshuffled=np.stack([xarray,chandshuffled],axis=2).tolist()

f,ax=plt.subplots(3,sharex=True)
plt.subplots_adjust(hspace=.0)
f.tight_layout()
ax[0].plot(perispikerange,meancellactivitychand,'k')
ax[0].set_ylim(chandflulimits)
ax[0].set_ylim([-2,2])

ax[1].plot(perispikerangevoltage,meanspike,'k')
ax[2].plot(perispikerange,meancellactivity_shuffled[0,:],'k')
ax[2].set_ylim(chandflulimits)
ax[2].set_ylim([-2,2])

ax[0].set_ylabel('Fluorescence (a.u.)',fontsize=16)
ax[1].set_ylabel('Voltage (mV)',fontsize=16)
ax[2].set_ylabel('Fluorescence (a.u.)',fontsize=16)
ax[-1].set_xlabel('Time (s)',fontsize=16)


line_segmentschand = LineCollection(chandarray,alpha=0.1,color='grey')
line_segmentschandshuffled = LineCollection(chandarrayshuffled,alpha=0.1,color='grey')

# ax[0,1].set_xlim(perispikerange[0], perispikerange[-1])
# ax[0,1].set_ylim(-5, 5)
ax[0].add_collection(line_segmentschand)
ax[2].add_collection(line_segmentschandshuffled)
for x in ax:
    x.margins(x=0)


#%% correlation summaries
sorted_by_alignedcorrelation
cell=0
f,ax=plt.subplots(3)
ax[0].plot(timestamps_video, zscore(smooth_trace(calcium[cell,:],10)),'r',label='Raw Calcium Non-Chan')
#set cell to 0 and comment previous line for chandelier only figure
f.tight_layout()


ax[0].plot(timestamps_video, zscore(smooth_trace(calcium[0,:],10)),'k',label='Raw Calcium Chan')
ax[0].plot(contimestamps,zscore(gaussiankernelfiringrate)-3,'b',label='Firing Rate')
ax[0].text(0.95, 0.95, f'Pearson correlation={np.round(alignedcorrelationsvolt[cell],3)}',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax[0].transAxes,fontsize=16)
ax[0].set_xlabel('Time(s)',fontsize=16)
ax[0].set_ylabel('Z-Score',fontsize=16)
ax[0].legend(loc='upper left',fontsize=16)


res=linregress(zscore(gaussiankernelfiringrate), y=zscore(calcium[cell,:]))
ax[1].scatter(zscore(gaussiankernelfiringrate),zscore(calcium[cell,:]),color='k')
ax[1].plot(zscore(gaussiankernelfiringrate), res.intercept + res.slope*zscore(gaussiankernelfiringrate), 'r', label='fitted line')
ax[1].text(0.95, 0.95, f"R-squared: {res.rvalue**2:.6f}",
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax[1].transAxes,fontsize=16)
ax[1].set_xlabel('Firing Rate Z-Score',fontsize=16)
ax[1].set_ylabel('Calcium Z-Score',fontsize=16)


corr = signal.correlate(gaussiankernelfiringrate-np.mean(gaussiankernelfiringrate), calcium[cell,:] - np.mean(calcium[cell,:]), mode="full")
lags = signal.correlation_lags(len(gaussiankernelfiringrate), len(calcium[cell,:]))
lag = lags[np.argmax(corr)]
corr /= np.max(corr)
ax[2].plot(lags,corr,'k')
ax[2].set_xlabel('Lags (frames)',fontsize=16)
ax[2].set_ylabel('Crosscorrelation',fontsize=16)

for x in ax:
    x.margins(x=0)

#%% global correlations heatmaps

        
cellmax=28+9
f,ax=plt.subplots(1,2)
f.tight_layout()
pos=ax[0].imshow(alignedcorrelations[:cellmax,:cellmax])
f.colorbar(pos, ax=ax[0],shrink=0.6)
ax[0].set_title("Pearson Correlations",fontsize=20)

pos2=ax[1].imshow(alignedcorrelationssorted[:cellmax,:cellmax])
f.colorbar(pos2, ax=ax[1],shrink=0.6)
ax[1].set_title("Pearson Correlations (Sorted by Correlation)",fontsize=20)


# pos3=ax[1,0].imshow(cosines[:28,:28])
# f.colorbar(pos3, ax=ax[1,0],shrink=0.6)
# ax[1,0].set_title("Cosine Similarity")

# pos4=ax[1,1].imshow(cosinessorted[:28,:28])
# f.colorbar(pos4, ax=ax[1,1],shrink=0.6)
# ax[1,1].set_title("Cosine Similarity (Sorted by Correlation)")


for x in ax:
    x.set_xlabel('Cell',fontsize=16)
    x.set_ylabel('Cell',fontsize=16)
    x.set_yticks(np.insert(np.arange(-1,cellmax,5)[1:],0,0))
    x.set_yticklabels(np.insert(np.arange(-1,cellmax,5)[1:],0,0)+1)
    x.set_xticks(np.insert(np.arange(-1,cellmax,5)[1:],0,0))
    x.set_xticklabels(np.insert(np.arange(-1,cellmax,5)[1:],0,0)+1)

#%% ALL STAs
cellmax=28+8
mm=np.zeros([cellmax,294])+perispikerange
mmm=np.stack([mm,meancellactivity[:cellmax,:]],axis=2)
ttt=mmm.tolist()
x = alignedcorrelationsvolt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_segments = LineCollection(ttt,array=x)
fig, ax = plt.subplots()
ax.set_xlim(perispikerange[0], perispikerange[-1])
ax.set_ylim(-1, 2)
axcb = fig.colorbar(line_segments)
ax.add_collection(line_segments)
ax.set_xlabel('Time(s)',fontsize=16)
ax.set_ylabel('Z-Score',fontsize=16)

#%% zscore tirggereaverage of cell with chandelier
cell=-1
allchand=activity[:,0,:]
allvolt=voltage
xarray=np.zeros(allchand.shape)+perispikerange
chandarray=np.stack([xarray,allchand],axis=2).tolist()
chandshuffled=shuffledactivity[:,0,:]
chandarrayshuffled=np.stack([xarray,chandshuffled],axis=2).tolist()
cellarray=np.stack([xarray,activity[:,cell,:]],axis=2).tolist()


f,ax=plt.subplots(4,sharex=True)
plt.subplots_adjust(hspace=.0)
f.tight_layout()
ax[0].plot(perispikerange,meancellactivitychand,'k')
ax[0].set_ylim([-2,2])

ax[1].plot(perispikerangevoltage,meanspike,'k')
ax[2].plot(perispikerange,meancellactivity[cell,:],'k')
ax[2].set_ylim([-2,2])
ax[3].plot(perispikerange,meancellactivity_shuffled[cell,:],'k')
ax[3].set_ylim([-2,2])

ax[0].set_ylabel('ZScore (a.u.)',fontsize=16)
ax[1].set_ylabel('Voltage (mV)',fontsize=16)
ax[2].set_ylabel('ZScore',fontsize=16)
ax[3].set_ylabel('ZScore',fontsize=16)
ax[-1].set_xlabel('Time (s)',fontsize=16)

ax[1].set_ylim([-5,1])
for i in range(voltage.shape[0]):
    ax[1].plot(perispikerangevoltage,voltage[i,:],alpha=0.01,color='grey')

    


line_segmentschand = LineCollection(chandarray,alpha=0.01,color='grey')
line_segmentschandshuffled = LineCollection(chandarrayshuffled,alpha=0.01,color='grey')
line_segmentscell = LineCollection(cellarray,alpha=0.01,color='grey')

# ax[0,1].set_xlim(perispikerange[0], perispikerange[-1])
# ax[0,1].set_ylim(-5, 5)
ax[0].add_collection(line_segmentschand)
ax[2].add_collection(line_segmentscell)
ax[3].add_collection(line_segmentschandshuffled)
for x in ax:
    x.margins(x=0)
    
    
    
    
 #%% Bursts definitions
 
bursts_times_seconds=spiketimes_seconds[bursts]
burstsfr_times_seconds=timestamps_video[firing_rate_bursts]

 
# plt.plot(thresholdedfr/5)
f,ax=plt.subplots(2,sharex=True)
ax[0].plot(contimestamps,gaussiankernelfiringrate/5)
ax[0].plot(burstsfr_times_seconds,gaussiankernelfiringrate[firing_rate_bursts]/5,'o')
ax[-1].set_xlabel('Time (s)',fontsize=16)
ax[0].set_ylabel('Firing Rate (hz)',  fontsize=18)
ax[1].plot(spiketimes_seconds,np.ones_like(spiketimes_seconds),'o')
ax[1].vlines(burstsfr_times_seconds,ymin=0.98,ymax=1.02)


f,ax=plt.subplots(2,sharex=True)
ax[0].plot(spiketimes_seconds,np.ones_like(spiketimes_seconds),'o')
ax[0].vlines(bursts_times_seconds,ymin=0.98,ymax=1.02)
ax[1].plot(spiketimes_seconds,np.ones_like(spiketimes_seconds),'o')
ax[1].vlines(burstsfr_times_seconds,ymin=0.98,ymax=1.02)
ax[-1].set_xlabel('Time (s)',fontsize=16)
