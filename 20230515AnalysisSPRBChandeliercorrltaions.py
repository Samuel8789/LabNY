# -*- coding: utf-8 -*-
"""
Created on Sun May 14 20:01:46 2023

@author: sp3660
"""
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.rcParams["figure.autolayout"] = True

def smooth_trace(trace,window):
    framenumber=len(trace)
    frac=window/framenumber
    filtered = lowess(trace, np.arange(framenumber), frac=frac)
    
    return filtered[:,1]
dfdtsmoothed=full_data['imaging_data']['Plane1']['Traces']['dfdt_raw']
demixed=full_data['imaging_data']['Plane1']['Traces']['demixed']
chands=[13,16,23,115,15,24]

cres=dtset.most_updated_caiman.CaimanResults_object


chnafssorted=np.searchsorted(full_data['imaging_data']['Plane1']['CellIds'], chands)


pyrsorted=np.delete(np.arange(len(full_data['imaging_data']['Plane1']['CellIds'])), chnafssorted)


chanftraces=demixed[chnafssorted,:]
pyrtraces=demixed[pyrsorted,:]

thres=0.18
claciumthreshold=1.5
window=5
f,axs=plt.subplots(3,sharex=True)
for i in range(3):
    axs[0].plot(smooth_trace(demixed[chnafssorted[i],:],window))
    axs[1].plot(smooth_trace(dfdtsmoothed[chnafssorted[i],:],window))
    axs[1].hlines(y=thres, linewidth=2,xmin=0, xmax=[len(demixed[chnafssorted[i],:])], linestyles='-', color='r')
    
    smothedtrace=smooth_trace(demixed[chnafssorted[i],:],window)
    threshoildedtrace=smothedtrace
    dd=np.where([threshoildedtrace < claciumthreshold] )[1]
    
    
    
    thresholdes=smooth_trace(dfdtsmoothed[chnafssorted[i],:],window)
    
    thresholdes[thresholdes < thres] = 0
    thresholdes[dd] = 0

    axs[2].plot(thresholdes)

    
for i in range(3):
    axs[i].margins(x=0)

    
manualchandelierpeaks=[1100,1634,2066,2328,2616,2894,3088,3462,3796,4294,4582,5133,7000,7535,8414,9463,10189,10573,12153,13057,13389]

fr=dtset.metadata.translated_imaging_metadata['FinalFrequency']
framen=len(dtset.most_updated_caiman.CaimanResults_object.raw[1,:])
lengthtime=framen/fr
period=1/fr
fulltimevector=np.arange(0,lengthtime,period)
smoothwindows=10
framevector=np.arange(framen)


f,ax=plt.subplots(1)
f.tight_layout()
for i in range(3):
    trace=demixed[chnafssorted[i],:]
    ax.plot(framevector,smooth_trace(trace,smoothwindows))
    
    ax.margins(x=0)
    for j in range(len(manualchandelierpeaks)):
        ax.axvline(x=framevector[manualchandelierpeaks[j]],c='r')  
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Activity(a.u.)')
    # f.suptitle('Optogenetic Stimulation of Chandelier Cells', fontsize=16)
    
    
    

smoothedchandtraces=np.zeros_like(chanftraces)        
smoothedpyrtraces=np.zeros_like(pyrtraces)        

for i in range(chanftraces.shape[0]):
    smoothedchandtraces[i,:]=smooth_trace(chanftraces[i,:],smoothwindows)
for i in range(pyrtraces.shape[0]):
    smoothedpyrtraces[i,:]=smooth_trace(pyrtraces[i,:],smoothwindows)

      


pretime=0.5#s
posttime=3#S
prestim=int(pretime*fr)
poststim=int(posttime*fr)

optotrialarraysmoothedchands=np.zeros((smoothedchandtraces.shape[0],len(manualchandelierpeaks),prestim+poststim))
optotrialarraysmoothedpyr=np.zeros((smoothedpyrtraces.shape[0],len(manualchandelierpeaks),prestim+poststim))

trialtimevector=np.linspace(-pretime,posttime,prestim+poststim)

colors=['k','b','r']
    

f,ax=plt.subplots(1)
f.tight_layout()
ax.imshow(smoothedchandtraces[:5,:],aspect='auto')
for j in range(len(manualchandelierpeaks)):
    ax.axvline(x=framevector[manualchandelierpeaks[j]],c='y')  



f,ax=plt.subplots(1)
f.tight_layout()
ax.imshow(smoothedpyrtraces,aspect='auto')
for j in range(len(manualchandelierpeaks)):
    ax.axvline(x=framevector[manualchandelierpeaks[j]],c='y')  


corr = np.correlate(a=sig1, v=sig2)

   
for i,opto  in enumerate(manualchandelierpeaks):
    for j in range(optotrialarraysmoothedchands.shape[0]):
        optotrialarraysmoothedchands[j,i,:]=smoothedchandtraces[j,opto-prestim:opto+poststim]
    for k in range(optotrialarraysmoothedpyr.shape[0]):
        optotrialarraysmoothedpyr[k,i,:]=smoothedpyrtraces[k,opto-prestim:opto+poststim]



f,ax=plt.subplots(1)
f.tight_layout()
ax.imshow(optotrialarraysmoothedchands[1,:,:],aspect='auto',extent=[trialtimevector[0], trialtimevector[-1], 0, optotrialarraysmoothedchands[1,:,:].shape[0]])

import scipy.stats as stats
chandcorrelations=np.zeros([6]+list(optotrialarraysmoothedchands.shape[:2]))
chandpyrcorr=np.zeros([6]+list(optotrialarraysmoothedpyr.shape[:2]))
for j in  range(optotrialarraysmoothedchands.shape[0]):
    for k in  range(optotrialarraysmoothedpyr.shape[0]):
        for i in range(optotrialarraysmoothedpyr.shape[1]):
        
            r, p = stats.pearsonr(optotrialarraysmoothedchands[j,i,:], optotrialarraysmoothedpyr[k,i,:])

            chandpyrcorr[j,k,i] = r
            
for j in  range(optotrialarraysmoothedchands.shape[0]):
    for k in  range(optotrialarraysmoothedchands.shape[0]):
        for i in range(optotrialarraysmoothedchands.shape[1]):
        
            r, p = stats.pearsonr(optotrialarraysmoothedchands[j,i,:], optotrialarraysmoothedchands[k,i,:])

            chandcorrelations[j,k,i] = r


    
for j in range(chandpyrcorr.shape[2]):
    f,ax=plt.subplots()
    f.tight_layout()
    ax.imshow(chandpyrcorr[:3,:,j], aspect='auto',cmap='viridis')

    
f,ax=plt.subplots()
f.tight_layout()
ax.imshow(meaqncorr[:3,:],aspect='auto',cmap='viridis')