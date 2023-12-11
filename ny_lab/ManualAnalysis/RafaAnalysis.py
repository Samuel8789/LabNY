# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:18:45 2023

@author: sp3660
"""
import matplotlib.pyplot as plt
import scipy
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import caiman as cm
# script to prepare figure for data grant 230420 base on rafas meeting 230405
#daSATES 230321_SPRB_3MinTest_25x_920_51020_63075_with-001_Plane1_Gree,loaded with app dataset focus
from matplotlib.patches import Rectangle
import scipy.signal as sg
import scipy.stats as st


def smooth_trace(trace,window):
    framenumber=len(trace)
    frac=window/framenumber
    filtered = lowess(trace, np.arange(framenumber), frac=frac)
    
    return filtered[:,1]
dtset=analysis.calcium_datasets[list(analysis.calcium_datasets.keys())[0]]
dtset.most_updated_caiman.CaimanResults_object.load_all()
f,ax=plt.subplots(2)
for i in range(2):
    trace=dtset.most_updated_caiman.CaimanResults_object.raw[i,:]
    ax[0].plot(trace)
    ax[1].plot(smooth_trace(trace,10))
    
    


    #%% manually getting optovoltageindexes
    
volt=dtset.associated_aquisiton.voltage_signal_object
dtset.most_updated_caiman.CaimanResults_object.load_caiman_hdf5_results()
cnm=dtset.most_updated_caiman.CaimanResults_object.cnm.estimates
cnm.view_components()
full_raw=cnm.C+cnm.YrA

chandeliers=np.array([ 3, 12, 15, 18])
chandeliers=np.array([ 0,])


chandeliertraces=full_raw[chandeliers,:]


sig=volt.voltage_signals_dictionary
sig['PhotoStim']
sig['PhotoTrig'].plot()
    
phototriggers=sig['PhotoTrig'].values.flatten()
photosignals=sig['PhotoStim'].values.flatten()
timestamps=dtset.associated_aquisiton.metadata_object.timestamps['Plane1']

    

def detect_opto_trigg():
    
        
    plt.plot(phototriggers)
    dif=np.diff(phototriggers)
    median=sg.medfilt(dif, kernel_size=1)
    rounded=np.round(median)
    plt.plot(rounded)
    uptransitions_medina_rounded=np.where(rounded>4.5)[0]
    downtransitions_medina_rounded=np.where(rounded<-4.5)[0]
    
    segments_indexes=[]
    segments=[]

    pad=5
    trial_number=10
    reps=5
    cellnumber=3
    
    for i in range(trial_number):
        if i!=9:
            segment=np.arange(uptransitions_medina_rounded[i]-pad,uptransitions_medina_rounded[i+1]+pad)
            segments_indexes.append(segment)
            segments.append(photosignals[segment])
        else:
            segment=np.arange(uptransitions_medina_rounded[i]-pad,uptransitions_medina_rounded[i]+len(segments_indexes[i-1])+pad)
            segments_indexes.append(segment)
            segments.append(photosignals[segment])
            
    
    f,axs=plt.subplots(5,2)
    for i in range(trial_number):
        row = i // 2  # determine the row index based on the iteration index
        col = i % 2  
        axs[row,col].plot(segments_indexes[i],segments[i])
        
        
        
    photostim_indexes_array=np.zeros([cellnumber,trial_number,reps,2])  
    for j in range(trial_number):
        trial=segments[j]
        dif=np.diff(trial)
        median=sg.medfilt(dif, kernel_size=1)
        rounded=np.round(median)
        plt.plot(rounded)
        uptransitions_medina_rounded=np.where(rounded>4.5)[0]+1
        downtransitions_medina_rounded=np.where(rounded<-4.5)[0]+1
       
        
        celluptransitions=[]
        celldowntransitions=[]
    
        for i in range(cellnumber):
    
            celluptransitions.append(  uptransitions_medina_rounded[i*reps:(i+1)*reps])
            celldowntransitions.append(  downtransitions_medina_rounded[i*reps:(i+1)*reps])
    
            celluporiginalindex=np.take(segments_indexes[j], celluptransitions[i])
            celldownoriginalindex=np.take(segments_indexes[j], celldowntransitions[i])

        
            photostim_indexes_array[i,j,:,0]=celluporiginalindex
            photostim_indexes_array[i,j,:,1]=celldownoriginalindex
            
        

    f,ax=plt.subplots()
    ax.plot(photosignals)
    colors=['r','b','k']
    shape=['x','o','v','^','<']
    for cell in range(cellnumber):
        for trial in range(trial_number):
            for rep in range(reps):
                x=int(photostim_indexes_array[cell,trial,rep,0])
                y=photosignals[x]
  
                ax.plot(x,y,color=colors[cell],marker= shape[rep])


    resampler=  lambda t: (np.abs(np.array(timestamps) - t/1000)).argmin() 
    vf = np.vectorize(resampler)
    phototimestamps=vf(photostim_indexes_array)

   
    cell1=phototimestamps[0,:,:,0]
    estimatedoptotimes=cell1[:,0]
    estimatedoptotimesallchands=phototimestamps[:,:,0,0]
    
   
    plt.rcParams["figure.figsize"] = [16, 5]
    plt.rcParams["figure.autolayout"] = True
    plt.close('all')
    smoothwindows=10
    fr=dtset.metadata.translated_imaging_metadata['FinalFrequency']
    framen=len(dtset.most_updated_caiman.CaimanResults_object.raw[1,:])
    lengthtime=framen/fr
    period=1/fr
    fulltimevector=np.arange(0,lengthtime,period)
    
    pretime=1#s
    posttime=3#S
    prestim=int(pretime*fr)
    poststim=int(posttime*fr)
    
    
    
    
    smoothwindows=10
    colors=['k','b','r']
    fig,ax=plt.subplots(3,sharex=True)
    fig.tight_layout()
    for i in range(len(chandeliers)):
        trace=chandeliertraces[i,:]
        ax[i].plot(fulltimevector,smooth_trace(trace,smoothwindows))
        ax[i].margins(x=0)
        for j in range(len(estimatedoptotimesallchands[0])):
            ax[i].axvline(x=fulltimevector[estimatedoptotimesallchands[i,j]],c=colors[i])  
        ax[i].set_xlabel('Time(s)')
        ax[i].set_ylabel('Activity(a.u.)')
        fig.suptitle('Optogenetic Stimulation of Chandelier Cells', fontsize=16)





    smoothedtraces=np.zeros_like(chandeliertraces)        
    for i in range(len(chandeliers)):
        smoothedtraces[i,:]=smooth_trace(chandeliertraces[i,:],smoothwindows)
    
    optotrialarraysmoothed=np.zeros((3,10,prestim+poststim))
    trialtimevector=np.linspace(-pretime,posttime,prestim+poststim)
    
    colors=['k','b','r']
    
    for l in range(len(chandeliers)):
        stimulatecdcell=l
        
        # f,ax=plt.subplots(5,2)
        for i,opto  in enumerate(estimatedoptotimesallchands[stimulatecdcell]):
            for j in range(len(chandeliers)):
        #         row = i // 2  # determine the row index based on the iteration index
        #         col = i % 2   # determine the column index based on the iteration index
                optotrialarraysmoothed[j,i,:]=smoothedtraces[j,opto-prestim:opto+poststim]
    
        stimnumber=5
        frequency=10#hz
        duration=20/1000#ms
        stimperiod=1/frequency #s
        isi=stimperiod-duration
        
        optotimes=np.arange(0,5*stimperiod,stimperiod)
    
       
        f,ax=plt.subplots(5,2)
        for i,opto  in enumerate(estimatedoptotimesallchands[stimulatecdcell]):
            for j in range(len(chandeliers)):
                row = i // 2  # determine the row index based on the iteration index
                col = i % 2   # determine the column index based on the iteration index
                ax[row, col].plot(trialtimevector,optotrialarraysmoothed[j,i,:])
                ax[row, col].axvline(x=0,c=colors[l])  
                # ax[row, col].set_ylim(-3,8)
                ax[row, col].margins(x=0)
                for m in optotimes:
                    ax[row, col].add_patch(Rectangle((m, 8), 0.01, 0.2,color='r'))
                ax[row, col].set_xlabel('Time(s)')
                ax[row, col].set_ylabel('Activity(a.u.)')
                
            f.suptitle('Single Trial Optogenetic Stimulation', fontsize=16)
    
    #%%

    [0,4,6,9]
# this come form the preliminary matlab analysis i did for rafa
estimatedoptotimes=np.array([1003,1422, 1647, 2110, 2221, 2923, 3525, 3666, 3806, 4209])
estimatedoptotimes=estimatedoptotimes-1

estimatedlocomotionbouts=np.array([1230, 1985, 2698,3325,  4010, 4500])
estimatedlocomotionbouts=estimatedlocomotionbouts-1
prestim=20
poststim=40
mc_movie=cm.load(dtset.mc_onacid_path)

#%%

background=mc_movie.mean(axis=(1,2))

backgroundcorrection=background[2786:2877]

firstcorrectionlenght=len(np.arange(994,1079))
secondtcorrectionlenght=len(np.arange(1432,1494))
thirdcorrectionlenght=len(np.arange(3575,3619))
fourthcorrectionlenght=len(np.arange(4214,4252))


background[994:1079]=np.random.choice(backgroundcorrection, firstcorrectionlenght)
background[1432:1494]=np.random.choice(backgroundcorrection, secondtcorrectionlenght)
background[3575:3619]=np.random.choice(backgroundcorrection, thirdcorrectionlenght)
background[4214:4252]=np.random.choice(backgroundcorrection, fourthcorrectionlenght)



f,ax=plt.subplots(2, sharex=True)
ax[0].plot(smooth_trace(background,10))
baseline=138
background[background<baseline]=0
background[background>baseline]=1
ax[1].plot(smooth_trace(background,10))

x=background
t=np.arange(len(background))
threshold=baseline




#%%

# Load the recorded voltage trace data
input_trace = background

# Define the threshold voltage value
threshold = 0.5

# Create a new output trace array initialized with all zeros
output_trace = np.zeros_like(input_trace)

# Define the parameters for the variable frequency oscillation
freq_range = (1, 10)  # Hz
amplitude = 1.0  # V
duration = 0.1  # seconds

# Iterate over the input trace array
for i, v in enumerate(input_trace):
    if v > threshold:
        # Set the corresponding segment in the output trace to the variable frequency oscillation
        freq = np.random.uniform(*freq_range)
        t = np.linspace(0, duration, int(duration * 1000), endpoint=False)
        oscillation = amplitude * np.sin(2 * np.pi * freq * t)
        output_trace[i:i+len(oscillation)] = oscillation
    else:
        # Set the corresponding segment in the output trace to zero
        output_trace[i] = 0

# Plot the input and output traces
plt.plot(input_trace, label='Input')
plt.plot(output_trace, label='Output')
plt.legend()
plt.show()

#%% aligning to the opto
stimnumber=5
frequency=20#hz
duration=20/1000#ms
stimperiod=1/frequency #s
isi=stimperiod-duration

optotimes=np.arange(0,5*stimperiod,stimperiod)

plt.rcParams["figure.figsize"] = [16, 5]
plt.rcParams["figure.autolayout"] = True
plt.close('all')
smoothwindows=10
fr=dtset.metadata.translated_imaging_metadata['FinalFrequency']
framen=len(dtset.most_updated_caiman.CaimanResults_object.raw[1,:])
lengthtime=framen/fr
period=1/fr
fulltimevector=np.arange(0,lengthtime,period)

f,ax=plt.subplots(6)    
for i in range(6):
    trace=dtset.most_updated_caiman.CaimanResults_object.raw[i,:]
    if i==3:
        ax[i].plot(fulltimevector,smooth_trace(trace,10),c='y')
    else:
        ax[i].plot(fulltimevector,smooth_trace(trace,10),c='g')

    for i,a in enumerate(ax):
        a.margins(x=0)
        if i<len(ax)-1:
            a.axis('off')
     
        elif i==len(ax)-1:
            
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.spines['left'].set_visible(False)
            a.get_yaxis().set_ticks([])
            
f,ax=plt.subplots(1)
f.tight_layout()
for i in range(3,6):
    trace=dtset.most_updated_caiman.CaimanResults_object.raw[i,:]
    ax.plot(fulltimevector,smooth_trace(trace,smoothwindows))
    
    ax.margins(x=0)
    for j in range(len(estimatedoptotimes)):
        ax.axvline(x=fulltimevector[estimatedoptotimes[j]],c='r')  
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Activity(a.u.)')
    f.suptitle('Optogenetic Stimulation of Chandelier Cells', fontsize=16)

   
smoothedtraces=np.zeros_like(dtset.most_updated_caiman.CaimanResults_object.raw)        
for i in range(6):
    smoothedtraces[i,:]=smooth_trace(dtset.most_updated_caiman.CaimanResults_object.raw[i,:],smoothwindows)
      
optotrialarraysmoothed=np.zeros((3,10,60))
trialtimevector=np.arange(-period*prestim,period*poststim,period)


# f,ax=plt.subplots(5,2)
for i,opto  in enumerate(estimatedoptotimes):
    for j in range(3,6):
#         row = i // 2  # determine the row index based on the iteration index
#         col = i % 2   # determine the column index based on the iteration index
        optotrialarraysmoothed[j-3,i,:]=smoothedtraces[j,opto-prestim:opto+poststim]
#         ax[row, col].plot(trialtimevector,optotrialarraysmoothed[j-3,i,:])

#         ax[row, col].axvline(x=0,c='r')  
#         ax[row, col].set_ylim(-1,3)
#         ax[row, col].margins(x=0)
#         ax[row, col].set_xlabel('Time(s)')
#         ax[row, col].set_ylabel('Activity(a.u.)')
#         for m in optotimes:
#             ax[row, col].add_patch(Rectangle((m, 2.8), 0.01, 0.2))
               
   # 

            
#plotting all trials and mean for chandelier
# fullmean=optotrialarraysmoothed.mean(axis=1)
# for j in range(3,6):
#     f,ax=plt.subplots(1)
#     for i,opto  in enumerate(estimatedoptotimes):
#         ax.plot(trialtimevector,optotrialarraysmoothed[j-3,i,:],c='k',alpha=0.2)
#         ax.plot(trialtimevector,fullmean[j-3,:],c='k')
#         ax.axvline(x=0,c='r')  
#         ax.set_ylim(-1,3)
#         ax.margins(x=0)
        
#     for m in optotimes:
#         ax.add_patch(Rectangle((m, 2.8), 0.01, 0.2))
        
meanactivations=optotrialarraysmoothed[:,[0,4,6,9],:].mean(axis=1)
meanalocomotion=optotrialarraysmoothed[:,7,:]
meannonactivations=optotrialarraysmoothed[:,[1,2,3,5,8],:].mean(axis=1)

# f,ax=plt.subplots(3)
# for i in range(3):
#     ax[0].plot(trialtimevector,meanactivations[i,:])
#     ax[1].plot(trialtimevector,meannonactivations[i,:])  
#     ax[2].plot(trialtimevector,meanalocomotion[i,:])
#     for j in [0,4,6,9]:
#         ax[0].plot(trialtimevector,optotrialarraysmoothed[i,j,:],alpha=0.2)
#     for j in [7]:
#         ax[2].plot(trialtimevector,optotrialarraysmoothed[i,j,:],alpha=0.2)  
#     for j in [1,2,3,5,8]:
#         ax[1].plot(trialtimevector,optotrialarraysmoothed[i,j,:],alpha=0.2)

#     for n in range(3):
#         ax[n].margins(x=0)
#         ax[n].set_ylim(-1,3)
#         ax[n].axvline(x=0,c='r')  
#         for m in optotimes:
#             ax[n].add_patch(Rectangle((m, 2.8), 0.01, 0.2))
#         ax[n].set_xlabel('Time(s)',fontsize=20)
#         ax[n].set_ylabel('Activity(a.u.)')
#     ax[0].set_title('Responsive Trials')
#     ax[1].set_title('Unresponsive Trials')
#     ax[2].set_title('Running Trials')

            
#     f.suptitle('Trial segmented PSTH', fontsize=25)
        

        
 #baseline substractes       
        
optotrialarraysmoothedDF=np.zeros_like(optotrialarraysmoothed)
meanbaselinesmoothed=optotrialarraysmoothed[:,:,0:20].mean(axis=2)
for i in range(3):
    for j in range(10):
        optotrialarraysmoothedDF[i,j,:]=(optotrialarraysmoothed[i,j,:]-meanbaselinesmoothed[i,j])


f,ax=plt.subplots(5,2)
for i,opto  in enumerate(estimatedoptotimes):
    for j in range(3,6):
        row = i // 2  # determine the row index based on the iteration index
        col = i % 2   # determine the column index based on the iteration index
        ax[row, col].plot(trialtimevector,optotrialarraysmoothedDF[j-3,i,:])
        ax[row, col].axvline(x=0,c='r')  
        ax[row, col].set_ylim(-1,3)
        ax[row, col].margins(x=0)
        for m in optotimes:
            ax[row, col].add_patch(Rectangle((m, 2.8), 0.01, 0.2,color='r'))
        ax[row, col].set_xlabel('Time(s)')
        ax[row, col].set_ylabel('Activity(a.u.)')
        
    f.suptitle('Single Trial Optogenetic Stimulation', fontsize=16)

               
#plotting all trials and mean for chandelier
fullmeandf=optotrialarraysmoothedDF.mean(axis=1)
cells=['Chand','Pyr1','Pyr2']
for j in range(3,6):
    f,ax=plt.subplots(1)
    for i,opto  in enumerate(estimatedoptotimes):
        ax.plot(trialtimevector,optotrialarraysmoothedDF[j-3,i,:],c='k',alpha=0.2)
        ax.plot(trialtimevector,fullmeandf[j-3,:],c='k')
        ax.axvline(x=0,c='r')  
        ax.set_ylim(-1,3)
        ax.margins(x=0)
        
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Activity(a.u.)')
        for m in optotimes:
            ax.add_patch(Rectangle((m, 2.8), 0.01, 0.2,color='r'))
        
    f.suptitle(f'Global Optostimulation PSTH {cells[j-3]}', fontsize=16)

                        
 
meanactivationsbasesub=optotrialarraysmoothedDF[:,[0,4,6,9],:].mean(axis=1)
meanalocomotionbasesub=optotrialarraysmoothedDF[:,7,:]
meannonactivationsbasesub=optotrialarraysmoothedDF[:,[1,2,3,5,8],:].mean(axis=1)       
for i in range(3):
    f,ax=plt.subplots(3)

    ax[0].plot(trialtimevector,meanactivationsbasesub[i,:])
    ax[1].plot(trialtimevector,meannonactivationsbasesub[i,:])  
    ax[2].plot(trialtimevector,meanalocomotionbasesub[i,:])
    for k in [0,4,6,9]:
        ax[0].plot(trialtimevector,optotrialarraysmoothedDF[i,k,:],c='k',alpha=0.2)
    for k in [7]:
        ax[2].plot(trialtimevector,optotrialarraysmoothedDF[i,k,:],c='k',alpha=0.2)  
    for k in [1,2,3,5,8]:
        ax[1].plot(trialtimevector,optotrialarraysmoothedDF[i,k,:],c='k',alpha=0.2)
    
    for j in range(3):
        ax[j].margins(x=0)
        ax[j].set_ylim(-1,3)
        ax[j].axvline(x=0,c='r')  
        for m in optotimes:
            ax[j].add_patch(Rectangle((m, 2.8), 0.01, 0.2,color='r'))
        
        ax[j].set_xlabel('Time(s)')
        ax[j].set_ylabel('Activity(a.u.)')
    ax[0].set_title('Responsive Trials')
    ax[1].set_title('Unresponsive Trials')
    ax[2].set_title('Running Trials')


        
    f.suptitle(f'Trial segmented PSTH {cells[i]}', fontsize=16)
    
#%% aligning to the locomotion

f,ax=plt.subplots(1)
for i in range(3,6):
    trace=dtset.most_updated_caiman.CaimanResults_object.raw[i,:]
    ax.plot(smooth_trace(trace,10))
    for j in range(len(estimatedlocomotionbouts)):
        ax.axvline(x=estimatedlocomotionbouts[j],c='r')  
   
smoothedtraces=np.zeros_like(dtset.most_updated_caiman.CaimanResults_object.raw)        
for i in range(6):
    smoothedtraces[i,:]=smooth_trace(dtset.most_updated_caiman.CaimanResults_object.raw[i,:],10)
      
    
locomotiontrialarraysmoothed=np.zeros((3,6,60))
locomotiontrialarraydenoised=np.zeros((3,6,60))

f,ax=plt.subplots(3,2)
for i,opto  in enumerate(estimatedlocomotionbouts):

    for j in range(3,6):
        row = i // 2  # determine the row index based on the iteration index
        col = i % 2   # determine the column index based on the iteration index
        locomotiontrialarraysmoothed[j-3,i,:]=smoothedtraces[j,opto-prestim:opto+poststim]

        ax[row, col].plot(locomotiontrialarraysmoothed[j-3,i,:])

        
            
        ax[row, col].axvline(x=prestim,c='r')  
        ax[row, col].set_ylim(-1,5)
        
            
# locomotiontrialarraysmoothedDF=np.zeros_like(locomotiontrialarraysmoothed)
# locomotiontrialarraydenoisedDF=np.zeros_like(locomotiontrialarraydenoised)
# meanbaselinesmoothed=locomotiontrialarraysmoothed[:,:,0:20].mean(axis=2)
# meanbaselinedenoised=locomotiontrialarraydenoisedDF[:,:,0:20].mean(axis=2)


# for i in range(3):
#     for j in range(5):
#         locomotiontrialarraysmoothedDF[i,j,:]=(locomotiontrialarraysmoothed[i,j,:]-meanbaselinesmoothed[i,j])/meanbaselinesmoothed[i,j]
#         locomotiontrialarraydenoisedDF[i,j,:]=(locomotiontrialarraydenoised[i,j,:]-meanbaselinedenoised[i,j])/meanbaselinedenoised[i,j]
        
# for k in range(2):
#     f,ax=plt.subplots(3,2)
#     for i,opto  in enumerate(estimatedlocomotionbouts):
#         for j in range(3,6):
#             row = i // 2  # determine the row index based on the iteration index
#             col = i % 2   # determine the column index based on the iteration index
#             if k==0 :
#                 ax[row, col].plot(locomotiontrialarraysmoothedDF[j-3,i,:])
#             else:
#                 ax[row, col].plot(locomotiontrialarraydenoisedDF[j-3,i,:])
#             ax[row, col].axvline(x=prestim,c='r')  
#             # ax[row, col].set_ylim(-1,3)


locomeanactivations=locomotiontrialarraysmoothed[:,:,:].mean(axis=1)

f,ax=plt.subplots(2)
for i in range(6):
    ax[0].plot(locomeanactivations[i,:])
    ax[1].plot(meanactivations[i,:])

    ax[1].axvline(x=20,c='r')  
    ax[0].axvline(x=20,c='r')  
    ax[0].set_ylim(-1,4)
    ax[1].set_ylim(-1,4)
    
    #%% making some scatter for rafa
    
                
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import statsmodels.api as sm

    
    responsivetrials=[0,4,6,9]
    locomotiontrials=[7]
    nonresponsivetrials=[1,2,3,5,8]
    
    trislstruct=[responsivetrials,nonresponsivetrials,locomotiontrials]
    meantrislstructsubs=[meanactivationsbasesub,meannonactivationsbasesub,meanalocomotionbasesub]
    meantrislstruct=[meanactivations,meannonactivations,meanalocomotion]

    colors=['r','b','y','tab:orange','k']
    shape=['x','o','v','^','<']
    legend=['Pyr1','Pyr2']

    f,ax=plt.subplots(3)
    line_handles=[]
    for m in range(3):
        for j in range(1,3):
            activity=optotrialarraysmoothed[:,trislstruct[m],:]
            for i in range(activity.shape[1]):
                scatter=ax[m].scatter(activity[0,i,:],activity[j,i,:],color=colors[i],marker=shape[j-1])
                line_handles.append(scatter)
                ax[m].set_ylim(-1,2.5)               
            ax[0].set_title('Responsive Trials')
            ax[1].set_title('Unresponsive Trials')
            ax[2].set_title('Running Trials')            
            
            
        # ax[0].legend(,
        #    ,
        #    scatterpoints=1,
        #    loc='lower left',
        #    ncol=3,
        #    fontsize=8)
            
    f.suptitle(f'Trial segmented Scatter Raw', fontsize=16)
    

  

    f,ax=plt.subplots(3)
    for m in range(3):
        for j in range(1,3):
            activity=meantrislstruct[m]
            ax[m].scatter(activity[0,:],activity[j,:],color=colors[j-1])
            ax[m].set_ylim(-0.6,1.5)
            ax[m].set_xlim(-0.5,2)
            
            x=activity[0,:].reshape((-1, 1))
            y=activity[j,:]
            model = LinearRegression().fit(x,y)
            r_sq = model.score(x, y)
            print(f"coefficient of determination: {r_sq}")

            x_new=np.linspace(-0.5,2).reshape((-1, 1))
            y_new = model.predict(x_new)

            ax[m].plot(x_new,y_new,color=colors[j-1])
            ax[m].set_xlabel('Time(s)')
            ax[m].set_ylabel('Activity(a.u.)')
            ax[m].legend(legend)
            x = sm.add_constant(x)
            #fit linear regression model
            model = sm.OLS(y, x).fit()            
            #view model summary
            print(model.summary())
    ax[0].set_title('Responsive Trials')
    ax[1].set_title('Unresponsive Trials')
    ax[2].set_title('Running Trials')
    f.suptitle(f'Mean Trial segmented Scatter Raw', fontsize=16)
    

    f,ax=plt.subplots(3)
    for m in range(3):
        for j in range(1,3):
            activity=optotrialarraysmoothedDF[:,trislstruct[m],:]
            for i in range(activity.shape[1]):
                ax[m].scatter(activity[0,i,:],activity[j,i,:],color=colors[i],marker=shape[j-1])
                ax[m].set_ylim(-1,2.5)                
            ax[0].set_title('Responsive Trials')
            ax[1].set_title('Unresponsive Trials')
            ax[2].set_title('Running Trials')   
    f.suptitle(f'Trial segmented Scatter Substracted', fontsize=16)


    f,ax=plt.subplots(3)
    for m in range(3):
        for j in range(1,3):
            activity=meantrislstructsubs[m]
            ax[m].scatter(activity[0,:],activity[j,:],color=colors[j-1])
            ax[m].set_ylim(-0.5,1)
            ax[m].set_xlim(-0.5,2.1)
            
            x=activity[0,:].reshape((-1, 1))
            y=activity[j,:]
            model = LinearRegression().fit(x,y)
            r_sq = model.score(x, y)
            print(f"coefficient of determination: {r_sq}")

            x_new=np.linspace(-0.5,2).reshape((-1, 1))
            y_new = model.predict(x_new)
            ax[m].plot(x_new,y_new,color=colors[j-1])
            
            ax[m].set_xlabel('Time(s)')
            ax[m].set_ylabel('Activity(a.u.)')
            ax[m].legend(legend) 
    ax[0].set_title('Responsive Trials')
    ax[1].set_title('Unresponsive Trials')
    ax[2].set_title('Running Trials')
    f.suptitle(f'Mean Trial segmented Scatter Substracted', fontsize=16)
