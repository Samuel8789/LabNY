# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 17:12:41 2023

@author: sp3660
"""
from sklearn.preprocessing import normalize
from scipy.linalg import norm
from scipy import signal
from scipy import stats
full_data
analysis
import scipy.stats as stats


plt.plot(full_data['voltage_traces']['Speed'])
plt.plot(full_data['voltage_traces']['VisStim'])

cells=[2,3,5,6]
# activity_arrays= analysis.get_raster_with_selections(trace_type,plane,selected_cells, paradigm)

trace_type='mcmc_scored_binary'
paradigm='Movie1'
# analysis.plot_sliced_raster(trace_type,plane,cells,paradigm)
analysis.plot_sliced_raster(trace_type,plane,cells,'Full')
#%%
timex=np.linspace(0,4000,113000)

cells=[2,3,5,6]
ac=full_data['imaging_data']['Plane1']['Traces']['denoised']
# f,ax =plt.subplots(len (cells)+1)
f,ax =plt.subplots(len (cells)+1)

ax[-1].plot(timex,full_data['voltage_traces']['Speed'],'b')
# ax[-1].plot(timex, full_data['voltage_traces']['VisStim'])

# for cell in range(len (ac)):
#     ax[cell+1].plot(timex,ac[cell,:])
    
for i,cell in enumerate(cells):
    # ax[i+1].plot(timex,ac[cell,:])
    ax[i].plot(timex,ac[cell,:])

   
   
for i,a in enumerate(ax):
    a.margins(x=0)
    if i<len(ax)-1:
        a.axis('off')
 
    elif i==len(ax)-1:
        
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.get_yaxis().set_ticks([])


    
for a in ax[:-1]:
    a.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
    
ax[-1].set_xlabel('Time(s)')
# f.supylabel('Cell Number')
# f.suptitle('Raster_scored_mcmc')

filename = os.path.join( 'test.pdf')
analysis.save_multi_image(filename)
#%%

ac=full_data['imaging_data']['Plane1']['Traces']['denoised']
f,ax =plt.subplots(2)
ax[0].plot(full_data['voltage_traces']['Speed'])
ax[1].plot(full_data['voltage_traces']['VisStim'])

#%% ploting activity in one ovie
cell=6
ac=full_data['imaging_data']['Plane1']['Traces']['denoised']
tt=full_data['visstim_info']['Paradigm_Indexes']['natural_movie_one_set_first']
ttl=full_data['visstim_info']['Paradigm_Indexes']['natural_movie_one_set_last']


inti=analysis.signals_object.transitions_dictionary['natural_movie_one_set_first']
endi=analysis.signals_object.transitions_dictionary['natural_movie_one_set_last']
mov=analysis.signals_object.rounded_vis_stim['Prairire']['VisStim'][inti:endi]

len(mov)
len(ac[cell,tt:ttl])
x1=np.linspace(0,len(mov),len(ac[4,tt:ttl]))

x2=np.linspace(0,len(mov),len(mov))

activity=full_data['imaging_data']['Plane1']['Traces']['mcmc_smoothed'][cell,:]
activity2=full_data['imaging_data']['Plane1']['Traces']['dfdt_smoothed'][cell,:]
bin1=full_data['imaging_data']['Plane1']['Traces']['mcmc_scored_binary'][cell,:]
bin2=full_data['imaging_data']['Plane1']['Traces']['dfdt_binary'][cell,:]

f,ax =plt.subplots(7)
ax[2].plot(x1,ac[cell,tt:ttl])
ax[1].plot(x1,full_data['voltage_traces']['Speed'][tt:ttl])
ax[0].plot(x2,mov)
ax[3].plot(x1,activity[tt:ttl])
ax[4].plot(x1,bin1[tt:ttl])
ax[5].plot(x1,activity2[tt:ttl])
ax[6].plot(x1,bin2[tt:ttl])
for a in ax:
    a.margins(x=0)
    
ax[-1].set_xlabel('Time(s)')
for a in ax[:-1]:
    a.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
labs=ax[-1].get_xticklabels()
# newlab=[str(int(la.get_text())/1000) for la in labs]
# ax[-1].set_xticklabels(newlab)
f.suptitle('Natural Movies Activity Extraction')
plt.tight_layout()

# ploting activity in one session of images
ac=full_data['imaging_data']['Plane1']['Traces']['denoised']
tt=full_data['visstim_info']['Paradigm_Indexes']['third_images_set_first']
ttl=full_data['visstim_info']['Paradigm_Indexes']['third_images_set_last']


inti=analysis.signals_object.transitions_dictionary['third_images_set_first']
endi=analysis.signals_object.transitions_dictionary['third_images_set_last']
mov=analysis.signals_object.rounded_vis_stim['Prairire']['VisStim'][inti:endi]

len(mov)
len(ac[cell,tt:ttl])
x1=np.linspace(0,len(mov),len(ac[4,tt:ttl]))

x2=np.linspace(0,len(mov),len(mov))

activity=full_data['imaging_data']['Plane1']['Traces']['mcmc_smoothed'][cell,:]
activity2=full_data['imaging_data']['Plane1']['Traces']['dfdt_smoothed'][cell,:]
bin1=full_data['imaging_data']['Plane1']['Traces']['mcmc_scored_binary'][cell,:]
bin2=full_data['imaging_data']['Plane1']['Traces']['dfdt_binary'][cell,:]

f,ax =plt.subplots(7)
ax[2].plot(x1,ac[cell,tt:ttl])
ax[1].plot(x1,full_data['voltage_traces']['Speed'][tt:ttl])
ax[0].plot(x2,mov)
ax[3].plot(x1,activity[tt:ttl])
ax[4].plot(x1,bin1[tt:ttl])
ax[5].plot(x1,activity2[tt:ttl])
ax[6].plot(x1,bin2[tt:ttl])
for a in ax:
    a.margins(x=0)
ax[-1].set_xlabel('Time(s)')
        
for a in ax[:-1]:
    a.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
labs=ax[-1].get_xticklabels()
# newlab=[str(int(la.get_text())/1000) for la in labs]
# ax[-1].set_xticklabels(newlab)
f.suptitle('Natural Images Activity Extraction')
plt.tight_layout()


#%%






full_data['visstim_info']['Natural_Images']['stimulus_table']
full_data['visstim_info']['Static_Gratings']['stimulus_table']


activity=full_data['imaging_data']['Plane1']['Traces']['mcmc_smoothed'][0,:]
locomotion=full_data['voltage_traces']['Speed']

locomotion_normalized=(locomotion-locomotion.mean())/norm(locomotion-locomotion.mean())
activity_normalized=(activity-activity.mean())/norm(activity-activity.mean())
locomsignal=locomotion_normalized*np.dot(activity_normalized,locomotion_normalized)


f,ax=plt.subplots()
ax.plot(locomotion_normalized)
# ax.plot(activity_normalized)
ax.plot(locomsignal)

ax.plot(activity_normalized-locomsignal)



f,ax=plt.subplots()
# ax.plot(full_data['imaging_data']['Plane1']['Traces']['mcmc_raw'][0,:])
# ax.plot(full_data['imaging_data']['Plane1']['Traces']['mcmc_smoothed'][0,:])
ax.plot(full_data['imaging_data']['Plane1']['Traces']['dfdt_smoothed'][0,:])


f2,ax2=plt.subplots()
ax2.plot(full_data['imaging_data']['Plane1']['Traces']['mcmc_scored_binary'][0,:])
ax2.plot(full_data['imaging_data']['Plane1']['Traces']['mcmc_binary'][0,:])
ax2.plot(full_data['imaging_data']['Plane1']['Traces']['dfdt_binary'][0,:])




traces=full_data['imaging_data']['Plane1']['Traces']['mcmc_smoothed']
plt.imshow(traces,aspect='auto')

f,ax=plt.subplots()
ax.plot(stats.zscore(traces[2,:]))
ax.plot(stats.zscore(full_data['voltage_traces']['Speed']))


slope, intercept, r, p, se =stats.linregress(stats.zscore(traces[2,:]),stats.zscore(full_data['voltage_traces']['Speed']))
# print(result.intercept, result.intercept_stderr)


co=signal.correlate(stats.zscore(traces[2,:]),stats.zscore(full_data['voltage_traces']['Speed']))

for cell in range(traces.shape[0]):
    corr = signal.correlate(stats.zscore(traces[cell,:]),stats.zscore(full_data['voltage_traces']['Speed']))
    lags = signal.correlation_lags(len(traces[cell,:]), len(full_data['voltage_traces']['Speed']))
    # corr /= np.max(corr)
    fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, figsize=(4.8, 4.8))
    
    ax_orig.plot(stats.zscore(traces[cell,:]))
    
    ax_orig.set_title('Original signal')
    
    ax_orig.set_xlabel('Sample Number')
    
    ax_noise.plot(stats.zscore(full_data['voltage_traces']['Speed']))
    
    ax_noise.set_title('Signal with noise')
    
    ax_noise.set_xlabel('Sample Number')
    
    ax_corr.plot(lags, corr)
    
    ax_corr.set_title('Cross-correlated signal')
    
    ax_corr.set_xlabel('Lag')
    
    ax_orig.margins(0, 0.1)
    
    ax_noise.margins(0, 0.1)
    
    ax_corr.margins(0, 0.1)
    ax_corr.set_ylim(0,10000)
    
    fig.tight_layout()
    
    plt.show()


selected_images=full_data['visstim_info']['Natural_Images']['stimulus_table'][full_data['visstim_info']['Natural_Images']['stimulus_table']['Image_ID']==1]





#%%

pre_time=1000     #ms
evoke_time=500
stim_time=1000

pre_frames=np.ceil(pre_time/analysis.milisecond_period).astype(int)
evoke_frames=np.ceil(evoke_time/analysis.milisecond_period).astype(int)
stim_frames=np.ceil(stim_time/analysis.milisecond_period).astype(int)
evoked_ranges= np.arange(pre_frames,pre_frames+evoke_frames)

cell1=traces[0,:]
from numpy.random import Generator, PCG64
rng = Generator(PCG64(12345))
trace_types=['demixed', 'denoised', 
 'dfdt_raw',  'dfdt_smoothed', 'dfdt_binary',
 'foopsi_raw', 'foopsi_smoothed','foopsi_binary',
 'mcmc_raw','mcmc_smoothed','mcmc_binary','mcmc_scored_binary']
trace_type=trace_types[3]
traces=full_data['imaging_data']['Plane1']['Traces'][trace_type]
numberimages=len([int(i) for i in list(set(full_data['visstim_info']['Natural_Images']['stimulus_table']['Image_ID'].tolist()))])

trialaveraged=np.zeros([traces.shape[0],  len(np.arange(-pre_frames,stim_frames)), numberimages])
trialaveragedevoked=np.zeros([traces.shape[0],  len(np.arange(0,evoke_frames)), numberimages])
averagedevokedactivty=np.zeros([traces.shape[0], numberimages])
trialavergaeddfdt=np.zeros([traces.shape[0],  len(np.arange(-pre_frames,stim_frames)), numberimages])
trialnumer=[]
mintrila=42
samples=1000
distributions=np.zeros([traces.shape[0],numberimages,samples])
maxtrials=len(full_data['visstim_info']['Natural_Images']['stimulus_table'][full_data['visstim_info']['Natural_Images']['stimulus_table']['Image_ID']==0])
allrunningspeeds=np.zeros([traces.shape[0],maxtrials,  len(np.arange(-pre_frames,stim_frames)), numberimages])
for cell in range( traces.shape[0]):
    for image in range(numberimages):
        selected_images=full_data['visstim_info']['Natural_Images']['stimulus_table'][full_data['visstim_info']['Natural_Images']['stimulus_table']['Image_ID']==image]
        random_trials=full_data['visstim_info']['Natural_Images']['stimulus_table'][full_data['visstim_info']['Natural_Images']['stimulus_table']['Image_ID']!=image]
        
        for i in range(samples):
            randomtrials=np.sort(rng.integers(0,len(random_trials),mintrila))
            onsets=random_trials.iloc[randomtrials]['start'].values
            ranges=onsets[:,None] + np.arange(-29,29)
            trialactivity=traces[cell,ranges[:,:]]
            
            averagedtrace=trialactivity.mean(0)
            averagedactivity=averagedtrace[evoked_ranges].mean()
            distributions[cell,image,i]=averagedactivity

        onsets=selected_images['start'].values
        ranges=onsets[:,None] + np.arange(-29,29)
        baseline_ranges=onsets[:,None] + np.arange(-pre_frames,0)
        
        
        running_speed=full_data['voltage_traces']['Speed'][ranges[:,:]]
        
        allrunningspeeds[cell,:len(selected_images),:,image]=running_speed
        #
        
        
        trialactivity=traces[cell,ranges[:,:]]
        
        averagedtrace=trialactivity.mean(0)
        averagedactivity=averagedtrace[evoked_ranges].mean()
        averagedevokedactivty[cell,image]=averagedactivity
        
        baseline_activity=traces[cell,baseline_ranges[:,:]].mean()
        
        dff=trialactivity-baseline_activity
        
        
        evoked_activity=dff.mean(0)[evoked_ranges]
        
        
        
        
        trialaveraged[cell,:,image]=dff.mean(0)
        trialavergaeddfdt[cell,:,image]=averagedtrace
        trialaveragedevoked[cell,:,image]=evoked_activity
        

intervals=np.zeros([traces.shape[0],numberimages,3])
for cell in range( traces.shape[0]):
    for image in range(numberimages):
        intervals[cell,image,0] = np.quantile(distributions[cell,image,:], 0.05) 
        intervals[cell,image,1] = np.quantile(distributions[cell,image,:], 0.95)
        intervals[cell,image,2]=averagedevokedactivty[cell,image]> intervals[cell,image,1]

#%%

cell=0
imageid=16

tt=allrunningspeeds[cell,:,:,:].max()
mean_running_speed=running_speed.mean(1)
# zscoredrunningspeed=(mean_running_speed-mean_running_speed.mean())/mean_running_speed.std()




denoised=full_data['imaging_data']['Plane1']['Traces']['denoised']
dfdtsmoothed=full_data['imaging_data']['Plane1']['Traces']['dfdt_smoothed']
mcmc_smoothed=full_data['imaging_data']['Plane1']['Traces']['mcmc_smoothed']




plt.close('all')
for i in range(7):
    number_of_signigficant=len(trialavergaeddfdt[i,:,np.where(intervals[i,:,2])[0]])

    meanvalues=pd.DataFrame(averagedevokedactivty[i,:])
    sortedindexes=meanvalues.sort_values(by=[0]).index.values
    f,axs=plt.subplots(1)
    pos=axs.imshow(trialavergaeddfdt[i,:,sortedindexes],aspect='auto', extent=[-1, 1, 1, 121], cmap='viridis',origin='lower')
    # pso=axs[1].imshow(stats.zscore(trialavergaeddfdt)[i,:,sortedindexes],aspect='auto', extent=[-1, 1, 1, 121], cmap='viridis',origin='lower')

    # for ax in axs:
    for j in   np.linspace(-1,1,9)  :
        axs.axvline(x = j, color = 'y',ls='--')
    axs.axvspan(0, 0.5, alpha=0.4, color='grey')
    axs.axhline(y = 121-number_of_signigficant, color = 'y',ls='--')

    axs.set_xlabel('Time(s)')

    f.colorbar(pos, ax=axs)
    # f.colorbar(pso, ax=axs[1])

    f.suptitle(f'Chandelier {int(i+1)}')
    
    
    axs.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        )
    axs.axes.get_yaxis().set_visible(False)

    
cell=0
imageid=16

for imageid in  np.where(intervals[cell,:,2])[0]:
    
    selected_images=full_data['visstim_info']['Natural_Images']['stimulus_table'][full_data['visstim_info']['Natural_Images']['stimulus_table']['Image_ID']==imageid]
    onsets=selected_images['start'].values
    ranges=onsets[:,None] + np.arange(-29,29)
    trialactivity=traces[cell,ranges[:,:]]
    from scipy.stats import sem
    
    f0,axss=plt.subplots(5)
    for i in range(len(trialactivity)):
        axss[0].plot(np.linspace(-1,1,pre_frames+stim_frames),denoised[cell,ranges[:,:]][i,:])
        axss[1].plot(np.linspace(-1,1,pre_frames+stim_frames),dfdtsmoothed[cell,ranges[:,:]][i,:])
        axss[2].plot(np.linspace(-1,1,pre_frames+stim_frames),mcmc_smoothed[cell,ranges[:,:]][i,:])
        axss[3].plot(np.linspace(-1,1,pre_frames+stim_frames),trialactivity[i,:])
        # axss[4].plot(np.linspace(-1,1,pre_frames+stim_frames),trialavergaeddfdt[i,:,imageid],'k')

        
    # for i in   np.linspace(-1,1,9)  :
    #     ax.axvline(x = i, color = 'y',ls='--')
    # ax.axvspan(0, 0.5, alpha=0.5, color='grey')

    
    
    f,ax=plt.subplots()
    for i in range(len(trialactivity)):
        ax.plot(np.linspace(-1,1,pre_frames+stim_frames),trialactivity[i,:])
        for i in   np.linspace(-1,1,9)  :
            ax.axvline(x = i, color = 'y',ls='--')
        ax.axvspan(0, 0.5, alpha=0.5, color='grey')
    
    f,ax=plt.subplots()
    ax.hist(distributions[cell,image,:])
    ax.axvline(averagedevokedactivty[cell,imageid])
    ax.axvspan(intervals[cell,imageid,0], intervals[cell,imageid,1], alpha=0.5, color='grey')
    
    
    for i in range(7):
        f,axs=plt.subplots(3)
        pos=axs[0].imshow(trialavergaeddfdt[i,:,imageid-1:imageid+2].T, aspect='auto', extent=[-1, 1, imageid+1, imageid-1], cmap='inferno')
        pos=axs[1].imshow(trialavergaeddfdt[i,:,:].T, aspect='auto', extent=[-1, 1, 121, 1], cmap='inferno')
        # for trial in range(traces[cell,ranges[:,:]].shape[0]):
        #     axs[2].plot(np.linspace(-1,1,pre_frames+stim_frames),traces[cell,ranges[trial,:]],color='grey')
        axs[2].plot(np.linspace(-1,1,pre_frames+stim_frames),trialavergaeddfdt[i,:,imageid],'k')
        axs[2].fill_between(np.linspace(-1,1,pre_frames+stim_frames), trialavergaeddfdt[i,:,imageid]+sem(trialavergaeddfdt[i,:,imageid], axis=0, ddof=1, nan_policy='propagate'),                          trialavergaeddfdt[i,:,imageid]-sem(trialavergaeddfdt[i,:,imageid], axis=0, ddof=1, nan_policy='propagate'),
                            alpha=0.3,
                            linestyle='dashdot', antialiased=True)
        axs[2].margins(x=0)
        
        for ax in axs:
            for j in   np.linspace(-1,1,9)  :
                ax.axvline(x = j, color = 'y',ls='--')
            ax.axvspan(0, 0.5, alpha=0.5, color='grey')
            f.colorbar(pos, ax=ax)
    
    
            cellimagesover=np.where(intervals[:,:,2])







"""
get an stimulus
get all trials for that stimuls
check average activity
do some linear regression with locomotiopn

"""