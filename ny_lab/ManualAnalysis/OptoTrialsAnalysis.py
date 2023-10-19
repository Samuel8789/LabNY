# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:09:45 2023

@author: sp3660
"""

#optoanalysis
    

analysis=selected_analysis[0]['analysis']
full_data=selected_analysis[0]['full_data']

%matplotlib inline
#%%
analysis.photostim_stim_table_and_optanalysisrafatemp()

#fullltraces

deriv=lambda x:np.diff(x,prepend=x[0] )
rectified=lambda x:np.absolute(x)
#to do sacling and smoothing

# analysis.full_data['opto_analysis']


final_voltage_timestamps=full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['Locomotion'].index.values
speed_array=rectified(deriv(full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['Locomotion'].values))
photostim_array=full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['PhotoStim'].values
phototrig_array=full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['PhotoTrig'].values
fr=full_data['imaging_data']['Frame_rate']   

 
# reference_image=
# opto_coordinates=

# opto_parameters=


final_movie_timestamps=analysis.mov_timestamps_seconds['shifted']*1000

full_accepted_traces=full_data['imaging_data']['Plane1']['Traces']['demixed']
# cell_identity
# opto_cell_caiman_ids=
opto_cell_accpeted_indexes=analysis.array_ids
# non_opto_cell_caiman_ids
non_opto_cell_accepted_indexes=analysis.non_opto_ids


# trial_times_transitions
# opto_time_transitions 

#here I have to create a method for changing pre and post times to play with the analysis
#here i will also have to include a mtehod for scaling and normalizing the traces to comapre
# pre_trial_time
# post_trial_time

trial_dissected_traces_array=analysis.fulloptotrialarray
dissected_array_movie_timestamps=analysis.trialtimevector

trial_dissected_signals_array=analysis.speedtrialrarray
dissected_array_voltagetimestamps=analysis.trialspeedtimevector

#this importnat to do as soon as posible to compare the correlations in these times with correlations in opto times   
manual_locomotion_transtions=[[1184,1331],[1884,2120],[2238,2258],[3174,3240],[3378,3597],[4000,4038],[4611,4697],[5593,5754],[6157,6333],[6480,6543],]
manual_locomotion_transtions=[[int(k*1e2) for k in i] for i in manual_locomotion_transtions]
manual_locomotion_time_transtions=[ final_voltage_timestamps[i] for i in manual_locomotion_transtions]


locomotion_length=[loc[1]-loc[0] for loc in manual_locomotion_time_transtions]




maxlocomotionbout=int(max(locomotion_length))
maxlocomotionboutframes=int(maxlocomotionbout/1000*fr)

    



smoothwindows=10
pretime=10#s
posttime=10#S
prestim=int(pretime*fr)
poststim=int(posttime*fr)        
prestimspeed=int(pretime*1000)
poststimspeed=int(posttime*1000)  

trialtimevector=np.linspace(-pretime,maxlocomotionbout/1000+posttime,prestim+poststim+maxlocomotionboutframes)
trialspeedtimevector=np.linspace(-pretime,posttime+maxlocomotionbout/1000,prestimspeed+poststimspeed+maxlocomotionbout)


fulllocotrialarray=np.zeros((full_accepted_traces.shape[0],len(manual_locomotion_transtions),prestim+poststim+maxlocomotionboutframes))
for cell in range(full_accepted_traces.shape[0]):
    for loco_idx in range(len(manual_locomotion_transtions)):
            start=manual_locomotion_transtions[loco_idx][0]
            start=int(np.argmin(np.abs(final_movie_timestamps-start)))
            fulllocotrialarray[cell,loco_idx,:]=full_accepted_traces[cell,start-prestim:start+poststim+maxlocomotionboutframes]


fulllocospeedtrialarray=np.zeros((len(manual_locomotion_transtions),prestimspeed+poststimspeed+maxlocomotionbout))
for loco_idx in range(len(manual_locomotion_transtions)):
        fulllocospeedtrialarray[loco_idx,:]=speed_array[manual_locomotion_transtions[loco_idx][0]-prestimspeed:manual_locomotion_transtions[loco_idx][0]+poststimspeed+maxlocomotionbout]

for cell in opto_cell_accpeted_indexes:
    f,ax=plt.subplots(len(manual_locomotion_transtions))
    for i in range(len(manual_locomotion_transtions)):
        ax[i].plot(trialspeedtimevector,fulllocospeedtrialarray[i,:])
        ax[i].plot(trialtimevector,analysis.smooth_trace(fulllocotrialarray[cell,i,:],smoothwindows))

    



    
    '''
    THings to plot
        Full opto traces with optosignals and locomotion
        Big figure, raster with all opto with trials indicated and as many pyramidal as can be vissible and locomotion
        single trial plots for every opto cell single cell with that cels activity and locomotion
        single trial plots for every opt cell with all chandleier cellscels activity and locomotion
        mean trials (with grayed out signal) for all  opto cell in an optocell by opto cell array

    '''
   
    
    #%%
    
   