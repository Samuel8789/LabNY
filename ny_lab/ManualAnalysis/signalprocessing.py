# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:57:35 2023

@author: sp3660
"""

import caiman as cm
import os
import glob
import scipy.io as spio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import scipy.signal as sg
from scipy import signal
from scipy import interpolate

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

def load_full_csv(csv_path):
    voltage_signals_dictionary={}
    if csv_path:
        voltage_signals = pd.read_csv(csv_path)            
        for signal in voltage_signals.columns.tolist()[1:]:
            if 'Locomotion' in signal or  ' Locomotion' in signal:
                voltage_signals_dictionary['Locomotion']=voltage_signals[signal].to_frame()
            if 'VisStim' in signal or ' VisStim' in signal:
                voltage_signals_dictionary['VisStim']=voltage_signals[signal].to_frame()
            if 'LED' in signal or ' LED' in signal:
                voltage_signals_dictionary['LED']=voltage_signals[signal].to_frame()
            if 'PhotoDiode' in signal or 'Photo Diode' in signal :   
                voltage_signals_dictionary['PhotoDiode']=voltage_signals[signal].to_frame()
            if 'PhotoStim' in signal or ' Photostim' in signal or ' UncagiingPokels' in signal:   
                voltage_signals_dictionary['PhotoStim']=voltage_signals[signal].to_frame()
            if 'Optotrigger' in signal or ' Optotrigger' in signal:  
                voltage_signals_dictionary['PhotoTrig']=voltage_signals[signal].to_frame()
            if 'StartEnd' in signal or ' StartEnd' in signal or ' Start/End' in signal: 
                voltage_signals_dictionary['AcqTrig']=voltage_signals[signal].to_frame()
            if 'Frames' in signal:
                voltage_signals_dictionary['Frames']=voltage_signals[signal].to_frame()
            if 'Input2' in signal or 'Input 2' in signal:       
                voltage_signals_dictionary['Input2']=voltage_signals[signal].to_frame()
            if 'Input7' in signal or 'Input 7' in signal:
                voltage_signals_dictionary['Input7']=voltage_signals[signal].to_frame()
                
                
        voltage_signals_dictionary['Time']=pd.DataFrame(np.arange(voltage_signals.shape[0]), columns=['Time'])
        return voltage_signals_dictionary
    
def load_full_daq(extra_daq_path):
    voltage_signals_dictionary_daq={'Locomotion': pd.DataFrame({'Locomotion' : []}),
                                'VisStim':pd.DataFrame({'VisStim' : []}),
                                'LED':pd.DataFrame({'LED' : []}),
                                'PhotoDiode':pd.DataFrame({'PhotoDiode' : []}),
                                'Frames':pd.DataFrame({'Frames' : []}),
                                'Input2':pd.DataFrame({'Input2' : []}),
                                'Time':pd.DataFrame({'Time' : []}),
                                'Input7':pd.DataFrame({'Input7' : []}),
                                'PhotoStim':pd.DataFrame({'PhotoStim' : []}),
                                'PhotoTrig':pd.DataFrame({'PhotoTrig' : []}),   
                                'AcqTrig':pd.DataFrame({'AcqTrig' : []})
                                }


    mat=spio.loadmat(extra_daq_path)# load mat-file           
    mdata = mat['daq_data']  # variable in mat file 
    ndata = {n: mdata[n][0,0] for n in mdata.dtype.names}
    time_array=ndata['time']
    volt_array=ndata['voltage']
    fullsignals_extra_daq=np.hstack((time_array,volt_array)).T
    
    new_daq_keys=['Time', 'VisStim', 'Optopockels', 'Start/End', 'LED', 'PhotoTrigger', 'Locomotion']
    old_daq_keys=['Time', 'VisStim', 'Photodiode', 'Locomotion', 'LED/Frames', 'Optopockels',]
    if volt_array.shape[1]==5:
        daq_keys=old_daq_keys
    elif volt_array.shape[1]>=6:
        daq_keys=new_daq_keys

    for i, key in enumerate(daq_keys):
        if 'Locomotion' in key:
            voltage_signals_dictionary_daq['Locomotion']=voltage_signals_dictionary_daq['Locomotion'].assign(Locomotion=fullsignals_extra_daq[i,:].T.tolist())
        if 'VisStim' in key:
            voltage_signals_dictionary_daq['VisStim']=voltage_signals_dictionary_daq['VisStim'].assign(VisStim=fullsignals_extra_daq[i,:].T.tolist())
        if 'LED' in key:
            voltage_signals_dictionary_daq['LED']=voltage_signals_dictionary_daq['LED'].assign(LED=fullsignals_extra_daq[i,:].T.tolist()) 
        if 'Photodiode' in key:
            voltage_signals_dictionary_daq['PhotoDiode']=voltage_signals_dictionary_daq['PhotoDiode'].assign(PhotoDiode=fullsignals_extra_daq[i,:].T.tolist()) 
        if  'Optopockels' in key:
            voltage_signals_dictionary_daq['PhotoStim']=voltage_signals_dictionary_daq['PhotoStim'].assign(PhotoStim=fullsignals_extra_daq[i,:].T.tolist())
        if 'PhotoTrigger' in key:
            voltage_signals_dictionary_daq['PhotoTrig']=voltage_signals_dictionary_daq['PhotoTrig'].assign(PhotoTrig=fullsignals_extra_daq[i,:].T.tolist())
        if 'Start/End' in key:
            voltage_signals_dictionary_daq['AcqTrig']=voltage_signals_dictionary_daq['AcqTrig'].assign(AcqTrig=fullsignals_extra_daq[i,:].T.tolist()) 
        if 'Time' in key:
            voltage_signals_dictionary_daq['Time']=voltage_signals_dictionary_daq['Time'].assign(Time=fullsignals_extra_daq[i,:].T.tolist()) 
        if 'Frames' in key:
            voltage_signals_dictionary_daq['Frames']=voltage_signals_dictionary_daq['Frames'].assign(Frames=fullsignals_extra_daq[i,:].T.tolist()) 
            
    return voltage_signals_dictionary_daq
#%%loading from raw
pth=r'G:\Projects\TemPrairireSSH\20230328\Calibrations\SensoryStimulation\TestAcquisitions\230328_Test_Test_1z_3min_2x_opto_25x_920_51020_63075_with-000'
voltagerecordings=r'G:\Projects\TemPrairireSSH\20230328\Calibrations\SensoryStimulation\UnprocessedDaq'
visstiminfo=r'G:\Projects\TemPrairireSSH\20230328\Calibrations\SensoryStimulation\UnprocessedVisStim'

movie = cm.load(glob.glob(os.path.join(pth, "**Ch2**.tif")))
visstim=loadmat(glob.glob(visstiminfo+'\**.mat')[0])
csv_path=glob.glob(os.path.join(pth, "**.csv"))[0]
prairievoltagefinal=load_full_csv(csv_path)
voltagesfinal=load_full_daq(glob.glob(voltagerecordings+'\**.mat')[0])

#%% loading from acq object

movie = cm.load(glob.glob(os.path.join(acq.database_acq_raw_path, 'Ch2Green','plane1',"**Ch2**.tif")))
visstim=acq.mat
prairievoltagefinal=acq.voltage_signal_object.voltage_signals_dictionary
daqpath=acq.slow_storage_all_paths['voltage_signals_daq']
daq_file_path=glob.glob(daqpath+'\**.mat')[0]

voltagesfinal=load_full_daq(daq_file_path)

menamov=movie.mean(axis=(1, 2))    
#%% process vistim file


visstimdict={'is135':visstim['is135'],
             'isi_color':visstim['isi_color'],
             'isi_color_texture':visstim['isi_color_texture'],
             'ops':visstim['ops'],
             'opto':visstim['opto'],
             'full_info':'',  
    }

titles0=[visstim['full_info'][0][i] for i in range(5)]
visstiminfofull={}
for i in range(5):
    if i in range(0,4):
        visstiminfofull[titles0[i]]=visstim['full_info'][1][i]
    elif i==4 :
        titles1=[visstim['full_info'][1][i][0][j] for j in range(4)]
        trisldiact={}
        for k in range(4):
            trisldiact[titles1[k]]=visstim['full_info'][1][i][1][k]
        visstiminfofull[titles0[i]]=trisldiact

visstiminfofull['ParadigmDuration']=visstiminfofull['EndParadigmTime'][0]-visstiminfofull['StartParadigmTime']
visstiminfofull['Trials']['TrialTime']=visstiminfofull['Trials']['TrialEnd'][0]-visstiminfofull['Trials']['TrialStart'][0]
visstimdict['full_info']=visstiminfofull


#%% proces voltage files

labels=['Paradigm','Optopockels','StartEndTrigger','LED ON_OFF', 'Optotrigger','Locomotion' ]

signalstoplot=['VisStim', 'PhotoStim','AcqTrig','LED','PhotoTrig','Locomotion']
f,ax=plt.subplots()
for i in range(6):
    ax.plot(voltagesfinal[signalstoplot[i]],label=labels[i])
    
ax.legend()


plt.close('all')
alldelays=[]
for i in range(5):
    
    paradigm=sg.medfilt(np.squeeze(voltagesfinal[signalstoplot[i]]), kernel_size=1)
    paradigm2=sg.medfilt(np.squeeze(prairievoltagefinal[signalstoplot[i]]), kernel_size=1)
    
    np.arange(0,len(paradigm2),1)
    f,ax=plt.subplots()
    ax.plot(paradigm)
    ax.plot(paradigm2,'r')
    
    if i==2:
        x1=np.argwhere(np.diff(paradigm2)<-2).flatten()[0]
        
        y1=np.argwhere(np.diff(paradigm[:len(paradigm2)])<-2).flatten()[0]
        
        
    else:
        x1=np.argwhere(np.diff(paradigm2)>2).flatten()[0]
        
        y1=np.argwhere(np.diff(paradigm[:len(paradigm2)])>2).flatten()[0]
    
    volt_delay= y1 - x1
    
    paradigm2corrected= np.concatenate([np.zeros(volt_delay),paradigm2])


    alldelays.append(volt_delay)
    
    f,ax=plt.subplots()
    ax.plot(paradigm)
    ax.plot(paradigm2corrected,'r')
    
    
correctingpadding=np.arange(alldelays[0])
sumestampst=np.squeeze(voltagesfinal['Time']*1000+alldelays[0])
correctedtimestanps=np.concatenate([correctingpadding,sumestampst])
prairirestamps=np.squeeze(prairievoltagefinal['Time'].values)
movstamps=np.array(acq.metadata_object.timestamps['Plane1'])*1000 
    

ledsignal=sg.medfilt(np.squeeze(prairievoltagefinal[signalstoplot[3]]), kernel_size=1)



f = signal.resample(ledsignal, len(movstamps))
plt.plot(movstamps,sg.medfilt(f,kernel_size=5)/np.linalg.norm(sg.medfilt(f,kernel_size=5)))
# plt.plot(prairirestamps,ledsignal/np.linalg.norm(ledsignal),'r')
# plt.plot(movstamps,menamov/np.linalg.norm(menamov),'g')

def resample(x, factor, kind='linear'):
    n = int(np.floor(x.size / factor))
    f = interpolate.interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, n))     
from cycler import cycler
custom_cycler = (cycler(color=['c', 'm', 'y', 'k']) +
                 cycler(lw=[1, 2, 3, 4]))

f,ax=plt.subplots()
testimestamps=resample(prairirestamps, factor=650000/17495, kind='linear').squeeze()
plt.plot(movstamps,menamov/np.linalg.norm(menamov))

for i in range(6):
    
    if i==3:
        plt.plot(prairirestamps,sg.medfilt(np.squeeze(prairievoltagefinal[signalstoplot[i]]), kernel_size=1)/np.linalg.norm(sg.medfilt(np.squeeze(prairievoltagefinal[signalstoplot[i]]), kernel_size=1)))
        resampledled=resample(sg.medfilt(np.squeeze(prairievoltagefinal[signalstoplot[i]]), kernel_size=1), factor=650000/17495, kind='linear').squeeze()
        plt.plot(testimestamps,resampledled/np.linalg.norm(resampledled))


    elif i==4:
        plt.plot(prairirestamps,sg.medfilt(np.squeeze(prairievoltagefinal[signalstoplot[i]]), kernel_size=1)/np.linalg.norm(sg.medfilt(np.squeeze(prairievoltagefinal[signalstoplot[i]]), kernel_size=1)))
    else:
        plt.plot(prairirestamps,sg.medfilt(np.squeeze(prairievoltagefinal[signalstoplot[i]]), kernel_size=1)/np.linalg.norm(sg.medfilt(np.squeeze(prairievoltagefinal[signalstoplot[i]]), kernel_size=1)))





#%% final step is aligning the prairire vltage that starts eralier than the aquisition(when pressed, and before the triger, same as aligning the pprevious signals but at diffent time resolutin so will have to check timestamps)
# seems they are almost fully aligne dso review this to be more precise


paradigm=sg.medfilt(resampledled, kernel_size=1)
paradigm2=sg.medfilt(menamov, kernel_size=1)

f,ax=plt.subplots()
ax.plot(paradigm/np.linalg.norm(paradigm))
ax.plot(paradigm2/np.linalg.norm(paradigm2),'r')

if i==2:
    x1=np.argwhere(np.diff(paradigm2)<-2).flatten()[0]
    
    y1=np.argwhere(np.diff(paradigm[:len(paradigm2)])<-2).flatten()[0]
    
    
else:
    x1=np.argwhere(np.diff(paradigm2)>2).flatten()[0]
    
    y1=np.argwhere(np.diff(paradigm[:len(paradigm2)])>2).flatten()[0]

volt_delay= y1 - x1

paradigm2corrected= np.concatenate([np.zeros(volt_delay),paradigm2])


alldelays.append(volt_delay)

f,ax=plt.subplots()
ax.plot(paradigm)
ax.plot(paradigm2corrected,'r')

#%% here lets get the opto trigger traces and the pockels indexes
# i have to do the correction for the cut movie(make a clipped_timestamps using the file)
     
optotrigger_index_full_recording =np.zeros((1))
phototriggersignal=np.squeeze(voltagesfinal[signalstoplot[4]])
fix, ax=plt.subplots(1)
ax.plot( phototriggersignal)
    

temp=np.diff(np.around(sg.medfilt(phototriggersignal, kernel_size=1),1))
temp2=np.around(temp,3)

fix, ax=plt.subplots(1)
ax.plot(temp)
ax.plot(temp2,'r')
phototrigger_voltage_slice_filtered_rounded_corrected,phototrigger_diff_voltage_slice_filtered_rounded_corrected,phototrigger_diff_voltage_slice_filtered_rounded_corrected_rerounded,phototrigger_errors_pairs = correct_voltage_split_transitions(phototriggersignal)

initial_transitions_odd=np.argwhere(np.logical_and(phototrigger_diff_voltage_slice_filtered_rounded_corrected!=2, phototrigger_diff_voltage_slice_filtered_rounded_corrected<2))


movie_trial_starts=np.argwhere(np.logical_and(phototrigger_diff_voltage_slice_filtered_rounded_corrected!=2 , phototrigger_diff_voltage_slice_filtered_rounded_corrected>1)).flatten()
movie_trial_ends=np.argwhere(phototrigger_diff_voltage_slice_filtered_rounded_corrected<-2).flatten()

phototrigger_frame_indexes_by_trial=np.zeros([10,900,2])

fix, ax=plt.subplots(1)
ax.plot( phototrigger_diff_voltage_slice_filtered_rounded_corrected)
for i, start in enumerate(movie_trial_starts):
    if i==9:
       movie_trial_starts=np.insert(movie_trial_starts,10,len(phototrigger_diff_voltage_slice_filtered_rounded_corrected))
    
    movietrial=phototrigger_diff_voltage_slice_filtered_rounded_corrected[start:movie_trial_starts[i+1]]
    ups=np.argwhere(movietrial==2).flatten()+start
    down=np.argwhere(movietrial==-2).flatten()+start

    
    ups=np.insert(ups, 0, start)
    down=np.insert(down, 0, movie_trial_ends[i])
    if i!=9:
        down=np.append(down, movie_trial_starts[i+1])
    else:
        down=np.append(down, movie_trial_starts[-1]-1)
        
       
    phototrigger_frame_indexes_by_trial[i,:,0]=ups
    phototrigger_frame_indexes_by_trial[i,:,1]=down

  
    ax.plot(np.arange(start,movie_trial_starts[i+1]),phototrigger_diff_voltage_slice_filtered_rounded_corrected[start:movie_trial_starts[i+1]])
    ax.plot( ups, phototrigger_diff_voltage_slice_filtered_rounded_corrected[ups],'bo')
    ax.plot( down, phototrigger_diff_voltage_slice_filtered_rounded_corrected[down],'ko')

    framelengths= phototrigger_frame_indexes_by_trial[i,:,1]- phototrigger_frame_indexes_by_trial[i,:,0]





def correct_voltage_split_transitions(voltage_slice):
    # this is for transition that were split betwen 2 samples, I always get the inital transition to the sample with at tleast some 
    # voltage as voltages is send after the image and for the end transition i get the also the first as the image has chnaged before voltage change
    voltage_slice_filtered=sg.medfilt(voltage_slice, kernel_size=1)
    voltage_slice_filtered_rounded=np.around(voltage_slice_filtered, 1)
    
    voltage_slice_filtered_rounded_corrected=np.copy(voltage_slice_filtered_rounded)

    diff_voltage_slice_filtered_rounded= np.diff(voltage_slice_filtered_rounded)
    diff_voltage_slice_filtered_rounded_rerounded =np.around(diff_voltage_slice_filtered_rounded, 1)

    #correcting  voltage transitions betwen samples
    errors_pairs=[]
    for i in range(0, diff_voltage_slice_filtered_rounded_rerounded.size-2):
        if diff_voltage_slice_filtered_rounded_rerounded[i+1]!=0 and diff_voltage_slice_filtered_rounded_rerounded[i]!=0:
            errors_pairs.append((i+1, i+2))
            voltage_slice_filtered_rounded_corrected[i+1]=voltage_slice_filtered_rounded_corrected[i+2]
            voltage_slice_filtered_rounded_corrected[i+2]=voltage_slice_filtered_rounded_corrected[i+3]
            
            
    plt.plot(voltage_slice_filtered_rounded)
    plt.plot(voltage_slice_filtered_rounded_corrected)

                    
    diff_voltage_slice_filtered_rounded_corrected = np.diff(voltage_slice_filtered_rounded_corrected)
    diff_voltage_slice_filtered_rounded_corrected_rerounded =np.around(diff_voltage_slice_filtered_rounded_corrected, 1)    

    maxplots=8
    figures=int(np.ceil(len(errors_pairs)/maxplots))
    
    for n in range(figures):
        indexes=np.arange(n*maxplots,(n+1)*maxplots,1, 'int')
        if n==figures-1:
            indexes=np.arange(n*maxplots,len(errors_pairs),1, 'int')
        fig,axa=plt.subplots(len(indexes), sharex=True)
        for i in  range(len(indexes)) :  
            l1=axa[i].plot(diff_voltage_slice_filtered_rounded_rerounded[errors_pairs[indexes[i]][0]-10:errors_pairs[indexes[i]][0]+10])
            l2=axa[i].plot(diff_voltage_slice_filtered_rounded_corrected_rerounded[errors_pairs[indexes[i]][0]-10:errors_pairs[indexes[i]][0]+10])

        fig,axo=plt.subplots(len(indexes), sharex=True)
        for i  in range(len(indexes)) :  
            axo[i].plot(voltage_slice_filtered_rounded[errors_pairs[indexes[i]][0]-10:errors_pairs[indexes[i]][0]+10])
            axo[i].plot(voltage_slice_filtered_rounded_corrected[errors_pairs[indexes[i]][0]-10:errors_pairs[indexes[i]][0]+10])
      
    return voltage_slice_filtered_rounded_corrected, diff_voltage_slice_filtered_rounded_corrected, diff_voltage_slice_filtered_rounded_corrected_rerounded, errors_pairs













    
#%%
x1,x2=np.argwhere(np.diff(ledsignal)<-2).flatten()
f,ax=plt.subplots()
ax.plot(ledsignal)
ax.plot(np.diff(ledsignal),'r')
ax.plot(x1,np.diff(ledsignal)[x1],'xr')
ax.plot(x1,ledsignal[x1],'oy')
ax.plot(x2,np.diff(ledsignal)[x1],'xr')
ax.plot(x2,ledsignal[x1],'oy')





limits=np.argwhere(np.diff(menamov)<-1000).flatten()
f,ax=plt.subplots()
ax.plot(movstamps,menamov)
ax.plot(movstamps[:-1],np.diff(menamov),'r')
ax.plot(movstamps[:-1][limits],np.diff(menamov)[limits],'xr')
ax.plot(movstamps[limits],menamov[limits],'oy')
y1=limits[1]
y2=limits[-1]

f,ax=plt.subplots()
ax.plot(prairirestamps,ledsignal/np.linalg.norm(ledsignal))
ax.plot(movstamps,menamov/np.linalg.norm(menamov),'y')


ax.plot(movstamps[:-1],np.diff(menamov),'g')
ax.plot(prairirestamps[:-1],np.diff(ledsignal),'r')
normalizedData = data/np.linalg.norm(data)


ax.plot(prairirestamps[:-1][x1],np.diff(ledsignal)[x1],'xr')
ax.plot(prairirestamps[x1],ledsignal[x1],'oy')





scaling_factor = (y2-y1)/(x2-x1)

shift = np.round(y1 - (x1*scaling_factor))

paradigm2, scaling_factor, shift

shift = np.round(shift);

d1 = paradigm2.shape[0]
num_samp =d1
num_chan = 1

if scaling_factor > 1 or scaling_factor < 1:
       
    np.arange(num_samp)*scaling_factor+1
    x = ((1:num_samp)-1)*scaling_factor+1;
    xq = 1:1:num_samp*scaling_factor;
    scaled_trace = interp1(x,trace,xq);
    if num_chan == 1
        scaled_trace = scaled_trace';
    end

