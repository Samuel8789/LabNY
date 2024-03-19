#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:07:59 2024

@author: sp3660

tzi tzi data
"""
 
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import pandas as pd

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

#%%
stempth=Path(r'/home/sp3660/Documents/Projects/LabNY/0. DataFigures/Data From Other People/From TziTzi/ChC_Data_Elec_Visual_Stim/')
dtsets=['1_17Junio2023_chandelier,solo Estim, sin control','2_21Junio2023_Chandelier','4_21Agosto2023_chande','5_24Agosto2023_Chande','6_17Nov2013_Chande']
treatlists=[['Estim'],['Control', 'Estim'],['Control', 'Estim', 'Vstim', 'Vtim_Estim']]

alldtsets_dict={}
for toanalyze in range(len(dtsets)):
    # toanalyze=1
    
    dtset=dtsets[toanalyze]
    if toanalyze==4:
        toload=2
    elif toanalyze in [1,2,3]:
        toload=1
    else:
        toload=0
    treatlist=treatlists[toload]
    filedict={kk:{'data':[m for m in glob.glob(str(stempth / dtset / kk / '**.mat')) if 'Variables' not in m][0], 'var':[m for m in glob.glob(str(stempth / dtset / kk / '**.mat')) if 'Variables' in m][0]} for kk in treatlist}
    
    
    allexpdatas={}
    minvector=[]
    maxvector=[]
    for treatment in range(len(treatlist)):
        allexpdatas[treatment]={}
        allexpdatas[treatment]['data']=loadmat(filedict[treatlist[treatment]]['data'])
        allexpdatas[treatment]['vars']=loadmat(filedict[treatlist[treatment]]['var'])
    
        
        # visstim= allexpdatas[treatment]['data']['Vstim']
        rasternomr=  np.stack(allexpdatas[treatment]['vars']['Data']['CaTransients'])
        # rasternomr=  allexpdatas[treatment]['vars']['Ca_raster_norm']
        raster=  allexpdatas[treatment]['vars']['Ca_raster']
        start=  allexpdatas[treatment]['vars']['col_start']
        end=  allexpdatas[treatment]['vars']['col_end']
        # chnadindx=  allexpdatas[treatment]['vars']['cell']
        allexpdatas[treatment]['vol_df'] = pd.read_csv(Path(filedict[treatlist[treatment]]['data']).parents[0]/ allexpdatas[treatment]['data']['VoltageRecording'])
    
        minvector.append(rasternomr.min(axis=1))
        maxvector.append(rasternomr.max(axis=1))
    
    
        
        # f,ax=plt.subplots(3,sharex=True)
        # ax[0].plot( allexpdatas[treatment]['vol_df']['Time(ms)'].values, allexpdatas[treatment]['vol_df'][' Voltage1'].values)
        # ax[1].plot( allexpdatas[treatment]['vol_df']['Time(ms)'].values, allexpdatas[treatment]['vol_df'][' VisualStim'].values)
        # ax[2].plot( allexpdatas[treatment]['vol_df']['Time(ms)'].values, allexpdatas[treatment]['vol_df'][' Current1'].values)
        # plt.show()
        # f,ax=plt.subplots(2,sharex=True)
        # ax[0].imshow(raster, aspect='auto')
        # ax[1].imshow(rasternomr, aspect='auto')
        # plt.show()
    
    minmaxarray=np.zeros([rasternomr.shape[0],2])
    
    minmaxarray[:,0]=np.stack(maxvector).max(axis=0)
    minmaxarray[:,1]=np.stack(minvector).min(axis=0)
    
    
    
    plt.close('all')
    all_sliced_array={}
    all_sliced_current={}
    allchandindex={}
    allrasternomr={}
    allrasternomr_scaled={}
    alldfs={}
    for treatment in range(len(treatlist)):
        print(treatlist[treatment])
        # allexpdatas[treatment]['vars']['col_start']
        # allexpdatas[treatment]['vars']['col_end']
        rasternomr=  np.stack(allexpdatas[treatment]['vars']['Data']['CaTransients'])
        # rasternomr=  allexpdatas[treatment]['vars']['Ca_raster_norm']
        rasternomr_scaled=np.zeros_like(rasternomr)
        
        chandindex=0
        allchandindex[treatment]=chandindex
        angles=np.linspace(0,360,9)[:-1]
        angle_numbers=len(angles)
        anglevalues = np.reshape(np.arange(1,9), (1, 8))
        frame_correction=1
        
        all_rows=[]
    
        if treatment<2:
            mintrial=allexpdatas[treatment]['vars']['col_start'].shape[0]
            orient=np.float32(np.nan)
            if treatment==0:
                blnak=np.float32(1)
                opt=np.float32(0)
    
            elif treatment==1:
                blnak=np.float32(0)
                opt=np.float32(1)
    
            for trial in range(allexpdatas[treatment]['vars']['col_start'].shape[0]):
                all_rows.append((orient, blnak,opt,np.int32(allexpdatas[treatment]['vars']['col_start'][trial]-frame_correction), np.int32(allexpdatas[treatment]['vars']['col_end'][trial]-frame_correction) ))
            df = pd.DataFrame(all_rows, columns =['orientation', 'blank_sweep','opto','start', 'end',])
        
        elif treatment>=2:
            oris=allexpdatas[treatment]['vars']['col_start'].shape[0]
            mintrial=allexpdatas[treatment]['vars']['col_start'].shape[1]    
            blnak=np.float32(0)
    
            if treatment==2:
                opt=np.float32(0)  
            elif treatment==3:
                opt=np.float32(1)
    
            for ori in range(8):
                angled=angles[ori]
                for trial in range(allexpdatas[treatment]['vars']['col_start'].shape[1]):
                    all_rows.append((np.float32(angled),blnak,opt,np.int32(allexpdatas[treatment]['vars']['col_start'][ori][trial]-frame_correction), np.int32(allexpdatas[treatment]['vars']['col_end'][ori][trial]-frame_correction) ))
                    
            df = pd.DataFrame(all_rows, columns =['orientation', 'blank_sweep','opto','start', 'end',])
            
       
    
        
        
         
        f,ax=plt.subplots(3,sharex=True)
        ax[0].plot( allexpdatas[treatment]['vol_df']['Time(ms)'].values, allexpdatas[treatment]['vol_df'][' Voltage1'].values)
        ax[1].plot( allexpdatas[treatment]['vol_df']['Time(ms)'].values, allexpdatas[treatment]['vol_df'][' VisualStim'].values)
        ax[2].plot( allexpdatas[treatment]['vol_df']['Time(ms)'].values, allexpdatas[treatment]['vol_df'][' Current1'].values)
        plt.show()
        f,ax=plt.subplots(2,sharex=True)
        ax[0].imshow(raster, aspect='auto')
        ax[1].imshow(rasternomr, aspect='auto')
        plt.show()       
        
        isostim=[]
               
        pretime=1
        stimtime=2
        postime=2
        preframes=int(np.round((pretime*1/ allexpdatas[treatment]['data']['frame_period'])))
        stimframes=int(np.round((stimtime*1/ allexpdatas[treatment]['data']['frame_period'])))
        postframes=int(np.round((postime*1/ allexpdatas[treatment]['data']['frame_period'])))
        
        pretimems=1*1000
        stimtimems=2*1000
        postimems=2*1000
        trialtimevector=np.linspace(-pretime,postime,preframes+postframes)
        trialtimevectorms=np.linspace(-pretime,postime,pretimems*10+postimems*10)
                     
        mstimes=allexpdatas[treatment]['vars']['stim_start_time']*10000
        mstimes=mstimes.astype('int')
    
        if treatment>1:
    
            trial_sliced_activity={}
            slicedarrayc=np.zeros((rasternomr.shape[0],oris,mintrial,preframes+postframes))
            cell=0
            for cell in range(rasternomr.shape[0]):
                min_to_use=minmaxarray[cell,1]
                max_to_use=minmaxarray[cell,0]
                min_to_use=minmaxarray[:,1].min()
                max_to_use=minmaxarray[:,0].max()
    
                scaled_trace = (rasternomr[cell]- min_to_use)/(max_to_use- min_to_use)
                # scaled_trace = rasternomr[cell]
                rasternomr_scaled[cell,:]=scaled_trace
    
    
                for i,ori in enumerate(angles):
                    for trial in range(mintrial):
                        if df[df['orientation']==ori].iloc[trial,3]-preframes>0:
                            slicedarrayc[cell,i,trial,:]=scaled_trace[df[df['orientation']==ori].iloc[trial,3]-preframes:df[df['orientation']==ori].iloc[trial,3]+postframes]
           
            slicedcurrent=np.zeros((oris,mintrial,pretimems*10+postimems*10))     
            for ori in range(oris):
                for trial in range(mintrial):
                    if mstimes[ori,trial]-pretimems*10>0:
                        slicedcurrent[ori,trial,:]=allexpdatas[treatment]['vol_df'][' Current1'].values[mstimes[ori,trial]-pretimems*10:mstimes[ori,trial]+postimems*10]
            
            trialaverageca=slicedarrayc.mean(axis=2)[:,0,:]  
            trialaveragecu=slicedcurrent.mean(axis=1)[0,:]  
    
            
            f,ax=plt.subplots(4)
            ax[0].imshow(trialaverageca[1:,:], aspect='auto')
            ax[1].plot(trialtimevector,trialaverageca[1:,:].mean(axis=0))
            ax[2].plot(trialtimevector,trialaverageca[0,:])
            ax[3].plot(trialtimevectorms,trialaveragecu)
            plt.show()
    
    
            trialaverage=slicedarrayc.mean(axis=2)   
            f,ax=plt.subplots(8,2)
            for row in range(4):
                for col in range(2):
                    rasterrow=2*row
                    meanrow=1+2*row
                    ori=row+col
                    menatrace=trialaverage[1:,ori,:]
                    ax[rasterrow,col].imshow(menatrace, aspect='auto')
                    ax[meanrow,col].plot(trialtimevector,menatrace.mean(axis=0))
            plt.show()
            
            slicedarrayc_dff=np.zeros_like(slicedarrayc)
            for i in range(slicedarrayc.shape[0]):
                for j in range(slicedarrayc.shape[1]):
                    for k in range(slicedarrayc.shape[2]):
                        preonset_trace=slicedarrayc[i,j,k,:preframes]
                        dff=100 * ((slicedarrayc[i,j,k,:] / np.nanmean(preonset_trace)) - 1)
                        slicedarrayc_dff[i,j,k,:]=dff
            
        else:
            stimtrace=allexpdatas[treatment]['vol_df'][' Voltage1'].values
            slicedarrayc=np.zeros((rasternomr.shape[0],len(allexpdatas[treatment]['vars']['col_start']),preframes+postframes))
            slicedcurrent=np.zeros((len(mstimes),pretimems*10+postimems*10))     
    
            for cell in range(rasternomr.shape[0]):
                min_to_use=minmaxarray[cell,1]
                max_to_use=minmaxarray[cell,0]
                min_to_use=minmaxarray[:,1].min()
                max_to_use=minmaxarray[:,0].max()
    
                scaled_trace = (rasternomr[cell]- min_to_use)/(max_to_use- min_to_use)
                # scaled_trace = rasternomr[cell]
                rasternomr_scaled[cell,:]=scaled_trace
    
    
    
                for trial in range(len(allexpdatas[treatment]['vars']['col_start'])):
                    if allexpdatas[treatment]['vars']['col_start'][trial]-preframes>0:
    
                        slicedarrayc[cell,trial,:]=scaled_trace[  allexpdatas[treatment]['vars']['col_start'][trial]-preframes:  allexpdatas[treatment]['vars']['col_start'][trial]+postframes]
                        
                    if mstimes[trial]-pretimems*10>0:
    
                        slicedcurrent[trial,:]=allexpdatas[treatment]['vol_df'][' Current1'].values[mstimes[trial]-pretimems*10:mstimes[trial]+postimems*10]
    
        
    
    
    
            
            trialaverageca=slicedarrayc.mean(axis=1)  
            trialaveragecu=slicedcurrent.mean(axis=0)  
    
            f,ax=plt.subplots(3)
            ax[0].imshow(trialaverageca[1:,:], aspect='auto')
            ax[1].plot(trialtimevector,trialaverageca[1:,:].mean(axis=0))
            ax[2].plot(trialtimevectorms,trialaveragecu)
            plt.show()
            
            
            slicedarrayc_dff=np.zeros_like(slicedarrayc)
            for i in range(slicedarrayc.shape[0]):
                for j in range(slicedarrayc.shape[1]):
                        preonset_trace=slicedarrayc[i,j,:preframes]
                        dff=100 * ((slicedarrayc[i,j,:] / np.nanmean(preonset_trace)) - 1)
                        slicedarrayc_dff[i,j,:]=dff
    
       
                    
                    
        all_sliced_array[treatlist[treatment]]=slicedarrayc_dff
        all_sliced_current[treatlist[treatment]]=slicedcurrent
        allrasternomr[treatlist[treatment]]=rasternomr
        allrasternomr_scaled[treatlist[treatment]]=rasternomr_scaled
    
        alldfs[treatlist[treatment]]=df
    
        
       
        
    mykeys=['control_blank','opto_blank','control_grating','opto_grating']
    adapted_all_sliced_array={mykeys[i]:v for i,(key,v) in enumerate(all_sliced_array.items())}
    adapted_all_sliced_current={mykeys[i]:v for i,(key,v) in enumerate(all_sliced_current.items())}
    adapted_all_rasternorm={mykeys[i]:v for i,(key,v) in enumerate(allrasternomr.items())}
    adapted_all_dfs={mykeys[i]:v for i,(key,v) in enumerate(alldfs.items())}
    adapted_all_rasternorm_scaled={mykeys[i]:v for i,(key,v) in enumerate(allrasternomr_scaled.items())}
    
    
    
    
    adapted_all_sliced_array['imaging_time_vector']=trialtimevector
    adapted_all_sliced_current['current_time_vector']=trialtimevectorms
    
    if 'control_grating' in adapted_all_sliced_array.keys():
        for ori in range(len(angles)):
            adapted_all_sliced_array[f'control_{int(angles[ori])}']=adapted_all_sliced_array['control_grating'][:,ori,:,:]
            adapted_all_sliced_array[f'opto_{int(angles[ori])}']=adapted_all_sliced_array['opto_grating'][:,ori,:,:]
            adapted_all_sliced_current[f'control_{int(angles[ori])}']=adapted_all_sliced_current['control_grating'][ori,:,:]
            adapted_all_sliced_current[f'opto_{int(angles[ori])}']=adapted_all_sliced_current['opto_grating'][ori,:,:]
        
        del adapted_all_sliced_array['control_grating']
        del adapted_all_sliced_array['opto_grating']
        del adapted_all_sliced_current['control_grating']
        del adapted_all_sliced_current['opto_grating']
        
    
    
        
    alldtsets_dict[dtset]={'adapted_all_sliced_array':adapted_all_sliced_array,
                            'adapted_all_sliced_current':adapted_all_sliced_current,
                            'adapted_all_rasternorm':adapted_all_rasternorm,
                            'adapted_all_dfs':adapted_all_dfs,
                            'adapted_all_rasternorm_scaled':adapted_all_rasternorm_scaled}


  


