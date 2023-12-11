# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:09:14 2023

@author: sp3660
\
    
    
    
   THINGS MISSING
       visual tuning
       differneces in effect of visual tuning on opto responses
       selecteing significnatly incxreased and decreased cell and comparing peak opto reposnse on this only
       analysis of blank sweep controls
       analysis of no opsin aquisition
       some kinf of correlation analysis maybe
       analysis of no injection acuisition
       
       

    
"""
import numpy as np
from PIL import Image
import scipy
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr, ttest_ind, zscore, mode
import scipy
from statsmodels.nonparametric.smoothers_lowess import lowess
import caiman as cm
from matplotlib.patches import Rectangle
import scipy.signal as sg
import scipy.stats as st
import pandas as pd
import shutil
import copy
import os
import seaborn as sns
from math import sqrt
import pickle
import glob
import scipy.stats as st

#%% PROCESSING FUNCTIONS
deriv=lambda x:np.diff(x,prepend=x[0] )
rectified=lambda x:np.absolute(x)

def save_temp_data(multiple_analysis,datapath):
    if not os.path.isfile(datapath):
        with open(datapath, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(multiple_analysis, f, pickle.HIGHEST_PROTOCOL)
    return datapath
            
def check_temp_data(tempprocessingpat) :
    temp_data_list=[]
    temp_data_list=glob.glob(tempprocessingpat+f'\\**{experimentalmousename}**', recursive=False)
    
    return temp_data_list

def load_temp_data(temp_data_list,dataindex):
    multiple_analysis={}
    if temp_data_list:
        selected_temp_data_path=temp_data_list[dataindex]
        with open( selected_temp_data_path, 'rb') as file:
            multiple_analysis=  pickle.load(file)
    return multiple_analysis
    
def get_peak(full_data,response,sweep_response,mean_sweep_response):
    ''' Computes metrics related to each cell's peak response condition.
 
    Returns
    -------
    Pandas data frame containing the following columns (_dg suffix is
    for drifting grating):
        * ori_dg (orientation)
        * tf_dg (temporal frequency)
        * reliability_dg
        * osi_dg (orientation selectivity index)
        * dsi_dg (direction selectivity index)
        * peak_dff_dg (peak dF/F)
        * ptest_dg
        * p_run_dg
        * run_modulation_dg
        * cv_dg (circular variance)
    '''
    df=full_data['visstim_info']['OptoDrift']['stimulus_table']
    numbercells=full_data['imaging_data']['All_planes_rough']['CellNumber']
    orivals=df.dropna().orientation.sort_values().unique()
    tfvals=df.dropna().temporal_frequency.sort_values().unique()

 
    peak = pd.DataFrame(index=range(numbercells),
                        columns=('ori_dg', 'tf_dg', 'reliability_dg',
                                 'osi_dg', 'dsi_dg', 'peak_dff_dg',
                                 'ptest_dg', 'p_run_dg',
                                 'run_modulation_dg',
                                 'cv_os_dg', 'cv_ds_dg', 'tf_index_dg',
                                 'cell_specimen_id'))
    # cids = self.drift_obj.data_set.get_cell_specimen_ids()
    cids=np.arange(numbercells)
    orivals_rad = np.deg2rad(orivals)
    for nc in range(numbercells):
        cell_peak = np.where(response[:, :, nc, 0] == np.nanmax(
            response[:, :, nc, 0]))
        prefori = cell_peak[0][0]
        preftf = cell_peak[1][0] 
        peak.cell_specimen_id.iloc[nc] = cids[nc]
        peak.ori_dg.iloc[nc] = prefori
        peak.tf_dg.iloc[nc] = preftf
 
        pref = response[prefori, preftf, nc, 0]
        orth1 = response[np.mod(prefori + 2, 8), preftf, nc, 0]
        orth2 = response[np.mod(prefori - 2, 8), preftf, nc, 0]
        orth = (orth1 + orth2) / 2
        null = response[np.mod(prefori + 4, 8), preftf, nc, 0]
 
        tuning = response[:, preftf, nc, 0]
        tuning = np.where(tuning > 0, tuning, 0)
        # new circular variance below
        CV_top_os = np.empty((8), dtype=np.complex128)
        CV_top_ds = np.empty((8), dtype=np.complex128)
        for i in range(8):
            CV_top_os[i] = (tuning[i] * np.exp(1j * 2 * orivals_rad[i]))
            CV_top_ds[i] = (tuning[i] * np.exp(1j * orivals_rad[i]))
        peak.cv_os_dg.iloc[nc] = np.abs(CV_top_os.sum()) / tuning.sum()
        peak.cv_ds_dg.iloc[nc] = np.abs(CV_top_ds.sum()) / tuning.sum()
 
        peak.osi_dg.iloc[nc] = (pref - orth) / (pref + orth)
        peak.dsi_dg.iloc[nc] = (pref - null) / (pref + null)
        peak.peak_dff_dg.iloc[nc] = pref
 
        groups = []
        for ori in orivals:
            for tf in tfvals[:]:
                groups.append(
                    mean_sweep_response[
                        (df.temporal_frequency == tf) &
                        (df.orientation == ori)][str(nc)])
        groups.append(mean_sweep_response[
                          df.temporal_frequency == 0][
                          str(nc)])
        _, p = st.f_oneway(*groups)
        peak.ptest_dg.iloc[nc] = p
 
        subset = mean_sweep_response[
            (df.temporal_frequency == tfvals[preftf]) &
            (df.orientation == orivals[prefori])]
 
        # running modulation
        subset_stat = subset[subset.dx < 1000]
        subset_run = subset[subset.dx >= 1000]
        if (len(subset_run) > 2) & (len(subset_stat) > 2):
            (_, peak.p_run_dg.iloc[nc]) = st.ttest_ind(subset_run[str(nc)],
                                                       subset_stat[
                                                           str(nc)],
                                                       equal_var=False)
 
            if subset_run[str(nc)].mean() > subset_stat[str(nc)].mean():
                peak.run_modulation_dg.iloc[nc] = (subset_run[
                                                       str(nc)].mean() -
                                                   subset_stat[
                                                       str(nc)].mean()) \
                                                  / np.abs(
                    subset_run[str(nc)].mean())
            elif subset_run[str(nc)].mean() < subset_stat[str(nc)].mean():
                peak.run_modulation_dg.iloc[nc] = \
                    (-1 * (subset_stat[str(nc)].mean() -
                           subset_run[str(nc)].mean()) /
                     np.abs(subset_stat[str(nc)].mean()))
 
        else:
            peak.p_run_dg.iloc[nc] = np.NaN
            peak.run_modulation_dg.iloc[nc] = np.NaN
 
        # reliability
        subset = sweep_response[
            (df.temporal_frequency == tfvals[preftf]) &
            (df.orientation == orivals[prefori])]
        corr_matrix = np.empty((len(subset), len(subset)))
        for i in range(len(subset)):
            for j in range(len(subset)):
                r, p = st.pearsonr(subset[str(nc)].iloc[i][int(np.ceil(len(subset[str(nc)].iloc[i])*1/4)):int(np.ceil(len(subset[str(nc)].iloc[i])*3/4))],
                                   subset[str(nc)].iloc[j][int(np.ceil(len(subset[str(nc)].iloc[i])*1/4)):int(np.ceil(len(subset[str(nc)].iloc[i])*3/4))])
                corr_matrix[i, j] = r
        mask = np.ones((len(subset), len(subset)))
        for i in range(len(subset)):
            for j in range(len(subset)):
                if i >= j:
                    mask[i, j] = np.NaN
        corr_matrix *= mask
        peak.reliability_dg.iloc[nc] = np.nanmean(corr_matrix)
 
        # TF index
        # tf_tuning = response[prefori, 1:, nc, 0]
        # trials = mean_sweep_response[
        #     (df.temporal_frequency != 0) &
        #     (df.orientation == orivals[prefori])
        # ][str(nc)].values
        # SSE_part = np.sqrt(
        #     np.sum((trials - trials.mean()) ** 2) / (len(trials) - 5))
        # peak.tf_index_dg.iloc[nc] = (np.ptp(tf_tuning)) / (
        #             np.ptp(tf_tuning) + 2 * SSE_part)
 
    return peak
def get_response(analysis,full_data,mean_sweep_response,pval ):
    ''' Computes the mean response for each cell to each stimulus
    condition.  Return is
    a (# orientations, # temporal frequencies, # cells, 3) np.ndarray.
    The final dimension
    contains the mean response to the condition (index 0), standard
    error of the mean of the response
    to the condition (index 1), and the number of trials with a
    significant response (p < 0.05)
    to that condition (index 2).

    Returns
    -------
    Numpy array storing mean responses.
    '''
    df=full_data['visstim_info']['OptoDrift']['stimulus_table']
    
    response = np.empty(
        (     len(df.dropna().orientation.unique()), len(df.dropna().temporal_frequency.unique()), full_data['imaging_data']['All_planes_rough']['CellNumber'] + 1, 3))

    def ptest(x):
        if x.empty:
            return np.nan
        return len(np.where(x < (0.05 / (8 * 5)))[0])
    orivals=df.dropna().orientation.sort_values().unique()
    tfvals=df.dropna().temporal_frequency.sort_values().unique()

    for ori in orivals:
        ori_pt = np.where(orivals == ori)[0][0]
        for tf in tfvals:
            tf_pt = np.where(tfvals == tf)[0][0]
            subset_response = mean_sweep_response[
                (df.temporal_frequency == tf) & (
                            df.orientation == ori)]
            subset_pval = pval[
                (df.temporal_frequency == tf) & (
                        df.orientation == ori)]
            response[ori_pt, tf_pt, :, 0] = subset_response.mean(axis=0)
            response[ori_pt, tf_pt, :, 1] = subset_response.std(
                axis=0) / sqrt(len(subset_response))
            response[ori_pt, tf_pt, :, 2] = subset_pval.apply(
                ptest, axis=0)
            
    return response

def get_stimulus_table(analysis,visstimtranstionts,pre_time=500,post_time=1000):

   #ms
    pre_frames=np.ceil(pre_time/analysis.milisecond_period).astype(int)
    post_frames=np.ceil(post_time/analysis.milisecond_period).astype(int)
    
    
    
    grating_number=visstimtranstionts.shape[0]
    grating_repetitions=visstimtranstionts.shape[1]*visstimtranstionts.shape[2]
    blnaksweepreps=4*visstimtranstionts.shape[1]
    angles=np.linspace(0,360,9)[:-1]
    angle_numbers=len(angles)
    frequencies=np.array([2])
    frequency_numbers=len(frequencies)
    angles_xv, frequencies_yv = np.meshgrid(angles,frequencies)
    anglevalues = np.reshape(np.arange(1,9), (1, 8))
    
    all_rows=[]
    for ori in range(1,9):

        angled=angles_xv[:,np.where(anglevalues==ori)[1][0]][0]
        freq=float(frequencies[np.where(anglevalues==ori)[0][0]])
        
        indexes=list(zip(np.reshape(visstimtranstionts[ori-1,:,:,0],(1,grating_repetitions)),         np.reshape(visstimtranstionts[ori-1,:,:,1],(1,grating_repetitions))))[0]
              
        
        for i in range(grating_repetitions):
            opto=np.float32(0)

            if ori==analysis.acquisition_object.visstimdict['opto']['randomoptograting'] and i % 2 == 0:
                opto=np.float32(1)
            
        
            all_rows.append((np.float32(freq),np.float32(angled), np.float32(0),opto,np.int32(indexes[0][i]), np.int32(indexes[1][i]) ))
          

          
    blankindexes=list(zip(np.reshape(analysis.signals_object.optodrift_info['Blank']['ArrayFinal_downsampled_LED_clipped'],(1,blnaksweepreps)), np.reshape(analysis.signals_object.optodrift_info['Blank']['ArrayFinalOffset_downsampled_LED_clipped'],(1,blnaksweepreps))))[0]
    for i in range(blnaksweepreps):
        opto=np.float32(0)

        if blank_opt and i in np.array([np.array([4*i, 4*i+1]) for i in range(20)]).ravel():
            opto=np.float32(1)
        
        all_rows.append((np.float32(np.nan), np.float32(np.nan), np.float32(1),opto,np.int32(blankindexes[0][i]), np.int32(blankindexes[1][i]) ))

    df = pd.DataFrame(all_rows, columns =['temporal_frequency','orientation', 'blank_sweep','opto','start', 'end',])
    sorted_df=df.sort_values(by=['start'])
    sorted_df= sorted_df.reset_index(drop=True)
    analysis.full_data['visstim_info']['OptoDrift']={}
    analysis.full_data['visstim_info']['OptoDrift']['stimulus_table']=sorted_df
    
    return sorted_df,pre_frames,post_frames

def get_sweep_response(analysis,full_data,traces,blank_opt,visstimtranstionts,use_scaled=False,smoothwindow=False,dff=False):
   
    titstr='raw'
    substracted=None
    if use_scaled:
        traces=scale_to_range(traces)
        titstr=titstr+'_scaled'
    trace_type=select_trace_type(substracted=substracted,smoothed=smoothwindow)
    
    if smoothwindow:
  
      traces,smoothwindow=smooth_activity_array(traces, smoothwindow)
      titstr=titstr+'_smoothed'
    else:
        traces=traces
    

    sorted_df,pre_frames,post_frames=get_stimulus_table(analysis,visstimtranstionts)
    
    def do_mean(x,pre_frames=13,post_frames=25):
        # +1])
        return np.mean(
            x[pre_frames:
              pre_frames + post_frames])

    def do_p_value(x,pre_frames=13,post_frames=25):
        (_, p) = \
            st.f_oneway(
                x[:pre_frames],
                x[pre_frames:
                  pre_frames + post_frames])
        return p

    numberc=len(traces)
    speed=full_data['voltage_traces']['Speed']
    full_data['imaging_data']['All_planes_rough']['Timestamps'][0]
    
    sweep_response = pd.DataFrame(index=sorted_df.index.values,
                                  columns=list(map(str, range(
                                      numberc + 1))))   
    sweep_response.rename(
        columns={str(numberc): 'dx'}, inplace=True)
    
    
    for index, row in sorted_df.iterrows():
        start = int(row['start'] - pre_frames)
        end = int(row['start'] + post_frames + pre_frames)

        for nc in range(numberc):
            temp = traces[int(nc), start:end]
            
            if dff:
                sweep_response[str(nc)][index] = \
                          100 * ((temp / np.mean(temp[:pre_frames])) - 1)
            else:
                sweep_response[str(nc)][index] =temp 
            
            
            
        sweep_response['dx'][index] = speed[start:end]

    mean_sweep_response = sweep_response.applymap(do_mean,pre_frames=pre_frames, post_frames=post_frames)

    
    pval = sweep_response.applymap(do_p_value,pre_frames=pre_frames, post_frames=post_frames)
    return sweep_response, mean_sweep_response, pval, full_data

def process_vis_stim(analysis):
    
    analysis.signals_object.extract_transitions_optodrift('VisStim', led_clipped=True, plot=False)
    analysis.signals_object.downsample_optodrift_onsets()

    optograting=analysis.acquisition_object.visstimdict['opto']['randomoptograting']


    final_stim_onsets={}
    for key, stim in analysis.signals_object.optodrift_info.items():
        final_stim_onsets[key]=stim['ArrayFinal_downsampled_LED_clipped']
    final_stim_onsets['opto_grating']=analysis.acquisition_object.visstimdict['opto']['randomoptograting']
  
    
    return   final_stim_onsets
    
        
    
def load_chandelier_cell_file(analysis):
    
    all_planes_chand_indexes={}
    planes=['Plane1','Plane2']
    
    for i,dtset in enumerate(analysis.calcium_datasets.values()):

        filepath=    os.path.split(dtset.selected_dataset_mmap_path)[0]+os.sep+'ChandIdentity.csv'
        temppath=r'C:\Users\sp3660\Desktop\ChandIdentity.csv'
        tempfile=   shutil.copyfile( filepath,temppath)
    
        df=pd.read_csv(tempfile,header=0)
        all_planes_chand_indexes[planes[i]]=df['Chandeliers Python ID'].values
        
    return all_planes_chand_indexes

def create_opto_transitions_array(analysis,stimulated_cells_number,nTrials,opto_repetitions,c,blank_opt):
    up=analysis.full_data['voltage_traces']['Full_signals_transitions']['PhotoStim']['aligned_downsampled_LEDshifted']['Prairie']['up']
    down=analysis.full_data['voltage_traces']['Full_signals_transitions']['PhotoStim']['aligned_downsampled_LEDshifted']['Prairie']['down']
    
    if led_opt:
        up=analysis.full_data['voltage_traces']['Full_signals_transitions']['PhotoTrig']['aligned_downsampled_LEDshifted']['Prairie']['up']
        down=analysis.full_data['voltage_traces']['Full_signals_transitions']['PhotoTrig']['aligned_downsampled_LEDshifted']['Prairie']['down']
    
    #THIS IS TOTALY WROMH DONT TRUST
    transition_array=np.zeros([stimulated_cells_number,nTrials,opto_repetitions,2],dtype=int)
    stimt=analysis.full_data['visstim_info']['OptoDrift']['stimulus_table']
    


    stimt.opto==1
    stimt.blank_sweep==0
    optotrialsst=stimt[(stimt.opto==1) & (stimt.blank_sweep==0)]
    onsets=up[0::opto_repetitions]

    
    it=0
    
    for i in  range(len(onsets)):
        
        if np.any(np.where(abs(onsets[i]- np.insert(optotrialsst.start.values,0,0))<5)):

            for cell in range(stimulated_cells_number):
                transition_array[cell,it,:,0]=  up[np.where(up==onsets[i])[0][0]:np.where(up==onsets[i])[0][0]+opto_repetitions]
                transition_array[cell,it,:,1]=  down[np.where(up==onsets[i])[0][0]:np.where(up==onsets[i])[0][0]+opto_repetitions]
            
            it=it+1
            
            
           
            
    return transition_array
        
def find_optostim_cells(analysis,all_planes_chand_indexes):
  
    res=analysis.caiman_results[list(analysis.caiman_results.keys())[0]]
    
    all_planes_distances_from_optocell={}
    accepted_all_plane_distances=None
    
    
    if analysis.acquisition_object.metadata_object.photostim_file:
        ref_image=Image.fromarray(analysis.acquisition_object.reference_image_dic[[i for i in analysis.acquisition_object.reference_image_dic.keys() if ('Red-8bit' in i) or ('Red-Green-8bit' in i)][0]])
        coordinates=[]
        for key, v in   analysis.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries'].items():
          coordinates.append((float(v['points']['Point_1']['x_pos'])*256,float(v['points']['Point_1']['y_pos'])*256,float(v['points']['Point_1']['spiral_width'])*256))
          
          
        all_planes_cell_coordinates={}  
        for plane in analysis.caiman_results.keys():
            res=analysis.caiman_results[plane]
            # res.data['est']['contours']
            A=res.data['est']['A']
            nA = np.sqrt(np.ravel(A.power(2).sum(axis=0)))
            nA_inv_mat = scipy.sparse.spdiags(1. / nA, 0, nA.shape[0], nA.shape[0])
            A = A * nA_inv_mat
            A=A.toarray().reshape([256,256,len(res.data['est']['contours'])])
               
            
            cell_coordinates=np.vstack(res.centers_of_mass)
            all_planes_cell_coordinates[plane]=cell_coordinates
     
        optoplane=[m for m in all_planes_cell_coordinates.keys() if 'Plane1' in m][0]
        optocellids=np.zeros(len(coordinates),dtype=int)
        for cell in range(len(coordinates)):
            dist=[]
            for i in range(len(all_planes_cell_coordinates[optoplane][:,0])):
                dist.append(np.linalg.norm(np.array((all_planes_cell_coordinates[optoplane][i,0], all_planes_cell_coordinates[optoplane][i,1]))- np.array((coordinates[cell][0], coordinates[cell][1]))))
            optocellids[cell]=np.argmin(dist)
 
        if len(optocellids)==1:
            for plane, cell_coordinates in  all_planes_cell_coordinates.items():
                distances_from_optocell=np.zeros(cell_coordinates.shape[0])
                for cell in range(len(cell_coordinates)):
                    distances_from_optocell[cell]=np.linalg.norm(np.array((cell_coordinates[cell,0], cell_coordinates[cell,1]))- np.array((all_planes_cell_coordinates[optoplane][optocellids][0][0],all_planes_cell_coordinates[optoplane][optocellids][0][1])))
                all_planes_distances_from_optocell[plane]=distances_from_optocell
                
            accepted_all_plane_distances={}
            for key in all_planes_distances_from_optocell.keys():
                plane=key[key.find('Plane'):key.find('Plane')+6]
                accepted_all_plane_distances[plane]=all_planes_distances_from_optocell[key][analysis.full_data['imaging_data']['All_planes_rough']['CellIds'][plane]]
            accepted_all_plane_distances=np.hstack(accepted_all_plane_distances.values())
            
                
            
        f,ax=plt.subplots(1,3)
        ax[0].imshow(A[:,:,analysis.full_data['imaging_data']['Plane1']['CellIds']].sum(axis=2).T)
        ax[1].imshow(A[:,:,optocellids].sum(axis=2).T)
        ax[2].imshow(ref_image.resize((256, 256)))
        for coord in coordinates:
            circles=[]
            for i in range(3):
                ax[i].add_patch(plt.Circle((coord[0], coord[1]), coord[2]/2, color='r', fill=False))
            
            optocellids.sort()
            
        
        alloptocellids={'Plane1':optocellids ,'Plane2':np.empty([0])}
            
    else:
        
        alloptocellids=all_planes_chand_indexes
       
        
        
        
    optocellindex_dict={'chand':{'all':{},
                                 'opto':{},
                                 'non_opto':{}}
                        ,
                        'non_chand':{'all':{},
                                     'opto':{},
                                     'non_opto':{}}
                        ,
                        'opto':{'all':{},
                                }
                        ,
                        'non_opto':{'all':{},
                                     },
                        'all':{'all':{}}
                        }
    

   
        
    for plane,ids in analysis.full_data['imaging_data']['All_planes_rough']['CellIds'].items():
        
        optocellindex_dict['opto']['all'][plane]=np.where(np.in1d(ids, alloptocellids[plane]))[0]
        optocellindex_dict['non_opto']['all'][plane]=np.where(np.in1d(ids, alloptocellids[plane],invert=True))[0]
        optocellindex_dict['chand']['all'][plane]=np.where(np.in1d(ids,all_planes_chand_indexes[plane]))[0]
        optocellindex_dict['non_chand']['all'][plane]=np.where(np.in1d(ids, all_planes_chand_indexes[plane],invert=True))[0]

        optocellindex_dict['chand']['opto'][plane]=optocellindex_dict['chand']['all'][plane][np.in1d(optocellindex_dict['chand']['all'][plane], optocellindex_dict['opto']['all'][plane])]
        optocellindex_dict['non_chand']['opto'][plane]=optocellindex_dict['non_chand']['all'][plane][np.in1d(optocellindex_dict['non_chand']['all'][plane], optocellindex_dict['opto']['all'][plane])]

        optocellindex_dict['chand']['non_opto'][plane]=optocellindex_dict['chand']['all'][plane][np.in1d(optocellindex_dict['chand']['all'][plane], optocellindex_dict['opto']['all'][plane],invert=True)]
        optocellindex_dict['non_chand']['non_opto'][plane]=optocellindex_dict['non_chand']['all'][plane][np.in1d(optocellindex_dict['non_chand']['all'][plane], optocellindex_dict['opto']['all'][plane],invert=True)]
        optocellindex_dict['all']['all'][plane]=np.where(np.in1d(ids, ids))[0]
      
    for key, di in  optocellindex_dict.items():
        for k,treat in di.items():   
            if  len(optocellindex_dict[key][k].keys())>1:
                temp=optocellindex_dict[key][k]['Plane1']
                planes=list(treat.keys())
                for l,(plane, cl) in  enumerate(treat.items()):
                    if 'All' not in plane and l>0:
                        temp=np.concatenate([temp,treat[list(treat.keys())[l]]+len(analysis.full_data['imaging_data']['All_planes_rough']['CellIds'][planes[l-1]])])
                optocellindex_dict[key][k]['All_planes']=temp
            else:
                optocellindex_dict[key][k]['All_planes']=optocellindex_dict[key][k]['Plane1']
      

    traces_dict={}  
    for key, di in  optocellindex_dict.items():
        traces_dict[key]={}
        for k,d in di.items():  
            traces_dict[key][k]={}
            for plane,idx in d.items():  
                traces_dict[key][k][plane]={}
                traces_dict[key][k][plane]['demixed']=analysis.full_data['imaging_data']['All_planes_rough']['Traces']['demixed'][idx,:]

 

    return optocellindex_dict,traces_dict,accepted_all_plane_distances

def slice_by_optotrial(stimulated_cells_number,pretime,posttime,fr,transition_array,traces_dict,gratingcontrol,use_scaled=False): 
    prestim=int(pretime*fr)
    poststim=int(posttime*fr)        
    trialtimevector=np.linspace(-pretime,posttime,prestim+poststim)
    smoothwindows=10

    prestimspeed=int(pretime*1000)
    poststimspeed=int(posttime*1000)  
    
    trialspeedtimevector=np.linspace(-pretime,posttime,prestimspeed+poststimspeed)
    interoptoframes=mode(np.diff(transition_array[0,0,:,0]))[0]
    # interoptoframes=mode(np.diff(transition_array[0,0,:,0]))[0][0]
    
    
    build_dict={'raw':np.zeros((traces_dict['all']['all']['All_planes']['demixed'].shape[0],stimulated_cells_number,nTrials,prestim+poststim)),
                'substracted':np.zeros((traces_dict['all']['all']['All_planes']['demixed'].shape[0],stimulated_cells_number,nTrials,prestim+poststim)),
                'raw_smoothed':np.zeros((traces_dict['all']['all']['All_planes']['demixed'].shape[0],stimulated_cells_number,nTrials,prestim+poststim)),
                'substracted_smoothed':np.zeros((traces_dict['all']['all']['All_planes']['demixed'].shape[0],stimulated_cells_number,nTrials,prestim+poststim))
                }
    build_mean_dict={'raw':np.zeros((traces_dict['all']['all']['All_planes']['demixed'].shape[0],stimulated_cells_number,nTrials,prestim+poststim)).mean(axis=2),
                     'substracted':np.zeros((traces_dict['all']['all']['All_planes']['demixed'].shape[0],stimulated_cells_number,nTrials,prestim+poststim)).mean(axis=2),
                     'raw_smoothed':np.zeros((traces_dict['all']['all']['All_planes']['demixed'].shape[0],stimulated_cells_number,nTrials,prestim+poststim)).mean(axis=2),
                     'substracted_smoothed':np.zeros((traces_dict['all']['all']['All_planes']['demixed'].shape[0],stimulated_cells_number,nTrials,prestim+poststim)).mean(axis=2) 
                     }
    
    activity_dict={'opto':  {'trials':copy.deepcopy(build_dict),
                             'mean': copy.deepcopy(build_mean_dict)  
                             },
                   'control':{'trials':copy.deepcopy(build_dict),
                              'mean':  copy.deepcopy(build_mean_dict)
                              },
                   'speed':{'trials': {'raw':np.empty(0),
                                        'raw_smoothed':np.zeros((stimulated_cells_number,nTrials,prestimspeed+poststimspeed)),
                                        },                            
                              }
                   }
                   

    print('Creating optostim arrays (recordeccell, stimulated cell, trials, trace,)')
   
        
    # f,ax=plt.subplots()
    # ax.imshow(traces_dict['all']['all']['All_planes']['demixed'],aspect='auto')
    
    if use_scaled:
        traces_dict['all']['all']['All_planes']['demixed']=scale_to_range(traces_dict['all']['all']['All_planes']['demixed'])
    
    
    for cell in range(traces_dict['all']['all']['All_planes']['demixed'].shape[0]):
        for opto_idx in range(stimulated_cells_number):
            for i, opto  in enumerate(transition_array[opto_idx,:,0,0]):
                activity_dict['opto']['trials']['raw'][cell,opto_idx,i,:]=traces_dict['all']['all']['All_planes']['demixed'][cell,opto-prestim:opto+poststim]
                
    if gratingcontrol.any():
        for cell in range(traces_dict['all']['all']['All_planes']['demixed'].shape[0]):
            for opto_idx in range(stimulated_cells_number):
                for i, cont  in enumerate(gratingcontrol):
                    activity_dict['control']['trials']['raw'][cell,opto_idx,i,:]=traces_dict['all']['all']['All_planes']['demixed'][cell,cont-prestim:cont+poststim]
                    
    activity_dict['control']['mean']['raw']=activity_dict['control']['trials']['raw'].mean(axis=2)
    activity_dict['opto']['mean']['raw']=activity_dict['opto']['trials']['raw'].mean(axis=2)

    
    print('Substracting baseline all cells')
    for cell in range(traces_dict['all']['all']['All_planes']['demixed'].shape[0]):
        for opto_idx in range(stimulated_cells_number):
            for trial  in range(nTrials):
                for treat in ['opto','control']:

                    activity_dict[treat]['trials']['substracted'][cell,opto_idx,trial,:]=      (activity_dict[treat]['trials']['raw'][cell,opto_idx,trial,:]- activity_dict[treat]['trials']['raw'][cell,opto_idx,trial,:prestim].mean())/activity_dict[treat]['trials']['raw'][cell,opto_idx,trial,:prestim].mean()
    
    activity_dict['opto']['mean']['substracted']= activity_dict['opto']['trials']['substracted'].mean(axis=2)
    activity_dict['control']['mean']['substracted']=   activity_dict['control']['trials']['substracted'].mean(axis=2)
    
    #smooth sigblas for plotting after processing
    print('Smoothing signals')
  
    for opto_idx in range(stimulated_cells_number):
        for treat in ['opto','control']:
            for p in ['raw','substracted']:
                activity_dict[treat]['mean'][p+'_smoothed'][:,opto_idx,:],_=smooth_activity_array(activity_dict[treat]['mean'][p][:,opto_idx,:],smoothwindows)
                for trial  in range(nTrials):

                    activity_dict[treat]['trials'][p+'_smoothed'][:,opto_idx,trial,:],_=smooth_activity_array(activity_dict[treat]['trials'][p][:,opto_idx,trial,:],smoothwindows)

    
    print('Scaling and smoothing speed')
    for opto_idx in range(stimulated_cells_number):
        for i, opto  in enumerate(transition_array[opto_idx,:,0,0]):
            voltopto= np.round(analysis.mov_timestamps_miliseconds['shifted'][opto]).astype(int)
            activity_dict['speed']['trials']['raw_smoothed'][opto_idx,i,:]=analysis.scale_signal(speed)[voltopto-prestimspeed:voltopto+poststimspeed]
    
    activity_dict['parameters']={'prestim':prestim,
                                 'poststim':poststim,
                                 'pretime':pretime,
                                 'posttime':posttime,
                                'interoptoframes':interoptoframes,
                                'trialtimevector':trialtimevector,
                                'trialspeedtimevector':trialspeedtimevector,
                                'use_scaled':use_scaled,
                                'smoothingWindow':smoothwindows,
                                'fr':fr}
    
    
    return activity_dict
           
def smooth_activity_array(ac_array, smoothwindow):
    
    
    smoothed_array=np.zeros_like(ac_array)
    for row in range(ac_array.shape[0]):
        smoothed_array[row,:]=analysis.smooth_trace(ac_array[row,:],smoothwindow)
    
    return smoothed_array, smoothwindow
    
def select_trace_type(substracted=False,smoothed=False):
    if substracted:
        trace_type='substracted'
    else:
        trace_type='raw'
      
    if smoothed:
        trace_type=trace_type+'_smoothed'

    return  trace_type    

def scale_to_range(selectedtraces):
    scaled_traces=np.zeros_like(selectedtraces)
    for cell in range(selectedtraces.shape[0]):
        x=selectedtraces[cell,:]
        scaled_traces[cell,:] = (x-np.min(x))/(np.max(x)-np.min(x))
    return scaled_traces

def mean_opto_response(activity_dict,stat='mean',stimtime=2):
    tracetypes=list(activity_dict['opto']['trials'].keys())
    new_activity_dict=copy.deepcopy(activity_dict)
    if stat=='mean':
        measure=lambda x:x.mean()
    elif stat=='max':
        measure=lambda x:x.max()
    

    for treat in ['opto','control']:
        for tracetype in tracetypes:
            new_activity_dict[treat]['trials'][tracetype+'_stim_peak']=np.zeros([activity_dict[treat]['trials'][tracetype].shape[0],activity_dict[treat]['trials'][tracetype].shape[1],activity_dict[treat]['trials'][tracetype].shape[2],1])
            new_activity_dict[treat]['mean'][tracetype+'_stim_peak']=np.zeros([activity_dict[treat]['mean'][tracetype].shape[0],activity_dict[treat]['trials'][tracetype].shape[1],1])
            for opto_trial in range( activity_dict[treat]['trials'][tracetype].shape[1]):
                for cell in range( activity_dict[treat]['trials'][tracetype].shape[0]):
                    new_activity_dict[treat]['mean'][tracetype+'_stim_peak'][cell,opto_trial,0]=measure(activity_dict[treat]['mean'][tracetype][cell,opto_trial,activity_dict['parameters']['prestim']:activity_dict['parameters']['prestim']+int(stimtime*activity_dict['parameters']['fr'])])
                    for trial in range( activity_dict[treat]['trials'][tracetype].shape[2]):
                        new_activity_dict[treat]['trials'][tracetype+'_stim_peak'][cell,opto_trial,trial,0]=measure(activity_dict[treat]['trials'][tracetype][cell,opto_trial,trial, activity_dict['parameters']['prestim']:activity_dict['parameters']['prestim']+int(stimtime*activity_dict['parameters']['fr'])])
                        
                        
                        
    return   new_activity_dict

#%% plotting functions

def check_transitions(info_list):
    
    """
    this function is to review that all opto and visstim onsets and offsets are adequate.
    """
    
    analysis=info_list[1]
    
    
    analysis.mean_movie_path =analysis.calcium_datasets[list(analysis.calcium_datasets.keys())[0]].bidishift_object.mean_movie_path
    mean_mov=np.load(analysis.mean_movie_path)
    if len(mean_mov)-len(analysis.mov_timestamps_miliseconds['raw'])==1:
        mean_mov=mean_mov[:-1]
    timestamps_voltage_signals=analysis.full_data['voltage_traces']['Full_signals']['Prairie']['LED_aligned']['traces']['PhotoTrig'].index.values
    mov=analysis.scale_signal(mean_mov)
    surrounddown=np.array([[i-1,i,i+1] for i in [analysis.start_frame]])
    surroundup=np.array([[i-1,i,i+1] for i in [analysis.end_frame]])
       
     
    
    
    deriv=lambda x:np.diff(x,prepend=x[0] )
    rectified=lambda x:np.absolute(x)
    
    signals_to_plot=[ 'VisStim',  'LED',  'PhotoStim',  'PhotoTrig',  'AcqTrig']
    analysis.full_data['voltage_traces']['Full_signals']['Prairie']['LED_aligned']['traces']
    
    
  
    
    
    #plotting clipped signal to video length
    signals_to_plot=[  'LED',  'AcqTrig']
    for record_to_plot, process in analysis.full_data['voltage_traces']['Full_signals'].items():
        f,ax=plt.subplots()
        for sig in  signals_to_plot:
            ax.plot(analysis.scale_signal(process['Raw']['traces'][sig].values),label=sig+' Raw')
            ax.plot(analysis.scale_signal(process['Movie_length_clipped']['traces'][sig].values),label=sig+' Movie_length_clipped')   
        ax.plot(analysis.mov_timestamps_miliseconds['raw'],mov,label='Video')
        ax.legend()
        f.suptitle(f'{record_to_plot} Check Video Length Clipping, this just cuts from the back')
    
    plt.show()
    
    
    #plot LED aligned not clipped signals
    # plt.close('all')
    for record_to_plot, process in analysis.full_data['voltage_traces']['Full_signals'].items():
        f,ax=plt.subplots(len(signals_to_plot)+1,sharex=True)
        for k,sigk in enumerate(signals_to_plot):
            sig=analysis.scale_signal(process['LED_aligned']['traces'][sigk].values)
            locomotion=process['LED_aligned']['traces']['Locomotion'].values
            speed=analysis.scale_signal(rectified(deriv(locomotion)))
            speedtimestamps=process['LED_aligned']['traces']['Locomotion'].index.values
            upt=analysis.full_data['voltage_traces']['Full_signals_transitions'][sigk]['aligned'][record_to_plot]['up']
            downt=analysis.full_data['voltage_traces']['Full_signals_transitions'][sigk]['aligned'][record_to_plot]['down']
            upt_downsampled=analysis.full_data['voltage_traces']['Full_signals_transitions'][sigk]['aligned_downsampled'][record_to_plot]['up']
            downt_downsampled=analysis.full_data['voltage_traces']['Full_signals_transitions'][sigk]['aligned_downsampled'][record_to_plot]['down']
            surrounddown=np.array([[i-1,i,i+1] for i in downt_downsampled])
            surroundup=np.array([[i-1,i,i+1] for i in upt_downsampled])
    
    
            ax[k].plot(analysis.mov_timestamps_miliseconds['raw'], mov,label='meanmov')
            ax[k].plot(speedtimestamps, speed,label='speed')
            ax[k].vlines(analysis.mov_timestamps_miliseconds['raw'][analysis.start_frame],0,1,linestyles='solid',alpha=1)
            ax[k].vlines(analysis.mov_timestamps_miliseconds['raw'][analysis.end_frame],0,1,linestyles='solid',alpha=1)
            ax[k].vlines(analysis.mov_timestamps_miliseconds['raw'][surrounddown], 0,1,linestyles='dashed',alpha=0.2)
            if surroundup.any():
                ax[k].vlines(analysis.mov_timestamps_miliseconds['raw'][surroundup], 0,1,linestyles='dashdot',alpha=0.2)
            ax[k].plot( timestamps_voltage_signals,sig ,label=f'{sigk}')
            ax[k].plot( upt,sig[upt] ,'^',label=f'{sigk} up')
            ax[k].plot( downt,sig[downt] ,'v',label=f'{sigk} down')
            ax[k].plot(analysis.mov_timestamps_miliseconds['raw'][upt_downsampled].astype(int),sig[ analysis.mov_timestamps_miliseconds['raw'][upt_downsampled].astype(int)] ,'<',label=f'{sigk} up downsampled')
            ax[k].plot(analysis.mov_timestamps_miliseconds['raw'][downt_downsampled].astype(int),sig[analysis.mov_timestamps_miliseconds['raw'][downt_downsampled].astype(int)] ,'>',label=f'{sigk} down downsampled')
            ax[k].vlines(analysis.mov_timestamps_miliseconds['raw'][surrounddown], 0,1,linestyles='dashed',alpha=0.2)
            if surroundup.any():
                ax[k].vlines(analysis.mov_timestamps_miliseconds['raw'][surroundup], 0,1,linestyles='dashdot',alpha=0.2)
                
            ax[k].set_title(f'{sigk}')
            ax[k].legend()
    
        ax[-1].plot(analysis.mov_timestamps_miliseconds['raw'], mov)
        ax[-1].plot(speedtimestamps, speed,alpha=0.2) 
        ax[-1].set_ylim(0,0.02)       
        ax[-1].set_title(f'Raw mean movie')
    
        f.suptitle(f'{record_to_plot} Alignment of VIdeo and signals to LED ')
    
                
    #plot led clipeed signals
    signals_to_plot=[ 'VisStim', 'PhotoStim',  'PhotoTrig', ]
    for record_to_plot, process in analysis.full_data['voltage_traces']['Full_signals'].items():
        f,ax=plt.subplots(len(signals_to_plot)+1,sharex=True)
        for k,sigk in enumerate(signals_to_plot):
            sig=analysis.scale_signal(process['LED_clipped']['traces'][sigk].values)
            locomotion=process['LED_clipped']['traces']['Locomotion'].values
            speed=analysis.scale_signal(rectified(deriv(locomotion)))
            speedtimestamps=process['LED_clipped']['traces']['Locomotion'].index.values
            mov_clipped=analysis.scale_signal(mean_mov[analysis.start_frame:analysis.end_frame])
            upt=analysis.full_data['voltage_traces']['Full_signals_transitions'][sigk]['aligned_downsampled_LEDshifted'][record_to_plot]['up']
            downt=analysis.full_data['voltage_traces']['Full_signals_transitions'][sigk]['aligned_downsampled_LEDshifted'][record_to_plot]['down']
     
            
            ax[k].plot( upt,sig[upt] ,'^',label=f'{sigk} up')
            ax[k].plot( downt,sig[downt] ,'v',label=f'{sigk} down')
            ax[k].plot(analysis.mov_timestamps_miliseconds['shifted'], mov_clipped)
            ax[k].plot(speedtimestamps,sig )
            # ax[k].plot(speedtimestamps,speed )
            ax[k].plot(analysis.mov_timestamps_miliseconds['shifted'][upt].astype(int),sig[ analysis.mov_timestamps_miliseconds['shifted'][upt].astype(int)] ,'<',label=f'{sigk} up downsampled')
            ax[k].plot(analysis.mov_timestamps_miliseconds['shifted'][downt].astype(int),sig[analysis.mov_timestamps_miliseconds['shifted'][downt].astype(int)] ,'>',label=f'{sigk} down downsampled')
            ax[k].vlines(analysis.mov_timestamps_miliseconds['shifted'][surrounddown], 0,1,linestyles='dashed',color='r',alpha=0.5)
            ax[k].vlines(analysis.mov_timestamps_miliseconds['shifted'][surroundup], 0,1,linestyles='dashdot',alpha=0.5)

            ax[k].set_title(f'{sigk}')
            ax[k].legend()
    
        ax[-1].plot(analysis.mov_timestamps_miliseconds['shifted'], mov_clipped)
        ax[-1].plot(speedtimestamps, speed,alpha=0.2) 
        ax[-1].set_title(f'Raw mean movie')

              
        f.suptitle(f'{record_to_plot} Alignment ofClipped  VIdeo and signals to LED ')

    
    plt.show()
    
def plot_optostimulated_cell_activities(analysis,speedtimestamps,speed, optocellindex_dict,traces_dict,transition_array,use_scaled=False,smoothwindow=False):
    
    titstr='raw'
    substracted=None
    selectedtraces=traces_dict['opto']['all']['All_planes']['demixed']
    if use_scaled:
        selectedtraces=scale_to_range(selectedtraces)
        titstr=titstr+'_scaled'
    trace_type=select_trace_type(substracted=substracted,smoothed=smoothwindow)
    
    if smoothwindow:

      optotraces,smoothwindow=smooth_activity_array(selectedtraces, smoothwindow)
      titstr=titstr+'_smoothed'
    else:
        optotraces=selectedtraces
        
        
    fig,ax=plt.subplots(optotraces.shape[0],sharex=True)
    if not isinstance(ax,np.ndarray ):
        ax=np.array([ax])
    fig.tight_layout()
    for i in range(optotraces.shape[0]):
        trace=optotraces[i,:]
        ax[i].plot(analysis.mov_timestamps_miliseconds['shifted']/1000,analysis.smooth_trace(trace,smoothwindow),'k')
        ax[i].plot(speedtimestamps/1000,analysis.scale_signal(speed),'r',alpha=0.5)
        ax[i].margins(x=0)
        if transition_array.shape[0]>1:
            m=i
        else:
            m=0
        for j in range(len(transition_array[m,:,0,0])):
            ax[i].axvline(x=analysis.mov_timestamps_miliseconds['shifted'][transition_array[m,j,0,0]]/1000)  
        ax[i].set_xlabel('Time(s)')
        ax[i].set_ylabel('Activity(a.u.)')
        ax[i].set_title(f'Optogenetic Stimulation of Chandelier Cell {i+1} {titstr}', fontsize=16)
    plt.show()
        

def plot_several_traces(analysis,speedtimestamps,speed, optocellindex_dict,traces_dict,transition_array,use_scaled=False,smoothwindow=False):
    substracted=None
    titstr='raw'

    selectedtraces=traces_dict['opto']['all']['All_planes']['demixed']
    selectedtraces_non_chand=traces_dict['non_chand']['non_opto']['All_planes']['demixed']
    
    all_traces=[selectedtraces,selectedtraces_non_chand]
    processed={}
    for i,traces in enumerate(all_traces):
        if use_scaled:
            traces=scale_to_range(traces)
        trace_type=select_trace_type(substracted=substracted,smoothed=smoothwindow)
        titstr=titstr+'scaled'

        if smoothwindow:
    
          traces,smoothwindow=smooth_activity_array(traces, smoothwindow)
          titstr=titstr+'_smoothed'
        processed[f'Plane{i+1}']=traces
   
    optotraces,traces_non_chand=processed.values()

    # #lpot clean traces of all opt and some non oppto cells
    plt.rcParams["figure.figsize"] = [16, 5]
    plt.rcParams["figure.autolayout"] = True
    non_chand_toplot=10
   
    totalcells=optotraces.shape[0]+non_chand_toplot
    f,ax=plt.subplots(totalcells+1,sharex=True)    
    for i in range(optotraces.shape[0]):
        trace=optotraces[i,:]
        ax[i].plot(analysis.mov_timestamps_miliseconds['shifted']/1000,analysis.smooth_trace(trace,10),c='y')
        for j in range(len(transition_array[i,:,0,0])):
            ax[i].axvline(x=analysis.mov_timestamps_miliseconds['shifted'][transition_array[i,j,0,0]]/1000)  
        
    for i in range(non_chand_toplot):
        trace=  traces_non_chand[i,:]
    
        ax[i+optotraces.shape[0]].plot(analysis.mov_timestamps_miliseconds['shifted']/1000,analysis.smooth_trace(trace,10),c='g')
        
    ax[-1].plot(speedtimestamps/1000,speed,'r')
    for i,a in enumerate(ax):
        a.margins(x=0)
        if i<len(ax)-1:
            a.axis('off')
        elif i==len(ax)-1:
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.spines['left'].set_visible(False)
            a.get_yaxis().set_ticks([])
            a.set_xlabel('Time(s)')
            
    f.suptitle(f'Raw Activity Examples {titstr}', fontsize=16)
    plt.show()

         
#%plot single optotirals individually 
def plot_optostimcell_single_trials(transition_array,optocellindex_dict,activity_dict,opto_repetitions,gratingcontrol,led_opt,substracted=False,smoothed=False):
    
    trace_type=select_trace_type(substracted=substracted,smoothed=smoothed)

    
    for l in range(len(optocellindex_dict['opto']['all']['All_planes'])):
        f,ax=plt.subplots(int(transition_array.shape[1]/2),2)
        if led_opt:
            k=0
        else:
            k=l
        for i,opto  in enumerate(transition_array[k,:,0,0]):
            row = i // 2  # determine the row index based on the iteration index
            col = i % 2   # determine the column index based on the iteration index
            # ax[row, col].plot(analysis.activity_dict['parameters']['trialtimevector'],analysis.scale_signal(smoothedoptotraces[l,l,i,:]),'k')
            ax[row, col].plot(activity_dict['parameters']['trialtimevector'],activity_dict['opto']['trials'][trace_type][optocellindex_dict['opto']['all']['All_planes'][l],k,i,:]),'k'
    
            # ax[row, col].plot(analysis.activity_dict['parameters']['trialtimevector'],smoothedoptotracessubstracetd[l,l,i,:],'b')
            ax[row, col].plot(  activity_dict['parameters']['trialspeedtimevector'],   activity_dict['speed']['trials']['raw_smoothed'][0,i,:],'r')
            ax[row, col].axvline(x=0)  
            # ax[row, col].set_ylim(-3,8)
            ax[row, col].margins(x=0)
            for m in range(opto_repetitions):               
                ax[row, col].add_patch(Rectangle((activity_dict['parameters']['trialtimevector'][activity_dict['parameters']['prestim']+activity_dict['parameters']['interoptoframes']*m], 0.8), 0.01, 0.2,color='r'))
    
            ax[row, col].set_xlabel('Time(s)')
            ax[row, col].set_ylabel('Activity(a.u.)')
            
        f.suptitle(f'Single Trial Optogenetic Stimulation Cell{str(l+1)} {trace_type}', fontsize=16)
        plt.show()

        
    if gratingcontrol.any():
        
        
        #plot single optotirals individually 
        for l in range(len(optocellindex_dict['opto']['all']['All_planes'])):
            f,ax=plt.subplots(int(gratingcontrol.shape[0]/2),2)
            for i,opto  in enumerate(gratingcontrol):
                row = i // 2  # determine the row index based on the iteration index
                col = i % 2   # determine the column index based on the iteration index
                # ax[row, col].plot(analysis.activity_dict['parameters']['trialtimevector'],analysis.scale_signal(smoothedoptotraces[l,l,i,:]),'k')
                ax[row, col].plot(activity_dict['parameters']['trialtimevector'],activity_dict['control']['trials'][trace_type][optocellindex_dict['opto']['all']['All_planes'][l],l,i,:]),'k'
    
                # ax[row, col].plot(analysis.activity_dict['parameters']['trialtimevector'],smoothedoptotracessubstracetd[l,l,i,:],'b')
                ax[row, col].plot(  activity_dict['parameters']['trialspeedtimevector'],activity_dict['speed']['trials']['raw_smoothed'][l,i,:],'r')
                ax[row, col].axvline(x=0)  
                # ax[row, col].set_ylim(-3,8)
                ax[row, col].margins(x=0)
           
                ax[row, col].set_xlabel('Time(s)')
                ax[row, col].set_ylabel('Activity(a.u.)')
                
            f.suptitle(f'Single Trial Grating Control Stimulation Cell{str(l+1)} {trace_type}', fontsize=16)
            plt.show()

def plot_optostimcell_tiled_cell(stimulated_cells_number,transition_array,optocellindex_dict,activity_dict,opto_repetitions,gratingcontrol,substracted=False,smoothed=False):
    
    trace_type=select_trace_type(substracted=substracted,smoothed=smoothed)

    # plot opto cells in tiled array with mean of all traces and then a individual figure for single trials

    if stimulated_cells_number>1:
        axestodo=stimulated_cells_number
        f,ax=plt.subplots(axestodo,axestodo,sharex=True, sharey=True)
    else:
        axestodo=1
        f,ax=plt.subplots(axestodo,sharex=True, sharey=True)
        if not isinstance(ax,np.ndarray):
            ax=np.array([[ax],[ax]])
            
    f.suptitle(f'OptoStimulated Chandelier Mean Activity {trace_type}', fontsize=16)

    plt.show()

    
    for stim_cell_trials in range(stimulated_cells_number):
        for cell_trace in range(stimulated_cells_number):
            if stim_cell_trials==cell_trace:
                color='r'
            else:
                color='b'
    
            ax[stim_cell_trials, cell_trace].plot(activity_dict['parameters']['trialtimevector'],activity_dict['opto']['mean'][trace_type][optocellindex_dict['opto']['all']['All_planes'][cell_trace],stim_cell_trials,:],color)
            ax[stim_cell_trials, cell_trace].axvline(x=0)  
            ax[stim_cell_trials, cell_trace].axis('off')
      
    for trial in range(nTrials):
        if stimulated_cells_number>1:
            axestodo=stimulated_cells_number
            f,ax=plt.subplots(axestodo,axestodo,sharex=True, sharey=True)
        else:
            axestodo=1
            f,ax=plt.subplots(axestodo,sharex=True, sharey=True)
            if not isinstance(ax,np.ndarray):
                ax=np.array([[ax],[ax]])
                
        f.suptitle(f'OptoStimulated Chandelier Trial {trial+1} Activity {trace_type}', fontsize=16)

        
        for stim_cell_trials in range(stimulated_cells_number):
            for cell_trace in range(stimulated_cells_number):
                if stim_cell_trials==cell_trace:
                    color='r'
                else:
                    color='b'
       
                ax[stim_cell_trials, cell_trace].plot(activity_dict['parameters']['trialtimevector'],activity_dict['opto']['trials'][trace_type][optocellindex_dict['opto']['all']['All_planes'][cell_trace],stim_cell_trials,trial,:],color)
                ax[stim_cell_trials, cell_trace].axvline(x=0)  
                ax[stim_cell_trials, cell_trace].axis('off')
        plt.show()

                
def plot_rasters_treatment(optocellindex_dict,activity_dict,substracted=False,smoothed=False):
    optocell=optocellindex_dict['opto']['all']['All_planes'][0]
    optocell=0
    trace_type=select_trace_type(substracted=substracted,smoothed=smoothed)

    ordered_by_distance=np.argsort(accepted_all_plane_distances)
    
    non_opto_mean_trial_traces=  activity_dict['opto']['mean'][trace_type][optocellindex_dict['non_chand']['non_opto']['All_planes'],optocell,:]
    meanoptoresponse=  activity_dict['opto']['mean'][trace_type][optocellindex_dict['non_chand']['non_opto']['All_planes'],optocell,np.argmin(abs(activity_dict['parameters']['trialtimevector']-0.5)):].mean(axis=1)
    non_opto_mean_trial_control_traces=  activity_dict['control']['mean'][trace_type][optocellindex_dict['non_chand']['non_opto']['All_planes'],optocell,:]
                    
    f, ax = plt.subplots(2,1, figsize=(12, 12),sharex=True)
    ax[0].imshow( non_opto_mean_trial_traces[np.flip(np.argsort(meanoptoresponse)),:])      
    ax[1].imshow(non_opto_mean_trial_control_traces[np.flip(np.argsort(meanoptoresponse)),:])   
    
    # f, ax = plt.subplots(2,1, figsize=(12, 12),sharex=True)
    # ax[0].imshow( non_opto_mean_trial_traces[np.flip(np.argsort(accepted_all_plane_distances[optocellindex_dict['non_chand']['non_opto']['All_planes'][0]])),:])      
    # ax[1].imshow(non_opto_mean_trial_control_traces[np.flip(np.argsort(accepted_all_plane_distances[optocellindex_dict['non_chand']['non_opto']['All_planes'][0]])),:]) 
     
       
    ticks =np.linspace(0,non_opto_mean_trial_traces.shape[1], 9)
    lab=np.arange(scaled_activity_dict['parameters']['trialtimevector'][0],scaled_activity_dict['parameters']['trialtimevector'][-1]+0.5,0.5)
    ticklabels = ["{:6.2f}".format(i) for i in lab]
    for i in range(len(ax)):
        ax[i].set_xticks(ticks)
        ax[i].set_xticklabels(ticklabels)
        ax[i].axvline(x=ticks[np.argwhere(lab==0)[0][0]])  
        
        ax[i].set_xlabel('Time(s)')
        ax[i].set_ylabel('Cell')
        ax[i].set_aspect('equal')
        ax[i].margins(x=0)
    ax[0].set_title('Opto')
    ax[1].set_title('Ctrl')


    f.suptitle(f'Tomato- Averaged Activity (OptoTrial Grating) {trace_type}', fontsize=16)

    # f.suptitle(f'Tomato- Averaged Activity (OptoTrial Grating)', fontsize=16)
    # f.suptitle(f'Tomato- Averaged Activity (Non Chand Grating) {trace_type}', fontsize=16)
    # f.suptitle(f'Tomato- Averaged Activity (OptoTrial Background)', fontsize=16)
    # f.suptitle(f'Tomato- Averaged Activity (Background Grating)', fontsize=16)

    
    f.tight_layout()
    plt.show()
def plot_nono_opto_single_trials(transition_array,optocellindex_dict,activity_dict,opto_repetitions,led_opt,substracted=False,smoothed=False):
    
    trace_type=select_trace_type(substracted=substracted,smoothed=smoothed)

#%% basic setup
all_analysis=selected_analysis
selected=all_analysis[0]

tempprocessingpat= r'C:\Users\sp3660\Desktop\TempPythonObjects'
experimentalmousename= selected['analysis'].acquisition_object.mouse_imaging_session_object.mouse_object.mouse_name
datapath=os.path.join(tempprocessingpat,experimentalmousename)
dataindex=0


temp_data_list=check_temp_data(tempprocessingpat)
multiple_analysis=load_temp_data(temp_data_list,dataindex)


if not multiple_analysis:
    for selected in all_analysis:   
        analysis=selected['analysis']
        full_data=selected['full_data']
        # tt=analysis.signals_object.signal_transitions
        # analysis.signals_object.extract_transitions_optodrift('VisStim', led_clipped=True, plot=False)
        # analysis.signals_object.optodrift_info
    
        final_stim_onsets=process_vis_stim(analysis) # carfeul here i run downsample onsets and might give error
    
        optograting=final_stim_onsets['opto_grating']
        aq=analysis.acquisition_object
        # analysis.review_aligned_signals_and_transitions()
        plane=1
        dtset=analysis.calcium_datasets[list(analysis.calcium_datasets.keys())[plane-1]]
        # dtset.most_updated_caiman.CaimanResults_object.open_caiman_sorter()
        chand_indexes=load_chandelier_cell_file(analysis)
        
    
        visstimtranstionts=np.zeros((8,20,2,2)).astype(np.uint16)
        for i in range(1,9):
            visstimtranstionts[i-1,:,:,0]=analysis.signals_object.optodrift_info[f'Grat_{i}']['ArrayFinal_downsampled_LED_clipped']
            visstimtranstionts[i-1,:,:,1]=analysis.signals_object.optodrift_info[f'Grat_{i}']['ArrayFinalOffset_downsampled_LED_clipped']
    
            
        gratingcontrol=np.empty(0)
        try:
            optograting=analysis.acquisition_object.visstimdict['opto']['randomoptograting']
            optogratinfo=analysis.signals_object.optodrift_info[f'Grat_{optograting}']
            optoblankinfo=analysis.signals_object.optodrift_info['Blank']
        
            gratingcontrol=optogratinfo['ArrayFinal_downsampled_LED_clipped'][:,1]
            # opto_gratdifference=analysis.full_data['voltage_traces']['Full_signals_transitions']['PhotoTrig']['aligned_downsampled_LEDshifted']['Prairie']['up']-optogratinfo['ArrayFinal_downsampled_LED_clipped'][:,0]
            
        except:
            print('Non optograting')
        
        
        ext=analysis.caiman_extractions[list(analysis.caiman_results.keys())[0]]
        cnmf=ext.cnm_object
        dff=cnmf.estimates.detrend_df_f()
        analysis.accepteddff=dff.F_dff[analysis.full_data['imaging_data']['Plane1']['CellIds'],:]
        
        locomotion=analysis.full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['Locomotion'].values
        
        speed=rectified(deriv(locomotion))
        speedtimestamps=analysis.full_data['voltage_traces']['Full_signals']['Prairie']['LED_clipped']['traces']['Locomotion'].index.values
        
        # analysis.full_data['voltage_traces']['Full_signals']
        # analysis.full_data['voltage_traces']['Full_signals_transitions']
        # analysis.signals_object.all_final_signals['Prairie']['LED_clipped']['traces']['PhotoStim']
        
        zz=analysis.acquisition_object.visstimdict
        total_opt_number=len(analysis.full_data['voltage_traces']['Full_signals_transitions']['PhotoStim']['aligned_downsampled_LEDshifted']['Prairie']['up'])
        # trial_delay=analysis.acquisition_object.visstimdict['opto']['intertrialtime']
        if hasattr(analysis.acquisition_object.metadata_object, 'mark_points_experiment'):
            nTrials=analysis.acquisition_object.metadata_object.mark_points_experiment['Iterations']
            stimulated_cells_number=len(analysis.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries'])
            opto_repetitions=int(analysis.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries']['PhotostimExperiment_1']['sequence']['Repetitions'])
            frequency=analysis.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries']['PhotostimExperiment_1']['sequence']['RepFrequency']
            opto_duration=analysis.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries']['PhotostimExperiment_1']['sequence']['StimDuration']
            iteration_duration=analysis.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries']['PhotostimExperiment_1']['sequence']['StimDuration']
            inter_rep_time=analysis.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries']['PhotostimExperiment_1']['sequence']['RepTime']
            inter_point_time=analysis.acquisition_object.metadata_object.mark_points_experiment['PhotoStimSeries']['PhotostimExperiment_1']['sequence']['InterpointDuration']
            led_opt=False
            triggers=nTrials
        
        
        else:
            
            optoinfo=aq.visstimdict['opto']
            opto_repetitions=optoinfo['number_of_pulses']
            stimulated_cells_number=1
            nTrials=len(aq.visstimdict['ops']['paradigm_sequence'])
            frequency=optoinfo['pulse_frequency']
            inter_rep_time=optoinfo['period']
            led_opt=True   
            triggers=len(analysis.full_data['voltage_traces']['Full_signals_transitions']['PhotoTrig']['aligned_downsampled_LEDshifted']['Prairie']['up'])
        
            
        metadata_total_opt_number=opto_repetitions*stimulated_cells_number*nTrials
        blank_opt=False
        
        if len(analysis.full_data['voltage_traces']['Full_signals_transitions']['PhotoTrig']['aligned_downsampled_LEDshifted']['Prairie']['up'])>400:
            blank_opt=True
            opto_blank_sweeps_pulses=20*2
            control_blank_sweeps=20*2
            total_opto_blank_swweps_pulses=opto_blank_sweeps_pulses*nTrials
            metadata_total_opt_number=metadata_total_opt_number+total_opto_blank_swweps_pulses
            'no photostim'
            analysis.full_data['voltage_traces']['Full_signals_transitions']['PhotoStim']=analysis.full_data['voltage_traces']['Full_signals_transitions']['PhotoTrig']
        
        signals_to_review=['PhotoStim','PhotoTrig']
        metadata_transitions=[metadata_total_opt_number,triggers]
        for sig in zip(signals_to_review,metadata_transitions):
            transitions=len(analysis.full_data['voltage_traces']['Full_signals_transitions'][sig[0]]['aligned_downsampled_LEDshifted']['Prairie']['up'])
            if sig[1]==transitions:
                good='Correct'
            else:
                good='Incorrect'
            print(f'{good} number of {sig[0]} transitions detected')
        if 1/frequency==inter_rep_time:
            print('Seems correct opto timings)')
        fr=analysis.full_data['imaging_data']['Frame_rate']   
        
        review=0
        if review:
            analysis.review_aligned_signals_and_transitions()
    
    
# process cells and tramsitions
        
    
        sweepscale=True
        sweepsmoothed=True
        sweepdff=True
        optocellindex_dict,traces_dict,accepted_all_plane_distances=find_optostim_cells(analysis,chand_indexes)
        traces=traces_dict['all']['all']['All_planes']['demixed']
        sweep_response, mean_sweep_response, pval, full_data=get_sweep_response(analysis,full_data,traces,blank_opt,visstimtranstionts,use_scaled=sweepscale,smoothwindow=sweepsmoothed,dff=sweepdff)
        response=get_response(analysis,full_data,mean_sweep_response,pval )
        peak=get_peak(full_data,response,sweep_response,mean_sweep_response)
    
    
        transition_array=create_opto_transitions_array(analysis,stimulated_cells_number,nTrials,opto_repetitions,led_opt,blank_opt)
        pretime=1
        posttime=3
        
        activity_dict= slice_by_optotrial(stimulated_cells_number,pretime,posttime,fr,transition_array,traces_dict,gratingcontrol,use_scaled=False)
        scaled_activity_dict= slice_by_optotrial(stimulated_cells_number,pretime,posttime,fr,transition_array,traces_dict,gratingcontrol,use_scaled=True)   
        
        
        activity_dict_peak=mean_opto_response(activity_dict,stat='mean',stimtime=2)       
        scaled_activity_dict_peak=mean_opto_response(scaled_activity_dict,stat='max',stimtime=2)    
        
        
        # ac=activity_dict_peak
        ac=scaled_activity_dict_peak
        li1 = {'mean_peak':ac['opto']['mean']['raw_smoothed_stim_peak'].flatten(), 'treatment':['opto'] * len(ac['opto']['mean']['raw_smoothed_stim_peak'].flatten())}
        li2 = {'mean_peak':ac['control']['mean']['raw_smoothed_stim_peak'].flatten() ,'treatment':['control'] * len(ac['opto']['mean']['raw_smoothed_stim_peak'].flatten())}
        df1=pd.DataFrame(li1)
        df2=pd.DataFrame(li2)
        peaks=pd.concat([df1,df2])
    
        all_info={'activity_dict':activity_dict,
                  'activity_dict_peak':activity_dict_peak,
                  'scaled_activity_dict':scaled_activity_dict,
                  'scaled_activity_dict_peak':scaled_activity_dict_peak,
                  'peaks':peaks,
                  'optocellindex_dict':optocellindex_dict,
                  'traces_dict':traces_dict,
                  'transition_array':transition_array,
                  'chand_indexes':chand_indexes,
                  'stimulated_cells_number':stimulated_cells_number,
                  'nTrials':nTrials,
                  'opto_repetitions':opto_repetitions,
                  'led_opt':led_opt,
                  'pretime':pretime,
                  'posttime':posttime,
                  'fr':fr,
                  'gratingcontrol':gratingcontrol,
                  'speed':speed,
                  'speedtimestamps':speedtimestamps,
                  'accepted_all_plane_distances':accepted_all_plane_distances,
                  'visstimtranstionts':visstimtranstionts,
                  'sweep_response':sweep_response,
                  'mean_sweep_response':mean_sweep_response,
                  'pval':pval,
                  'response':response,
                  'peak':peak,               
                  'sweepscale':sweepscale,
                  'sweepsmoothed':sweepsmoothed ,
                  'dff':sweepdff
                  }
        
        multiple_analysis[analysis.acquisition_object.aquisition_name]=all_info
        
    datapath=save_temp_data(multiple_analysis,datapath)

else:
    for selected in all_analysis:  
        analysis=selected['analysis']
        multiple_analysis[analysis.acquisition_object.aquisition_name]['visstimtranstionts']


        sorted_df,pre_frames,post_frames=get_stimulus_table(analysis,visstimtranstionts)
     
#%% manual play


i=0
experiment=list(multiple_analysis.keys())[i]
aq_analysis=all_analysis[i]['analysis']
aq_all_info=multiple_analysis[experiment]


aq_all_info['gratingcontrol']
aq_all_info['transition_array'][0,:,0,0]
aq_all_info.keys()

    
stimt=aq_analysis.full_data['visstim_info']['OptoDrift']['stimulus_table']
stimt.opto==1
stimt.blank_sweep==0
optotrialsst=stimt[(stimt.opto==1) & (stimt.blank_sweep==0)]
orival=optotrialsst.orientation.unique()[0]
optotrialcontrolsst=stimt[(stimt.opto==0) & (stimt.orientation==orival)]


optotrialblanksst=stimt[(stimt.opto==1) & (stimt.blank_sweep==1)]
optotrialblankcontrolsst=stimt[(stimt.opto==0) & (stimt.blank_sweep==1)]



tt=aq_all_info['sweep_response']
tt=aq_all_info['mean_sweep_response']
tt=aq_all_info['pval']
tt=aq_all_info['response']
tt=aq_all_info['peak']



#%%plotting full recordings traces
plt.close('all')
# plot_optostimulated_cell_activities(aq_all_info['analysis'],aq_all_info['speedtimestamps'],aq_all_info['speed'], aq_all_info['optocellindex_dict'],aq_all_info['traces_dict'],aq_all_info['transition_array'],use_scaled=False,smoothwindow=False)
# plot_optostimulated_cell_activities(aq_all_info['analysis'],aq_all_info['speedtimestamps'],aq_all_info['speed'], aq_all_info['optocellindex_dict'],aq_all_info['traces_dict'],aq_all_info['transition_array'],use_scaled=False,smoothwindow=True)
# plot_optostimulated_cell_activities(aq_all_info['analysis'],aq_all_info['speedtimestamps'],aq_all_info['speed'], aq_all_info['optocellindex_dict'],aq_all_info['traces_dict'],aq_all_info['transition_array'],use_scaled=True,smoothwindow=False)
plot_optostimulated_cell_activities(aq_all_info['analysis'],aq_all_info['speedtimestamps'],aq_all_info['speed'], aq_all_info['optocellindex_dict'],aq_all_info['traces_dict'],aq_all_info['transition_array'],use_scaled=True,smoothwindow=True)

# plot_several_traces(aq_all_info['analysis'],aq_all_info['speedtimestamps'],aq_all_info['speed'],aq_all_info['optocellindex_dict'],aq_all_info['traces_dict'],aq_all_info['transition_array'],use_scaled=False,smoothwindow=False)
# plot_several_traces(aq_all_info['analysis'],aq_all_info['speedtimestamps'],aq_all_info['speed'],aq_all_info['optocellindex_dict'],aq_all_info['traces_dict'],aq_all_info['transition_array'],use_scaled=False,smoothwindow=True)
# plot_several_traces(aq_all_info['analysis'],aq_all_info['speedtimestamps'],aq_all_info['speed'],aq_all_info['optocellindex_dict'],aq_all_info['traces_dict'],aq_all_info['transition_array'],use_scaled=True,smoothwindow=False)
plot_several_traces(aq_all_info['analysis'],aq_all_info['speedtimestamps'],aq_all_info['speed'],aq_all_info['optocellindex_dict'],aq_all_info['traces_dict'],aq_all_info['transition_array'],use_scaled=True,smoothwindow=True)

   
#%% processng trials

# meanswepresponse filter for non cha ndlier and significancly opt have to select the optograting and comare opto vs non opto

aq_all_info['mean_sweep_response']
aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes']
aq_analysis.full_data['visstim_info']['OptoDrift']['stimulus_table']
aq_all_info['pval']
 

optononchandpval=aq_all_info['pval'].iloc[optotrialsst.index.values,aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes']]
optononchandpvalcontrol=aq_all_info['pval'].iloc[optotrialcontrolsst.index.values,aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes']]
optononchandblankpval=aq_all_info['pval'].iloc[optotrialblanksst.index.values,aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes']]
optononchandpvalblankcontrol=aq_all_info['pval'].iloc[optotrialblankcontrolsst.index.values,aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes']]

optononchandmeansweep=aq_all_info['mean_sweep_response'].iloc[optotrialsst.index.values,aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes']]
optononchandmeansweepcontrol=aq_all_info['mean_sweep_response'].iloc[optotrialcontrolsst.index.values,aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes']]
optononchandblankmeansweep=aq_all_info['mean_sweep_response'].iloc[optotrialblanksst.index.values,aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes']]
optononchandblankmeansweepcontrol=aq_all_info['mean_sweep_response'].iloc[optotrialblankcontrolsst.index.values,aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes']]


optononchandsweep=aq_all_info['sweep_response'].iloc[optotrialsst.index.values,aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes']]
optononchandsweepcontrol=aq_all_info['sweep_response'].iloc[optotrialcontrolsst.index.values,aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes']]
optononchandblanksweep=aq_all_info['sweep_response'].iloc[optotrialblanksst.index.values,aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes']]
optononchandblanksweepcontrol=aq_all_info['sweep_response'].iloc[optotrialblankcontrolsst.index.values,aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes']]




df=optononchandpval<0.05
df2=optononchandpvalcontrol<0.05
df3=optononchandblankpval<0.05
df4=optononchandpvalblankcontrol<0.05

numbersignificanttrials=df.sum()
numbersignificancontrol=df2.sum()
numbersignificanttrialsblank=df4.sum()
numbersignificanblankcontrol=df2.sum()

tt=numbersignificanttrials>=10
ttt=numbersignificancontrol>=10
tttt=numbersignificanttrialsblank>=10
ttttt=numbersignificanblankcontrol>=10

significantcells=[int(c) for c in tt.index[np.argwhere(tt.values).flatten()]]
significantcellscontrol=[int(c) for c in ttt.index[np.argwhere(ttt.values).flatten()]]
significantcellsblank=[int(c) for c in tttt.index[np.argwhere(tttt.values).flatten()]]
significantcellsblankcontrol=[int(c) for c in ttttt.index[np.argwhere(ttttt.values).flatten()]]


nonsignificantcells= np.delete(aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes'], np.where(np.in1d(aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes'], significantcells))[0])
nonsignificantcellscontrol= np.delete(aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes'], np.where(np.in1d(aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes'], significantcellscontrol))[0])

nonsignificantcellsblank= np.delete(aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes'], np.where(np.in1d(aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes'], significantcellsblank))[0])
nonsignificantcellsblankcontrol= np.delete(aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes'], np.where(np.in1d(aq_all_info['optocellindex_dict']['non_chand']['non_opto']['All_planes'], significantcellsblankcontrol))[0])



significanmeansweep=aq_all_info['mean_sweep_response'].iloc[optotrialsst.index,significantcells].mean()
nonsignificantmeansweep=aq_all_info['mean_sweep_response'].iloc[optotrialsst.index,nonsignificantcells].mean()
significanmeansweepcontrol=aq_all_info['mean_sweep_response'].iloc[optotrialcontrolsst.index,significantcells].mean()
nonsignificantmeansweepcontrol=aq_all_info['mean_sweep_response'].iloc[optotrialcontrolsst.index,nonsignificantcells].mean()


significansweep=aq_all_info['sweep_response'].iloc[optotrialsst.index,significantcells]
nonsignificantsweep=aq_all_info['sweep_response'].iloc[optotrialsst.index,nonsignificantcells]
significanmweepcontrol=aq_all_info['sweep_response'].iloc[optotrialcontrolsst.index,significantcells]
nonsignificantsweepcontrol=aq_all_info['sweep_response'].iloc[optotrialcontrolsst.index,nonsignificantcells]

significanblankmeansweep=aq_all_info['mean_sweep_response'].iloc[optotrialblanksst.index,significantcellsblank].mean()
nonsignificantblankmeansweep=aq_all_info['mean_sweep_response'].iloc[optotrialblanksst.index,nonsignificantcellsblank].mean()
significanblankmeansweepcontrol=aq_all_info['mean_sweep_response'].iloc[optotrialblankcontrolsst.index,significantcellsblank].mean()
nonsignificantblankmeansweepcontrol=aq_all_info['mean_sweep_response'].iloc[optotrialblankcontrolsst.index,nonsignificantcellsblank].mean()

significanblanksweep=aq_all_info['sweep_response'].iloc[optotrialblanksst.index,significantcellsblank]
significanblanksweepcontrol=aq_all_info['sweep_response'].iloc[optotrialblankcontrolsst.index,significantcellsblank]



all_rows=[]

for i,cell in enumerate(significantcells):
    sig=1
    contsig=0
  
    if cell in significantcellscontrol:
        contsig=1

    all_rows.append((cell,sig,contsig, significanmeansweep.iloc[i],significanmeansweepcontrol.iloc[i]))

for i,cell in enumerate(nonsignificantcells):
    sig=0
    contsig=0
  
    if cell in significantcellscontrol:
        contsig=1
        
        
    all_rows.append((cell,sig,contsig, nonsignificantmeansweep.iloc[i],nonsignificantmeansweepcontrol.iloc[i]))


optogratingdf = pd.DataFrame(all_rows, columns =['cellid','optosignificant','controlsignigficant','meanopto','meancontrol'])
                                      
                                      

all_rows=[]

for i,cell in enumerate(significantcellsblank):
    sig=1
    contsig=0
  
    if cell in significantcellsblankcontrol:
        contsig=1

    all_rows.append((cell,sig,contsig, significanblankmeansweep.iloc[i],significanblankmeansweepcontrol.iloc[i]))

for i,cell in enumerate(nonsignificantcellsblank):
    sig=0
    contsig=0
  
    if cell in significantcellsblankcontrol:
        contsig=1
        
        
    all_rows.append((cell,sig,contsig, nonsignificantblankmeansweep.iloc[i],nonsignificantblankmeansweepcontrol.iloc[i]))



optoblankdf = pd.DataFrame(all_rows, columns =['cellid','optosignificant','controlsignigficant','meanopto','meancontrol'])



f,axs=plt.subplots(1,len(multiple_analysis))
f,axs=plt.subplots(1,2)

for i, aq_all_info in enumerate(multiple_analysis.values()):
    sns.barplot(ax=axs[i], x="optosignificant", y="meanopto", data=df, capsize=.1, ci="sd")
    sns.swarmplot(ax=axs[i], x="optosignificant", y="meanopto", data=df, color="0", alpha=.35)
    axs[i].set_ylim(0,1)
plt.show()

signif=optogratingdf[(optogratingdf['meanopto']-optogratingdf['meancontrol']>0) & (optogratingdf['controlsignigficant']==0) & (optogratingdf['optosignificant']==1)]
nonsignif=optogratingdf[(optogratingdf['controlsignigficant']==0) & (optogratingdf['optosignificant']==0)]
result = pd.concat([signif,nonsignif])
f,axs=plt.subplots(1,2,figsize=(9,4.5),dpi=300)
sns.barplot(ax=axs[0], x="optosignificant", y="meanopto", data=result, capsize=.1, ci="sd")
sns.swarmplot(ax=axs[0], x="optosignificant", y="meanopto", data=result, color="0", alpha=.35)
sns.barplot(ax=axs[1], x="optosignificant", y="meancontrol", data=result, capsize=.1, ci="sd")
sns.swarmplot(ax=axs[1], x="optosignificant", y="meancontrol", data=result, color="0", alpha=.35)
axs[0].set_title('Opto Trials')
axs[1].set_title('Control Trials')
f.suptitle('Opto + Grating Activated')


signif=optogratingdf[(optogratingdf['meanopto']-optogratingdf['meancontrol']<0) & (optogratingdf['controlsignigficant']==0) & (optogratingdf['optosignificant']==1)]
nonsignif=optogratingdf[(optogratingdf['controlsignigficant']==0) & (optogratingdf['optosignificant']==0)]
result = pd.concat([signif,nonsignif])
f2,axs2=plt.subplots(1,2,figsize=(9,4.5),dpi=300)
sns.barplot(ax=axs2[0], x="optosignificant", y="meanopto", data=result, capsize=.1, ci="sd")
sns.swarmplot(ax=axs2[0], x="optosignificant", y="meanopto", data=result, color="0", alpha=.35)
sns.barplot(ax=axs2[1], x="optosignificant", y="meancontrol", data=result, capsize=.1, ci="sd")
sns.swarmplot(ax=axs2[1], x="optosignificant", y="meancontrol", data=result, color="0", alpha=.35)
axs2[0].set_title('Opto Trials')
axs2[1].set_title('Control Trials')
f2.suptitle('Opto + Grating Inhibited')




signif=optoblankdf[(optoblankdf['meanopto']-optoblankdf['meancontrol']>0) & (optoblankdf['controlsignigficant']==0) & (optoblankdf['optosignificant']==1)]
nonsignif=optoblankdf[(optoblankdf['controlsignigficant']==0) & (optoblankdf['optosignificant']==0)]
result = pd.concat([signif,nonsignif])
f3,axs3=plt.subplots(1,2,figsize=(9,4.5),dpi=300)
sns.barplot(ax=axs3[0], x="optosignificant", y="meanopto", data=result, capsize=.1, ci="sd")
sns.swarmplot(ax=axs3[0], x="optosignificant", y="meanopto", data=result, color="0", alpha=.35)
sns.barplot(ax=axs3[1], x="optosignificant", y="meancontrol", data=result, capsize=.1, ci="sd")
sns.swarmplot(ax=axs3[1], x="optosignificant", y="meancontrol", data=result, color="0", alpha=.35)
axs3[0].set_title('Opto Trials')
axs3[1].set_title('Control Trials')
f3.suptitle('Opto Only Activated')


signif=optoblankdf[(optoblankdf['meanopto']-optoblankdf['meancontrol']<0) & (optoblankdf['controlsignigficant']==0) & (optoblankdf['optosignificant']==1)]
nonsignif=optoblankdf[(optoblankdf['controlsignigficant']==0) & (optoblankdf['optosignificant']==0)]
result = pd.concat([signif,nonsignif])
f4,axs=plt.subplots(1,2,figsize=(9,4.5),dpi=300)
sns.barplot(ax=axs[0], x="optosignificant", y="meanopto", data=result, capsize=.1, ci="sd")
sns.swarmplot(ax=axs[0], x="optosignificant", y="meanopto", data=result, color="0", alpha=.35)
sns.barplot(ax=axs[1], x="optosignificant", y="meancontrol", data=result, capsize=.1, ci="sd")
sns.swarmplot(ax=axs[1], x="optosignificant", y="meancontrol", data=result, color="0", alpha=.35)
axs[0].set_title('Opto Trials')
axs[1].set_title('Control Trials')
f4.suptitle('Opto Only Inhibited')
plt.show()

plt.figure()
plt.plot(np.vstack(significansweep.iloc[:,0].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,1].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,2].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,3].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,4].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,5].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,6].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,7].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,8].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,9].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,10].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,11].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,12].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,13].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,14].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,15].values).mean(axis=0))
plt.plot(np.vstack(significansweep.iloc[:,16].values).mean(axis=0))
plt.show()


plt.figure()
plt.plot(np.vstack(significanmweepcontrol.iloc[:,0].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,1].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,2].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,3].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,4].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,5].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,6].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,7].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,8].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,9].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,10].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,11].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,12].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,13].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,14].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,15].values).mean(axis=0))
plt.plot(np.vstack(significanmweepcontrol.iloc[:,16].values).mean(axis=0))
plt.show()


plt.figure()
plt.plot(np.vstack(significanblanksweep.iloc[:,0].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,1].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,2].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,3].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,4].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,5].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,6].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,7].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,8].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,9].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,10].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,11].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,12].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,13].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,14].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,15].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweep.iloc[:,16].values).mean(axis=0))
plt.show()

plt.figure()
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,0].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,1].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,2].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,3].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,4].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,5].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,6].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,7].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,8].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,9].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,10].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,11].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,12].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,13].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,14].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,15].values).mean(axis=0))
plt.plot(np.vstack(significanblanksweepcontrol.iloc[:,16].values).mean(axis=0))
plt.show()
#%% plotting
# non scaled activity
plt.close('all')
# 
aq_all_info
f,axs=plt.subplots(1,2)
aq_all_info=multiple_analysis[list(multiple_analysis.keys())[0]]
for i, aq_all_info in multiple_analysis.items():
    
    # analysis=aq_all_info['analysis']
    # speedtimestamps=aq_all_info['speedtimestamps']
    # optocellindex_dict=aq_all_info['optocellindex_dict']
    # traces_dict=aq_all_info['traces_dict']
    # transition_array=aq_all_info['transition_array']
    
    
    plot_optostimulated_cell_activities(aq_all_info['analysis'],aq_all_info['speedtimestamps'],aq_all_info['speed'],aq_all_info['optocellindex_dict'],aq_all_info['traces_dict'],aq_all_info['transition_array'],use_scaled=True,smoothwindow=True)
    # plot_several_traces(analysis,speedtimestamps,aq_all_info['speed'],  optocellindex_dict,traces_dict,transition_array,use_scaled=True,smoothwindow=True)
    plot_rasters_treatment(aq_all_info['optocellindex_dict'],aq_all_info['scaled_activity_dict'],substracted=True,smoothed=True)
    plot_rasters_treatment(aq_all_info['optocellindex_dict'],aq_all_info['activity_dict'],substracted=True,smoothed=True)
    plot_optostimcell_single_trials(aq_all_info['transition_array'],aq_all_info['optocellindex_dict'],aq_all_info['scaled_activity_dict'],aq_all_info['opto_repetitions'],aq_all_info['gratingcontrol'],aq_all_info['led_opt'],substracted=True,smoothed=True)
    # plot_optostimcell_single_trials(aq_all_info['transition_array'],aq_all_info['optocellindex_dict'],aq_all_info['activity_dict'],aq_all_info['opto_repetitions'],aq_all_info['gratingcontrol'],aq_all_info['led_opt'],substracted=True,smoothed=True)



#%%

    
    
# plot_rasters_treatment(optocellindex_dict,activity_dict,substracted=False,smoothed=False)
# plot_rasters_treatment(optocellindex_dict,activity_dict,substracted=False,smoothed=True)
# plot_rasters_treatment(optocellindex_dict,activity_dict,substracted=True,smoothed=False)
# plot_rasters_treatment(optocellindex_dict,activity_dict,substracted=True,smoothed=True)

# plot_optostimcell_tiled_cell(aq_all_info['stimulated_cells_number'],aq_all_info['transition_array'],aq_all_info['optocellindex_dict'],aq_all_info['activity_dict'],aq_all_info['opto_repetitions'],aq_all_info['gratingcontrol'],substracted=False,smoothed=False)
# plot_optostimcell_tiled_cell(aq_all_info['stimulated_cells_number'],aq_all_info['transition_array'],aq_all_info['optocellindex_dict'],aq_all_info['activity_dict'],aq_all_info['opto_repetitions'],aq_all_info['gratingcontrol'],substracted=False,smoothed=True)
# plot_optostimcell_tiled_cell(aq_all_info['stimulated_cells_number'],aq_all_info['transition_array'],aq_all_info['optocellindex_dict'],aq_all_info['activity_dict'],aq_all_info['opto_repetitions'],aq_all_info['gratingcontrol'],substracted=True,smoothed=False)
plot_optostimcell_tiled_cell(aq_all_info['stimulated_cells_number'],aq_all_info['transition_array'],aq_all_info['optocellindex_dict'],aq_all_info['activity_dict'],aq_all_info['opto_repetitions'],aq_all_info['gratingcontrol'],substracted=True,smoothed=True)

# plot_optostimcell_single_trials(transition_array,optocellindex_dict,activity_dict,opto_repetitions,gratingcontrol,led_opt,substracted=False,smoothed=False)
plot_optostimcell_single_trials(transition_array,optocellindex_dict,activity_dict,opto_repetitions,gratingcontrol,led_opt,substracted=False,smoothed=True)
# plot_optostimcell_single_trials(transition_array,optocellindex_dict,activity_dict,opto_repetitions,gratingcontrol,led_opt,substracted=True,smoothed=False)
# plot_optostimcell_single_trials(transition_array,optocellindex_dict,activity_dict,opto_repetitions,gratingcontrol,led_opt,substracted=True,smoothed=True)



# scaled activity

# plot_rasters_treatment(optocellindex_dict,scaled_activity_dict,substracted=False,smoothed=False)
# plot_rasters_treatment(optocellindex_dict,scaled_activity_dict,substracted=False,smoothed=True)
# plot_rasters_treatment(optocellindex_dict,scaled_activity_dict,substracted=True,smoothed=False)
# plot_rasters_treatment(optocellindex_dict,scaled_activity_dict,substracted=True,smoothed=True)
# plot_rasters_treatment(optocellindex_dict,scaled_activity_dict,substracted=True,smoothed=True)

        
# plot_optostimcell_tiled_cell(stimulated_cells_number,transition_array,optocellindex_dict,scaled_activity_dict,opto_repetitions,gratingcontrol,substracted=False,smoothed=False)
# plot_optostimcell_tiled_cell(stimulated_cells_number,transition_array,optocellindex_dict,scaled_activity_dict,opto_repetitions,gratingcontrol,substracted=False,smoothed=True)
# plot_optostimcell_tiled_cell(stimulated_cells_number,transition_array,optocellindex_dict,scaled_activity_dict,opto_repetitions,gratingcontrol,substracted=True,smoothed=False)
plot_optostimcell_tiled_cell(stimulated_cells_number,transition_array,optocellindex_dict,scaled_activity_dict,opto_repetitions,gratingcontrol,substracted=True,smoothed=True)

# plot_optostimcell_single_trials(transition_array,optocellindex_dict,scaled_activity_dict,opto_repetitions,gratingcontrol,led_opt,substracted=False,smoothed=False)
# plot_optostimcell_single_trials(transition_array,optocellindex_dict,scaled_activity_dict,opto_repetitions,gratingcontrol,led_opt,substracted=False,smoothed=True)
# plot_optostimcell_single_trials(transition_array,optocellindex_dict,scaled_activity_dict,opto_repetitions,gratingcontrol,led_opt,substracted=True,smoothed=False)
plot_optostimcell_single_trials(transition_array,optocellindex_dict,scaled_activity_dict,opto_repetitions,gratingcontrol,led_opt,substracted=True,smoothed=True)
        #%%
for selectedgrating in range(8):
    grattrialarray=np.zeros((stimulated_cells_number,nTrials,1,1)).astype(np.uint16)
    gratingtrialctrl=visstimtranstionts[selectedgrating,:,1,0]
    grattrialarray[0,:,0,0]=visstimtranstionts[selectedgrating,:,0,0]
    
    activity_dict= slice_by_optotrial(stimulated_cells_number,pretime,posttime,fr,grattrialarray,traces_dict,gratingtrialctrl,use_scaled=True)
    plot_rasters_treatment(optocellindex_dict,activity_dict,substracted=False,smoothed=False)

    #%%

# for l in range(2):
#     f,ax=plt.subplots(int(transition_array.shape[1]/2),2)
#     for i,opto  in enumerate(transition_array[0,:,0,0]):
#         row = i // 2  # determine the row index based on the iteration index
#         col = i % 2   # determine the column index based on the iteration index
#         # ax[row, col].plot(analysis.activity_dict['parameters']['trialtimevector'],analysis.scale_signal(smoothedoptotraces[l,l,i,:]),'k')
#         ax[row, col].plot(activity_dict['parameters']['trialtimevector'],activity_dict['opto']['trials']['raw_smoothed'][l,0,i,:]),'k'

#         # ax[row, col].plot(analysis.activity_dict['parameters']['trialtimevector'],smoothedoptotracessubstracetd[l,l,i,:],'b')
#         ax[row, col].plot(  activity_dict['parameters']['trialspeedtimevector'],   activity_dict['speed']['trials']['raw_smoothed'][0,i,:],'r')
#         ax[row, col].axvline(x=0)  
#         # ax[row, col].set_ylim(-3,8)
#         ax[row, col].margins(x=0)
#         for m in range(opto_repetitions):               
#             ax[row, col].add_patch(Rectangle((activity_dict['parameters']['trialtimevector'][activity_dict['parameters']['prestim']+activity_dict['parameters']['interoptoframes']*m], 0.8), 0.01, 0.2,color='r'))

#         ax[row, col].set_xlabel('Time(s)')
#         ax[row, col].set_ylabel('Activity(a.u.)')
        
#     f.suptitle(f'Single Trial Optogenetic Stimulation Pyr Cell{str(l+1)}', fontsize=16)
    
# if gratingcontrol.any():

#     #plot single optotirals individually 
#     for l in range(10):
#         f,ax=plt.subplots(int(gratingcontrol.shape[0]/2),2)
#         for i,opto  in enumerate(gratingcontrol):
#             row = i // 2  # determine the row index based on the iteration index
#             col = i % 2   # determine the column index based on the iteration index
#             # ax[row, col].plot(analysis.activity_dict['parameters']['trialtimevector'],analysis.scale_signal(smoothedoptotraces[l,l,i,:]),'k')
#             ax[row, col].plot(analysis.activity_dict['parameters']['trialtimevector'],activity_dict['control']['trials']['raw_smoothed'][l,0,i,:]),'k'

#             # ax[row, col].plot(analysis.activity_dict['parameters']['trialtimevector'],smoothedoptotracessubstracetd[l,l,i,:],'b')
#             ax[row, col].plot(analysis.  activity_dict['parameters']['trialspeedtimevector'],analysis.   activity_dict['speed']['trials']['raw_smoothed'][0,i,:],'r')
#             ax[row, col].axvline(x=0)  
#             # ax[row, col].set_ylim(-3,8)
#             ax[row, col].margins(x=0)
       
#             ax[row, col].set_xlabel('Time(s)')
#             ax[row, col].set_ylabel('Activity(a.u.)')
            
#         f.suptitle(f'Single Trial Grating Control Pyr Stimulation Cell{str(l+1)}', fontsize=16)
    



# #%% in rafaaanalysis
# dtset=analysis.calcium_datasets[list(analysis.calcium_datasets.keys())[0]]
# dtset.most_updated_caiman.CaimanResults_object.load_all()
# f,ax=plt.subplots(2)
# for i in range(2):
#     trace=dtset.most_updated_caiman.CaimanResults_object.raw[i,:]
#     ax[0].plot(trace)
#     ax[1].plot(analysis.smooth_trace(trace,10))
    
# volt=dtset.associated_aquisiton.voltage_signal_object
# dtset.most_updated_caiman.CaimanResults_object.load_caiman_hdf5_results()
# cnm=dtset.most_updated_caiman.CaimanResults_object.cnm.estimates
# cnm.view_components()
# full_raw=cnm.C+cnm.YrA   



# sig=volt.voltage_signals_dictionary
# sig['PhotoStim']
# sig['PhotoTrig'].plot()
    
# phototriggers=sig['PhotoTrig'].values.flatten()
# photosignals=sig['PhotoStim'].values.flatten()
# timestamps=dtset.associated_aquisiton.metadata_object.timestamps['Plane1']

    
# volt=dtset.associated_aquisiton.voltage_signal_object
# dtset.most_updated_caiman.CaimanResults_object.load_caiman_hdf5_results()
# cnm=dtset.most_updated_caiman.CaimanResults_object.cnm.estimates
# cnm.view_components()
# full_raw=cnm.C+cnm.YrA

# chandeliers=np.array([ 0,])


# chandeliertraces=full_raw[chandeliers,:]


# sig=volt.voltage_signals_dictionary
# sig['PhotoStim']
# sig['PhotoTrig'].plot()
    
# phototriggers=sig['PhotoTrig'].values.flatten()
# photosignals=sig['PhotoStim'].values.flatten()
# timestamps=dtset.associated_aquisiton.metadata_object.timestamps['Plane1']

# stimnumber=5
# frequency=20#hz
# duration=20/1000#ms
# stimperiod=1/frequency #s
# isi=stimperiod-duration

# optotimes=np.arange(0,5*stimperiod,stimperiod)

# plt.rcParams["figure.figsize"] = [16, 5]
# plt.rcParams["figure.autolayout"] = True
# plt.close('all')
# smoothwindows=10
# fr=dtset.metadata.translated_imaging_metadata['FinalFrequency']
# framen=len(dtset.most_updated_caiman.CaimanResults_object.raw[1,:])
# lengthtime=framen/fr
# period=1/fr
# fulltimevector=np.arange(0,lengthtime,period)

# f,ax=plt.subplots(6)    
# for i in range(6):
#     trace=dtset.most_updated_caiman.CaimanResults_object.raw[i,:]
#     if i==3:
#         ax[i].plot(fulltimevector,smooth_trace(trace,10),c='y')
#     else:
#         ax[i].plot(fulltimevector,smooth_trace(trace,10),c='g')

#     for i,a in enumerate(ax):
#         a.margins(x=0)
#         if i<len(ax)-1:
#             a.axis('off')
     
#         elif i==len(ax)-1:
            
#             a.spines['top'].set_visible(False)
#             a.spines['right'].set_visible(False)
#             a.spines['left'].set_visible(False)
#             a.get_yaxis().set_ticks([])
            
# f,ax=plt.subplots(1)
# f.tight_layout()
# for i in range(3,6):
#     trace=dtset.most_updated_caiman.CaimanResults_object.raw[i,:]
#     ax.plot(fulltimevector,smooth_trace(trace,smoothwindows))
    
#     ax.margins(x=0)
#     for j in range(len(estimatedoptotimes)):
#         ax.axvline(x=fulltimevector[estimatedoptotimes[j]],c='r')  
#     ax.set_xlabel('Time(s)')
#     ax.set_ylabel('Activity(a.u.)')
#     f.suptitle('Optogenetic Stimulation of Chandelier Cells', fontsize=16)

   
# smoothedtraces=np.zeros_like(dtset.most_updated_caiman.CaimanResults_object.raw)        
# for i in range(6):
#     smoothedtraces[i,:]=smooth_trace(dtset.most_updated_caiman.CaimanResults_object.raw[i,:],smoothwindows)
      
# optotrialarraysmoothed=np.zeros((3,10,60))
# activity_dict['parameters']['trialtimevector']=np.arange(-period*activity_dict['parameters']['prestim'],period*poststim,period)


# # f,ax=plt.subplots(5,2)
# for i,opto  in enumerate(estimatedoptotimes):
#     for j in range(3,6):
# #         row = i // 2  # determine the row index based on the iteration index
# #         col = i % 2   # determine the column index based on the iteration index
#         optotrialarraysmoothed[j-3,i,:]=smoothedtraces[j,opto-activity_dict['parameters']['prestim']:opto+poststim]
# #         ax[row, col].plot(activity_dict['parameters']['trialtimevector'],optotrialarraysmoothed[j-3,i,:])

# #         ax[row, col].axvline(x=0,c='r')  
# #         ax[row, col].set_ylim(-1,3)
# #         ax[row, col].margins(x=0)
# #         ax[row, col].set_xlabel('Time(s)')
# #         ax[row, col].set_ylabel('Activity(a.u.)')
# #         for m in optotimes:
# #             ax[row, col].add_patch(Rectangle((m, 2.8), 0.01, 0.2))
               
#    # 

            
# #plotting all trials and mean for chandelier
# # fullmean=optotrialarraysmoothed.mean(axis=1)
# # for j in range(3,6):
# #     f,ax=plt.subplots(1)
# #     for i,opto  in enumerate(estimatedoptotimes):
# #         ax.plot(activity_dict['parameters']['trialtimevector'],optotrialarraysmoothed[j-3,i,:],c='k',alpha=0.2)
# #         ax.plot(activity_dict['parameters']['trialtimevector'],fullmean[j-3,:],c='k')
# #         ax.axvline(x=0,c='r')  
# #         ax.set_ylim(-1,3)
# #         ax.margins(x=0)
        
# #     for m in optotimes:
# #         ax.add_patch(Rectangle((m, 2.8), 0.01, 0.2))
        
# meanactivations=optotrialarraysmoothed[:,[0,4,6,9],:].mean(axis=1)
# meanalocomotion=optotrialarraysmoothed[:,7,:]
# meannonactivations=optotrialarraysmoothed[:,[1,2,3,5,8],:].mean(axis=1)

# # f,ax=plt.subplots(3)
# # for i in range(3):
# #     ax[0].plot(activity_dict['parameters']['trialtimevector'],meanactivations[i,:])
# #     ax[1].plot(activity_dict['parameters']['trialtimevector'],meannonactivations[i,:])  
# #     ax[2].plot(activity_dict['parameters']['trialtimevector'],meanalocomotion[i,:])
# #     for j in [0,4,6,9]:
# #         ax[0].plot(activity_dict['parameters']['trialtimevector'],optotrialarraysmoothed[i,j,:],alpha=0.2)
# #     for j in [7]:
# #         ax[2].plot(activity_dict['parameters']['trialtimevector'],optotrialarraysmoothed[i,j,:],alpha=0.2)  
# #     for j in [1,2,3,5,8]:
# #         ax[1].plot(activity_dict['parameters']['trialtimevector'],optotrialarraysmoothed[i,j,:],alpha=0.2)

# #     for n in range(3):
# #         ax[n].margins(x=0)
# #         ax[n].set_ylim(-1,3)
# #         ax[n].axvline(x=0,c='r')  
# #         for m in optotimes:
# #             ax[n].add_patch(Rectangle((m, 2.8), 0.01, 0.2))
# #         ax[n].set_xlabel('Time(s)',fontsize=20)
# #         ax[n].set_ylabel('Activity(a.u.)')
# #     ax[0].set_title('Responsive Trials')
# #     ax[1].set_title('Unresponsive Trials')
# #     ax[2].set_title('Running Trials')

            
# #     f.suptitle('Trial segmented PSTH', fontsize=25)
        

        
#  #baseline substractes       
        
# optotrialarraysmoothedDF=np.zeros_like(optotrialarraysmoothed)
# meanbaselinesmoothed=optotrialarraysmoothed[:,:,0:20].mean(axis=2)
# for i in range(3):
#     for j in range(10):
#         optotrialarraysmoothedDF[i,j,:]=(optotrialarraysmoothed[i,j,:]-meanbaselinesmoothed[i,j])


# f,ax=plt.subplots(5,2)
# for i,opto  in enumerate(estimatedoptotimes):
#     for j in range(3,6):
#         row = i // 2  # determine the row index based on the iteration index
#         col = i % 2   # determine the column index based on the iteration index
#         ax[row, col].plot(activity_dict['parameters']['trialtimevector'],optotrialarraysmoothedDF[j-3,i,:])
#         ax[row, col].axvline(x=0,c='r')  
#         ax[row, col].set_ylim(-1,3)
#         ax[row, col].margins(x=0)
#         for m in optotimes:
#             ax[row, col].add_patch(Rectangle((m, 2.8), 0.01, 0.2,color='r'))
#         ax[row, col].set_xlabel('Time(s)')
#         ax[row, col].set_ylabel('Activity(a.u.)')
        
#     f.suptitle('Single Trial Optogenetic Stimulation', fontsize=16)

               
# #plotting all trials and mean for chandelier
# fullmeandf=optotrialarraysmoothedDF.mean(axis=1)
# cells=['Chand','Pyr1','Pyr2']
# for j in range(3,6):
#     f,ax=plt.subplots(1)
#     for i,opto  in enumerate(estimatedoptotimes):
#         ax.plot(activity_dict['parameters']['trialtimevector'],optotrialarraysmoothedDF[j-3,i,:],c='k',alpha=0.2)
#         ax.plot(activity_dict['parameters']['trialtimevector'],fullmeandf[j-3,:],c='k')
#         ax.axvline(x=0,c='r')  
#         ax.set_ylim(-1,3)
#         ax.margins(x=0)
        
#         ax.set_xlabel('Time(s)')
#         ax.set_ylabel('Activity(a.u.)')
#         for m in optotimes:
#             ax.add_patch(Rectangle((m, 2.8), 0.01, 0.2,color='r'))
        
#     f.suptitle(f'Global Optostimulation PSTH {cells[j-3]}', fontsize=16)

                        
 
# meanactivationsbasesub=optotrialarraysmoothedDF[:,[0,4,6,9],:].mean(axis=1)
# meanalocomotionbasesub=optotrialarraysmoothedDF[:,7,:]
# meannonactivationsbasesub=optotrialarraysmoothedDF[:,[1,2,3,5,8],:].mean(axis=1)       
# for i in range(3):
#     f,ax=plt.subplots(3)

#     ax[0].plot(activity_dict['parameters']['trialtimevector'],meanactivationsbasesub[i,:])
#     ax[1].plot(activity_dict['parameters']['trialtimevector'],meannonactivationsbasesub[i,:])  
#     ax[2].plot(activity_dict['parameters']['trialtimevector'],meanalocomotionbasesub[i,:])
#     for k in [0,4,6,9]:
#         ax[0].plot(activity_dict['parameters']['trialtimevector'],optotrialarraysmoothedDF[i,k,:],c='k',alpha=0.2)
#     for k in [7]:
#         ax[2].plot(activity_dict['parameters']['trialtimevector'],optotrialarraysmoothedDF[i,k,:],c='k',alpha=0.2)  
#     for k in [1,2,3,5,8]:
#         ax[1].plot(activity_dict['parameters']['trialtimevector'],optotrialarraysmoothedDF[i,k,:],c='k',alpha=0.2)
    
#     for j in range(3):
#         ax[j].margins(x=0)
#         ax[j].set_ylim(-1,3)
#         ax[j].axvline(x=0,c='r')  
#         for m in optotimes:
#             ax[j].add_patch(Rectangle((m, 2.8), 0.01, 0.2,color='r'))
        
#         ax[j].set_xlabel('Time(s)')
#         ax[j].set_ylabel('Activity(a.u.)')
#     ax[0].set_title('Responsive Trials')
#     ax[1].set_title('Unresponsive Trials')
#     ax[2].set_title('Running Trials')


        
#     f.suptitle(f'Trial segmented PSTH {cells[i]}', fontsize=16)