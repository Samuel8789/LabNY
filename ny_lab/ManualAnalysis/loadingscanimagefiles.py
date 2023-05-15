# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:11:20 2022

@author: sp3660
"""
import os
import sys
import glob
from ScanImageTiffReader import ScanImageTiffReader
import caiman as cm
import numpy as np
import scipy.signal as sg
import pandas as pd
import json
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as spio
sys.path.append(r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\dataManaging\classes\standalone')
sys.path.append(r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\dataManaging\classes')
from voltageSignals import VoltageSignals
from scipy.ndimage import gaussian_filter1d
import time
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.summary_images import local_correlations_movie_offline
from scipy import interpolate
from caiman.source_extraction import cnmf as cnmf
import time
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

def resample(x, factor, kind='linear'):
    n = int(np.floor(x.size / factor))
    f = interpolate.interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, n))     
#%% location managing++
"""
Raw files
    scanimage singel tiff
    prairie ome directory fill
    separated praire channels and planes
mmap files raw
hd5 raw files

mmap averaged files
hd5 averaged files

mmap 


"""
mouseflder=r'F:\Projects\LabNY\Imaging\2022\20220525Hakim\Mice\SPKU'


aquis='FOV_1\Aq_1'
aq_name=os.path.split(glob.glob(os.path.join(mouseflder,aquis)+'\\**')[0])[1]
tosavedir=os.path.join(r'G:\Projects\TempProcessing',aq_name)
ocord='0CoordinateAcquisiton'
surface='FOV_1\SurfaceImage'
plane='FOV_1\OtherAcq'
full=aquis+os.sep+aq_name
facecam='FOV_1\Aq_1\FaceCamera'
visstim='FOV_1\Aq_1\VisStim'




ocrodfile=glob.glob(os.path.join(mouseflder,ocord)+'\**.tif')[0]
surfacefile=glob.glob(os.path.join(mouseflder,surface)+'\**.tif')[0]
planefile=glob.glob(os.path.join(mouseflder,plane)+'\**.tif')[0]
fullmoviefile=glob.glob(os.path.join(mouseflder,full)+'\**.tif')[0]
visstimfile=glob.glob(os.path.join(mouseflder,visstim)+'\**.mat')[0]
voltagefile=glob.glob(os.path.join(mouseflder,full)+'\**.csv')[0]

output=os.path.join(mouseflder, r'ToTrack')

allfiles=[ocrodfile, surfacefile, planefile, fullmoviefile]
filenames=['0Coord.tiff', 'Surface.tiff', 'Plane.tiff', 'Fullmovie.tiff' ]
i=-1


#%% metadtat
meta=ScanImageTiffReader(allfiles[i]).metadata();
# desc=ScanImageTiffReader(allfiles[i]).description(0);

parameter='SI.hDisplay.displayRollingAverageFactor = '
paddedstring=meta[meta.find(parameter)+len(parameter):meta.find(parameter)+len(parameter)+20]
value=paddedstring[:paddedstring.find('\n')]


parameters=['SI.hBeams.powers',
            'SI.hDisplay.displayRollingAverageFactor',
            'SI.hRoiManager.linesPerFrame',
            'SI.hRoiManager.pixelsPerLine',
            'SI.hRoiManager.scanFramePeriod',
            'SI.hRoiManager.scanFrameRate',
            'SI.hRoiManager.linePeriod',
            'SI.hChannels.channelsActive',
            'SI.hFastZ.numVolumes']

meta_extract={}
for parameter in parameters:
    paddedstring=meta[meta.find(parameter +' = ')+len(parameter+' = '):meta.find(parameter+' = ')+len(parameter+' = ')+20]
    if 'powers' in parameter:
        tempstring=paddedstring[:paddedstring.find('\n')]

        meta_extract[parameter]=[float(i) for i in tempstring[1:-1].split(" ")]
    elif 'channelsActive' in parameter:
        tempstring=paddedstring[:paddedstring.find('\n')]
        meta_extract[parameter]=[int(i) for i in tempstring[1:-1].split(";")]

    else:
        meta_extract[parameter]=float(paddedstring[:paddedstring.find('\n')])

#%% movie laoding
fr =meta_extract['SI.hRoiManager.scanFrameRate']
px=int(meta_extract['SI.hRoiManager.linesPerFrame'])
colors=['green', 'red']

with ScanImageTiffReader(allfiles[i]) as vol:
    lmov = vol.__len__()
    for i in range(2):
        im=cm.movie(vol.data(0, lmov ).reshape([int(lmov/2), 2, px, px])  [ :,i, :, : ])
        # print(f'Type: {im.dtype}')
        # print(f'Mean: {im.mean():.2f}')
        # print(f'Minimum: {im.min()}')
        # print(f'Maximum: {im.max()}')
        
        # f,ax=plt.subplots()
        # ax.imshow(red.mean(0))
        corrected=im-im.min()
        del im
        # print(f'Type: {corrected.dtype}')
        # print(f'Mean: {corrected.mean():.2f}')
        # print(f'Minimum: {corrected.min()}')
        # print(f'Maximum: {corrected.max()}')
        
        unsigned=corrected.astype('uint16')
        del corrected
        # print(f'Type: {unsigned.dtype}')
        # print(f'Mean: {unsigned.mean():.2f}')
        # print(f'Minimum: {unsigned.min()}')
        # print(f'Maximum: {unsigned.max()}')
        
        f2,ax2=plt.subplots()
        ax2.hist(unsigned.flatten(), bins=100)
        
         # reducing size either avergaing frames or reducing bit depth update meta if averaging frAEMS
        averagin=4
        averagedmov=np.mean(unsigned.reshape((int(unsigned.shape[0]/averagin), averagin, unsigned.shape[1], unsigned.shape[2])), axis=1)
        
        meta_extract['SI.hRoiManager.scanFrameRate']=meta_extract['SI.hRoiManager.scanFrameRate']/averagin
        
        averagedmov.astype('uint16').save(os.path.join(tosavedir,aq_name)+f'_4mean_negcorrected_{colors[i]}.mmap',to32=False)
        averagedmov.astype('uint16').save(os.path.join(tosavedir,aq_name)+f'_4mean_negcorrected_{colors[i]}.tiff',to32=False, bigtiff=True)
        del averagedmov
#%% reloading mmaps movies
from pprint import pprint
allmmapfiles=glob.glob(os.path.join(tosavedir)+'\**.mmap')
pprint(allmmapfiles)

mov=cm.load(allmmapfiles[0])

mov16=mov.astype('uint16')
mov16.save(os.path.join(tosavedir,aq_name)+'_test16.tiff',to32=False,bigtiff=True)


sigma=3

averagedmov.save(os.path.join(tosavedir,aq_name)+'_4mean.mmap',to32=False)
averagedmov.save(os.path.join(tosavedir,aq_name)+'_4mean.tiff',to32=False,bigtiff=True)

correctedmov=mov-mov.min()



smoothemov=cm.movie(gaussian_filter1d(averagedmov, sigma, axis=0))
smoothemov.play(gain=0.3, fr=300)

movie.save(os.path.join(tosavedir,aq_name)+'.mmap',to32=False)
movie.save(os.path.join(tosavedir,aq_name)+'.tiff',to32=False, imagej=True)
#%% motion correct
mc=cm.motion_correction.MotionCorrect(filepath, pw_rigid = False)
start=time.time()
mc.motion_correct(save_movie=True)
end=time.time()-start
print(end)
rigid_mc_file=max(glob.glob(os.path.join(tosavedir)+'\**'), key=os.path.getctime)
to_remove=rigid_mc_file.find('_d1')
to_keep=rigid_mc_file.find('_rig__d1')
rigid_mc_file=os.rename(rigid_mc_file, rigid_mc_file[:to_remove]+'_caiman_mc_rigid'+rigid_mc_file[to_keep+5:])
rigid_registered_mov=cm.load(glob.glob(os.path.join(tosavedir,aq_name)+'**_mc_rigid_d1**.mmap')[0])
# cm.movie(sp.ndimage.gaussian_filter1d(rigid_registered_mov, sigma, axis=0)).save(os.path.join(tosavedir,aqname)+f'_LED_clipped_bidicorrected_caiman_mc_rigid_smoothed-{sigma}-sigma.tiff')
rigid_registered_mov.save(os.path.join(tosavedir,aq_name)+'_4mean_negcorrected_rigid_mc.tiff',to32=False, bigtiff=True)
#cant open big hdf5 files on imagej
#%% ciman extraction
#%% CNMF
dview.terminate()
fnames=r'G:\Projects\TempProcessing\220525_SPKU_FOV1_AllenB_940_25x_hakim_00001\220525_SPKU_FOV1_AllenB_940_25x_hakim_00001_4mean_negcorrected_caiman_mc_rigid_orderc_d1_256_d2_256_d3_1_order_C_frames_55000_.mmap'
# groups=['data','spatial_params','temporal_params','init_params','preprocess_params','patch_params','online','quality','merging','motion','ring_CNN']
Yr, dims, T = cm.load_memmap(fnames)
images = np.reshape(Yr.T, [T] + list(dims), order='F')# mov.save(movie_to_process+'orderc.mmap',   order='C')
images=np.copy(images)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
fr = 14             # imaging rate in frames per second
decay_time = 0.4    # length of a typical transient in seconds
dxy = (2., 2.)      # spatial resolution in x and y in (um per pixel)
# note the lower than usual spatial resolution here

# motion correction parameters
# maximum allowed rigid shift in pixels
# start a new patch for pw-rigid motion correction every x pixels
# overlap between pathes (size of patch in pixels: strides+overlaps)
# maximum deviation allowed for patch with respect to rigid shifts

mc_dict = {
    'fnames': fnames,
    'fr': fr,
    'decay_time': decay_time,
    'dxy': dxy,
    'border_nan': 'copy'
}

opts = params.CNMFParams(params_dict=mc_dict)
p = 1                    # order of the autoregressive system
gnb = 2                  # number of global background components
merge_thr = 0.85         # merging threshold, max correlation allowed
rf = 15
# half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 6          # amount of overlap between the patches in pixels
K = 4                    # number of components per patch
gSig = [3, 3]            # expected half size of neurons in pixels
# initialization method (if analyzing dendritic data using 'sparse_nmf')
method_init = 'greedy_roi'
ssub = 2                     # spatial subsampling during initialization
tsub = 2                     # temporal subsampling during intialization

# parameters for component evaluation
opts_dict = {'fnames': fnames,
             'p': p,
             'fr': fr,
             'nb': gnb,
             'rf': rf,
             'K': K,
             'gSig': gSig,
             'stride': stride_cnmf,
             'method_init': method_init,
             'rolling_sum': True,
             'merge_thr': merge_thr,
             'n_processes': n_processes,
             'only_init': True,
             'ssub': ssub,
             'tsub': tsub}

opts.change_params(params_dict=opts_dict);
# First extract spatial and temporal components on patches and combine them
# for this step deconvolution is turned off (p=0). If you want to have
# deconvolution within each patch change params.patch['p_patch'] to a
# nonzero value

#opts.change_params({'p': 0})
cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
cnm = cnm.fit(images)

#   you can also perform the motion correction plus cnmf fitting steps
#   simultaneously after defining your parameters object using
#  cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
#  cnm1.fit_file(motion_correct=True)

Cns = local_correlations_movie_offline(fnames,
                                       remove_baseline=True, window=1000, stride=1000,
                                       winSize_baseline=100, quantil_min_baseline=10,
                                       dview=dview)
Cn = Cns.max(axis=0)
Cn[np.isnan(Cn)] = 0
cnm.estimates.plot_contours(img=Cn)
plt.title('Contour plots of found components')
cnm.estimates.Cn = Cn
cnm.save(fnames[:-5]+'_init.hdf5')

cnm2 = cnm.refit(images, dview=dview)
cnm3 = cnm2.refit(images, dview=dview)
cnm3.estimates.view_components()
cnm3.estimates.plot_contours(img=Cn)
plt.title('Contour plots of found components')

min_SNR = 2  # signal to noise ratio for accepting a component
rval_thr = 0.85  # space correlation threshold for accepting a component
cnn_thr = 0.99  # threshold for CNN based classifier
cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

cnm3.params.set('quality', {'decay_time': decay_time,
                           'min_SNR': min_SNR,
                           'rval_thr': rval_thr,
                           'use_cnn': True,
                           'min_cnn_thr': cnn_thr,
                           'cnn_lowest': cnn_lowest})
cnm3.estimates.evaluate_components(images, cnm3.params, dview=dview)
cnm3.estimates.plot_contours(img=Cn, idx=cnm3.estimates.idx_components)
cnm3.estimates.Cn = Cn
cnm3.save(fnames[:-5]+'_init.hdf5')

#%% ONACID

fnames=r'G:\Projects\TempProcessing\220525_SPKU_FOV1_AllenB_940_25x_hakim_00001\220525_SPKU_FOV1_AllenB_940_25x_hakim_00001_4mean_negcorrected_caiman_mc_rigid_orderc_d1_256_d2_256_d3_1_order_C_frames_55000_.mmap'
loaded_movie=cm.load(fnames)

cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

fr = 14             # imaging rate in frames per second
decay_time = 0.06# 2 for s 0.5 for f # approximate length of transient event in seconds
gSig = (3,3)  # expected half size of neurons
p = 2  # order of AR indicator dynamics
min_SNR = 1.5   # minimum SNR for accepting new components
ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
gnb = 1  # number of background components
gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int')) # recompute gSig if downsampling is involved
mot_corr = False  # flag for online motion correction
pw_rigid = False  # flag for pw-rigid motion correction (slower but potentially more accurate)
max_shifts_online = 10  # maximum allowed shift during motion correction
sniper_mode = True  # use a CNN to detect new neurons (o/w space correlation)
rval_thr = 0.8  # soace correlation threshold for candidate components
# set up some additional supporting parameters needed for the algorithm
# (these are default values but can change depending on dataset properties)
init_batch = 100 # number of frames for initialization (presumably from the first file)
K = 2  # initial number of components
epochs = 1 # number of passes over the data
show_movie = False # show the movie as the data gets processed
merge_thr = 0.8
use_cnn = True  # use the CNN classifier
min_cnn_thr = 0.90  # if cnn classifier predicts below this value, reject
cnn_lowest = 0.3  # neurons with cnn probability lowe
fudge_factor=0.99 #defqault is 0.96


dataset_caiman_parameters = {'fnames': fnames,
                                   'fr': fr,
                                   'decay_time': decay_time,
                                   'gSig': gSig,
                                   'p': p,
                                   'min_SNR': min_SNR,
                                   'rval_thr': rval_thr,
                                   'merge_thr': merge_thr,
                                   'ds_factor': ds_factor,
                                   'nb': gnb,
                                   'motion_correct': mot_corr,
                                   'init_batch': init_batch,
                                   'init_method': 'bare',
                                   'normalize': True,
                                   'sniper_mode': sniper_mode,
                                   'K': K,
                                   'max_shifts_online': max_shifts_online,
                                   'pw_rigid': pw_rigid,
                                   'dist_shape_update': True,
                                   'min_num_trial': 10,
                                   'show_movie': show_movie,
                                   'epochs':epochs,
                                   'use_cnn': use_cnn,
                                   'min_cnn_thr': min_cnn_thr,
                                   'cnn_lowest': cnn_lowest,
                                   'fudge_factor':fudge_factor
                                    }

start_t=time.time()
fnamestemp=dataset_caiman_parameters['fnames']
opts = cnmf.params.CNMFParams(params_dict=dataset_caiman_parameters)
opts.set('temporal', {'fudge_factor':0.99})

cnm = cnmf.online_cnmf.OnACID(params=opts)
preprocetime=time.time()
cnm.fit_online()
postprocetime=time.time()

images = cm.load(fnamestemp)
MC_movie = images  
Cn = MC_movie.local_correlations(swap_dim=False, frames_per_chunk=500)
cnm.estimates.Cn = Cn


# Cn = MC_movie.local_correlations(swap_dim=False, frames_per_chunk=frames_per_chunk)

cnm.mmap_file = fnamestemp
Yr, dims, T = cm.load_memmap(fnamestemp)

images = np.reshape(Yr.T, [T] + list(dims), order='F')
cnm.estimates.evaluate_components(images, cnm.params, dview=None)
timestr = time.strftime("%Y%m%d-%H%M%S")
caiman_results_path=fnames[:-5]+'_init.hdf5'  


cnm.save(caiman_results_path)

#%% voltage loading



signals=VoltageSignals(scanimage_voltage_csv_path=voltagefile)
signals.signal_extraction_object()
signalextractions=signals.extraction_object
signalextractions.process_all_signals(vis_stim_protocol='AllenB')

signalextractions.plot_all_basics()
signalextractions.plotting_paradigm_transitions()


signalextractions.static_grat_even_index_full_recording
signalextractions.static_grat_odd_index_full_recording
all_static_grating_onsets=np.sort(np.append(signalextractions.static_grat_even_index_full_recording,signalextractions.static_grat_odd_index_full_recording))
all_natural_images_onsets=np.sort(np.append(signalextractions.natural_image_even_index_full_recording,signalextractions.natural_image_odd_index_full_recording))



#%% visstim info



mat = loadmat(visstimfile)
outarray=mat['full_info']
durationseconds=np.array([outarray[1:,2][i][0] for i in range(outarray[1:,2].shape[0])])-outarray[1:,1]
durationminutes=durationseconds/60
outarray[1:,3]=durationseconds
stimparadigms=outarray[:,0].tolist()
staticparadigms=np.where(['Static' in paradigm for paradigm in stimparadigms])[0]
imagesparadigms=np.where(['Images' in paradigm for paradigm in stimparadigms])[0]
movieparadigms=np.where(['Movie' in paradigm for paradigm in stimparadigms])[0]

statictrialinfo=outarray[staticparadigms,:]
imagestrialinfo=outarray[imagesparadigms,:]
movietrialinfo=outarray[movieparadigms,:]

firststaticgratings=statictrialinfo[0][4][1:-1,4].astype(int)
secondstaticgratings=statictrialinfo[1][4][1:-1,4].astype(int)
thirdstaticgratings=statictrialinfo[2][4][1:-1,4].astype(int)

allstaticindexes=np.hstack([firststaticgratings,secondstaticgratings,thirdstaticgratings])

static_gratings_trial_structure=pd.DataFrame({'onsets':all_static_grating_onsets,
                                              'grating_id':allstaticindexes[:all_static_grating_onsets.shape[0]]})


firstnaturalimages=statictrialinfo[0][4][1:-1,4].astype(int)
secondnaturalimages=statictrialinfo[1][4][1:-1,4].astype(int)
thirdnaturalimages=statictrialinfo[2][4][1:-1,4].astype(int)

allnatiuralindexes=np.hstack([firstnaturalimages,secondnaturalimages,thirdnaturalimages])

natural_images_trial_structure=pd.DataFrame({'onsets':all_natural_images_onsets,
                                              'grating_id':allnatiuralindexes[:all_natural_images_onsets.shape[0]]})




#%% assignig frame onsets to static grating stimulus
full_data={'imaging_data':{'Frame_rate':58.2/4,
                                'Interplane_period':''
                                },
                'voltage_traces':{},
                'visstim_info':{}}
milisecond_period=1000/(58.2/4)
voltagerate=1000
full_data['voltage_traces']['Speed']=resample(signalextractions.rectified_speed_array['Prairire']['Locomotion'][:], factor=milisecond_period, kind='linear').squeeze()
full_data['voltage_traces']['Acceleration']=resampled_acceleration_matrix=resample(signalextractions.rectified_acceleration_array['Prairire']['Locomotion'][:], factor=milisecond_period, kind='linear').squeeze() 
if signalextractions.rounded_vis_stim:
    full_data['voltage_traces']['VisStim']=resample(signalextractions.rounded_vis_stim['Prairire']['VisStim'][:], factor=milisecond_period, kind='linear').squeeze()
full_data['voltage_traces']['Photodiode']=''
full_data['voltage_traces']['Start_End']=''
full_data['voltage_traces']['LED']=''
full_data['voltage_traces']['Optopockels']=''
full_data['voltage_traces']['OptoTrigger']=''

if signalextractions.vis_stim_protocol and signalextractions.transitions_dictionary:
    full_data['visstim_info']['Paradigm_Indexes']={key:(np.abs(full_data['imaging_data']['Plane1']['Timestamps'][0] - index/1000)).argmin() for key, index in signalextractions.transitions_dictionary.items()}
    full_data['visstim_info']['Movie1']={'Indexes':'',
                                              'Binary_Maytrix':''
                                                }
    full_data['visstim_info']['Spontaneous']={}
    full_data['visstim_info']['Spontaneous']['stimulus_table']= pd.DataFrame( ([full_data['visstim_info']['Paradigm_Indexes']['spont_first'],full_data['visstim_info']['Paradigm_Indexes']['spont_last']] ,), columns =['start', 'end'])

    if signalextractions.vis_stim_protocol =='AllenB':    
        
        full_data['visstim_info']['Static_Gratings']={'Indexes':{'Grating_onsets':''},'Binary_Maytrix':'','Ref_matrix':'' }
        full_data['visstim_info']['Natural_Images']={'Indexes':'',
                                                  'Binary_Maytrix':'',
                                                  'Ref_matrix':''
                                                    }
        
        
        
        
# {'Indexes':{'Drift_on':np.vstack([[(np.abs( self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.tuning_stim_on_index_full_recording]),
#                                                                 'Drift_off':np.vstack([[(np.abs(self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.tuning_stim_off_index_full_recording]),
#                                                                 'Blank_sweep_on':np.vstack([[(np.abs( self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.blank_sweep_on_index_full_recording]),
#                                                                 'Blank_sweep_off':np.vstack([[(np.abs( self.full_data['imaging_data']['Plane1']['Timestamps'][0] - rep/voltagerate)).argmin()   for rep in ori] for ori in self.signals_object.blank_sweep_off_index_full_recording])
#                                                                 },
#                                                      # 'Binary_Maytrix_downsampled':np.vstack([self.resample(self.signals_object.full_stimuli_binary_matrix[srtim], 
#                                                      #                                                       factor=self.milisecond_period, kind='linear').squeeze() for srtim in range (self.signals_object.full_stimuli_binary_matrix.shape[0])]),
#                                                      # 'Binary_Maytrix_recreated':'',
#                                                      'Resampled_sliced_speed':self.resample(np.concatenate((self.signals_object.first_drifting_set_speed, 
#                                                                                                             self.signals_object.second_drifting_set_speed, 
#                                                                                                             self.signals_object.third_drifting_set_speed)), factor=self.milisecond_period, kind='linear').squeeze(),
#                                                      'Resampled_sliced_visstim':self.resample(np.concatenate((self.signals_object.first_drifting_set, 
#                                                                                                               self.signals_object.second_drifting_set,
#                                                                                                               self.signals_object.third_drifting_set)), factor=self.milisecond_period, kind='linear').squeeze()
                                                     
#                                                      }






#%% facecam



#%%


    
# frame=fr
# sigma=100#ms

# def gaussian_smooth_kernel_convolution(signal, fr, sigma):
#     dt = 1000/fr
#     sigma_frames = sigma/dt
#     # make kernel
#     kernel_half_size = int(np.ceil(np.sqrt(-np.log(0.05)*2*sigma_frames**2)))
#     gaus_win =list(range( -kernel_half_size,kernel_half_size+1))
#     gaus_kernel = [np.exp(-(i**2)/(2*sigma_frames**2)) for i in gaus_win]
#     gaus_kernel = gaus_kernel/sum(gaus_kernel)
#     conv_trace = sg.convolve2d(np.expand_dims(signal,1), np.expand_dims(gaus_kernel,1), mode='same')
#     return conv_trace.flatten()


# for j, mov in enumerate(movs):
#     smoothed=np.zeros_like(mov)
#     for x in np.arange(mov.shape[1]):
#         for y in np.arange(mov.shape[2]):
#             smoothed[:,x,y]=gaussian_smooth_kernel_convolution(mov[:,x,y],frame,sigma)
#     smoothed_motion_corrected=cm.movie(smoothed)
#     proje=smoothed_motion_corrected.mean(axis=0)
#     proje.save(os.path.join(output,'Ch_{}_'.format(j+1)+ filenames[i]))