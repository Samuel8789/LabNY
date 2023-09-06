#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete pipeline for online processing using CaImAn Online (OnACID).
The demo demonstates the analysis of a sequence of files using the CaImAn online
algorithm. The steps include i) motion correction, ii) tracking current 
components, iii) detecting new components, iv) updating of spatial footmodule_logger.infos.
The script demonstrates how to construct and use the params and online_cnmf
objects required for the analysis, and presents the various parameters that
can be passed as options. A plot of the processing time for the various steps
of the algorithm is also included.
@author: Eftychios Pnevmatikakis @epnev
Special thanks to Andreas Tolias and his lab at Baylor College of Medicine
for sharing the data used in this demo.
"""
import logging 
module_logger = logging.getLogger(__name__)
from datetime import datetime
import numpy as np
import os
try:
    if __IPYTHON__:
        # this is used for debugging purposes only.
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
import time
import pickle
import glob


# %%
def galois_caiman(initial_shallow=False):
    pass # For compatibility between running under Spyder and the CLI

    module_logger.info('running ' +__name__)
    
    
    dir_path=os.path.dirname(os.path.realpath(__file__))
    parameter_dict_path=glob.glob(os.path.join(dir_path,'parameter_dict.pkl'))[0]

    with open( parameter_dict_path, 'rb') as file:
        parameter_dict=  pickle.load(file)

    filename_append='cnmf_results.hdf5'
    if initial_shallow:
        filename_append='cnmf_results_1run_initial.hdf5'
    
    
#%%
    if 'dview' in locals():
        cm.stop_server(dview=dview)
        del dview
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)    
    
#%%

    start_t=time.time()
    fnamestemp=parameter_dict['fnames']
    pipeline=parameter_dict['pipeline']
    refit=parameter_dict['refit']
    parameter_dict.pop('pipeline')
    parameter_dict.pop('refit')

    
    opts = cnmf.params.CNMFParams(params_dict=parameter_dict)
    opts.set('temporal', {'fudge_factor':0.99})

    
    
    Yr, dims, T = cm.load_memmap(fnamestemp)
    MC_movie = np.reshape(Yr.T, [T] + list(dims), order='C') 
    
    
   
    if pipeline=='cnmf':
        
        filename_append='deep_cnmf_results.hdf5'
        
        cnm = cnmf.CNMF(n_processes, params= opts, dview=dview)
        
        module_logger.info('start processing')
        preprocetime=time.time()
        cnm.fit(MC_movie)
        postprocetime=time.time()
        module_logger.info(postprocetime-preprocetime)
        
        if refit==True:
            cnm2 = cnm.refit(MC_movie, dview=dview)

    elif pipeline=='onacid':
        
        filename_append='deep_onacid_cnmf_results.hdf5'

        try:
            cnm = cnmf.online_cnmf.OnACID(params=opts,dview=dview)
            module_logger.info('start processing')
            preprocetime=time.time()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            module_logger.info("Current Time ="+ current_time)
            module_logger.info(preprocetime-start_t)
            cnm.fit_online()
            module_logger.info('Finsihed processing')
            postprocetime=time.time()
            module_logger.info(postprocetime-preprocetime)
        except:
                module_logger.exception('cnmf crashed')

    images = cm.load(fnamestemp)

    mmap_directory, caiman_filename=os.path.split(fnamestemp)
    good_filename=caiman_filename[:caiman_filename.find('_d1_')]   
    if initial_shallow:
        MC_onacid_file_path='_'.join([os.path.join(mmap_directory, good_filename),'MC_OnACID'])
    else:
        MC_onacid_file_path='_'.join([os.path.join(mmap_directory, good_filename)])



    MC_movie = images  
    Cn = MC_movie.local_correlations(swap_dim=False, frames_per_chunk=500)
    cnm.estimates.Cn = Cn

    # cnm.estimates.evaluate_components(MC_movie, cnm.params, dview=dview)  
    # cnm.estimates.detrend_df_f(quantileMin=8, frames_window=250)
    # cnm.estimates.select_components(use_object=True)
    cnm.estimates.Cn = Cn



    cnm.mmap_file = fnamestemp
    Yr, dims, T = cm.load_memmap(fnamestemp)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    cnm.estimates.evaluate_components(images, cnm.params, dview=None)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    caiman_results_path='_'.join([MC_onacid_file_path, timestr,filename_append])  
      
 

    try:
        cnm.save(caiman_results_path)
    except:
         module_logger.exception('caiman not propery saved '+ fnamestemp)   
    #%%
    duration = time.time() - start_t
    module_logger.info(duration/60)
    module_logger.info('caiman finished')
    
    
    #%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    # for log_file in log_files:
    #     os.remove(log_file)
#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    galois_caiman()