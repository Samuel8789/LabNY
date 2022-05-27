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



# %%
def run_on_acid(caiman_extraction_object, parameter_dict, dataset_object=False, mot_corretc=False, save_mot_correct=False, initial_shallow=False):
    module_logger.info('running ' +__name__)

    pass # For compatibility between running under Spyder and the CLI
    filename_append='cnmf_results.hdf5'
    if initial_shallow:
        filename_append='cnmf_results_1run_initial.hdf5'
    
    
    if save_mot_correct:
        mot_corretc=True
#%%
    start_t=time.time()
    fnamestemp=parameter_dict['fnames']
    opts = cnmf.params.CNMFParams(params_dict=parameter_dict)

    # %% fit online
    #stadard cnmf
        # images = cm.load(fnamestemp)
        # cnm = cnmf.CNMF(params_dict= opts)
        # cnm.fit_file(include_eval=True)

    
    
    try:
        cnm = cnmf.online_cnmf.OnACID(params=opts)
        module_logger.info('start processing')
        preprocetime=time.time()
        module_logger.info(preprocetime-start_t)
        cnm.fit_online()
        module_logger.info('Finsihed processing')
        postprocetime=time.time()
        module_logger.info(postprocetime-preprocetime)
    except:
            module_logger.exception('cnmf crashed')

    # %% plot contours (this may take time)
    # module_logger.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))  
    images = cm.load(fnamestemp)
    # module_logger.info('calculating dff')
    # cnm.estimates.detrend_df_f()

    mmap_directory, caiman_filename=os.path.split(fnamestemp)
    good_filename=caiman_filename[:caiman_filename.find('_d1_')]   
    if initial_shallow:
        MC_onacid_file_path='_'.join([os.path.join(mmap_directory, good_filename),'MC_OnACID'])
    else:
        MC_onacid_file_path='_'.join([os.path.join(mmap_directory, good_filename)])

    if mot_corretc:
        shifts = cnm.estimates.shifts[-cnm.estimates.C.shape[-1]:]   
        if save_mot_correct:
            module_logger.info('calculating motion corrected movie')
            module_logger.info('saving motion corrected movie')
            MC_movie_to_save= cm.motion_correction.apply_shift_online(images, shifts,
                                                            save_base_name=MC_onacid_file_path)
            MC_onacid_shifts_file_path='.'.join( ['_'.join([os.path.join(mmap_directory, good_filename),'MC_OnACID_shifts']),'pkl'])
            # cnm.estimates.total_template_rig
            with open(MC_onacid_shifts_file_path, "wb") as fp:   #Pickling
                pickle.dump(shifts, fp)
        else:
            module_logger.info('calculating motion corrected movie')

            MC_movie = cm.movie(cm.motion_correction.apply_shift_online(images, shifts)) 
            
       
    else:    
        MC_movie = images  
        Cn = MC_movie.local_correlations(swap_dim=False, frames_per_chunk=500)
        cnm.estimates.Cn = Cn


    # Cn = MC_movie.local_correlations(swap_dim=False, frames_per_chunk=frames_per_chunk)

    cnm.mmap_file = fnamestemp
    Yr, dims, T = cm.load_memmap(fnamestemp)

    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    cnm.estimates.evaluate_components(images, cnm.params, dview=None)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    caiman_results_path='_'.join([MC_onacid_file_path, timestr,filename_append])  

    #%%
    try:
        cnm.save(caiman_results_path)
    except:
         module_logger.exception('caiman not propery saved '+ fnamestemp)   
    #%%
    duration = time.time() - start_t
    module_logger.info(duration/60)
    return cnm
#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
