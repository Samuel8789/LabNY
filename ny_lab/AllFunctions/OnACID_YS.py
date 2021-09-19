#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete pipeline for online processing using CaImAn Online (OnACID).
The demo demonstates the analysis of a sequence of files using the CaImAn online
algorithm. The steps include i) motion correction, ii) tracking current 
components, iii) detecting new components, iv) updating of spatial footprints.
The script demonstrates how to construct and use the params and online_cnmf
objects required for the analysis, and presents the various parameters that
can be passed as options. A plot of the processing time for the various steps
of the algorithm is also included.
@author: Eftychios Pnevmatikakis @epnev
Special thanks to Andreas Tolias and his lab at Baylor College of Medicine
for sharing the data used in this demo.
"""

import numpy as np
import os
import logging
import matplotlib.pyplot as plt
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

# %%
def run_on_acid(caiman_extraction_object,parameter_dict, dataset_object=False ):
    pass # For compatibility between running under Spyder and the CLI

#%%
    start_t=time.time()
    fnamestemp=caiman_extraction_object.temporarypath
    # if dataset_object:
    #     motion_correct_file_to_save=os.path.join(dataset_object.selected_dataset_mmap_path, os.path.splitext(dataset_object.image_sequence_changed_type_file)[0] )
    # else:
    #     motion_correct_file_to_save=os.path.splitext(caiman_extraction_object.dataset_mouse_path)[0]+'test'
    opts = cnmf.params.CNMFParams(params_dict=parameter_dict)

    # %% fit online
    
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    print('start processing')
    preprocetime=time.time()
    print(preprocetime-start_t)
    cnm.fit_online()
    print('Finsihed processing')
    postprocetime=time.time()
    print(postprocetime-preprocetime)
    # %% plot contours (this may take time)
    logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
    images = cm.load(fnamestemp)
    Cn = images.local_correlations(swap_dim=False, frames_per_chunk=500)
    # cnm.estimates.plot_contours(img=Cn, display_numbers=False)
    
    # %% view components
    # cnm.estimates.view_components(img=Cn)
    
    # %% plot timing performance (if a movie is generated during processing, timing
    # will be severely over-estimated)
    
    # T_motion = 1e3*np.array(cnm.t_motion)
    # T_detect = 1e3*np.array(cnm.t_detect)
    # T_shapes = 1e3*np.array(cnm.t_shapes)
    # T_track = 1e3*np.array(cnm.t_online) - T_motion - T_detect - T_shapes
    # plt.figure()
    # plt.stackplot(np.arange(len(T_motion)), T_motion, T_track, T_detect, T_shapes)
    # plt.legend(labels=['motion', 'tracking', 'detect', 'shapes'], loc=2)
    # plt.title('Processing time allocation')
    # plt.xlabel('Frame #')
    # plt.ylabel('Processing time [ms]')
    #%% RUN IF YOU WANT TO VISUALIZE THE RESULTS (might take time)
    
    # if 'dview' in locals():
    #     cm.stop_server(dview=dview)
    # c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None,
    #                                   single_thread=False)
    
    
    # if opts.online['motion_correct']:
    #     shifts = cnm.estimates.shifts[-cnm.estimates.C.shape[-1]:]
    #     if not opts.motion['pw_rigid']:
    #         memmap_file = cm.motion_correction.apply_shift_online(images, shifts,
    #                                                     save_base_name=(motion_correct_file_to_save + 'MC'))
    #     else:
    #         mc = cm.motion_correction.MotionCorrect(fnames, dview=dview,
    #                                                 **opts.get_group('motion'))
            
    #         mc.x_shifts_els = [[sx[0] for sx in sh] for sh in shifts]
    #         mc.y_shifts_els = [[sx[1] for sx in sh] for sh in shifts]
            
    #         memmap_file = mc.apply_shifts_movie(fnames, rigid_shifts=False,
    #                                             save_memmap=True,
    #                                             save_base_name=(motion_correct_file_to_save+'MC'))
    # else:  # To do: apply non-rigid shifts on the fly
    #     memmap_file = images.save(fnames[:-4] + 'mmap')
        
    # if 'dview' in locals():
    #     cm.stop_server(dview=dview)    
        
    # cnm.mmap_file = memmap_file
    cnm.mmap_file = fnamestemp
    Yr, dims, T = cm.load_memmap(fnamestemp)
    
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
  
  
    cnm.estimates.evaluate_components(images, cnm.params, dview=None)
    cnm.estimates.Cn = Cn
    

    
    cnm.save(caiman_extraction_object.dataset_mouse_path[0:-5] + 'cnmf_results.hdf5')
    # dview.terminate()
    
    duration = time.time() - start_t
    print(duration/60)
    # return cnm
#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
