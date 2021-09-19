# # -*- coding: utf-8 -*-
# """
# Created on Wed Apr 21 12:31:11 2021

# @author: sp3660
# """

#  opts = cnmf.params.CNMFParams(params_dict=params_dict)
 
 
 

#     cnm = cnmf.online_cnmf.OnACID(params=opts)
#     cnm.fit_online()
#     cnm.estimates.plot_contours(img=Cn, display_numbers=False)
#     cnm.estimates.view_components(img=Cn)
#     cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
#     cnm.estimates.Cn = Cn
#     cnm.save(os.path.splitext(fnames[0])[0]+'_results.hdf5')

#     cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
#     cnm = cnm.fit_file()
#     cnm.estimates.plot_contours(img=Cn)
#     cnm2 = cnm.refit(images, dview=dview)
#     cnm2.estimates.play_movie(images, magnification=4)
    
    
#     Yr, dims, T = cm.load_memmap(cnm.mmap_file)

    
#     images = cm.load(fnames)
#     Cn = images.local_correlations(swap_dim=False, frames_per_chunk=500)
#     Cns = local_correlations_movie_offline(fnames[0],
#                                            remove_baseline=True,
#                                            swap_dim=False, window=1000, stride=1000,
#                                            winSize_baseline=100, quantil_min_baseline=10,
#                                            dview=dview)
#     Cn = Cns.max(axis=0)
#     cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False)
#     inspect_correlation_pnr(cn_filter, pnr)
