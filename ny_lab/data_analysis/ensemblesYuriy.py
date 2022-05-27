# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 16:03:07 2021

@author: sp3660
"""

import networkx as nx
import math
import matplotlib.pyplot as plt
import os
import mplcursors
import copy
import numpy as np
import matplotlib as mlp
# from TestPLot import SnappingCursor
import matplotlib as mpl
import scipy.signal as sig
from numpy import exp, abs, angle
from scipy import stats, interpolate
import scipy.io
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b"]) 


class EnsemblesYuriy():
    
    def __init__(self, analysis_object ):
        self.analysis_object=analysis_object
        self.full_data= self.analysis_object.full_data

        self.normalize_activity_matrix(activity_matrix, 'norm_mean_std')

        pass

        

        # np.roll(, random number of places)
        # np.random.permutation
        # from sklearn.decomposition import PCA
        # PCA
        # np.convolve
        # np.linalg.svd
        # from sklearn.decomposition import NMF
        # NMF
        # from sklearn.decomposition import FastICA
        # FastICA(n_components=7,
        # from sklearn.cluster import AgglomerativeClustering

        # AgglomerativeClustering().fit(X)


    def permute_cells(self, activity_matrix):
        
        permuted_cells=np.random.permutation(activity_matrix)
        
        return permuted_cells

        
    def remove_inactive_cells(self, activity_matrix):
        
        inactive_cells_removed=copy.copy(activity_matrix)
        active_cells = np.sum(inactive_cells_removed,2) > 0
        inactive_cells=1-active_cells
        inactive_cells_removed[inactive_cells,:] = []

        
        return inactive_cells_removed
    
    def shuffle_data(self, activity_matrix):

        shuflled=np.apply_along_axis(np.roll, 1, activity_matrix,np.ceil(np.random.uniform()*activity_matrix.shape[0]) )
        
        return shuflled
    
    def normalize_activity_matrix(self, activity_matrix, method):
        from sklearn.metrics import mean_squared_error
        
        if 'norm_mean_std' in method:
            activity_matrix_norm = activity_matrix - np.mean(activity_matrix,2)
            activity_matrix_norm = activity_matrix_norm/np.std(activity_matrix_norm,1) 
            # %firing_rate_cont(isnan(firing_rate_cont)) = 0;
        elif 'norm_mean' in method:
            activity_matrix_norm = activity_matrix - np.mean(activity_matrix,2)
        elif 'norm_std' in method:
            activity_matrix_norm = activity_matrix/np.std(activity_matrix,[],2) 
        elif 'norm_rms' in method:
            activity_matrix_norm = activity_matrix/mean_squared_error(activity_matrix,2) 
        elif 'none' in method:
            activity_matrix_norm = activity_matrix

        activity_matrix_norm[np.isnan(activity_matrix_norm)] = 0;

        
        
        
        
if __name__ == "__main__":
    pass
    # plt.close('all')
    
    
    
    # activity=analysis.activity_matrixes_resampled_bordercuts['dfdtmatrix']
    # firing_rate=activity
    # #%% input parameters for cross validation estimation of smooth window and number of correlated components / ensembles
    # # **params** are best params


    # estimate_params = 1   # do estimation?
    # include_shuff_version = 1
    # est_params={}
    # est_params['ensamble_method'] = 'svd'            # options: svd, nmf, ica                % SVD is most optimal for encoding, NMF rotates components into something that is real and interpretable
    # est_params['normalize'] = 'norm_mean_std' # **'norm_mean_std'**, 'norm_mean' 'none'   % either way, need to normalize the power of signal in each cell, otherwise dimred will pull out individual cells
    # est_params['smooth_SD'] = 0;#50:50:250;       # range of values to estimate across    % larger window will capture 'sequences' of ensembles, if window is smaller than optimal, you will end up splitting those into more components
    # est_params['num_comp'] = range(2,20,4)               # range of values to estimate across    
    # est_params['shuffle_data_chunks'] = 0; # 1 or 0, keeping cell correlations   % if the sequence of trial presentation contains information, you will need to shuffle. Also need to do in chunks because adjacent time bins are slightly correlated
    # est_params['reps'] = 1                   # how many repeats per param 


    # est_params['n_rep'] = rnage(est_params['reps'])
    # est_params_list = f_build_param_list(est_params, {'smooth_SD', 'num_comp', 'n_rep'});
    # if include_shuff_version:
    #     est_params_list_s = est_params_list;

    # # %% input paramseters for ensemble analysis
    # # NMF ensemble detection is best with thresh extraction
    # # for NMF best to use norm_rms(keep values positive), otherwise can also use norm_mean_std
    # ens_params={}
    # ens_params['ensamble_method'] = 'nmf' # options: svd, **nmf**, ica     % here NMF is
    # ens_params['num_comp'] = 10
    # ens_params['smooth_SD'] = 120 # 110 is better?
    # ens_params['normalize'] = 'norm_mean_std' # 'norm_mean_std', 'norm_mean' 'none'
    # ens_params['ensamble_extraction'] = 'thresh' #  **'thresh'(only for nmf)** 'clust'(for all)
    # # --- for thresh detection (only nmf)
    # ens_params['ensamble_extraction_thresh'] = 'signal_z' # 'shuff' 'signal_z' 'signal_clust_thresh'
    # ens_params['signal_z_thresh'] = 2.5
    # ens_params['shuff_thresh_percent'] = 95
    # # --- for clust detection and general sorting 
    # ens_params['hcluster_method'] = 'average' # ward(inner square), **average**, single(shortest)     
    # ens_params['hcluster_distance_metric'] = 'cosine'  # none, euclidean, squaredeuclidean, **cosine**, hammilarity, rbf% for low component number better euclidean, otherwise use cosine
    # ens_params['corr_cell_thresh_percent'] = 95  # to remove cells with no significant correlations
    # # --- other
    # ens_params['plot_stuff'] = 0

    # ens_params['vol_period'] = 1/(frame_rate*1000)

    # # %% remove inactive cells

    # active_cells = active_cells[np.where(np.sum(firing_rate,1) > 0)]
    # select = np.in1d(range(firing_rate.shape[0]), active_cells)

    # firing_rate[~select,:] = []

    # num_cells = firing_rate.shape[0]

    # firing_rate = firing_rate[np.random.permutation(num_cells),:]

    # # % shuffle
    # firing_rate_s = f_shuffle_data(firing_rate);

    # firing_rate_norm = f_normalize(firing_rate, est_params['normalize'])
    # firing_rate_norm_s = f_normalize(firing_rate_s, est_params['normalize'])

    # # %% estimate best smoothing window
    # if estimate_params:
    #     est_params_list = f_ens_estimate_dim_params(firing_rate_norm, est_params_list, ens_params['vol_period'])
    #     # [~, min_ind] = min([est_params_list.test_err]);
    #     print('From provided range, optimal smooth_SD = %d; Number of CV %s num_comp = %d\n', est_params_list(min_ind).smooth_SD, est_params.ensamble_method, est_params_list(min_ind).num_comp);

    #     if include_shuff_version
    #         fprintf('Estimating params shuff n/%d reps: ',numel(est_params_list_s));
    #         %dim_corr = zeros(numel(estimate_smooth_list),1);
    #         for n_par = 1:numel(est_params_list_s)

    #             params1 = est_params_list_s(n_par);
    #             params1.vol_period = ens_params.vol_period;
    #             accuracy = f_ens_estimate_corr_dim_cv(firing_rate_norm_s, params1);

    #             temp_fields = fields(accuracy);
    #             for n_fl = 1:numel(temp_fields)
    #                 est_params_list_s(n_par).(temp_fields{n_fl}) = accuracy.(temp_fields{n_fl});
    #             end
    #             fprintf('--%d',n_par);
    #         end
    #         fprintf('\nDone\n');
    #         [~, min_ind] = min([est_params_list_s.test_err]);
    #         fprintf('From provided range, optimal smooth_SD = %d; Number of CV %s num_comp = %d\n', est_params_list_s(min_ind).smooth_SD, est_params.ensamble_method, est_params_list(min_ind).num_comp);
    #     end
    #     f_plot_cv_error_3D(est_params_list, est_params_list_s, 'smooth_SD', 'num_comp', 'test_err');
    #     ax1 = gca;
    #     ax1.Title.String = sprintf('%s L2 error from raw, (%s)',est_params.ensamble_method, ax1.Title.String);
    # end

    # # %% Smooth data
    # firing_rate_sm = f_smooth_gauss(firing_rate, ens_params.smooth_SD*frame_rate/1000);

    # # %% extract ensambles
    # disp('Extracting ensambles...');
    # ens_out = f_ensemble_analysis_YS_raster(firing_rate_sm, ens_params);

    # # %% plotting ensembles
    # f_plot_raster_mean(firing_rate_sm(ens_out.ord_cell,:), 1);
    # title('raster cell sorted');
    # f_plot_raster_mean(firing_rate_sm(ens_out.ord_cell,:), 1);
    # title('raster shuffeled cell sorted');

    # for n_comp = 1:numel(ens_out.cells.ens_list)
    #     cells1 = ens_out.cells.ens_list{n_comp};
    #     trials1 = ens_out.trials.ens_list{n_comp};
    #     scores1 = ens_out.cells.ens_scores(n_comp,:);

    #     f_plot_ensamble_deets(firing_rate_sm, cells1, trials1, scores1);
    #     title([ens_params.ensamble_method ' ensamble ' num2str(n_comp)]);
    # end
            
    # # %%
    # disp('Done');

    #  fn = [directs '\\test'];  %in this example, we'll save to a temp directory.
    # figures =  findall(0,'Type','figure'); 
    # % Generate figures
    # %[your code]
    # % Resize and output figures
    # figSize = [21, 29];            % [width, height]
    # figUnits = 'Centimeters';
    # for f = 1:numel(figures)
    #       fig = figures(f);
    #       % Resize the figure
    #       set(fig, 'Units', figUnits);
    #       pos = get(fig, 'Position');
    #       pos = [pos(1), pos(4)+figSize(2), pos(3)+figSize(1), pos(4)];
    #       set(fig, 'Position', pos);
    #       % Output the figure
    #       filename = sprintf('Figure%02d.pdf', f);
    #       print( fig, '-dpdf', filename );
    # end 