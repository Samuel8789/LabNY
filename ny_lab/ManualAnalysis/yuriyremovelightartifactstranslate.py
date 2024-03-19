#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:45:10 2024

@author: sp3660
"""


from IPython import get_ipython
import logging
import matplotlib.pyplot as plt
import numpy as np

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
import time 

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.INFO)

try:
    if __IPYTHON__:
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass

import bokeh.plotting as bpl
from pathlib import Path

datasetpath=Path(os.path.join(os.path.expanduser('~'),r'Desktop/CaimanTemp/'))
fname=datasetpath / '230813_SPRM_FOV1_1z_ShortDrift_LED_opto_25x_920_51020_63075_without-000-000_plane1_Shifted_Movie_MC_OnACID_d1_256_d2_256_d3_1_order_F_frames_48841.mmap'


Y =cm.load(fname)

# %%
[T, d1, d2] = Y.shape
Y2d = np.reshape(Y,[T,d1*d2])

Ymeanim = Y2d.mean(axis=0)
Ymeantrace = Y2d.mean(axis=1)
Y2dn = Y2d - Ymeanim;

prc_thresh1 = np.percentile(Ymeanim, 95)
Ymask = Ymeanim<prc_thresh1
mask_trace =Y2dn[:,Ymask].mean(axis=1)

idx_art  = np.argmax(mask_trace)

# %% remove components 1 by 1
num_split = 10;
update_mask = 1;
thresh_clip = 20;

Y2d_cut = Y2d[(idx_art-100):(idx_art+100),:]
T2= Y2d_cut.shape[0]

Y2d_in = Y2d_cut

for n_iter = 1:10
    # % normalize
    Y2d_inn = Y2d_in - np.mean(Y2d_in,0)
    
    # % create initialize comp
    if n_iter == 0 or update_mask
        mask_trace_cut = np.mean(Y2d_inn[:,Ymask],axis=1)
        mask_trace_cut[mask_trace_cut<0] = 0
        mask_trace_cut = mask_trace_cut/norm(mask_trace_cut);
    end
    
    % compute spacial mask
    mask_spac_temp = Y2d_inn*mask_trace_cut';
    
    low_thresh = prctile(mask_spac_temp, thresh_clip);
    high_thresh = prctile(mask_spac_temp, 100 - thresh_clip);
    mask_spac_temp(mask_spac_temp<low_thresh) = low_thresh;
    mask_spac_temp(mask_spac_temp>high_thresh) = high_thresh;
    
    %[f,xi] = ksdensity(mask_spac_temp, linspace(min(mask_spac_temp(:)), max(mask_spac_temp(:)),1000), 'bandwidth', 10);
    %figure; plot(xi, f)
    
    % compute bkg;
    Ybkg_temp = mask_spac_temp*mask_trace_cut;
    
    Y2d_in = Y2d_in - Ybkg_temp;
    Y2d_in(Y2d_in<0) = 0;
    
    figure; 
    subplot(3,1,1);
    plot(mask_trace_cut); title('temporal comp')
    subplot(3,1,2:3); 
    imagesc(reshape(mask_spac_temp, d1, d2)); title('spatial comp')
    sgtitle(sprintf('inter %d', n_iter))
    if_plot_prc_split_mean(Y2d_inn, Ymeanim, num_split);
    sgtitle(sprintf('inter %d, normalized', n_iter))
end

if_plot_prc_split_mean(Y2d_in, Ymeanim, num_split);
sgtitle('after artif removal')


Ybkg = Y2d_cut - Y2d_in;

f_save_mov_YS(uint16(reshape(Y2d_cut, d1, d2, T2)), 'test_Y_cut.h5');
f_save_mov_YS(uint16(reshape(Y2d_in, d1, d2, T2)), 'test_Y_cut_dn.h5');
f_save_mov_YS(uint16(reshape(Ybkg - min(Ybkg(:)), d1, d2, T2)), 'test_Y_cut_res.h5');

%%

Y2d_out = Y2d;
Y2d_out(:,(idx_art-100):(idx_art+100)) = Y2d_out(:,(idx_art-100):(idx_art+100)) - Ybkg;
Y2d_out = uint16(round(Y2d_out));

%%
[~, fname2, ext] = fileparts(fname);

%%
f_save_mov_YS(reshape(Y2d_out, [d1, d2, T]), [data_dir, fname2, '_denoised', ext], '/mov')

%%
% 
% 
% % create initialize comp
% mask_trace_cut = mean(Y2d_cutn(Ymask,:));
% mask_trace_cutn = mask_trace_cut;
% mask_trace_cutn(mask_trace_cut<0) = 0;
% mask_trace_cutn = mask_trace_cutn/norm(mask_trace_cutn);
% 
% 
% 
% if_plot_prc_split_mean(Y2d_cutn, Ymeanim, num_split);
% 
% 
% % run nmf
% num_comp = 1;
% opt = statset('MaxIter',1000,'Display','final', 'TolFun', 1e-20, 'TolX', 1e-20);
% [W, H] = nnmf(Y2d_cutn, num_comp, 'H0', mask_trace_cutn, 'Replicates',5,...
%                    'Options',opt,...
%                    'Algorithm','mult');
% 
% %W1 = Y2d_cutn*H';
% Ybkg = W*H;
% 
% Y2d_cutdn = Y2d_cut - Ybkg;
% 
% Y2d_cutdn2 = Y2d_cutdn;
% Y2d_cutdn2(Y2d_cutdn<0) = 0;
% 
% 
% figure; subplot(4,1,1);
% plot(mean(Y2d_cutn,1)); title('artifact in normalized cut vid')
% subplot(4,1,2); 
% plot(H'); title('nmf comp of afrtifact')
% subplot(4,1,3:4);  imagesc(reshape(W(:,1), d1, d2)); title('nmf comp of afrtifact')
% 
% f_save_mov_YS(uint16(reshape(Y2d_cut, d1, d2, T2)), 'test_Y_cut.h5');
% f_save_mov_YS(uint16(reshape(Y2d_cutdn, d1, d2, T2)), 'test_Y_cut_dn.h5');
% 
% 
% if_plot_prc_split_mean(Y2d_cutdn2, Ymeanim, num_split);
% 
% 
% Y2d_cutn = Y2d_cutdn2 - mean(Y2d_cutdn2,2);
% 
% % create initialize comp
% mask_trace_cut = mean(Y2d_cutn(Ymask,:));
% mask_trace_cutn = mask_trace_cut;
% mask_trace_cutn(mask_trace_cut<0) = 0;
% mask_trace_cutn = mask_trace_cutn/norm(mask_trace_cutn);
% 
% num_split = 10;
% 
% if_plot_prc_split_mean(Y2d_cutn, Ymeanim, num_split);
% 
% 
% prc_thresh2 = prctile(Ymeanim, 5);
% Ymask2 = Ymeanim<prc_thresh2;
% Ymask_trace2 = mean(Y2d_cutn(Ymask2,:));
% Ymask_trace2(Ymask_trace2 < 0) = 0;
% Ymask_trace2 = Ymask_trace2/norm(Ymask_trace2);
% figure; plot(Ymask_trace2)
% 
% H = Ymask_trace2;
% W1 = Y2d_cutn*H';
% 
% % run nmf
% num_comp = 1;
% opt = statset('MaxIter',1000,'Display','final', 'TolFun', 1e-10, 'TolX', 1e-10);
% [W, H] = nnmf(Y2d_cutn, num_comp, 'H0', Ymask_trace2, 'Replicates',5,...
%                    'Options',opt,...
%                    'Algorithm','mult');
%            
% %W1 = Y2d_cutn*H';
% Ybkg = W1*H;
% 
% Y2d_cutdn3 = Y2d_cutdn2 - Ybkg;
% 
% figure; subplot(4,1,1);
% plot(mean(Y2d_cutn,1)); title('artifact in normalized cut vid')
% subplot(4,1,2); 
% plot(H'); title('nmf comp of afrtifact')
% subplot(4,1,3:4);  imagesc(reshape(W1(:,1), d1, d2)); title('nmf comp of afrtifact')
% 
% if_plot_prc_split_mean(Y2d_cutdn3, mean(Y2d_cutdn2,2), num_split);
% 
% 
% Y2d_cutdn4 = Y2d_cutdn3;
% Y2d_cutdn4(Y2d_cutdn3<0) = 0;
% 
% if_plot_prc_split_mean(Y2d_cutdn4, mean(Y2d_cutdn2,2), num_split);
% 
% figure; plot(std(Y2d_cutdn4))
% 
% 
% 
% f_save_mov_YS(uint16(reshape(Y2d_cut, d1, d2, T2)), 'test_Y_cut.h5');
% f_save_mov_YS(uint16(reshape(Y2d_in, d1, d2, T2)), 'test_Y_cut_dn4.h5');
% 
% 
% Y_cutdn2 = reshape(Y2d_cutdn2, d1, d2, T2);
% figure; imagesc(mean(Y_cutdn2,3))
% 
% figure; plot(squeeze(mean(mean(Y_cutdn2(230:end,:,:),1),2)))
% 
% 
% figure; plot(squeeze(Y_cutdn2(250,50:55,:))')

%%
function if_plot_prc_split_mean(data_2d, Ymeanim, num_split)

prc_range = linspace(0, 100, num_split);
prc_thresh = prctile(Ymeanim, prc_range);

figure;
for n_prc = 2:num_split
    subplot(num_split-1,1,n_prc-1)
    Ymask = and(Ymeanim>prc_thresh(n_prc-1),Ymeanim<prc_thresh(n_prc));
    mask_trace = mean(data_2d(Ymask,:));
    plot(mask_trace); axis tight;
end


end