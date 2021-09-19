% clear;
close all;
% clear all
addpath('C:\Users\sp3660\Documents\GitHub\YuryEnsembles')
caima
%%

directs='C:\\Users\\sp3660\\Desktop\\TemporaryProcessing';
dfdtmatrix='\\allplanedfdt.mat';
foopsimatrix='\\allplanefoopsi.mat';
foopsigratingmatrix='\\allplanefoopsigrating.mat';
Cmatrix='\\allplanesC.mat';

dfdtmatrixchand='\\allchanddfdt.mat';

laod1 = load(append(directs,dfdtmatrixchand));
firing_rate = laod1.alchandnedfdt;

SpeedToPlot4=abs(diff(Locomotion))';
SpeedToPlot4(0:62:(length(vistim)-1))



raster=firing_rate';
imagesc(raster)
%SpeedToPlot=SpeedGrats;
%raster=double(TotalDriftSpikes>0);
%raster=double(SpikSpont>0);
frame_rate=16;
%%
% raster load
% load('collected_data_spikes.mat');
[num_cells, num_frames] = size(raster);

% % frame times load
% XML_data = extract_frame_data_from_XML2('a2-RFP-spASAP-s1-jRGECO1a-PoM-014');
% frame_times = XML_data.frame_times(1:num_frames)/1000;
% 
% %volt times load 
% volt_data = csvread('a2-RFP-spASAP-s1-jRGECO1a-PoM-014_Cycle00001_VoltageRecording_001.csv',1);
% volt_t = volt_data(:,1)/1000;
% stim_times_ms = volt_data(:,2);

%% find best smoothing window
if 1
    smooth_range = 50:50:300;
    num_shuff_reps = 100;
    dim_corr_sm = zeros(numel(smooth_range),1);
    for n_sm = 1:numel(smooth_range)
        sigma_frames2 = smooth_range(n_sm)/(1000/frame_rate);
        % make kernel
        smooth_raster2 = f_smooth_gaus(raster, sigma_frames2, 1);
        dim_corr_sm(n_sm) = f_vh_get_ens_number(smooth_raster2, num_shuff_reps);
    end
    figure('Name','Range'); plot(smooth_range, dim_corr_sm);
    title('num ensembles vs smooth');
    xlabel('Smooth window (ms)');
    ylabel('correlations dimensionality')
end

[~, MaxIdx]=max(dim_corr_sm);


%% smooth
%frame_rate = 1000/XML_data.frame_period;

frame_times=1:length(raster);
% 80 200 450
sigma = 200;
%sigma=smooth_range(MaxIdx);
sigma_frames = sigma/(1000/frame_rate);

smooth_raster = f_smooth_gaus(raster, sigma_frames, 1);

%normalize
raster_norm = smooth_raster - mean(smooth_raster,2);
raster_norm = raster_norm./std(raster_norm,[],2); 

%% get number of ensembles
num_shuff_reps = 500;
dim_corr = f_vh_get_ens_number(raster_norm, num_shuff_reps);
fprintf('Threre are probably %.2f ensembles\n', dim_corr);

%% get top SVD components
num_comp = round(dim_corr); % calculate with get ens number script 
[U,S,V] = svd(raster_norm);
raster_LR = (U(:,1:num_comp)*S(1:num_comp,1:num_comp)*V(:,1:num_comp)');

% sort cell order
[dend_order_cell,~,Z] = f_hcluster(raster_LR, 'cosine');
dend_order_cell=[flipud(dend_order_cell')]';

%% get top NMF components
[d_W,d_H] = nnmf(raster_norm,3);
raster_LR_NMF = d_W*d_H;

%f_vh_plot_raster(raster_LR_NMF(dend_order_cell,:), frame_times, volt_t, stim_times_ms);
f_vh_plot_raster(raster_LR_NMF(dend_order_cell,:), frame_times);
title(sprintf('raster Low Rank NMF; smooth=%dms, %d %d comp', sigma, num_comp))
set(gcf,'Name','NMFRaster')

%% plot stuff
%f_vh_plot_raster(raster(dend_order_cell,:), frame_times, volt_t, stim_times_ms);
test=f_vh_plot_raster(raster(dend_order_cell,:), frame_times);
title(sprintf('raster'))
set(gcf,'Name','Raster')
%f_vh_plot_raster(raster_norm(dend_order_cell,:), frame_times, volt_t, stim_times_ms);
f_vh_plot_raster( raster_norm(dend_order_cell,:), frame_times);
title(sprintf('raster norm; smooth=%dms', sigma))
set(gcf,'Name','NormRaster')
%f_vh_plot_raster(raster_LR(dend_order_cell,:), frame_times, volt_t, stim_times_ms);
f_vh_plot_raster(raster_LR(dend_order_cell,:), frame_times);
title(sprintf('raster Low Rank SVD; smooth=%dms, %d %d comp', sigma, num_comp))
set(gcf,'Name','SVDRaster')
DendFrig=figure; dendrogram(Z,0,'Orientation','left')
title(sprintf('cell sorted tree; smooth=%dms', sigma));
set(gcf,'Name','Dendrogram')

%% similarity 
SI_cells_norm = 1-pdist2(raster_norm(dend_order_cell,:),raster_norm(dend_order_cell,:), 'cosine');
%SI_cells_norm = SI_cells_norm - diag(diag(SI_cells_norm));
SI_cells_LR = 1-pdist2(raster_LR(dend_order_cell,:),raster_LR(dend_order_cell,:), 'cosine');
%SI_cells_LR = SI_cells_LR - diag(diag(SI_cells_LR));
figure('Name','SI')
subplot(2,1,1);
imagesc(SI_cells_norm);
title(sprintf('cosine similarity raster norm; smooth=%dms', sigma));
subplot(2,1,2);
imagesc(SI_cells_LR);
title(sprintf('cosine similarity low rank; smooth=%dms, %d %d comp', sigma, num_comp));




