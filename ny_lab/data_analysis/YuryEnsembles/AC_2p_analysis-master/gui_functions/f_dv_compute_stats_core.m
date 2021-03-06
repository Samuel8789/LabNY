function stats = f_dv_compute_stats_core(app, params)
% get significant peaks
% get sig onset and offset windows resp
% then extract some measure and do stat with sampling

peak_stats = 'shuff_pool'; % 'shuff_pool', 'shuff_locwise', 'z_thresh'
peak_bin_size = 7;
peak_prcntle = 99.9;
num_samp = 1000;

stat_window = [-2 3];
stat_trial_window = [0 1];
stat_resp_window = [.05 1];

%stat_resp_window = [-2 3];

z_thresh = params.z_thresh_new;

%%
n_pl = params.n_pl;
ddata = params.ddata;

num_cells = params.cdata.num_cells;
stim_times = ddata.stim_frame_index{n_pl};
%trig_window = app.working_ops.trial_num_baseline_resp_frames;
trial_types = ddata.trial_types{1};
MMN_freq = ddata.MMN_freq{1};
fr = 1000/double(ddata.proc_data{1}.frame_data.volume_period);

%%
stat_window_t = (ceil(stat_window(1)*fr):floor(stat_window(2)*fr))/fr;
stat_window_num_baseline_resp_frames = [sum(stat_window_t<=0) sum(stat_window_t>0)];   

stat_trial_window_t = (ceil(stat_trial_window(1)*fr):floor(stat_trial_window(2)*fr))/fr;
stat_trial_window_num_baseline_resp_frames = [sum(stat_trial_window_t<=0) sum(stat_trial_window_t>0)];   

%%
win1 = stat_trial_window_num_baseline_resp_frames;
trial_data_sort = f_get_stim_trig_resp(params.cdata.S, stim_times, win1);
[trial_data_sort_wctx, trial_types_wctx] =  f_s3_add_ctx_trials(trial_data_sort, trial_types, MMN_freq, app.ops);

%% choose population for shuffle
if strcmpi(params.stat_source, 'All')
    pop_stim_times = stim_times;
    trial_data_sort_stat = trial_data_sort;
elseif strcmpi(params.stat_source, 'Freqs')
    stim_idx = logical(sum(trial_types == 1:app.ops.stim.num_freqs,2));
    pop_stim_times = stim_times(stim_idx);
    trial_data_sort_stat = trial_data_sort(:,:,stim_idx);
end

num_tt = numel(app.ops.context_types_all);
num_trials = numel(pop_stim_times);
num_trial_per_stim = num_trials/app.ops.stim.num_freqs;
num_t = sum(stat_trial_window_num_baseline_resp_frames);

trial_data_sort_stat_mean = squeeze(mean(trial_data_sort_stat,2));
%% convert to z scores

trial_data_stat_mean = mean(trial_data_sort_stat,3);
trial_data_stat_sem = std(trial_data_sort_stat,[],3)/sqrt(num_trial_per_stim-1);

%% get peak resp

peak_vals = zeros(num_cells, num_tt);
peak_locs = zeros(num_cells, num_tt);
for n_tt = 1:num_tt
    trial_data_sort2 = trial_data_sort_wctx(:,:, trial_types_wctx==app.ops.context_types_all(n_tt));
    [peak_vals(:,n_tt), peak_locs(:,n_tt)] = f_get_trial_peak(trial_data_sort2, peak_bin_size);
end

if strcmpi(peak_stats, 'shuff_pool') || strcmpi(peak_stats, 'shuff_locwise')
    % compute stats for peaks
    samp_peak_vals = zeros(num_cells, num_samp);
    samp_peak_locs = zeros(num_cells, num_samp);
    wb = f_waitbar_initialize(app, 'Stats: sampling...');
    for n_cell = 1:num_cells
        f_waitbar_update(wb, n_cell/num_cells, 'Stats: sampling...');
        for n_samp = 1:num_samp
            samp_idx = randsample(num_trials, num_trial_per_stim, 1);
            samp_trial_data_sort = trial_data_sort_wctx(n_cell, :, samp_idx);
            [samp_peak_vals(n_cell,n_samp), samp_peak_locs(n_cell,n_samp)] = f_get_trial_peak(samp_trial_data_sort, peak_bin_size);
        end
        %fprintf('%d-', n_cell)
    end
    f_waitbar_close(wb);
    %fprintf('\n')
end

% figure; histogram(peak_locs(:))
% figure; histogram(samp_peak_locs(n_cell,:))
% figure; histogram(samp_peak_vals(n_cell,:))

%% get resp cell

if strcmpi(peak_stats, 'shuff_pool')
    resp_thresh = repmat(prctile(samp_peak_vals', peak_prcntle)', [1 num_t]);
    resp_cells = peak_vals>resp_thresh(:,1);
elseif strcmpi(peak_stats, 'shuff_locwise')
    resp_thresh = zeros(num_cells, num_t);
    resp_cells = zeros(num_cells, num_tt);
    for n_cell = 1:num_cells
        for n_loc = 1:num_t
            temp_th = prctile(samp_peak_vals(n_cell,samp_peak_locs(n_cell,:) == n_loc), peak_prcntle);
            if isnan(temp_th)
                resp_thresh(n_cell, n_loc) = 0;
            else
                resp_thresh(n_cell, n_loc) = temp_th;
            end
        end
        for n_tt = 1:num_tt
            resp_cells(n_cell, n_tt) = peak_vals(n_cell, n_tt) > resp_thresh(n_cell,peak_locs(n_cell,n_tt));
        end
    end
else
    resp_thresh = zeros(num_cells, num_t);
    resp_cells = zeros(num_cells, num_tt);
    for n_cell = 1:num_cells
        trial_data1 = squeeze(trial_data_sort_stat(n_cell,:,:));
        resp_thresh(n_cell,:) = mean(trial_data1,2) + z_thresh*std(trial_data1,[],2)/sqrt(num_trial_per_stim-1);
        for n_tt = 1:num_tt
            resp_cells(n_cell, n_tt) = peak_vals(n_cell, n_tt) > resp_thresh(n_cell,peak_locs(n_cell,n_tt));
        end
    end
end

%%
% 
% peak_in_win = false(num_cells, num_tt);
% peak_is_sig = false(num_cells, num_tt);
% cell_is_resp = false(num_cells, num_tt);
% peak_val_all = zeros(num_cells, num_tt);
% peak_t_all = zeros(num_cells, num_tt);
% for n_cell = 1:num_cells
%     for n_tt = 1:num_tt
%         temp_resp = trial_data_sort_wctx(n_cell,:,trial_types_wctx == app.ops.context_types_all(n_tt));
%         temp_resp2 = squeeze(temp_resp);
%         
%         mean_resp = mean(temp_resp2,2);
%         
%         [pks,locs] = findpeaks(mean_resp);
%         [peak_val, max_idx2] = max(pks);
%         max_idx = locs(max_idx2);
%         
%         peak_t_all(n_cell, n_tt) = stat_window_t(max_idx);
%         peak_val_all(n_cell, n_tt) = peak_val;
%         peak_in_win(n_cell, n_tt) = and(stat_window_t(max_idx)>=stat_resp_window(1),stat_window_t(max_idx)<=stat_resp_window(2));
%         peak_is_sig(n_cell, n_tt) = peak_val>(pop_mean{n_cell}(max_idx)+pop_z_factor{n_cell}(max_idx)*z_thresh);
%                 
%         if and(peak_in_win(n_cell, n_tt),peak_is_sig(n_cell, n_tt))
%             cell_is_resp(n_cell, n_tt) = 1;
%         end
%         
%     end
% end

% col1 = jet(10);
% figure; hold on;
% for n_tt = 1:10
%     [f, x] = ksdensity(peak_t_all(peak_is_sig(:,n_tt),n_tt));
%     plot(x, f, 'Color', col1(n_tt,:));
% end
%%

volt_dat = params.ddata.proc_data{1}.volt_data_binned{n_pl};

loco1 = volt_dat(:,3);
firing_rate = params.cdata.S;

cell_corr = corr(loco1, firing_rate');

num_shuff = 50;
samp_data = zeros(num_shuff, num_cells);

for n_shuff = 1:num_shuff
    firing_rate_s = f_shuffle_data(firing_rate);
    samp_data(n_shuff,:) = corr(loco1, firing_rate_s');
end
loco_thresh = prctile(samp_data(:), 99);

% [f_d,x_d] = ecdf(cell_corr);
% [f_s,x_s] = ecdf(samp_data(:));
% figure; hold on;
% plot(x_d, f_d, 'LineWidth', 2);
% plot(x_s, f_s, 'LineWidth', 2);
% l1 = line([loco_thresh, loco_thresh], [0 1]);
% l1.LineStyle = '--';
% l1.Color = 'r';
% legend('loco data', 'shuff', '99% thresh');

z_factor = std(samp_data(:));

loco_cell = cell_corr>loco_thresh;
loco_corr = cell_corr;
loco_z = cell_corr/z_factor;

%%
stats.pop_mean_trace = trial_data_stat_mean;
stats.pop_sem_trace = trial_data_stat_sem;
stats.pop_mean_val = mean(trial_data_stat_mean,2);
stats.pop_z_factor = mean(trial_data_stat_sem,2);
stats.cell_is_resp = resp_cells;
stats.resp_thresh = resp_thresh;
stats.peak_val_all = peak_vals;
stats.peak_t_all = stat_trial_window_t(peak_locs);
stats.stat_window_t = stat_trial_window_t; % stat_window_t
stats.z_thresh = z_thresh;
stats.peak_stats = peak_stats;
stats.peak_bin_size = peak_bin_size;
stats.peak_prcntle = peak_prcntle;
stats.num_samp = num_samp;
stats.num_cells = num_cells;
stats.accepted_cells = params.cdata.accepted_cells;
stats.loco_cell = loco_cell;
stats.loco_corr = loco_corr;
stats.loco_z = loco_z;
end