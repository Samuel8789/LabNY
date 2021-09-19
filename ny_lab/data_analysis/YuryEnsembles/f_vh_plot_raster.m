function fig=f_vh_plot_raster(raster, frame_times,speed, t_ms, stim_times )

[num_cells, num_frames] = size(raster);

if ~exist('frame_times', 'var')
    frame_times = 1:num_frames;
end


mean_ras = mean(raster);

fig=figure; 
if exist('speed')
    ax1 = subplot(5,1,1:3);
    imagesc(frame_times, 1:num_cells, raster)
    ax2 = subplot(5,1,4);
    plot(frame_times, mean_ras/max(mean_ras)); axis tight;
    ax3 = subplot(5,1,5);
    plot(frame_times, speed(1,:)); axis tight;
    linkaxes([ax1 ax2 ax3], 'x');
    subplot(ax1);
else   
    ax1 = subplot(4,1,1:3);
    imagesc(frame_times, 1:num_cells, raster)
    ax2 = subplot(4,1,4);
    plot(frame_times, mean_ras/max(mean_ras)); axis tight;
    if exist('stim_times', 'var')
        hold on;
        plot(t_ms, stim_times/max(stim_times));
    end
linkaxes([ax1 ax2], 'x');
subplot(ax1);

end

end