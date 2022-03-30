% This code calculates stimulus-induced movement behavior
% load vidget_trace (srp_vidget)

vidget_trace = srp1_ephys; %task; %srp7_ephys; %srp1_vidget;

figure,
subplot(2,1,1), plot(vidget_trace(:,2))
subplot(2,1,2), plot(vidget_trace(:,3))
ori = vidget_trace(:,2);  % visual stim trace
move = vidget_trace(:,3); %motion detection trace

% figure,
% subplot(2,1,1), plot(vidget_trace(:,3))
% subplot(2,1,2), plot(vidget_trace(:,4))
% ori = vidget_trace(:,3);  % visual stim trace
% move = vidget_trace(:,4);  %motion detection trace

% figure,
% subplot(2,1,1), plot(vidget_trace(:,4))
% subplot(2,1,2), plot(vidget_trace(:,5))
% ori = vidget_trace(:,4);   % visual stim trace
% move = vidget_trace(:,3);  %motion detection trace

sr = 10000; %input('sampling rate: ');
prestim = 29*sr; %2sec before stim onset
poststim = 100*sr; %5sec after stim onset
nsession = 5;

% find indices for stim-on
ori = roundn(ori,-1);
for i = 1:length(ori)
    if ori(i)<0.2
        ori(i)=0;
    else
        ori(i)=1;
    end
end

df_ori = diff(ori);

for i = 1:size(df_ori,1)
    if df_ori(i) < 0 % find indices for stim-on
        df_ori(i) = 0;
    end 
end

pos_ori = find(df_ori); % indices for stim-on

% calculate movement
move = roundn(move,-4);
df_m = diff(move); %difference in moving
vidget = abs(df_m); % make all value positive
vidget_win = vidget(pos_ori(1,1)-prestim:pos_ori(1,1)+poststim-1);
vidget_win(:,2) = vidget(pos_ori(2,1)-prestim:pos_ori(2,1)+poststim-1);
vidget_win(:,3) = vidget(pos_ori(3,1)-prestim:pos_ori(3,1)+poststim-1);
vidget_win(:,4) = vidget(pos_ori(4,1)-prestim:pos_ori(4,1)+poststim-1);
vidget_win(:,5) = vidget(pos_ori(5,1)-prestim:pos_ori(5,1)+poststim-1);

% figure,
% subplot(5,1,1), plot(vidget_win(:,1),'k'), xlim([0 prestim+poststim])
% subplot(5,1,2), plot(vidget_win(:,2),'k'),xlim([0 prestim+poststim])
% subplot(5,1,3), plot(vidget_win(:,3),'k'), xlim([0 prestim+poststim])
% subplot(5,1,4), plot(vidget_win(:,4),'k'), xlim([0 prestim+poststim])
% subplot(5,1,5), plot(vidget_win(:,5),'k'), xlim([0 prestim+poststim])

figure,
subplot(4,1,1), plot(ori,'k')
subplot(4,1,2), plot(move,'k')
subplot(4,1,3), plot(df_m,'k')
subplot(4,1,4), plot(vidget,'k')


%%
% % down-sample to 100Hz
% ds = (prestim + poststim)*(100/sr); %700; % # of samples for 7 sec data699
% nsession = 5;
% temp = zeros(ds,nsession);
% counter = 1;
% for i = 1:sr/100:size(vidget_win,1)
%     temp(counter,:) = sum(vidget_win(i:i+(sr/100)-1,:));
%     counter = counter+1;
% end
% vidget_ds = temp;

% normalze to the mean during 2-s before stim onset
mean_prestim = mean(vidget_win(1:prestim,:));
norm_vg = zeros(size(vidget_win));
for i = 1:nsession
    norm_vg(:,i) = vidget_win(:,i)./mean_prestim(1,i);
end

% % plot 5 sessions of vidget
% % before normalization
% figure,
% subplot(5,1,1), plot(vidget_ds(:,1)), xlim([0 ds]),% ylim([0 3])
% subplot(5,1,2), plot(vidget_ds(:,2)), xlim([0 ds]), %ylim([0 3])
% subplot(5,1,3), plot(vidget_ds(:,3)), xlim([0 ds]), %ylim([0 3])
% subplot(5,1,4), plot(vidget_ds(:,4)), xlim([0 ds]),% ylim([0 3])
% subplot(5,1,5), plot(vidget_ds(:,5)), xlim([0 ds]),% ylim([0 3])

% after normalization
% figure,
% subplot(5,1,1), plot(norm_vg(:,1)), %xlim([0 ds]), ylim([0 300])
% subplot(5,1,2), plot(norm_vg(:,2)), %xlim([0 ds]), ylim([0 300])
% subplot(5,1,3), plot(norm_vg(:,3)), %xlim([0 ds]), ylim([0 300])
% subplot(5,1,4), plot(norm_vg(:,4)), %xlim([0 ds]), ylim([0 300])
% subplot(5,1,5), plot(norm_vg(:,5)), %xlim([0 ds]), ylim([0 300])

figure, plot(mean(norm_vg,2),'k'), xlim([0 prestim+poststim]), %ylim([0 300])

mean_vidget = mean(mean(norm_vg(prestim+1:end,:)));
mean_trial = mean(norm_vg(prestim+1:end,:));

% save('vidget_revision','mean_vidget','norm_vg')
%% Revision May 2020
%1sec post stim
mean_vidget_1s = mean(mean(norm_vg(prestim+1:prestim+1*sr,:)));
mean_vidget_2s = mean(mean(norm_vg(prestim+1:prestim+2*sr,:)));
mean_vidget_3s = mean(mean(norm_vg(prestim+1:prestim+3*sr,:)));
mean_vidget_4s = mean(mean(norm_vg(prestim+1:prestim+4*sr,:)));
mean_vidget_5s = mean(mean(norm_vg(prestim+1:prestim+5*sr,:)));
mean_vidget_6s = mean(mean(norm_vg(prestim+1:prestim+6*sr,:)));
mean_vidget_7s = mean(mean(norm_vg(prestim+1:prestim+7*sr,:)));
mean_vidget_8s = mean(mean(norm_vg(prestim+1:prestim+8*sr,:)));
mean_vidget_9s = mean(mean(norm_vg(prestim+1:prestim+9*sr,:)));
mean_vidget_10s = mean(mean(norm_vg(prestim+1:prestim+10*sr,:)));
mean_vidget_25s = mean(mean(norm_vg(prestim+1:prestim+25*sr,:)));
mean_vidget_50s = mean(mean(norm_vg(prestim+1:prestim+50*sr,:)));
mean_vidget_100s = mean(mean(norm_vg(prestim+1:prestim+100*sr,:)));


