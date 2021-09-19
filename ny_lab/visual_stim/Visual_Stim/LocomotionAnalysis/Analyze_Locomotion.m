function [mean_vidget mean_trial]=Analyze_Locomotion(file_path,file_name, voltage_rate, fg1, fg2, fg3, fg4, fg5)
file_to_process=append(file_path,file_name);
T = readtable(file_to_process);
behavior = table2array(T);
voltage_duration=(size(behavior,1)/voltage_rate)/60;
[tempi,tempi2,ext] = fileparts(file_to_process);






%%
% This code calculates stimulus-induced movement behavior
% load vidget_trace (srp_vidget)

vidget_trace = behavior; %task; %srp7_ephys; %srp1_vidget;

ori = vidget_trace(:,2);  % visual stim trace
move = vidget_trace(:,3); %motion detection trace
times_second=vidget_trace(:,1)/voltage_rate;
times_minutes=times_second/60;
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

prestim = 29*voltage_rate; %2sec before stim onset
poststim = 100*voltage_rate; %5sec after stim onset
nsession = 5;

%% first chec if there is any vis stim
set(0,'CurrentFigure',fg5);
plot(times_second,ori,'k')

if isempty (find(ori>0.5))
    pos_ori=zeros(5,1);
    pos_ori(1)=610000;
    for j=2:5
        pos_ori(j)=pos_ori(j-1)+130000;
    end
else
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
end

% calculate movement
move = roundn(move,-4);
df_m = diff(move); %difference in moving
vidget = abs(df_m); % make all value positive
vidget_win(:,1) = vidget(pos_ori(1,1)-prestim:pos_ori(1,1)+poststim-1);
vidget_win(:,2) = vidget(pos_ori(2,1)-prestim:pos_ori(2,1)+poststim-1);
vidget_win(:,3) = vidget(pos_ori(3,1)-prestim:pos_ori(3,1)+poststim-1);
vidget_win(:,4) = vidget(pos_ori(4,1)-prestim:pos_ori(4,1)+poststim-1);
try
    vidget_win(:,5) = vidget(pos_ori(5,1)-prestim:pos_ori(5,1)+poststim-1);
catch
    checking_lol=vidget(pos_ori(5,1)-prestim:end);
    pad_to_do=129000-length(checking_lol)
    padded_vid=padarray(checking_lol,[pad_to_do 0],mean(checking_lol), 'post')
    vidget_win(:,5) = padded_vid
    
    
end    
% figure,
% subplot(5,1,1), plot(vidget_win(:,1),'k'), xlim([0 prestim+poststim])
% subplot(5,1,2), plot(vidget_win(:,2),'k'),xlim([0 prestim+poststim])
% subplot(5,1,3), plot(vidget_win(:,3),'k'), xlim([0 prestim+poststim])
% subplot(5,1,4), plot(vidget_win(:,4),'k'), xlim([0 prestim+poststim])
% subplot(5,1,5), plot(vidget_win(:,5),'k'), xlim([0 prestim+poststim])

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
norm_vg_average=mean(norm_vg,2);

times_stimuli_seconds=(1:length(norm_vg))/voltage_rate;
shifted_times_stimuli_seconds=times_stimuli_seconds-29;
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


mean_vidget = mean(mean(norm_vg(prestim+1:end,:)));
mean_trial = mean(norm_vg(prestim+1:end,:));


%%plot only -2 tio +5 seconds
analyzed_times=(shifted_times_stimuli_seconds>-2) & (shifted_times_stimuli_seconds<5);
analyzed_times_idx=find(analyzed_times==1);





% save('vidget_revision','mean_vidget','norm_vg')

%% ALl PLotting

set(0,'CurrentFigure',fg1);

subplot(4,1,1), plot(times_second,ori,'k')
subplot(4,1,2), plot(times_second,move,'k')
subplot(4,1,3), plot(times_second(1:end-1),df_m,'k')
subplot(4,1,4), plot(times_second(1:end-1),vidget,'k')






set(0,'CurrentFigure',fg2);

for n=1:5
AX(n) = subplot(6,1,n);
subplot(6,1,n), plot(shifted_times_stimuli_seconds,norm_vg(:,n)), %xlim([0 ds]), ylim([0 300])  
end
AX(6)=subplot(6,1,6);
subplot(6,1,6), plot(shifted_times_stimuli_seconds,mean(norm_vg,2),'k')%, xlim([0 prestim+poststim]) %ylim([0 300])
set(AX,'XLim',[-29 100])








set(0,'CurrentFigure',fg3);

for n=1:5
AX(n) = subplot(6,1,n);
subplot(6,1,n), plot(shifted_times_stimuli_seconds,norm_vg(:,n)), %xlim([0 ds]), ylim([0 300])  
end
AX(6)=subplot(6,1,6);
subplot(6,1,6), plot(shifted_times_stimuli_seconds,mean(norm_vg,2),'k')%, xlim([0 prestim+poststim]) %ylim([0 300])
set(AX,'XLim',[-29 100])
set(AX,'YLim',[0 max(norm_vg,[],'all') ])



set(0,'CurrentFigure',fg4);

for n=1:5;
AX(n) = subplot(6,1,n);
subplot(6,1,n), plot(shifted_times_stimuli_seconds(analyzed_times_idx),norm_vg(analyzed_times_idx,n)), %xlim([0 ds]), ylim([0 300])

end;
AX(6)=subplot(6,1,6);
subplot(6,1,6), plot(shifted_times_stimuli_seconds(analyzed_times_idx),norm_vg_average(analyzed_times_idx),'k')%, xlim([0 prestim+poststim]) %ylim([0 300])
set(AX,'YLim',[0 max(norm_vg(analyzed_times_idx,:),[],'all') ])
set(AX,'XLim',[-2 5]);


saveas(fg1,append(tempi,'\',tempi2,'_raw_locomotion'),'pdf')
saveas(fg2,append(tempi,'\',tempi2,'_non_normalized_trials'),'pdf')
saveas(fg3,append(tempi,'\',tempi2,'_normalized_trials'),'pdf')
saveas(fg4,append(tempi,'\',tempi2,'_habituation_window'),'pdf')

mean_vidget_5trials=mean_vidget;
mean_single_trial=mean_trial;
save(append(tempi,'\',tempi2,'mean_locomotion_5s_after_stim'),'mean_vidget_5trials','mean_single_trial')



