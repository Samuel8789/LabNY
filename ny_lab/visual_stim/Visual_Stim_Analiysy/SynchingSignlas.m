%% get behavior
framePeriod=0.017699014
etl_frame_period=0.020699002
rastersPerFrame=2
plane_period=etl_frame_period*rastersPerFrame
number_planes=3
volume_period=number_planes*plane_period
volume_rate=1/volume_period



voltage_rate=1000

upsampling_factor=voltage_rate/volume_rate



%%
number_of_folders=6;
folders_to_process={};
for np=1:number_of_folders
    
    folders_to_process{np} = uigetdir('G:\CodeTempRawData\LabData','Select Folder');
   
end

%%
close all
fg=figure(1);
for i=1:length(folders_to_process)
    listing = dir(folders_to_process{i});
    path=folders_to_process{i};
    zz={listing.name};

    mc_to_kalmna=~cellfun('isempty',strfind(zz,'MC.hdf5'));
    fileNames=zz(logical(mc_to_kalmna));
    for pp=1:length(fileNames)
        fileName=fileNames{pp};
        file = append(path,'\', fileName);
        h5disp(file);
        h5info(file);
        movieMC=h5read (file,'/mov');
        
        kalman = Kalman_Stack_Filter(movieMC,0.95);
        for n=1:length(kalman)
            kalman(:,:,n)=flip(flip(kalman(:,:,n,1),2));

        end

        h5create(append(file,'_Kalman_Filtered.hdf5'),'/mov',size(kalman));
        h5write(append(file,'_Kalman_Filtered.hdf5'),'/mov',kalman);
     end
    

%     raw_to_split=zeros(1,length(zz));  
%     for k=1:length(zz)
%        raw_to_split(1,k) =endsWith(zz(k),'Full_GCaMP.hdf5');
%     end  
%     fileNamesRaw=zz(logical(raw_to_split));
%     
%     for dd=1:length(fileNamesRaw)
%         fileNameRaw=fileNamesRaw{dd};
%         file = append(path,'\', fileNameRaw);
%         
%         h5disp(file);
%         movie=h5read (file,'/mov');
%         movie_duration=(size(movie,3)*volume_period)/60;
%         movie_in_time=((1:size(movie,3))-1)*volume_period;
%         
%         mean_fluorescence=mean(movie,[1,2]);
%         mean_fluorescence=mean_fluorescence(:);
%         change_mean_fluorescence=diff(mean_fluorescence);
%         
% 
% 
%         %%select cuttoffs
%         spontaneous_value=mean_fluorescence(1500);
%         cutoff=spontaneous_value+5;
%         filtered_mean=mean_fluorescence;
%         filtered_mean(1:250)=filtered_mean(1:250)+cutoff;
%         filtered_mean(8000:end)=filtered_mean(8000:end)+cutoff;
%         
%         set(0,'CurrentFigure',fg);
%         subplot(4,2,[1,2])
%         plot(mean_fluorescence)
%         subplot(4,2,[3,4])
%         plot(movie_in_time,mean_fluorescence)
%         subplot(4,2,[5,6])
%         plot(movie_in_time(1:end-1),change_mean_fluorescence)
%         
%         
%         
%         spont_period_idx=find(filtered_mean<cutoff);
%         spont_fluorescence=movie(:,:,spont_period_idx);
%         spont_fluorescence_times=movie_in_time(spont_period_idx);
% 
%         movie_after_spont=movie(:,:,spont_period_idx(end)+1:end);
%         movie_after_spont_times=movie_in_time(spont_period_idx(end)+1:end);
%         after_spont=mean_fluorescence(spont_period_idx(end)+1:end);
%         trials_idx=find(after_spont>115);
%         only_trials=movie_after_spont(:,:,trials_idx);
%         only_trials_time=movie_after_spont_times(trials_idx);
%         
%         set(0,'CurrentFigure',fg);
%         
%         roro=mean(spont_fluorescence,[1,2]);
%         roro=roro(:);
%         subplot(4,2,7)
%         plot(roro)
% 
%         lolo=mean(only_trials,[1,2]);
%         lolo=lolo(:);
%         subplot(4,2,8)
%         plot(lolo)
%         
%         pause
%         
%         save(append('G:\CodeTempRawData\LabData\Chandelier_Imaging\LocomotionHAbit','\','SPHB','_averaegs.mat'),'fidgeting_habit')

        
%     %% save files
% 
%         h5create(append(file,'_Spontaneous.hdf5'),'/mov',size(spont_fluorescence));
%         h5write(append(file,'_Spontaneous.hdf5'),'/mov',spont_fluorescence);
% 
%         h5create(append(file,'_Trials.hdf5'),'/mov',size(only_trials));
%         h5write(append(file,'_Trials.hdf5'),'/mov',only_trials);
%      end
end




















%%

[locfileName,locpath]=uigetfile('*.csv','Select the INPUT DATA FILE(s)','MultiSelect','on');

fidgeting=zeros(length(locfileName),6);
close all
for i=1:length(locfileName)
    file_path=locpath;
    file_name=locfileName{i};
    [fidgeting(i,1) fidgeting(i,2:end)]=Analyze_Locomotion(file_path, file_name, voltage_rate, figure(1),figure(2),figure(3),figure(4), figure(5) );
    
end
norm_fidgeting=zeros(length(locfileName),6);

for col=1:size(norm_fidgeting,2)
    norm_fidgeting(:,col)=fidgeting(:,col)/fidgeting(1,col)
end

hab_index=1-norm_fidgeting(:,1)
habit_hab_idx=hab_index
habit_control_idx=hab_index

habit_com=[habit_hab_idx habit_control_idx]


figure(6)
hold on
plot(habit_com(:,1), 'k')
plot(habit_com(:,2), 'r')

%%


fidgeting_control=zeros(7,6);
fidgeting_control(1,1)=mean_vidget_5trials
fidgeting_control(1,2:end)=mean_single_trial
fidgeting_control(2,1)=mean_vidget_5trials
fidgeting_control(2,2:end)=mean_single_trial
fidgeting_control(3,1)=mean_vidget_5trials
fidgeting_control(3,2:end)=mean_single_trial
fidgeting_control(4,1)=mean_vidget_5trials
fidgeting_control(4,2:end)=mean_single_trial
fidgeting_control(5,1)=mean_vidget_5trials
fidgeting_control(5,2:end)=mean_single_trial
fidgeting_control(6,1)=mean_vidget_5trials
fidgeting_control(6,2:end)=mean_single_trial
fidgeting_control(7,1)=mean_vidget_5trials
fidgeting_control(7,2:end)=mean_single_trial

fidgeting_habit=zeros(7,6);
fidgeting_habit(1,1)=mean_vidget_5trials
fidgeting_habit(1,2:end)=mean_single_trial
fidgeting_habit(2,1)=mean_vidget_5trials
fidgeting_habit(2,2:end)=mean_single_trial
fidgeting_habit(3,1)=mean_vidget_5trials
fidgeting_habit(3,2:end)=mean_single_trial
fidgeting_habit(4,1)=mean_vidget_5trials
fidgeting_habit(4,2:end)=mean_single_trial
fidgeting_habit(5,1)=mean_vidget_5trials
fidgeting_habit(5,2:end)=mean_single_trial
fidgeting_habit(6,1)=mean_vidget_5trials
fidgeting_habit(6,2:end)=mean_single_trial
fidgeting_habit(7,1)=mean_vidget_5trials
fidgeting_habit(7,2:end)=mean_single_trial


save(append('G:\CodeTempRawData\LabData\Chandelier_Imaging\LocomotionHAbit','\','SPHB','_averaegs'),'fidgeting_habit')
save(append('G:\CodeTempRawData\LabData\Chandelier_Imaging\LocomotionHAbit','\','SPHC','_averaegs'), 'fidgeting_control')

norm_fidgeting_habit=zeros(7,6);
norm_fidgeting_control=zeros(7,6);

for col=1:size(norm_fidgeting_habit,2)
    norm_fidgeting_habit(:,col)=fidgeting_habit(:,col)/fidgeting_habit(1,col)
end

for col=1:size(norm_fidgeting_control,2)
    norm_fidgeting_control(:,col)=fidgeting_control(:,col)/fidgeting_control(1,col)
end

habit_idx_habit=1-norm_fidgeting_habit(:,1)
habit_idx_control=1-norm_fidgeting_control(:,1)

habit_com=[habit_idx_habit habit_idx_control]


figure(6)
hold on
scatter([1:7], habit_com(:,1), 'k')
scatter([1:7], habit_com(:,2), 'r')
plot(habit_com(:,1), 'k')
plot(habit_com(:,2), 'r')
xlabel('Experimental Day') 
ylabel('Habituation Index') 
legend('Habituation', 'Control')