%video1
%plane1 SPHB_Habit_Day1_940_1634_2planes-000_plane1
habit_day1_cell1_MCMCspikes=proc.deconv.MCMC.S{3, 1};
plot(habit_day1_cell1_MCMCspikes)
habit_day1_cell1_MCMCspikes_binarized=habit_day1_cell1_MCMCspikes>0;

habit_day1_cell2_MCMCspikes=proc.deconv.MCMC.S{4, 1};
plot(habit_day1_cell2_MCMCspikes)
habit_day1_cell2_MCMCspikes_binarized=habit_day1_cell2_MCMCspikes>0;

%video1
%plane2 SPHB_500Filter_940_3planes_Habituation_Day5-000_plane1


habit_day1_cell3_MCMCspikes=proc.deconv.MCMC.S{3, 1};
plot(habit_day1_cell3_MCMCspikes)
habit_day1_cell3_MCMCspikes_binarized=habit_day1_cell3_MCMCspikes>0;


%video1
%plane1 SPHB_500Filter_940_3planes_Habituation_Day5-000_plane2


habit_day1_cell4_MCMCspikes=proc.deconv.c_foopsi.S{3, 1}  ;
plot(habit_day1_cell4_MCMCspikes)
habit_day1_cell4_MCMCspikes_binarized=habit_day1_cell4_MCMCspikes>0;

habit_day1_cell5_MCMCspikes=proc.deconv.c_foopsi.S{6, 1}  
plot(habit_day1_cell5_MCMCspikes)
habit_day1_cell5_MCMCspikes_binarized=habit_day1_cell5_MCMCspikes>0;

habit_day1_cell6_MCMCspikes=proc.deconv.c_foopsi.S{8, 1}  
plot(habit_day1_cell6_MCMCspikes)
habit_day1_cell6_MCMCspikes_binarized=habit_day1_cell6_MCMCspikes>0;

habit_day1_cell7_MCMCspikes=proc.deconv.c_foopsi.S{11, 1}  
plot(habit_day1_cell7_MCMCspikes)
habit_day1_cell7_MCMCspikes_binarized=habit_day1_cell7_MCMCspikes>0;

habit_day1_cell8_MCMCspikes=proc.deconv.c_foopsi.S{13, 1}  
plot(habit_day1_cell8_MCMCspikes)
habit_day1_cell8_MCMCspikes_binarized=habit_day1_cell8_MCMCspikes>0;

habit_day1_cell9_MCMCspikes=proc.deconv.c_foopsi.S{14, 1}  
plot(habit_day1_cell9_MCMCspikes)
habit_day1_cell9_MCMCspikes_binarized=habit_day1_cell9_MCMCspikes>0;



%%

%% get behavior
framePeriod=0.017699014
etl_frame_period=0.020699002
rastersPerFrame=2
plane_period=etl_frame_period*rastersPerFrame
number_planes=2
volume_period=number_planes*plane_period
volume_rate=1/volume_period



voltage_rate=1000

upsampling_factor=voltage_rate/volume_rate
%%
number_of_folders=1;
folders_to_process={};
for np=1:number_of_folders
    
    folders_to_process{np} = uigetdir('G:\CodeTempRawData\LabData','Select Folde');
   
end

%%
close all
fg=figure(1);
for i=1:length(folders_to_process)
    listing = dir(folders_to_process{i});
    path=folders_to_process{i};
    zz={listing.name};

%     mc_to_kalmna=~cellfun('isempty',strfind(zz,'MC.hdf5'));
%     fileNames=zz(logical(mc_to_kalmna));
%     for pp=1:length(fileNames)
%         fileName=fileNames{pp};
%         file = append(path,'\', fileName);
%         h5disp(file);
%         h5info(file);
%         movieMC=h5read (file,'/mov');
%         
%         kalman = Kalman_Stack_Filter(movieMC,0.95);
%         for n=1:length(kalman)
%             kalman(:,:,n)=flip(flip(kalman(:,:,n,1),2));
% 
%         end
% 
%         h5create(append(file,'_Kalman_Filtered.hdf5'),'/mov',size(kalman));
%         h5write(append(file,'_Kalman_Filtered.hdf5'),'/mov',kalman);
%      end
%     

    raw_to_split=zeros(1,length(zz));  
    for k=1:length(zz)
       raw_to_split(1,k) =endsWith(zz(k),'Full_GCaMP.hdf5');
    end  
    fileNamesRaw=zz(logical(raw_to_split));
    
    for dd=1:length(fileNamesRaw)
        fileNameRaw=fileNamesRaw{dd};
        file = append(path,'\', fileNameRaw);
        
        h5disp(file);
        movie=h5read (file,'/mov');
        movie_duration=(size(movie,3)*volume_period)/60;
        movie_in_time=((1:size(movie,3))-1)*volume_period;
        MOVIE_FRAMSE=size(movie,3)
        
        mean_fluorescence=mean(movie,[1,2]);
        mean_fluorescence=mean_fluorescence(:);
        change_mean_fluorescence=diff(mean_fluorescence);
        


        %%select cuttoffs
        spontaneous_value=mean_fluorescence(1500);
        cutoff=spontaneous_value+5;
        filtered_mean=mean_fluorescence;
        filtered_mean(1:250)=filtered_mean(1:250)+cutoff;
        filtered_mean(8000:end)=filtered_mean(8000:end)+cutoff;
        
        set(0,'CurrentFigure',fg);
        subplot(4,2,[1,2])
        plot(mean_fluorescence)
        subplot(4,2,[3,4])
        plot(movie_in_time,mean_fluorescence)
        subplot(4,2,[5,6])
        plot(movie_in_time(1:end-1),change_mean_fluorescence)
        
        
        
        spont_period_idx=find(filtered_mean<cutoff);
        spont_fluorescence=movie(:,:,spont_period_idx);
        spont_fluorescence_times=movie_in_time(spont_period_idx);

        movie_after_spont=movie(:,:,spont_period_idx(end)+1:end);
        movie_after_spont_times=movie_in_time(spont_period_idx(end)+1:end);
        after_spont=mean_fluorescence(spont_period_idx(end)+1:end);
        trials_idx=find(after_spont>115);
        only_trials=movie_after_spont(:,:,trials_idx);
        only_trials_time=movie_after_spont_times(trials_idx);
        
        set(0,'CurrentFigure',fg);
        
        roro=mean(spont_fluorescence,[1,2]);
        roro=roro(:);
        subplot(4,2,7)
        plot(roro)

        lolo=mean(only_trials,[1,2]);
        lolo=lolo(:);
        subplot(4,2,8)
        plot(lolo)
        
        pause
        
        save(append(file,'_stiminfo.mat'),'movie_duration', 'movie_in_time', 'fileNameRaw', 'spont_period_idx', 'MOVIE_FRAMSE', 'trials_idx')

        
%     %% save files
% 
%         h5create(append(file,'_Spontaneous.hdf5'),'/mov',size(spont_fluorescence));
%         h5write(append(file,'_Spontaneous.hdf5'),'/mov',spont_fluorescence);
% 
%         h5create(append(file,'_Trials.hdf5'),'/mov',size(only_trials));
%         h5write(append(file,'_Trials.hdf5'),'/mov',only_trials);
     end
end


%%

habit_day1_cell1_MCMCspikes_spont=habit_day1_cell1_MCMCspikes(1,spont_period_idx);
habit_day1_cell1_MCMCspikes_spont_times=movie_in_time(spont_period_idx);

habit_day1_cell1_MCMCspikes_after_spont=habit_day1_cell1_MCMCspikes(1,spont_period_idx(end)+1:end);
habit_day1_cell1_MCMCspikes_after_spont_times=movie_in_time(spont_period_idx(end)+1:end);

habit_day1_cell1_MCMCspikes_only_trials=habit_day1_cell1_MCMCspikes_after_spont(1,trials_idx);
habit_day1_cell1_MCMCspikes_only_trials_time=habit_day1_cell1_MCMCspikes_after_spont_times(trials_idx);




habit_day1_cell2_MCMCspikes_spont=habit_day1_cell2_MCMCspikes(1,spont_period_idx);
habit_day1_cell2_MCMCspikes_spont_times=movie_in_time(spont_period_idx);

habit_day1_cell2_MCMCspikes_after_spont=habit_day1_cell2_MCMCspikes(1,spont_period_idx(end)+1:end);
habit_day1_cell2_MCMCspikes_after_spont_times=movie_in_time(spont_period_idx(end)+1:end);

habit_day1_cell2_MCMCspikes_only_trials=habit_day1_cell2_MCMCspikes_after_spont(1,trials_idx);
habit_day1_cell2_MCMCspikes_only_trials_time=habit_day1_cell2_MCMCspikes_after_spont_times(trials_idx);

%%
habit_day1_cell3_MCMCspikes_spont=habit_day1_cell3_MCMCspikes(1,spont_period_idx);
habit_day1_cell3_MCMCspikes_spont_times=movie_in_time(spont_period_idx);

habit_day1_cell3_MCMCspikes_after_spont=habit_day1_cell3_MCMCspikes(1,spont_period_idx(end)+1:end);
habit_day1_cell3_MCMCspikes_after_spont_times=movie_in_time(spont_period_idx(end)+1:end);

habit_day1_cell3_MCMCspikes_only_trials=habit_day1_cell3_MCMCspikes_after_spont(1,trials_idx);
habit_day1_cell3_MCMCspikes_only_trials_time=habit_day1_cell3_MCMCspikes_after_spont_times(trials_idx);


habit_day1_cell4_MCMCspikes_spont=habit_day1_cell4_MCMCspikes(1,spont_period_idx);
habit_day1_cell4_MCMCspikes_spont_times=movie_in_time(spont_period_idx);

habit_day1_cell4_MCMCspikes_after_spont=habit_day1_cell4_MCMCspikes(1,spont_period_idx(end)+1:end);
habit_day1_cell4_MCMCspikes_after_spont_times=movie_in_time(spont_period_idx(end)+1:end);

habit_day1_cell4_MCMCspikes_only_trials=habit_day1_cell4_MCMCspikes_after_spont(1,trials_idx);
habit_day1_cell4_MCMCspikes_only_trials_time=habit_day1_cell4_MCMCspikes_after_spont_times(trials_idx);


habit_day1_cell5_MCMCspikes_spont=habit_day1_cell5_MCMCspikes(1,spont_period_idx);
habit_day1_cell5_MCMCspikes_spont_times=movie_in_time(spont_period_idx);

habit_day1_cell5_MCMCspikes_after_spont=habit_day1_cell5_MCMCspikes(1,spont_period_idx(end)+1:end);
habit_day1_cell5_MCMCspikes_after_spont_times=movie_in_time(spont_period_idx(end)+1:end);

habit_day1_cell5_MCMCspikes_only_trials=habit_day1_cell5_MCMCspikes_after_spont(1,trials_idx);
habit_day1_cell5_MCMCspikes_only_trials_time=habit_day1_cell5_MCMCspikes_after_spont_times(trials_idx);


habit_day1_cell6_MCMCspikes_spont=habit_day1_cell6_MCMCspikes(1,spont_period_idx);
habit_day1_cell6_MCMCspikes_spont_times=movie_in_time(spont_period_idx);

habit_day1_cell6_MCMCspikes_after_spont=habit_day1_cell6_MCMCspikes(1,spont_period_idx(end)+1:end);
habit_day1_cell6_MCMCspikes_after_spont_times=movie_in_time(spont_period_idx(end)+1:end);

habit_day1_cell6_MCMCspikes_only_trials=habit_day1_cell6_MCMCspikes_after_spont(1,trials_idx);
habit_day1_cell6_MCMCspikes_only_trials_time=habit_day1_cell6_MCMCspikes_after_spont_times(trials_idx);


habit_day1_cell7_MCMCspikes_spont=habit_day1_cell7_MCMCspikes(1,spont_period_idx);
habit_day1_cell7_MCMCspikes_spont_times=movie_in_time(spont_period_idx);

habit_day1_cell7_MCMCspikes_after_spont=habit_day1_cell7_MCMCspikes(1,spont_period_idx(end)+1:end);
habit_day1_cell7_MCMCspikes_after_spont_times=movie_in_time(spont_period_idx(end)+1:end);

habit_day1_cell7_MCMCspikes_only_trials=habit_day1_cell7_MCMCspikes_after_spont(1,trials_idx);
habit_day1_cell7_MCMCspikes_only_trials_time=habit_day1_cell7_MCMCspikes_after_spont_times(trials_idx);


habit_day1_cell8_MCMCspikes_spont=habit_day1_cell8_MCMCspikes(1,spont_period_idx);
habit_day1_cell8_MCMCspikes_spont_times=movie_in_time(spont_period_idx);

habit_day1_cell8_MCMCspikes_after_spont=habit_day1_cell8_MCMCspikes(1,spont_period_idx(end)+1:end);
habit_day1_cell8_MCMCspikes_after_spont_times=movie_in_time(spont_period_idx(end)+1:end);

habit_day1_cell8_MCMCspikes_only_trials=habit_day1_cell8_MCMCspikes_after_spont(1,trials_idx);
habit_day1_cell8_MCMCspikes_only_trials_time=habit_day1_cell8_MCMCspikes_after_spont_times(trials_idx);


habit_day1_cell9_MCMCspikes_spont=habit_day1_cell9_MCMCspikes(1,spont_period_idx);
habit_day1_cell9_MCMCspikes_spont_times=movie_in_time(spont_period_idx);

habit_day1_cell9_MCMCspikes_after_spont=habit_day1_cell9_MCMCspikes(1,spont_period_idx(end)+1:end);
habit_day1_cell9_MCMCspikes_after_spont_times=movie_in_time(spont_period_idx(end)+1:end);

habit_day1_cell9_MCMCspikes_only_trials=habit_day1_cell9_MCMCspikes_after_spont(1,trials_idx);
habit_day1_cell9_MCMCspikes_only_trials_time=habit_day1_cell9_MCMCspikes_after_spont_times(trials_idx);
%% have to downsample the signals form the 2 plane mocies
% spont=7248
% trails=10328
% spont =4831
% trials=5237
% downsampledspont 4832
% downsamplestrials= 6886
framesrate2=12.0779
framesrate3=8.0519
difference_frames=6886-5237
downsampling_factor=1/(framesrate2/framesrate3)

habit_day1_cell1_MCMCspikes_spont_times_downsampled = resample(habit_day1_cell1_MCMCspikes_spont_times,2,3);
habit_day1_cell1_MCMCspikes_spont_downsampled = resample(double(habit_day1_cell1_MCMCspikes_spont),2,3);
habit_day1_cell1_MCMCspikes_spont_downsampled>0;

habit_day1_cell1_MCMCspikes_only_trials_times_downsampled = resample(habit_day1_cell1_MCMCspikes_only_trials_time,2,3);
habit_day1_cell1_MCMCspikes_only_trials_downsampled = resample(double(habit_day1_cell1_MCMCspikes_only_trials),2,3);
habit_day1_cell1_MCMCspikes_only_trials_downsampled>0;

habit_day1_cell2_MCMCspikes_spont_times_downsampled = resample(habit_day1_cell2_MCMCspikes_spont_times,2,3);
habit_day1_cell2_MCMCspikes_spont_downsampled = resample(double(habit_day1_cell2_MCMCspikes_spont),2,3);
habit_day1_cell2_MCMCspikes_spont_downsampled>0;

habit_day1_cell2_MCMCspikes_only_trials_times_downsampled = resample(habit_day1_cell2_MCMCspikes_only_trials_time,2,3);
habit_day1_cell2_MCMCspikes_only_trials_downsampled = resample(double(habit_day1_cell2_MCMCspikes_only_trials),2,3);
habit_day1_cell2_MCMCspikes_only_trials_downsampled>0;


habit_day1_cell1_MCMCspikes_spont_times_downsampled=habit_day1_cell1_MCMCspikes_spont_times_downsampled(1:end-1);
habit_day1_cell1_MCMCspikes_spont_downsampled=habit_day1_cell1_MCMCspikes_spont_downsampled(1:end-1);

habit_day1_cell1_MCMCspikes_only_trials_times_downsampled=habit_day1_cell1_MCMCspikes_only_trials_times_downsampled(1:5236);
habit_day1_cell1_MCMCspikes_only_trials_downsampled=habit_day1_cell1_MCMCspikes_only_trials_downsampled(1:5237);

habit_day1_cell2_MCMCspikes_spont_times_downsampled=habit_day1_cell2_MCMCspikes_spont_times_downsampled(1:end-1);
habit_day1_cell2_MCMCspikes_spont_downsampled=habit_day1_cell2_MCMCspikes_spont_downsampled(1:end-1);

habit_day1_cell2_MCMCspikes_only_trials_times_downsampled=habit_day1_cell2_MCMCspikes_only_trials_times_downsampled(1:5236);
habit_day1_cell2_MCMCspikes_only_trials_downsampled=habit_day1_cell2_MCMCspikes_only_trials_downsampled(1:5237);
%%
raster_matrix_spont=zeros(9,4831);
raster_matrix_trials=zeros(9,5237);

raster_matrix_spont(1,:)=habit_day1_cell1_MCMCspikes_spont_downsampled;
raster_matrix_spont(2,:)=habit_day1_cell2_MCMCspikes_spont_downsampled;
raster_matrix_spont(3,:)=habit_day1_cell3_MCMCspikes_spont;
raster_matrix_spont(4,:)=habit_day1_cell4_MCMCspikes_spont;
raster_matrix_spont(5,:)=habit_day1_cell5_MCMCspikes_spont;
raster_matrix_spont(6,:)=habit_day1_cell6_MCMCspikes_spont;
raster_matrix_spont(7,:)=habit_day1_cell7_MCMCspikes_spont;
raster_matrix_spont(8,:)=habit_day1_cell8_MCMCspikes_spont;
raster_matrix_spont(9,:)=habit_day1_cell9_MCMCspikes_spont;

raster_matrix_trials(1,:)=habit_day1_cell1_MCMCspikes_only_trials_downsampled;
raster_matrix_trials(2,:)=habit_day1_cell2_MCMCspikes_only_trials_downsampled;
raster_matrix_trials(3,:)=habit_day1_cell3_MCMCspikes_only_trials;
raster_matrix_trials(4,:)=habit_day1_cell4_MCMCspikes_only_trials;
raster_matrix_trials(5,:)=habit_day1_cell5_MCMCspikes_only_trials;
raster_matrix_trials(6,:)=habit_day1_cell6_MCMCspikes_only_trials;
raster_matrix_trials(7,:)=habit_day1_cell7_MCMCspikes_only_trials;
raster_matrix_trials(8,:)=habit_day1_cell8_MCMCspikes_only_trials;
raster_matrix_trials(9,:)=habit_day1_cell9_MCMCspikes_only_trials;


save(append(file,'_rastersday1.mat'),'raster_matrix_spont', 'raster_matrix_trials')
%%

random_perm=randperm(size(raster_matrix_spont, 1));
random_raster_matrix_spont = raster_matrix_spont(random_perm, :);
random_raster_matrix_trials = raster_matrix_trials(random_perm, :);

spont_fluorescence_times(1:end-1)
only_trials_time(1:end-1)
save(append(file,'_rastersday_rand.mat'),'random_raster_matrix_spont', 'random_raster_matrix_trials')


figure(3)
Plot_Raster(random_raster_matrix_spont)
figure(4)
Plot_Raster(random_raster_matrix_trials)
%%
figure(1)
% subplot(9,1,1)
% plot( habit_day1_cell1_MCMCspikes_binarized)
% subplot(9,1,2)
% plot( habit_day1_cell2_MCMCspikes_binarized)
% subplot(9,1,3)
% plot( habit_day1_cell3_MCMCspikes_binarized)
subplot(9,1,4)
plot( habit_day1_cell4_MCMCspikes_binarized)
subplot(9,1,5)
plot( habit_day1_cell5_MCMCspikes_binarized)
subplot(9,1,6)
plot( habit_day1_cell6_MCMCspikes_binarized)
subplot(9,1,7)
plot( habit_day1_cell7_MCMCspikes_binarized)
subplot(9,1,8)
plot( habit_day1_cell8_MCMCspikes_binarized)
subplot(9,1,8)
plot( habit_day1_cell9_MCMCspikes_binarized)
%%
