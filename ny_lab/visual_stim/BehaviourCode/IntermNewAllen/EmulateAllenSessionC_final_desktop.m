sca()
clear all
desktop_testing=1;

acquisition_name='SPJL finaldesktop';

load('locally_sparse_noise_full.mat')
load('natural_movie_one.mat')
load('natural_movie_two.mat')
fullnoise=locally_sparse_noise_all_warped_frames_full;
clear locally_sparse_noise_all_warped_frames_full 
naturalmovie1big=natural_movie_one_all_warped_frames;
clear natural_movie_one_all_warped_frames 
naturalmovie2big=natural_movie_two_all_warped_frames;
clear natural_movie_two_all_warped_frames 

%% parameters
% ------ Paradigm sequence ------
ops.paradigm_sequence = {'Noise1','Spont','Movie1','Intergrey','Noise2','Intergrey','Movie2', 'Spont','Noise3'};  
ops.paradigm_trial_num =    [2880,   1,    10,  1,  2880,   1,    10,   1,  3120];   
% ops.paradigm_stim_time=     [ 1/4, 300,  1/30, 30,   1/4,  30,  1/30, 300,   1/4];
ops.paradigm_stim_time=     [ 1/4, 300,    30, 30,   1/4,  30,    30, 300,   1/4];

ops.paradigm_isi_time=      [   0,   0,     0,  0,     0,   0,     0,   0,     0];
ops.paradigm_frame_number=  [   1,   1,   900,  1,     1,   1,   900,   1,     1];
ops.isicolor=127;      %appx middle                       % Shade of gray on screen during isi (1 if black, 255/2 if gray)
isi_color = [ops.isicolor ops.isicolor ops.isicolor];

%% initialize
Screen('Preference', 'SkipSyncTests', 0);
AssertOpenGL; % Make sure this is running on OpenGL Psychtoolbox:
screenid = max(Screen('Screens')); % Choose screen with maximum id - the secondary display on a dual-display setup for display
if desktop_testing
    screenid=1;
end
[win, rect] = Screen('OpenWindow',screenid, [ops.isicolor ops.isicolor ops.isicolor]); % rect is the coordinates of the screen
ops.flipInterval = Screen('GetFlipInterval', win);
resolution=Screen('Resolution', screenid);
reswidth=resolution.width;
resheight=resolution.height;

%% sampling the noise matrix

ncol = 8880;
splitrandperm=randperm(size(fullnoise,3));
Ar=fullnoise(:,:,splitrandperm);
clear fullnoise
noise1=Ar(:,:,1:2880);
noise2=Ar(:,:,2881:5760);
noise3=Ar(:,:,5761:8880);
clear Ar
texnoise1=zeros(1,size(noise1,3));
texnoise2=zeros(1,size(noise2,3));
texnoise3=zeros(1,size(noise3,3));

for i=1:size(noise1,3)
    texnoise1(i)=Screen('MakeTexture', win, noise1(:,:,i)); 
end
for i=1:size(noise2,3)
    texnoise2(i)=Screen('MakeTexture', win, noise2(:,:,i)); 
end
for i=1:size(noise3,3)
    texnoise3(i)=Screen('MakeTexture', win, noise3(:,:,i)); 
end
clear noise1 noise2 noise3 

noiseindexes1=splitrandperm(1:2880)';
noiseindexes2=splitrandperm(2881:5760)';
noiseindexes3=splitrandperm(5761:8880)';


%% loading the movies
[~,~,frame_number_movie_1]=size(naturalmovie1big);
texmov1=zeros(1,frame_number_movie_1);
for i=1:frame_number_movie_1
    texmov1(i)=Screen('MakeTexture', win, naturalmovie1big(:,:,i)); 
end
texmov1=reshape(repmat(texmov1,2,1),size(texmov1,1),2*size(texmov1,2));

[~,~,frame_number_movie_2]=size(naturalmovie2big);
texmov2=zeros(1,frame_number_movie_2);
for i=1:frame_number_movie_2
    texmov2(i)=Screen('MakeTexture', win, naturalmovie2big(:,:,i)); 
end
texmov2=reshape(repmat(texmov2,2,1),size(texmov2,1),2*size(texmov2,2));

clear naturalmovie1big naturalmovie2big
ops.paradigm_frame_number=  [   1,   1,   frame_number_movie_1,  1,     1,   1,   frame_number_movie_2,   1,     1];

%% VoltageSignals
if ~desktop_testing
    session=daq.createSession('ni');
    session.addAnalogOutputChannel('Dev1','ao0','Voltage');
    session.IsContinuous = true;
    session.Rate = 10000;
    session.outputSingleScan(5);
    session.outputSingleScan(0);
end

%%
full_info=cell(numel(ops.paradigm_sequence)+1,5);
full_info{1,1}='Paradigms';
full_info{1,2}='StartParadigmTime';
full_info{1,3}='EndParadigmTime';
full_info{1,4}='ParadigmDuration';
full_info{1,5}='Trials';
for parad_num=1:numel(ops.paradigm_sequence)
    full_info{parad_num+1,1}=ops.paradigm_sequence(parad_num);
    if contains(ops.paradigm_sequence{parad_num}, 'Movie')
        full_info{parad_num+1,5}=cell(ops.paradigm_trial_num(parad_num)+1,5);
        full_info{parad_num+1,5}(1,:)={'Trial', 'TrialStart', 'TrialEnd', 'TrialTime','Frames'};
        for trial=1:ops.paradigm_trial_num(parad_num)
            full_info{parad_num+1,5}{trial+1,1}=trial;
            frame_number=ops.paradigm_frame_number(parad_num);
            full_info{parad_num+1,5}{trial+1,5}=cell(frame_number+1,3);
            full_info{parad_num+1,5}{trial+1,5}(1,:)={'Frame', 'FrameStart', 'FrameTime',};
            for frame=1:frame_number
                full_info{parad_num+1,5}{trial+1,5}{frame+1,1}=frame;
            end           
        end
    elseif contains(ops.paradigm_sequence{parad_num}, 'Noise')
        full_info{parad_num+1,5}=cell(ops.paradigm_trial_num(parad_num)+1,5);
        full_info{parad_num+1,5}(1,:)={'Trial', 'TrialStart', 'TrialEnd', 'TrialTime','NoiseIndex'};
            if contains(ops.paradigm_sequence{parad_num}, 'Noise1')
                noiseindexes=noiseindexes1;
            elseif contains(ops.paradigm_sequence{parad_num}, 'Noise2')
                noiseindexes=noiseindexes2;
            elseif contains(ops.paradigm_sequence{parad_num}, 'Noise3')
                noiseindexes=noiseindexes3;
            end
            for k = 1:length(noiseindexes);
                full_info{parad_num+1,5}{k+1,5} = noiseindexes(k);
            end    
            for trial=1:ops.paradigm_trial_num(parad_num)
                full_info{parad_num+1,5}{trial+1,1}=trial;
            end
    else
        full_info{parad_num+1,5}=cell(ops.paradigm_trial_num(parad_num)+1,4);
        full_info{parad_num+1,5}(1,:)={'Trial', 'TrialStart', 'TrialEnd', 'TrialTime'};
        for trial=1:ops.paradigm_trial_num(parad_num)
                full_info{parad_num+1,5}{trial+1,1}=trial;
        end
    end   
end
diodeBox=30;
clear noiseindexes noiseindexes1 noiseindexes2 noiseindexes3 splitrandperm
waitforbuttonpress; 
startexp=GetSecs();

%% FIRST GRATING TIME 10min
for parad_num = 3:numel(ops.paradigm_sequence)
    full_info{parad_num+1,2}=GetSecs();
%     session.outputSingleScan(10); 
%     pause(0.2)
%     session.outputSingleScan(10);  
    noise=0;
    movi=0;
 % check what paradigm
    if strcmpi(ops.paradigm_sequence{parad_num}, 'Noise1')
        texture=texnoise1   ; 
        noise=1;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Noise2')
        texture=texnoise2;
        noise=1;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Noise3')
        texture=texnoise3;
        noise=1;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Movie1')
        texture=texmov1;
        movi=1;
        frame_number=frame_number_movie_1;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Movie2')
        texture=texmov2;
        movi=1;
        frame_number=frame_number_movie_2;
    end    
%     session.outputSingleScan(0); 
    if noise
         for trl=1:ops.paradigm_trial_num(parad_num) 
            full_info{parad_num+1,5}{trl+1,2}=GetSecs();
            noise_volt=9*full_info{parad_num+1,5}{trl+1,5}/9000;
            start_stim = GetSecs();
%             session.outputSingleScan(noise_volt);
            if trl==1
                now=GetSecs();
                Screen('DrawTexture', win, texture(1,trl));
                Screen('Flip',win);
                Screen('DrawTexture', win, texture(1,trl+1));
                while (now-start_stim)<ops.paradigm_stim_time(parad_num)
                        now=GetSecs();
                end
                full_info{parad_num+1,5}{trl+1,3}=GetSecs();
                Screen('Flip',win);
            elseif trl>1 & trl<ops.paradigm_trial_num(parad_num)
                Screen('DrawTexture', win, texture(1,trl+1));
                while (now-start_stim)<ops.paradigm_stim_time(parad_num)
                    now=GetSecs();
                end
                full_info{parad_num+1,5}{trl+1,3}=GetSecs();                
                Screen('Flip',win);
            else
                Screen('FillRect', win, isi_color, rect);
                while (now-start_stim)<ops.paradigm_stim_time(parad_num)
                    now=GetSecs();
                end 
                full_info{parad_num+1,5}{trl+1,3}=GetSecs(); 
                Screen('Flip',win);
            end         
         end     
    elseif movi 
          for trl=1:ops.paradigm_trial_num(parad_num)
                ct=0;
                full_info{parad_num+1,5}{trl+1,2}=GetSecs();          
                frame_volt=9*rem(ct,frame_number)/frame_number;
                start_stim = GetSecs();
        %             session.outputSingleScan(grat_volt);
                now=GetSecs();                
                while (now-start_stim)<ops.paradigm_stim_time(parad_num)
                        full_info{parad_num+1,5}{trl+1,5}{ct+2,2}=GetSecs();
                        now=GetSecs();
                        Screen('DrawTexture', win,texture(1,rem(ct,frame_number)+1));
                        Screen('Flip',win);
                        ct=ct+1;
                end
                full_info{parad_num+1,5}{trl+1,3}=GetSecs();
          end
    else
        for trl=1:ops.paradigm_trial_num(parad_num)
%             session.outputSingleScan(6);
            full_info{parad_num+1,5}{trl+1,2}=GetSecs();
            start_stim = GetSecs();
            now=GetSecs();
            Screen('FillRect', win, isi_color, rect);
            Screen('Flip',win, 1,1);
            while (now-start_stim)<ops.paradigm_stim_time(parad_num)
                      now=GetSecs();
            end 
            full_info{parad_num+1,5}{trl+1,3}=GetSecs();
        end
    end
%     session.outputSingleScan(0);
    full_info{parad_num+1,3}=GetSecs();      
 end

Screen('Close');  
sca();
%%
for parad_num=1:numel(ops.paradigm_sequence)
    full_info{parad_num+1,4}=full_info{parad_num+1,3}-full_info{parad_num+1,2};
    for trial=1:ops.paradigm_trial_num(parad_num)
      full_info{parad_num+1,5}{trial+1,4}=full_info{parad_num+1,5}{trial+1,3}-full_info{parad_num+1,5}{trial+1,2};
      if contains(ops.paradigm_sequence{parad_num}, 'Movie')
        full_info{parad_num+1,5}{trial+1,4}=full_info{parad_num+1,5}{trial+1,3}-full_info{parad_num+1,5}{trial+1,2};
        full_info{parad_num+1,5}{trial+1,5}{2,3}=full_info{parad_num+1,5}{trial+1,5}{3,2}-full_info{parad_num+1,5}{trial+1,5}{2,2};
        for frame=3:length( full_info{parad_num+1,5}{trial+1,5}(:,2))          
            if frame==length( full_info{parad_num+1,5}{trial+1,5}(:,2))
                full_info{parad_num+1,5}{trial+1,5}{frame,1}=frame-1; 
                full_info{parad_num+1,5}{trial+1,5}{frame,3}= full_info{parad_num+1,5}{trial+1,3}-full_info{parad_num+1,5}{trial+1,5}{frame,2};  
            else
                 full_info{parad_num+1,5}{trial+1,5}{frame,3}=full_info{parad_num+1,5}{trial+1,5}{frame+1,2}-full_info{parad_num+1,5}{trial+1,5}{frame,2};
                 full_info{parad_num+1,5}{trial+1,5}{frame,1}=frame-1; 
            end
        end
       end     
    end
end 

pwd2 = fileparts(which('EmulateAllenSessionC_v3.m')); %mfilename
save_path = pwd2;
temp_time = clock;
file_name = sprintf([acquisition_name, 'AllenSessionC_%d_%d_%d_stim_data_%dh_%dm'],temp_time(1)-2000, temp_time(2), temp_time(3), temp_time(4), temp_time(5));
clear temp_time;

%% save info
fprintf('Saving...\n');
save([save_path,'\', file_name, '.mat'],'ops', 'full_info');
fprintf('Done\n');
%% ploting noise timings
for j=[2,6,10]
   figure
    plot(cell2mat(full_info{j, 5}(2:end,4)));
    ylim([0.249,0.252]);
    mean(cell2mat(full_info{j, 5}(2:end,4)));
    std(cell2mat(full_info{j, 5}(2:end,4)));  
end

%% ploting movies timings
for j=[4,8]
    figure
    plot(cell2mat(full_info{j, 5}(2:end,4)));
    ylim([29.9,30.1])
    mean(cell2mat(full_info{j, 5}(2:end,4)))
    std(cell2mat(full_info{j, 5}(2:end,4)))
    figure
    hold on
    for i=2:size(full_info{j, 5},1)
        plot(cell2mat(full_info{j, 5}{i,5}(2:end,3)));       
    end
    ylim([0.000,0.04])
end
