sca()
clear all
close all
desktop_testing=1;

acquisition_name='SessionC_timecount_fliptimes';
stim_dir=fullfile(fileparts(pwd),'AllenStimuli', 'Smalles');

load(fullfile(stim_dir,'locally_sparse_noise_full_135.mat'))
load(fullfile(stim_dir,'natural_movie_one.mat'))
load(fullfile(stim_dir,'natural_movie_two.mat'))
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
ops.isicolor=135;      %appx middle                       % Shade of gray on screen during isi (1 if black, 255/2 if gray)
isi_color = [ops.isicolor ops.isicolor ops.isicolor];
waitframes=1;
%% initialize
PsychDefaultSetup(1);
Screen('Preference', 'SkipSyncTests', 0);
AssertOpenGL; % Make sure this is running on OpenGL Psychtoolbox:
screenid = max(Screen('Screens')); % Choose screen with maximum id - the secondary display on a dual-display setup for display
if desktop_testing==1
    screenid=1;
end
white = WhiteIndex(screenid);
black = BlackIndex(screenid);
% Do a simply calculation to calculate the luminance value for grey. This
% will be half the luminace value for white
grey = white / 2;
grey=135;

[win, rect] = Screen('OpenWindow',screenid, isi_color); % rect is the coordinates of the screen
ops.flipInterval = Screen('GetFlipInterval', win);
resolution=Screen('Resolution', screenid);
reswidth=resolution.width;
resheight=resolution.height;
topPriorityLevel = MaxPriority(win);


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


%% loading the movies twice the size without waitframes
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
    maxvol=10;
    greyvol=6
    movievol=4

else
    session=daq.createSession('ni');
    session.addAnalogOutputChannel('Dev1','ao1','Voltage');
%     session.IsContinuous = true;
%     session.Rate = 10000;
    session.outputSingleScan(5);
    session.outputSingleScan(0);
    maxvol=5;
    greyvol=2
    movievol=2.5
end

%%
full_info=cell(numel(ops.paradigm_sequence)+1,5);
full_info{1,1}='Paradigms';
full_info{1,2}='StartParadigmTime';
full_info{1,3}='EndParadigmTime';
full_info{1,4}='ParadigmDuration';
full_info{1,5}='Trials';
for parad_num=1:numel(ops.paradigm_sequence)
    full_info{1+parad_num,1}=ops.paradigm_sequence(parad_num);
    if contains(ops.paradigm_sequence{parad_num}, 'Movie')
        full_info{1+parad_num,5}=cell(ops.paradigm_trial_num(parad_num)+1,5);
        full_info{1+parad_num,5}(1,:)={'Trial', 'TrialStart', 'TrialEnd', 'TrialTime','Frames'};
        for trial=1:ops.paradigm_trial_num(parad_num)
            full_info{1+parad_num,5}{trial+1,1}=trial;
            frame_number=ops.paradigm_frame_number(parad_num);
            full_info{1+parad_num,5}{trial+1,5}=cell(frame_number+1,3);
            full_info{1+parad_num,5}{trial+1,5}(1,:)={'Frame', 'FrameStart', 'FrameTime',};
            for frame=1:frame_number
                full_info{1+parad_num,5}{trial+1,5}{frame+1,1}=frame;
            end           
        end
    elseif contains(ops.paradigm_sequence{parad_num}, 'Noise')
        full_info{1+parad_num,5}=cell(ops.paradigm_trial_num(parad_num)+1,5);
        full_info{1+parad_num,5}(1,:)={'Trial', 'TrialStart', 'TrialEnd', 'TrialTime','NoiseIndex'};
            if contains(ops.paradigm_sequence{parad_num}, 'Noise1')
                noiseindexes=noiseindexes1;
            elseif contains(ops.paradigm_sequence{parad_num}, 'Noise2')
                noiseindexes=noiseindexes2;
            elseif contains(ops.paradigm_sequence{parad_num}, 'Noise3')
                noiseindexes=noiseindexes3;
            end
            for k = 1:length(noiseindexes)
                full_info{1+parad_num,5}{k+1,5} = noiseindexes(k);
            end    
            for trial=1:ops.paradigm_trial_num(parad_num)
                full_info{1+parad_num,5}{trial+1,1}=trial;
            end
    else
        full_info{1+parad_num,5}=cell(ops.paradigm_trial_num(parad_num)+1,4);
        full_info{1+parad_num,5}(1,:)={'Trial', 'TrialStart', 'TrialEnd', 'TrialTime'};
        for trial=1:ops.paradigm_trial_num(parad_num)
                full_info{1+parad_num,5}{trial+1,1}=trial;
        end
    end   
end
diodeBox=30;
clear noiseindexes noiseindexes1 noiseindexes2 noiseindexes3 splitrandperm
mydlg = warndlg('Ok to strat VisStim.', 'A Warning Dialog');
waitfor(mydlg);
disp('VisStimStarted.');startexp=GetSecs();

%% FIRST GRATING TIME 10min
for parad_num = 3:numel(ops.paradigm_sequence)
    full_info{1+parad_num,2}=GetSecs();
    if exist ('session', 'var')
        session.outputSingleScan(maxvol); 
        pause(0.2)
        session.outputSingleScan(maxvol);     
    end
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
    if exist ('session', 'var')
        session.outputSingleScan(0);
        session.outputSingleScan(0);
    end
    if noise
         for trl=1:ops.paradigm_trial_num(parad_num) 
%             noise_volt=(maxvol-1)*full_info{1+parad_num,5}{1+trl,5}/9000;
            if rem(trl, 2) == 0
                trial_volt=4
            else
                trial_volt=2
            end
            if exist ('session', 'var')
%                 session.outputSingleScan(noise_volt);
                session.outputSingleScan(trial_volt);
            end
            if trl==1
                now=GetSecs();
                Screen('DrawTexture', win, texture(1,trl));
                [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);
                full_info{1+parad_num,5}{1+trl,2}=[VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
                Screen('DrawTexture', win, texture(1,trl+1));
                while (now- VBLTimestamp)<ops.paradigm_stim_time(parad_num)
                        now=GetSecs();
                end               
                [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);
%                 full_info{1+parad_num,5}{1+trl,3}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos];
                full_info{1+parad_num,5}{1+trl+1,2}=[VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
            elseif trl>1 && trl<ops.paradigm_trial_num(parad_num)
                Screen('DrawTexture', win, texture(1,trl+1));
                while (now-VBLTimestamp)<ops.paradigm_stim_time(parad_num)
                    now=GetSecs();
                end
                [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);
%                 full_info{1+parad_num,5}{1+trl,3}=[VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos];    
                full_info{1+parad_num,5}{1+trl+1,2}=[VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
            else
                Screen('FillRect', win, isi_color, rect);
                while (now-VBLTimestamp)<ops.paradigm_stim_time(parad_num)
                    now=GetSecs();
                end 
                [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);
                full_info{1+parad_num,5}{1+trl,3}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()]; 
            end         
         end     
    elseif movi 
          for trl=1:ops.paradigm_trial_num(parad_num)
%                 frame_volt=9*rem(ct,frame_number)/frame_number;
                trial_volt=movievol*trl/ops.paradigm_trial_num(parad_num);
                Screen('DrawTexture', win,texture(1,1));
                [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);
                ct=1
                if exist ('session', 'var')
                    session.outputSingleScan(trial_volt);
                end
                full_info{1+parad_num,5}{1+trl,2}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
                while (now- full_info{1+parad_num,5}{1+trl,2}(1))<ops.paradigm_stim_time(parad_num)-ops.paradigm_stim_time(parad_num)/frame_number
                        now=GetSecs();
                        Screen('DrawTexture', win,texture(1,rem(ct,frame_number)+1));
                        [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);
                        ct=ct+1;
                        if rem(ct, 2) == 0
                            frame_volt=0.7
                        else
                            frame_volt=0.3
                        end
                        if exist ('session', 'var')
                           session.outputSingleScan(frame_volt);
                        end
                        full_info{1+parad_num,5}{1+trl,5}{1+ct+1,2}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
                end
                Screen('FillRect', win, isi_color, rect);
                [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);
                full_info{1+parad_num,5}{1+trl,3}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
          end
    else
        for trl=1:ops.paradigm_trial_num(parad_num)
            if exist ('session', 'var')
                    session.outputSingleScan(greyvol);
            end
            now=GetSecs();      
            full_info{1+parad_num,5}{1+trl,2}= full_info{1+parad_num-1,5}{end,3};
            while (now- full_info{1+parad_num,5}{1+trl,2}(1))<ops.paradigm_stim_time(parad_num)-1/60
                      now=GetSecs();
            end 
            [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win,1,1); % last flip
            full_info{1+parad_num,5}{1+trl,3}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
        end
    end
    if exist ('session', 'var')
        session.outputSingleScan(0);
        session.outputSingleScan(0);
    end
    full_info{1+parad_num,3}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];      
 end

Screen('Close');  
sca();
%%
for parad_num=1:numel(ops.paradigm_sequence)
    full_info{1+parad_num,4}=full_info{1+parad_num,3}-full_info{1+parad_num,2};
    for trial=1:ops.paradigm_trial_num(parad_num)
      full_info{1+parad_num,5}{trial+1,4}=full_info{1+parad_num,5}{trial+1,3}-full_info{1+parad_num,5}{trial+1,2};
      if contains(ops.paradigm_sequence{parad_num}, 'Movie')
        full_info{1+parad_num,5}{trial+1,4}=full_info{1+parad_num,5}{trial+1,3}-full_info{1+parad_num,5}{trial+1,2};
        full_info{1+parad_num,5}{trial+1,5}{2,3}=full_info{1+parad_num,5}{trial+1,5}{3,2}-full_info{1+parad_num,5}{trial+1,5}{2,2};
        for frame=3:length( full_info{1+parad_num,5}{trial+1,5}(:,2))          
            if frame==length( full_info{1+parad_num,5}{trial+1,5}(:,2))
                full_info{1+parad_num,5}{trial+1,5}{frame,1}=frame-1; 
                full_info{1+parad_num,5}{trial+1,5}{frame,3}= full_info{1+parad_num,5}{trial+1,3}-full_info{1+parad_num,5}{trial+1,5}{frame,2};  
            else
                 full_info{1+parad_num,5}{trial+1,5}{frame,3}=full_info{1+parad_num,5}{trial+1,5}{frame+1,2}-full_info{1+parad_num,5}{trial+1,5}{frame,2};
                 full_info{1+parad_num,5}{trial+1,5}{frame,1}=frame-1; 
            end
        end
       end     
    end
end 

pwd2 = fileparts(which('EmulateAllenSessionC_v3.m')); %mfilename
pwd2 = pwd; %mfilename

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
