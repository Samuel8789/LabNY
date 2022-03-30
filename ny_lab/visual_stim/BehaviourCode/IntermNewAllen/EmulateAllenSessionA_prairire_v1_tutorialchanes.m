sca()
clear all
close all

desktop_testing=1;

acquisition_name='SPJL';

load('drifiting_gratings_full.mat')
driftinggratings=all_warped_drifting_gratings_full;
clear all_warped_drifting_gratings_full 

load('natural_movie_one.mat')
naturalmovie1big=natural_movie_one_all_warped_frames;
clear natural_movie_one_all_warped_frames 

load('natural_movie_three_full.mat')
naturalmovie3big=natural_movie_three_all_warped_frames_full;
clear natural_movie_three_all_warped_frames_full 


%% parameters
% ------ Paradigm sequence ------
ops.paradigm_sequence = {'Drifting1', 'Intergrey','Movie3', 'Intergrey','Movie1','Intergrey','Drifting2' , 'Spont','Movie3','Intergrey','Drifting3'};  
ops.paradigm_trial_num =    [ 200,   1,     5,    1,    10,    1,   200,   1,     5,   1,  200];   
ops.paradigm_stim_time=     [   2,  30,   120,   30,    30,   30,     2, 300,   120,  30,    2];
ops.paradigm_isi_time=      [   1,   0,     0,    0,     0,    0,     1,   0,     0,   0,    1];
ops.paradigm_frame_number=  [   1,   1,  3600,    1,   900,    1,     1,   1,  3600,   1,    1];
ops.isicolor=127;      %appx middle                       % Shade of gray on screen during isi (1 if black, 255/2 if gray)
isi_color = [0 0 ops.isicolor];
repetitions=15;
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
grey=135

[win, rect] = Screen('OpenWindow',screenid, isi_color ); % rect is the coordinates of the screen
ops.flipInterval = Screen('GetFlipInterval', win);
resolution=Screen('Resolution', screenid);
reswidth=resolution.width;
resheight=resolution.height;
topPriorityLevel = MaxPriority(win);


%% 

totalgratings=size(driftinggratings,3)*size(driftinggratings,4);
seepfreq=20;
periodtimes=[10 10 10];
periodreps=periodtimes*20;
numbersweeps=ceil(periodreps/seepfreq);
total_trials = totalgratings*repetitions;
totalstim=totalgratings;
tex = zeros(size(driftinggratings,3),size(driftinggratings,4),size(driftinggratings,5));
 for i=1:size(driftinggratings,3)
     for j=1:size(driftinggratings,4)
         for k=1:size(driftinggratings,5)    
            tex(i,j,k)=Screen('MakeTexture', win,0.8*driftinggratings(:,:,i,j,k)); 
         end
     end
 end
driftingtex=reshape(tex,[size(driftinggratings,3)*size(driftinggratings,4),size(driftinggratings,5)]);
restoredtex=squeeze(reshape(driftingtex,[1,size(driftinggratings,3), size(driftinggratings,4), size(driftinggratings,5)]));
all_grating_indexes=1:40;
drifitng_grating_index_array=squeeze(reshape(all_grating_indexes,[1,size(driftinggratings,3), size(driftinggratings,4)]));
trial_array=zeros(totalgratings,repetitions);
for i=1:repetitions
    trial_array(:,i)=randperm(totalgratings);
end
jitter=zeros(3,max(numbersweeps));
for i=1:3
    jitter(i,:)=randi(3,1,max(numbersweeps));
end
jitter(1:2,78:end)=NaN;
jitter(jitter==1)=-1;
jitter(jitter==2)=0;
jitter(jitter==3)=1;
sweepperiods=jitter+seepfreq;
sweepindexes=cumsum(sweepperiods,2);
sampled_grating_indexes=trial_array(randperm(total_trials, total_trials));
sampled_grating_indexes_parts = reshape(sampled_grating_indexes, 3, []);

for i=1:3
    sweepindexes(i,sweepindexes(i,1:numbersweeps(i))>periodreps(i))=NaN;
    sampled_grating_indexes_parts(i,sweepindexes(i,(~isnan(sweepindexes(i,:)))))=0;
end
%% loading the movies
[~,~,frame_number_movie_1]=size(naturalmovie1big);
texmov1=zeros(1,frame_number_movie_1);
for i=1:frame_number_movie_1
    texmov1(i)=Screen('MakeTexture', win, naturalmovie1big(:,:,i)); 
end

[~,~,frame_number_movie_3]=size(naturalmovie3big);
texmov3=zeros(1,frame_number_movie_3);
for i=1:frame_number_movie_3
    texmov3(i)=Screen('MakeTexture', win, naturalmovie3big(:,:,i)); 
end

clear naturalmovie1big naturalmovie2big
ops.paradigm_frame_number=  [   1,   1,  frame_number_movie_3,    1,   frame_number_movie_1,    1,     1,   1,  frame_number_movie_3,   1,    1];
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
    elseif contains(ops.paradigm_sequence{parad_num}, 'Drifting')
        full_info{parad_num+1,5}=cell(ops.paradigm_trial_num(parad_num)+1,7);
        full_info{parad_num+1,5}(1,:)={'Trial', 'TrialStart','StimStart' 'TrialEnd', 'TrialTime','DrifitingIndex', 'Phases'};
            if contains(ops.paradigm_sequence{parad_num}, 'Drifting1')
                gratingindexes=sampled_grating_indexes_parts(1,(~isnan(sampled_grating_indexes_parts(1,:))));
            elseif contains(ops.paradigm_sequence{parad_num}, 'Drifting2')
                gratingindexes=sampled_grating_indexes_parts(1,(~isnan(sampled_grating_indexes_parts(1,:))));
            elseif contains(ops.paradigm_sequence{parad_num}, 'Drifting3')
                gratingindexes=sampled_grating_indexes_parts(1,(~isnan(sampled_grating_indexes_parts(1,:))));
            end
            for k = 1:length(gratingindexes)
                full_info{parad_num+1,5}{k+1,6} = gratingindexes(k);
            end    
            for trial=1:ops.paradigm_trial_num(parad_num)
                full_info{parad_num+1,5}{trial+1,1}=trial;
                phase_number=120;
                full_info{parad_num+1,5}{trial+1,7}=cell(phase_number+1,4);
                full_info{parad_num+1,5}{trial+1,7}(1,:)={'Phase', 'PahseStart', 'PhaseTime','PhaseValue'};
                for phase=1:phase_number
                    full_info{parad_num+1,5}{trial+1,7}{phase+1,1}=phase;
                end     
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
waitforbuttonpress; 

%% FIRST GRATING TIME 10min
Priority(topPriorityLevel);  
for parad_num = 1:numel(ops.paradigm_sequence)
    full_info{parad_num+1,2}=Screen('Flip', win); % paradigm start
%     session.outputSingleScan(10); 
%     pause(0.010)
%     session.outputSingleScan(10);  
    grating=0;
    movi=0;
 % check what paradigm
    if strcmpi(ops.paradigm_sequence{parad_num}, 'Drifting1')
        texindexes=sampled_grating_indexes_parts(1,(~isnan(sampled_grating_indexes_parts(1,:))));
        texture=driftingtex   ;    
        grating=1;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Drifting2')
        texindexes=sampled_grating_indexes_parts(2,(~isnan(sampled_grating_indexes_parts(3,:))));
        texture=driftingtex;
        grating=1;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Drifting3')
        texindexes=sampled_grating_indexes_parts(3,(~isnan(sampled_grating_indexes_parts(3,:))));
        texture=driftingtex;
        grating=1;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Movie1')
        texture=texmov1;
        movi=1;  
        frame_number=frame_number_movie_1;
        totalframenumer=ops.paradigm_trial_num(parad_num) *frame_number;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Movie3')
        texture=texmov3;
        movi=1;
        frame_number=frame_number_movie_3; 
        totalframenumer=ops.paradigm_trial_num(parad_num) *frame_number;
    end   
    numSecs = ops.paradigm_stim_time(parad_num);
    waitframes=round(numSecs*1/ops.flipInterval);
%     session.outputSingleScan(0); 
%     session.outputSingleScan(0);
    if grating
        waitframes=1;
        waitframesisi=round(ops.paradigm_isi_time(parad_num) *1/ops.flipInterval);
        waitframessweep=round(ops.paradigm_stim_time(parad_num) *1/ops.flipInterval);
        numFrames = round(numSecs / ops.flipInterval);
        for trl=1:ops.paradigm_trial_num(parad_num)
            full_info{parad_num+1,5}{trl+1,2}=Screen('Flip', win);% trial start previous trial end
            session.outputSingleScan(0);
            session.outputSingleScan(0);
            VBLTimestamp=full_info{parad_num+1,5}{trl+1,2}(1);          
            grat_volt=4*texindexes(trl)/totalstim;
            if texindexes(trl)==0
                grat_volt=4.5;
            end
            if texindexes(trl)==0
                waitframes=waitframessweep;
                Screen('DrawTexture', win,texture(texindexes(trl),1), [],[],[],[],[], isi_color));   
            else
                waitframes=1;
                Screen('FillRect', win, isi_color, rect);
            end             
            full_info{parad_num+1,5}{trl+1,3}=Screen('Flip', win, VBLTimestamp + (waitframesisi - 0.2) * ops.flipInterval );     % stim start, phase 1 start
%             session.outputSingleScan(grat_volt);
            VBLTimestamp=full_info{parad_num+1,5}{trl+1,3}(1);
            if texindexes(trl)~=0
                for frame=1:numFrames-1
                   Screen('DrawTexture', win,texture(texindexes(trl),rem(frame,60)+1), [],[],[],[],[], isi_color));
                   full_info{parad_num+1,5}{trl+1,7}{frame+2,2}= Screen('Flip', win, VBLTimestamp+ (waitframes - 0.2) * ops.flipInterval );  % phase start 2:end
                   VBLTimestamp= full_info{parad_num+1,5}{trl+1,7}{frame+2,2}(1);
                end
            else
                full_info{parad_num+1,5}{trl+1,7}{frame+2,2}= Screen('Flip', win, VBLTimestamp+ (waitframes - 0.2) * ops.flipInterval );  % last frame of blank start            
            end
        end 
    elseif movi 
        waitframes=2;
        trl=1;
        trial_volt=4*1/ops.paradigm_trial_num(parad_num);
        Screen('DrawTexture', win,texture(1,1));
        full_info{parad_num+1,5}{trl+1,2}=Screen('Flip', win); % start trial 1 frame 1
%         session.outputSingleScan(trial_volt);
        VBLTimestamp=full_info{parad_num+1,5}{trl+1,2}(1);
        trl=trl+1;
        trial_volt=4*trl/ops.paradigm_trial_num(parad_num);
        for frame=1:totalframenumer-1
             Screen('DrawTexture', win,texture(1,rem(frame,frame_number)+1), [],[],[],[],[], isi_color));
             full_info{parad_num+1,5}{trl+1,5}{frame+1,2}=Screen('Flip', win, VBLTimestamp + (waitframes - 0.2) * ops.flipInterval );   % frame start
             if rem(frame,frame_number)+1==1
%                 session.outputSingleScan(trial_volt);
                trl=trl+1;
                trial_volt=4*trl/ops.paradigm_trial_num(parad_num);
             end
             VBLTimestamp= full_info{parad_num+1,5}{trl+1,5}{frame+1,2}(1);
        end
    else
        full_info{parad_num+1,5}{trl+1,2} = Screen('Flip', win); % start trial 1
%         session.outputSingleScan(6); strat trial grey
        full_info{parad_num+1,5}{trl+1,3} = Screen('Flip', win, full_info{parad_num+1,5}{trl+1,2}(1) + (waitframes - 0.5) * ops.flipInterval ); % start last frame grey    
    end
    Screen('FillRect', win, isi_color, rect);      
    full_info{parad_num+1,3}=Screen('Flip', win);  %paradigm end , last trial last frame end, gery end
%     session.outputSingleScan(0);
%     session.outputSingleScan(0);
    Screen('FillRect', win, isi_color, rect);      
end
Priority(0);
Screen('Close');  
sca();
%%
for parad_num=1:numel(ops.paradigm_sequence)
    full_info{parad_num+1,4}=full_info{parad_num+1,3}(1)-full_info{parad_num+1,2}(1);
%     full_info{parad_num+1,4}=full_info{parad_num+1,3}-full_info{parad_num+1,2};
    for trial=1:ops.paradigm_trial_num(parad_num)        
      if contains(ops.paradigm_sequence{parad_num}, 'Movie')
        full_info{parad_num+1,5}{trial+1,4}=full_info{parad_num+1,5}{trial+1,3}(1)-full_info{parad_num+1,5}{trial+1,2}(1);
        full_info{parad_num+1,5}{trial+1,5}{2,3}=full_info{parad_num+1,5}{trial+1,5}{3,2}(1)-full_info{parad_num+1,5}{trial+1,5}{2,2}(1);
        for frame=3:length( full_info{parad_num+1,5}{trial+1,5}(:,2))           
            if frame==length( full_info{parad_num+1,5}{trial+1,5}(:,2))
                full_info{parad_num+1,5}{trial+1,5}{frame,1}=frame-1; 
                full_info{parad_num+1,5}{trial+1,5}{frame,3}= full_info{parad_num+1,5}{trial+1,3}(1)-full_info{parad_num+1,5}{trial+1,5}{frame,2}(1);  
            else
                 full_info{parad_num+1,5}{trial+1,5}{frame,3}=full_info{parad_num+1,5}{trial+1,5}{frame+1,2}(1)-full_info{parad_num+1,5}{trial+1,5}{frame,2}(1);
                 full_info{parad_num+1,5}{trial+1,5}{frame,1}=frame-1; 
            end
        end
      end     
      if contains(ops.paradigm_sequence{parad_num}, 'Drifting') 
        full_info{parad_num+1,5}{trial+1,5}=full_info{parad_num+1,5}{trial+1,4}(1)-full_info{parad_num+1,5}{trial+1,2}(1);
        full_info{parad_num+1,5}{trial+1,7}{2,3}=full_info{parad_num+1,5}{trial+1,7}{3,2}(1)-full_info{parad_num+1,5}{trial+1,7}{2,2}(1);
        for phase=3:length( full_info{parad_num+1,5}{trial+1,7}(:,2)) 
                if phase==length( full_info{parad_num+1,5}{trial+1,7}(:,2))
                    full_info{parad_num+1,5}{trial+1,7}{phase,3}= full_info{parad_num+1,5}{trial+1,4}(1)-full_info{parad_num+1,5}{trial+1,7}{phase,2}(1);
                    full_info{parad_num+1,5}{trial+1,7}{phase,1}=phase-1;  

                else
                     full_info{parad_num+1,5}{trial+1,7}{phase,3}=full_info{parad_num+1,5}{trial+1,7}{phase+1,2}(1)-full_info{parad_num+1,5}{trial+1,7}{phase,2}(1);
                     full_info{parad_num+1,5}{trial+1,7}{phase,1}=phase-1;    
                end
        end
      end
    end
end
pwd2 = fileparts(which('EmulateAllenSessionC_v3.m')); %mfilename
save_path = pwd2;

temp_time = clock;
file_name = sprintf([acquisition_name, 'AllenSessionA_%d_%d_%d_stim_data_%dh_%dm'],temp_time(1)-2000, temp_time(2), temp_time(3), temp_time(4), temp_time(5));
clear temp_time;

%% save info
fprintf('Saving...\n');
save([save_path,'\', file_name, '.mat'],'ops', 'full_info', 'drifitng_grating_index_array');
fprintf('Done\n');

%% ploting gratings timings
for j=[2,8,12]
    figure
    plot(cell2mat(full_info{j, 5}(2:end,5)));
    ylim([0,3])
    figure
    hold on
    for i=2:size(full_info{j, 5},1)
        plot(cell2mat(full_info{j, 5}{i,7}(2:end,3)));
    end
    ylim([0,0.06])
end

%% ploting movies timings
for j=[4,6,10]
    figure
    plot(cell2mat(full_info{j, 5}(2:end,4)));
    ylim([0,130])
    figure
    hold on
    for i=2:size(full_info{j, 5},1)
        plot(cell2mat(full_info{j, 5}{i,5}(2:end,3)));
    end
    ylim([0,0.08])
end
