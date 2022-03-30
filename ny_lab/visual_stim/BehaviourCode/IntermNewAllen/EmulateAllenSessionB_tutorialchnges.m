sca()
clear all
close all

desktop_testing=1;

acquisition_name='SPJL';

load('natural_movie_one.mat');
naturalmovie1big=natural_movie_one_all_warped_frames;
clear natural_movie_one_all_warped_frames 
load('natural_scenes.mat');
scenes=natural_scenes_all_warped_frames;
clear natural_scenes_all_warped_frames 
load('static_gratings.mat')
gratings=all_warped_static_gratings;
clear all_warped_static_gratings 


% scenes_normalized=zeros(size(scenes));
% for i =1:118
%     scenes_normalized(:,:,i)=imadjust(scenes(:,:,i),stretchlim(scenes(:,:,i)),[]);
% end
%% parameters
% ------ Paradigm sequence ------
ops.paradigm_sequence = {'Static1','Intergrey','Images1','Spont','Images2','Intergrey','Static2', 'Intergrey','Movie1','Intergrey', 'Images3','Intergrey','Static3'};  
ops.paradigm_trial_num =    [1920,   1,  1920,   1,  1920,   1,  1920,   1,    10,    1,  1920,   1,  1920];   
% ops.paradigm_stim_time=     [ 1/4,  30,   1/4, 300,   1/4,  30,   1/4,  30,  1/30,   30,   1/4,  30,   1/4];
ops.paradigm_stim_time=     [ 1/4,  30,   1/4, 300,   1/4,  30,   1/4,  30,    30,   30,   1/4,  30,   1/4];

ops.paradigm_isi_time=      [   0,   0,     0,   0,     0,   0,     0,   0,     0,    0,     0,   0,     0];
ops.paradigm_frame_number=  [   1,   1,     1,   1,     1,   1,     1,   1,   900,    1,     1,   1,     1];
ops.isicolor=127;      %appx middle                       % Shade of gray on screen during  isi (1 if black, 255/2 if gray)
isi_color = [0 0 ops.isicolor];
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
grey=135

[win, rect] = Screen('OpenWindow',screenid, isi_color ); % rect is the coordinates of the screen
ops.flipInterval = Screen('GetFlipInterval', win);
resolution=Screen('Resolution', screenid);
reswidth=resolution.width;
resheight=resolution.height;
topPriorityLevel = MaxPriority(win);


%% preparing gratings

tex = zeros(size(gratings,3),size(gratings,4),size(gratings,5));
 for i=1:size(gratings,3)
     for j=1:size(gratings,4)
         for k=1:size(gratings,5)    
            tex(i,j,k)=Screen('MakeTexture', win, 0.8*gratings(:,:,i,j,k)); 
         end
     end
 end
texgratings=reshape(tex,[1,size(gratings,3)*size(gratings,4)*size(gratings,5)]);
restoredtex=squeeze(reshape(texgratings,[1,size(gratings,3), size(gratings,4), size(gratings,5)]));

all_grating_indexes=1:120;
grating_index_array=squeeze(reshape(all_grating_indexes,[1,size(gratings,3), size(gratings,4), size(gratings,5)]));

totalpresentations=(8+8+9)*60*1000/250;
totalgratings=size(gratings,3)*size(gratings,4)*size(gratings,5);


full_texture_indexes = repmat(1:length(texgratings),1,floor(totalpresentations/totalgratings));
splitrandperm=randperm(size(full_texture_indexes,2));
permuted_texture_indexes=full_texture_indexes(:,splitrandperm);


seepfreq=25;
totalgratings=size(gratings,3)*size(gratings,4)*size(gratings,5);
periodtimes=[8 8 9];
periodreps=periodtimes*60*1000/250;
numbersweeps=ceil(periodreps/seepfreq);

jitter=zeros(3,max(numbersweeps));
for i=1:3
    jitter(i,:)=randi(3,1,max(numbersweeps));
end
jitter(1:2,numbersweeps(1)+1:end)=NaN;
jitter(jitter==1)=-1;
jitter(jitter==2)=0;
jitter(jitter==3)=1;
sweepperiods=jitter+seepfreq;
sweepindexes=cumsum(sweepperiods,2);

sampled_grating_indexes_parts=zeros(3,max(periodreps));
ct=0;
for i=1:3
    sampled_grating_indexes_parts(i,1:periodreps(i))=permuted_texture_indexes(1+ct:ct+periodreps(i));
    ct=ct+periodreps(i);
end
sampled_grating_indexes_parts(1:2,min(periodreps)+1:end)=NaN;

for i=1:3
    sweepindexes(i,sweepindexes(i,1:numbersweeps(i))>periodreps(i))=NaN;
    sampled_grating_indexes_parts(i,sweepindexes(i,(~isnan(sweepindexes(i,:)))))=0;
end
% clear tex  sampled_grating_indexes

%% preparing natural scens

tex = zeros(1,size(scenes,3));
 for i=1:size(scenes,3)
     tex(i)=Screen('MakeTexture', win, scenes(:,:,i)); 
 end
texscenes=tex;
seepfreq=100;
totalscenes=size(scenes,3);
totalpresentations=(8+8+9)*60*1000/250;

full_texture_indexes = repmat(1:length(texscenes),1,ceil(totalpresentations/totalscenes));
splitrandperm=randperm(size(full_texture_indexes,2));
permuted_texture_indexes=full_texture_indexes(:,splitrandperm);

periodtimes=[8 8 9];
periodreps=periodtimes*60*1000/250;
numbersweeps=ceil(periodreps/seepfreq);

jitter=zeros(3,max(numbersweeps));
for i=1:3
    jitter(i,:)=randi(3,1,max(numbersweeps));
end
jitter(1:2,numbersweeps(1)+1:end)=NaN;
jitter(jitter==1)=-1;
jitter(jitter==2)=0;
jitter(jitter==3)=1;
sweepperiods=jitter+seepfreq;
sweepindexes=cumsum(sweepperiods,2);

sampled_scene_indexes_parts=zeros(3,max(periodreps));

ct=0;
for i=1:3
    sampled_scene_indexes_parts(i,1:periodreps(i))=permuted_texture_indexes(1+ct:ct+periodreps(i));
    ct=ct+periodreps(i);
end
sampled_scene_indexes_parts(1:2,min(periodreps)+1:end)=NaN;

for i=1:3
    sweepindexes(i,sweepindexes(i,1:numbersweeps(i))>periodreps(i))=NaN;
    sampled_scene_indexes_parts(i,sweepindexes(i,(~isnan(sweepindexes(i,:)))))=0;
end

clear tex  

%% loading the movies
[~,~,frame_number_movie_1]=size(naturalmovie1big);
texmov1=zeros(1,frame_number_movie_1);
for i=1:frame_number_movie_1
    texmov1(i)=Screen('MakeTexture', win, naturalmovie1big(:,:,i)); 
end

clear naturalmovie1big 
ops.paradigm_frame_number=  [   1,   1,     1,   1,     1,   1,     1,   1,   frame_number_movie_1,    1,     1,   1,     1];

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
    elseif contains(ops.paradigm_sequence{parad_num}, 'Image')
        full_info{parad_num+1,5}=cell(ops.paradigm_trial_num(parad_num)+1,5);
        full_info{parad_num+1,5}(1,:)={'Trial', 'TrialStart', 'TrialEnd', 'TrialTime','SceneIndex'};
        if contains(ops.paradigm_sequence{parad_num}, 'Images1')
            sceneindexes=sampled_scene_indexes_parts(1,(~isnan(sampled_scene_indexes_parts(1,:))));
        elseif contains(ops.paradigm_sequence{parad_num}, 'Images2')
            sceneindexes=sampled_scene_indexes_parts(2,(~isnan(sampled_scene_indexes_parts(2,:))));
        elseif contains(ops.paradigm_sequence{parad_num}, 'Images3')
            sceneindexes=sampled_scene_indexes_parts(3,(~isnan(sampled_scene_indexes_parts(3,:))));
        end
        for k = 1:length(sceneindexes)
            full_info{parad_num+1,5}{k+1,5} = sceneindexes(k);
        end    
        for trial=1:ops.paradigm_trial_num(parad_num)
            full_info{parad_num+1,5}{trial+1,1}=trial;
        end
     elseif contains(ops.paradigm_sequence{parad_num}, 'Static')
        full_info{parad_num+1,5}=cell(ops.paradigm_trial_num(parad_num)+1,5);
        full_info{parad_num+1,5}(1,:)={'Trial', 'TrialStart', 'TrialEnd', 'TrialTime','GratingIndex'};
        if contains(ops.paradigm_sequence{parad_num}, 'Static1')
            gratingindexes=sampled_grating_indexes_parts(1,(~isnan(sampled_grating_indexes_parts(1,:))));
        elseif contains(ops.paradigm_sequence{parad_num}, 'Static2')
            gratingindexes=sampled_grating_indexes_parts(2,(~isnan(sampled_grating_indexes_parts(2,:))));
        elseif contains(ops.paradigm_sequence{parad_num}, 'Static3')
            gratingindexes=sampled_grating_indexes_parts(3,(~isnan(sampled_grating_indexes_parts(3,:))));
        end
        for k = 1:length(gratingindexes)
            full_info{parad_num+1,5}{k+1,5} = gratingindexes(k);
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
Priority(topPriorityLevel);  
for parad_num = 1:numel(ops.paradigm_sequence)
    full_info{parad_num+1,2}=Screen('Flip', win); % paradigm start
%     session.outputSingleScan(10); 
%     pause(0.010)
%     session.outputSingleScan(10);  
    grating=0;
    movi=0;
    images=0;
 % check what paradigm
    if strcmpi(ops.paradigm_sequence{parad_num}, 'Static1')
        texture=texgratings   ; 
        grating=1;
        texindexes=sampled_grating_indexes_parts(1,(~isnan(sampled_grating_indexes_parts(1,:))));
        totalstim=totalgratings;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Static2')
        texture=texgratings;
        grating=1;
        texindexes=sampled_grating_indexes_parts(2,(~isnan(sampled_grating_indexes_parts(2,:))));
        totalstim=totalgratings;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Static3')
        texture=texgratings;
        grating=1;
        texindexes=sampled_grating_indexes_parts(3,(~isnan(sampled_grating_indexes_parts(3,:))));
        totalstim=totalgratings;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Images1')
        texture=texscenes;
        images=1;
        texindexes=sampled_scene_indexes_parts(1,(~isnan(sampled_scene_indexes_parts(1,:))));
        totalstim=totalscenes;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Images2')
        texture=texscenes;
        images=1;
        texindexes=sampled_scene_indexes_parts(2,(~isnan(sampled_scene_indexes_parts(2,:))));
        totalstim=totalscenes;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Images3')
        texture=texscenes;
        images=1;
        texindexes=sampled_scene_indexes_parts(3,(~isnan(sampled_scene_indexes_parts(3,:))));
        totalstim=totalscenes;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Movie1')
        texture=texmov1;
        movi=1;
        frame_number=frame_number_movie_1;
        totalframenumer=ops.paradigm_trial_num(parad_num) *frame_number;

    end    
    numSecs = ops.paradigm_stim_time(parad_num);
    waitframes=round(numSecs*1/ops.flipInterval);
%     session.outputSingleScan(0); 
%     session.outputSingleScan(0); 
    if grating || images
         Screen('DrawTexture', win, texture(1,texindexes(1)), [],[],[],[],[], isi_color));          
         full_info{parad_num+1,5}{1+1,2} = Screen('Flip', win); % start trial 1(single frame)
         VBLTimestamp= full_info{parad_num+1,5}{1+1,2}(1);
         for trl=2:ops.paradigm_trial_num(parad_num) 
            Screen('DrawTexture', win, texture(1,texindexes(trl)), [],[],[],[],[], isi_color)); 
            Screen('DrawingFinished', win);
%             grat_volt=5*texindexes(trl)/totalstim;         
            full_info{parad_num+1,5}{trl+1,2}= Screen('Flip', win, VBLTimestamp+ (waitframes - 0.5) * ops.flipInterval ); % trial start 2:end
%             session.outputSingleScan(grat_volt);
            VBLTimestamp= full_info{parad_num+1,5}{trl+1,2}(1);
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
        full_info{parad_num+1,5}{trl+1,3} = Screen('Flip', win, full_info{parad_num+1,5}{trl+1,2}(1) + (waitframes - 0.5) * ops.flipInterval ); % end trial -1       
    end
    Screen('FillRect', win, isi_color, rect);      
    full_info{parad_num+1,3}=Screen('Flip', win); %paradigm end , last trial last frame end, gery end
%     session.outputSingleScan(0);
%     session.outputSingleScan(0);
    Screen('FillRect', win, isi_color, rect);      

end
Priority(0);
Screen('Close');  
sca();
%%
for parad_num=1:numel(ops.paradigm_sequence)
    full_info{parad_num+1,4}=full_info{parad_num+1,3}-full_info{parad_num+1,2};
    for trial=1:ops.paradigm_trial_num(parad_num)
      full_info{parad_num+1,5}{trial+1,4}=full_info{parad_num+1,5}{trial+1,3}-full_info{parad_num+1,5}{trial+1,2};
      %          full_info{1+parad_num,5}{1+trl-1,3}=full_info{parad_num+1,5}{trl+1,2}; %previoustrial end       

      
      
      
      if contains(ops.paradigm_sequence{parad_num}, 'Movie')
%           full_info{parad_num+1,5}{trl+1,5}{frame+1,2}=full_info{parad_num+1,5}{trl+1,2};% trialstart
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
file_name = sprintf([acquisition_name, 'AllenSessionB_%d_%d_%d_stim_data_%dh_%dm'],temp_time(1)-2000, temp_time(2), temp_time(3), temp_time(4), temp_time(5));
clear temp_time;

%% save info
fprintf('Saving...\n');
save([save_path,'\', file_name, '.mat'],'ops', 'full_info', 'grating_index_array');
fprintf('Done\n');
%% ploting images timings
for j=[4,6,12]
   figure
    plot(cell2mat(full_info{j, 5}(2:end,4)));
    ylim([0.249,0.252])
    mean(cell2mat(full_info{j, 5}(2:end,4)))
    std(cell2mat(full_info{j, 5}(2:end,4)))
end

%% ploting gratings timings
for j=[2,8,14]
    figure
    plot(cell2mat(full_info{j, 5}(2:end,4)));
    ylim([0.249,0.252])
    mean(cell2mat(full_info{j, 5}(2:end,4)))
    std(cell2mat(full_info{j, 5}(2:end,4)))
end

%% ploting movies timings
for j=[10]
    figure
    plot(cell2mat(full_info{j, 5}(2:end,4)));
    ylim([29.9,30.1])
    figure
    hold on
    for i=2:size(full_info{j, 5},1)
        plot(cell2mat(full_info{j, 5}{i,5}(2:end,3)));
    end
    ylim([0.011,0.021])
end
