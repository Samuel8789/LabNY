sca()
clear 
close all
desktop_testing=1;
stimdate='20220610';
mouse='Test';
fov='Test';
opto='None';
blue=true;
is135=false;

acquisition_name=[stimdate '_' mouse '_' fov '_' opto];
stim_dir=fullfile(fileparts(pwd),'AllenStimuli', 'Smalles');

if is135==true
    ops.isicolor=131;
else
    ops.isicolor=ceil(255/2);
end


load(fullfile(stim_dir,'drifiting_gratings_full.mat'))
load(fullfile(stim_dir,'natural_movie_one.mat'))
load(fullfile(stim_dir,'natural_movie_three_full.mat'))
driftinggratings=all_warped_drifting_gratings_full;
clear all_warped_drifting_gratings_full 
naturalmovie1big=natural_movie_one_all_warped_frames;
clear natural_movie_one_all_warped_frames 
naturalmovie3big=natural_movie_three_all_warped_frames_full;
clear natural_movie_three_all_warped_frames_full 

%% parameters
% ------ Paradigm sequence ------
ops.paradigm_sequence = {'Drifting1', 'Intergrey','Movie3', 'Intergrey','Movie1','Intergrey','Drifting2' , 'Spont','Movie3','Intergrey','Drifting3'};  
ops.paradigm_trial_num =    [ 200,   1,     5,    1,    10,    1,   200,   1,     5,   1,  200];   
ops.paradigm_stim_time=     [   2,  30,   120,   30,    30,   30,     2, 300,   120,  30,    2];
ops.paradigm_isi_time=      [   1,   0,     0,    0,     0,    0,     1,   0,     0,   0,    1];
ops.paradigm_frame_number=  [   1,   1,  3600,    1,   900,    1,     1,   1,  3600,   1,    1];
ops.paradigm_optotest=      [   1,   0,     0,    0,     0,    0,     1,   1,     0,   0,    1];

isi_color = [ops.isicolor ops.isicolor ops.isicolor];
isi_color_texture=[255 255 255];
if blue==true
    isi_color = [0 0 ops.isicolor];
    isi_color_texture=[0 0 255];
end
repetitions=15;
waitframes=1;
random_jitter= randi([1 20]);
stim_delay=200+random_jitter;
iterations=1;

%% set opto tirggers for which trial
% chandelier one photon stimulation select 1 trial to do opto and also do
% opto duirg the spont, not during movies
%select angle to do opto, I need to count first count total repetitions
% with one photon I have to trigger by myself at a given frequency
optotrial=10;
% slect movie frame to do opto
optoframe=1;
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
% grey=135;

[win, rect] = Screen('OpenWindow',screenid, isi_color); % rect is the coordinates of the screen
ops.flipInterval = Screen('GetFlipInterval', win);
resolution=Screen('Resolution', screenid);
reswidth=resolution.width;
resheight=resolution.height;
topPriorityLevel = MaxPriority(win);


%% drifitng gratings

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
jitter(find(jitter==1))=-1;
jitter(find(jitter==2))=0;
jitter(find(jitter==3))=1;
sweepperiods=jitter+seepfreq;
sweepindexes=cumsum(sweepperiods,2);
sampled_grating_indexes=trial_array(randperm(total_trials, total_trials));
sampled_grating_indexes_parts = reshape(sampled_grating_indexes, 3, []);

for i=1:3
    sweepindexes(i,find(sweepindexes(i,1:numbersweeps(i))>periodreps(i)))=NaN;
    sampled_grating_indexes_parts(i,sweepindexes(i,(~isnan(sweepindexes(i,:)))))=0;
end


%% loading the movies twice the size without waitframes
%% loading the movies
[~,~,frame_number_movie_1]=size(naturalmovie1big);
texmov1=zeros(1,frame_number_movie_1);
for i=1:frame_number_movie_1
    texmov1(i)=Screen('MakeTexture', win, naturalmovie1big(:,:,i)); 
end
texmov1=reshape(repmat(texmov1,2,1),size(texmov1,1),2*size(texmov1,2));

[~,~,frame_number_movie_3]=size(naturalmovie3big);
texmov3=zeros(1,frame_number_movie_3);
for i=1:frame_number_movie_3
    texmov3(i)=Screen('MakeTexture', win, naturalmovie3big(:,:,i)); 
end
texmov3=reshape(repmat(texmov3,2,1),size(texmov3,1),2*size(texmov3,2));
clear naturalmovie1big naturalmovie2big
ops.paradigm_frame_number=  [   1,   1,  frame_number_movie_3,    1,   frame_number_movie_1,    1,     1,   1,  frame_number_movie_3,   1,    1];

%% VoltageSignals
if ~desktop_testing
    session=daq.createSession('ni');
    counter_trigger=daq('ni');
    usb_session=daq.createSession('ni');
    resetcounters(counter_trigger);
    addinput(counter_trigger,'Dev1','ctr0','EdgeCount');
    
    session.addAnalogOutputChannel('Dev1','ao0','Voltage');
    session.addAnalogOutputChannel('Dev1','ao1','Voltage');
    
    usb_session.addAnalogOutputChannel('Dev2','ao0','Voltage');
    usb_session.addAnalogOutputChannel('Dev2','ao1','Voltage');

    session.IsContinuous = true;
    session.Rate = 10000;
    maxvol=10;
    greyvol=2;
    movievolmax=9;
    movievolmin=4;

else
    session=daq('ni');
    session=daq.createSession('ni');
    session.addAnalogOutputChannel('Dev1','ao1','Voltage');
%     session.IsContinuous = true;
%     session.Rate = 10000;
    session.outputSingleScan([5 0]);
    session.outputSingleScan([0 0]);
    maxvol=5;
    greyvol=1;
    movievolmax=4.5;
    movievolmin=3;
end
%% create info arrays
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
   elseif contains(ops.paradigm_sequence{parad_num}, 'Drifting')
        full_info{1+parad_num,5}=cell(ops.paradigm_trial_num(parad_num)+1,7);
        full_info{1+parad_num,5}(1,:)={'Trial', 'TrialStart','StimStart' 'TrialEnd', 'TrialTime','DrifitingIndex', 'Phases'};
            if contains(ops.paradigm_sequence{parad_num}, 'Drifting1')
                gratingindexes=sampled_grating_indexes_parts(1,(~isnan(sampled_grating_indexes_parts(1,:))));
            elseif contains(ops.paradigm_sequence{parad_num}, 'Drifting2')
                gratingindexes=sampled_grating_indexes_parts(2,(~isnan(sampled_grating_indexes_parts(2,:))));
            elseif contains(ops.paradigm_sequence{parad_num}, 'Drifting3')
                gratingindexes=sampled_grating_indexes_parts(3,(~isnan(sampled_grating_indexes_parts(3,:))));
            end
            for k = 1:length(gratingindexes);
                full_info{1+parad_num,5}{k+1,6} = gratingindexes(k);
            end    
            for trial=1:ops.paradigm_trial_num(parad_num)
                full_info{1+parad_num,5}{trial+1,1}=trial;
                phase_number=120;
                full_info{1+parad_num,5}{trial+1,7}=cell(phase_number+1,4);
                full_info{1+parad_num,5}{trial+1,7}(1,:)={'Phase', 'PahseStart', 'PhaseTime','PhaseValue'};
                for phase=1:phase_number
                    full_info{1+parad_num,5}{trial+1,7}{phase+1,1}=phase;
                end     
            end
    else
        full_info{1+parad_num,5}=cell(ops.paradigm_trial_num(parad_num)+1,4);
        full_info{1+parad_num,5}(1,:)={'Trial', 'TrialStart', 'TrialEnd', 'TrialTime'};
        for trial=1:ops.paradigm_trial_num(parad_num)
                full_info{1+parad_num,5}{trial+1,1}=trial;
        end
    end   
end
% clear noiseindexes noiseindexes1 noiseindexes2 noiseindexes3 splitrandperm
% mydlg = warndlg('Ok to strat VisStim.', 'A Warning Dialog');
% waitfor(mydlg);
% disp('VisStimStarted.');startexp=GetSecs();
%% set up tirgger
counter_data=read(counter_trigger);
fprintf('Waiting For DaqRec trigger');

while counter_data.Dev1_ctr0==0
    counter_data=read(counter_trigger);
end
fprintf('Trigering Acquisition');
% trigger scquisition
pause(5);
usb_session.outputSingleScan([5 0]);
usb_session.outputSingleScan([5 0]);
pause(1)
usb_session.outputSingleScan([0 0]);
usb_session.outputSingleScan([0 0]);
pause(3);
% triggerLED start
usb_session.outputSingleScan([0 5]);
usb_session.outputSingleScan([0 5]);
pause(1);
usb_session.outputSingleScan([0 0]);
usb_session.outputSingleScan([0 0]);
pause(5);

%% RUN STIMULI
for parad_num = 1:numel(ops.paradigm_sequence)
    full_info{1+parad_num,2}=GetSecs();
    session.outputSingleScan([maxvol 0]); 
    session.outputSingleScan([maxvol 0]); 
    pause(0.05)
    session.outputSingleScan([maxvol 0]);    
    session.outputSingleScan([maxvol 0]); 
    grating=0;
    movi=0;
    
    optotest=0;

 % check what paradigm
    if strcmpi(ops.paradigm_sequence{parad_num}, 'Drifting1')
        texture=driftingtex   ;    
        texindexes=sampled_grating_indexes_parts(1,(~isnan(sampled_grating_indexes_parts(1,:))));
        grating=1;
        optotest=ops.paradigm_optotest(parad_num);
        
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Drifting2')
        texindexes=sampled_grating_indexes_parts(2,(~isnan(sampled_grating_indexes_parts(2,:))));
        texture=driftingtex;
        grating=1;
        optotest=ops.paradigm_optotest(parad_num);

    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Drifting3')
        texindexes=sampled_grating_indexes_parts(3,(~isnan(sampled_grating_indexes_parts(3,:))));
        texture=driftingtex;
        grating=1;
        optotest=ops.paradigm_optotest(parad_num);

    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Movie1')
        texture=texmov1;
        movi=1;  
        frame_number=frame_number_movie_1*2;
        optotest=ops.paradigm_optotest(parad_num);

    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Movie3')
        texture=texmov3;
        movi=1;
        frame_number=frame_number_movie_3*2;
        optotest=ops.paradigm_optotest(parad_num);
        
    end    
    
    session.outputSingleScan([0 0]);
    session.outputSingleScan([0 0]);
    pause(0.05)
    session.outputSingleScan([0 0]);
    session.outputSingleScan([0 0]);

    %% GRATING
    if grating
         for trl=1:ops.paradigm_trial_num(parad_num) 
            % define voltage depending on trial
            grat_volt=movievolmin+(movievolmax-movievolmin)*(texindexes(trl)-1)/(ops.paradigm_trial_num(parad_num)-1)
             if texindexes(trl)==0
                grat_volt=movievolmax+0.5;
             end
            % select wich grating will tirgger opto
            if optotest
                if texindexes(trl)==optotrial
                    optotriger=5
                else
                    optotriger=0;
                end
            else
                optotriger=0;

            end
            %% ISI LOOP 1S
            % start first ISI
            Screen('FillRect', win, isi_color, rect);
            now=GetSecs();
            [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);% strat trial start isi
            full_info{1+parad_num,5}{1+trl,2}=[VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()]; % strat trial start isi
            % send volateg after making screen blue
            session.outputSingleScan([grat_volt 0]);
            session.outputSingleScan([grat_volt 0]);
            pause(0.05)
            session.outputSingleScan([greyvol 0]);
            session.outputSingleScan([greyvol 0]);
            % prepare drawing the grating forwhen ISI FINISH
            if texindexes(trl)~=0
                %draw first frame of drifting grating
                Screen('DrawTexture', win,texture(texindexes(trl),1), [],[],[],[],[], isi_color_texture);
                ct=1;
            else
                Screen('FillRect', win, isi_color);
            end       

            % count IS TIME
            while (now-VBLTimestamp)<ops.paradigm_isi_time(parad_num)
                      now=GetSecs();
            end          
            %% GRATING LOOP 2s
            %flip first frame of grating
            [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);% stim start end isi
            full_info{1+parad_num,5}{1+trl,3}=[VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()]; % stim start end isi
            % this is to tirgger opto if is the proper trial
            session.outputSingleScan([greyvol optotriger]);
            session.outputSingleScan([greyvol optotriger]);
            pause(0.003)
            session.outputSingleScan([greyvol 0]);
            session.outputSingleScan([greyvol 0]);

            % select grating vs blank sweep
            if texindexes(trl)~=0
                while (now-full_info{1+parad_num,5}{1+trl,3}(1))<ops.paradigm_stim_time(parad_num)           
                        now=GetSecs();
                        % draw next frame of grating
                        Screen('DrawTexture', win,texture(texindexes(trl),rem(ct,60)+1), [],[],[],[],[], isi_color_texture);
                        % flip frame of grating
                        [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);
                        %send a flip flop voltage for every frame
                        if rem(ct, 2) == 0
                            frame_volt=greyvol+1;
                        else
                            frame_volt=greyvol-1;
                        end
                        session.outputSingleScan([frame_volt 0]);
                        session.outputSingleScan([frame_volt 0]);

                        %save timing
                        full_info{1+parad_num,5}{1+trl,7}{ct+2,2}=  [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
                        ct=ct+1;
                end
            else
                % blank sweep 2 second presentation
                while (now-VBLTimestamp)<ops.paradigm_stim_time(parad_num)
                    now=GetSecs();
                end 
            end
         end
    %% MOVIE     
    elseif movi
        for trl=1:ops.paradigm_trial_num(parad_num)
            % first set voltage for each trial
            trial_volt=movievolmin+(movievolmax-movievolmin)*(trl-1)/(ops.paradigm_trial_num(parad_num)-1);

            % set optotriggers
            if optotest
                if  optoframe==1
                    optotriger=5
                end
            else
                optotriger=0;
            end
            
            % draw and flip first frma eof the movie
            Screen('DrawTexture', win,texture(1,1), [],[],[],[],[], isi_color_texture);
            [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win); %strat first frame
            ct=1;
            %  senf voltage indicating trial and start opto in case is with frame 1
            session.outputSingleScan([trial_volt optotriger]);
            session.outputSingleScan([trial_volt optotriger]);
            pause(0.003)
            session.outputSingleScan([trial_volt 0]);
            session.outputSingleScan([trial_volt 0]);
            full_info{1+parad_num,5}{1+trl,2}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];%strat first frame
            now=GetSecs();
            while (now-full_info{1+parad_num,5}{1+trl,2}(1))<ops.paradigm_stim_time(parad_num)%-ops.paradigm_stim_time(parad_num)/frame_number
                now=GetSecs();
                % decide if doing otpotirgger
                if optotest
                    if  rem(ct,frame_number)+1==optoframe;
                        optotriger2=5
                    end
                else
                    optotriger2=0;
                end
                %draw and flip frame
                Screen('DrawTexture', win,texture(1,rem(ct,frame_number)+1), [],[],[],[],[], isi_color_texture);
                [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win); %strat frame highre  than one
                %flip flop voltage by frames and trigger optogentic
                if rem(ct, 2) == 0
                    frame_volt=greyvol+1;
                else
                    frame_volt=greyvol-1;
                end          
                session.outputSingleScan([frame_volt optotriger2]);
                session.outputSingleScan([frame_volt optotriger2]);
                pause(0.003)
                session.outputSingleScan([frame_volt 0]);
                session.outputSingleScan([frame_volt 0]);
                full_info{1+parad_num,5}{1+trl,5}{1+ct+1,2}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
                ct=ct+1;
            end                
        end
    %% GREYS    
    else
         for trl=1:ops.paradigm_trial_num(parad_num)
             % this fill rect is irrelevant
            Screen('FillRect', win, isi_color, rect);
            [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);
            full_info{1+parad_num,5}{1+trl,2}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()]; %grey period start
            session.outputSingleScan([greyvol 0]);
            session.outputSingleScan([greyvol 0]);

            now=GetSecs();   
            it=1;
            while (now- full_info{1+parad_num,5}{1+trl,2}(1))<ops.paradigm_stim_time(parad_num)%-1/60
                now=GetSecs();
                % this is to trigger the opto protocol during spontaneous activity
                if ops.paradigm_optotest(parad_num) && it==iterations && (now-full_info{1+parad_num,5}{1+trl,2}(1))>stim_delay  
                    session.outputSingleScan([greyvol 5]);
                    session.outputSingleScan([greyvol 5]);
                    pause(0.003)
                    session.outputSingleScan([greyvol 0]);
                    session.outputSingleScan([greyvol 0]);
                    it=it+1
                end 
            end 
        end
    end
    % finish paradign by drawing grey screen
    Screen('FillRect', win, isi_color, rect);
    % flip end of paradigm grey screen so next paradigm starts gery
    [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win); % paradigm end flip, gives last trial end and paradigm end
    full_info{1+parad_num,5}{1+trl,3}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()]; % paradigm end flip, gives last trial  end
    full_info{1+parad_num,3}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];      % paradigm end flip, and paradigm end 
    session.outputSingleScan([0 0]);
    session.outputSingleScan([0 0]);
    pause(0.05)
    session.outputSingleScan([0 0]);
    session.outputSingleScan([0 0]);
end

% finsih visual stimulation
session.outputSingleScan([maxvol 0]); 
session.outputSingleScan([maxvol 0]);   
pause(0.1)
session.outputSingleScan([0 0]);
session.outputSingleScan([0 0]);
Screen('Close');  
sca();

%% trigger led and stop acquisition
pause(5)
% END LED
usb_session.outputSingleScan([0 5])
usb_session.outputSingleScan([0 5])
pause(1)
usb_session.outputSingleScan([0 0])
usb_session.outputSingleScan([0 0])
pause(3)
% stop acq
usb_session.outputSingleScan([5 0])
usb_session.outputSingleScan([5 0])
pause(1)
usb_session.outputSingleScan([0 0])
usb_session.outputSingleScan([0 0])
%%
% for parad_num=1:numel(ops.paradigm_sequence)
%     full_info{1+parad_num,4}=full_info{1+parad_num,3}(1)-full_info{1+parad_num,2}(1); %paradigm time
%     for trl=1:ops.paradigm_trial_num(parad_num)
%       if contains(ops.paradigm_sequence{parad_num}, 'Noise')
%           if trl<ops.paradigm_trial_num(parad_num)
%             full_info{1+parad_num,5}{1+trl,3}=full_info{1+parad_num,5}{1+trl+1,2}; %trial end is same as next trial start
%           else
%             full_info{1+parad_num,5}{1+trl,3}=full_info{1+parad_num,3}; %trial end is same as paradigm end
%           end
%           full_info{1+parad_num,5}{1+trl,4}=full_info{1+parad_num,5}{1+trl,3}(1)-full_info{1+parad_num,5}{1+trl,2}(1); % trial time 
% 
%       elseif contains(ops.paradigm_sequence{parad_num}, 'Movie') 
%             full_info{1+parad_num,5}{1+trl,5}{1+1,2}=full_info{1+parad_num,5}{1+trl,2} ;%first fram start same as trial star
%             if trl<ops.paradigm_trial_num(parad_num)             
%                 full_info{1+parad_num,5}{1+trl,3}=full_info{1+parad_num,5}{1+trl+1,2};  %trial end is same as next trial start
%             else
%                 full_info{1+parad_num,5}{1+trl,3}=full_info{1+parad_num,3}; %trial end is same as paradigm end
%             end
%             full_info{1+parad_num,5}{1+trl,4}=full_info{1+parad_num,5}{1+trl,3}(1)-full_info{1+parad_num,5}{1+trl,2}(1); %trial time
% 
%             for frame=1:length(full_info{1+parad_num,5}{1+trl,5}(:,2))-1   
%                 if frame<length(full_info{1+parad_num,5}{1+trl,5}(:,2))-1   
% %                     full_info{1+parad_num,5}{1+trl,5}{1+frame,1}=frame-1; 
%                     full_info{1+parad_num,5}{1+trl,5}{1+frame,3}=full_info{1+parad_num,5}{1+trl,5}{1+frame+1,2}; %  frame ens same as next frame start
%                 elseif frame==length(full_info{1+parad_num,5}{1+trl,5}(:,2))-1
%                     if trl<ops.paradigm_trial_num(parad_num)     
%                         full_info{1+parad_num,5}{1+trl,5}{1+frame,3}=   full_info{1+parad_num,5}{1+trl+1,2}; %  frame ens same astrialstart
%                     else
%                         full_info{1+parad_num,5}{1+trl,5}{1+frame,3}=   full_info{1+parad_num,5}{1+trl,3}; %  last trials last frame end is same as paradigm end
%                     end
%                 else
%                      full_info{1+parad_num,5}{1+trl,5}{frame,3}=full_info{1+parad_num,3}; %trial end is same as paradigm end
% %                      full_info{1+parad_num,5}{1+trl,5}{frame,1}=frame-1; 
%                 end
%                 full_info{1+parad_num,5}{1+trl,5}{1+frame,4}=full_info{1+parad_num,5}{1+trl,5}{1+frame,3}(1)-full_info{1+parad_num,5}{1+trl,5}{1+frame,2}(1); %frame time
%             end
%       else
%           full_info{1+parad_num,5}{1+trl,3}=full_info{1+parad_num,3}; %trial end is same as paradigm end
%           full_info{1+parad_num,5}{1+trl,4}=full_info{1+parad_num,5}{1+trl,3}(1)-full_info{1+parad_num,5}{1+trl,2}(1); % trial time       
%       end     
%     end
% end 

filePath = matlab.desktop.editor.getActiveFilename;
[~,stiimname]=fileparts(filePath)
[ParentFolderPath] = fileparts(fileparts(filePath));
save_path = [ParentFolderPath '\Sessions\' stimdate '\Mice\' mouse '\UnprocessedVisStim' ];
if ~exist(save_path, 'dir')
    mkdir(save_path)
end

[~,stiimname]=fileparts(filePath)

temp_time = clock;
file_name = sprintf([acquisition_name,'_', stiimname, '_%d_%d_%d_stim_data_%dh_%dm'],temp_time(1)-2000, temp_time(2), temp_time(3), temp_time(4), temp_time(5));
clear temp_time;

%% save info
fprintf('Saving...\n');
save([save_path,'\', file_name, '.mat'],'ops', 'full_info', 'isi_color','is135');
fprintf('Done\n');
%% ploting noise timings
% for j=[2,6,10]
%    figure
%     plot(cell2mat(full_info{j, 5}(2:end,4)));
% %     ylim([0.249,0.252]);
%     mean(cell2mat(full_info{j, 5}(2:end,4)));
%     std(cell2mat(full_info{j, 5}(2:end,4)));  
% end
% 
% %% ploting movies timings
% for j=[4,8]
%     figure
%     plot(cell2mat(full_info{j, 5}(2:end,4)));
% %     ylim([29.9,30.1])
%     mean(cell2mat(full_info{j, 5}(2:end,4)))
%     std(cell2mat(full_info{j, 5}(2:end,4)))
%     figure
%     hold on
%     for i=2:size(full_info{j, 5},1)
%         plot(cell2mat(full_info{j, 5}{i,5}(2:end,4)));       
%     end
% %     ylim([0.000,0.04])
% end
