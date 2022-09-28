sca()
clear
close all
desktop_testing=0;
stimdate='20220610';
mouse='Test';
fov='Test';
opto='None';
blue=true; 
is135=false;

acquisition_name=[stimdate '_' mouse '_' fov '_' opto];
stim_dir=fullfile(fileparts(pwd),'AllenStimuli', 'Smalles');

if is135==true
    load(fullfile(stim_dir,'locally_sparse_noise_full_135.mat'))
    ops.isicolor=135;
else
    load(fullfile(stim_dir,'locally_sparse_noise_full.mat'))
    ops.isicolor=ceil(255/2);
end

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
ops.paradigm_trial_num =    [2900,   1,    10,  1,  2900,   1,    10,   1,  3200];   
ops.paradigm_stim_time=     [ 1/4, 300,    30, 30,   1/4,  30,    30, 300,   1/4];
ops.paradigm_isi_time=      [   0,   0,     0,  0,     0,   0,     0,   0,     0];
ops.paradigm_frame_number=  [   1,   1,   900,  1,     1,   1,   900,   1,     1];
ops.paradigm_optotest=      [   0,   0,     0,  0,     0,   0,     0,   0,     0,];

isi_color = [ops.isicolor ops.isicolor ops.isicolor];
isi_color_texture=[255 255 255]
if blue==true
    isi_color = [0 0 ops.isicolor];
    isi_color_texture=[0 0 255];
end

waitframes=1;
random_jitter= randi([1 20]);
stim_delay=200+random_jitter;
iterations=1;

%% set opto tirggers for which trial
%select angle to do opto

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


%% sampling the noise matrix

ncol = 9000;
splitrandperm=randperm(size(fullnoise,3));
Ar=fullnoise(:,:,splitrandperm);
clear fullnoise
noise1=Ar(:,:,1:2900);
noise2=Ar(:,:,2901:5800);
noise3=Ar(:,:,5801:9000);
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
% clear noise1 noise2 noise3 

noiseindexes1=splitrandperm(1:2900)';
noiseindexes2=splitrandperm(2901:5800)';
noiseindexes3=splitrandperm(5801:9000)';


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
% clear naturalmovie1big naturalmovie2big
ops.paradigm_frame_number=  [   1,   1,   frame_number_movie_1,  1,     1,   1,   frame_number_movie_2,   1,     1];

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
    noise=0;
    movi=0;
        
    optotest=0;
    
 % check what paradigm
    if strcmpi(ops.paradigm_sequence{parad_num}, 'Noise1')
        texture=texnoise1; 
        noise=1;
        optotest=ops.paradigm_optotest(parad_num);
        
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Noise2')
        texture=texnoise2;
        noise=1;
        optotest=ops.paradigm_optotest(parad_num);
        
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Noise3')
        texture=texnoise3;
        noise=1;
        optotest=ops.paradigm_optotest(parad_num);
        
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Movie1')
        texture=texmov1;
        movi=1;
        frame_number=frame_number_movie_1*2;
        optotest=ops.paradigm_optotest(parad_num);
        
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Movie2')
        texture=texmov2;
        movi=1;
        frame_number=frame_number_movie_2*2;
        optotest=ops.paradigm_optotest(parad_num);
        
     end    
    
    session.outputSingleScan([0 0]);
    session.outputSingleScan([0 0]);
    pause(0.05)
    session.outputSingleScan([0 0]);
    session.outputSingleScan([0 0]);
    
    %% NOISE
    if noise
         for trl=1:ops.paradigm_trial_num(parad_num) 
             
            % select wich grating will tirgger opto
            if optotest
                if texindexes(trl)==optotrial
                    optotriger=5
                    optotriger2=0;
                elseif texindexes(trl+1)==optotrial
                    optotriger2=5;
                    optotriger=0;
                end
            else
                optotriger=0;
                optotriger2=0;
            end
            % define voltage depending flip flop betwen trial, keep order in saved
            if rem(trl, 2) == 0
                trial_volt=greyvol+1;
            else
                trial_volt=greyvol-1;
            end
            % run first image and second image
            if trl==1
                now=GetSecs();
                %draw and flip frist trial
                Screen('DrawTexture', win, texture(1,trl), isi_color_texture);
                [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);% show first image start
                
                session.outputSingleScan([trial_volt optotriger]);
                session.outputSingleScan([trial_volt optotriger]);
                pause(0.003)
                session.outputSingleScan([trial_volt 0]);
                session.outputSingleScan([trial_volt 0]);
                full_info{1+parad_num,5}{1+trl,2}=[VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];% show first image start
                %prepare and draw nex texture
                Screen('DrawTexture', win, texture(1,trl+1), isi_color_texture);
                % wait 250ms with first image
                while (now- VBLTimestamp)<ops.paradigm_stim_time(parad_num)
                        now=GetSecs();
                end     
                %flip second image
                [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);% show second image start
                session.outputSingleScan([trial_volt optotriger2]);
                session.outputSingleScan([trial_volt optotriger2]);
                pause(0.003)
                session.outputSingleScan([trial_volt 0]);
                session.outputSingleScan([trial_volt 0]);
                full_info{1+parad_num,5}{1+trl+1,2}=[VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
                
            elseif trl>1 && trl<ops.paradigm_trial_num(parad_num)
                %prepare and draw nex texture
                Screen('DrawTexture', win, texture(1,trl+1), isi_color_texture);
                % wait 250ms with current trial
                while (now-VBLTimestamp)<ops.paradigm_stim_time(parad_num)
                    now=GetSecs();
                end
                % flip trl+1 image
                [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);
                session.outputSingleScan([trial_volt optotriger2]);
                session.outputSingleScan([trial_volt optotriger2]);
                pause(0.003)
                session.outputSingleScan([trial_volt 0]);
                session.outputSingleScan([trial_volt 0]);
                full_info{1+parad_num,5}{1+trl+1,2}=[VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
                
            else
                %prepare and draw final grey scren
                Screen('FillRect', win, isi_color, rect);
                % wait 250ms with last trial
                while (now-VBLTimestamp)<ops.paradigm_stim_time(parad_num)
                    now=GetSecs();
                end 
                [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);
                full_info{1+parad_num,5}{1+trl,3}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()]; 
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
            optotriger=0;
            full_info{1+parad_num,5}{1+trl,2}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];%strat first frame
            now=GetSecs();
            while (now-full_info{1+parad_num,5}{1+trl,2}(1))<ops.paradigm_stim_time(parad_num)%-ops.paradigm_stim_time(parad_num)/frame_number
                now=GetSecs();
                % decide if doing otpotirgger
                if optotest
                    if  rem(ct,frame_number)+1==optoframe;
                        optotriger2=5
                    end
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
                optotriger2=0
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
save([save_path,'\', file_name, '.mat'],'ops', 'full_info', 'isi_color', 'is135');
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
