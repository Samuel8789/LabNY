sca()
clear all
close all
desktop_testing=0;

date='20220517';
mouse='Test';
fov='Test';
opto='None';
blue=true
acquisition_name=[date '_' mouse '_' fov '_' opto];
% 10min spont with  a single trigger after 5 min +1-20s
%% parameters
% ------ Paradigm sequence ------
ops.paradigm_sequence = {'Spont'};  

ops.paradigm_trial_num =  [ 1  ];   
ops.paradigm_stim_time=     [ 600,  ];
ops.paradigm_isi_time=      [   0   ];
ops.paradigm_frame_number=  [   1  ];

ops.isicolor=127;      %appx middle                       % Shade of gray on screen during  isi (1 if black, 255/2 if gray)
isi_color = [ops.isicolor ops.isicolor ops.isicolor];

if blue
    isi_color = [0 0 ops.isicolor];
end
waitframes=1;

random_jitter= randi([1 20]);
stim_delay=300+random_jitter;
optotest=0;
iterations=1;

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

%% build info arrays
full_info=cell(numel(ops.paradigm_sequence)+1,5);
full_info{1,1}='Paradigms';
full_info{1,2}='StartParadigmTime';
full_info{1,3}='EndParadigmTime';
full_info{1,4}='ParadigmDuration';
full_info{1,5}='Trials';
for parad_num=1:numel(ops.paradigm_sequence)
    full_info{1+parad_num,1}=ops.paradigm_sequence(parad_num);
        full_info{1+parad_num,5}=cell(ops.paradigm_trial_num(parad_num)+1,4);
        full_info{1+parad_num,5}(1,:)={'Trial', 'TrialStart', 'TrialEnd', 'TrialTime'};
        for trial=1:ops.paradigm_trial_num(parad_num)
                full_info{1+parad_num,5}{trial+1,1}=trial;
        end
       
end


%% wait for tirgger
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

%% FIRST GRATING TIME 10min
for parad_num = 1:numel(ops.paradigm_sequence)
    full_info{1+parad_num,2}=GetSecs();
    if exist ('session', 'var')
        session.outputSingleScan([maxvol 0]); 
        pause(0.05)
        session.outputSingleScan([maxvol 0]);    
    end
 % check what paradigm
    if exist ('session', 'var')
        session.outputSingleScan([0 0]);
        pause(0.05)
        session.outputSingleScan([0 0]);
    end

    for trl=1:ops.paradigm_trial_num(parad_num)
        Screen('FillRect', win, isi_color, rect);
        [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);
        full_info{1+parad_num,5}{1+trl,2}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
        if exist ('session', 'var')
            session.outputSingleScan([greyvol 0]);
        end
        now=GetSecs();      
        it=1;
        while (now- full_info{1+parad_num,5}{1+trl,2}(1))<ops.paradigm_stim_time(parad_num)%-1/60
                  now=GetSecs();
              if optotest && it==iterations && (now-full_info{1+parad_num,5}{1+trl,2}(1))>stim_delay  
                session.outputSingleScan([greyvol 5]);
                session.outputSingleScan([greyvol 5]);
                pause(0.003)
                session.outputSingleScan([greyvol 0]);
                session.outputSingleScan([greyvol 0]);
                it=it+1;
              end
        end 
%             [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win,1,1); % extra final flip flip
%             full_info{1+parad_num,5}{1+trl,3}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
    end
    [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win,1,1); % extra final flip flip
    full_info{1+parad_num,5}{1+trl,3}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
    full_info{1+parad_num,3}= [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];      
    if exist ('session', 'var')
        session.outputSingleScan([0 0]);
        pause(0.05)
        session.outputSingleScan([0 0]);
    end
end
session.outputSingleScan([maxvol 0]); 
pause(0.05)
session.outputSingleScan([maxvol 0]);   
session.outputSingleScan([0 0]);
session.outputSingleScan([0 0]);
Screen('Close');  
sca();

%% trigger led and stop acquisition
usb_session.outputSingleScan([0 5])
pause(1)
usb_session.outputSingleScan([0 0])
pause(3)
% stop acq

usb_session.outputSingleScan([5 0])
usb_session.outputSingleScan([5 0])
usb_session.outputSingleScan([0 0])
usb_session.outputSingleScan([0 0])


%% fill info array
% for parad_num=1:numel(ops.paradigm_sequence)
%     full_info{1+parad_num,4}=full_info{1+parad_num,3}(1)-full_info{1+parad_num,2}(1); %paradigm time
%     for trl=1:ops.paradigm_trial_num(parad_num)
%      
%           full_info{1+parad_num,5}{1+trl,3}=full_info{1+parad_num,3}; %trial end is same as paradigm end
%           full_info{1+parad_num,5}{1+trl,4}=full_info{1+parad_num,5}{1+trl,3}(1)-full_info{1+parad_num,5}{1+trl,2}(1); % trial time       
%     end
% end 

%% save stuff
filePath = matlab.desktop.editor.getActiveFilename;
[ParentFolderPath] = fileparts(fileparts(filePath))
save_path = [ParentFolderPath '\Sessions\' date '\Mice\' mouse '\UnprocessedVisStim' ];
temp_time = clock;
file_name = sprintf([acquisition_name, '_Spont_%d_%d_%d_stim_data_%dh_%dm'],temp_time(1)-2000, temp_time(2), temp_time(3), temp_time(4), temp_time(5));
clear temp_time;

fprintf('Saving...\n');
save([save_path,'\', file_name, '.mat'],'ops', 'full_info');
fprintf('Done\n');

