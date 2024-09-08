sca()
clear 
close all
desktop_grat_inding=0;
stimdate='20240619';
mouse='Test';
fov='Test';
% opto='470_LED_Drift';
opto='617_LED_Drift';
blue=false;
is135=false;
triggered=true;
rng("shuffle")
acquisition_name=[stimdate '_' mouse '_' fov '_' opto];
stim_dir=fullfile(fileparts(fileparts(pwd)),'AllenStimuli', 'Smalles');

if is135==true
    ops.isicolor=131;
else
    ops.isicolor=ceil(255/2);
end

load(fullfile(stim_dir,'drifiting_gratings_full.mat'))
driftinggratings=all_warped_drifting_gratings_full;
clear all_warped_drifting_gratings_full 
temppfrequency=3;
driftinggratings=squeeze(driftinggratings(:,:,:,temppfrequency,:));


%waaaaa 
%% parameters
% ------ Paradigm sequence ------
ops.paradigm_sequence=cell(20,1);
ops.paradigm_trial_num =  zeros(20,1);
ops.paradigm_stim_time=   zeros(20,1) ; 
ops.paradigm_isi_time=  zeros(20,1);
ops.paradigm_frame_number=  zeros(20,1);
ops.paradigm_optograt_ind= zeros(20,1);
ops.preparadigm_time=zeros(20,1);

for i=1:20

    ops.paradigm_sequence{i,1}="Drifting"+i;
    ops.paradigm_trial_num(i,1)=20;
    ops.paradigm_stim_time(i,1)=2;
    ops.paradigm_isi_time(i,1)=2;
    ops.paradigm_frame_number(i,1)=1;
    ops.paradigm_optograt_ind(i,1)=1;
    ops.preparadigm_time(i,1)=5;

end


isi_color = [ops.isicolor ops.isicolor ops.isicolor];
isi_color_texture=[255 255 255];
if blue==true
    isi_color = [0 0 ops.isicolor];
    isi_color_texture=[0 0 255];
end

%% set opto tirggers for which trial
% chandelier one photon stimulation select 1 trial to do opto and also do
% opto duirg the spont, not during movies
%select angle to do opto, I need to count first count total repetitions
% with one photon I have to trigger by myself at a given frequency
% slect movie frame to do opto
ops.ledvoltage=2%v 
optoop.randomoptograting=false;
optoop.alloptoselect=zeros(20,8);
grat_ind=1:8;

recordingtime=ops.paradigm_stim_time(1);
optoop.stim_delay=0;
optoop.iterations=1;
optoop.interiteration_time=0;
stimtime=optoop.iterations+optoop.interiteration_time*optoop.iterations;

assert((optoop.iterations+optoop.interiteration_time*optoop.iterations)<recordingtime, 'Design longer than video')


optoop.pulse_frequency=20;%hz
optoop.pulse_width=0.020;%s
optoop.train_dur=1;%s


optoop.random_jitter= 0;
optoop.stim_delay=optoop.stim_delay+optoop.random_jitter;
optoop.number_of_pulses=optoop.pulse_frequency*optoop.train_dur;
assert(mod(optoop.number_of_pulses,1)==0, 'Pulse number is not an integer')

optoop.period=1/optoop.pulse_frequency;

assert(optoop.period>optoop.pulse_width, 'Pulse number is not an integer')

optoop.isi=optoop.period-optoop.pulse_width;




optoop.uptimes=zeros(1,optoop.number_of_pulses);
for i=1:optoop.number_of_pulses
   optoop.uptimes(i)=(i-1)*optoop.period;
end
optoop.downtimes=optoop.uptimes+optoop.pulse_width;

optoop.fulluptimes=cell(1,optoop.iterations);
optoop.fulldowntimes=cell(1,optoop.iterations);

for i=1:optoop.iterations
    if i==1
        optoop.fulluptimes{1,i}=optoop.uptimes;
        optoop.fulldowntimes{1,i}=optoop.downtimes;
    else
        optoop.fulluptimes{1,i}=optoop.fulluptimes{1,i-1}+optoop.interiteration_time+optoop.train_dur;
        optoop.fulldowntimes{1,i}=optoop.fulldowntimes{1,i-1}+optoop.interiteration_time+optoop.train_dur;
    
    end
end

optoop.uptimes=cell2mat(optoop.fulluptimes);
optoop.downtimes=cell2mat(optoop.fulldowntimes);

% stival=zeros(1,recordingtime*1000);
% for i=1:length(optoop.uptimes)
%     stival((optoop.uptimes(i)+1)*1000+optoop.stim_delay*1000:optoop.downtimes(i)*1000-1+optoop.stim_delay*1000)=1;
% 
% end
% 
% plot(stival)
% xlim([0 recordingtime*1000])


%% initialize
PsychDefaultSetup(1);
Screen('Preference', 'SkipSyncTests', 0);
AssertOpenGL; % Make sure this is running on OpenGL Psychtoolbox:
screenid = max(Screen('Screens')); % Choose screen with maximum id - the secondary display on a dual-display setup for display
if desktop_grat_inding==1
    screenid=1;
end
white = WhiteIndex(screenid);
black = BlackIndex(screenid);


[win, rect] = Screen('OpenWindow',screenid, isi_color); % rect is the coordinates of the screen
ops.flipInterval = Screen('GetFlipInterval', win);
resolution=Screen('Resolution', screenid);
reswidth=resolution.width;
resheight=resolution.height;
topPriorityLevel = MaxPriority(win);


%% drifitng gratings

optotrials=20;
gratreps=2;
present=20;
totalpresnetations=optotrials*gratreps;

totalgratings=size(driftinggratings,3);
numbersweeps=gratreps*2;
randomoptograting=randi(8);
totalstim=8;


tex = zeros(size(driftinggratings,3),size(driftinggratings,4));
% tex = zeros(size(driftinggratings));

for i=1:size(driftinggratings,3)
     for j=1:size(driftinggratings,4)
            tex(i,j)=Screen('MakeTexture', win,0.8*driftinggratings(:,:,i,j)); 
%             tex(:,:,i,j)=0.8*driftinggratings(:,:,i,j); 

     end
end
optoop.all_grating_indexes=[];

optoop.all_grating_indexes=[[1:20];[1:8 1:8 0 0 0 0]];

optoop.allrandperm=zeros(optotrials,present);
for i=1:optotrials
    optoop.allrandperm(i,:)=randperm(optotrials,present);
end


%% VoltageSignals
if ~desktop_grat_inding
    session=daq.createSession('ni');
    counter_trigger=daq('ni');
    usb_session=daq.createSession('ni');
    resetcounters(counter_trigger);
    addinput(counter_trigger,'Dev1','ctr2','EdgeCount');
    
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
    movievolmin=3;s
end
gre=greyvol;
session.outputSingleScan([0 0]); 
session.outputSingleScan([0 0]); 
%% create info arrays
full_info=cell(numel(ops.paradigm_sequence)+1,5);
full_info{1,1}='Paradigms';
full_info{1,2}='StartParadigmTime';
full_info{1,3}='EndParadigmTime';
full_info{1,4}='ParadigmDuration';
full_info{1,5}='Trials';
for parad_num=1:numel(ops.paradigm_sequence)
    full_info{1+parad_num,1}=ops.paradigm_sequence(parad_num);
  
   if contains(ops.paradigm_sequence{parad_num}, 'Drifting')
        full_info{1+parad_num,5}=cell(ops.paradigm_trial_num(parad_num)+1,7);
        full_info{1+parad_num,5}(1,:)={'Trial', 'TrialStart','StimStart' 'TrialEnd', 'TrialTime','DrifitingIndex', 'Phases'};
                gratingindexes=optoop.allrandperm;
            
            for k = 1:length(gratingindexes)
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
if triggered
    counter_data=read(counter_trigger);
    fprintf('Waiting For DaqRec trigger');
    
    while counter_data.Dev1_ctr2==0
    counter_data=read(counter_trigger);
    [keyIsDown, keysecs, keyCode] = KbCheck;
          if keyCode(KbName('ESCAPE'))
            Screen('CloseAll');
            break;
          end
    end
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
    random_opto_sel=8*(rand(1,8)<0.5);
    optoselect=grat_ind+random_opto_sel;
    optoop.alloptoselect(parad_num,:)=optoselect;

    full_info{1+parad_num,2}=GetSecs();
    session.outputSingleScan([maxvol 0]); 
    session.outputSingleScan([maxvol 0]); 
    pause(0.05)
    session.outputSingleScan([maxvol 0]);    
    session.outputSingleScan([maxvol 0]); 
    movi=0;
    

 % check what paradigm
    texture=tex;    
    texindexes=optoop.all_grating_indexes(2,optoop.allrandperm(parad_num,:));
        
    
    session.outputSingleScan([0 0]);
    session.outputSingleScan([0 0]);
    pause(0.05)
    session.outputSingleScan([0 0]);
    session.outputSingleScan([0 0]);
    pause(ops.preparadigm_time)

    alldone=zeros(1,8);
    done0=0;
    optosignal=0;
    optosignal0=0;
    pulse_counter=0;
    pulse_counter0=0;

    %% GRATING
    for trl=1:ops.paradigm_trial_num(parad_num) 
        pulse_counter=0;
        optosignal=0;
        doopto=false;

        % define voltage depending on trial
        grat_volt=movievolmin+(movievolmax-movievolmin)*(texindexes(trl))/totalstim;
        if texindexes(trl)==0
            grat_volt=movievolmax+0.5;
        end
        % select wich grating will tirgger opto
        
        %% ISI LOOP 2S
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
        
        % this is to tirgger opto if is the proper trialif the rrandomly
        % selected grating for opto is the oe who is going to be presented
        % once opto for one grating is done the alldone element for that
        % grating will set to one so here i only do opto when that values
        % is 0

        
        if texindexes(trl)~=0 && ~alldone(texindexes(trl))
            % next if is the first presentation of the gratin check  if
            % opto has to be done
             if sum(texindexes(trl)==optoselect)>0
                firstopto=true;
                if trl == find(texindexes==texindexes(trl),1)
                    doopto=true;
                end

             elseif sum(texindexes(trl)+8==optoselect)>0
                firstopto=false;
                if trl ~= find(texindexes==texindexes(trl),1)
                    doopto=true;
                end
             end
        end

        if texindexes(trl)~=0 && ~alldone(texindexes(trl)) && doopto  
                start=now;
                optosignal=ops.ledvoltage;
                pulse_counter=pulse_counter+1;
        end
        if done0<2 && 0==texindexes(trl)    
                start=now;
                optosignal0=ops.ledvoltage;
                pulse_counter0=1;
        end

        % this trigger the signal for the the prairire opto on that grating
        % is a single small pulse
        session.outputSingleScan([greyvol optosignal]);
        session.outputSingleScan([greyvol optosignal]);
  
        % select grating vs blank sweep
        if texindexes(trl)~=0
            while (now-full_info{1+parad_num,5}{1+trl,3}(1))<ops.paradigm_stim_time(parad_num)           
                now=GetSecs();
                [keyIsDown, keysecs, keyCode] = KbCheck;
                if keyCode(KbName('escape'))
                    Screen('CloseAll');
                    break;
                end
    
                %this is to calculate the voltage to do analysis later
                if rem(ct, 2) == 0
                    frame_volt=greyvol+1;
                else
                    frame_volt=greyvol-1;
                end

                if  optosignal~=0 && doopto
                        if now- start>optoop.downtimes(pulse_counter)
                            optosignal=0;
                            if pulse_counter==length(optoop.downtimes)
                                alldone(texindexes(trl))=1;

                            end
                        end
                elseif ~alldone(texindexes(trl)) && doopto
                 
                    if pulse_counter<length(optoop.uptimes) && now- start>optoop.uptimes(pulse_counter+1)
                            optosignal=ops.ledvoltage;
                            pulse_counter=pulse_counter+1;
                    end
                end
    
    
    
                % draw next frame of grating
                Screen('DrawTexture', win,texture(texindexes(trl),rem(ct,60)+1), [],[],[],[],[], isi_color_texture);
                % flip frame of grating
                [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos]=Screen('Flip',win);
                %send a flip flop voltage for every frame
               
                session.outputSingleScan([frame_volt optosignal]);
                session.outputSingleScan([frame_volt optosignal]);
    
                %save timing
                full_info{1+parad_num,5}{1+trl,7}{ct+2,2}=  [VBLTimestamp, StimulusOnsetTime, FlipTimestamp, Missed, Beampos, GetSecs()];
                ct=ct+1;
            end
        else
            % blank sweep 2 second presentation
            while (now-VBLTimestamp)<ops.paradigm_stim_time(parad_num)
                now=GetSecs();
                [keyIsDown, keysecs, keyCode] = KbCheck;
                if keyCode(KbName('escape'))
                    Screen('CloseAll');
                    break;
                end               
                if  optosignal0~=0 
                        if now- start>optoop.downtimes(pulse_counter0)
                            optosignal0=0;
                            gre=greyvol+1;
                            if pulse_counter0==length(optoop.downtimes)
                                gre=greyvol;
                            end
                        end
                elseif done0<2
                    if pulse_counter0<length(optoop.uptimes) && now- start>optoop.uptimes(pulse_counter0+1)
                            optosignal0=ops.ledvoltage;
                            pulse_counter0=pulse_counter0+1;
                            gre=greyvol+1;
                    end
                end
                session.outputSingleScan([gre optosignal0]);
                session.outputSingleScan([gre optosignal0]);
                session.outputSingleScan([gre optosignal0]);
                [keyIsDown, keysecs, keyCode] = KbCheck;
                if keyCode(KbName('escape'))
                    Screen('CloseAll');
                    break;
                end
            end 
            done0=done0+1;
            pulse_counter0=1;
            gre=greyvol;
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


filePath = matlab.desktop.editor.getActiveFilename;
[~,stiimname]=fileparts(filePath);
[ParentFolderPath] = fileparts(fileparts(fileparts(filePath)));
save_path = [ParentFolderPath '\Sessions\' stimdate '\Mice\' mouse '\UnprocessedVisStim' ];
if ~exist(save_path, 'dir')
    mkdir(save_path)
end

temp_time = clock;
file_name = sprintf([acquisition_name,'_', stiimname, '_%d_%d_%d_stim_data_%dh_%dm'],temp_time(1)-2000, temp_time(2), temp_time(3), temp_time(4), temp_time(5));
clear temp_time;

%% save info
fprintf('Saving...\n');
save([save_path,'\', file_name, '.mat'],'ops','optoop', 'full_info', 'isi_color','is135');
fprintf('Done\n');
