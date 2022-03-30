%%
clear
desktop_testing=1
%% parameters
% ------ Stim params ------
ops.spont_time=60*10
ops.stim_time =100;                                         % sec
ops.isi_time = 30;
% ------ Paradigm sequence ------
ops.paradigm = {'Control', 'Habituation'};     % 3 options {'Control', 'MMN', 'flip_MMN'}, concatenate as many as you want
ops.paradigm_day=[-2:7];
ops.paradigm_trial_num=5

% ------ Stim params ------
ops.squarewave=1;                                           % do squarewaves instead of sinewaves
ops.isicolor=135;      %appx middle                         % Shade of gray on screen during isi (1 if black, 255/2 if gray)
ops.ctrsts=[.015625 .03125 .0625 .125 .25 .5 .85 1];          % Range of contrasts to use
ops.ctrst = 8;                                                  % pick 
ops.spfrqs=[.01  .02 .04 .05 .08 .16 .32];                      % Range of spatial freqs to use
ops.spfrq = 4;                                                  % pick 
ops.angs_rad = pi*(0:7)/8;                                  % Orientations to use, in rad
% ------ Moving Gratings ------
ops.driftingGrating = 1;                                    % use if you want to make it drifting grating
if ops.driftingGrating
    ops.angs_rad = 2 * pi*(0:7)/8;
    ops.cps=2;   %movement velocity. cycles per secondo 
%     cpd = .2967; % cm per degree when distance between eye and monitor is 15 cm i.e. tand.5 x 15 x 2 =  0.2618
%     spfreq_new = ops.spfrq/(cpd*reswidth/(40.64)); % convert # of cycles per degree to # of cycles per pixel; 40.64 is the screen width in cm 
end

%% predicted run time
% add maybe

%% paths and saves
% 
% temp_time = clock;
% file_name = sprintf('vMMN_driftG_%d_%d_%d_stim_data_%dh_%dm.mat',temp_time(2), temp_time(3), temp_time(1)-2000, temp_time(4), temp_time(5));
% clear temp_time;


%% gui selector and previous dataloading
prompt = {'Enter Mouse Code:','Enter Treatment:', 'Enter ExpDay:'};
dlgtitle = 'Input';
dims = [1 35];
definput = {'SP','Control', '0'};
MouseInfo = inputdlg(prompt,dlgtitle,dims,definput);

datinfo = datestr(datetime);
savefile = [date '_' MouseInfo{1} '_' MouseInfo{2} '_Day' MouseInfo{3} ];
[fnm pnm] = uiputfile('*.mat','Save File A..',savefile);
treatment=MouseInfo{2};
expday=str2num(MouseInfo{3});
PreviousExp=dir( [[datestr(datetime('yesterday')) '_' MouseInfo{1} '_' MouseInfo{2} '_Day' MouseInfo{3}-1] '.mat']);
if size(PreviousExp,1)>0;
    Yester=load(PreviousExp.name);
end

TodayHabit=dir( [datestr(datetime('today')) '_SP*_' 'Habituation' '_Day1.mat']);
if size(TodayHabit,1)>0;
    Tody=load(TodayHabit.name);
end

%% initialize
Screen('Preference', 'SkipSyncTests', 1);
AssertOpenGL; % Make sure this is running on OpenGL Psychtoolbox:
screenid = max(Screen('Screens')); % Choose screen with maximum id - the secondary display on a dual-display setup for display
if desktop_testing
    screenid=1
end
[win, rect] = Screen('OpenWindow',screenid, [255/2 255/2 255/2]); % rect is the coordinates of the screen
ops.flipInterval = Screen('GetFlipInterval', win);
%% DAQ CONFIG
% session=daq.createSession('ni');
% session.addAnalogOutputChannel('Dev1','ao0','Voltage');
% session.addAnalogOutputChannel('Dev1','ao1','Voltage');
% session.IsContinuous = true;
% %session.Rate = 10000;
% session.outputSingleScan([0,0]);

%% create stim

isi_color = [ops.isicolor ops.isicolor ops.isicolor];
tex = zeros(numel(ops.angs_rad)*numel(ops.ctrst)*numel(ops.spfrq),30);
angsy = sin(ops.angs_rad);
angsx = cos(ops.angs_rad);
for cc=ops.ctrst %contrast (out of 7)
    contrast=ops.ctrsts(cc);
    white = WhiteIndex(win); % pixel value for white
    black = BlackIndex(win); % pixel value for black
    gray = (white+black)/2;
    inc = white-gray;
    for s=ops.spfrq %determine spatial frequency
        scrsz = rect;
        [x,y] = meshgrid((-scrsz(3)/2)+1:(scrsz(3)/2)-1, (-scrsz(4)/2)+1:(scrsz(4)/2)-1);
        sp1=(.5799/10.2)*ops.spfrqs(s); %10.2 is just some scaling factor that i calibrated. do not change unless you know what youre doing!
%     cpd = .2967; % cm per degree when distance between eye and monitor is 15 cm i.e. tand.5 x 15 x 2 =  0.2618
%     spfreq_new = ops.spfrqs(s)/(cpd*scrsz(3)/(40.64)); % convert # of cycles per degree to # of cycles per pixel; 40.64 is the screen width in cm 
        for ang=1:numel(ops.angs_rad)
            if ops.driftingGrating
                ops.mon_framerate=60;
                for ii=1:30
                    ang1=ii*(2*pi)/(ops.mon_framerate/ops.cps);
                    if ops.squarewave==1
                        m1 = sign(sin(angsy(ang)*(sp1*2*pi*y)+angsx(ang)*(sp1*2*pi*x)+ang1)); 
                    else 
                        m1 = sin(angsy(ang)*(sp1*2*pi*y)+angsx(ang)*(sp1*2*pi*x)+ang1); 
                    end
                    tex(ang,ii)=Screen('MakeTexture', win, gray+((contrast*gray)*m1)); 
                end
            else
                m1 = sin(angsy(ang)*(sp1*2*pi*y)+angsx(ang)*(sp1*2*pi*x));

                if ops.squarewave
                    m1 = sign(m1);
                end
                tex(ang)=Screen('MakeTexture', win, gray+((contrast*gray)*m1));
            end

        end
    end
end



times = zeros(100,1);
tic;
for ii = 1:100
    Screen('FillRect', win, isi_color, rect);
    Screen('Flip',win);
    times(ii) = toc;
end



%%
% stim_times = cell(numel(ops.paradigm_sequence),1);
% stim_ang = cell(numel(ops.paradigm_sequence),1);
% stim_ctx_stdcount = cell(numel(ops.paradigm_sequence),1);
start_paradigm=GetSecs();
h = waitbar(0, 'initializeing...');
fprintf('Paradigm %d: %s, %d trials:\n',parad_num, ops.paradigm_sequence{parad_num}, ops.paradigm_trial_num(parad_num));
    
% check what paradigm
if strcmpi(ops.paradigm_sequence{parad_num}, 'control')
    cont_parad = 1;
else
    cont_parad = 0;
    if strcmpi(ops.paradigm_sequence{parad_num}, 'mmn')
        curr_MMN_pattern = ops.MMN_patterns(ops.paradigm_MMN_pattern(parad_num),:);
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'flip_mmn')
        curr_MMN_pattern = fliplr(ops.MMN_patterns(ops.paradigm_MMN_pattern(parad_num),:));
    else
        error('unknown paradigm type, line 140');          
    end 
    stdcounts = 0;
end

% stim_times{parad_num} = zeros(ops.paradigm_trial_num(parad_num),1);
% stim_ang{parad_num} = zeros(ops.paradigm_trial_num(parad_num),1);
% stim_ctx_stdcount{parad_num} = zeros(ops.paradigm_trial_num(parad_num),2);

% run trials
for trl=1:ops.paradigm_trial_num(parad_num)
    start_trial1 = GetSecs();

   %%% CREAT CONDITION FOR EACH DAY 

    waitbar(trl/ops.paradigm_trial_num(parad_num), h, sprintf('Paradigm %d of %d: Trial %d, angle %d',parad_num, numel(ops.paradigm_sequence), trl, ang));
    % pause for isi
    now=GetSecs();
    while (now-start_trial1)<(ops.isi_time)%+rand(1)/20)
        now=GetSecs();
    end
    Screen('Flip',win);

    % draw
    start_stim = GetSecs();
    now=GetSecs();
    ct = 0;
    while (now-start_stim)<ops.stim_time
        now=GetSecs();
        Screen('DrawTexture', win, tex(ang,rem(ct,30)+1));
        Screen('Flip',win);
%             session.outputSingleScan([vis_volt,0]);
        if ops.driftingGrating
            ct=ct+1;
        end
    end
%         session.outputSingleScan([0,0]);

    % reset screen
    Screen('FillRect', win, isi_color, rect);
    Screen('Flip',win);
    Screen('FillRect', win, isi_color, rect);
    Screen('Flip',win);

    % record times
     stim_times{parad_num}(trl) = start_stim-start_paradigm;
     stim_ang{parad_num}(trl) = ang;

    %fprintf('; Angle %d\n', ang);
end



close(h);

%% close all
% session.outputSingleScan([0,0]);
sca();

%% save info





savefilefull = strcat(pnm,fnm);
save(savefilefull,'parameters');