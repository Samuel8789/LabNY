%%
clear
 desktop_testing=1;

%% parameters
% ------ Stim params ------

% ------ Stim params ------

% ------ Paradigm sequence ------
% ops.paradigm_sequence = {'Spont', 'SingleGrsting', 'RandomGrating','TuningGratin'};     % 3 options {'Control', 'MMN', 'flip_MMN'}, concatenate as many as you want
ops.paradigm_sequence = {'Spont', 'Habituation'};  


ops.paradigm_trial_num =[1,5] ;                                                       % 1= horz/vert; 2= 45deg;
ops.paradigm_stim_time=[600,100];
ops.paradigm_isi_time=[0,30];
% ------ Stim params ------
ops.squarewave=0;                                           % do squarewaves instead of sinewaves
ops.isicolor=135;      %appx middle                         % Shade of gray on screen during isi (1 if black, 255/2 if gray)
ops.ctrsts=[.015625 .03125 .0625 .125 .25 .5 .80 1];          % Range of contrasts to use
ops.ctrst = 7;                                                  % pick 
ops.spfrqs=[.01  .02 .04 .05 .08 .16 .32];                      % Range of spatial freqs to use
ops.spfrq = 3;                                                  % pick 
ops.angs_rad = pi*(0:7)/  8;                                  % Orientations to use, in rad
% ------ Moving Gratings ------
ops.driftingGrating = 1;                                    % use if you want to make it drifting grating
if ops.driftingGrating
    ops.angs_rad = 2 * pi*(0:7)/8;
    ops.cps=2;   %movement velocity. cycles per secondo         
    ops.cpss=[1,2,4,8,15];   %movement velocity. cycles per secondo         
end
ops.angs_deg=radtodeg(ops.angs_rad);

%% gui selector and previous dataloading


%%

    ops.paradigm_sequence = {'Spont', 'Habituation', 'TuningGrating'};  
    ops.paradigm_trial_num =[1,5,80] ;                                                      
    ops.paradigm_stim_time=[600,100,5];
    ops.paradigm_isi_time=[0,30,5];
    tuning_angles=zeros(0);
    for i=1:ops.paradigm_trial_num(3)/8
        tuning_angles=horzcat(tuning_angles,randperm(8));
    end
  
   %% predicted run time
run_time=0
for i = 1:numel(ops.paradigm_sequence)
    run_time=run_time+((ops.paradigm_stim_time(i)+ops.paradigm_isi_time(i)+0.025)*ops.paradigm_trial_num(i));

end

fprintf('Expected run duration: %.1fmin (%.fsec)\n',run_time/60,run_time);
%% initialize
Screen('Preference', 'SkipSyncTests', 1);
AssertOpenGL; % Make sure this is running on OpenGL Psychtoolbox:
screenid = max(Screen('Screens')); % Choose screen with maximum id - the secondary display on a dual-display setup for display
if desktop_testing
    screenid=1;
end
[win, rect] = Screen('OpenWindow',screenid, [ops.isicolor ops.isicolor ops.isicolor]); % rect is the coordinates of the screen
ops.flipInterval = Screen('GetFlipInterval', win);
%%
if ~desktop_testing
    session=daq.createSession('ni');
    session.addAnalogOutputChannel('Dev1','ao0','Voltage');
    session.IsContinuous = true;
    session.Rate = 10000;
    session.outputSingleScan([5]);
end


%% create stim
for designstim=1
    isi_color = [0 0 ops.isicolor];
    tex = zeros(numel(ops.angs_rad)*numel(ops.ctrst)*numel(ops.spfrq),30,numel(ops.cpss));
    angsy = sin(ops.angs_rad);
    angsx = cos(ops.angs_rad);
    for cc=ops.ctrst %contrast (out of 7)
        contrast=ops.ctrsts(cc);
        white = WhiteIndex(win); % pixel value for white
        black = BlackIndex(win); % pixel value for black
        gray = (white+black)/2;
        inc = white-gray;
        for s=ops.spfrqs(ops.spfrq) %determine spatial frequency
            scrsz = rect;
            [x,y] = meshgrid((-scrsz(3)/2)+1:(scrsz(3)/2)-1, (-scrsz(4)/2)+1:(scrsz(4)/2)-1);
            sp1=(.5799/10.2)*s; %10.2 is just some scaling factor that i calibrated. do not change unless you know what youre doing!
            for ang=1:numel(ops.angs_rad)  
                if ops.driftingGrating
                    ops.mon_framerate=60;
                    for idx=(1:length(ops.cpss))
                        ops.cps=ops.cpss(idx);
                        for ii=1:30
                            ang1=ii*(2*pi)/(ops.mon_framerate/ops.cps);
                            if ops.squarewave==1
                                m1 = sign(sin(angsy(ang)*(sp1*2*pi*y)+angsx(ang)*(sp1*2*pi*x)+ang1)); 
                            else 
                                m1 = sin(angsy(ang)*(sp1*2*pi*y)+angsx(ang)*(sp1*2*pi*x)+ang1);                           
                            end
                            tex(ang,ii,idx)=Screen('MakeTexture', win, gray+((contrast*gray)*m1)); 
                        end
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
end


times = zeros(100,1);
tic;
for ii = 1:100
    Screen('FillRect', win, [ops.isicolor ops.isicolor ops.isicolor], rect);
    Screen('Flip',win);
    times(ii) = toc;
end
%% select angles
rng(0,'twister');
if expday==1 
    ang_habit = ceil(rand * size(tex,1))  ;  
    habituation_angles=repelem(ang_habit,5);
elseif strcmpi(treatment, 'Habituation')
    ang_habit=day1angle  ;
    habituation_angles=repelem(ang_habit,5);
elseif strcmpi(treatment, 'Control')    
    if expday==7
        ang_habit=day1angle;
        habituation_angles=repelem(ang_habit,5);
    else
        habituation_angles=zeros(5);
        for i=1:5
            habituation_angles(i)= ceil(rand * size(tex,1));
        end
    end
end

%%
stim_times = cell(numel(ops.paradigm_sequence),1);
stim_ang = cell(numel(ops.paradigm_sequence),1);
start_paradigm=GetSecs();
h = waitbar(0, 'initializeing...');
for parad_num = 1:numel(ops.paradigm_sequence)
    fprintf('Paradigm %d: %s, %d trials:\n',parad_num, ops.paradigm_sequence{parad_num}, ops.paradigm_trial_num(parad_num));
    stim_times{parad_num} = zeros(ops.paradigm_trial_num(parad_num),1);
    stim_ang{parad_num} = zeros(ops.paradigm_trial_num(parad_num),1);   
    if strcmpi(ops.paradigm_sequence{parad_num}, 'TuningGrating')
        angles=tuning_angles
    elseif  strcmpi(ops.paradigm_sequence{parad_num}, 'Habituation')
        angles=habituation_angles
    end
    
    
    if strcmpi(ops.paradigm_sequence{parad_num}, 'Spont')
        vis_volt=1;
        trl=1;
        start_trial1 = GetSecs();
        waitbar(trl/ops.paradigm_trial_num(parad_num), h, sprintf('Paradigm %d of %d: Trial %d,',parad_num, numel(ops.paradigm_sequence), trl));
        % pause for isi
        now=GetSecs();        
        while (now-start_trial1)<(ops.paradigm_stim_time(parad_num)+rand(1)/20)
            now=GetSecs();
            if ~desktop_testing
                session.outputSingleScan([vis_volt]);
            end
        end
        if ~desktop_testing
            session.outputSingleScan([0]);
        end
        Screen('Flip',win);
        stim_times{parad_num}(trl) = start_trial1-start_paradigm;       
        stim_ang{parad_num}(trl) = 0;
      
    
    else
        
        for trl=1:ops.paradigm_trial_num(parad_num)          

            start_trial1 = GetSecs();
            ang=angles(trl);
            vis_volt = ang/size(tex,1)*4;
            waitbar(trl/ops.paradigm_trial_num(parad_num), h, sprintf('Paradigm %d of %d: Trial %d, angle %d',parad_num, numel(ops.paradigm_sequence), trl, ang));
            % pause for isi
            now=GetSecs();
            while (now-start_trial1)<(ops.paradigm_isi_time(parad_num)+rand(1)/20)
                now=GetSecs();
            end
            Screen('Flip',win);

            % draw
            start_stim = GetSecs();
            now=GetSecs();
            ct = 0;
            while (now-start_stim)<ops.paradigm_stim_time(parad_num)
                now=GetSecs();
                Screen('DrawTexture', win, tex(ang,rem(ct,30)+1));
                Screen('Flip',win);
                if ~desktop_testing
                    session.outputSingleScan([vis_volt]);
                end
                if ops.driftingGrating
                    ct=ct+1;
                end
            end
            if ~desktop_testing
                session.outputSingleScan([0]);
            end

            % reset screen
            Screen('FillRect', win, [ops.isicolor ops.isicolor ops.isicolor], rect);
            Screen('Flip',win);
            Screen('FillRect', win, [ops.isicolor ops.isicolor ops.isicolor], rect);
            Screen('Flip',win);

            % record
            stim_times{parad_num}(trl) = start_stim-start_paradigm;
            stim_ang{parad_num}(trl) = ang;

            fprintf('; Angle %d\n', ang);
        end
    end
end
%%
now=GetSecs();
final_time=now-start_paradigm

fprintf('Saving...\n');
save([pnm, fnm, '.mat'],'ops', 'stim_times', 'stim_ang','final_time');
fprintf('Done\n');        
         

sca();
