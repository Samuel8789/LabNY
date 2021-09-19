function Habituation_SineGrating(tempfreq,spfreq,sduration,iduration,nsessions)
Screen('Preference', 'SkipSyncTests', 0)
% tempfreq=2;
% spfreq=0.05;
% sduration=100;
% iduration=30;
% nsessions=5;
% treatment='Habituation';
% % treatment='Control';
% expday=5;
%Habituation_SineGrating(2,0.05,100,30,5)

WarmUp=500;
Habit=10;
PreStim=2;


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
lum_center = 138;
AssertOpenGL; % Make sure this is running on OpenGL Psychtoolbox:
%screenid = max(Screen('Screens')); % Choose screen with maximum id - the secondary display on a dual-display setup for display
screenid=2;
x = ((0:255).^(1/2.2))'; % gamma correction
table = repmat(x/max(x),1,3);
Screen('LoadNormalizedGammaTable',screenid,table);

resolution = Screen('Resolution', screenid); % # of pixels
resheight = resolution.height;
reswidth = resolution.width;
con=1;
%stimwidth = 15*(resolution.width/16); %full field (with 1/16 of the screen blank for the diode)
stimwidth = resolution.width;
stimheight = resolution.height;
stimwindow = [stimwidth stimheight];

% create data acquisition session
  session = daq.createSession ('ni');
  session.addAnalogOutputChannel('Dev1','ao0','Voltage'); % visual stim
%  session.IsContinuous = true;
%  session.Rate = 10000;
  session.outputSingleScan(0); % create output value and output a single scan

cpd = .2967; % cm per degree when distance between eye and monitor is 15 cm i.e. tand.5 x 15 x 2 =  0.2618
spfreq_new = spfreq/(cpd*reswidth/(37.5)); % convert # of cycles per degree to # of cycles per pixel; 40.64 is the screen width in cm 
rotateMode = kPsychUseTextureMatrixForRotation;
angles = [0 45 90 135 180 225 270 315];
contrasts = 1; %[0.08 0.16 0.32 0.64 0.8 1];
all_combs = combn(1:max([length(angles),length(contrasts)]),2);
all_combs(length(contrasts)*length(angles)+1:end,:) = [];
if length(angles) > length(contrasts)
    all_combs = fliplr(all_combs);
end

comb_all_sessions = zeros(size(all_combs,1),nsessions);
for i = 1:nsessions
    comb_all_sessions(:,i) = randperm(length(contrasts)*length(angles))';
end

amplitude = .5;
phase = 0; % Phase is the phase shift in degrees (0-360 etc.) applied to the sine grating:
win = Screen('OpenWindow', screenid, lum_center); % Open a fullscreen onscreen window on that display, choose a background color of 128 = gray, i.e. 50% max intensity
AssertGLSL; % Make sure the GLSL shading language is supported
ifi = Screen('GetFlipInterval', win, lum_center); % Retrieve video redraw interval for later control of our animation timing
phaseincrement = (tempfreq * 360) * ifi; % Compute increment of phase shift per redraw

offset = lum_center/255;




if expday<=0 
    Screen('FillRect', win, lum_center);
    Screen('Flip', win);


    tic
   % h = waitbar(0,'0','Name','Warming Up', 'Position', [500 500 270 100]);
    while (toc<=WarmUp) % wait 50 seconds for the screen to adjust
        steps = WarmUp;
        tim = toc;
        %waitbar(toc / steps,h, sprintf('%d',round(toc)));
        [keyIsDown,secs,keyCode] = KbCheck;
        if keyCode(27); break; end
    end
    %close(h);

    Screen('FillRect', win, lum_center);
    Screen('Flip', win);


    tic
   % h = waitbar(0,'0','Name','Pre Stimulus', 'Position', [500 500 270 100]);
    while (toc<=Habit)
        steps = Habit;% 10 MIN PRE STIMULS PERIOD
        tim = toc;
        %waitbar(toc / steps,h, sprintf('%d',round(toc)));
        [keyIsDown,secs,keyCode] = KbCheck;
        if keyCode(27); break; end
    end
   % close(h);    
    
    parameters.spfreq = 0;
    parameters.tempfreq = 0;
    parameters.sduration = Habit;
    parameters.warmup = WarmUp;
    parameters.prestim = 0;
    parameters.iduration = 0;
    parameters.nsessions = 0;
    parameters.treatment = treatment;
    parameters.expday =expday;
    parameters.mousecode = MouseInfo{1};
    
       
else if expday>0    
    
        
    Screen('FillRect', win, lum_center);
    Screen('Flip', win);    
        
    
    tic
    %h = waitbar(0,'0','Name','Warming Up', 'Position', [500 500 270 100]);
    while (toc<=WarmUp) % wait 50 seconds for the screen to adjust
        steps = WarmUp;
        tim = toc;
       % waitbar(toc / steps,h, sprintf('%d',round(toc)));
        [keyIsDown,secs,keyCode] = KbCheck;
        if keyCode(27); break; end
    end
    %close(h);

    Screen('FillRect', win, lum_center);
    Screen('Flip', win);


    tic
   % h = waitbar(0,'0','Name','Pre Stimulus', 'Position', [500 500 270 100]);
    while (toc<=PreStim)
        steps = PreStim;% 10 MIN PRE STIMULS PERIOD
        tim = toc;
       % waitbar(toc / steps,h, sprintf('%d',round(toc)));
        [keyIsDown,secs,keyCode] = KbCheck;
        if keyCode(27); break; end
    end
    %close(h);

    ori=zeros(1,5);
    if strcmp(treatment, 'Habituation') & expday==1 
        ori(:)=randsample(angles,1);
    elseif strcmp(treatment, 'Habituation') & expday~=1
%         [indx,~] = listdlg('ListString',string(angles),'PromptString','Select a Color')
%         ori(:) = angles(indx);
           ori(:) = Yester.parameters.sequence;
    elseif strcmp(treatment, 'Control') & expday==1|expday==7
%         [indx,~] = listdlg('ListString',string(angles),'PromptString','Select a Color')
%         ori(:) = angles(indx);
          ori(:) = Tody.parameters.sequence;
                  
    elseif strcmp(treatment, 'Control') & expday~=1&expday~=7
        ori=randsample(angles,5)
    end

    sequence = nan(length(ori),1); %initialize the sequence array


    abort = 0; count = 0;


    for j = 1:nsessions % loop over orientations
        disp(['Session ',int2str(j),'/',int2str(nsessions)])
        count = count+1;
        sequence(count,1) = ori(count);
        ang=ori(count)
        gratingtex = CreateProceduralSineGrating(win, stimwindow(1), stimwindow(2), offset*[1,1,1,0], [], con);
        Screen('FillRect', win, lum_center);
        Screen('Flip', win);

        tic
       % h = waitbar(0,'0','Name','ISI', 'Position', [2500 500 270 100]);
        while (toc<=iduration)
            tim = toc;
            %waitbar(toc / iduration,h, sprintf('%d',round(toc)));
        end
        %close(h);

        [keyIsDown,secs,keyCode] = KbCheck;
        if keyCode(27); abort = 1; break; end

        tic
        %h = waitbar(0,'0','Name','Stim', 'Position', [2500 500 270 100]);
        tim = Screen('Flip', win);
        vblStime = tim + sduration;
        while (tim<vblStime)
            phase = phase + phaseincrement; % Increment phase by 1 degree
            Screen('DrawTexture', win, gratingtex, [], [1 1 stimwidth stimheight], ang, [], [], [], [], rotateMode, [phase, spfreq_new, amplitude, 0])

            tim = Screen('Flip', win, tim + 0.5 * ifi); % Show it at next retrace
             session.outputSingleScan((ang/360)+1); % create output value and output a single scan
             %waitbar(toc / sduration,h, sprintf('%d',round(toc)));
        end
        %close(h);
             session.outputSingleScan(0); % create output value and output a single scan
        [keyIsDown,secs,keyCode] = KbCheck;
        if keyCode(27); abort = 1; break; end

        if abort==1 ; break ; end  
        Screen('FillRect', win, 0);
        Screen('Flip', win);
    end
    parameters.spfreq = spfreq;
    parameters.tempfreq = tempfreq;
    parameters.sduration = sduration;
    parameters.iduration = iduration;
    parameters.warmup = WarmUp;
    parameters.prestim = PreStim;
    parameters.nsessions = nsessions;
    parameters.sequence = sequence;
    parameters.treatment = treatment;
    parameters.expday =expday;
    parameters.mousecode = MouseInfo{1};
    end
end
% Close the window. This will also release all other ressources:
Screen('CloseAll');
savefilefull = strcat(pnm,fnm);
save(savefilefull,'parameters');

return;
