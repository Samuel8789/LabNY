sca;
close all;
clear;

load('drifiting_gratings_full.mat')
driftinggratings=all_warped_drifting_gratings_full;
clear all_warped_drifting_gratings_full 

% Here we call some default settings for setting up Psychtoolbox
PsychDefaultSetup(1);

% Get the screen numbers. This gives us a number for each of the screens
% attached to our computer.
screens = Screen('Screens');

% To draw we select the maximum of these numbers. So in a situation where we
% have two screens attached to our monitor we will draw to the external
% screen.
screenNumber = max(screens);
screenNumber=1
% Define black and white (white will be 1 and black 0). This is because
% in general luminace values are defined between 0 and 1 with 255 steps in
% between. All values in Psychtoolbox are defined between 0 and 1
white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);

% Do a simply calculation to calculate the luminance value for grey. This
% will be half the luminace values for white
grey = white / 2;

% Open an on screen window using PsychImaging and color it grey.
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, grey);
win=window
rect=windowRect
% Measure the vertical refresh rate of the monitor
ifi = Screen('GetFlipInterval', window);
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% Retreive the maximum priority number
topPriorityLevel = MaxPriority(window);

% Length of time and number of frames we will use for each drawing test
numSecs = 2;
numFrames = round(numSecs / ifi);

% Numer of frames to wait when specifying good timing. Note: the use of
% wait frames is to show a generalisable coding. For example, by using
% waitframes = 2 one would flip on every other frame. See the PTB
% documentation for details. In what follows we flip every frame.
waitframes = 1;

%-------------------------------------------------------------------------
repetitions=15;
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
            tex(i,j,k)=Screen('MakeTexture', window,0.8*driftinggratings(:,:,i,j,k)); 
         end
     end
 end
graytext=127+zeros(size(driftinggratings(:,:,i,j,k)));
graytextex=Screen('MakeTexture', window,graytext); 
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


texindexes=sampled_grating_indexes_parts(1,(~isnan(sampled_grating_indexes_parts(1,:))));
texture=driftingtex   ;    

%--------------------------------------------------------------------------
% NOTE: The aim in the following is to demonstrate how one might setup code
% to present a stimulus that changes on each frame. One would not for
% instance present a uniform screen of a fixed colour using this approach.
% The only reason I do this here to to make the code as simple as possible,
% and to avoid a screen which flickers a different colour every, say,
% 1/60th of a second. Virtually all the remianing demos show a stimulus
% which changes on each frame, so I want to show an approach which will
% generalise to the rest of the demos. Therefore, one would clearly not
% write a script in this form for an experiment; it is to demonstrate
% principles.
%
% Specifically,
%
% vbl + (waitframes - 0.5) * ifi
%
% is the same as
%
% vbl + 0.5 * ifi
%
% As here waitframes is set to 1 (i.e. (1 - 0.5) == 0.5)
%
% For discussion see PTB forum thread 20178 for discussion.
%
%--------------------------------------------------------------------------

%------------
% EXAMPLE #1
%------------

% First we will demonstrate a poor way in which to get good timing of
% visually presented stimuli. We generally use this way of presenting in
% the demos in order to allow the demos to run on potentially defective
% hardware. In this way of presenting we leave much to chance as regards
% when our stimuli get to the screen, so it is not reccomended that you use
% this approach.
for frame = 1:numFrames

    % Color the screen grey
    Screen('FillRect', window, grey);
    Screen('DrawTexture', window, texture(texindexes(1),rem(frame,60)+1));
    % Flip to the screen
    Screen('Flip', window);

end



%------------
% EXAMPLE #2
%------------

% Here we now specify a time at which PTB should be ready to draw to the
% screen by. In this example we use half a inter-frame interval. This
% specification allows us to get an accurate idea of whether PTB is making
% the stimulus timings we want.
%%
numSecsISI = 1;
isiframes = round(numSecsISI / ifi);



    [VBLTimestamp StimulusOnsetTime FlipTimestamp Missed Beampos] = Screen('Flip', window);
    for frame = 1:numFrames

        % Color the screen red
    %     Screen('FillRect', window, [0.5 0 0]);
        Screen('DrawTexture', window, tex(1,rem(frame,30)+1));

        % Flip to the screen
        [VBLTimestamp StimulusOnsetTime FlipTimestamp Missed Beampos] = Screen('Flip', window, VBLTimestamp + (waitframes - 0.5) * ifi, 2);

    end


        Screen('FillRect', window, black);
        Screen('Flip',window)

    
    

 %%

[VBLTimestamp StimulusOnsetTime FlipTimestamp Missed Beampos] = Screen('Flip', window, VBLTimestamp + (waitframes - 0.5) * ifi, 1,1);

% [VBLTimestamp StimulusOnsetTime FlipTimestamp Missed Beampos] = Screen('Flip', window, VBLTimestamp + (waitframes - 0.5) * ifi, 1,1);

  ct = 0;
        while (now-start_stim)<ops.stim_time
            now=GetSecs();
            Screen('DrawTexture', win, tex(ang,rem(ct,30)+1));
            Screen('Flip',win);
%              session.outputSingleScan(vis_volt);
            if ops.driftingGrating
                ct=ct+1;
            end
        end

%%

%------------
% EXAMPLE #3
%------------

% Here we do exactly the same as the second example, but we additionally
% first set the PTB prority level to maximum. This means PTB will take
% processing priority over other system and applicaiton processes. I
% normally do this before and after stimulus presentation, however, on
% modern multi-core processors keeping Priority on is unlikely to overload
% system resources. Plus, on Linux this operation can take much much longer
% then on Windows and OSX (up to minutes in some use cases). So it is now
% suggested that you set Priority once at the start of a script after
% setting up your onscreen window.
% See PTB forum thread 20178 (and those linked to it) for discussion.
Priority(topPriorityLevel);
vbl = Screen('Flip', window);
for frame = 1:numFrames

    % Color the screen purple
    Screen('FillRect', window, [0.5 0 0.5]);

    % Flip to the screen
    vbl = Screen('Flip', window, vbl + (waitframes - 0.5) * ifi);

end
Priority(0);


%------------
% EXAMPLE #4
%------------

% Finally we do the same as the last example except now we additionally
% tell PTB that no more drawing commands will be given between coloring the
% screen and the flip command. This, can help acheive good timing when one
% is needing to do additional non-PTB processing between setting up drawing
% and flipping to the screen. Thus, you would only use this technique if
% you were doing this. So, if you are not, go with example #3
Priority(topPriorityLevel);
vbl = Screen('Flip', window);
for frame = 1:numFrames

    % Color the screen blue
    Screen('FillRect', window, [0 0 0.5]);

    % Tell PTB no more drawing commands will be issued until the next flip
    Screen('DrawingFinished', window);

    % One would do some additional stuff here to make the use of
    % "DrawingFinished" meaningful / useful

    % Flip to the screen
    vbl = Screen('Flip', window, vbl + (waitframes - 0.5) * ifi);

end
Priority(0);

% Clear the screen.