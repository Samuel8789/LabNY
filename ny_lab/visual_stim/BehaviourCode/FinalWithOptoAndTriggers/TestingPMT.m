% test grey transition betwen noise and grey and movies and grey
sca()
clear all
close all
desktop_testing=0;

acquisition_name='SessionC_timecount_fliptimes';
stim_dir=fullfile(fileparts(pwd),'AllenStimuli', 'SmallMonitor');

load(fullfile(stim_dir,'locally_sparse_noise_full_135.mat'))
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
ops.paradigm_trial_num =    [2880,   1,    10];   
ops.paradigm_stim_time=     [ 1/4, 30,    30];
ops.paradigm_isi_time=      [   0,   0,     0];
ops.paradigm_frame_number=  [   1,   900,   900];
ops.isicolor=127;      %appx middle                       % Shade of gray on screen during isi (1 if black, 255/2 if gray)
isi_color = [ ops.isicolor ops.isicolor ops.isicolor];
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
grey=135;

[win, rect] = Screen('OpenWindow',screenid, isi_color); % rect is the coordinates of the screen
ops.flipInterval = Screen('GetFlipInterval', win);
resolution=Screen('Resolution', screenid);
reswidth=resolution.width;
resheight=resolution.height;
topPriorityLevel = MaxPriority(win);

%% sampling the noise matrix

ncol = 8880;
splitrandperm=randperm(size(fullnoise,3));
Ar=fullnoise(:,:,splitrandperm);
clear fullnoise
noise1=Ar(:,:,1:2880);
noise2=Ar(:,:,2881:5760);
noise3=Ar(:,:,5761:8880);
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
clear noise1 noise2 noise3 

noiseindexes1=splitrandperm(1:2880)';
noiseindexes2=splitrandperm(2881:5760)';
noiseindexes3=splitrandperm(5761:8880)';


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

clear naturalmovie1big naturalmovie2big
ops.paradigm_frame_number=  [   1,   1,   frame_number_movie_1,  1,     1,   1,   frame_number_movie_2,   1,     1];

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

clear noiseindexes noiseindexes1 noiseindexes2 noiseindexes3 splitrandperm
waitforbuttonpress; 
startexp=GetSecs();
% Screen('FillRect', win, [0 0 127], rect);
% Screen('Flip',win,1,1);
% FIRST GRATING TIME 10min
for parad_num = 1:numel(ops.paradigm_sequence)
    if desktop_testing==0
        session.outputSingleScan(10); 
        pause(0.2)
        session.outputSingleScan(10);     
    end
    noise=0;
    movi=0;
 % check what paradigm
    if strcmpi(ops.paradigm_sequence{parad_num}, 'Noise1')
        texture=texnoise1   ; 
        noise=1;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Noise2')
        texture=texnoise2;
        noise=1;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Noise3')
        texture=texnoise3;
        noise=1;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Movie1')
        texture=texmov1;
        movi=1;
        frame_number=frame_number_movie_1;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Movie2')
        texture=texmov2;
        movi=1;
        frame_number=frame_number_movie_2;
    end    
    if desktop_testing==0
        session.outputSingleScan(0);
        session.outputSingleScan(0);
    end
    if noise
         for trl=1:ops.paradigm_trial_num(parad_num) 
         
                Screen('DrawTexture', win, texture(1,trl)); 
                Screen('Flip',win);
                stratstim=GetSecs(); 
                now=GetSecs(); 
                while (now- stratstim)<ops.paradigm_stim_time(parad_num)
                        now=GetSecs();
                end
                stratstim=GetSecs(); 
                now=GetSecs(); 
                while (now-stratstim)<ops.paradigm_stim_time(parad_num)
                    now=GetSecs();
                    Screen('FillRect', win, isi_color, rect);
                    Screen('Flip',win);
                end
         end     
    elseif movi 
          for trl=1:ops.paradigm_trial_num(parad_num)
                ct=0;
                starttrial=GetSecs(); 
                now=GetSecs(); 
                while (now-starttrial)<ops.paradigm_stim_time(parad_num)
                        now=GetSecs();
                        Screen('DrawTexture', win,texture(1,rem(ct,2*frame_number)+1), [],[],[],[],[], isi_color); 
                        Screen('Flip',win,1,1);
                        ct=ct+1;
                end     
             
          end
    end
    Screen('FillRect', win, isi_color, rect);
    Screen('Flip',win);    
 end

Screen('Close');  
sca();



