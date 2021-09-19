sca()
clear all
close all
desktop_testing=1;
 load('mov.mat');
 mov1=double(permute(z,[2,3,1]));
% mov2=double(permute(y,[2,3,1]));
% x=1:304;
% y=1:608;
% z=1:900;
% zz=1:3600;
% [X,Y,Z]=meshgrid(y, x, z);
% [XX,YY,ZZ]=meshgrid(y, x, zz);
% x2=1:304;
% y2=1:608;
% z2=0.5:0.5:900;
% zz2=0.5:0.5:3600;
% [Xq,Yq,Zq] = meshgrid(y2,x2,z2);
% [Xxq,Yyq,ZZq] = meshgrid(y2,x2,zz2);
% 
% naturalmovie1 = interp3(X,Y,Z,mov1,Xq,Yq,Zq);
% naturalmovie2 = interp3(XX,YY,ZZ,mov2,Xxq,Yyq,ZZq);
% 
% naturalmovie1big=imresize(uint8(naturalmovie1), [1024 1280]);
% naturalmovie2big=imresize(uint8(naturalmovie2), [1024 1280]);
load('nATURALmOVIE1.mat')
load('naturalmovie3.mat')
naturalmovie1big=naturalmovie1big(:,:,2:end);
naturalmovie2big=naturalmovie2big(:,:,2:end);

%% parameters
% ------ Stim params ------

% ------ Stim params ------

% ------ Paradigm sequence ------
% ops.paradigm_sequence = {'Spont', 'SingleGrsting', 'RandomGrating','TuningGratin'};     % 3 options {'Control', 'MMN', 'flip_MMN'}, concatenate as many as you want
ops.paradigm_sequence = {'Drifting1', 'Intergrey','Movie3', 'Intergrey','Movie1','Drifting2' , 'Spont','Movie3','Intergrey','Drifting3'};  
% ops.paradigm_sequence = {'Movie3'};  

ops.paradigm_trial_num =[210,1,5,1,10,210,1,5,1,210] ;   
ops.paradigm_stim_time=[2,30,120,30,30,2,300,120,30,2];
ops.paradigm_isi_time=[1,0,0,0,0,1,0,0,0,1];

% ------ Stim params ------
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
    ops.cpss=[1,2,4,8,15]; 
%     ops.cpss=[4];%movement velocity. cycles per secondo         
end
ops.angs_deg=rad2deg(ops.angs_rad);
repetitions=15;

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
texmov1=zeros(1,length(naturalmovie1big));
for i=1:length(naturalmovie1big)
    texmov1(i)=Screen('MakeTexture', win, naturalmovie1big(:,:,i)); 
end
texmov2=zeros(1,length(naturalmovie2big));
for i=1:length(naturalmovie2big)
    texmov2(i)=Screen('MakeTexture', win, naturalmovie2big(:,:,i)); 
end
%%

stim_times = cell(1,1);
end_stim_times = cell(1,1);
framestarttimes= cell(1,1);
frameendtimes= cell(1,1);
stim_ang = cell(1,1);


startexp=GetSecs();
h = waitbar(0, 'initializing...');
z= waitbar(0, 'initializing...');
%% FIRST GRATING TIME 10min
d=1;
c=0;
if d
    texture=texmov2;
elseif c
    texture=texmov1;
end    

stim_times{1} = zeros(1,1);
end_stim_times{1} = zeros(1,1);
stim_ang{1} = zeros(length(texture),1);
framestarttimes{1}= zeros(length(texture),1);
frameendtimes{1}= zeros(length(texture),1);


% run trials
start_paradigm=GetSecs();

start_trial1 = GetSecs();

    % draw
start_stim = GetSecs();
now=GetSecs();
ct = 1;
%              while (now-start_stim)<ops.paradigm_stim_time(parad_num)
while ct<=length(texture)

    now=GetSecs();
    framestarttimes{1}(ct,1)= now-start_stim;
    Screen('DrawTexture', win, texture(1,ct));
    Screen('Flip',win);
%                       session.outputSingleScan([vis_volt,0]);

    frameendtimes{1}(ct,1)= GetSecs()-start_stim;
    ct=ct+1;

end   
 endstim=GetSecs();
%             session.outputSingleScan([0,0]);
% reset screen
%         Screen('FillRect', win, 0, rect);
%         Screen('Flip',win);
% record
stim_times{1}(1) = start_stim-start_paradigm;
end_stim_times{1}(1) = endstim-start_paradigm;



 

Screen('Close');  
close(h);
%%
% IF_pause_synch(10, session, ops.synch_pulse)

%% close all
% session.outputSingleScan([0,0]);
sca();

% %% save info
% fprintf('Saving...\n');
% save([pnm, fnm, '.mat'],'ops', 'stim_times', 'stim_ang','final_time');
% fprintf('Done\n');        
%          
% 
% sca();

plot((frameendtimes{1,1}-framestarttimes{1,1})*1000);
mean((frameendtimes{1,1}-framestarttimes{1,1})*1000);
std((frameendtimes{1,1}-framestarttimes{1,1})*1000);
%%
pwd2 = fileparts(which('EmulateAllen2.m')); %mfilename
save_path = [pwd2];

temp_time = clock;
file_name = sprintf('AllenSessionA_%d_%d_%d_stim_data_%dh_%dm.mat',temp_time(2), temp_time(3), temp_time(1)-2000, temp_time(4), temp_time(5));
clear temp_time;


%% save info
fprintf('Saving...\n');
% save([save_path, '\',file_name, '.mat'],'ops', 'stim_times', 'stim_ang', ,'end_times');
save([save_path,'\', file_name, '.mat'],'ops', 'stim_times' ,'end_stim_times','framestarttimes','frameendtimes','stim_ang');

fprintf('Done\n');



close all
for paradigm=1:10
    figure(paradigm)
    plot((frameendtimes{paradigm,1}-framestarttimes{paradigm,1})*1000);
%     plot(stim_times{paradigm})
end
%%

mean(mean(mov1,[1,2]))
mean(mean(naturalmovie1big(:,:,2:end),[1,2]))
figure(1)
plot(squeeze(mean(mov1,[1,2])))
figure(2)
plot(squeeze(mean(naturalmovie1big(:,:,2:end),[1,2])))

noramlizemov1=zeros(1024,1280,1800);
for i=1:1800
   noramlizemov1(:,:,i) = rescale(naturalmovie1big(:,:,i),0,255);
end
figure(4)
plot(squeeze(mean(noramlizemov1(:,:,2:end),[1,2])))
mean(mean(noramlizemov1(:,:,2:end),[1,2]))
implay(uint8(noramlizemov1))

noramlizemov2=zeros(304,608,900);
for i=1:900
   noramlizemov2(:,:,i) = rescale(mov1(:,:,i),0,255);
end
figure(3)
plot(squeeze(mean(noramlizemov2,[1,2])))
implay(uint8(noramlizemov2))
mean(mean(noramlizemov2,[1,2]))
