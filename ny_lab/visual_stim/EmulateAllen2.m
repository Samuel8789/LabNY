sca()
clear all
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


%% create stim
for designstim=1
    isi_color = [ops.isicolor ops.isicolor ops.isicolor];
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
                    for idx=1:length(ops.cpss)
                        ops.cps=ops.cpss(idx);
                        for ii=1:60
                            ang1=ii*(2*pi)/(ops.mon_framerate/ops.cps);
                            m1 = sin(angsy(ang)*(sp1*2*pi*y)+angsx(ang)*(sp1*2*pi*x)+ang1);                           
                            tex(ang,ii,idx)=Screen('MakeTexture', win, gray+((contrast*gray)*m1)); 
                        end
                    end
               
                end
                
            end
        end
    end
end
%% 

totaltrialtypes=numel(ops.angs_rad)*numel(ops.cpss);

fullidx=zeros(totaltrialtypes);
if length(ops.cpss)>1
    habituation_angles=cat(1,tex(:,:,1),tex(:,:,2),tex(:,:,3),tex(:,:,4),tex(:,:,5));
else 
    habituation_angles=tex;
end   
resp=zeros(totaltrialtypes,repetitions);
for i=1:repetitions
    resp(:,i)=randperm(totaltrialtypes);
end

msize = numel(resp);
allindx=resp(randperm(msize, msize));
n = 3;
indxinparts = reshape(allindx', [], n);
a =1;
b = 10;
numbersweps=ceil(length(indxinparts)/20);

y = ceil(a.*randn(numbersweps,1) + b);
y2 = ceil(a.*randn(numbersweps,1) + b);
y3= ceil(a.*randn(numbersweps,1) + b);


indixfirst=indxinparts(:,1);
indixsecond=indxinparts(:,2);
indixthird=indxinparts(:,3);
ct=0;
for i=1:numbersweps
    indixfirst=[indixfirst(1:ct+y(i)); 0; indixfirst(ct+y(i)+1:end)]    ;  
    indixsecond=[indixsecond(1:ct+y2(i)); 0 ;indixsecond(ct+y2(i)+1:end)] ; 
    indixthird=[indixthird(1:ct+y3(i)); 0; indixthird(ct+y3(i)+1:end)] ; 
    ct=ct+20;
end


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

stim_times = cell(numel(ops.paradigm_sequence),1);
end_stim_times = cell(numel(ops.paradigm_sequence),1);
framestarttimes= cell(numel(ops.paradigm_sequence),1);
frameendtimes= cell(numel(ops.paradigm_sequence),1);
stim_ang = cell(numel(ops.paradigm_sequence),1);


startexp=GetSecs();
h = waitbar(0, 'initializing...');
z= waitbar(0, 'initializing...');
%% FIRST GRATING TIME 10min
for parad_num = 1:numel(ops.paradigm_sequence)
     start_paradigm=GetSecs();
     movi=0;
 % check what paradigm
    stim_ang{parad_num} = zeros(ops.paradigm_trial_num(parad_num),1);

    if strcmpi(ops.paradigm_sequence{parad_num}, 'Drifting1')
        indx=indixfirst;
        texture=habituation_angles   ;    
        grating=1;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Drifting2')
        indx=indixsecond;
        texture=habituation_angles;
                grating=1;

    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Drifting3')
        indx=indixthird;
        texture=habituation_angles;
                grating=1;

    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Movie3')
        texture=texmov2;
        movi=1;
    elseif strcmpi(ops.paradigm_sequence{parad_num}, 'Movie1')
        texture=texmov1;
        movi=1;
    end    

    stim_times{parad_num} = zeros(ops.paradigm_trial_num(parad_num),1);
    end_stim_times{parad_num} = zeros(ops.paradigm_trial_num(parad_num),1);
    framestarttimes{parad_num}= zeros(120,ops.paradigm_trial_num(parad_num));
    frameendtimes{parad_num}= zeros(120,ops.paradigm_trial_num(parad_num));

    if movi
        stim_ang{parad_num} = zeros(length(texture),1);
        framestarttimes{parad_num}= zeros(length(texture),ops.paradigm_trial_num(parad_num));
        frameendtimes{parad_num}= zeros(length(texture),ops.paradigm_trial_num(parad_num));
    end
  % run trials
    for trl=1:ops.paradigm_trial_num(parad_num)
       start_trial1 = GetSecs();
     
       waitbar(trl/ops.paradigm_trial_num(parad_num), h, sprintf('Paradigm %d of %d: Trial %d, angle %d',parad_num, numel(ops.paradigm_sequence), trl, indx(trl)));
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
        if ~movi
            counter=1;
            while (now-start_stim)<ops.paradigm_stim_time(parad_num)
                now=GetSecs();
                framestarttimes{parad_num}(counter,trl)= now;
                if grating
                    if indx(trl)~=0
%                         vis_volt=indx(trl)/size(texture,1)*4;

                        Screen('DrawTexture', win, texture(indx(trl),rem(ct,60)+1));
                        Screen('Flip',win);
            %             session.outputSingleScan([vis_volt,0]);
                        if ops.driftingGrating
                            ct=ct+1;
                        end
                    else
                        vis_volt=5;
%                       session.outputSingleScan([vis_volt,0]);
                        Screen('FillRect', win, isi_color, rect);
                        Screen('Flip',win, 1,1);  
                        
                    end
                    
                else
                    Screen('FillRect', win, isi_color, rect);
                    Screen('Flip',win, 1,1);
                    indx(trl)=1;
   
                end
                frameendtimes{parad_num}(counter,trl)= GetSecs();
                counter=counter+1;

            end   
            endstim=GetSecs();
            
            Screen('FillRect', win, 0, rect);
            Screen('Flip',win);
        elseif movi  
            ct = 1;
%              while (now-start_stim)<ops.paradigm_stim_time(parad_num)
             while ct<=length(texture)
                    waitbar(ct/length(texture), z, sprintf('Paradigm %d of %d: Frame %d',parad_num, numel(ops.paradigm_sequence), ct));

                    now=GetSecs();
                    framestarttimes{parad_num}(ct,trl)= now-start_stim;
                    Screen('DrawTexture', win, texture(1,ct));
                    Screen('Flip',win);
%                       session.outputSingleScan([vis_volt,0]);

                    frameendtimes{parad_num}(ct,trl)= GetSecs()-start_stim;
                    ct=ct+1;
                 	
             end   
             endstim=GetSecs();
             indx(trl)=1;
        end
%             session.outputSingleScan([0,0]);
        % reset screen
%         Screen('FillRect', win, 0, rect);
%         Screen('Flip',win);
        % record
        stim_times{parad_num}(trl) = start_stim-start_paradigm;
        stim_ang{parad_num}(trl) = indx(trl);
        end_stim_times{parad_num}(trl) = endstim-start_paradigm;


    end
    grating=0;
    movi=0;
    [keyIsDown,secs,keyCode] = KbCheck;
    if keyCode(27); break; end
 end

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
