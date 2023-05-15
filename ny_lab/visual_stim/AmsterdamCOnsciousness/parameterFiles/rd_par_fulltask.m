function Par = rd_par_fulltask
Par = struct; %structure with parameters

%% General variable definitions
Par.dblVersion          = 1.0; % version of the output file
Par.strTask             = 'rd_fullx';

%% Keypress parameters
Par.strExitKey          = 'F2';
Par.strRewardKeyLeft    = 'F3';
Par.strRewardKeyRight   = 'F4';

%% Servo Parameters:
Par.UseServo            = 0;

%% Piezo Parameters:
Par.UsePiezo            = 0;

%% Screen Parameters
Par.intFrameRate                = 60;
Par.dblScreenDistance_cm        = 25; % cm;  
Par.dblScreenWidth_cm           = 36.1; % cm;
Par.dblScreenHeight_cm          = 20.3; % cm;

%% Visual Stimulus Parameters
Par.strVisStimType              = 'fs_grating';
Par.str90Deg                    = '0 degrees is leftward motion; 90 degrees is upward motion';
Par.dblSpeed                    = 1.5;  %TF of moving grating in cycles/sec in glickfeld 1-2 for maximal RL, 1.5 for optomotor umino2008
Par.dblSpatFreq                 = 0.05; %SF of moving gratings in cycles/degree in glickfeld 0.02 for maximal RL and for optomotor 0.13 umino2008
Par.dblGratingContrast          = [.2 .4 .6 .8 1];    %0-1
Par.vecOrientations             = [30 120]%[0 30 60 90 120 150 180]; %possible orientations (picked randomly every trial)
Par.bgInt                       = 0.5; %background intensity [0-1]
Par.bgInt                       = round(Par.bgInt*255);
Par.boolUseSquareTex        = true; % Bool for using square tex or normal texture
Par.GratingMethod               = 2; %1 = only one orientation, 2 make adhoc, 3 = load i

%% Auditory Stimulus Parameters
Par.strAuStimType               = 'Beep';
Par.BeepFreq                    = 3500; %Bandpassed white noise around center frequency
Par.intSamplingRate             = 48000; %Default sampling rate
% Par.dblSoundLevel             = 0.4; %How to control volume?

%% TRAINING PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Par.preTrainingSecs             = 5;        %before training starts

%%Trial Structure
Par.intSecsITI                  = [5 8 15];       %ITI duration parameters for exponential probability: [onset mean cut]

Par.StimEpochDur                = 3.5;       % Duration of stimulus epoch in seconds. 3.9 tr1 3.5 tr2 

Par.PostDur                     = 0.3;        % Post-stimulus epoch duration in senconds. 

Par.LFROnset                    = 0.1;       % LFR onset(Mouse can lick for reward). In seconds. Since StimEpoch start. 
Par.LFROffset                   = 2;       % LFR offset(Mouse can lick for reward). In seconds. Since StimEpoch start. 2.4tr1 2.0tr2
Par.GivePassiveReward           = 0;       % Give Automatic Rewards at the end of LFR if no licks?
Par.PassiveRewardSize           = 4;         % AR size in uL

Par.boolSingleResp              = 0;        %Mouse has only one shot to lick correctly.
Par.boolRewardPenality          = 0;        %If mouse licks incorrectly, the next correct lick will trigger a different reward size.
Par.PenaltyRewardPercent        = 0.5;         %Percent of the current reward size that the mouse will get if it licks correctly after penality.

%%Session Structure
Par.UseTrialBlocks              = 1;        %Use Blocks or not
Par.BlockLength                 = [2 2];        %[L R] Block length in number of trials.

Par.LeftTrialsFreq              = 0.5; %frequency of left trials
Par.UpdateTrials                = 0; %Button for updating session structure values. Irrelevant initial state.

Par.MultimodalFreq              = 0; %frequency of multimodal trials / total trials.

Par.CatchTrialsFreq             = 0.1; %frequency of catch trials.

% Size, quantity and duration of reward giving:
Par.SessionDuration             = 60; %Planned session duration in minutes; 
Par.SessionReward               = 1200/0.8;    %Planned total reward in microliter; I take here performance into account: about 80% is actually received
Par.RewardConsumptionSecs       = Par.StimEpochDur + Par.PostDur + Par.intSecsITI(1);        %Time window for collecting reward

Par.intTrialNum                 = ceil(Par.SessionDuration*60/Par.RewardConsumptionSecs);
Par.CorrectRewardSizeLeft           = ceil(Par.SessionReward/Par.intTrialNum);        %Reward size upon licking (in uL)
Par.CorrectRewardSizeRight          = ceil(Par.SessionReward/Par.intTrialNum);        %Reward size upon licking (in uL)

%Create tria-types and stim-data
[Par] = create_trialtypes_pie(Par);