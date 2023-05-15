function Par = pie_par_fulltask
Par = struct; %structure with parameters

%% General variable definitions
Par.dblVersion          = 1.0; % version of the output file
Par.strTask             = 'mol_pie_fullx';

%% Keypress parameters
Par.strExitKey          = 'F2';
Par.strRewardKeyLeft    = 'F3';
Par.strRewardKeyRight   = 'F4';

%% Servo Parameters:
Par.UseServo            = 0;

%% Piezo Parameters:
Par.UsePiezo            = 1;

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
Par.dblGratingContrast          = [0.8];    %0-1
Par.vecOrientations             = [30]%[0 30 60 90 120 150 180]; %possible orientations (picked randomly every trial)
Par.bgInt                       = 0.5; %background intensity [0-1]
Par.bgInt                       = round(Par.bgInt*255);
Par.boolUseSquareTex        = true; % Bool for using square tex or normal texture
Par.GratingMethod               = 2; %1 = only one orientation, 2 make adhoc, 3 = load i

%% Tactile Stimulus Parameters
Par.DeflectionIntensities       = [100];%0-100


%% TRAINING PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Par.preTrainingSecs             = 5;        %before training starts

%%Trial Structure
Par.intSecsITI                  = [4.5 5.8 9];       %ITI duration parameters for exponential probability: [onset mean cut]

Par.StimEpochDur                = 1.8;       % Duration of stimulus epoch in seconds.  

Par.PostDur                     = 0.05;        % Post-stimulus epoch duration in senconds. 

Par.LFROnset                    = 0.1;       % LFR onset(Mouse can lick for reward). In seconds. Since StimEpoch start. 
Par.LFROffset                   = 0.99;       % LFR offset(Mouse can lick for reward). In seconds. Since StimEpoch start.
Par.GivePassiveReward           = 1;       % Give Automatic Rewards at the end of LFR if no licks?
Par.PassiveRewardSize           = 6;         % AR size in uL

Par.boolSingleResp              = 0;        %Mouse has only one shot to lick correctly.
Par.boolRewardPenality          = 0;        %If mouse licks incorrectly, the next correct lick will trigger a different reward size.
Par.PenaltyRewardPercent        = 0.5;         %Percent of the current reward size that the mouse will get if it licks correctly after penality.

%%Session Structure
Par.UseTrialBlocks              = 1;        %Use Blocks or not
Par.BlockLength                 = [2 2];        %[L R] Block length in number of trials.

Par.LeftTrialsFreq              = 0.5; %frequency of left trials
Par.UpdateTrials                = 0.5; %Button for updating session structure values. For some reason must be not zero.

Par.MultimodalFreq              = 0.0001; %frequency of multimodal trials / total trials.
Par.TactileTrialsFreq           = 0.9999; %frequency of tactile trials / unimodal trials.

Par.CatchTrialsFreq             = 0.0001; %frequency of catch trials.

% Size, quantity and duration of reward giving:
Par.SessionDuration             = 60; %Planned session duration in minutes; 
Par.SessionReward               = 1200/0.8;    %Planned total reward in microliter; I take here performance into account: about 80% is actually received
Par.RewardConsumptionSecs       = Par.StimEpochDur + Par.PostDur + Par.intSecsITI(1);        %Time window for collecting reward

Par.intTrialNum                 = ceil(Par.SessionDuration*60/Par.RewardConsumptionSecs);
Par.CorrectRewardSizeLeft           = ceil(Par.SessionReward/Par.intTrialNum);        %Reward size upon licking (in uL)
Par.CorrectRewardSizeRight          = ceil(Par.SessionReward/Par.intTrialNum);        %Reward size upon licking (in uL)

%GUI
Par.useHistPanel = 1; %Use new panel with past and future trials.

%Create tria-types and stim-data
[Par] = create_trialtypes_pie(Par);