function Par = pie_par_detection
Par = struct; %structure with parameters

%% General variable definitions
Par.dblVersion          = 1.0; % version of the output file
Par.strTask             = 'mol_pie_detectx';

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
Par.dblScreenDistance_cm        = 25; % cm;  should be changed later
Par.dblScreenWidth_cm           = 36.1; % cm;
Par.dblScreenHeight_cm          = 20.3; % cm;

%% Visual Stimulus Parameters
Par.strVisStimType              = 'fs_grating';
Par.str90Deg                    = '0 degrees is leftward motion; 90 degrees is upward motion';
Par.dblSpeed                    = 0.4;  %TF of moving grating in cycles/sec in glickfeld 1-2 for maximal RL
Par.dblSpatFreq                 = 0.05; %SF of moving gratings in cycles/degree in glickfeld 0.02 for maximal RL
Par.dblGratingContrast          = 1;    %0-1
Par.vecOrientations             = [120]%[0 30 60 90 120 150 180]; %possible orientations (picked randomly every trial)
Par.bgInt                       = 0.5; %background intensity [0-1]
Par.bgInt                       = round(Par.bgInt*255);
Par.boolUseSquareTex        = true; % Bool for using square tex or normal texture
Par.GratingMethod               = 2; %1 = only one orientation, 2 make adhoc, 3 = load i


%% TRAINING PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Par.preTrainingSecs             = 5;        %before training starts

%%Structure
Par.intSecsITI                  = [2 3 4];       %ITI duration parameters for exponential probability: [onset mean cut]

Par.StimEpochDur                = 3.500;       % Duration of stimulus epoch in seconds.  

Par.PostDur                     = 0.500;        % Post-stimulus epoch duration in senconds. 

Par.LFROnset                    = 0;       % LFR onset(Mouse can lick for reward). In seconds. Since StimEpoch start. 
Par.LFROffset                   = 0.600;       % LFR offset(Mouse can lick for reward). In seconds. Since StimEpoch start.
Par.GivePassiveReward           = 1;       % Give Automatic Rewards at the end of LFR if no licks?
Par.PassiveRewardSize           = 5;         % AR size in uL


% Size, quantity and duration of reward giving:
Par.SessionDuration             = 30; %Planned session duration in minutes; 
Par.SessionReward               = 1100;    %Planned total reward in microliter;
Par.RewardConsumptionSecs       = Par.StimEpochDur + Par.PostDur + Par.intSecsITI(1);        %Time window for collecting reward

Par.intTrialNum                 = ceil(Par.SessionDuration*60/Par.RewardConsumptionSecs);
Par.CorrectRewardSizeLeft           = ceil(Par.SessionReward/Par.intTrialNum);        %Reward size upon licking (in uL)
Par.CorrectRewardSizeRight      = Par.CorrectRewardSizeLeft;

%Simulate the genetare trial and stim functions

Par.Stim.vecOri = Par.vecOrientations(randi(length(Par.vecOrientations),Par.intTrialNum,1)); %Generates an Ori for each trial, randomly drawn from vecOrientations

Par.Stim.vecVisualSide(1:Par.intTrialNum)        = 'A'; %All screen, both sides
