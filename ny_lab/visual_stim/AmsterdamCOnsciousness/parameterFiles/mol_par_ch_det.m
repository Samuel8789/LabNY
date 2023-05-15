function Par = mol_par_ch_det(varargin)

if nargin == 2
    Par                 = varargin{1};
    Par.TrainingStage   = varargin{2};
else
    Par = struct; %structure with parameters
    Par.TrainingStage = 1; %default setting
end

%% General variable definitions
Par.strTask             = 'mol_ch_det';

%% Keypress parameters
Par.strExitKey          = 'F2';
Par.strRewardKeyLeft    = 'F3';
Par.strRewardKeyRight   = 'F4';

%% Servo Parameters:
Par.UseServo            = false;

%% Piezo Parameters:
Par.UsePiezo            = false;

%% TRAINING PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Trials -- in this script these parameters make sure that the side of
%reward is alternating, so be careful when you adjust them

%% Timing parameters
Par.ExpVariability              = 0.7;      %variablility of random exponential durs 1=no var, 0.5 = a lot
Par.preTrainingSecs             = 5;        %before training starts
Par.intSecsITI                  = 5;        %mean of exp dist randomly chosen ITI
% Par.StimPreChangeSecs           = 0.1;      %mean of exp dist randomly chosen stim duration before stimulus change
% Par.StimPostChangeSecs          = 2;        %stim duration after stimulus change
Par.RespWinSecs                 = 1;        %Time window for responding after stimulus offset
% Par.RewardConsumptionSecs       = 2;        %Time window for collecting reward
Par.UseTimeOut                  = 0;        %whether to use a time out as punishment
Par.TimeOutSecs                 = 2;        %duration of time out
Par.dblSecsBlankAtEnd           = 10;       %second pause after session is done
Par.GracePeriod                 = 0.1;      %Seconds after prechange that licks are not counted

%% Behavioral Parameters (these should be under control by GUI or command line)
Par.VisualLeftCorrectSide       = 1;        %Parameter that governs correct side per modality
Par.TolerateLicksITI            = 1;        %Whether to tolerate licks during ITI
Par.SecsNoLickITI            	  = 2.5;      %If licks are not tolerated how long no lick
% Par.TolerateLicksPreChange      = 1;        %Whether to tolerate licks during stimulus pre change
Par.TolerateLicksIncorrect      = 1;        %Whether to tolerate incorrect licks during stimulus post change or response window
Par.GivePassiveReward           = 1;        %Aid training give reward correct side
Par.PassiveRewardDelay          = 1.6;      %Delay after stim change before reward
Par.PassiveRewardSize           = 3;        %Reward size during Pavlovian conditioning (in uL)
Par.CorrectRewardSize           = 7;        %Reward size upon correct response (in uL)
% Par.StimPreChangeIntensity      = 1;        %0-1 intensity of volume and contrast before change to aid detection
Par.StimSpanSides               = 1;        %Span of stimuli across other speaker (0 is one side only, 1 equal both)
Par.CounterBias                 = 1;        %Whether to apply algorithm to counter any developing biases
Par.CheckLick                   = 1;        %Whether to check whether the animal has licked before continuing

%% STIMULUS PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Screen Parameters
Par.intFrameRate                = 60;
Par.dblScreenDistance_cm        = 26; % cm;  should be changed later
Par.dblScreenWidth_cm           = 41; % cm;
Par.dblScreenHeight_cm          = 23; % cm;

%% Visual Stimulus Parameters
Par.strVisStimType              = 'movingGratingChange';
Par.str90Deg                    = '0 degrees is leftward motion; 90 degrees is upward motion';
Par.dblSpeed                    = 1;        %TF of moving grating in cycles/sec
Par.dblSpatFreq                 = 0.05;     %SF of moving gratings in cycles/degree
Par.dblGratingContrast          = 0.7;      %0-1
Par.bgInt                       = 0.5;      %background intensity [0-1]
Par.bgInt                       = round(Par.bgInt*255);Par.boolUseSquareTex        = true; % Bool for using square tex or normal texture
Par.GratingMethod               = 3;        %1 = only one orientation, 2 make adhoc, 3 = load in

Par.vecOrientations             = 0:1:359;  %Possible orientations (picked randomly every trial)
Par.vecOriChange                = 90;
Par.vecOrientations             = [0:30:330];  %Possible orientations (picked randomly every trial)

%% Auditory Stimulus Parameters
Par.strAuStimType               = 'shepardtone';
% Par.intSamplingRate             = 48000;    %Default sampling rate
Par.dblSoundLevel               = 0.65;     %in dB SPL of continuous sound
Par.vecCenterFreq               = 8000:100:15900; %Center frequency with octaves below and above as well
Par.ShepardTones                = 9;
Par.ShepardWeights              = gausswin(length(Par.vecCenterFreq) * Par.ShepardTones);
Par.vecFreqChange               = 4000;

%% Trial Types and Task Structure:
Par.UseBimodalTrials            = 1; %Main type, w/change       Trial code = 'X' (Visual Change) or 'Y' (Auditory Change)
%Other type of trials:
Par.UseUnimodalTrials           = 0; %Unimodal stimulation      Trial code = 'A' or 'V'
Par.FractionUnimodalTrials      = 0;
Par.UseBiProbeTrials            = 1; %No change                 Trial code = 'P'
Par.FractionBiProbeTrials       = 0.15;
Par.UseUniProbeTrials           = 0; %Unimodal stim no change   Trial code = 'Q' or 'R'
Par.FractionUniProbeTrials      = 0;
Par.UseConflictTrials           = 0; %Double change             Trial code = 'C'
Par.FractionConflictTrials      = 0;

Par.UseTrialBlocks              = 0;      %Whether to use blocks or pseudorandom
Par.BlockSize                   = 20;     %Number of trials in a block, should be even in habituation phase

%% Overrule default settings with trainingstage settings:
%Get parameters from mouse database, follow progress of mouse 
%training to automatize the training stages. So get mouse and date specific
%parameters.
Par.ProceedStages               = 1; %Setting whether stage is advanced or not;

%Par.AllTrainingStages          = [1        2       3           4           5           6               7           ];
Stage.GivePassiveReward         = {1        0       0           0           0           0               0           };      %Aid training give reward correct side
Stage.CorrectRewardSize         = {7        6       5           5           5           5               5           };      %Reward size upon correct response (in uL) correct side
Stage.RespWinSecs               = {3        2       1           2           1           1               1           };      %%Time window for responding after stimulus offset
Stage.TolerateLicksIncorrect    = {1        0       0           0           0           0               0           };      %Whether to tolerate incorrect licks
Stage.UseBiProbeTrials          = {0        0       1           1           1           1               0           };      %Unimodal stimulation      Trial code = 'A' or 'V'
Stage.FractionBiProbeTrials     = {0        0       0.15        0.15        0.15        0.15            0           };

Stage.vecOriChange              = {90       90      90          [45 90]     [45 90]     [5 15 45 90]    [5 15 45 90]}; 
Stage.vecFreqChange             = {4000     4000    4000        [2000 4000] [2000 4000] [200 600 2000 4000] [200 600 2000 4000]}; 

% Stage.StimPreChangeSecs         = {3        3       3           3           3           3       };      %randomly chosen stim duration before stimulus change
% Stage.StimPostChangeSecs        = {2        2       1.2         2           1.2         2       };      %mean of exp dist randomly chosen stim duration after stimulus change

%Get the right parameters depending on trainingstage:
StageFieldNames = fieldnames(Stage);
Par.TrainingStage = min([Par.TrainingStage length(Stage.(StageFieldNames{1}))]);
for stgfield =1:length(StageFieldNames)
    Par.(StageFieldNames{stgfield}) = Stage.(StageFieldNames{stgfield}){Par.TrainingStage};
end

%% Create stim struct with trial-by-trial stimulus data
Par                             = create_trialtypes_change(Par);    %Generate trial type sequence:
Par                             = create_stimtypes_change(Par);     %Generate trial specific stimulus settings with these trials 

