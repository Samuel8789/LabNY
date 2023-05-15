function Par = mol_par_mod_discr(varargin)

if nargin == 2
    Par                 = varargin{1};
    Par.TrainingStage   = varargin{2};
else
    Par = struct; %structure with parameters
    Par.TrainingStage = 1; %default setting
end

%% General variable definitions
Par.strTask             = 'mol_mod_discr';

%% Keypress parameters
Par.strExitKey          = 'F2';
Par.strRewardKeyLeft    = 'F3';
Par.strRewardKeyRight   = 'F4';

%% Servo Parameters:
Par.UseServo            = false;

%% Piezo Parameters:
Par.UsePiezo            = false;

%% TTL arduino Parameters
Par.serialPortStr       = '/dev/ttyACM1';
Par.TTL_time            = 0.001;

%% TRAINING PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Trials -- in this script these parameters make sure that the side of
%reward is alternating, so be careful when you adjust them

%% Timing parameters
Par.ExpVariability              = 0.7;      %variablility of random exponential durs 1=no var, 0.5 = a lot
Par.preTrainingSecs             = 5;        %before training starts
Par.intSecsITI                  = 5;        %mean of exp dist randomly chosen ITI
Par.StimSecs                    = 5;        %mean of exp dist randomly chosen stim duration before stimulus change
Par.RespWinSecs                 = 1;        %Time window for responding after stimulus offset
Par.RewardConsumptionSecs       = 2;        %Time window for collecting reward
Par.UseTimeOut                  = 0;        %whether to use a time out as punishment
Par.TimeOutSecs                 = 2;        %duration of time out
Par.dblSecsBlankAtEnd           = 10;       %second pause after session is done
Par.GracePeriod                 = 0.1;      %Seconds after prechange that licks are not counted

%% Behavioral Parameters (these should be under control by GUI or command line)
Par.VisualLeftCorrectSide       = 1;        %Parameter that governs correct side per modality
Par.TolerateLicksITI            = 1;        %Whether to tolerate licks during ITI
Par.SecsNoLickITI            	  = 2.5;      %If licks are not tolerated how long no lick
Par.TolerateLicksPreChange      = 1;        %Whether to tolerate licks during stimulus pre change
Par.TolerateLicksIncorrect      = 1;        %Whether to tolerate incorrect licks during stimulus post change or response window
Par.GivePassiveReward           = 1;        %Aid training give reward correct side
Par.PassiveRewardDelay          = 1.6;      %Delay after stim before passive reward is delivered
Par.PassiveRewardSize           = 3;        %Reward size during Pavlovian conditioning (in uL)
Par.CorrectRewardSize           = 8;        %Reward size upon correct response (in uL)
Par.StimSpanSides               = 1;        %Span of stimuli across other speaker (0 is one side only, 1 equal both)
Par.CounterBias                 = 1;        %Whether to apply algorithm to counter any developing biases
Par.CheckLick                   = 0;        %Whether to check whether the animal has licked before continuing

%% STIMULUS PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Screen Parameters
Par.intFrameRate                = 60;
Par.dblScreenDistance_cm        = 26; % cm;  should be changed later
Par.dblScreenWidth_cm           = 41; % cm;
Par.dblScreenHeight_cm          = 23; % cm;

%% Visual Stimulus Parameters
Par.vecOrientations             = 0:45:315; %Possible orientations (picked randomly every trial)
Par.vecVisStimIntensities       = 1;         %Vector of 0-1 intensities of contrast
Par.strVisStimType              = 'movingGrating';
Par.str90Deg                    = '0 degrees is leftward motion; 90 degrees is upward motion';
Par.dblSpeed                    = 1;        %TF of moving grating in cycles/sec
Par.dblSpatFreq                 = 0.05;     %SF of moving gratings in cycles/degree
Par.dblGratingContrast          = 1;        %0-1
Par.bgInt                       = 0.5;      %background intensity [0-1]
Par.bgInt                       = round(Par.bgInt*255); 
Par.boolUseSquareTex            = true; % Bool for using square tex or normal texture
Par.GratingMethod               = 3;        %1 = only one orientation, 2 make adhoc, 3 = load in

%% Auditory Stimulus Parameters
Par.vecCenterFreq               = 8000:1000:15000; %Center frequency with octaves below and above as well
Par.vecAuStimIntensities        = 1;        %Vector of 0-1 intensities of volume
Par.strAuStimType               = 'shepardtone';
Par.ShepardTones                = 9;
Par.ShepardWeights              = gausswin(length(Par.vecCenterFreq) * Par.ShepardTones);

%% Visual: (For Psychophysics:)
%P3.2
%psyvis                          = [0.012  0.035   0.095   0.29    1];
%P3.3
%psyvis                          = [0.015   0.055   0.11    0.26    1];
%2.14
psyvis                          = [0.01   0.05    0.09    0.15    1];
%% Auditory: (For Psychophysics:)
% P3.2
%psyau                           = [0.67   0.81    0.85    0.89    0.95];
% P3.3
%psyau                           = [0.69   0.78    0.83    0.88    0.95];
% 2.14
psyau                           = [0.30   0.38    0.45    0.64    0.85];

%% Trial Types and Task Structure:
%Other type of trials:
Par.UseProbeTrials              = 0; %No stimulus                 Trial code = 'P'
Par.FractionProbeTrials         = 0.2;
Par.UseConflictTrials           = 0; %Both stimuli             Trial code = 'C'
Par.FractionConflictTrials      = 0.1;

Par.UseTrialBlocks              = 0;        %Whether to use blocks or pseudorandom
Par.BlockSize                   = 20;       %Number of trials in a block, should be even in habituation phase

%% Overrule default settings with trainingstage settings:
%Get parameters from mouse database, follow progress of mouse 
%training to automatize the training stages. So get mouse and date specific
%parameters.
Par.ProceedStages               = 1; %Setting whether stage is advanced or not;

%Par.AllTrainingStages          = [1        2       3       4           5           6];
Stage.GivePassiveReward         = {1        1       0       0           0           0};      %Aid training give reward correct side
Stage.CorrectRewardSize         = {10       7       5       5           5           5};      %Reward size upon correct response (in uL) correct side
Stage.StimSecs                  = {5        4       2       2           2           2};      %mean of exp dist randomly chosen stim duration after stimulus change
Stage.UseTimeOut                = {0        0       0       0           0           0};      %whether to use a time out as punishment
Stage.TolerateLicksITI          = {1        1       1       1           1           1};      %Whether to tolerate licks during ITI
Stage.TolerateLicksPreChange    = {1        1       1       1           1           1};      %Whether to tolerate licks during stimulus pre change
Stage.TolerateLicksIncorrect    = {1        1       0       0           0           0};      %Whether to tolerate incorrect licks during stimulus post change or response window
Stage.UseTrialBlocks            = {1        0       0       0           0           0};      %Whether to use blocks or pseudorandom
Stage.BlockSize                 = {800      0       0       0           0           0};      %Size of the blocks
Stage.vecVisStimIntensities     = {1        1       1       psyvis      psyvis      psyvis};      %Vector of 0-1 intensities of contrast
Stage.vecAuStimIntensities      = {0.8      0.8     0.85    psyau       psyau       psyau};      %Vector of 0-1 intensities of volume
Stage.UseProbeTrials            = {0        0       1       1           1           0};      %No stimulus trial code = 'P'
Stage.FractionProbeTrials       = {0        0       0.05    0.08        0.1         0.1};      %Fraction of probe trials

%Get the right parameters depending on trainingstage:
StageFieldNames = fieldnames(Stage);
Par.TrainingStage = min([Par.TrainingStage length(Stage.(StageFieldNames{1}))]);
for stgfield =1:length(StageFieldNames)
    Par.(StageFieldNames{stgfield}) = Stage.(StageFieldNames{stgfield}){Par.TrainingStage};
end

%% Create stim struct with trial-by-trial stimulus data
Par                             = create_trialtypes_mod_discr(Par);    %Generate trial type sequence:
Par                             = create_stimtypes_mod_discr(Par);     %Generate trial specific stimulus settings with these trials 

end