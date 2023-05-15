function Par = mol_par_sifi(varargin)

if nargin == 2
    Par                 = varargin{1};
    Par.TrainingStage   = varargin{2};
else
    Par = struct; %structure with parameters
    Par.TrainingStage = 1; %default setting
end

%% General variable definitions
Par.strTask             = 'mol_sifi';

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
Par.intSecsITI                  = 4;        %randomly chosen ITI
Par.RespWinSecs                 = 4;        %Time window for responding after stimulus offset
Par.RewardConsumptionSecs       = 3;        %Time window for collecting reward
Par.UseTimeOut                  = 0;        %whether to use a time out as punishment
Par.TimeOutSecs                 = 5;        %duration of time out
Par.dblSecsBlankAtEnd           = 10;       %second pause after session is done
Par.GracePeriod                 = 0.1;      %Seconds after stimulus that licks are not counted

%% Behavioral Parameters (these should be under control by GUI or command line)
Par.OneFlashLeft                = 1;        %Parameter that governs correct side for F1 vs F2
Par.TolerateLicksITI            = 1;        %Whether to tolerate licks during ITI
Par.SecsNoLickITI            	= 3;        %If licks are not tolerated how long no lick
Par.TolerateLicksIncorrect      = 1;        %Whether to tolerate incorrect licks during stimulus post change or response window
Par.GivePassiveReward           = 1;        %Aid training give reward correct side
Par.PassiveRewardDelay          = 1;        %Delay after stim change before reward
Par.PassiveRewardSize           = 3;        %Reward size during Pavlovian conditioning (in uL)
Par.CorrectRewardSize           = 8;        %Reward size upon correct response (in uL)
Par.CounterBias                 = 1;        %Whether to apply algorithm to counter any developing biases
Par.CheckLick                   = 1;        %Whether to check whether the animal has licked before continuing

%% STIMULUS PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Screen Parameters
Par.intFrameRate                = 60;
Par.dblScreenDistance_cm        = 26; % cm;  should be changed later
Par.dblScreenWidth_cm           = 41; % cm;
Par.dblScreenHeight_cm          = 23; % cm;
Par.ScreenHeight_deg            = atand((Par.dblScreenHeight_cm / 2) / Par.dblScreenDistance_cm) * 2;

%% Visual Stimulus Parameters
%Par.strVisStimType              = 'Flash';
%Par.FlashInt                    = 50;   % Flash intensity       [0-255]
Par.strVisStimType              = 'Checker';
Par.bgInt                       = 128;  % Background intensity  [0-255]
Par.StimSizeRetinalDegrees      = 25;   % Retinal degrees of flashing disk

%% Auditory Stimulus Parameters
Par.strAuStimType               = 'Beep';
Par.BeepFreq                    = 15000; %Beep frequency
Par.intSamplingRate             = 48000; %Default sampling rate
% Par.dblSoundLevel             = 0.4; %How to control volume?

%% Trial Types and Task Structure:
Par.vecFlashes      = [1 2];
Par.vecBeeps        = [0];

Par.FlashDur        = 0.01667*5;      %Will be rounded to number of frames (each frame ~16.67 ms)
Par.IFI             = 0.01667*30;     %Will be rounded to number of frames (each frame ~16.67 ms)

Par.BeepDur         = 0.050;          %In seconds
Par.IBI             = 0.056;          %In seconds

Par.RandomBeeps     = 1;
Par.StimLateralize  = 1;              % Whether stim location matches response location (lateralized to help) [0-1] %1 is full lateralization

% Par.UseBimodalTrials            = 1; %Main type, w/change       Trial code = 'X' (Visual Change) or 'Y' (Auditory Change)
% 
% %Other type of trials:
% Par.UseUnimodalTrials           = 1; %Unimodal stimulation      Trial code = 'A' or 'V'
% Par.FractionUnimodalTrials      = 0.2;
% Par.UseProbeTrials              = 0; %No change                 Trial code = 'P'
% Par.FractionProbeTrials         = 0.1;
% Par.UseConflictTrials           = 0; %Double change             Trial code = 'C'
% Par.FractionConflictTrials      = 0.1;

Par.UseTrialBlocks              = 0;        %Whether to use blocks or pseudorandom
Par.BlockSize                   = 20;       %Number of trials in a block, should be even in habituation phase

%% Overrule default settings with trainingstage settings:
%Get parameters from mouse database, follow progress of mouse 
%training to automatize the training stages. So get mouse and date specific
%parameters.
Par.ProceedStages               = 1; %Setting whether stage is advanced or not;

%Par.AllTrainingStages          = [1        2        3       4       5       6       7       8       9      ];
Stage.FlashDur                  = {0.2      0.200    0.150   0.100   0.010   0.010   0.010   0.010   1      };      %Duration of flashes in seconds     %Inter Beep Interval in seconds
Stage.IFI                       = {0.15     0.15     0.15    0.15    0.1    0.060   1       1       1      };      %Inter Flash Interval in seconds
Stage.BeepDur                   = {0.05     0.050    0.050   0.050   0.05    0.010   0.010   0.010   1      };      %Duration of beeps in seconds
Stage.IBI                       = {0.08     0.08     0.08    0.08    0.08    1       1       1       1      };      %Inter Beep Interval in seconds
Stage.RandomBeeps               = {0        0        1       1       0       0       0       0       0      };      %Use Random Beeps played
Stage.GivePassiveReward         = {1        0        1       1       0       0       0       0       0      };      %Aid training give reward correct side
Stage.CorrectRewardSize         = {10       6        6       6       4       4       4       4       10      };     %Reward size upon correct response (in uL) correct side
Stage.UseTimeOut                = {0        0        0       0       0       1       1       1       1      };      %whether to use a time out as punishment
Stage.TolerateLicksITI          = {1        1        1       1       1       0       0       0       0      };      %Whether to tolerate licks during ITI
Stage.TolerateLicksIncorrect    = {1        0        1       0       1       0       0       0       0      };      %Whether to tolerate licks during ITI
Stage.UseTrialBlocks            = {1        1        0       0       0       0       0       0       0      };      %Whether to use blocks or pseudorandom
Stage.BlockSize                 = {150      150      0       0       0       0       0       0       0      };      %Whether to use blocks or pseudorandom
Stage.StimLateralize            = {1        1        1       1       0       0       0       0       0      };      % Whether stim location matches response location (lateralized to help) [0-1] %1 is full lateralization
Stage.RespWinSecs               = {4        4        5       3       1.5     1.5     1.5     1.5     1.5    };      %Secs time to respond after stim offset

%Get the right parameters depending on trainingstage:
StageFieldNames = fieldnames(Stage);
Par.TrainingStage = min([Par.TrainingStage length(Stage.(StageFieldNames{1}))]);
for stgfield =1:length(StageFieldNames)
    Par.(StageFieldNames{stgfield}) = Stage.(StageFieldNames{stgfield}){Par.TrainingStage};
end

%% Create stim struct with trial-by-trial stimulus data
Par                             = create_trialtypes_sifi(Par);    %Generate trial type sequence:

