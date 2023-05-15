function Par = mol_par_mod_discr_conf(varargin)

if nargin == 2
    Par                 = varargin{1};
    Par.TrainingStage   = varargin{2};
else
    Par = struct; %structure with parameters
    Par.TrainingStage = 1; %default setting
end
Par.ProceedStages               = 0; %Setting whether stage is advanced or not;

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
Par.StimSecs                    = 2;        %mean of exp dist randomly chosen stim duration before stimulus change
Par.RespWinSecs                 = 0;        %Time window for responding after stimulus offset
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
Par.TolerateLicksIncorrect      = 0;        %Whether to tolerate incorrect licks during stimulus post change or response window
Par.GivePassiveReward           = 0;        %Aid training give reward correct side
Par.PassiveRewardDelay          = 1.6;      %Delay after stim before passive reward is delivered
Par.PassiveRewardSize           = 3;        %Reward size during Pavlovian conditioning (in uL)
Par.CorrectRewardSize           = 5;        %Reward size upon correct response (in uL)
Par.StimSpanSides               = 1;        %Span of stimuli across other speaker (0 is one side only, 1 equal both)
Par.CounterBias                 = 0;        %Whether to apply algorithm to counter any developing biases
Par.CheckLick                   = 0;        %Whether to check whether the animal has licked before continuing

%% STIMULUS PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Screen Parameters
Par.intFrameRate                = 60;
Par.dblScreenDistance_cm        = 26; % cm;  should be changed later
Par.dblScreenWidth_cm           = 41; % cm;
Par.dblScreenHeight_cm          = 23; % cm;

%% Visual Stimulus Parameters
Par.vecOrientations             = 0:45:315; %Possible orientations (picked randomly every trial)
Par.ConflictOri                 = 180;
%passive
Par.vecVisStimIntensities       = 0.1;         %Vector of 0-1 intensities of contrast
%P3.2
%Par.vecVisStimIntensities       = 0.058;         %Vector of 0-1 intensities of contrast
%P3.3
%Par.vecVisStimIntensities       = 0.13;         %Vector of 0-1 intensities of contrast
%2.14
Par.vecVisStimIntensities       = 0.097;         %Vector of 0-1 intensities of contrast
%psyvis                          = [0.05 0.1 0.3 0.5 0.9];

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
Par.ConflictFreq                = 12000;
%Passive
Par.vecAuStimIntensities        = 0.82;         %Vector of 0-1 intensities of volume
%P3.2
%Par.vecAuStimIntensities        = 0.832;         %Vector of 0-1 intensities of volume
% P3.3
%Par.vecAuStimIntensities        = 0.75;         %Vector of 0-1 intensities of volume
% 2.14
%Par.vecAuStimIntensities        = 0.56;         %Vector of 0-1 intensities of volume
%psyau                           = [0.4 0.6 0.8 1];

Par.strAuStimType               = 'shepardtone';
Par.ShepardTones                = 9;
Par.ShepardWeights              = gausswin(length(Par.vecCenterFreq) * Par.ShepardTones);

%% Trial Types and Task Structure:
%Other type of trials:
Par.UseProbeTrials              = 1; %No stimulus                 Trial code = 'P'
Par.FractionProbeTrials         = 0.08;
Par.UseConflictTrials           = 1; %Both stimuli             Trial code = 'C'
Par.FractionConflictTrials      = 0.08;

Par.UseTrialBlocks              = 0;        %Whether to use blocks or pseudorandom
Par.BlockSize                   = 0;       %Number of trials in a block, should be even in habituation phase

%% Create stim struct with trial-by-trial stimulus data
Par                             = create_trialtypes_mod_discr(Par);    %Generate trial type sequence:
Par                             = create_stimtypes_mod_discr(Par);     %Generate trial specific stimulus settings with these trials 

%% Give conflict trials just one stimulus type:
Par.Stim.vecOri(find(Par.Stim.vecTrialType=='C'))   = Par.ConflictOri;
Par.Stim.vecFreq(find(Par.Stim.vecTrialType=='C'))  = Par.ConflictFreq;
%% Reward conflict trials with random probability 0.7
Par.Stim.leftCorrect(find(Par.Stim.vecTrialType=='C'))        = rand(numel(Par.Stim.leftCorrect(find(Par.Stim.vecTrialType=='C'))),1)<=0.7;
Par.Stim.rightCorrect(find(Par.Stim.vecTrialType=='C'))        = rand(numel(Par.Stim.leftCorrect(find(Par.Stim.vecTrialType=='C'))),1)<=0.7;

end