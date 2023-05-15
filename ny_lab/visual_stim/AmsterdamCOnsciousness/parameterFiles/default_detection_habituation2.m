function Par = default_detection_habituation
Par = struct; %structure with parameters

%% General variable definitions
Par.dblVersion          = 2.0; % version of the output file
Par.strTask             = 'detection';

%% Keypress parameters
Par.strExitKey          = 'F1';
Par.strRewardKey        = 'F9';
Par.strRewardKeyLeft    = 'F10';
Par.strRewardKeyRight   = 'F11';

%% Mouse and box parameters
Par.strTrialType        = 'Trial type V = Visual, A = Auditory, P = Probe';
Par.str90Deg            = '0 degrees is leftward motion; 90 degrees is upward motion';

%% Box specific params
Par.intAudioChannel     = 1;

%% Servo Parameters:
Par.UseServo            = true;

%% Piezo Parameters:
Par.UsePiezo            = false;

%% Screen Parameters
Par.intFrameRate            = 60;
Par.dblScreenDistance_cm    = 16; % cm;  should be changed later
Par.dblScreenWidth_cm       = 34; % cm;
Par.dblScreenHeight_cm      = 27; % cm;
Par.dblScreenWidth_deg      = atand((Par.dblScreenWidth_cm / 2) / Par.dblScreenDistance_cm) * 2;
Par.dblScreenHeight_deg     = atand((Par.dblScreenHeight_cm / 2) / Par.dblScreenDistance_cm) * 2;

%% Visual Stimulus Parameters
Par.strVisStimType          = 'movingGrating';
Par.dblSpeed                = 1.5; %TF of moving grating in cycles/sec
Par.vecOrientations         = [0 120 240]; %possible orientations (picked randomly every trial)
Par.stimBackInt             = 0.5; %background intensity during stimulus
Par.bgIntStim               = round(Par.stimBackInt*255);
Par.boolUseSquareTex        = true; % Bool for using square tex or normal texture

%% Auditory Stimulus Parameters
Par.dblPitch                = 0; %carrier frequency of A1 stimulus in hz not used here
Par.vecNoiseRange           = [8000 22000];
Par.dblAudioRampDur         = 0.001; % Duration of ramp for audio stim
% Par.dblModulationIndex    = 1000; %for auditory frequency modulation
% Par.dblModulationRate     = 1.5;
Par.intSamplingRate         = 44100;
Par.dblSoundLevel           = 0.4;

Par.bgInt                   = 0.5; %background intensity (dbl, [0 1])
Par.bgInt                   = round(Par.bgInt*255);

%% TRAINING PARAMETERS

%Trials -- in this script these parameters make sure that the side of
%reward is alternating, so be careful when you adjust them
Par.strStartStim                = 'Probe';  % The starting stimulus is the probe stimulus
Par.intBlockSizeVisual          = 20;       %Number of trials in a visual block, should be even in habituation phase
Par.intBlockSizeAudio           = 20;       %Number of trials in an audio block, should be even in habituation phase
Par.intProbeTrialNumVisual      = Par.intBlockSizeVisual/2; %15 %Number of probe trials in visual blocks (out of intBlockSize, they don't add up)
Par.intProbeTrialNumAudio       = Par.intBlockSizeAudio/2; %10 %Number of probe trials in audio blocks (out of intBlockSize, they don't add up)
Par.intVisualBlocks             = 6;       %Number of visual blocks
Par.intAudioBlocks              = 6;        %Number of audio blocks
Par.intTrialNum                 = (Par.intBlockSizeVisual*Par.intVisualBlocks) + (Par.intBlockSizeAudio*Par.intAudioBlocks);
Par.intMaxSameBlank             = 2;        %Maximum number of consecutive blank trials
Par.charYesSide                 = 'R';      % 'R' for right as yes and 'L' for left as yes
Par.charNoSide                  = 'L';      % 'R' for right as no and 'L' for left as no

%Timing
Par.preTrainingSecs             = 2;        %before training starts
Par.dblSecsBlankAtStart         = 1;        %black screen cameras
Par.dblSecsBlankPre             = [3 5];    %randomly chosen ITI
Par.dblSecsStimDur              = 3;        %stim duration
Par.dblSecsPreStim              = [0.1 0.1];%length of timewindow before stim onset random value between these two values
Par.dblSecsPreReward            = 1;      %length of timewindow before reward onset
Par.dblSecsBlankAtEnd           = 10;       %second pause after session is done

Par.RespWinDur                  = 3;        %Time window for responding after stimulus offset

Par.TolerateLicksITI            = true;     %Whether to tolerate licks during ITI
Par.SecsNoLickITI            	= 3;        %If licks are not tolerated how long no lick

Par.RewardSize                  = 4;        %Adapt script to deal with multiple reward size, calibrate beforehand
Par.boolWaitForResponse         = true;     %Wait for response of mouse before continuing
%Time-out
Par.UseTimeOut                  = true;        %whether to use a time out as punishment
Par.TimeOutSecs                 = 5;        %duration of time out

% Wait for response of mouse before starting new trial
Par.boolWaitForResponse         = false;

%Reward Consumption
Par.RewardConsumptionSecs       = 2;        %duration of reward consumption if rewarded

%% presentation times
Par.vecStimStartPhase           = nan(1,Par.intTrialNum);
Par.vecSecsITI                  = rand(1,Par.intTrialNum).*(Par.dblSecsBlankPre(2) - Par.dblSecsBlankPre(1)) + Par.dblSecsBlankPre(1);

%% CREATE STIM STRUCT WITH VECTORS CONTAINING STIMULUS INFO
Par.Stim                        = struct;
Par.Stim.vecType                = create_block_vectors(Par);
Par.Stim.vecSide(1:numel(Par.Stim.vecType)) = 'C';
Par.Stim.vecOri                 = create_orientation_vectors(Par);
% Determine the correct response for the trials (in this case probe trial is ´ no´ side
Par.Stim.CorrectSide(1:numel(Par.Stim.vecType)) = Par.charYesSide;
Par.Stim.CorrectSide(Par.Stim.vecType == 'P')            = Par.charNoSide;

