function Par = default_localization_habituation

%% General variable definitions
Par.dblVersion          = 2.0; % version of the output file
Par.strTask             = 'localization';
%% Keypress parameters
Par.strExitKey          = 'F1';
Par.strRewardKey        = 'F9';
Par.strRewardKeyLeft    = 'F10';
Par.strRewardKeyRight   = 'F11';

%% Mouse and box parameters
Par.strTrialType        = 'Trial type V = Visual, A = Auditory, P = Probe';%'Trial type V = Visual, A = Auditory, P = Probe';
Par.str90Deg            = '0 degrees is leftward motion; 90 degrees is upward motion';

%% Box specific params
Par.intAudioChannel             = 1;

%% Servo Parameters:
Par.UseServo                    = false;

%% Piezo Parameters:
Par.UsePiezo                    = false;

%% screen parameters
Par.intFrameRate                = 60;
Par.dblScreenDistance_cm        = 16; % cm;  should be changed later
Par.dblScreenWidth_cm           =  34; % cm; 
Par.dblScreenHeight_cm          = 27; % cm; 

Par.dblScreenWidth_deg          = atand((Par.dblScreenWidth_cm / 2) / Par.dblScreenDistance_cm) * 2;
Par.dblScreenHeight_deg         = atand((Par.dblScreenHeight_cm / 2) / Par.dblScreenDistance_cm) * 2;

Par.bgInt                       = 0.3; %background intensity (dbl, [0 1])
Par.bgInt                       = round(Par.bgInt*255);

%% stimulus params
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




Par.dblPitch                    = 0; %carrier frequency of A1 stimulus in hz not used here
Par.vecNoiseRange               = [8000 22000];
Par.dblAudioRampDur             = 0.001; % Duration of ramp for audio stim
% Par.dblModulationIndex = 1000; %for auditory frequency modulation
% Par.dblModulationRate = 1.5;
Par.intSamplingRate             = 44100;
Par.dblSoundLevel               = 0.4;

%% TRAINING PARAMETERS

%Trials
Par.intBlockSizeVisual          = 4; %60 %Number of trials in a visual block  
Par.intBlockSizeAudio           = 4;%4; %40 %Number of trials in an audio block  
Par.intProbeTrialNumVisual      = 0; %Number of probe trials in visual blocks (out of intBlockSize, they don't add up)
Par.intProbeTrialNumAudio       = 0;
Par.intVisualBlocks             = 100; %Number of visual blocks %was 12
Par.intAudioBlocks              = 1;%100; %Number of audio blocks  %was 12
Par.intMaxSameBlank             = 4; 
Par.intTrialNum                 = (Par.intBlockSizeVisual*Par.intVisualBlocks) + (Par.intBlockSizeAudio*Par.intAudioBlocks);

%Timing
Par.preTrainingSecs             = 0; %before training starts
Par.dblSecsBlankAtStart         = 1; %black screen cameras
Par.dblSecsBlankPre             = [4 6]; %randomly chosen ITI
Par.dblSecsNoLick               = [1 2]; %minimum time the animal should not have licked before a trial starts (randomly chosen for each trial)
Par.dblSecsPreStim              = [0 0]; %length of timewindow before stim onset (one of these two is randomly selected)
Par.dblSecsPreReward            = 0.8;      %length of timewindow before reward onset
Par.boolAbortTrial              = false; %whether to abort the trial if the animal licks in the pre-stim period
Par.dblSecsStimDur              = 10; %stim duration
Par.RespWinDur                  = 3;        %Time window for responding after stimulus offset
Par.RewardSize                  = 4;        %Adapt script to deal with multiple reward size, calibrate beforehand

Par.TolerateLicksITI            = true;     %Whether to tolerate licks during ITI
Par.SecsNoLickITI            	= 3;        %If dllicks are not tolerated how long no lick

%Time-out
Par.UseTimeOut                  = false;        %whether to use a time out as punishment %false
Par.TimeOutSecs                 = 5;        %duration of time out

Par.dblSecsBlankAtEnd           = 10; %second pause after session is done

% Wait for response of mouse before starting new trial
Par.boolWaitForResponse         = true;

%Reward Consumption
Par.RewardConsumptionSecs       = 4;        %duration of reward consumption if rewarded
Par.PassiveRewardSize           = 3;        %Reward size during Pavlovian conditioning (in uL)
Par.CorrectRewardSize           = 8;        %Reward size upon correct response (in uL)

%% presentation times
Par.vecStimStartPhase           = nan(1,Par.intTrialNum);
Par.vecSecsITI                  = rand(1,Par.intTrialNum).*(Par.dblSecsBlankPre(2) - Par.dblSecsBlankPre(1)) + Par.dblSecsBlankPre(1);

%% CREATE STIM STRUCT WITH VECTORS CONTAINING STIMULUS INFO
Par.Stim                        = struct;
Par.Stim.vecType                = create_block_vectors(Par);
Par.Stim.vecSide                = create_side_vectors(Par);
Par.Stim.vecOri                 = create_orientation_vectors(Par);
% Determine the correct response for the trials (in this case side of stim
% is correct side)
Par.Stim.CorrectSide = Par.Stim.vecSide;