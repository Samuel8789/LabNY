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
Par.strTrialType        = 'Trial type V = Visual, A = Auditory, P = Probe';
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
% Visual
Par.strVisStimType              = 'gabor';
Par.dblSpatialFrequency         = 0.01; %spatial frequency of grating (cycle/degree)
Par.dblSigma                    = 100; %standard deviation of gaussian window
Par.vecCoordinatesRight         = [0.8,0.5]; % Coordinates of center of gabor 0 --> 1 normalized (x, y)
Par.vecCoordinatesLeft          = [0.2,0.5]; % Coordinates of center of gabor 0 --> 1 normalized (x, y)

Par.vecOrientations             = [0 120 240]; %possible orientations (picked randomly every trial)
Par.stimBackInt                 = 0.5; %background intensity during stimulus
Par.bgIntStim                   = round(Par.stimBackInt*255);

Par.bgInt                       = 0.5; %background intensity (dbl, [0 1])
Par.bgInt                       = round(Par.bgInt*255);

Par.dblPitch                    = 0; %carrier frequency of A1 stimulus in hz not used here
Par.vecNoiseRange               = [8000 22000];
Par.dblAudioRampDur             = 0.001; % Duration of ramp for audio stim
% Par.dblModulationIndex = 1000; %for auditory frequency modulation
% Par.dblModulationRate = 1.5;
Par.intSamplingRate             = 44100;
Par.dblSoundLevel               = 0.4;

%% TRAINING PARAMETERS

%Trials
Par.intBlockSizeVisual          = 40; %60 %Number of trials in a visual block
Par.intBlockSizeAudio           = 4; %40 %Number of trials in an audio block
Par.intProbeTrialNumVisual      = 0; %Number of probe trials in visual blocks (out of intBlockSize, they don't add up)
Par.intProbeTrialNumAudio       = 0;
Par.intVisualBlocks             = 12; %Number of visual blocks
Par.intAudioBlocks              = 0; %Number of audio blocks
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
Par.SecsNoLickITI            	= 3;        %If licks are not tolerated how long no lick

%Time-out
Par.UseTimeOut                  = false;        %whether to use a time out as punishment
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
