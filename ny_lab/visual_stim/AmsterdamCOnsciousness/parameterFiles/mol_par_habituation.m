function Par = mol_par_habituation
Par = struct; %structure with parameters

%% General variable definitions
Par.dblVersion          = 1.0; % version of the output file
Par.strTask             = 'mol_habituation';

%% Keypress parameters
Par.strExitKey          = 'F2';
Par.strRewardKeyLeft    = 'F3';
Par.strRewardKeyRight   = 'F4';

%% Servo Parameters:
Par.UseServo            = 0;

%% Piezo Parameters:
Par.UsePiezo            = 0;

%% TRAINING PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Par.preTrainingSecs             = 2;        %before training starts
Par.bgInt                       = 0.4;

% Size, quantity and duration of reward giving:
Par.SessionDuration             = 30;        %Planned session duration in minutes; 
Par.SessionReward               = 1500;     %Planned total reward in microliter;
Par.RewardConsumptionSecs       = 4;        %Time window for collecting reward
Par.RespWinSecs                 = 10;       %Time window for responding after stimulus offset

Par.intTrialNum                 = Par.SessionDuration*60/Par.RewardConsumptionSecs;
Par.GivePassiveReward           = 0;        %Aid training give reward correct side
Par.PassiveRewardSize           = 4;        %Reward size during Pavlovian conditioning (in uL)
Par.PassiveRewardInterval       = 5;        %Interval when not licking for passive reward
Par.CorrectRewardSize           = ceil(Par.SessionReward/Par.intTrialNum);        %Reward size upon licking (in uL)

