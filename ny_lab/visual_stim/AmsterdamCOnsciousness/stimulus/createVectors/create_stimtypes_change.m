function [Par] = create_stimtypes_change(Par)
%% Initialize variables:
Par.Stim.vecOriPreChange    = NaN(1,Par.intTrialNum);
Par.Stim.vecOriPostChange   = NaN(1,Par.intTrialNum);
Par.Stim.vecFreqPreChange   = NaN(1,Par.intTrialNum);
Par.Stim.vecFreqPostChange  = NaN(1,Par.intTrialNum);
RandPermOri                 = NaN(1,Par.intTrialNum);
RandPermOriChange           = NaN(1,Par.intTrialNum);
RandPermFreq                = NaN(1,Par.intTrialNum);
RandPermFreqChange          = NaN(1,Par.intTrialNum);

%% Distribute orientations over trial types that have visual pre and or post change 

for trial = 1:Par.intTrialNum
    RandPermOri(trial)    = randperm(length(Par.vecOrientations),1);
end
Par.Stim.vecOriPreChange(Par.Stim.hasvisual)        = Par.vecOrientations(RandPermOri(Par.Stim.hasvisual));

for trial = 1:Par.intTrialNum
    RandPermOriChange(trial)    = randperm(length(Par.vecOriChange),1);
end
updown          = (round(rand(1,Par.intTrialNum))-0.5)*2;
Par.Stim.vecOriChange                           = Par.vecOriChange(RandPermOriChange) .* updown;
Par.Stim.vecOriPostChange(Par.Stim.hasvisual)   = Par.Stim.vecOriPreChange(Par.Stim.hasvisual) + Par.Stim.vecOriChange(Par.Stim.hasvisual);
Par.Stim.vecOriPostChange                       = mod(Par.Stim.vecOriPostChange,360); % Make circular

%% Distribute auditory frequencies over trial types that have audio pre and or post change 
for trial = 1:Par.intTrialNum
    RandPermFreq(trial)    = randperm(length(Par.vecCenterFreq),1);
end
Par.Stim.vecFreqPreChange(Par.Stim.hasauditory)        = Par.vecCenterFreq(RandPermFreq(Par.Stim.hasauditory));

for trial = 1:Par.intTrialNum
    RandPermFreqChange(trial)    = randperm(length(Par.vecFreqChange),1);
end
updown          = (round(rand(1,Par.intTrialNum))-0.5)*2;
Par.Stim.vecFreqChange = Par.vecFreqChange(RandPermFreqChange) .* updown;
Par.Stim.vecFreqPostChange(Par.Stim.hasauditory)    = Par.Stim.vecFreqPreChange(Par.Stim.hasauditory) + Par.Stim.vecFreqChange(Par.Stim.hasauditory);
Par.Stim.vecFreqPostChange    =  mod(Par.Stim.vecFreqPostChange-min(Par.vecCenterFreq),8000)+min(Par.vecCenterFreq); % Make circular

%% Verify with histogram that changes are distributed:
% close all; figure;
% histogram(Par.Stim.vecOriPostChange-Par.Stim.vecOriPreChange,1000)
% close all; figure;
% histogram(Par.Stim.vecFreqPostChange-Par.Stim.vecFreqPreChange,1000)
% close all; figure;
% histogram(Par.Stim.vecOriPostChange,1000)
% close all; figure;
% histogram(Par.Stim.vecFreqPostChange,1000)

%% Set the side of presentation
if Par.VisualLeftCorrectSide == 1
    Par.Stim.vecVisualSide(1:Par.intTrialNum)        = 'L';
    Par.Stim.vecAuditorySide(1:Par.intTrialNum)      = 'R';
else
    Par.Stim.vecVisualSide(1:Par.intTrialNum)        = 'R';
    Par.Stim.vecAuditorySide(1:Par.intTrialNum)      = 'L';
end

end
