function [Par] = create_stimtypes_mod_discr(Par)

%% Initialize variables:
Par.Stim.vecOri             = NaN(1,Par.intTrialNum);
Par.Stim.vecFreq            = NaN(1,Par.intTrialNum);
RandPermOri                 = NaN(1,Par.intTrialNum);
RandPermFreq                = NaN(1,Par.intTrialNum);

Par.Stim.vecVisStimInt      = NaN(1,Par.intTrialNum);
Par.Stim.vecAuStimInt       = NaN(1,Par.intTrialNum);
RandPermVisInt              = NaN(1,Par.intTrialNum);
RandPermAuInt               = NaN(1,Par.intTrialNum);

%% Distribute orientations over trial types 

for trial = 1:Par.intTrialNum
    RandPermOri(trial)    = randperm(length(Par.vecOrientations),1);
end
Par.Stim.vecOri(Par.Stim.hasvisual)             = Par.vecOrientations(RandPermOri(Par.Stim.hasvisual));

%% Distribute visual intensties over trial types 

for trial = 1:Par.intTrialNum
    RandPermVisInt(trial)    = randperm(length(Par.vecVisStimIntensities),1);
end
Par.Stim.vecVisStimInt(Par.Stim.hasvisual)             = Par.vecVisStimIntensities(RandPermVisInt(Par.Stim.hasvisual));

%% Distribute auditory frequencies over trial types 
for trial = 1:Par.intTrialNum
    RandPermFreq(trial)    = randperm(length(Par.vecCenterFreq),1);
end
Par.Stim.vecFreq(Par.Stim.hasauditory)          = Par.vecCenterFreq(RandPermFreq(Par.Stim.hasauditory));

%% Distribute auditory intensities over trial types 
for trial = 1:Par.intTrialNum
    RandPermAuInt(trial)    = randperm(length(Par.vecAuStimIntensities),1);
end
Par.Stim.vecAuStimInt(Par.Stim.hasauditory)          = Par.vecAuStimIntensities(RandPermAuInt(Par.Stim.hasauditory));

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
