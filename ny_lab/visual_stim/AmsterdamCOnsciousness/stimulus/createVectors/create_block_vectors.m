%% CREATE_BLOCK_VECTORS() create vectors with stimulus type info
% Create a vector that contains the stimulus type info, i.e.
% visual/auditory/ or blank trial.
function vecPresStimType = create_block_vectors(Par)
intMaxSame = Par.intMaxSameBlank; %Maximum number of consecutive blank trials

%Create alternating blocks of visual only and audio only trials
intVisualBlocks = Par.intVisualBlocks;
intAudioBlocks = Par.intAudioBlocks;
vecPresStimType = [];
intFirstBlock = round(rand); %randomize whether visual or audio block is the first block
boolRunning = true;
startTime = tic;
while boolRunning
    if intFirstBlock %first audio block then visual block
        if intAudioBlocks ~= 0
            vecAddTrialBlock = Shuffle([ones(1,Par.intBlockSizeAudio-Par.intProbeTrialNumAudio)*2 ones(1,Par.intProbeTrialNumAudio)*5]);
            if Par.intProbeTrialNumAudio > 0
                while max(find(diff([2 vecAddTrialBlock 2])==-3)-find(diff([2 vecAddTrialBlock 2])==3)) > intMaxSame || sum(vecAddTrialBlock(1:2)) == 10 || sum(vecAddTrialBlock(end-1:end)) == 10
                    vecAddTrialBlock = Shuffle([ones(1,Par.intBlockSizeAudio-Par.intProbeTrialNumAudio)*2 ones(1,Par.intProbeTrialNumAudio)*5]);
                end
            end
            vecPresStimType = cat(2,vecPresStimType,vecAddTrialBlock);
            intAudioBlocks = intAudioBlocks - 1;
        end
        if intVisualBlocks ~= 0
            vecAddTrialBlock = Shuffle([ones(1,Par.intBlockSizeVisual-Par.intProbeTrialNumVisual) ones(1,Par.intProbeTrialNumVisual)*5]);
            if Par.intProbeTrialNumVisual > 0
                while max(find(diff([1 vecAddTrialBlock 1])==-4)-find(diff([1 vecAddTrialBlock 1])==4)) > intMaxSame || sum(vecAddTrialBlock(1:2)) == 10 || sum(vecAddTrialBlock(end-1:end)) == 10
                    vecAddTrialBlock = Shuffle([ones(1,Par.intBlockSizeVisual-Par.intProbeTrialNumVisual) ones(1,Par.intProbeTrialNumVisual)*5]);
                end
            end
            vecPresStimType = cat(2,vecPresStimType,vecAddTrialBlock);
            intVisualBlocks = intVisualBlocks - 1;
        end
    else %first visual block then audio block
        if intVisualBlocks ~= 0
            vecAddTrialBlock = Shuffle([ones(1,Par.intBlockSizeVisual-Par.intProbeTrialNumVisual) ones(1,Par.intProbeTrialNumVisual)*5]);
            if Par.intProbeTrialNumVisual > 0
                while max(find(diff([1 vecAddTrialBlock 1])==-4)-find(diff([1 vecAddTrialBlock 1])==4)) > intMaxSame || sum(vecAddTrialBlock(1:2)) == 10 || sum(vecAddTrialBlock(end-1:end)) == 10
                    vecAddTrialBlock = Shuffle([ones(1,Par.intBlockSizeVisual-Par.intProbeTrialNumVisual) ones(1,Par.intProbeTrialNumVisual)*5]);
                end
            end
            vecPresStimType = cat(2,vecPresStimType,vecAddTrialBlock);
            intVisualBlocks = intVisualBlocks - 1;
        end
        if intAudioBlocks ~= 0
            vecAddTrialBlock = Shuffle([ones(1,Par.intBlockSizeAudio-Par.intProbeTrialNumAudio)*2 ones(1,Par.intProbeTrialNumAudio)*5]);
            if Par.intProbeTrialNumAudio > 0
                while max(find(diff([2 vecAddTrialBlock 2])==-3)-find(diff([2 vecAddTrialBlock 2])==3)) > intMaxSame || sum(vecAddTrialBlock(1:2)) == 10 || sum(vecAddTrialBlock(end-1:end)) == 10
                    vecAddTrialBlock = Shuffle([ones(1,Par.intBlockSizeAudio-Par.intProbeTrialNumAudio)*2 ones(1,Par.intProbeTrialNumAudio)*5]);
                end
            end
            vecPresStimType = cat(2,vecPresStimType,vecAddTrialBlock);
            intAudioBlocks = intAudioBlocks - 1;
        end
    end
    if toc(startTime) > 10
        error('Trial sequence generation timed out. Try putting Par.intMaxSameBlank higher');
        break;
    end
    if intVisualBlocks == 0 && intAudioBlocks == 0
        boolRunning = false;
    end
end
% Loop over trials and change ints to chars
for iTrial = 1:numel(vecPresStimType)
    intStimType = vecPresStimType(iTrial);
    switch intStimType;
        % visual stim
        case 1
            vecPresStimType(iTrial) = 'V';
        % audio stim
        case 2
            vecPresStimType(iTrial) = 'A';
        % Probe trial
        case 5
            vecPresStimType(iTrial) = 'P';
    end
end
end