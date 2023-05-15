function [Par] = create_trialtypes_sifi(Par)

Par.Stim            = struct;

% Parameters that govern the trial structure generation:
% Par.UseBimodalTrials   Trial code = 'X' (Visual Change) or 'Y' (Auditory Change)
% Par.UseUnimodalTrials  Trial code = 'A' (Only Auditory) or 'V' (Only Visual)
% Par.UseProbeTrials     Trial code = 'P'
% Par.UseConflictTrials  Trial code = 'C'
% (+ their respective fractions of incidence)

Par.nConditions = numel(Par.vecFlashes)* numel(Par.vecBeeps);

%% Calc total trial num
if Par.UseTrialBlocks
    Par.nBlocksperCondition     = ceil(600/Par.BlockSize/Par.nConditions);        %Number of  blocks
    Par.intTrialNum             = Par.BlockSize*Par.nBlocksperCondition*Par.nConditions;
else
    Par.intTrialNum             = 600;      %make sure sufficient number of trials are generated
end

%% Generate trial sequence: (blockwise or pseudorandom)
if Par.UseTrialBlocks
       
    if isfield(Par,'intFirstBlock')
        intFirstBlock = Par.intFirstBlock;
    else intFirstBlock = round(rand);
    end
    if intFirstBlock
        Par.Stim.vecTrialType           = repmat([repmat('S',1,Par.BlockSize) repmat('S',1,Par.BlockSize)],1,Par.nBlocksperCondition);
    else
        Par.Stim.vecTrialType           = repmat([repmat('D',1,Par.BlockSize) repmat('S',1,Par.BlockSize)],1,Par.nBlocksperCondition);
    end
    
    % Implement Unimodal, Probe and/or Conflict Trials
    % Strategy: Look for normal trials and replace by special trials. Randperm
    % is not sensitive for multiple trial types in a row!
%     if Par.UseUnimodalTrials
%         AllNormalTrials     = find(Par.Stim.vecTrialType == 'X');
%         SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionUnimodalTrials*numel(AllNormalTrials)));
%         Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'V';
%         
%         AllNormalTrials     = find(Par.Stim.vecTrialType == 'Y');
%         SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionUnimodalTrials*numel(AllNormalTrials)));
%         Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'A';
%     end
%     
%     if Par.UseProbeTrials
%         AllNormalTrials     = find(Par.Stim.vecTrialType == 'X');
%         SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionProbeTrials*numel(AllNormalTrials)));
%         Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'P';
%         
%         AllNormalTrials     = find(Par.Stim.vecTrialType == 'Y');
%         SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionProbeTrials*numel(AllNormalTrials)));
%         Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'P';
%     end
%     
%     if Par.UseConflictTrials
%         AllNormalTrials     = find(Par.Stim.vecTrialType == 'X');
%         SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionConflictTrials*numel(AllNormalTrials)));
%         Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'C';
%         
%         AllNormalTrials     = find(Par.Stim.vecTrialType == 'Y');
%         SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionConflictTrials*numel(AllNormalTrials)));
%         Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'C';
%     end
else %Pseudorandom sequence of trials
    block = 10;
    seq = [];
%     if Par.UseUnimodalTrials
%         seq = [seq repmat('AV',1,ceil(Par.FractionUnimodalTrials*block/2))];
%     end
%     if Par.UseProbeTrials
%         seq = [seq repmat('P',1,ceil(Par.FractionProbeTrials*block))];
%     end
%     if Par.UseConflictTrials
%         seq = [seq repmat('C',1,ceil(Par.FractionConflictTrials*block))];
%     end
    
    seq = [seq repmat('SD',1,ceil((block-length(seq))/2))];
    seq = seq(1:block);
    
    Par.Stim.vecTrialType = [];
    for bl = 1:ceil(Par.intTrialNum/block)
        Par.Stim.vecTrialType = [Par.Stim.vecTrialType seq(randperm(10))];
    end
    
end


%% Assign modalities to trials, i.e. convert strings to logical commands:
Par.Stim.hasvisual              = false(1,Par.intTrialNum);
Par.Stim.hasauditory            = false(1,Par.intTrialNum);
Par.Stim.hastactile             = false(1,Par.intTrialNum);

%Visual
Par.Stim.hasvisual(   Par.Stim.vecTrialType == 'S'...
                    | Par.Stim.vecTrialType == 'D'...
                    ) = true; 

%Visual
Par.Stim.hasauditory( Par.Stim.vecTrialType == 'G'...
                    | Par.Stim.vecTrialType == 'Z'...
                    ) = true; 

Par.Stim.nFlashes               = zeros(1,Par.intTrialNum);
Par.Stim.nBeeps                 = zeros(1,Par.intTrialNum);

Par.Stim.nFlashes(   Par.Stim.vecTrialType == 'S'...
                    ) = 1; 
Par.Stim.nFlashes(   Par.Stim.vecTrialType == 'D'...
                    ) = 2;
                
Par.Stim.nBeeps(   Par.Stim.vecTrialType == 'G'...
                    ) = 1; 
Par.Stim.nBeeps(   Par.Stim.vecTrialType == 'Z'...
                    ) = 2;
                
%% Define correct sides to respond:            
Par.Stim.leftCorrect                = false(1,Par.intTrialNum);
Par.Stim.rightCorrect               = false(1,Par.intTrialNum);

if Par.OneFlashLeft
    Par.Stim.leftCorrect(Par.Stim.nFlashes == 1)     = true;
    Par.Stim.rightCorrect(Par.Stim.nFlashes == 2)    = true;
else
    Par.Stim.leftCorrect(Par.Stim.nFlashes == 2)     = true;
    Par.Stim.rightCorrect(Par.Stim.nFlashes == 1)    = true;
end

end