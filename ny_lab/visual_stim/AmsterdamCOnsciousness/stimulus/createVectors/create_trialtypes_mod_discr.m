function [Par] = create_trialtypes_mod_discr(Par)

Par.Stim            = struct;

% Parameters that govern the trial structure generation:
% Standard trial types:     Trial code = 'A' (Only Auditory) or 'V' (Only Visual)
% Par.UseProbeTrials        Trial code = 'P'
% Par.UseConflictTrials     Trial code = 'C'
% (+ their respective fractions of incidence)

%% Calc total trial num
if Par.UseTrialBlocks
    Par.nBlocksModality         = ceil(1000/Par.BlockSize*0.5);        %Number of  blocks
    Par.intTrialNum             = (Par.BlockSize*Par.nBlocksModality*2);
else
    Par.intTrialNum             = 1000;      %make sure sufficient number of trials are generated
end

%% Generate trial sequence: (blockwise or pseudorandom)
if Par.UseTrialBlocks
       
    if isfield(Par,'intFirstBlock')
        intFirstBlock = Par.intFirstBlock;
    else intFirstBlock = round(rand);
    end
    
    if intFirstBlock 
        Par.Stim.vecTrialType           = repmat([repmat('V',1,Par.BlockSize) repmat('A',1,Par.BlockSize)],1,Par.nBlocksModality);
    else
        Par.Stim.vecTrialType           = repmat([repmat('A',1,Par.BlockSize) repmat('V',1,Par.BlockSize)],1,Par.nBlocksModality);
    end
    
    % Implement Probe and/or Conflict Trials
    % Strategy: Look for normal trials and replace by special trials. Randperm
    % is not sensitive for multiple trial types in a row!
    
    if Par.UseProbeTrials
        AllNormalTrials     = find(Par.Stim.vecTrialType == 'V');
        SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionProbeTrials*numel(AllNormalTrials)));
        Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'P';
        
        AllNormalTrials     = find(Par.Stim.vecTrialType == 'A');
        SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionProbeTrials*numel(AllNormalTrials)));
        Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'P';
    end

    if Par.UseConflictTrials
        AllNormalTrials     = find(Par.Stim.vecTrialType == 'V');
        SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionConflictTrials*numel(AllNormalTrials)));
        Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'C';
        
        AllNormalTrials     = find(Par.Stim.vecTrialType == 'A');
        SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionConflictTrials*numel(AllNormalTrials)));
        Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'C';
    end
    
else %Pseudorandom sequence of trials
    block = 10;
    seq = [];
    if Par.UseProbeTrials
        seq = [seq repmat('P',1,ceil(Par.FractionProbeTrials*block))];
    end
    if Par.UseConflictTrials
        seq = [seq repmat('C',1,ceil(Par.FractionConflictTrials*block))];
    end
    
    seq = [seq repmat('VA',1,ceil((block-length(seq))/2))];
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
Par.Stim.hasvisual(   Par.Stim.vecTrialType == 'V'... %All but Auditory trials
                    | Par.Stim.vecTrialType == 'C'...
                    ) = true; 
                
%Auditory
Par.Stim.hasauditory(   Par.Stim.vecTrialType == 'A'... %All but visual trials
                    | Par.Stim.vecTrialType == 'C'...
                    ) = true;

%% Define correct sides to respond:            
Par.Stim.leftCorrect                = false(1,Par.intTrialNum);
Par.Stim.rightCorrect               = false(1,Par.intTrialNum);

if Par.VisualLeftCorrectSide
    Par.Stim.leftCorrect(Par.Stim.hasvisual)        = true;
    Par.Stim.rightCorrect(Par.Stim.hasauditory)     = true;
else
    Par.Stim.leftCorrect(Par.Stim.hasauditory)      = true;
    Par.Stim.rightCorrect(Par.Stim.hasvisual)    	= true;
end


end