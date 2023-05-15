function [Par] = create_trialtypes_change(Par)

Par.Stim            = struct;

% Parameters that govern the trial structure generation:
% Par.UseBimodalTrials      Trial code = 'X' (Visual Change) or 'Y' (Auditory Change)
% Par.UseUnimodalTrials     Trial code = 'V' (Only Visual) or 'A' (Only Auditory)
% Par.UseBiBiProbeTrials    Trial code = 'P'
% Par.UseUniBiProbeTrials   Trial code = 'Q' (Only Visual) or 'R' (Only Auditory)
% Par.UseConflictTrials     Trial code = 'C'
% (+ their respective fractions of incidence)

%% Calc total trial num
if Par.UseTrialBlocks
    Par.nBlocksModality         = ceil(800/Par.BlockSize*0.5);        %Number of  blocks
    Par.intTrialNum             = (Par.BlockSize*Par.nBlocksModality*2);
else
    Par.intTrialNum             = 800;      %make sure sufficient number of trials are generated
end

%% Generate trial sequence: (blockwise or pseudorandom)
if Par.UseTrialBlocks
       
    if isfield(Par,'intFirstBlock')
        intFirstBlock = Par.intFirstBlock;
    else intFirstBlock = round(rand);
    end
    
    if intFirstBlock 
        Par.Stim.vecTrialType           = repmat([repmat('X',1,Par.BlockSize) repmat('Y',1,Par.BlockSize)],1,Par.nBlocksModality);
    else
        Par.Stim.vecTrialType           = repmat([repmat('Y',1,Par.BlockSize) repmat('X',1,Par.BlockSize)],1,Par.nBlocksModality);
    end
    
    % Implement Unimodal, BiProbe, UniProbe and/or Conflict Trials
    % Strategy: Look for normal trials and replace by special trials. Randperm
    % is not sensitive for multiple trial types in a row!
    if Par.UseUnimodalTrials
        AllNormalTrials     = find(Par.Stim.vecTrialType == 'X');
        SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionUnimodalTrials*numel(AllNormalTrials)));
        Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'V';
        
        AllNormalTrials     = find(Par.Stim.vecTrialType == 'Y');
        SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionUnimodalTrials*numel(AllNormalTrials)));
        Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'A';
    end
    
    if Par.UseBiProbeTrials
        AllNormalTrials     = find(Par.Stim.vecTrialType == 'X');
        SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionBiProbeTrials*numel(AllNormalTrials)));
        Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'P';
        
        AllNormalTrials     = find(Par.Stim.vecTrialType == 'Y');
        SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionBiProbeTrials*numel(AllNormalTrials)));
        Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'P';
    end
    
    if Par.UseUniProbeTrials    %Unimodal Probe trials:
        AllNormalTrials     = find(Par.Stim.vecTrialType == 'V');
        SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionUniProbeTrials*numel(AllNormalTrials)));
        Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'Q';
        
        AllNormalTrials     = find(Par.Stim.vecTrialType == 'A');
        SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionUniProbeTrials*numel(AllNormalTrials)));
        Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'R';
    end

    if Par.UseConflictTrials
        AllNormalTrials     = find(Par.Stim.vecTrialType == 'X');
        SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionConflictTrials*numel(AllNormalTrials)));
        Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'C';
        
        AllNormalTrials     = find(Par.Stim.vecTrialType == 'Y');
        SelectedTrials      = randperm(numel(AllNormalTrials),round(Par.FractionConflictTrials*numel(AllNormalTrials)));
        Par.Stim.vecTrialType(AllNormalTrials(SelectedTrials)) = 'C';
    end
    
else %Pseudorandom sequence of trials
    block = 10;
    seq = [];
    if Par.UseUnimodalTrials
        seq = [seq repmat('AV',1,ceil(Par.FractionUnimodalTrials*block/2))];
    end
    if Par.UseBiProbeTrials
        seq = [seq repmat('P',1,ceil(Par.FractionBiProbeTrials*block))];
    end
    if Par.UseUniProbeTrials
        seq = [seq repmat('QR',1,ceil(Par.FractionBiProbeTrials*block/2))];
    end
    if Par.UseConflictTrials
        seq = [seq repmat('C',1,ceil(Par.FractionConflictTrials*block))];
    end
    
    seq = [seq repmat('XY',1,ceil((block-length(seq))/2))];
    seq = seq(1:block);
    
    Par.Stim.vecTrialType = [];
    for bl = 1:ceil(Par.intTrialNum/block)
        Par.Stim.vecTrialType = [Par.Stim.vecTrialType seq(randperm(10))];
    end
    
end


%% Assign changes and modalities to trials, i.e. convert strings to logical commands:
Par.Stim.visualchange           = false(1,Par.intTrialNum);
Par.Stim.hasvisual              = false(1,Par.intTrialNum);
Par.Stim.hasauditory            = false(1,Par.intTrialNum);
Par.Stim.audiochange            = false(1,Par.intTrialNum);
Par.Stim.hastactile             = false(1,Par.intTrialNum);

%Visual
Par.Stim.hasvisual(   Par.Stim.vecTrialType == 'X'... %All but auditory trials
                    | Par.Stim.vecTrialType == 'Y'...
                    | Par.Stim.vecTrialType == 'V'... 
                    | Par.Stim.vecTrialType == 'P'...
                    | Par.Stim.vecTrialType == 'C'...
                    | Par.Stim.vecTrialType == 'Q'...
                    ) = true; 
                
Par.Stim.visualchange(   Par.Stim.vecTrialType == 'X'...
                    | Par.Stim.vecTrialType == 'V'...
                    | Par.Stim.vecTrialType == 'C'...
                    ) = true;

%Auditory
Par.Stim.hasauditory(   Par.Stim.vecTrialType == 'X'... %All but visual trials
                    | Par.Stim.vecTrialType == 'Y'...
                    | Par.Stim.vecTrialType == 'A'... 
                    | Par.Stim.vecTrialType == 'P'...
                    | Par.Stim.vecTrialType == 'C'...
                    | Par.Stim.vecTrialType == 'R'...
                    ) = true;

Par.Stim.audiochange(   Par.Stim.vecTrialType == 'Y'...
                    | Par.Stim.vecTrialType == 'A'...
                    | Par.Stim.vecTrialType == 'C'...
                    ) = true;

%% Define correct sides to respond:            
Par.Stim.leftCorrect                = false(1,Par.intTrialNum);
Par.Stim.rightCorrect               = false(1,Par.intTrialNum);

if Par.VisualLeftCorrectSide
    Par.Stim.leftCorrect(Par.Stim.visualchange)     = true;
    Par.Stim.rightCorrect(Par.Stim.audiochange)     = true;
else
    Par.Stim.leftCorrect(Par.Stim.audiochange)      = true;
    Par.Stim.rightCorrect(Par.Stim.visualchange)    = true;
end


end