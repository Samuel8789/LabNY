%% Create vector with side to present stimulus
function vecStimSide = create_side_vectors(Par)
    % Determine side  (i.e. random L or R )
    vecSideSymbol = ['L', 'R'];
    vecRandom = (rand(1,numel(Par.Stim.vecType)) < 0.5) +1;
    vecStimSide = vecSideSymbol(vecRandom);
end