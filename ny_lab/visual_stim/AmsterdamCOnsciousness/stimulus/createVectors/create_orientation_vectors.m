%% Create vectors with orientation to present (999 = no stim)
function vecPresStimOri = create_orientation_vectors(Par)

vecRandOri = randi(length(Par.vecOrientations),1,length(Par.Stim.vecType)); %randomly assign one of the ori's to each trial
vecPresStimOri = ones(1,length(Par.Stim.vecType))*999;
vecPresStimOri(Par.Stim.vecType == 'V') = Par.vecOrientations(vecRandOri(Par.Stim.vecType == 'V'));
end