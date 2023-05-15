%% Create vectors with pitch to present (999 = no stim)
function vecPresStimAudio = create_audio_vectors(Par)

vecPresStimAudio = ones(1,length(Par.vecPresStimType))*999;
if strcmp(Par.strTask, 'localization')
    % Determine side if this is for a localization task (i.e. random 1 or 2)
    vecPresStimAudio(Par.vecPresStimType == 2) = (rand(1,sum(Par.vecPresStimType == 2)) < 0.5) +1;
else
    % If this is a detection task, the auditory stimulus is presented on both
    % speakers and there is no pitch
    vecPresStimAudio(Par.vecPresStimType == 2) = 111;
end

end