%% Load the auditory stimulus
% Return varAudio, a struct which contains the information about the audio
% stimulus
function varAudio = load_auditory(Par,intThisTrial)

if isfield(Par,'strAuStimType')    % Generate Audio

    varAudio = []; % variable for storing auditory stimulus
    % Use sufficient time for stimulus, stimulus is stopped by task script
    Stimdur     =   3; %seconds
    
    %Generate auditory stimulus Pre-Change
    if strcmp(Par.strAuStimType,'bandpassfreq')
        vecSoundOneSide = white_noise_band_ramp(Stimdur, Par.intSamplingRate,...
            Par.Stim.vecFreq(intThisTrial)-Par.FreqBandwidth/2,...
            Par.Stim.vecFreq(intThisTrial)+Par.FreqBandwidth/2,...
            Par.dblAudioRampDur);
    elseif strcmp(Par.strAuStimType,'shepardtone')
        window               = Par.ShepardWeights(find(Par.vecCenterFreq == Par.Stim.vecFreq(intThisTrial)):length(Par.vecCenterFreq):Par.ShepardTones*length(Par.vecCenterFreq));
        vecSoundOneSide      = shepardtone(Par.Stim.vecFreq(intThisTrial),Par.ShepardTones,'length',Stimdur,'samplerate',Par.intSamplingRate,'weight',window);
        vecSoundOneSide      = vecSoundOneSide';
    end
    
    %if the first one is tuned down compared to the second:
    mappingintenstodb = 1/(3^((1-Par.Stim.vecAuStimInt(intThisTrial))*10));
    vecSoundOneSide = vecSoundOneSide * mappingintenstodb;
    
    % Set auditory intensity for both sides and store in variable varaudio
    % In psychtoolbox is first row right, second row left
    if Par.Stim.vecAuditorySide(1:Par.intTrialNum) == 'L'  % left
        varAudio.vecSound    = [vecSoundOneSide * Par.StimSpanSides;   vecSoundOneSide];
    elseif Par.Stim.vecAuditorySide(1:Par.intTrialNum) == 'R'  % right
        varAudio.vecSound    = [vecSoundOneSide;   vecSoundOneSide * Par.StimSpanSides];
    end
    
else
    % retrieve side for current trial
    charSide = Par.Stim.vecSide(intThisTrial);
    varAudio = []; % variable for storing auditory stimulus
    
    %Generate auditory stimulation (white noise)
    vecSoundOneSide = white_noise_band_ramp(Par.dblSecsStimDur, Par.intSamplingRate,...
        Par.vecNoiseRange(1),Par.vecNoiseRange(2),Par.dblAudioRampDur); % in other file
    % right
    if charSide == 'R'
        varAudio.vecSound = [vecSoundOneSide; zeros(1,numel(vecSoundOneSide))];
        % left
    elseif charSide == 'L'
        varAudio.vecSound = [zeros(1,numel(vecSoundOneSide)); vecSoundOneSide];
        % center
    elseif charSide == 'C'
        varAudio.vecSound = [vecSoundOneSide; vecSoundOneSide];
    end
    varAudio.intAudioChannel = Par.intAudioChannel;
    
end








end
