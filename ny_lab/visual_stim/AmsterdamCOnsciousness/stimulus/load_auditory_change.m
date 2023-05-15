%% Load the auditory stimulus
% Return varAudio, a struct which contains the information about the audio
% stimulus
function varAudio = load_auditory_change(Par,intThisTrial)

varAudio = []; % variable for storing auditory stimulus

%% Generate Pre-Change

% Use sufficient time for stimulus, stimulus is stopped by task script
Stimdur     =   7; %seconds

%Generate auditory stimulus Pre-Change
if strcmp(Par.strAuStimType,'bandpassfreq')
    vecSoundOneSidePre = white_noise_band_ramp(Stimdur, Par.intSamplingRate,...
    Par.Stim.vecFreqPreChange(intThisTrial)-Par.FreqBandwidth/2,...
    Par.Stim.vecFreqPreChange(intThisTrial)+Par.FreqBandwidth/2,...
    Par.dblAudioRampDur);
elseif strcmp(Par.strAuStimType,'shepardtone')
    window                  = Par.ShepardWeights(find(Par.vecCenterFreq == Par.Stim.vecFreqPreChange(intThisTrial)):length(Par.vecCenterFreq):Par.ShepardTones*length(Par.vecCenterFreq));
    vecSoundOneSidePre      = shepardtone(Par.Stim.vecFreqPreChange(intThisTrial),Par.ShepardTones,'length',Stimdur,'samplerate',Par.intSamplingRate,'weight',window);
    vecSoundOneSidePre      = vecSoundOneSidePre';
end

%if the first one is tuned down compared to the second:
% vecSoundOneSidePre = vecSoundOneSidePre * Par.StimPreChangeIntensity^2;

%% Generate auditory stimulus Post-Change
if strcmp(Par.strAuStimType,'bandpassfreq')
    vecSoundOneSidePost = white_noise_band_ramp(Stimdur, Par.intSamplingRate,...
    Par.Stim.vecFreqPostChange(intThisTrial)-Par.FreqBandwidth/2,...
    Par.Stim.vecFreqPostChange(intThisTrial)+Par.FreqBandwidth/2,...
    Par.dblAudioRampDur/100);
elseif strcmp(Par.strAuStimType,'shepardtone')
    window                  = Par.ShepardWeights(find(Par.vecCenterFreq == Par.Stim.vecFreqPostChange(intThisTrial)):length(Par.vecCenterFreq):Par.ShepardTones*length(Par.vecCenterFreq));
    vecSoundOneSidePost     = shepardtone(Par.Stim.vecFreqPostChange(intThisTrial),Par.ShepardTones,'length',Stimdur,'samplerate',Par.intSamplingRate,'weight',window);
    vecSoundOneSidePost = vecSoundOneSidePost';
end

% Set auditory intensity for both sides and store in variable varaudio
%In psychtoolbox is first row right, second row left
if Par.Stim.vecAuditorySide(1:Par.intTrialNum) == 'L'  % left
    varAudio.vecSoundPre    = [vecSoundOneSidePre*Par.StimSpanSides^2;   vecSoundOneSidePre];
    varAudio.vecSoundPost   = [vecSoundOneSidePost*Par.StimSpanSides^2;  vecSoundOneSidePost];
elseif Par.Stim.vecAuditorySide(1:Par.intTrialNum) == 'R'  % right
    varAudio.vecSoundPre    = [vecSoundOneSidePre;   vecSoundOneSidePre*Par.StimSpanSides^2];
    varAudio.vecSoundPost   = [vecSoundOneSidePost;  vecSoundOneSidePost*Par.StimSpanSides^2];
end

end
