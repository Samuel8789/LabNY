%% This script is used to test sound presentation through psychtoolbox.
intChannelNum = 2; % Number of channels
dblSoundLevel = 0.4;
dblSecsStimDur = 1.5;
dblModulationRate = 1.5;
dblModulationIndex = 1000;
intSamplingRate = 44100;
vecNoiseRange = [4000 16000]; % range of noise in Hz
dblPitch = 15000;
dblRampDur = 0.001;
% INITIALIZE AUDIO
% Initialize Sounddriver
InitializePsychSound(1);
% Open Psych-Audio port, with the follow arguements
% (1) [] = default sound device
% (2) 1 = sound playback only
% (3) 1 = default level of latency
% (4) Requested frequency in samples per second
% (5) 2 = stereo output
ptrAudio = PsychPortAudio('Open', [], [], 0, intSamplingRate, intChannelNum);

%% Frequency modulated pure tone
%Generate auditory stimulation
%     vecTime = 0 : 1/intSamplingRate : dblSecsStimDur;
%     varAudio.vecSound  = sin((2 * pi * dblPitch * vecTime) ...
%         + (-dblModulationIndex * sin(2 * dblModulationRate * pi * vecTime)));
%     varAudio.vecSound = varAudio.vecSound * dblSoundLevel;

%% White noise
 vecSoundOneSide = white_noise_band_ramp(dblSecsStimDur, intSamplingRate,vecNoiseRange(1),vecNoiseRange(2),dblRampDur);
varAudio.vecSound = [zeros(1,numel(vecSoundOneSide)); vecSoundOneSide];
varAudio.intAudioChannel = 0;

% Fill the audio playback buffer with the audio data 'varAudio.vecSound':

PsychPortAudio('FillBuffer', ptrAudio, varAudio.vecSound);

PsychPortAudio('Start', ptrAudio, 1, 0, 1); % start audio
% Wait for the length of time of the beep is playing for
WaitSecs(dblSecsStimDur);
% Close psychportaudio
PsychPortAudio('Close')