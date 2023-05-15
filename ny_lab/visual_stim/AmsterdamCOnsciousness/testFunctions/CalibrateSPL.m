%% CalibrateSPL

    Par = mol_par_mod_discr;
    
    %% INITIALIZE SCREEN
    %open screen
    AssertOpenGL;
    KbName('UnifyKeyNames');
    Screen('Preference', 'SkipSyncTests', 0)
    % get screen id
    Par.intUseScreen = max(Screen('Screens'));

    [Objects.ptrWindow , windowRect] = Screen('OpenWindow',Par.intUseScreen,255);  % This is the psychtoolbox window object
    
    %% INITIALIZE AUDIO
    % Initialize Sounddriver
    InitializePsychSound;
    DeviceInfo = struct2cell(PsychPortAudio('GetDevices')); %Get available sound ports
    AudioDevices = squeeze(DeviceInfo(4,:,:));
    intDevice = find(~cellfun(@isempty,strfind(AudioDevices,'Multichannel Playback')))-1; %Get the right audiocard
    if intDevice
        Par.intSamplingRate = DeviceInfo {11,1,intDevice}; %Get sampling rate of audiocard (48kHz for Audigy)
    else Par.intSamplingRate = 44100; %Default sampling rate if Audigy Soundcard is not detected
    end
    LatencyLevel = 3;
    intChannelNum = 2; % Number of channels
    Objects.ptrAudio = PsychPortAudio('Open', intDevice, [], LatencyLevel, Par.intSamplingRate, intChannelNum);
    
    intThisTrial = 1;
    Par.Stim.hasauditory(intThisTrial) = 1;
    Par.Stim.vecTrialType(intThisTrial) = 'A';
    
%    Par.Stim.vecFreq(intThisTrial)    = 8000;
    Par.Stim.vecFreq(intThisTrial) = 10000;
%    Par.Stim.vecFreq(intThisTrial) = 12000;
 %    Par.Stim.vecFreq(intThisTrial) = 14000;
%    Par.Stim.vecFreq(intThisTrial) = 15800;

    psyau = [0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9];
    
    for int = psyau;
      disp(int)
      Par.Stim.vecAuStimInt(intThisTrial) = int;
      varaudio            = load_auditory(Par,intThisTrial);
      PsychPortAudio('FillBuffer', Objects.ptrAudio, varaudio.vecSound);
      PsychPortAudio('Start', Objects.ptrAudio, 1, 0, 1);
      pause(4)
      PsychPortAudio('Stop', Objects.ptrAudio);
    end
    
    try Screen('CloseAll'); PsychPortAudio('Close');
    catch; end
    
    