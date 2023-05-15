function dl_initialize_session

global uihandles
if exist('OCTAVE_VERSION', 'builtin') ~= 0; warning("off","Octave:num-to-str"); warning("off","Octave:divide-by-zero"); end %Turn off nonsense warning

try
    % Get time vector
    vecClock = clock;
    
    %% Get info about the session:
    sessionData                 = struct();
    sessionData.Mouse           = get(uihandles.choose_mouse,'String');
    sessionData.Date            = sprintf('%04d%02d%02d',vecClock(1),vecClock(2),vecClock(3));
    sessionData.Hour            = sprintf('%02d%02d',vecClock(4),vecClock(5));
    sessionData.Box             = get(uihandles.choose_box,'Value');
    sessionData.MouseWeight     = get(uihandles.mouse_weight,'String');
    sessionData.Experimenter    = uihandles.Experimenters{get(uihandles.choose_experimenter,'Value')};
    sessionData.Parameterfile   = uihandles.Parameters{get(uihandles.choose_parameters,'Value')};
    sessionData.Protocolfile    = uihandles.Protocols{get(uihandles.choose_protocol,'Value')};
    sessionData.TrainingStage   = get(uihandles.choose_trainingstage,'Value');
    
    try %try saving in Dropbox
        strSaveDir = fullfile(uihandles.pcsettings.dropboxFolder,sessionData.Experimenter);
    catch; strSaveDir = uigetdir();
    end
    
    % Define output filename
    strFilename = sprintf('%04d%02d%02d_%02d%02d_%s',vecClock(1),vecClock(2),vecClock(3),vecClock(4),vecClock(5),sessionData.Mouse);
    if isa(strFilename,'char') && ~isempty(strFilename)
        if exist([strSaveDir filesep strFilename],'file') || exist([strSaveDir filesep strFilename '.mat'],'file')
            strFilename = [strFilename, '_1']; % add a one to the filename
        end
    end
    
    fprintf('Saving output to file "%s.mat"\n',strFilename);
    sessionData.FullFileName = fullfile(strSaveDir,strcat(strFilename,'.mat'));
    
    %% Get parameters     (If trainingstage specified it will take parameters of the corresponding training stage (MOL))
    if strfind(sessionData.Protocolfile,'mol')
        Par = struct(); Par.Mouse = sessionData.Mouse; eval(strcat('Par=',sessionData.Parameterfile,'(Par,sessionData.TrainingStage);'));
    else eval(strcat('Par=',sessionData.Parameterfile,';'));
    end
    Par.pcsettings = uihandles.pcsettings; %Transfer PC settings to parameters file
    set(uihandles.updateline1,'String','Parameters OK');

    %% Set up Objects struct
    Objects = struct;
    
    %% Setup lick detector
%     Objects.ldObj = LickDetector; % This constructs the LickDetector object
    Objects.ldObj = LickDetector('serialStr',Par.pcsettings.ldserialStr,'box',sessionData.Box); % This constructs the LickDetector object

    set(uihandles.updateline2,'String','LickDetector OK');
    
    %% Setup servo controller
    if Par.UseServo %#ok<*NODEF>
        Objects.servoObj = ServoController;
        %Put servo in far position, and wait because the classbuilding is like stupid slow
        pause(2);
        Objects.servoObj.moveServo('F');
        set(uihandles.updateline3,'String','ServoController OK');
    else
        set(uihandles.updateline3,'String','ServoController not used');
    end
    
    %% Setup piezo controller
    if Par.UsePiezo %#ok<*NODEF>
        Objects.pzObj = PiezoController('serialStr',Par.pcsettings.pzserialStr);
        set(uihandles.updateline4,'String','PiezoController OK');
    else
        set(uihandles.updateline4,'String','PiezoController not used');
    end
    
    %% Setup PsychToolbox
    fprintf('Starting PsychToolBox extension...\n');
    
    %% INITIALIZE SCREEN
    %open screen
    AssertOpenGL;
    KbName('UnifyKeyNames');
    Screen('Preference', 'SkipSyncTests', 0)
    % get screen id
    Par.intUseScreen = max(Screen('Screens'));

    [Objects.ptrWindow , windowRect] = Screen('OpenWindow',Par.intUseScreen,Par.bgInt);  % This is the psychtoolbox window object
    Par.FrameDur = Screen('GetFlipInterval', Objects.ptrWindow); %Get interval between frames ~1/framerate
    
    %size of screen variables
    Par.intScreenWidth_pix          = windowRect(3);
    Par.intScreenHeight_pix         = windowRect(4);
    
    %start with blank screen
    Screen('FillRect',Objects.ptrWindow, Par.bgInt);
    Screen('Flip',Objects.ptrWindow);
    
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
    
    % Open Psych-Audio port, with the follow arguements
    % (1) [] = default sound device
    % (2) 1 = sound playback only
    % (3) 1 = default level of latency
    % (4) Requested frequency in samples per second
    % (5) 2 = stereo output
    
    if strcmp(Par.strTask,'mol_ch_det')
        Objects.ptrAudio = PsychPortAudio('Open', intDevice, 1+8, LatencyLevel, Par.intSamplingRate, intChannelNum);
        PsychPortAudio('Start', Objects.ptrAudio, 0, 0, 1);
        Objects.ptrAudioPre = PsychPortAudio('OpenSlave',Objects.ptrAudio, 1);
        Objects.ptrAudioPost = PsychPortAudio('OpenSlave',Objects.ptrAudio, 1);
    else
        Objects.ptrAudio = PsychPortAudio('Open', intDevice, [], LatencyLevel, Par.intSamplingRate, intChannelNum);
    end
    set(uihandles.updateline5,'String','PsychToolBox OK');
    
    %% Close SetupFig
    pause(0.5)
    close(uihandles.setup_fig); clear uihandles;
    
    %% Setup Control Window
    if strfind(Par.strTask,'mol') || strfind(Par.strTask,'pie')
        Objects.cwObj = ControlWindowMatthijs(Objects.ldObj, Par, sessionData);
    else
        Objects.cwObj = ControlWindow(Objects.ldObj, Par, sessionData);
    end
    
    %% Wait for mouse to calm down
    intCountDown = Par.preTrainingSecs;
    waitRef = tic;
    fprintf('\nWaiting %d seconds before starting experiment:\n %03d seconds remaining',Par.preTrainingSecs,intCountDown);
    while toc(waitRef) < Par.preTrainingSecs
        pause(0.1);
        if (Par.preTrainingSecs + 1 - toc(waitRef)) < intCountDown
            intCountDown = intCountDown - 1;
            fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b%03d seconds remaining',intCountDown);
        end
    end
    
catch errorMessage
    %% Catch me and Throw me
    fprintf('\nRegistered error during initialization...\n');
    
    %Clean up
    fprintf('\nTrying to close down and clean up...\n');
    try Screen('CloseAll'); PsychPortAudio('Close');
    catch; end
    
    if exist('Objects','var')
        ObjectFields = fieldnames(Objects);
        for obj = 1:length(ObjectFields)
            try delete(Objects.(ObjectFields{obj})); clear Objects.(ObjectFields{obj};
            catch; end
        end
    end
    
    %Show error
    rethrow(errorMessage);
    
end

%% Run actual task:
if ~exist('errorMessage','var')
    %starting message
    vecTime = clock;
    fprintf('\nStarting training session (type %s) now: time is %02.0f:%02.0f\n',sessionData.Protocolfile,vecTime(4),vecTime(5));
    %Start selected protocol
    eval(strcat(sessionData.Protocolfile,'(Par,Objects,sessionData);'));
end

end