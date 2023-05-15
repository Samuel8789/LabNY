function rd_av_uni_detection_block(Par,Objects,sessionData)
% Task description here

%% Preallocate struct with arrays for each event/info 
%  Pre-allocate the fields that will be used by your task
vecTrial = struct;
vecTrial.trialStart = zeros(1,Par.intTrialNum);
vecTrial.trialEnd = zeros(1,Par.intTrialNum);
vecTrial.itiStart = zeros(1,Par.intTrialNum);
vecTrial.itiEnd = zeros(1,Par.intTrialNum);
vecTrial.preStimStart = zeros(1,Par.intTrialNum);
vecTrial.stimStart = zeros(1,Par.intTrialNum);
vecTrial.lickTime = cell(1,Par.intTrialNum);
vecTrial.lickSide = cell(1,Par.intTrialNum);
vecTrial.correct = zeros(1,Par.intTrialNum);
vecTrial.rewardSide = zeros(1,Par.intTrialNum);
vecTrial.rewardTime = zeros(1,Par.intTrialNum);
vecTrial.rewardSize = zeros(1,Par.intTrialNum);
vecTrial.servoCloseTime = zeros(1,Par.intTrialNum);
vecTrial.servoFarTime = zeros(1,Par.intTrialNum);
vecTrial.noResponse = zeros(1,Par.intTrialNum);

%% Task internal variables:
intThisTrial                = 1;     %Indicator trial number (increments)
boolTaskRunning             = true;  %boolean whether task should run or not
state                       = 'PrepareNewTrial'; %Initial state

%% Set time:
refTime                     = tic;   %Reference time, start of task
Objects.ldObj.syncTime();   %Make reference time known for lickdetector object

%% Run Task
try
    % put servo in far position. It is in off position when initialized.
    Objects.servoObj.moveServo('F');
    
    while boolTaskRunning
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Check Arduino lickdetector for licks
        boolLickDetected                                = false; %bool whether lick is just detected
        [charSide, dblLickTs]                           = Objects.ldObj.detectLick;
        if ~strcmp (charSide, '')                           % If there was a lick, set & save:
            try vecTrial.lickTime{intThisTrial}         = [vecTrial.lickTime{intThisTrial} dblLickTs]; %Save lick time with others
                vecTrial.lickSide{intThisTrial}         = [vecTrial.lickSide{intThisTrial} charSide]; %Save lick side with previous
            catch
                vecTrial.lickTime{intThisTrial}         = dblLickTs; %save first lick time this trial
                vecTrial.lickSide{intThisTrial}         = charSide; %save first lick side this trial
            end
            boolLickDetected                            = true;
        end
                
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update ControlWindow - Check pending callbacks at figures
%         Objects.cwObj.update(vecTrial,Par,intThisTrial);
        
        drawnow;
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Check for pressed escape or reward key press
        [~, ~, keyCode] = KbCheck();
        if keyCode(KbName(Par.strExitKey))
            error([mfilename ':ExitButton'],'Exit button has been pressed: exiting...')
            boolTaskRunning = false;
        elseif keyCode(KbName(Par.strRewardKey))
            Objects.ldObj.giveReward(Par.Stim.CorrectSide(intThisTrial));
            fprintf('Dispensed reward on the correct side of this trial: %s\n', Par.Stim.CorrectSide(intThisTrial));
        elseif keyCode(KbName(Par.strRewardKeyLeft))
            Objects.ldObj.giveReward('L');
            fprintf('Dispensed reward on the left side\n');
        elseif keyCode(KbName(Par.strRewardKeyRight))
            Objects.ldObj.giveReward('R');
            fprintf('Dispensed reward on the right side\n');
        end
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Main Task % A la finite state-machine: each state corresponds to 
        % a logical situation in the task. Each state transition to a next 
        % state upon events, e.g. ITI is over, or lick detected.
        
        switch state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'PrepareNewTrial' %State: prepare config for new trial
                
                vecTrial.trialStart(intThisTrial) = toc(refTime);
                dblPhaseRand            = rand(1); %randomize starting phase of grating

                %Define the trial type and what kind of stimuli are presented
                if Par.Stim.vecType(intThisTrial)   == 'V'
                    boolVisualStim                  = true;
                    boolAudioStim                   = false;
%                     boolTactileStim = false;
                elseif Par.Stim.vecType(intThisTrial) == 'A'
                    boolVisualStim                  = false;
                    boolAudioStim                   = true;
%                     boolTactileStim = false;
                else
                    boolVisualStim                  = false;
                    boolAudioStim                   = false;
%                     boolTactileStim = false;
                end

                % Save trial types in vecTrial
                vecTrial.stimType(intThisTrial)     = Par.Stim.vecType(intThisTrial);
                vecTrial.stimSide(intThisTrial)     = Par.Stim.vecSide(intThisTrial);
                vecTrial.stimOri(intThisTrial)      = Par.Stim.vecOri(intThisTrial);
                
                % Print trial info
                fprintf('Starting trial %d at %3.2f secs, trial type %c:\n',...
                    intThisTrial, vecTrial.trialStart(intThisTrial), Par.Stim.vecType(intThisTrial));
                
                % Visual: Start with blank screen
                Screen('FillRect',Objects.ptrWindow, Par.bgInt);
                Screen('Flip', Objects.ptrWindow);
                vecTex = load_texture(Par,Objects.ptrWindow,intThisTrial); % Load in textures
                
                % Audio: make auditory and load in buffer
                if boolAudioStim
                    varAudio = load_auditory(Par,intThisTrial);
                    % Fill the audio playback buffer with the audio data 'varAudio.vecSound':
                    PsychPortAudio('FillBuffer', Objects.ptrAudio, varAudio.vecSound);
                end
                
                % Transition if everything is prepared --> go to next state (ITI) 
                state = 'ITI';
                vecTrial.ITIStart(intThisTrial) = toc(refTime); %save trial start time
                
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'ITI' %State: Inter-trial interval
                
                %if                       %If animals shouldnt lick for some secs
                %&& (~Par.TolerateLicksITI && toc(refTime) - vecTrial.lickTime{intThisTrial}(end) > Par.SecsNoLickITI) 
                %end
                
%                 Transition if ITI is over --> Start Stim
%                 if toc(refTime) - vecTrial.ITIStart(intThisTrial) > Par.vecSecsITI(intThisTrial) %If ITI is over
%                     vecTrial.ITIend(intThisTrial) = toc(refTime); % save ITI end time
%                     state = 'StimStart';
%                     vecTrial.StimStart(intThisTrial) = toc(refTime); % save Stimulus start time
%                 end
                
%                Use this if you are using a prestimulus period:
                if toc(refTime) - vecTrial.ITIStart(intThisTrial) > Par.vecSecsITI(intThisTrial) %If ITI is over
                    vecTrial.ITIend(intThisTrial) = toc(refTime); % save ITI end time
                    dblWaitPreStim = rand * (Par.dblSecsPreStim(2) - Par.dblSecsPreStim(1)) + Par.dblSecsPreStim(1);
                    state = 'PreStim';
                    vecTrial.PreStimStart(intThisTrial) = toc(refTime); % save PreStim start time
                end
                
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'PreStim' %State: waiting period before simulus start
                
                if Par.UseServo %Move servo to close position
                    Objects.servoObj.moveServo('C');
                    
                    vecTrial.servoCloseTime(intThisTrial) = toc(refTime);
                end               
                % Transition if pre-stimulus period is over --> Start Stim
                if toc(refTime)-vecTrial.PreStimStart(intThisTrial) > dblWaitPreStim
                    state = 'StimStart';
                    vecTrial.StimStart(intThisTrial) = toc(refTime);
                end
                
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'StimStart' %State: stimulus start
                
                %Present audio stim
                if boolAudioStim
                    PsychPortAudio('Start', Objects.ptrAudio, 1, 0, 1);
                    boolAudioStim = false;
                end
                
                %Present visual stimulus
                if boolVisualStim
                    tStamp = mod((toc(refTime) + dblPhaseRand) * Par.dblSpeed, 1);
                    intThisFrame = ceil(tStamp * Par.intFrameRate);
                    Screen('DrawTexture',Objects.ptrWindow,vecTex(intThisFrame));
                    Screen('Flip', Objects.ptrWindow);
                end
                
                %Transition after stim onset --> state is during stim
                state = 'StimDuring';
                
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'StimDuring' %State: keep presenting stimuli and detect response
                
                boolStimEnd = false; %Var to end stim when necessary
                
                % Keep presenting visual stimulus:
                if boolVisualStim
                    tStamp = mod((toc(refTime) + dblPhaseRand) * Par.dblSpeed, 1);
                    intThisFrame = ceil(tStamp * Par.intFrameRate);
                    Screen('DrawTexture',Objects.ptrWindow,vecTex(intThisFrame));
                    Screen('Flip', Objects.ptrWindow);
                end
                
                % Transition if correct lick during stimulus--> end stim/reward consumption
                if boolLickDetected 
                    if vecTrial.lickSide{intThisTrial}(end) == Par.Stim.CorrectSide(intThisTrial) %correct
                        vecTrial.correct(intThisTrial) = 1;
                        Objects.ldObj.giveReward(Par.Stim.CorrectSide(intThisTrial)); % give reward on arduino
                        vecTrial.rewardSide(intThisTrial) = Par.Stim.CorrectSide(intThisTrial);
                        vecTrial.rewardSize(intThisTrial) = Par.RewardSize;
                        vecTrial.rewardTime(intThisTrial) = toc(refTime);
                        fprintf('\b        Correct (%s)\n',vecTrial.lickSide{intThisTrial}(end));
                        state = 'RewardConsumption';
                        boolStimEnd = true;
                    end
                end
                
                % Transition if premature incorrect lick --> end stim/Time-out?
                if boolLickDetected 
                    if vecTrial.lickSide{intThisTrial}(end) ~= Par.Stim.CorrectSide(intThisTrial) %Incorrect
                        fprintf('\b        Error (%s)\n',vecTrial.lickSide{intThisTrial}(end));
                        vecTrial.correct(intThisTrial) = 0;
                        boolStimEnd = true;
                        state = 'TimeOut';  % Give a time out
                        vecTrial.TimeOutStart(intThisTrial) = toc(refTime);
                    end
                end
                
                % Transition if stimulus duration is over --> end stim
                if toc(refTime) - vecTrial.StimStart(intThisTrial) > Par.dblSecsStimDur
                    boolStimEnd = true;
                    state = 'ResponseWindow';
                    vecTrial.RespWinStart(intThisTrial) = toc(refTime);
                end
                
                % End the stimuli if indicated:
                if boolStimEnd 
                    vecTrial.StimEnd(intThisTrial) = toc(refTime);
                    %Set screen to grey
                    Screen('FillRect',Objects.ptrWindow, Par.bgInt);
                    Screen('Flip', Objects.ptrWindow);
                    % Stop any ongoing audio stimulus
                    PsychPortAudio('Stop',Objects.ptrAudio);
                    boolVisualStim = false; boolAudioStim = false; %This makes sure no stim will appear                    
                end
                
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'ResponseWindow' %State: time window for response
                
                % Transition if correct lick --> Reward Consumption
                if boolLickDetected 
                    if strcmp(vecTrial.lickSide{intThisTrial}(end),Par.Stim.CorrectSide(intThisTrial)) %correct
                        vecTrial.correct(intThisTrial) = 1;
                        Objects.ldObj.giveReward(Par.Stim.CorrectSide(intThisTrial)); % give reward on arduino
                        vecTrial.rewardSide(intThisTrial) = Par.Stim.CorrectSide(intThisTrial);
                        vecTrial.rewardSize(intThisTrial) = Par.RewardSize;
                        vecTrial.rewardTime(intThisTrial) = toc(refTime);
                        state = 'RewardConsumption';
                        fprintf('\b        Correct (%s)\n',vecTrial.lickSide{intThisTrial}(end));
                    end
                end
                
                % Transition if incorrect lick --> Time-out
                if boolLickDetected 
                    if ~strcmp(vecTrial.lickSide{intThisTrial}(end),Par.Stim.CorrectSide(intThisTrial)) %Incorrect
                        vecTrial.correct(intThisTrial) = 0;
                        fprintf('\b        Error (%s)\n',vecTrial.lickSide{intThisTrial}(end));
                        state = 'TimeOut';
                        vecTrial.TimeOutStart(intThisTrial) = toc(refTime);
                    end
                end
                
                % Transition if no response --> End the trial
                % + save trial as ommission
                if toc(refTime)-vecTrial.RespWinStart(intThisTrial) > Par.RespWinDur; %Window length for response
                    vecTrial.correct(intThisTrial) = 0;
                    vecTrial.RespWinEnd(intThisTrial) = toc(refTime);
                    state = 'EndTrial';
                    vecTrial.noResponse(intThisTrial) = 1;
                    fprintf('\b        No Response\n');
                end
                
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'RewardConsumption' %State: time window for reward consumption
                
                %Extra time window for reward consumption
                if toc(refTime)-vecTrial.rewardTime(intThisTrial) > Par.RewardConsumptionSecs; 
                    state = 'EndTrial';
                end             

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'TimeOut'  
                
                %Give a time-out of few seconds as punishment
                if toc(refTime)-vecTrial.TimeOutStart(intThisTrial) > Par.TimeOutSecs;
                    vecTrial.TimeOutEnd(intThisTrial) = toc(refTime);
                    state = 'EndTrial';
                end
                
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'EndTrial'
                
                vecTrial.trialEnd(intThisTrial) = toc(refTime);
                
                if Par.UseServo   
                    Objects.servoObj.moveServo('F');  %Put servo in far position
                    vecTrial.servoFarTime(intThisTrial) = toc(refTime); %Save time
                end
                
                state = 'PrepareNewTrial';
 
                % Update the control window, provide the relevant
                % trialoutcome parameters
                Objects.cwObj.update(vecTrial.correct(intThisTrial),vecTrial.noResponse(intThisTrial),Par,intThisTrial); 
                
                % End of this trial, increment trial number:
                intThisTrial = intThisTrial + 1;
                


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            otherwise
                error('Unknown state')
                
        end
    end
catch errorMessage
    %% Catch me and Throw me
    fprintf('\nError while executing task...\n');
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Abort Session
fprintf('\nEnding Session...\n');

%% Close Psychtoolbox
try Screen('CloseAll'); PsychPortAudio('Close');
catch; end


%% Calculate total reward given in ul
% try if intThisTrial > 1 %#ok<ALIGN>
%     vecTrial.rewardSize     = ones(1,length(vecTrial.correct))*Par.rewardSize;
%     sessionData.TotalReward    = sum(vecTrial.correct.*vecTrial.rewardSize);end
% catch
%     sessionData.TotalReward = 0; end
% fprintf('Total reward given to the animal so far: %3.0f (uL)',sessionData.TotalReward)

try     vecTrial.trialEnd(intThisTrial) = toc(refTime); %#ok<ALIGN>
        vecTrial.correct(intThisTrial)  = 0;
catch; end

%% Save data and check if data is saved:
fprintf('\nTrying to save data and clean up...\n');
save('-mat7-binary',sessionData.FullFileName,'Par','vecTrial','sessionData');
if exist(strcat(sessionData.FullFileName,'.mat'),'file') || exist(sessionData.FullFileName,'file')
    fprintf('\nSuccesfully saved data...\n');
else error('Data not saved');
end

% servo gets put in off position when the object is deleted

%% Close objects:
ObjectFields = fieldnames(Objects);
for obj = 1:length(ObjectFields)
    try delete(Objects.(ObjectFields{obj})); clear Objects.(ObjectFields{obj};
    catch; end
end

%% Rough backup-method to delete lickdetector:
% delete(instrfindall); %Important to close serial port connection does not
% work on octave

%% Show error
rethrow(errorMessage);

end

