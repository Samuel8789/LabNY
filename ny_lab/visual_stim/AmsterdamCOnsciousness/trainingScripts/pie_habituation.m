function pie_habituation(Par,Objects,sessionData)
%Habituation script where the animals just have to lick for a reward

PsychPortAudio('Close');

%% Preallocate struct with arrays for each event/info 
%  Pre-allocate the fields that will be used by your task
vecTrial = struct;

%On time:
vecTrial.rewardTime             = NaN(Par.intTrialNum,1);
vecTrial.trialStart             = NaN(Par.intTrialNum,1);
vecTrial.trialEnd               = NaN(Par.intTrialNum,1);

%On response:
vecTrial.lickTime               = cell(Par.intTrialNum,1);
vecTrial.lickSide               = cell(Par.intTrialNum,1);
vecTrial.correctResponse        = NaN(Par.intTrialNum,1);
vecTrial.responseSide           = cell(Par.intTrialNum,1);
vecTrial.rewardSide             = cell(Par.intTrialNum,1);
vecTrial.rewardSize             = NaN(Par.intTrialNum,1);
vecTrial.noResponse             = NaN(Par.intTrialNum,1);

%% Task internal variables:
intThisTrial                = 1;     %Indicator trial number (increments)
boolTaskRunning             = true;  %boolean whether task should run or not
state                       = 'PrepareNewTrial'; %Initial state

%% Set time:
refTime                     = tic;   %Reference time, start of task
Objects.ldObj.syncTime();   %Make reference time known for lickdetector object

%% Run Task
try
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
            disp('Lick Detected')
        end
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Check for pressed escape or reward key press
        [~, ~, keyCode] = KbCheck();
        if keyCode(KbName(Par.strExitKey))
            error([mfilename ':ExitButton'],'Exit button has been pressed: exiting...')
            boolTaskRunning = false;
        elseif keyCode(KbName(Par.strRewardKeyLeft))
            Objects.ldObj.giveReward('L',Par.CorrectRewardSize);
            fprintf('Dispensed reward on the left side\n');
        elseif keyCode(KbName(Par.strRewardKeyRight))
            Objects.ldObj.giveReward('R',Par.CorrectRewardSize);
            fprintf('Dispensed reward on the right side\n');
        end
        
        drawnow
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Main Task % A la finite state-machine: each state corresponds to 
        % a logical situation in the task. Each state transition to a next 
        % state upon events, e.g. ITI is over, or lick detected.
        
        switch state
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'PrepareNewTrial' %State: prepare config for new trial

                vecTrial.trialStart(intThisTrial)   = toc(refTime);
                % Print trial number:
                fprintf('Starting trial %d at %.0f min %2.0f secs:\n',intThisTrial,floor(vecTrial.trialStart(intThisTrial)/60),mod(vecTrial.trialStart(intThisTrial),60));
                
                state = 'WaitForLick';
                
            case 'WaitForLick' %State: prepare config for new trial
                if boolLickDetected
                    fprintf('\b        Rewarded Lick (%s)\n',vecTrial.lickSide{intThisTrial}(end));
                    side = vecTrial.lickSide{intThisTrial}(end);
                    vecTrial.responseSide{intThisTrial}     = side;
                    vecTrial.correctResponse(intThisTrial)  = 1;
                    Objects.ldObj.giveReward(side,Par.CorrectRewardSize); % give reward on arduino 
                    vecTrial.rewardSide{intThisTrial}       = side;
                    vecTrial.rewardSize(intThisTrial)       = Par.CorrectRewardSize;
                    vecTrial.rewardTime(intThisTrial)       = toc(refTime);
                    state = 'RewardConsumption';
                end

                if Par.GivePassiveReward && toc(refTime) - vecTrial.trialStart(intThisTrial) > Par.PassiveRewardInterval
                    randvec = ['R' 'L']; side = randvec(round(rand)+1);
                    fprintf('\b        Passive Reward  (%s)\n',side);
                    vecTrial.correctResponse(intThisTrial)  = 0;
                    vecTrial.noResponse(intThisTrial)       = 1;
                    Objects.ldObj.giveReward(side,Par.PassiveRewardSize); % give reward on random side
                    vecTrial.rewardSide{intThisTrial}       = side;
                    vecTrial.rewardSize(intThisTrial)       = Par.PassiveRewardSize;
                    vecTrial.rewardTime(intThisTrial)       = toc(refTime);
                    state = 'RewardConsumption';
                end
                    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'RewardConsumption' %State: time window for reward consumption
                
                %Time window for reward consumption and not rewarding licks
                if toc(refTime)-vecTrial.rewardTime(intThisTrial) > Par.RewardConsumptionSecs; 
                    state = 'EndTrial';
                end             
                
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'EndTrial'
                vecTrial.trialEnd(intThisTrial) = toc(refTime);

                % Update the control window
                % Get also changed parameters from the figure
                Par = Objects.cwObj.update(Par,vecTrial,intThisTrial);
%                 Objects.cwObj.update(Objects.cwObj,boolLickDetected,boolLickDetected,Par,intThisTrial);
                
                % End of this trial, increment trial number:
                intThisTrial = intThisTrial + 1;
                
                if intThisTrial > Par.intTrialNum %All trials done
                    error('Protocol Finished')
                end
                state = 'PrepareNewTrial';
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            otherwise
                error('Task in unknown state')
                
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
sessionData.TotalReward = 0;
try if intThisTrial > 1 %#ok<ALIGN>
        sessionData.TotalReward = nansum(vecTrial.rewardSize);
    end
catch; end
fprintf('Total reward given to the animal in this session: %3.0f (uL)\n',sessionData.TotalReward)

%% Trim all trialfields to the correct length:
try     trialfields = fieldnames(vecTrial);     %#ok<ALIGN>
    for trialfield = 1:length(trialfields)
        vecTrial.(trialfields{trialfield}) = vecTrial.(trialfields{trialfield})(1:intThisTrial);
    end
catch; end

%% Save data and check if data is saved:
fprintf('\nTrying to save data and clean up...\n');
trialData = vecTrial; %#ok<NASGU>
if exist('OCTAVE_VERSION', 'builtin') ~= 0 
    save('-mat7-binary',sessionData.FullFileName,'trialData','sessionData');
else
    save(sessionData.FullFileName,'trialData','sessionData');
end
%Check if data is saved:
if exist(strcat(sessionData.FullFileName,'.mat'),'file') || exist(sessionData.FullFileName,'file')
    fprintf('\nSuccesfully saved data...\n');
else error('Data not saved');
end

if Par.UseServo    % put servo in far position
    Objects.servoObj.moveServo('F');
end

%% Close objects:
ObjectFields = fieldnames(Objects);
for obj = 1:length(ObjectFields)
    try delete(Objects.(ObjectFields{obj})); %clear Objects.(ObjectFields{obj};
    catch; end
end

%% Show error
rethrow(errorMessage);

end

