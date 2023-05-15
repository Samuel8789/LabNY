function pie_detection(Par,Objects,sessionData)
%Habituation script where the animals just have to lick for a reward

PsychPortAudio('Close');

%% Preallocate struct with arrays for each event/info 
%  Pre-allocate the fields that will be used by your task
vecTrial = struct;

%On time:
vecTrial.rewardTime             = NaN(Par.intTrialNum,1);
vecTrial.trialStart             = NaN(Par.intTrialNum,1);
vecTrial.trialEnd               = NaN(Par.intTrialNum,1);

vecTrial.itiStart               = NaN(Par.intTrialNum,1);
vecTrial.itiEnd                 = NaN(Par.intTrialNum,1);
vecTrial.stimStart              = NaN(Par.intTrialNum,1);
vecTrial.stimChange             = NaN(Par.intTrialNum,1);
vecTrial.stimEnd                = NaN(Par.intTrialNum,1);
vecTrial.respwinStart           = NaN(Par.intTrialNum,1);
vecTrial.respwinEnd             = NaN(Par.intTrialNum,1);
vecTrial.timeoutStart           = NaN(Par.intTrialNum,1);
vecTrial.timeoutEnd             = NaN(Par.intTrialNum,1);

%On response:
vecTrial.lickTime               = cell(Par.intTrialNum,1);
vecTrial.lickSide               = cell(Par.intTrialNum,1);
vecTrial.correctResponse        = NaN(Par.intTrialNum,1);
vecTrial.responseSide           = cell(Par.intTrialNum,1);
vecTrial.rewardSide             = cell(Par.intTrialNum,1);
vecTrial.rewardSize             = NaN(Par.intTrialNum,1);
vecTrial.noResponse             = NaN(Par.intTrialNum,1);
vecTrial.AutoRewReceived        = NaN(Par.intTrialNum,1);

%% Task internal variables:
intThisTrial                = 1;     %Indicator trial number (increments)
boolTaskRunning             = true;  %boolean whether task should run or not
state                       = 'PrepareNewTrial'; %Initial state
LFR                         = 'preLFR'; %initial LFR state

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
            Objects.ldObj.giveReward('L',Par.CorrectRewardSizeLeft);
            fprintf('Dispensed reward on the left side\n');
        elseif keyCode(KbName(Par.strRewardKeyRight))
            Objects.ldObj.giveReward('R',Par.CorrectRewardSizeRight);
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
                
                 %Generate random durations for ITI:
                ITIdur = min(Par.intSecsITI(1) + exprnd(Par.intSecsITI(2)-Par.intSecsITI(1)), Par.intSecsITI(3)); %Generate exponential random ITI duration
%                ITIdur = rand()*(Par.intSecsITI(2) - Par.intSecsITI(1)) + Par.intSecsITI(1); %Generate random ITI duration

                %Stimulus
                vecTex = load_texture_change(Par,Objects.ptrWindow,intThisTrial); % Load in textures
                Screen('FillRect',Objects.ptrWindow, Par.bgInt); %Show gray screen
                Screen('Flip', Objects.ptrWindow);
                
                % Print trial number:
                fprintf('Starting trial %d of %d / %.0f min %2.0f secs:\n',intThisTrial,Par.intTrialNum,floor(vecTrial.trialStart(intThisTrial)/60),mod(vecTrial.trialStart(intThisTrial),60));
                state = 'ITI';
                vecTrial.itiStart(intThisTrial) = toc(refTime);
                
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            case 'ITI'   %Mouse just has to wait through this period 
                
                %--> Continue to show grey screen.
                
                if toc(refTime) - vecTrial.trialStart(intThisTrial) > ITIdur
                      vecTrial.itiEnd(intThisTrial) = toc(refTime);
                      state = 'StimEpoch';
                      vecTrial.stimStart(intThisTrial) = toc(refTime);
                      LFR = 'preLFR';
                      fprintf('StimEpoch of trial %d at %.0f min %2.0f secs:\n',intThisTrial,floor(vecTrial.stimStart(intThisTrial)/60),mod(vecTrial.stimStart(intThisTrial),60));
                end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'StimEpoch'   %
            
                    dblPhaseRand    = 0; %rand(1); %randomize starting phase of grating
                    tStamp          = mod((toc(refTime) + dblPhaseRand) * Par.dblSpeed, 1);
                    intThisFrame    = ceil(tStamp * Par.intFrameRate);
                    Screen('DrawTexture',Objects.ptrWindow,vecTex(intThisFrame));
                    Screen('Flip', Objects.ptrWindow);
                %--> Display Stimulus according to vecTrial etc.
                
                switch LFR
                case 'preLFR'
                      if toc(refTime) - vecTrial.stimStart(intThisTrial) > Par.LFROnset %LFR starts
                      LFR = 'LFR';
                      vecTrial.respwinStart(intThisTrial) = toc(refTime);
                      end
                      
                case 'LFR'    
                      if boolLickDetected 
                          fprintf('\b        Rewarded Lick (%s)\n',vecTrial.lickSide{intThisTrial}(end));
                          side = vecTrial.lickSide{intThisTrial}(end);
                          vecTrial.responseSide{intThisTrial}     = side;
                          vecTrial.correctResponse(intThisTrial)  = 1;
                          Objects.ldObj.giveReward(side,Par.CorrectRewardSizeLeft); % give reward on arduino 
                          vecTrial.rewardSide{intThisTrial}       = side;
                          vecTrial.rewardSize(intThisTrial)       = Par.CorrectRewardSizeLeft;
                          vecTrial.rewardTime(intThisTrial)       = toc(refTime);    
                          LFR = 'postLFR'; 
                          vecTrial.respwinEnd(intThisTrial) = toc(refTime);
                      end
                      if toc(refTime) - vecTrial.stimStart(intThisTrial) > Par.LFROffset %LFR offset reached with no licks
                          if Par.GivePassiveReward         
                          % Give AutoRew
                          randvec = ['R' 'L']; side = randvec(round(rand)+1);
                          fprintf('\b        Passive Reward  (%s)\n',side);
                          vecTrial.correctResponse(intThisTrial)  = 0;
                          vecTrial.noResponse(intThisTrial)       = 1;
                          vecTrial.AutoRewReceived(intThisTrial)  = 1;
                          Objects.ldObj.giveReward(side,Par.PassiveRewardSize); % give reward on arduino 
                          vecTrial.rewardSide{intThisTrial}       = side;
                          vecTrial.rewardSize(intThisTrial)       = Par.PassiveRewardSize;
                          vecTrial.rewardTime(intThisTrial)       = toc(refTime); 
                          end
                          LFR = 'postLFR';
                          vecTrial.respwinEnd(intThisTrial) = toc(refTime);
                      end
               
                case 'postLFR'              
                       if toc(refTime) - vecTrial.stimStart(intThisTrial) > Par.StimEpochDur %if LFROffset>StimEpochDur then stimulus lasts until LFROffset. 
                              vecTrial.stimEndEnd(intThisTrial) = toc(refTime);
                              state = 'Post';  
                              vecTrial.timeoutStart(intThisTrial) = toc(refTime);
                              fprintf('Post epoch of trial %d at %.0f min %2.0f secs:\n',intThisTrial,floor(vecTrial.trialStart(intThisTrial)/60),mod(vecTrial.trialStart(intThisTrial),60));

                       end
                       
                otherwise
                error('LFR in unknown state')
                       
                end
               
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'Post'
            
            %Set screen to grey
                Screen('FillRect',Objects.ptrWindow, Par.bgInt); 
                Screen('Flip', Objects.ptrWindow);
                    
                      if toc(refTime) - vecTrial.timeoutStart(intThisTrial) > Par.PostDur
                              vecTrial.timeoutEnd(intThisTrial) = toc(refTime);
                              state = 'EndTrial';
                      end               
           
                
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'EndTrial'
                vecTrial.trialEnd(intThisTrial) = toc(refTime);

                % Update the control window
                % Get also changed parameters from the figure
                
                Par = Objects.cwObj.update(Par,vecTrial,intThisTrial);
                
                if exist('vecTex','var'); Screen('Close',vecTex); end               %Close open textures..
                
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

