function mol_mod_discr(Par,Objects,sessionData)
% Task description here
% This script runs the task for a 2AFC modality discrimination/detection
% task. Visual stimuli are associated with one side, while auditory stimuli
% are associated with the other side. 

fprintf('\n Training Stage is %2.0f\n',Par.TrainingStage);

%% Preallocate struct with arrays for each event/info
%  Pre-allocate the fields that will be used by your task
vecTrial = struct;

%On stimuli:
vecTrial.trialType              = cell(Par.intTrialNum,1);
vecTrial.hasauditory            = NaN(Par.intTrialNum,1);
vecTrial.hasvisual              = NaN(Par.intTrialNum,1);
vecTrial.hastactile             = NaN(Par.intTrialNum,1);
vecTrial.visualOri              = NaN(Par.intTrialNum,1);
vecTrial.audioFreq              = NaN(Par.intTrialNum,1);
vecTrial.visualInt              = NaN(Par.intTrialNum,1);
vecTrial.audioInt              = NaN(Par.intTrialNum,1);

%On timing:
vecTrial.trialStart             = NaN(Par.intTrialNum,1);
vecTrial.trialEnd               = NaN(Par.intTrialNum,1);
vecTrial.itiStart               = NaN(Par.intTrialNum,1);
vecTrial.itiEnd                 = NaN(Par.intTrialNum,1);
vecTrial.stimStart              = NaN(Par.intTrialNum,1);
vecTrial.stimEnd                = NaN(Par.intTrialNum,1);
vecTrial.respwinStart           = NaN(Par.intTrialNum,1);
vecTrial.respwinEnd             = NaN(Par.intTrialNum,1);
vecTrial.timeoutStart           = NaN(Par.intTrialNum,1);
vecTrial.timeoutEnd             = NaN(Par.intTrialNum,1);
vecTrial.rewardTime             = NaN(Par.intTrialNum,1);
vecTrial.passiveReward          = NaN(Par.intTrialNum,1);
vecTrial.passiveRewardTime      = NaN(Par.intTrialNum,1);

%On response:
vecTrial.lickTime               = cell(Par.intTrialNum,1);
vecTrial.lickSide               = cell(Par.intTrialNum,1);
vecTrial.responseSide           = cell(Par.intTrialNum,1);
vecTrial.rewardSide             = cell(Par.intTrialNum,1);
vecTrial.rewardSize             = NaN(Par.intTrialNum,1);
vecTrial.leftCorrect            = NaN(Par.intTrialNum,1);
vecTrial.rightCorrect           = NaN(Par.intTrialNum,1);
vecTrial.correctResponse        = NaN(Par.intTrialNum,1);
vecTrial.noResponse             = NaN(Par.intTrialNum,1);

%% Task internal variables:
intThisTrial                = 1;     %Indicator trial number (increments)
boolTaskRunning             = true;  %boolean whether task should run or not
state                       = 'PrepareNewTrial'; %Initial state

%% Set time:
refTime                     = tic;   %Reference time, start of task
Objects.ldObj.syncTime(refTime);   %Make reference time known for lickdetector object
sessionData.t_start         = toc(refTime);
sessionData.t_unit          = 's';

%% Run Task
try
    while boolTaskRunning
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Check Arduino lickdetector for licks      %takes around 0.6ms
        boolLickDetected                                = false; %bool whether lick is just detected
        [charSide, dblLickTs]                           = Objects.ldObj.detectLick;
        if ~strcmp (charSide, '')                           % If there was a lick, set & save:
            dblLickTs = toc(refTime) - 0.001;             %Use matlab timer because arduino goes out of sync
            try vecTrial.lickTime{intThisTrial}         = [vecTrial.lickTime{intThisTrial} dblLickTs]; %Save lick time with others
                vecTrial.lickSide{intThisTrial}         = [vecTrial.lickSide{intThisTrial} charSide]; %Save lick side with previous
            catch
                vecTrial.lickTime{intThisTrial}         = dblLickTs; %save first lick time this trial
                vecTrial.lickSide{intThisTrial}         = charSide; %save first lick side this trial
            end
            boolLickDetected                            = true;
        end
        %Check pending callbacks at figure              %Takes around 0.01ms
        drawnow;
        % Check for pressed escape or reward key press  %Takes around 0.6ms
        [ ~, ~, keyCode] = KbCheck();
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
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Main Task % A la finite state-machine: each state corresponds to
        % a logical situation in the task. Each state transition to a next
        % state upon events, e.g. ITI is over, or lick detected.
        switch state
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'PrepareNewTrial' %State: prepare config for new trial
                
                vecTrial.trialStart(intThisTrial)   = toc(refTime);
                vecTrial.itiStart(intThisTrial)     = toc(refTime); %save trial start time

                %Define the trial type and what kind of stimuli are presented
                %Save trial types in vecTrial
                vecTrial.trialType{intThisTrial}    = Par.Stim.vecTrialType(intThisTrial);
                vecTrial.leftCorrect(intThisTrial)  = Par.Stim.leftCorrect(intThisTrial);
                vecTrial.rightCorrect(intThisTrial) = Par.Stim.rightCorrect(intThisTrial);
                
                %Set which modalities are stimulated:
                vecTrial.hasauditory(intThisTrial)  = Par.Stim.hasauditory(intThisTrial);
                boolAudioStim                       = Par.Stim.hasauditory(intThisTrial);
                vecTrial.hasvisual(intThisTrial)    = Par.Stim.hasvisual(intThisTrial);
                boolVisualStim                      = Par.Stim.hasvisual(intThisTrial);
                vecTrial.hastactile(intThisTrial)   = Par.Stim.hastactile(intThisTrial);
                boolTactileStim                     = Par.Stim.hastactile(intThisTrial); %#ok<NASGU>
                
                % Print trial info
                fprintf('\nStarting trial %d at %.0f min %2.0f secs, trial type %c: ',...
                    intThisTrial,floor(vecTrial.trialStart(intThisTrial)/60),mod(vecTrial.trialStart(intThisTrial),60), Par.Stim.vecTrialType(intThisTrial));
                
                % Visual: Make stimuli and start with blank screen
                if boolVisualStim
                    vecTrial.visualOri(intThisTrial)            = Par.Stim.vecOri(intThisTrial);
                    vecTrial.visualInt(intThisTrial)            = Par.Stim.vecVisStimInt(intThisTrial);
                    vecTex                                      = load_texture_change(Par,Objects.ptrWindow,intThisTrial); % Load in textures
                end
                
                % Audio: make auditory and load in buffer
                if boolAudioStim
                    vecTrial.audioFreq(intThisTrial)            = Par.Stim.vecFreq(intThisTrial);
                    vecTrial.audioInt(intThisTrial)             = Par.Stim.vecAuStimInt(intThisTrial);
                    varaudio                                    = load_auditory(Par,intThisTrial);
                    % Fill the audio playback buffer with the audio data 'varAudio.vecSound':
                    PsychPortAudio('FillBuffer', Objects.ptrAudio, varaudio.vecSound);
                end
                
                % Transition if everything is prepared --> go to next state (ITI)
                state = 'ITI';
                
                %Generate random durations for ITI and secs no lick ITI:
                ITIdur              = min([Par.intSecsITI*Par.ExpVariability + exprnd(Par.intSecsITI-Par.intSecsITI*Par.ExpVariability) ...
                    Par.intSecsITI*(1/Par.ExpVariability)^2]); %Generate random ITI duration
                SecsNoLickITI       = min([Par.SecsNoLickITI*Par.ExpVariability + exprnd(Par.SecsNoLickITI-Par.SecsNoLickITI*Par.ExpVariability)...
                    Par.SecsNoLickITI*(1/Par.ExpVariability)^2]);
                    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'ITI' %State: Inter-trial interval
                
                %Transition if ITI is over --> Start Stim
                if Par.TolerateLicksITI || (~Par.TolerateLicksITI && ~any(vecTrial.lickTime{intThisTrial}))
                    if toc(refTime) - vecTrial.itiStart(intThisTrial) > ITIdur %If ITI is over
                        vecTrial.itiEnd(intThisTrial) = toc(refTime); % save ITI end time
                        state = 'stimStart';
                        vecTrial.stimStart(intThisTrial) = toc(refTime); % save Stimulus start time
                    end
                    %Transition if ITI is over AND animals havent licked --> Start Stim
                elseif any(vecTrial.lickTime{intThisTrial})
                    if (toc(refTime) - vecTrial.lickTime{intThisTrial}(end)) > SecsNoLickITI && ...
                            toc(refTime) - vecTrial.itiStart(intThisTrial) > ITIdur %If ITI is over
                        vecTrial.itiEnd(intThisTrial) = toc(refTime); % save ITI end time
                        state = 'stimStart';
                        vecTrial.stimStart(intThisTrial) = toc(refTime); % save Stimulus start time
                    end
                end
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'stimStart' %State: stimulus start
                
                %Present audio stim
                if boolAudioStim
                    PsychPortAudio('Start', Objects.ptrAudio, 1, 0, 1);
                end
                
                %Present visual stimulus
                if boolVisualStim
                    dblPhaseRand    = rand(1); %randomize starting phase of grating
                    tStamp          = mod((toc(refTime) + dblPhaseRand) * Par.dblSpeed, 1);
                    intThisFrame    = ceil(tStamp * Par.intFrameRate);
                    Screen('DrawTexture',Objects.ptrWindow,vecTex(intThisFrame));
                    Screen('Flip', Objects.ptrWindow);
                end
                
                %Transition after stim onset --> state during stim:
                state               = 'StimDuring';
                
                boolstimEnd         = false; %Var to end stim when necessary
                GivePassiveReward   = Par.GivePassiveReward;
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'StimDuring' %State: keep presenting stimuli and detect response
                
                % Keep presenting visual stimulus:
                if boolVisualStim
                    tStamp = mod((toc(refTime) + dblPhaseRand) * Par.dblSpeed, 1);
                    intThisFrame = ceil(tStamp * Par.intFrameRate);
                    Screen('DrawTexture',Objects.ptrWindow,vecTex(intThisFrame));
                    Screen('Flip', Objects.ptrWindow);
                end
                
                %If used, give a one time passive reward to aid learning
                if GivePassiveReward && toc(refTime)-vecTrial.stimStart(intThisTrial) > Par.PassiveRewardDelay
                    sides = [repmat('L',1,vecTrial.leftCorrect(intThisTrial)) repmat('R',1,vecTrial.rightCorrect(intThisTrial))];
                    if sides
                      switch length(sides) %If no side is to be rewarded, no passive reward. If two sides, pick one random
                          case 1
                              Objects.ldObj.giveReward(sides(1),Par.PassiveRewardSize); %give reward on arduino
                          case 2
                              Objects.ldObj.giveReward(sides(randi(2,1)),Par.PassiveRewardSize); %give reward on arduino
                      end
                      vecTrial.passiveRewardTime(intThisTrial) 	= toc(refTime);
                      vecTrial.passiveReward(intThisTrial)        = 1;
                    end
                    GivePassiveReward       = false; %Set variable to false --> only one reward given
                end
                
                % Transition if correct lick during stimulus--> reward consumption
                if boolLickDetected && toc(refTime)-vecTrial.stimStart(intThisTrial) > Par.GracePeriod && vecTrial.passiveReward(intThisTrial)~=1
                    side        = vecTrial.lickSide{intThisTrial}(end);
                    if (strcmp(side,'L')&&vecTrial.leftCorrect(intThisTrial)) || (strcmp(side,'R')&&vecTrial.rightCorrect(intThisTrial))   %correct
                        fprintf('\b        Correct (%s)',side);
                        vecTrial.responseSide{intThisTrial} = side;
                        vecTrial.correctResponse(intThisTrial) = 1;
                        Objects.ldObj.giveReward(side,Par.CorrectRewardSize); % give reward on arduino
                        vecTrial.rewardSide{intThisTrial} = side;
                        vecTrial.rewardSize(intThisTrial) = Par.CorrectRewardSize;
                        vecTrial.rewardTime(intThisTrial) = toc(refTime);
                        state = 'RewardConsumption';
                        boolstimEnd = false; %Continue with stimulus if animal was correct 
                    end
                end 
                
                % Transition if incorrect lick --> end stim/Time-out?
                if boolLickDetected && toc(refTime)-vecTrial.stimStart(intThisTrial) > Par.GracePeriod && ~Par.TolerateLicksIncorrect
                    side        = vecTrial.lickSide{intThisTrial}(end);
                    if (strcmp(side,'L')&&~vecTrial.leftCorrect(intThisTrial)) || (strcmp(side,'R')&&~vecTrial.rightCorrect(intThisTrial))   %Incorrect
                        vecTrial.responseSide{intThisTrial}     = side;
                        vecTrial.correctResponse(intThisTrial)  = 0;
                        fprintf('\b        Error (%s)',side);
                        if Par.UseTimeOut
                            state = 'timeout';  % Give a time out
                            vecTrial.timeoutStart(intThisTrial) = toc(refTime);
                        else
                            state = 'EndTrial';  %End the Trial
                        end
                        boolstimEnd = true;
                    end
                end
                
                % Transition if stimulus duration is over + no response --> End the trial  + save trial as miss (unless probe trial)
                if ~boolLickDetected && (toc(refTime) - vecTrial.stimStart(intThisTrial)) > Par.StimSecs; %Window length for response
                    boolstimEnd = true;
                    vecTrial.noResponse(intThisTrial)       = 1;
                    if ~vecTrial.leftCorrect(intThisTrial) && ~vecTrial.rightCorrect(intThisTrial)
                      vecTrial.correctResponse(intThisTrial)  = 1;
                      state = 'EndTrial';
                      fprintf('\b        Correct Rejection');
                    else vecTrial.correctResponse(intThisTrial)  = 0;
                      state = 'CheckLick'; ChecklickStart = toc(refTime);
                      fprintf('\b        No Response');
                    end
                end                
                
                % End the stimuli if indicated:
                if boolstimEnd
                    %Set screen to grey
                    Screen('FillRect',Objects.ptrWindow, Par.bgInt);
                    Screen('Flip', Objects.ptrWindow);
                    
                    % Stop any ongoing audio stimulus
                    PsychPortAudio('Stop',Objects.ptrAudio);
                    vecTrial.stimEnd(intThisTrial) = toc(refTime);
                    boolVisualStim = false; boolAudioStim = false; %This makes sure no stim will appear
                end
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'ResponseWindow' %State: time window for response
                
                % Transition if correct lick --> Reward Consumption
                if boolLickDetected && vecTrial.passiveReward(intThisTrial)~=1
                    side        = vecTrial.lickSide{intThisTrial}(end);
                    if (strcmp(side,'L')&&vecTrial.leftCorrect(intThisTrial)) || (strcmp(side,'R')&&vecTrial.rightCorrect(intThisTrial))   %correct
                        vecTrial.respwinEnd(intThisTrial)       = toc(refTime);
                        fprintf('\b        Correct (%s)',side);
                        vecTrial.responseSide{intThisTrial} = side;
                        vecTrial.correctResponse(intThisTrial) = 1;
                        Objects.ldObj.giveReward(side,Par.CorrectRewardSize); % give reward on arduino
                        vecTrial.rewardSide{intThisTrial} = side;
                        vecTrial.rewardSize(intThisTrial) = Par.CorrectRewardSize;
                        vecTrial.rewardTime(intThisTrial) = toc(refTime);
                        state = 'RewardConsumption';
                    end
                end
                
                % Transition if incorrect lick --> Time-out
                if boolLickDetected && ~Par.TolerateLicksIncorrect
                    side        = vecTrial.lickSide{intThisTrial}(end);
                    if (strcmp(side,'L')&&~vecTrial.leftCorrect(intThisTrial)) || (strcmp(side,'R')&&~vecTrial.rightCorrect(intThisTrial))   %Incorrect
                        vecTrial.respwinEnd(intThisTrial)       = toc(refTime);
                        vecTrial.responseSide{intThisTrial}     = side;
                        vecTrial.correctResponse(intThisTrial)  = 0;
                        fprintf('\b        Error (%s)',side);
                        if Par.UseTimeOut
                            state = 'timeout';
                            vecTrial.timeoutStart(intThisTrial) = toc(refTime);
                        else
                            state = 'EndTrial';  %End the Trial
                        end
                    end
                end
                
                % Transition if no response --> End the trial  + save trial as miss (unless probe trial)
                if ~boolLickDetected && toc(refTime)-vecTrial.respwinStart(intThisTrial) > Par.RespWinSecs; %Window length for response
                    vecTrial.respwinEnd(intThisTrial)       = toc(refTime);
                    vecTrial.noResponse(intThisTrial)       = 1;
                    if ~vecTrial.leftCorrect(intThisTrial) && ~vecTrial.rightCorrect(intThisTrial)
                      vecTrial.correctResponse(intThisTrial)  = 1;
                      state = 'EndTrial';
                      fprintf('\b        Correct Rejection');
                    else vecTrial.correctResponse(intThisTrial)  = 0;
                      state = 'CheckLick'; ChecklickStart = toc(refTime);
                      fprintf('\b        No Response');
                    end
                end
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'RewardConsumption' %State: time window for reward consumption
                
                % Keep presenting visual stimulus:
                if boolVisualStim
                    tStamp = mod((toc(refTime) + dblPhaseRand) * Par.dblSpeed, 1);
                    intThisFrame = ceil(tStamp * Par.intFrameRate);
                    Screen('DrawTexture',Objects.ptrWindow,vecTex(intThisFrame));
                    Screen('Flip', Objects.ptrWindow);
                end
                
                % End stim if stim duration is over
                if ~boolstimEnd && (toc(refTime) - vecTrial.stimStart(intThisTrial)) > Par.StimSecs
                    boolstimEnd = true;
                end   
                
                %Extra time window for reward consumption
                if toc(refTime)-vecTrial.rewardTime(intThisTrial) > Par.RewardConsumptionSecs;
                    state = 'EndTrial';
                    boolstimEnd = true;
                end
                
                % End the stimuli if indicated:
                if boolstimEnd && (boolAudioStim || boolVisualStim)
                    %Set screen to grey
                    Screen('FillRect',Objects.ptrWindow, Par.bgInt);
                    Screen('Flip', Objects.ptrWindow);
                    
                    % Stop any ongoing audio stimulus
                    PsychPortAudio('Stop',Objects.ptrAudio);
                    vecTrial.stimEnd(intThisTrial) = toc(refTime);
                    boolVisualStim = false; boolAudioStim = false; %This makes sure no stim will appear
                end
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'timeout'
                
                TimeOutFlashFreq = 5;
                %Set screen to grey
                if mod(toc(refTime),1/TimeOutFlashFreq)>(1/TimeOutFlashFreq)/2
                  TimeOutScrInt = 255;
                else TimeOutScrInt = 0;
                end
                Screen('FillRect',Objects.ptrWindow, TimeOutScrInt);
                Screen('Flip', Objects.ptrWindow);
                
                %Give a time-out of few seconds as punishment
                if toc(refTime)-vecTrial.timeoutStart(intThisTrial) > Par.TimeOutSecs;
                    Screen('FillRect',Objects.ptrWindow, Par.bgInt);
                    Screen('Flip', Objects.ptrWindow);                    
                    vecTrial.timeoutEnd(intThisTrial) = toc(refTime);
                    state = 'EndTrial';
                end
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'CheckLick' %State: Pause if no lick during entire trial before continuing
                
                if Par.CheckLick && intThisTrial>2
                  if ~isempty([vecTrial.lickTime{intThisTrial} vecTrial.lickTime{intThisTrial-1} vecTrial.lickTime{intThisTrial-2}])
                      state = 'EndTrial';
                  end
                else 
                    state = 'EndTrial';
                end
                
                if toc(refTime)-ChecklickStart>20     %Give passive reward if no action for 20 seconds to stimulate
                    ChecklickStart = toc(refTime); sides = ['R' 'L'];
                    Objects.ldObj.giveReward(sides(randi(2,1)),Par.PassiveRewardSize); %give reward on arduino
                end
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'EndTrial'
                
                vecTrial.trialEnd(intThisTrial) = toc(refTime);
                state = 'PrepareNewTrial';

                if Par.CounterBias && intThisTrial>10 %Counter if any bias present
                  nlast = 10; lastn = intThisTrial-(nlast-1):intThisTrial; lastn = lastn(lastn>0); %Take responses of nlast trials
                
                  if nansum(vecTrial.leftCorrect(lastn)==1 & vecTrial.correctResponse(lastn) == 0)/nansum(vecTrial.leftCorrect(lastn)==1) > 0.7;
                    biasside = 'R'; fprintf(' Countering bias on the %s side. ',biasside);
                  elseif nansum(vecTrial.rightCorrect(lastn)==1 & vecTrial.correctResponse(lastn) == 0)/nansum(vecTrial.rightCorrect(lastn)==1) > 0.7;
                    biasside = 'L'; fprintf(' Countering bias on the %s side. ',biasside);
                  else biasside = [];
                  end
                
                  if biasside
                    while ~((Par.Stim.leftCorrect(intThisTrial+1) && strcmp(biasside,'R')) || (Par.Stim.rightCorrect(intThisTrial+1) && strcmp(biasside,'L')))
                        Par = create_trialtypes_mod_discr(Par);
                        Par = create_stimtypes_mod_discr(Par);
                    end
                  end
                end
                  
                %Change stage/parameters depending on performance:
                nTrialPerfCheck = 25;
                if intThisTrial>nTrialPerfCheck/2 && mod(intThisTrial,nTrialPerfCheck)==0 %Check every n trials
                    RunningPercCorrect      = nansum(vecTrial.correctResponse((intThisTrial-nTrialPerfCheck+1):intThisTrial))/nTrialPerfCheck; %Compute performance:
                    if RunningPercCorrect > 0.7
                        ProceedStage = false;
                        switch Par.TrainingStage
%                            case 3
%                               Par.StimSpanSides = Par.StimSpanSides + 0.1;
%                                if Par.StimSpanSides >= 1
%                                    ProceedStage = true;
%                                else   fprintf('\n\n Training Stage is still %1.1f, StimSpanSides is %1.1f\n',Par.TrainingStage,Par.StimSpanSides);
%                                end
                            otherwise
                                ProceedStage = true;
                        end
                        
                        if ProceedStage && Par.ProceedStages
                              Par.TrainingStage = Par.TrainingStage + 1;
                              if Par.TrainingStage > 12
                                sessionData.Parameterfile = 'mol_par_ch_det_psy';
                              end
                              fprintf('\n\n Training Stage is now %2.0f\n',Par.TrainingStage);
                              eval(strcat('Par=',sessionData.Parameterfile,'(Par,Par.TrainingStage);'));
                              %% Set Parameter settings from the ControlWindowMatthijs figure
                              AllParams = fieldnames(Objects.cwObj.h.params);
                              for param = 1:length(AllParams)
                                  if Par.(AllParams{param}) == 0 || Par.(AllParams{param}) == 1 
                                    set(Objects.cwObj.h.params.(AllParams{param}),'value',Par.(AllParams{param}));
                                  elseif isnumeric(Par.(AllParams{param})) && isnumeric(str2num(get(Objects.cwObj.h.params.(AllParams{param}),'String')))
                                    set(Objects.cwObj.h.params.(AllParams{param}),'String',num2str(Par.(AllParams{param})));
                                  end
                              end
                        end
                        %If not performance >0.7:
                    else fprintf('\n\n Performance is %.2f, not advancing training stage.\n',RunningPercCorrect);
                    end
                end
                
                % Update the control window, provide the relevant
                % trialoutcome parameters
                Par = Objects.cwObj.update(Par,vecTrial,intThisTrial);
                
                if exist('vecTex','var'); Screen('Close',vecTex); end               %Close open textures..
                if exist('varAudio','var'); PsychPortAudio('DeleteBuffer'); end     %Clear auditory buffers...
                
%                if mod(intThisTrial,100)
%                  Objects.ldObj.syncTime(refTime);   %Make reference time known for lickdetector object
%                end
                
                % End of this trial, increment trial number:
                intThisTrial = intThisTrial + 1;
                
                if intThisTrial > Par.intTrialNum %If all trials done:
                    pause(Par.dblSecsBlankAtEnd)
                    error('Protocol Finished')
                end
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

%% Copy parameters to sessionData struct
parfields = fieldnames(Par);
for parfield = 1:length(parfields)
    if ~isstruct(Par.(parfields{parfield}))
        sessionData.(parfields{parfield}) = Par.(parfields{parfield});
    end
end
sessionData.t_stop         = toc(refTime); %Save stopping time

%% Calculate total reward given in ul
sessionData.TotalReward = 0;
try if intThisTrial > 1 %#ok<ALIGN>
        totalcorrectreward = nansum(vecTrial.correctResponse(1:intThisTrial-1) .*vecTrial.rewardSize(1:intThisTrial-1));
        totalpassivereward = nansum(vecTrial.passiveReward(1:intThisTrial-1)    *Par.PassiveRewardSize);
        sessionData.TotalReward = nansum([totalcorrectreward totalpassivereward]);
    end
catch; end
fprintf('Total reward given to the animal in this session: %4.0f (uL)\n',sessionData.TotalReward)

%% Save last trial as incorrect:
try     vecTrial.trialEnd(intThisTrial) = toc(refTime); %#ok<ALIGN>
    vecTrial.correctResponse(intThisTrial)  = 0;
catch; end

%% Trim all trialfields to the correct length:
try     trialfields = fieldnames(vecTrial);     %#ok<ALIGN>
    for trialfield = 1:length(trialfields)
        vecTrial.(trialfields{trialfield}) = vecTrial.(trialfields{trialfield})(1:intThisTrial);
    end
catch; end

%% Save data
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
else save('catch_save_error.mat','trialData','sessionData');
    error('Problem saving data!!!');
end

if Par.UseServo    % put servo in far position
    Objects.servoObj.moveServo('F');
end

%% Close objects:
ObjectFields = fieldnames(Objects);
for obj = 1:length(ObjectFields)
    try delete(Objects.(ObjectFields{obj})); clear Objects.(ObjectFields{obj};
    catch; end
end

%% Show error
rethrow(errorMessage);

end

