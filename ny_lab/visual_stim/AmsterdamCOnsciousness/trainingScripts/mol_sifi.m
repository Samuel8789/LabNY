function mol_sifi(Par,Objects,sessionData)
% Task description here

%% %% To do list for the script:
% - Priority level of gratings, timing better etc.
% - Check sessiondata + trialdata variables
% - Ability to change trial types within the task
% - give unique session_ID

fprintf('\n Training Stage is %2.0f\n',Par.TrainingStage);

%% Preallocate struct with arrays for each event/info
%  Pre-allocate the fields that will be used by your task
vecTrial = struct;

%On stimuli:
vecTrial.trialType              = cell(Par.intTrialNum,1);
vecTrial.hasauditory            = NaN(Par.intTrialNum,1);
vecTrial.hasvisual              = NaN(Par.intTrialNum,1);
vecTrial.hastactile             = NaN(Par.intTrialNum,1);
vecTrial.nFlashes               = NaN(Par.intTrialNum,1);
vecTrial.nBeeps                 = NaN(Par.intTrialNum,1);
vecTrial.FlashDur               = NaN(Par.intTrialNum,1);
vecTrial.BeepDur                = NaN(Par.intTrialNum,1);
vecTrial.IFI                    = NaN(Par.intTrialNum,1);
vecTrial.IBI                    = NaN(Par.intTrialNum,1);

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
lastRandomBeep              = 0;
randomBeepInterval          = rand()*40+10;

%% Set time:
refTime                     = tic;   %Reference time, start of task
Objects.ldObj.syncTime();   %Make reference time known for lickdetector object
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
            dblLickTs = toc(refTime) - 0.001;
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
        
        %Random Beeps
        if Par.RandomBeeps && (toc(refTime)-lastRandomBeep) > randomBeepInterval
            % Fill the audio playback buffer with varAudio and play:
            Beep                    = MakeBeep(Par.BeepFreq,Par.BeepDur,Par.intSamplingRate) / 5;
            PsychPortAudio('FillBuffer', Objects.ptrAudio, [Beep; Beep]);
            PsychPortAudio('Start', Objects.ptrAudio, 1, 0, 1);
            if rand()>0.5 %Randomly present two beeps or just one:
              pause(Par.IBI + Par.BeepDur)
              PsychPortAudio('Start', Objects.ptrAudio, 1, 0, 1);
            end
            lastRandomBeep          = toc(refTime);
            randomBeepInterval      = rand()*25+5;
        end
                
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Main Task % A la finite state-machine: each state corresponds to
        % a logical situation in the task. Each state transition to a next
        % state upon events, e.g. ITI is over, or lick detected.
        switch state
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'PrepareNewTrial' %State: prepare config for new trial
                
                vecTrial.trialStart(intThisTrial)   = toc(refTime);
                
                %Define the trial type and what kind of stimuli are presented
                vecTrial.trialType{intThisTrial}    = Par.Stim.vecTrialType(intThisTrial); %Save trial types in vecTrial
                vecTrial.leftCorrect(intThisTrial)  = Par.Stim.leftCorrect(intThisTrial);
                vecTrial.rightCorrect(intThisTrial) = Par.Stim.rightCorrect(intThisTrial);

                %Set which modalities are stimulated:
                vecTrial.hasauditory(intThisTrial)  = Par.Stim.hasauditory(intThisTrial);
                vecTrial.hasvisual(intThisTrial)    = Par.Stim.hasvisual(intThisTrial);
                vecTrial.hastactile(intThisTrial)   = Par.Stim.hastactile(intThisTrial);
                
                % Print trial info
                fprintf('\nStarting trial %d at %.0f min %2.0f secs, trial type %c: ',...
                    intThisTrial,floor(vecTrial.trialStart(intThisTrial)/60),mod(vecTrial.trialStart(intThisTrial),60), Par.Stim.vecTrialType(intThisTrial));
                
                %Prepare visual stimulus
                if vecTrial.hasvisual(intThisTrial)
                    Par.PixPerDeg           = Par.intScreenHeight_pix / Par.ScreenHeight_deg; %number of pixels in single retinal degree
                    Par.FlashSizePix        = round(Par.StimSizeRetinalDegrees * Par.PixPerDeg);
                    
                    offset                  = Par.intScreenWidth_pix/4 * Par.StimLateralize * (vecTrial.rightCorrect(intThisTrial)*2-1);
                    FlashCenter             = [Par.intScreenHeight_pix/2 Par.intScreenWidth_pix/2 + offset];
                    FlashDim                = [FlashCenter(1)-Par.FlashSizePix/2 FlashCenter(2)-Par.FlashSizePix/2 FlashCenter(1)+Par.FlashSizePix/2 FlashCenter(2)+Par.FlashSizePix/2];
                    FlashDim                = round(FlashDim);
                    
                    if strcmp(Par.strVisStimType,'Checker')
                        chk             = (checkerboard(80,ceil(Par.FlashSizePix/(80*2))) > 0.5);
                        chk             = chk(1:Par.FlashSizePix,1:Par.FlashSizePix) .* 255;
                        [rr, cc]        = meshgrid(1:Par.FlashSizePix);
                        aperture        = sqrt((rr-Par.FlashSizePix/2).^2+(cc-Par.FlashSizePix/2).^2)<=Par.FlashSizePix/2;
                        sq              = repmat(Par.bgInt,Par.FlashSizePix,Par.FlashSizePix);
                        sq(aperture)    = chk(aperture);
                        checkerstim     = repmat(Par.bgInt,Par.intScreenHeight_pix,Par.intScreenWidth_pix);
                        checkerstim(FlashDim(1):FlashDim(1)+Par.FlashSizePix-1,FlashDim(2):FlashDim(2)+Par.FlashSizePix-1) = sq;
                    end
                end
                
                %Prepare auditory stimulus
                if vecTrial.hasauditory(intThisTrial)
                    PsychPortAudio('DeleteBuffer');
                    % Fill the audio playback buffer with the audio data:
                    Beep                    = MakeBeep(Par.BeepFreq,Par.BeepDur,Par.intSamplingRate);
                    PsychPortAudio('FillBuffer', Objects.ptrAudio, [Beep; Beep]);
                end
                
                % Prepare stimulus timings:
                Par.IFI             = round(Par.IFI/Par.FrameDur)*Par.FrameDur;         %Round to number of frames (each frame ~16.67 ms)
                Par.FlashDur        = round(Par.FlashDur/Par.FrameDur)*Par.FrameDur;    %Round to number of frames (each frame ~16.67 ms)
                
                TotalFlashesDur     = Par.Stim.nFlashes(intThisTrial)*Par.FlashDur + (Par.Stim.nFlashes(intThisTrial)-1)*Par.IFI;
                TotalBeepsDur       = Par.Stim.nBeeps(intThisTrial)*Par.BeepDur + (Par.Stim.nBeeps(intThisTrial)-1)*Par.IBI;
                TotalStimDur        = max([TotalFlashesDur TotalBeepsDur]) + 0.05; %Calc total stimulus duration with some padding:
                
                FirstFlash          = TotalStimDur/2-TotalFlashesDur/2;
                FlashOnsets         = NaN(1,Par.Stim.nFlashes(intThisTrial));
                for f = 1:Par.Stim.nFlashes(intThisTrial)
                    FlashOnsets(f)  = FirstFlash + (f-1)*Par.FlashDur + (f-1)*Par.IFI;
                end
                
                FirstBeep           = TotalStimDur/2-TotalBeepsDur/2;
                BeepOnsets          = NaN(1,Par.Stim.nBeeps(intThisTrial));
                for b = 1:Par.Stim.nBeeps(intThisTrial)
                    BeepOnsets(b)   = FirstBeep + (b-1)*Par.BeepDur + (b-1)*Par.IBI;
                end
                
                % Prepare ordered list with sequence of beeps flashes + onset times
                StimSeq             = [repmat('F',1,Par.Stim.nFlashes(intThisTrial)) repmat('B',1,Par.Stim.nBeeps(intThisTrial))];
                StimOnsets          = [FlashOnsets BeepOnsets];
                [~, idx]            = sort(StimOnsets);
                StimSeq             = StimSeq(idx);
                StimOnsets          = StimOnsets(idx);
                
                %Save information about stimulus in vecTrial:
                vecTrial.nFlashes(intThisTrial)     = Par.Stim.nFlashes(intThisTrial);
                vecTrial.nBeeps(intThisTrial)       = Par.Stim.nBeeps(intThisTrial);
                vecTrial.FlashDur(intThisTrial)     = Par.FlashDur;
                vecTrial.BeepDur(intThisTrial)      = Par.BeepDur;
                vecTrial.IFI(intThisTrial)          = Par.IFI;
                vecTrial.IBI(intThisTrial)          = Par.IBI;
                
                % Transition if everything is prepared --> go to next state (ITI)
                state = 'ITI';
                vecTrial.itiStart(intThisTrial) = toc(refTime); %save trial start time
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
                        state = 'StartTrial';
                        vecTrial.TrialCue(intThisTrial) = toc(refTime); % save Stimulus start time
                    end
                    %Transition if ITI is over AND animals havent licked --> Start Stim
                elseif any(vecTrial.lickTime{intThisTrial})
                    if (toc(refTime) - vecTrial.lickTime{intThisTrial}(end)) > SecsNoLickITI && ...
                            toc(refTime) - vecTrial.itiStart(intThisTrial) > ITIdur %If ITI is over
                        vecTrial.itiEnd(intThisTrial) = toc(refTime); % save ITI end time
                        state = 'StartTrial';
                        vecTrial.TrialCue(intThisTrial) = toc(refTime); % save trialcue start time
                    end
                end
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'StartTrial'
                CueDur          = 0.5;
                PsychPortAudio('DeleteBuffer');
                % Fill the audio playback buffer with the white noise audio:
                noise = (2 * rand(1, CueDur * Par.intSamplingRate) - 1); %Range 0.5 to -0.5;
                PsychPortAudio('FillBuffer', Objects.ptrAudio, [noise; noise]);
                PsychPortAudio('Start', Objects.ptrAudio, 1, 0, 1);
                state = 'CueStimInterval';
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'CueStimInterval'
                CuetoStimDelay  = 1;
                if toc(refTime) - vecTrial.TrialCue(intThisTrial) > CuetoStimDelay
                  state = 'stim';
                  vecTrial.stimStart(intThisTrial) = toc(refTime); % save Stimulus start time
                  firststim = 1;
                end
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'stim' %State: stimulus
                % Stim loops through an order array of flashes and beeps
                % It waits for each stimulus to finish and records actual onset and offset
                % times
                
                % A switch to realtime-priority to reduce timing jitter and interruptions
                % caused by other applications and the operating system itself:
                Priority(MaxPriority(Objects.ptrWindow));
                
                tplanonset          = NaN(1,length(StimSeq));
                tplanoffset         = NaN(1,length(StimSeq));
                tactualonset        = NaN(1,length(StimSeq));
                tactualoffset       = NaN(1,length(StimSeq));
                
                Screen('FillRect', Objects.ptrWindow, Par.bgInt);
                tstart = Screen('Flip', Objects.ptrWindow);
                
                % Align the StimOnsets of visual stimulus to vertical retrace of the screen
                aligndiff           = mean(mod(StimOnsets(strfind(StimSeq,'F')),Par.FrameDur)); %mean of diff vertical retrace to onset flashes
                StimOnsets          = StimOnsets - aligndiff;   %Align all StimSeq to vertical retrace
                
                for nstim = 1:length(StimSeq)
                    if strcmp(StimSeq(nstim),'F')
                        
                        tplanonset(nstim)           = tstart + StimOnsets(nstim);
                        tplanoffset(nstim)          = tplanonset(nstim) + Par.FlashDur;
                        
                        switch Par.strVisStimType
                            case 'Flash'
                                Screen('FillOval', Objects.ptrWindow, Par.FlashInt, FlashDim); %Make flash
                            case 'Checker'
                                chckhandle = Screen('MakeTexture', Objects.ptrWindow, checkerstim); %Draw Checker Flash
                                Screen('DrawTexture', Objects.ptrWindow, chckhandle); %Draw Checker Flash
                        end
                        [~, tactualonset(nstim)]    = Screen('Flip', Objects.ptrWindow,tplanonset(nstim)); %- 0.5*Par.FrameDur
                        Screen('FillRect', Objects.ptrWindow, Par.bgInt); %Set to background
                        [~, tactualoffset(nstim)]   = Screen('Flip', Objects.ptrWindow,tactualonset(nstim) +  Par.FlashDur); %- 0.5*Par.FrameDur
                        
                    elseif strcmp(StimSeq(nstim),'B')
                        
                        tplanonset(nstim)               = tstart + StimOnsets(nstim);
                        tplanoffset(nstim)              = tplanonset(nstim) + Par.BeepDur;
                        tactualonset(nstim)             = PsychPortAudio('Start', Objects.ptrAudio, 1, tplanonset(nstim), 1);
                        [~,~,~,tactualoffset(nstim)]    = PsychPortAudio('Stop', Objects.ptrAudio, 1); %Wait for end of playback
                        
                    end
                end
                
                % Ending the stimulus:
                Priority(0);   %Switch to low priority
                Screen('FillRect',Objects.ptrWindow, Par.bgInt);
                Screen('Flip', Objects.ptrWindow);
                vecTrial.stimEnd(intThisTrial) = toc(refTime);
                
                %Transition after stimulus
                state = 'ResponseWindow';
                if firststim
                  vecTrial.respwinStart(intThisTrial) = toc(refTime);
                  GivePassiveReward = Par.GivePassiveReward;
                  firststim = 0;
                end
               
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'ResponseWindow' %State: time window for response
                
                %If used, give a one time passive reward to aid learning
                if GivePassiveReward && toc(refTime)-vecTrial.stimEnd(intThisTrial) > Par.PassiveRewardDelay
                    sides = [repmat('L',1,vecTrial.leftCorrect(intThisTrial)) repmat('R',1,vecTrial.rightCorrect(intThisTrial))];
                    if ~isempty(sides)
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
                
                % Transition if correct lick --> Reward Consumption
                if boolLickDetected && vecTrial.passiveReward(intThisTrial)~=1 && toc(refTime)-vecTrial.stimEnd(intThisTrial) > Par.GracePeriod
                    side        = vecTrial.lickSide{intThisTrial}(end);
                    if (strcmp(side,'L')&&vecTrial.leftCorrect(intThisTrial)) || (strcmp(side,'R') && vecTrial.rightCorrect(intThisTrial))   %correct
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
                
                % Repeat stimulus if no response 
                if ~boolLickDetected && toc(refTime)-vecTrial.stimEnd(intThisTrial) > 1.2; %Inter stimulus interval
                    state = 'stim';
                end
                
                % Transition if no response --> End the trial + save trial as miss
                if ~boolLickDetected && toc(refTime)-vecTrial.respwinStart(intThisTrial) > Par.RespWinSecs; %Window length for response
                    vecTrial.respwinEnd(intThisTrial)       = toc(refTime);
                    vecTrial.correctResponse(intThisTrial)  = 0;
                    vecTrial.noResponse(intThisTrial)       = 1;
                    state = 'CheckLick'; ChecklickStart = toc(refTime);
                    fprintf('\b        No Response');
                end
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'RewardConsumption' %State: time window for reward consumption
                
                %Extra time window for reward consumption
                if toc(refTime)-vecTrial.rewardTime(intThisTrial) > Par.RewardConsumptionSecs;
                    state = 'EndTrial';
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
                
                if Par.CounterBias %Counter if any bias present
                  nlast = 10; lastn = intThisTrial-(nlast-1):intThisTrial; lastn = lastn(lastn>0); %Take responses of nlast trials
                
                  if nansum(vecTrial.leftCorrect(lastn)==1 & vecTrial.correctResponse(lastn) == 0)/nansum(vecTrial.leftCorrect(lastn)==1) > 0.7;
                    biasside = 'R'; fprintf(' Countering bias on the %s side. ',biasside);
                  elseif nansum(vecTrial.rightCorrect(lastn)==1 & vecTrial.correctResponse(lastn) == 0)/nansum(vecTrial.rightCorrect(lastn)==1) > 0.7;
                    biasside = 'L'; fprintf(' Countering bias on the %s side. ',biasside);
                  else biasside = [];
                  end
                
                  if intThisTrial>5 && biasside
                    while ~((Par.Stim.leftCorrect(intThisTrial+1) && strcmp(biasside,'R')) || (Par.Stim.rightCorrect(intThisTrial+1) && strcmp(biasside,'L')))
                        Par = create_trialtypes_sifi(Par);    %Generate new trial type sequence
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
                            case 4
                               Par.StimLateralize = Par.StimLateralize - 0.1;
                                if Par.StimLateralize <= 0
                                    ProceedStage = true;
                                else   fprintf('\n\n Training Stage is still %1.1f, StimLateralize is %1.1f s\n',Par.TrainingStage,Par.StimLateralize);
                                end
                                
                            otherwise
                              ProceedStage = true;
                        end
                        
                        if ProceedStage && Par.ProceedStages
                              Par.TrainingStage = Par.TrainingStage + 1;
                              fprintf('\n\n Training Stage is now %2.0f\n',Par.TrainingStage);
                              eval(strcat('Par=',sessionData.Parameterfile,'(Par,Par.TrainingStage);'));
                              %% Set Parameter settings from the ControlWindowMatthijs figure
                              AllParams = fieldnames(Objects.cwObj.h.params);
                              for param = 1:length(AllParams)
                                  if Par.(AllParams{param}) == 0 || Par.(AllParams{param}) == 1 
                                    set(Objects.cwObj.h.params.(AllParams{param}),'value',Par.(AllParams{param}));
                                  elseif isnumeric(Par.(AllParams{param})) && isnumeric(str2double(get(Objects.cwObj.h.params.(AllParams{param}),'String')))
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

