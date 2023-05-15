function pie_fulltask(Par,Objects,sessionData)
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
vecTrial.stimEnd                = NaN(Par.intTrialNum,1);
vecTrial.respwinStart           = NaN(Par.intTrialNum,1);
vecTrial.respwinEnd             = NaN(Par.intTrialNum,1);
vecTrial.timeoutStart           = NaN(Par.intTrialNum,1);
vecTrial.timeoutEnd             = NaN(Par.intTrialNum,1);

%On stimulus:
vecTrial.trialType              = NaN(Par.intTrialNum,1);
vecTrial.stimSide               = cell(Par.intTrialNum,1);
vecTrial.leftCorrect            = NaN(Par.intTrialNum,1);
vecTrial.rightCorrect           = NaN(Par.intTrialNum,1);
vecTrial.stimOri                = NaN(Par.intTrialNum,1);
vecTrial.stimContrast           = NaN(Par.intTrialNum,1);
vecTrial.stimDeflection         = NaN(Par.intTrialNum,1);

%On response:
vecTrial.lickTime               = cell(Par.intTrialNum,1);
vecTrial.lickSide               = cell(Par.intTrialNum,1);
vecTrial.correctResponse        = NaN(Par.intTrialNum,1);
vecTrial.responseSide           = cell(Par.intTrialNum,1);
vecTrial.PenalizedCorrect       = cell(Par.intTrialNum,1);
vecTrial.rewardSide             = cell(Par.intTrialNum,1);
vecTrial.rewardSize             = NaN(Par.intTrialNum,1);
vecTrial.noResponse             = NaN(Par.intTrialNum,1);
vecTrial.AutoRewReceived        = NaN(Par.intTrialNum,1);
vecTrial.firstIncorrect         = NaN(Par.intTrialNum,1);
vecTrial.firstRespLFR           = NaN(Par.intTrialNum,1);

%% Task internal variables:
intThisTrial                = 1;     %Indicator trial number (increments)
boolTaskRunning             = true;  %boolean whether task should run or not
state                       = 'PrepareNewTrial'; %Initial state
LFR                         = 'preLFR'; %initial LFR state
flagBlocks                  = Par.UseTrialBlocks;

%% Set time:
refTime                     = tic;   %Reference time, start of task
Objects.ldObj.syncTime();   %Make reference time known for lickdetector object

%% Prepare the visual stimulus textures
Textures = struct;
for ori=1:length(Par.vecOrientations)
for con=1:length(Par.dblGratingContrast)
Textures(ori,con).tex = buildStimPie2(Par,Objects.ptrWindow,ori,con,1); %the index depends on the par vectors
end
end

%% Run Task
try
    while boolTaskRunning
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Check Arduino lickdetector for licks
        boolLickDetected                                = false; %bool whether lick is just detected
        [charSide, ~]                                   = Objects.ldObj.detectLick;
        dblLickTs                                       = toc(refTime); %matlab based cause arduino delays
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
                
                %Recreate Trials if UseTrialBlocks is on          
                if Par.UseTrialBlocks==1 && flagBlocks==0 %first time on since last off.
                    [Par] = create_trialtypes_pie(Par);
                    flagBlocks=1; %indicate that the change has been made
                end
                if Par.UseTrialBlocks==0 && flagBlocks==1 %first time off since last on.
                    [Par] = create_trialtypes_pie(Par);
                    flagBlocks=0;
                end
                
                % Update trials if requested
                if Par.UpdateTrials == 1 
                Par.UpdateTrials = 0.5;
                [Par] = create_trialtypes_pie(Par);              
                end    
                
                vecTex0 = buildGratingPie2(Par,Objects.ptrWindow,Par.vecOrientations(1),Par.dblGratingContrast(1),1);
                %Screen                
                Screen('FillRect',Objects.ptrWindow, Par.bgInt); %Show gray screen
                Screen('Flip', Objects.ptrWindow);
                
                %Check for the need of new textures
                
                
                %Save trial types in vecTrial
                vecTrial.trialType(intThisTrial)        = Par.Stim.vecTrialType(intThisTrial);
                vecTrial.stimSide{intThisTrial}         = Par.Stim.vecSide(intThisTrial);
                ThisTrialType                           = Par.Stim.vecTrialType(intThisTrial); %short, for use inside loop
                vecTrial.leftCorrect(intThisTrial)      = Par.Stim.leftCorrect(intThisTrial);
                vecTrial.rightCorrect(intThisTrial)     = Par.Stim.rightCorrect(intThisTrial);
                
                if ThisTrialType=='V' || ThisTrialType=='M'
                vecTrial.stimOri(intThisTrial)          = Par.Stim.vecOri(intThisTrial);
                vecTrial.stimContrast(intThisTrial)     = Par.Stim.vecContrast(intThisTrial);
                
                ind1=find(Par.vecOrientations==vecTrial.stimOri(intThisTrial)); ind1=ind1(1);
                ind2=find(Par.dblGratingContrast==vecTrial.stimContrast(intThisTrial)); ind2=ind2(1);
                fullstimgrat = Textures(ind1,ind2).tex;
                
                %Make the textures with the stimuli and store pointers in vecTex
                  for intFrame = 1:Par.intFrameRate
                      vecTex(intFrame) = Screen('MakeTexture', Objects.ptrWindow, fullstimgrat(:,:,intFrame), 0, 1);
                  end
                end
                
                if ThisTrialType=='T' || ThisTrialType=='M'
                vecTrial.stimDeflection(intThisTrial)   = Par.Stim.vecDeflection(intThisTrial);
                piezoflag=0;
                end
                
                 %Generate random durations for ITI:
                ITIdur = min(Par.intSecsITI(1) + exprnd(Par.intSecsITI(2)-Par.intSecsITI(1)), Par.intSecsITI(3)); %Generate exponential random ITI duration
%                ITIdur = rand()*(Par.intSecsITI(2) - Par.intSecsITI(1)) + Par.intSecsITI(1); %Generate random ITI duration         
                
                % Print trial number:
                fprintf('Starting %s trial %d of %d / %.0f min %2.0f secs:\n',ThisTrialType,intThisTrial,Par.intTrialNum,floor(vecTrial.trialStart(intThisTrial)/60),mod(vecTrial.trialStart(intThisTrial),60));
                
                %Initialize penality settings.
                boolPenalized = false;
                IncorrectRewardSizeLeft = ceil(Par.CorrectRewardSizeLeft*Par.PenaltyRewardPercent);
                IncorrectRewardSizeRight = ceil(Par.CorrectRewardSizeRight*Par.PenaltyRewardPercent);
                state = 'ITI';
                vecTrial.itiStart(intThisTrial)   = toc(refTime);
                
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            case 'ITI'   %Mouse just has to wait through this period 
                
                %--> Continue to show grey screen.           
                       
                if toc(refTime) - vecTrial.trialStart(intThisTrial) > ITIdur
                
                % Set LFR starting state.
                %VISUALS
                    if ThisTrialType=='V' || ThisTrialType=='C'
                    LFR = 'preLFR'; %preLFR only for visual so far, tactile is fast.
                    end
                %TACTILES
                    if ThisTrialType=='T' || ThisTrialType=='M'
                    LFR = 'preLFR'; %StimEpoch starts with LFR.
                    end
                
                      state = 'StimEpoch';          
                      vecTrial.itiEnd(intThisTrial) = toc(refTime);     
                      vecTrial.stimStart(intThisTrial) = toc(refTime); % save Stimulus start time
                      fprintf('StimEpoch of trial %d at %.0f min %2.0f secs:\n',intThisTrial,floor(vecTrial.stimStart(intThisTrial)/60),mod(vecTrial.stimStart(intThisTrial),60));                  
                end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'StimEpoch'   %
                    %VISUALS
                    if ThisTrialType=='V' || ThisTrialType=='M'
                    dblPhaseRand    = 0; %rand(1); %randomize starting phase of grating
                    tStamp          = mod((toc(refTime) + dblPhaseRand) * Par.dblSpeed, 1);
                    intThisFrame    = ceil(tStamp * Par.intFrameRate);
                    Screen('DrawTexture',Objects.ptrWindow,vecTex(intThisFrame));
                    Screen('Flip', Objects.ptrWindow);
                    end
                    %TACTILES : stimulus will be on LFR case.                   
                
                switch LFR
                case 'preLFR'
                      if toc(refTime) - vecTrial.stimStart(intThisTrial) > Par.LFROnset %LFR starts
                      LFR = 'LFR';
                      vecTrial.respwinStart(intThisTrial) = toc(refTime);
                      end
                      
                case 'LFR'    
                
                %TACTILES ONLY
                if (ThisTrialType=='T' || ThisTrialType=='M') 
                  if piezoflag==0
                %deliver tactile stimulus
                    piezoflag=1; %only one stim per trial.
                    Objects.pzObj.movePiezo(Par.Stim.vecSide(intThisTrial),vecTrial.stimDeflection(intThisTrial)); 
                  end
                end
                
                      if boolLickDetected 
                          side = vecTrial.lickSide{intThisTrial}(end);
                          vecTrial.firstRespLFR(intThisTrial) = side;
                          
                        if ThisTrialType=='C'
                            vecTrial.correctResponse(intThisTrial)  = 0;
                            vecTrial.firstIncorrect(intThisTrial)   = 1;
                            LFR = 'postLFR';
                            vecTrial.respwinEnd(intThisTrial) = toc(refTime);
                        else
                          
                          if (strcmp(side,'L')&&Par.Stim.leftCorrect(intThisTrial)) || (strcmp(side,'R')&&Par.Stim.rightCorrect(intThisTrial))   %correct
                          fprintf('\b        Correct Lick (%s)\n',side); 
                              if boolPenalized                                                  
                                  %Response side is still considered wrong. Only the first counts. This can be changed.
                                  vecTrial.PenalizedCorrect{intThisTrial} = side;
                                  if strcmp(side,'L')  
                                  Objects.ldObj.giveReward(side,IncorrectRewardSizeLeft); % give reward on arduino 
                                  vecTrial.rewardSize(intThisTrial)       = IncorrectRewardSizeLeft;
                                  elseif strcmp(side,'R')
                                  Objects.ldObj.giveReward(side,IncorrectRewardSizeRight); % give reward on arduino  
                                  vecTrial.rewardSize(intThisTrial)       = IncorrectRewardSizeRight;
                                  end
                                  vecTrial.rewardSide{intThisTrial}       = side;                                              
                              else      
                                  vecTrial.responseSide{intThisTrial}     = side;
                                  vecTrial.correctResponse(intThisTrial)  = 1;
                                  if strcmp(side,'L') 
                                  Objects.ldObj.giveReward(side,Par.CorrectRewardSizeLeft); % give reward on arduino / LEFT
                                  vecTrial.rewardSize(intThisTrial)       = Par.CorrectRewardSizeLeft;
                                  elseif strcmp(side,'R')
                                  Objects.ldObj.giveReward(side,Par.CorrectRewardSizeRight); % give reward on arduino / RIGHT
                                  vecTrial.rewardSize(intThisTrial)       = Par.CorrectRewardSizeRight;
                                  end
                                  vecTrial.rewardSide{intThisTrial}       = side;
                                               
                              end
                          vecTrial.rewardTime(intThisTrial)       = toc(refTime);      
                          LFR = 'postLFR'; 
                          vecTrial.respwinEnd(intThisTrial) = toc(refTime);                      
                          
                          else                                                                                             %incorrect                      
                          vecTrial.responseSide{intThisTrial}     = side;
                          vecTrial.correctResponse(intThisTrial)  = 0;
                          vecTrial.firstIncorrect(intThisTrial)   = 1; %works cause if lick correct then postLFR/ BADLY CODED THOUGH
                          fprintf('\b        Wrong Lick (%s)\n',side); 
                            if Par.boolSingleResp %Mouse only gets one lick to respond correcly. 
                                  state = 'Post'; 
                                  vecTrial.respwinEnd(intThisTrial) = toc(refTime);
                                  vecTrial.stimEnd(intThisTrial) = toc(refTime);
                                  vecTrial.timeoutStart(intThisTrial) = toc(refTime);
                            end
                            if Par.boolRewardPenality
                                  boolPenalized = true; %Indicates the mouse has been wrong already.
                            end
                            
                          end
                        end
                      end
                      if toc(refTime) - vecTrial.stimStart(intThisTrial) > Par.LFROffset %LFR offset reached (with no licks except..)
                      
                        if ThisTrialType=='C' && ~boolLickDetected
                            vecTrial.correctResponse(intThisTrial)  = 1;
                        end
                      
                          if Par.GivePassiveReward && ThisTrialType~='C'     
                          % Give AutoRew
                          side = Par.Stim.vecSide(intThisTrial);
                          fprintf('\b        Passive Reward  (%s)\n',side);
                          %vecTrial.correctResponse(intThisTrial)  = 0; %if no response just count it as a miss.
                          if ~boolLickDetected
                            vecTrial.noResponse(intThisTrial)       = 1;
                            vecTrial.correctResponse(intThisTrial)  = 0;
                          end
                          vecTrial.AutoRewReceived(intThisTrial)  = 1;
                          Objects.ldObj.giveReward(side,Par.PassiveRewardSize); % give reward on arduino 
                          vecTrial.rewardSide{intThisTrial}       = side;
                          vecTrial.rewardSize(intThisTrial)       = Par.PassiveRewardSize;
                          vecTrial.rewardTime(intThisTrial)       = toc(refTime); 
                          LFR = 'postLFR';
                          vecTrial.respwinEnd(intThisTrial) = toc(refTime);
                          else
                          %no PostLFR if they get no reward.
                          if ~boolLickDetected
                            vecTrial.noResponse(intThisTrial)       = 1;
                            vecTrial.correctResponse(intThisTrial)  = 0;
                          end
                          vecTrial.AutoRewReceived(intThisTrial)  = 0;
                          state = 'Post';
                          vecTrial.respwinEnd(intThisTrial) = toc(refTime);                         
                          vecTrial.stimEnd(intThisTrial) = toc(refTime);
                          vecTrial.timeoutStart(intThisTrial) = toc(refTime);
                              fprintf('Post epoch of trial %d at %.0f min %2.0f secs:\n',intThisTrial,floor(vecTrial.timeoutStart(intThisTrial)/60),mod(vecTrial.timeoutStart(intThisTrial),60));
                          end
                          
                      end
               
                case 'postLFR'              
                       if toc(refTime) - vecTrial.stimStart(intThisTrial) > Par.StimEpochDur
                              state = 'Post';  
                              vecTrial.stimEnd(intThisTrial) = toc(refTime);
                              vecTrial.timeoutStart(intThisTrial) = toc(refTime);
                              fprintf('Post epoch of trial %d at %.0f min %2.0f secs:\n',intThisTrial,floor(vecTrial.timeoutStart(intThisTrial)/60),mod(vecTrial.timeoutStart(intThisTrial),60));                       
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
                              state = 'EndTrial';
                              vecTrial.timeoutEnd(intThisTrial) = toc(refTime);
                      end               
           
                
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 'EndTrial'
                vecTrial.trialEnd(intThisTrial) = toc(refTime);

                % Update the control window
                % Get also changed parameters from the figure
                
                if strfind(Par.strTask,'mol') 
                Par = Objects.cwObj.update(Par,vecTrial,intThisTrial);
                else
                 Objects.cwObj.update(Objects.cwObj,boolLickDetected,boolLickDetected,Par,intThisTrial);
                end
                
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

