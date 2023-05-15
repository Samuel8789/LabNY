classdef ControlWindowMatthijs < handle
    %% ControlWindowMatthijs object that creates and handles an interactive gui for mouse
    % experiments
    
    properties
        % h is a struct with the handles to uicontrols in the figure
        h = struct;
        ldObj; % the lickdetector object handle, has to be passed to the constructor
    end
    
    methods
        %% Contstructor
        function obj = ControlWindowMatthijs(ldObj,Par,sessionData)
            % Create the figure
            ctrlfigure = figure('name', strcat(sessionData.Mouse,'_box',sessionData.Box),'Units','normalized','position', [0.5 0.05 0.5 0.9]);
            
            %% Training Parameters Panel
            trainingpanel = uipanel('Parent',ctrlfigure,'Title','Training Parameters', 'Units','normalized', 'Position',[0 0.5 1 0.5]);
            
            %%On Lick Detection: (threshold + reward buttons:)
            obj.ldObj = ldObj; % assign the lickdetector object handle
            
            xpos    = 0.02;
            ypos    = 0.85;
            xwidth  = 0.15;
            ywidth  = 0.1;
            
            %Threshold:
            uicontrol('Parent',trainingpanel,'Style','text','HorizontalAlignment','left','Units','normalized','Position', [xpos ypos xwidth ywidth], 'String', 'Threshold:');
            xpos = xpos + xwidth;
            uicontrol('Parent',trainingpanel,'Style','edit','Units','normalized','Position', [xpos ypos xwidth ywidth], 'String', num2str(ldObj.threshold),...
                'Callback',@(h,e)obj.editThreshold_callback(h,e));
            xpos = xpos + xwidth*1.5;
            % Push buttons passive reward
            uicontrol('Parent',trainingpanel,'Style','pushbutton','HorizontalAlignment','left','Units','normalized','Position',...
                [xpos ypos xwidth*2 ywidth], 'String', 'Give Reward Left','Callback',@(h,e)obj.pbRewardLeft_callback(h,e));
            xpos = xpos + xwidth*2;
            uicontrol('Parent',trainingpanel,'Style','pushbutton','HorizontalAlignment','left','Units','normalized','Position',...
                [xpos ypos xwidth*2 ywidth], 'String', 'Give Reward Right','Callback',@(h,e)obj.pbRewardRight_callback(h,e));
            
            %%On Parameters under control of User:
            AllParams = {'CounterBias' 'TolerateLicksITI' 'TolerateLicksPreChange' 'TolerateLicksIncorrect' 'GivePassiveReward' 'UseTimeOut'...
                'intSecsITI' 'SecsNoLickITI' 'StimEpochDur' 'LFROnset' 'LFROffset' 'UseTrialBlocks' 'BlockLength' 'LeftTrialsFreq'  'CorrectRewardSize' 'PassiveRewardSize' 'StimPreChangeIntensity'...
                'StimLateralize' 'ProceedStages' 'dblGratingContrast' 'DeflectionIntensities'  'CatchTrialsFreq' 'MultimodalFreq' 'TactileTrialsFreq'...
                'Contrasts' 'PiezoI' 'CatchFreq' 'MultimFreq' 'TactileFreq'};
            
            xpos    = 0.02;
            ypos    = 0.75;
            xwidth  = 0.25;
            ywidth  = 0.1;
            
            %Construct on/off checkbox for each that will be passed on to
            %the task script whenever the getparam function is called
            for param = 1:length(AllParams)
                if isfield(Par,(AllParams{param}))
                    uicontrol('Parent',trainingpanel,'Style','text','HorizontalAlignment','left','Units','normalized',...
                        'Position', [xpos ypos xwidth ywidth],'String', AllParams{param});                  
                    if Par.(AllParams{param}) == 0 || Par.(AllParams{param}) == 1
                        obj.h.params.(AllParams{param}) = uicontrol('Parent',trainingpanel,'Style','checkbox','HorizontalAlignment','left','Units','normalized',...
                            'Position', [xpos+xwidth ypos xwidth/2 ywidth],'Callback',''); ypos = ypos - ywidth;
                        set(obj.h.params.(AllParams{param}),'value',Par.(AllParams{param}));
                    elseif isnumeric(Par.(AllParams{param}))
                        uicontrol('Parent',trainingpanel,'Style','text','HorizontalAlignment','left','Units','normalized',...
                            'Position', [xpos ypos xwidth ywidth],'String', AllParams{param});
                        obj.h.params.(AllParams{param}) = uicontrol('Parent',trainingpanel,'Style','edit','Units','normalized',...
                            'Position', [xpos+xwidth ypos xwidth/2 ywidth],'String', num2str(Par.(AllParams{param}))); ypos = ypos - ywidth;                    
                    end                                   
                    if ypos < 0
                        ypos = 0.75; xpos = xpos + 1.6*xwidth;
                    end
                end
            end
            
            if isfield(Par,'UpdateTrials')
                      updb = uicontrol('Parent',trainingpanel,'Style','pushbutton','HorizontalAlignment','left','enable','on','Units','normalized','Position',...
                      [xpos+xwidth ypos xwidth/2 ywidth], 'String', 'UpdateTrials','Callback','Par.UpdateTrials=1;''yay'' ');ypos = ypos - ywidth;
                      
                      if ypos < 0
                        ypos = 0.75; xpos = xpos + 1.6*xwidth;
                      end
            end
            
            %% Mouse performance panel
            perfpanel = uipanel('Parent',ctrlfigure,'Title','Mouse performance','Units','normalized','Position',[0 0.2 0.5 0.3]);
            
            xpos    = 0.05;
%             ypos    = 0.8;
            xwidth  = 0.2;
            ywidth  = 0.1;
            
            OutcomeOptions = {'HitLeft' 'ErrorToLeft' 'FALeft' 'ErrorToRight' 'HitRight' 'FARight' 'MissLeft' 'MissRight' 'CorrectRejection'};
            
            for stim = 1:3
                ypos = 0.8;
                for resp = 1:3
                    uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos ypos xwidth ywidth], 'String', OutcomeOptions{(stim-1)*3+resp});
                    obj.h.(OutcomeOptions{(stim-1)*3+resp})   = uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos+xwidth ypos xwidth/2 ywidth], 'String', '0');
                    if ismember((stim-1)*3+resp,[1 5 9])
                        set(obj.h.(OutcomeOptions{(stim-1)*3+resp}),'ForeGroundColor','g')
                    else
                        set(obj.h.(OutcomeOptions{(stim-1)*3+resp}),'ForeGroundColor','r')
                    end
                    ypos = ypos - ywidth;
                end
                xpos    = xpos + xwidth*1.7;
            end
            
            
            %                uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos ypos xwidth ywidth], 'String', '% Perfect:');
            %                obj.h.PercPerfect= uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos+xwidth ypos xwidth/2 ywidth], 'String', '0');
            
            xpos    = 0.05;
            ypos    = 0.4;
            xwidth  = 0.3;
            
            uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos ypos xwidth ywidth], 'String', 'Perc. Correct:');
            obj.h.PercCorrect= uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos+xwidth ypos xwidth/2 ywidth], 'String', '0'); ypos = ypos - ywidth;
            
            uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos ypos xwidth ywidth], 'String', 'Lapse Rate:');
            obj.h.LapseRate= uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos+xwidth ypos xwidth/2 ywidth], 'String', '0'); ypos = ypos - ywidth;
            
            uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos ypos xwidth ywidth], 'String', 'D Prime:');
            obj.h.DPrime = uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos+xwidth ypos xwidth/2 ywidth], 'String', '0'); ypos = ypos - ywidth;
            
            uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos ypos xwidth ywidth], 'String', 'Bias:');
            obj.h.Bias = uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos+xwidth ypos xwidth/2 ywidth], 'String', '0'); ypos = ypos - ywidth;
            
            uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos ypos xwidth ywidth], 'String', 'Total Reward:');
            obj.h.TotalReward = uicontrol('Parent',perfpanel,'Style', 'text','Units','normalized','HorizontalAlignment', 'left','Position', [xpos+xwidth ypos xwidth/2 ywidth], 'String', '0');
            
            %% History and upcoming trial types visualization plot  
            if Par.useHistPanel == 1
                        historypanel = uipanel('Parent',ctrlfigure,'Title','Trial History', 'Units','normalized', 'Position',[0 0 0.5 0.2]);
                        obj.h.historyplot = subplot(1,1,1,'Parent',historypanel,'Position', [0.05 0.05 0.9 0.9],'units','normalized');
                        set(obj.h.historyplot,'Xtick',[],'YTick',[])
            end
            
            %% Trial visualization panel
            %            vispanel = uipanel('Parent',ctrlfigure,'Title','Trial visualization', 'Units','normalized', 'Position',[0.5 0.2 0.5 0.4]);
            vispanel = uipanel('Parent',ctrlfigure,'Title','Trial visualization', 'Units','normalized', 'Position',[0.5 0 0.5 0.5]);
            obj.h.visplot = subplot(1,1,1,'Parent',vispanel,'Position', [0.07 0.05 0.85 0.9],'units','normalized');
            set(obj.h.visplot,'Xtick',[],'YTick',[])
            
            drawnow;
            
        end
        
        %% Update the ControlWindowMatthijs figure, needs vecTrial
        function Par = update(obj,Par,vecTrial,intThisTrial)
            
            %OutcomeOptions = {'HitLeft' 'ErrorToLeft' 'FALeft' 'ErrorToRight' 'HitRight' 'FARight' 'MissLeft' 'MissRight' 'CorrectRejection'};
            %Compute over all trials:
            HitLeft         = nansum(vecTrial.leftCorrect==1                                & vecTrial.correctResponse == 1 & vecTrial.noResponse ~= 1);
            ErrorToLeft     = nansum(vecTrial.rightCorrect==1                               & vecTrial.correctResponse == 0 & vecTrial.noResponse ~= 1);
            FALeft          = nansum(vecTrial.leftCorrect~=1 & vecTrial.rightCorrect~=1     & vecTrial.correctResponse == 0 & strcmp(vecTrial.responseSide,'L'));
            ErrorToRight    = nansum(vecTrial.leftCorrect==1                                & vecTrial.correctResponse == 0 & vecTrial.noResponse ~= 1);
            HitRight        = nansum(vecTrial.rightCorrect==1                               & vecTrial.correctResponse == 1 & vecTrial.noResponse ~= 1);
            FARight         = nansum(vecTrial.leftCorrect~=1 & vecTrial.rightCorrect~=1     & vecTrial.correctResponse == 0 & strcmp(vecTrial.responseSide,'R'));
            MissLeft        = nansum(vecTrial.leftCorrect==1                                & vecTrial.correctResponse == 0 & vecTrial.noResponse == 1);
            MissRight       = nansum(vecTrial.rightCorrect==1                               & vecTrial.correctResponse == 0 & vecTrial.noResponse == 1);
            CorrectRejection= nansum(vecTrial.leftCorrect~=1  & vecTrial.rightCorrect~=1    & vecTrial.correctResponse == 1 & vecTrial.noResponse == 1);
            
            set(obj.h.HitLeft,     'String',sprintf('%.0f',HitLeft));
            set(obj.h.ErrorToLeft, 'String',sprintf('%.0f',ErrorToLeft));
            set(obj.h.FALeft,      'String',sprintf('%.0f',FALeft));
            set(obj.h.ErrorToRight,'String',sprintf('%.0f',ErrorToRight));
            set(obj.h.HitRight,    'String',sprintf('%.0f',HitRight));
            set(obj.h.FARight,     'String',sprintf('%.0f',FARight));
            set(obj.h.MissLeft,    'String',sprintf('%.0f',MissLeft));
            set(obj.h.MissRight,   'String',sprintf('%.0f',MissRight));
            set(obj.h.CorrectRejection,'String',sprintf('%.0f',CorrectRejection));
            
            LapseRate       = (MissLeft + MissRight + CorrectRejection)/intThisTrial;
            PercCorrect     = (HitLeft + HitRight + CorrectRejection)/intThisTrial;
            
            %Show performance:
            set(obj.h.LapseRate,'String',sprintf('%.2f',LapseRate));
            set(obj.h.PercCorrect,'String',sprintf('%.2f',PercCorrect));
            %                set(obj.h.PercPerfect,'String',sprintf('%.2f',TrialPerfect/numel(lastn)));
            
            %-- Calculate d-prime left and right
            dprimeleft      = norminv(HitLeft/(HitLeft+MissLeft)) - norminv(FALeft/(FALeft+ErrorToLeft));
            dprimeright     = norminv(HitRight/(HitRight+MissRight)) - norminv(FARight/(FARight+ErrorToRight));
            set(obj.h.DPrimeLeft,'String',sprintf('%2.2f',dprimeleft));
            set(obj.h.DPrimeRight,'String',sprintf('%2.2f',dprimeright));
            
            %-- Calculate bias
            zLeft       = norminv(HitLeft/(HitLeft+ErrorToLeft));
            zRight      = norminv(HitRight/(HitRight+ErrorToRight));
            bias        = (zLeft+zRight )/2;
            set(obj.h.Bias,'String',sprintf('%2.2f',bias));
            
            %Calculate Total Reward
            TotalReward = [];
            try if intThisTrial > 1 %#ok<ALIGN>
                    totalcorrectreward = nansum(vecTrial.correctResponse(1:intThisTrial-1) .*vecTrial.rewardSize(1:intThisTrial-1));
                    totalpassivereward = nansum(vecTrial.passiveReward(1:intThisTrial-1)    *Par.PassiveRewardSize);
                    TotalReward = nansum([totalcorrectreward totalpassivereward]);
                end
            catch; end
            set(obj.h.TotalReward,'String',sprintf('%3.0f',round(TotalReward)));
            
            %% Get Parameter settings from the ControlWindowMatthijs figure
            % Change the settings to the values in the control window
            if isfield(obj.h,'params')
                AllParams = fieldnames(obj.h.params);
                for param = 1:length(AllParams)
                    if Par.(AllParams{param}) == 0 || Par.(AllParams{param}) == 1
                        Par.(AllParams{param}) = get(obj.h.params.(AllParams{param}),'value');
                    elseif isnumeric(Par.(AllParams{param})) && isnumeric(str2num(get(obj.h.params.(AllParams{param}),'String')))
                        Par.(AllParams{param}) = str2num(get(obj.h.params.(AllParams{param}),'String'));
                    end
                end
            end
            
            %% Update trial visualization panel
            cla(obj.h.visplot,'reset')
            subplot(obj.h.visplot)
            lastn = intThisTrial-9:intThisTrial; lastn = lastn(lastn>0);
            
            states      = {'iti'          'stim'         'respwin'      'timeout'};
            statecolors = {[0.65 0.85 0.85], [0.9 0.7 0.3], [0.8 0.8 0.3], [0.7 0.6 0.6]};
            
            trialtypes  = {'V'          'T'          'M'            'C'         'Y'              'X'};
            trialcolors = {[0.8 0.2 0.2], [0.2 0.2 0.8], [0.7 0.2 0.7], [0.4 0.4 0.4] ,[0.95 0.5 0.5] ,[0.5 0.5 0.95]};
            
            RewardColor     = [0.1 0.1 0.1];
            AudioLickColor  = [1 0 0];
            VisualLickColor = [0 0 1];
            RightLickColor  = [1 0 0];
            LeftLickColor   = [0 1 0];

            for trial = lastn
                rectangle('Position',[0 trial vecTrial.trialEnd(trial)-vecTrial.trialStart(trial) 0.25],'FaceColor','k','EdgeColor','k'); hold all;
                text(0,trial+0.5,char(vecTrial.trialType(trial)),'FontSize',20);

                for state = 1:length(states)
                    if isfield(vecTrial,strcat(states{state},'Start')) && isfield(vecTrial,strcat(states{state},'End'))
                        
                        ts_start    = vecTrial.(strcat(states{state},'Start'))(trial) - vecTrial.trialStart(trial);
                        ts_end      = vecTrial.(strcat(states{state},'End'))(trial)  - vecTrial.trialStart(trial);
                        
                        if strcmp(states{state},'stim')
                            idx     = strcmp(trialtypes,char(vecTrial.trialType(trial)));
                            rectangle('Position',[ts_start trial+0.25 ts_end-ts_start 0.5],'FaceColor',trialcolors{idx},'EdgeColor',trialcolors{idx}); hold all;
                        elseif strcmp(states{state},'respwin')
                            rectangle('Position',[ts_start trial+0.25 ts_end-ts_start 0.2],'FaceColor',statecolors{state},'EdgeColor',statecolors{state}); hold all;
                        else
                            rectangle('Position',[ts_start trial+0.25 ts_end-ts_start 0.5],'FaceColor',statecolors{state},'EdgeColor',statecolors{state}); hold all;
                        end
                    end
                end
                
                if isfield(vecTrial,'stimChange')
                    rectangle('Position',[vecTrial.stimChange(trial) - vecTrial.trialStart(trial) trial+0.25 0.1 0.5],'FaceColor',[0.9 0.1 0.9],'EdgeColor',[0.1 0.1 0.1]); hold all;
                end
                
                if isfield(vecTrial,'rewardTime')
                    if ~isnan(vecTrial.rewardTime(trial))
                        rectangle('Position',[vecTrial.rewardTime(trial)-vecTrial.trialStart(trial) trial+0.75 0.3 0.25],'FaceColor',RewardColor,'EdgeColor',RewardColor); hold all;
                    end
                end
                
                if isfield(vecTrial,'passiveRewardTime')
                    if ~isnan(vecTrial.passiveRewardTime(trial))
                        rectangle('Position',[vecTrial.passiveRewardTime(trial)-vecTrial.trialStart(trial) trial+0.75 0.3 0.25],'FaceColor',RewardColor,'EdgeColor',RewardColor); hold all;
                    end
                end
                
                for lick = 1:length(vecTrial.lickTime{trial})
                    if isfield(Par,'VisualLeftCorrectSide')
                        if Par.VisualLeftCorrectSide
                            RightLickColor = AudioLickColor; LeftLickColor = VisualLickColor;
                        else RightLickColor = VisualLickColor; LeftLickColor = AudioLickColor;
                        end
                    end
                    licktime = vecTrial.lickTime{trial}(lick) - vecTrial.trialStart(trial);
                    if vecTrial.lickSide{trial}(lick) == 'R'
                        rectangle('Position',[licktime trial+0.75 0.05 0.25],'FaceColor',RightLickColor,'EdgeColor',RightLickColor,'LineWidth', 0.8); hold all;
                    elseif vecTrial.lickSide{trial}(lick) == 'L'
                        rectangle('Position',[licktime trial+0.75 0.05 0.25],'FaceColor',LeftLickColor,'EdgeColor',LeftLickColor,'LineWidth', 0.8); hold all;
                    end
                end
            end
            
            xmax = max(vecTrial.trialEnd(lastn) - vecTrial.trialStart(lastn));
            if xmax>15; xmax = 15; end
            xlim(obj.h.visplot,[0 xmax])
            ylim(obj.h.visplot,[min(lastn) max(lastn)+1])
            set(obj.h.visplot,'Xtick',2:2:floor(xmax),'YTick',[lastn+0.5],'yticklabel',cellstr(num2str(lastn'))')

            %% Update the trial history panel   
            if Par.useHistPanel == 1
                    cla(obj.h.historyplot,'reset')
                    subplot(obj.h.historyplot)
                    
                    CurrentWindowF = (intThisTrial+1):(intThisTrial+10); CurrentWindowF = CurrentWindowF(CurrentWindowF<Par.intTrialNum);
                    CurrentWindowP = (intThisTrial-10):(intThisTrial); CurrentWindowP = CurrentWindowP(CurrentWindowP>0);
                    
                    %Ypos:
                    %Visual : 4  ;  Tactile : 3  ;  Multi : 2   ;  Catch : 1  ;
                    Yposes= zeros(1,Par.intTrialNum);
                    Yposes(Par.Stim.vecTrialType=='V')=4;
                    Yposes(Par.Stim.vecTrialType=='T')=3;
                    Yposes(Par.Stim.vecTrialType=='M')=2;
                    Yposes(Par.Stim.vecTrialType=='C')=1;
                    
                    YposesP= zeros(1,Par.intTrialNum);
                    YposesP(vecTrial.trialType=='V')=4;
                    YposesP(vecTrial.trialType=='T')=3;
                    YposesP(vecTrial.trialType=='M')=2;
                    YposesP(vecTrial.trialType=='C')=1;
                    
                    XleftF=CurrentWindowF(Par.Stim.leftCorrect(CurrentWindowF)==1); %left trials numbers
                    YposvecleftF  = Yposes(XleftF); %Ypos depends on trial type : V,T,M,C
                    XrightF = CurrentWindowF(Par.Stim.rightCorrect(CurrentWindowF)==1); %right trials numbers
                    YposvecrightF = Yposes(XrightF); %Ypos depends on trial type : V,T,M,C
                    XcatchF = CurrentWindowF(Par.Stim.vecTrialType(CurrentWindowF)=='C'); %catch
                    YposveccatchF  = Yposes(XcatchF);
                    
                    XleftP=CurrentWindowP(vecTrial.leftCorrect(CurrentWindowP)==1); %left trials numbers
                    YposvecleftP  = YposesP(XleftP);  %create Ypos vector for left
                    XrightP = CurrentWindowP(vecTrial.rightCorrect(CurrentWindowP)==1); %right trials numbers
                    YposvecrightP = YposesP(XrightP); %create Ypos vector for right
                    XcatchP = CurrentWindowP(vecTrial.trialType(CurrentWindowP)=='C'); %catch
                    YposveccatchP  = YposesP(XcatchP);
                    
                    scatter(XleftF,YposvecleftF,15,[1 0 0]) %left trials future
                    hold on
                    scatter(XrightF,YposvecrightF,15,[0 1 0]) %right trials future
                    hold on
                    scatter(XleftP,YposvecleftP,15,[1 0 0]) %left trials past
                    hold on
                    scatter(XrightP,YposvecrightP,15,[0 1 0]) %right trials past
                    hold on
                    scatter(XcatchP,YposveccatchP,15,[0.4 0.4 0.4]) %C trials past
                    hold on
                    scatter(XcatchF,YposveccatchF,15,[0.4 0.4 0.4]) %C trials future
                    hold on
                
                  if sum(~isnan(vecTrial.correctResponse(CurrentWindowP)))>0
                      XcorrectP = CurrentWindowP(vecTrial.correctResponse(CurrentWindowP)==1); %correct trials past
                      XwrongP = CurrentWindowP(vecTrial.firstIncorrect(CurrentWindowP)==1);
                      Ycorrect=YposesP(XcorrectP);
                      Ywrong=YposesP(XwrongP);
                      
                      scatter(XcorrectP,Ycorrect,12,[0.95 0.85 0.05],'filled')
                      hold on
                      scatter(XwrongP,Ywrong,8,[0.1 0.1 0.1],'filled')
                      hold on
                  end
                
                if sum(~isnan(vecTrial.AutoRewReceived(CurrentWindowP)))>0
                    XpassiveP = CurrentWindowP(vecTrial.AutoRewReceived(CurrentWindowP)==1); %AutoRew trials past
                    YpassiveP = YposesP(XpassiveP);
                    scatter(XpassiveP,YpassiveP,6,[0.09 0.78 0.86],'filled')
                    hold on
                end
                
                line([intThisTrial+1 intThisTrial+1],[0 5],'Color',[0.6 0.6 0.6],'LineWidth', 3)
                
                hold off
                
                xlim(obj.h.historyplot, [min(CurrentWindowP) max(CurrentWindowF)])
                ylim(obj.h.historyplot, [0 5])
                set(obj.h.historyplot, 'YTick',[1 2 3 4],'yticklabel',{'C','M','T','V'})
            end
            % Check pending callbacks at figures (i.e. ControlWindowMatthijs)
            drawnow;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Callback functions
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Lickdetector editable texts callback functions
        
        function editThreshold_callback(obj, hObject,~)
            obj.ldObj.set('threshold',str2double(get(hObject,'String')));
        end
        
        % Push button passive reward callback functions
        function pbRewardRight_callback(obj, hObject,EventData) %#ok<INUSD>
            obj.ldObj.giveReward('R',5);
        end
        function pbRewardLeft_callback(obj, hObject,EventData) %#ok<INUSD>
            obj.ldObj.giveReward('L',5);
        end
        
    end
end


