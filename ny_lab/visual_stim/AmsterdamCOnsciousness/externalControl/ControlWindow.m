classdef ControlWindow < handle
    %% CONTROLWINDOW object that creates and handles an interactive gui for mouse
    % experiments
    
    properties
    % h is a struct with the handles to uicontrols in the figure
    h = struct;
    ldObj; % the lickdetector object handle, has to be passed to the constructor
    end
    
    methods
        %% Contstructor
        function obj = ControlWindow(ldObj,Par,sessionData)
            % assign the lickdetector object handle
            obj.ldObj = ldObj;
            % Create the figure
            f = figure('name', strcat(sessionData.Mouse,'_box',sessionData.Box), 'position', [30 70 500 300]);

            %% Lickdetector panel
            uipanel('Title','Lickdetector', 'Units','pixel', 'Position',[5 200 400 95]);

            % Lickdetector panel texts
            uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [10 250 200 20], 'String', 'rewardSizeRight:');
            uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [10 230 200 20], 'String', 'rewardSizeLeft:');
            uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [225 230 20 20], 'String', 'ms');
            uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [225 250 20 20], 'String', 'ms');
            uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [10 210 200 20], 'String', 'Threshold:');

            % Lickdetector editable texts
            uicontrol('Style', 'edit' , 'Position', [180 250 40 20], 'String', num2str(ldObj.rewardSizeRight),...
             'Callback', @(h,e)obj.editrewardSizeRight_callback(h,e));
            uicontrol('Style', 'edit' , 'Position', [180 230 40 20], 'String', num2str(ldObj.rewardSizeLeft),...
             'Callback',@(h,e)obj.editrewardSizeLeft_callback(h,e));
            uicontrol('Style', 'edit' , 'Position', [180 210 40 20], 'String', num2str(ldObj.threshold),...
             'Callback',@(h,e)obj.editThreshold_callback(h,e));

            % Push buttons passive reward
            uicontrol('Style', 'pushbutton','HorizontalAlignment', 'left','Position', [250 250 90 20], 'String', 'Right reward',...
                'Callback',@(h,e)obj.pbRewardRight_callback(h,e));
            uicontrol('Style', 'pushbutton','HorizontalAlignment', 'left','Position', [250 230 90 20], 'String', 'Left reward',...
                'Callback',@(h,e)obj.pbRewardLeft_callback(h,e));

            %% Mouse performance panel
            uipanel('Title','Mouse performance', 'Units','pixel', 'Position',[5 100 400 95]);
            
            if strcmp(Par.strTask,'detection')
                % Mouse performance panel static texts
                uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [10 140 100 20], 'String', 'Target:');
                uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [10 120 100 20], 'String', 'Non-target:');
                uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [160 160 40 20], 'String', ['no:' Par.charNoSide]);
                uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [110 160 40 20], 'String', ['yes:' Par.charYesSide]);
                uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [210 160 40 20], 'String', 'O');
                
                % Mouse performance panel texts that are being updated online
                obj.h.yesTarget = uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [110 140 20 20], 'String', '0');
                obj.h.noTarget = uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [160 140 20 20], 'String', '0');
                obj.h.yesNonTarget = uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [110 120 20 20], 'String', '0');
                obj.h.noNonTarget = uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [160 120 20 20], 'String', '0');
                obj.h.omission = uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [210 140 20 20], 'String', '0');
                
            elseif strcmp(Par.strTask, 'localization')
                % Mouse performance panel static texts
                uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [10 140 100 20], 'String', 'Target R:');
                uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [10 120 100 20], 'String', 'Target L:');
                uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [110 160 40 20], 'String', 'hit');
                uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [160 160 40 20], 'String', 'miss');
                uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [210 160 40 20], 'String', 'O');
                
                % Mouse performance panel texts that are being updated online
                obj.h.hitR = uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [110 140 20 20], 'String', '0');
                obj.h.missR = uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [160 140 20 20], 'String', '0');
                obj.h.hitL = uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [110 120 20 20], 'String', '0');
                obj.h.missL = uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [160 120 20 20], 'String', '0');
                obj.h.omission = uicontrol('Style', 'text','HorizontalAlignment', 'left','Position', [210 140 20 20], 'String', '0');
            end
            drawnow;
        end
        
        %% Update the ControlWindow figure, needs trial outcome, i.e. whether trial was correct,
        % whether this was a noresponse trial AS first 2 arguments, the Par
        % struct as 3rdd and the current trial int as 4RD argument
        function update(obj,boolCorrect,boolNoResponse,Par,intThisTrial)        
            
            if strcmp(Par.strTask,'detection')                
                    % if this trial was an ommission
                if boolNoResponse
                    set(obj.h.omission, 'String', num2str(str2num(get(obj.h.omission, 'String')) + 1));    
                elseif boolCorrect
                    % hit (i.e. yes side & target)
                    if Par.Stim.CorrectSide(intThisTrial) == Par.charYesSide
                        set(obj.h.yesTarget, 'String', num2str(str2num(get(obj.h.yesTarget, 'String')) + 1));
                    % correct rejection (i.e. no side & no target
                    elseif Par.Stim.CorrectSide(intThisTrial) == Par.charNoSide
                        set(obj.h.noNonTarget, 'String', num2str(str2num(get(obj.h.noNonTarget, 'String')) + 1));
                    end
                        
                      
                else
                    if Par.Stim.CorrectSide(intThisTrial) == Par.charYesSide
                        % Miss (i.e. wrong on target trial)
                        set(obj.h.noTarget, 'String', num2str(str2num(get(obj.h.noTarget, 'String')) + 1)); 
                    % False alarm (i.e. wrong non-target
                    elseif Par.Stim.CorrectSide(intThisTrial) == Par.charNoSide
                        set(obj.h.yesNonTarget, 'String', num2str(str2num(get(obj.h.yesNonTarget, 'String')) + 1));
                    end
                end
             
                % If this is the localization task NOT UPDATED YET!
            elseif strcmp(Par.strTask,'localization')
                % if tiral outcome was omission
                if boolNoResponse
                    set(obj.h.omission, 'String', num2str(str2num(get(obj.h.omission, 'String')) + 1));   
       
                % See where the stimulus was presented and trial outcome
                elseif Par.Stim.vecSide(intThisTrial) == 'R'
                    % targetSide = 'R';
                    if boolCorrect
                        set(obj.h.hitR, 'String', num2str(str2num(get(obj.h.hitR, 'String')) + 1));  
                    else
                        set(obj.h.missR, 'String', num2str(str2num(get(obj.h.missR, 'String')) + 1));
                    end
                elseif Par.Stim.vecSide(intThisTrial) == 'L'
                    % TargetSide = 'L';
                    if boolCorrect
                        set(obj.h.hitL, 'String', num2str(str2num(get(obj.h.hitL, 'String')) + 1));  
                    else
                        set(obj.h.missL, 'String', num2str(str2num(get(obj.h.missL, 'String')) + 1));
                    end    
                end                                     
            
            end
            % Check pending callbacks at figures (i.e. ControlWindow)
            drawnow;
            
        end
  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        %% Callback functions    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Lickdetector editable texts callback functions
        function editrewardSizeRight_callback(obj, hObject, eventData)
            obj.ldObj.set('rewardSizeRight',str2double(get(hObject,'String')));
        end
        function editrewardSizeLeft_callback(obj, hObject, eventData)
            obj.ldObj.set('rewardSizeLeft',str2double(get(hObject,'String')));  
        end
        function editThreshold_callback(obj, hObject, eventData)
            obj.ldObj.set('threshold',str2double(get(hObject,'String')));
        end

        % Push button passive reward callback functions
        function pbRewardRight_callback(obj, hObject, eventData)
            obj.ldObj.giveReward('R',4);
        end
        function pbRewardLeft_callback(obj, hObject, eventData)
            obj.ldObj.giveReward('L',4);
        end   
            
    end
end
    
