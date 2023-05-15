function flush_it()

global uihandles
close all;

ldObj = LickDetector;
boolRunning = true;
boolOpenLeft = false;
boolOpenRight = false;

uihandles.setup_fig=figure('Name','',...
    'units','normalized',...
    'outerposition',[0 0 0.3 1],...
    'Color',[1 1 1],...
    'Resize','on',...
    'NumberTitle','off'...
    ,'tag','_main');
    
% TITLE
    
x_pos=0.1;
y_pos=0.9;
x_width=0.4;
y_width=0.06;

uihandles.header_string = uicontrol('Parent',uihandles.setup_fig,'Style','text','units','normalized',...
    'Position',[x_pos y_pos 2*x_width 2*y_width],'String','FLUSH',...
    'BackgroundColor',[1 1 1],'fontsize',18); y_pos = y_pos-y_width;

%% Buttons for opening/closing the valves

x_pos   = 0.1;
y_pos   = 0.4;
y_width = 0.2;
x_width = 0.25;
uihandles.OpenLeft = uicontrol('Parent',uihandles.setup_fig,'Style','togglebutton','units','normalized',...
    'Position',[x_pos y_pos x_width y_width],'String','Open LEFT Valve','fontsize',12,...
    'backgroundcolor',[0.7,0.8,1],'ForegroundColor',[0,0,1]);
    
x_pos   = 0.65;
y_pos   = 0.4;
y_width = 0.2;
x_width = 0.25;  
uihandles.OpenRight = uicontrol('Parent',uihandles.setup_fig,'Style','togglebutton','units','normalized',...
'Position',[x_pos y_pos x_width y_width],'String','Open RIGHT Valve','fontsize',12,...
'backgroundcolor',[0.7,1,0.8],'ForegroundColor',[0,1,0]);

%STOP button
x_pos   = 0.25;
y_pos   = 0.1;
x_width = 0.75;
y_width = 0.2;
uihandles.Stop = uicontrol('Parent',uihandles.setup_fig,'Style','push','units','normalized',...
'Position',[x_pos y_pos x_width y_width],'String','EXIT','fontsize',12,...
'backgroundcolor',[1, 0.2, 0.2],'Callback','boolRunning=false;');

try
    while boolRunning
      boolOpenLeft = get(uihandles.OpenLeft,'Value');
      boolOpenRight = get(uihandles.OpenRight,'Value');
    
    if boolOpenLeft == 0 %its inverted cos button value starts at 1
          set(uihandles.OpenLeft,'String','Close LEFT Valve','BackgroundColor',[1,0,0]);
          set(uihandles.OpenRight,'Enable','off');
          set(uihandles.Stop,'Enable','off');
          ldObj.giveReward('L',500);
    elseif boolOpenRight == 0          %Can´t open RIGHT if LEFT open.
          set(uihandles.OpenRight,'String','Close RIGHT Valve','BackgroundColor',[1,0,0]);
          set(uihandles.OpenLeft,'Enable','off');
          set(uihandles.Stop,'Enable','off');
          ldObj.giveReward('R',500);
    else
          set(uihandles.OpenLeft,'Enable','on','String','Open LEFT Valve','BackgroundColor',[0,1,0]);
          set(uihandles.OpenRight,'Enable','on','String','Open RIGHT Valve','BackgroundColor',[0,1,0]);
          set(uihandles.Stop,'Enable','on');
    end
          pause(0.49)
    end

catch
    % delete LickDetector object
    delete(ldObj); clear ldObj
    %Show error
    rethrow(errorMessage);
end

delete(ldObj); clear ldObj
close(uihandles.setup_fig); clear uihandles;


end
