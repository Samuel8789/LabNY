function dl_setup_session_fig()
%% Main Startup Script to start behavioral training

global uihandles
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load in settings from current pc (dl_pc_settings function should bin in path)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
uihandles.pcsettings = dl_pc_settings;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%display figure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
uihandles.setup_fig=figure('Name','',...
    'units','normalized',...
    'outerposition',[0 0 0.3 1],...
    'Color',[1 1 1],...
    'Resize','on',...
    'NumberTitle','off'...
    ,'tag','_main');

x_pos=0.1;
y_pos=0.9;
x_width=0.4;
y_width=0.06;

uihandles.header_string = uicontrol('Parent',uihandles.setup_fig,'Style','text','units','normalized',...
    'Position',[x_pos y_pos 2*x_width 2*y_width],'String','BEHAVIORAL TRAINING',...
    'BackgroundColor',[1 1 1],'fontsize',18); y_pos = y_pos-y_width;

% The available boxes
uihandles.Boxes = {'1','2','3','4','5','6','7','8','ExpSetupMatthijs'};

uihandles.choose_box_header = uicontrol('Parent',uihandles.setup_fig,'Style', 'text','units','normalized',...
    'Position',[x_pos y_pos x_width*1.5 y_width],'String','Choose Box:','fontsize',11,'HorizontalAlignment','left');
uihandles.choose_box = uicontrol('Parent',uihandles.setup_fig,'Style','popupmenu','units','normalized',...
    'Position',[x_pos+x_width*1.5 y_pos x_width/2 y_width],'String',uihandles.Boxes,'fontsize',11); y_pos = y_pos-y_width;

uihandles.choose_mouse_header = uicontrol('Parent',uihandles.setup_fig,'Style', 'text','units','normalized',...
    'Position',[x_pos y_pos x_width y_width],'String','Choose Mouse:','fontsize',11,'HorizontalAlignment','left');
uihandles.choose_mouse = uicontrol('Parent',uihandles.setup_fig,'Style','edit','units','normalized',...
    'Position',[x_pos+x_width y_pos x_width y_width],'String','Mouse','fontsize',11); y_pos = y_pos-y_width;

uihandles.mouse_weight_header = uicontrol('Parent',uihandles.setup_fig,'Style', 'text','units','normalized',...
    'Position',[x_pos y_pos x_width y_width],'String','Weight(gr):','fontsize',11,'HorizontalAlignment','left');
uihandles.mouse_weight = uicontrol('Parent',uihandles.setup_fig,'Style','edit','units','normalized',...
    'Position',[x_pos+x_width y_pos x_width y_width],'String','','fontsize',11); y_pos = y_pos-y_width;

uihandles.Experimenters = {'Test' 'Matthijs' 'Jean' 'Jeanette' 'Reinder'};
uihandles.choose_experimenter_header = uicontrol('Parent',uihandles.setup_fig,'Style', 'text','units','normalized',...
    'Position',[x_pos y_pos x_width y_width],'String','Experimenter:','fontsize',11,'HorizontalAlignment','left');
uihandles.choose_experimenter = uicontrol('Parent',uihandles.setup_fig,'Style','popupmenu','units','normalized',...
    'Position',[x_pos+x_width y_pos x_width y_width],'String',uihandles.Experimenters,'fontsize',11,'Callback',@(hObject, e)experim_callback(hObject,e)); y_pos = y_pos-y_width;

%% List the trainingscripts in the protocolfolder according to experimenter
mFiles = dir(fullfile(uihandles.pcsettings.protocolFolder, '*.m'));
uihandles.Protocols = {mFiles.name};
for prot = 1:length(uihandles.Protocols) 
      uihandles.Protocols{prot} = uihandles.Protocols{prot}(1:end-2);
end

uihandles.choose_protocol_header = uicontrol('Parent',uihandles.setup_fig,'Style', 'text','units','normalized',...
    'Position',[x_pos y_pos x_width/1.5 y_width],'String','Protocol:','fontsize',11,'HorizontalAlignment','left');
uihandles.choose_protocol = uicontrol('Parent',uihandles.setup_fig,'Style','popupmenu','units','normalized',...
    'Position',[x_pos+x_width/1.5 y_pos x_width+x_width/3 y_width],'String',uihandles.Protocols,'fontsize',11); y_pos = y_pos-y_width;

%% List the parameterfiles in the parameterfolder (make sure it is loaded)
mFiles = dir(fullfile(uihandles.pcsettings.parametersFolder, '*.m'));
uihandles.Parameters = {mFiles.name};
for para = 1:length(uihandles.Parameters)
    uihandles.Parameters{para} = uihandles.Parameters{para}(1:end-2);
end

uihandles.choose_parameters_header = uicontrol('Parent',uihandles.setup_fig,'Style','text','units','normalized',...
    'Position',[x_pos y_pos x_width/1.5 y_width],'String','Parameters:','fontsize',11,'HorizontalAlignment','left');
uihandles.choose_parameters = uicontrol('Parent',uihandles.setup_fig,'Style','popupmenu','units','normalized',...
    'Position',[x_pos+x_width/1.5 y_pos x_width+x_width/3 y_width],'String',uihandles.Parameters,'fontsize',11); y_pos = y_pos-y_width;

uihandles.comments_header = uicontrol('Parent',uihandles.setup_fig,'Style', 'text','units','normalized',...
    'Position',[x_pos y_pos x_width/1.5 y_width],'String','Comments','fontsize',11,'HorizontalAlignment','left');
uihandles.comments = uicontrol('Parent',uihandles.setup_fig,'Style','edit','units','normalized',...
    'Position',[x_pos+x_width/1.5 y_pos x_width+x_width/3 y_width],'String','','fontsize',11); y_pos = y_pos-y_width;
    
uihandles.choose_trainingstage_header = uicontrol('Parent',uihandles.setup_fig,'Style','text','units','normalized',...
    'Position',[x_pos y_pos x_width*1.5 y_width],'String','Tr. Stage:','fontsize',11,'HorizontalAlignment','left');
uihandles.choose_trainingstage = uicontrol('Parent',uihandles.setup_fig,'Style','popupmenu','units','normalized','HorizontalAlignment','right',...
    'Position',[x_pos+x_width*1.5 y_pos x_width/2 y_width],'String',cellstr(num2str([1:8]'))','fontsize',11); y_pos = y_pos-y_width;
    
x_pos   = 0.1;
y_pos   = 0.15;
y_width = 0.2;
x_width = 0.7;

%% Button for starting the task
uihandles.starttask = uicontrol('Parent',uihandles.setup_fig,'Style','push','units','normalized',...
    'Position',[x_pos y_pos x_width y_width],'String','START','fontsize',15,...
    'backgroundcolor',[0.7,1,0.5],'Callback','dl_initialize_session');

y_width = 0.025;
uihandles.updateline1 = uicontrol('Parent',uihandles.setup_fig,'Style', 'text','units','normalized','foregroundcolor',[0.25 0.25 0.9], ...
    'Position',[x_pos y_pos-0.025 x_width y_width],'String','','fontsize',8,'backgroundcolor',[1 1 1]); y_pos = y_pos-y_width;
uihandles.updateline2 = uicontrol('Parent',uihandles.setup_fig,'Style', 'text','units','normalized','foregroundcolor',[0.25 0.25 0.9], ...
    'Position',[x_pos y_pos-0.025 x_width y_width],'String','','fontsize',8,'backgroundcolor',[1 1 1]); y_pos = y_pos-y_width;
uihandles.updateline3 = uicontrol('Parent',uihandles.setup_fig,'Style', 'text','units','normalized','foregroundcolor',[0.25 0.25 0.9], ...
    'Position',[x_pos y_pos-0.025 x_width y_width],'String','','fontsize',8,'backgroundcolor',[1 1 1]); y_pos = y_pos-y_width;
uihandles.updateline4 = uicontrol('Parent',uihandles.setup_fig,'Style', 'text','units','normalized','foregroundcolor',[0.25 0.25 0.9], ...
    'Position',[x_pos y_pos-0.025 x_width y_width],'String','','fontsize',8,'backgroundcolor',[1 1 1]); y_pos = y_pos-y_width;
uihandles.updateline5 = uicontrol('Parent',uihandles.setup_fig,'Style', 'text','units','normalized','foregroundcolor',[0.25 0.25 0.9], ...
    'Position',[x_pos y_pos-0.025 x_width y_width],'String','','fontsize',8,'backgroundcolor',[1 1 1]); y_pos = y_pos-y_width;
      
%Callback function for showing only relevant protocols     
function experim_callback(hObject, e) %#ok<INUSD>
global uihandles
%Get all protocols and parameters.
  mFiles = dir(fullfile(uihandles.pcsettings.protocolFolder, '*.m'));
  allprotocols = {mFiles.name};
  mFiles = dir(fullfile(uihandles.pcsettings.parametersFolder, '*.m'));
  allparameters = {mFiles.name};
%Get experimenter
  experim = uihandles.Experimenters{get(uihandles.choose_experimenter,'Value')};
%Subselect relevant  
  switch experim
          case 'Matthijs' 
            uihandles.Protocols = allprotocols(~cellfun(@isempty,strfind(allprotocols,'mol'))); 
            uihandles.Parameters = allparameters(~cellfun(@isempty,strfind(allparameters,'mol')));        
          case 'Jean' 
            uihandles.Protocols = allprotocols(~cellfun(@isempty,strfind(allprotocols,'pie')));
            uihandles.Parameters = allparameters(~cellfun(@isempty,strfind(allparameters,'pie')));          
          case 'Reinder'
            uihandles.Protocols = allprotocols(~cellfun(@isempty,strfind(allprotocols,'rd_')));
            uihandles.Parameters = allparameters(~cellfun(@isempty,strfind(allparameters,'rd_')));  
          case 'Jeanette' 
            uihandles.Protocols = allprotocols(~cellfun(@isempty,strfind(allprotocols,'jlo')));
            uihandles.Parameters = allparameters(~cellfun(@isempty,strfind(allparameters,'jlo')));  
          end
  for prot = 1:length(uihandles.Protocols) 
    uihandles.Protocols{prot} = uihandles.Protocols{prot}(1:end-2);
  end  
  for para = 1:length(uihandles.Parameters)
    uihandles.Parameters{para} = uihandles.Parameters{para}(1:end-2);
  end
 %Set the new popupmenu to the subselected strings
  set(uihandles.choose_protocol,'String',uihandles.Protocols)
  set(uihandles.choose_parameters,'String',uihandles.Parameters)

end
    
end
