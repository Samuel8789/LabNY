function acquireData_YS(channels, recording_length, pwd2)


global globalTime;
global globalData;
global timeIndex;

timeIndex = 1;

%% DAQ 

disp('Initializing DAQ...');
% Creating session
s = daq.createSession('ni');

% % which channels to record from
% channels = [8,9,10,11,12];

% Channels
% 8 - visstim
% 9 - photodiode
% 10 - locomotion
% 11 - starttrigers
% 12 - optopockels

% channel_key = [8,9,10,11,12];
% channel_names = {'visstim';
%                  'Acq Trigger';
%                  'locomotion';
%                  'starttrigers/LED';
%                  'optopockels';};
             
             
channel_key = [8,9,10,11,12,13,14];
channel_names = {'VisStim';
                 'OptoPockels';
                 'Start/End';
                 'LED';
                 'PhotoTrigger';
                 'Locomotion',
                 'SartFrame'};
             
daq_ai_chan_map = containers.Map(channel_key,channel_names);

% initialize 
s.addAnalogInputChannel('Dev2',channels,'voltage');
counter_ses=daq('ni')
addoutput(counter_ses,'Dev2',0,'voltage');


    




% make the data acquisition 'SingleEnded, to separate the '
for nchan = 1:length(channels)
%     if channel_key(nchan) <= 3 %strcmp(s.Channels(ii).ID, 'ai3')
        s.Channels(nchan).Range = [-10 10];
        s.Channels(nchan).TerminalConfig = 'SingleEnded';
        
%     end
end

s.Rate = 1000; % Cannot exceed 1666.6667 for six channels.
if recording_length == 0
    s.DurationInSeconds = input('duration in sec: '); % Change this to change duration of experiment.
else
    s.DurationInSeconds = recording_length;
end

%% Create temporary files to write  data from DAQ as it is recording
daq_data.time = fopen([pwd2 '\temp_data\temp_time.csv'], 'w');
for ii = 1:numel(channels)
    daq_data.voltage(ii) = fopen([pwd2 '\temp_data\temp_volt_data_', num2str(ii), '.csv'], 'w');
end

%% Plotting data

globalTime = zeros(s.Rate*120,1);
globalData = zeros(s.Rate*120,size(channels,2));

% change subplot dim if using fewer channels
if length(channels) == 6
    subplot_dim = [3, 2];
elseif length(channels) < 6
    subplot_dim = [length(channels), 1];
else
     subplot_dim = [4, 2];
    
end

% create plot for voltage data
figure;
for nchan = 1:length(channels)
    fig_plt.subplt(nchan) = subplot(subplot_dim(1), subplot_dim(2),nchan);
    fig_plt.plt(nchan) = plot(globalTime,globalData(:,nchan));
    xlim([0 120]);
    if channels(nchan) == 10
        ylim([-0.5 3]); % for locomotion
    elseif channels(nchan) == 8
        ylim([-0.5 10]);
    elseif channels(nchan) == 9
         ylim([-0.1 5.5]);
    elseif channels(nchan) == 11
         ylim([-0.5 5.5]);
     elseif channels(nchan) == 12
         ylim([-0.5 5.5]);
    end
    title(daq_ai_chan_map(channels(nchan)))
end

%% Data acqusition
mydlg = warndlg('Ok to strat DaqAcq', 'A Warning Dialog');
waitfor(mydlg);
disp('DaqAcq Started');

% Handle (whenever data is available, call the function inside)
lh = s.addlistener('DataAvailable', @(src,event)writeData_YS(daq_data, event, fig_plt));

disp('Recording voltage...');
s.startBackground();
write(counter_ses,5)
write(counter_ses,5)
pause(2)
write(counter_ses,0)
write(counter_ses,0)
pause(2)
write(counter_ses,5)
write(counter_ses,5)
pause(2)
write(counter_ses,0)
write(counter_ses,0)
s.wait();
delete(lh);
disp('Finished voltage recording...');

end