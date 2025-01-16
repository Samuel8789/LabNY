%% Run script to start voltage recording with NI-DAQ
clear;
close all;
tic

%% Input paramterets
expdate='20240619';
mouse='Test';
fov='Test';
acq='20x_Drift';
% % acq='Test';

% opto='470_1pOpto';
opto='617_1pOpto';
% opto='617_Mock';


% opto='Test';
% opto='1064_opto_NoChand1';


% opto='OptoNone';

recording_length= 1800; % in sec

if strcmp(acq,'TestConnections')
    recording_length =140; % in se
elseif strcmp(acq,'20x_Drift')
    recording_length= 1800; % in sec
    cameraframes=recording_length*31



elseif strcmp(acq,'15Trial')
    % frame for camera are 5500
    % for paririe voltage 150000
    % frame 1 aver 
    % frames 2 av er 200 
    % frames 2 av er 

        
    recording_length =1050; % in se
    cameraframes=recording_length*35
    
elseif strcmp(acq,'5minspontep')
    recording_length =400; % in se
    cameraframes=recording_length*35
    
elseif strcmp(acq,'CellScreen')
    % frame for camera are 5500
    % for paririe voltage 150000
    % frame 1 aver 
    % frames 2 av er 200 
    % frames 2 av er 

        
    recording_length =750; % in se
elseif strcmp(acq,'AllenA')
    % frame for camera are 115000
    % for paririe voltage 3785000
    % frame 1 aver 224000
    % frames 2 av er 112000
    % frames 2 av er 56000

        
    recording_length =3800; % in sec
elseif strcmp(acq,'AllenB')
 % frame for camera are 120000
    % for paririe voltage 3850000
    % frame 1 aver 
    % frames 2 av er 
    % frames 4 av er 58000

    recording_length =3900; % in sec
elseif strcmp(acq,'AllenC')
    % frame for camera are 112000
    % for paririe voltage 3700000
    % frame 1 aver 
    % frames 2 av er 
    % frames 4 av er 53000
    
    
    recording_length =3600; % in sec
elseif strcmp(acq,'Spont')
 % frame for camera are 20000

    recording_length =350; % in sec

end

% recording_length =4200; % in sec

acquisition_file_name = [expdate '_' mouse '_' fov '_' acq '_' opto];
display(acquisition_file_name)
display(cameraframes)
% Select NI-DAQ AI channels to record from:
% Channels
% 8 - visstim prairire 0 daqout 0
% 9 - photodiode prairire3
% 10 - locomotion prairire 7
% 11 - starttrigers/LED prairire 5
% 12 - optopockels prairire 2
% Select NI-DAQ AI channels to record from:
% Channels
% 8 - VisStim prairire 0 daqout 0
% 1-9 - OptoPockels prairire2
% 2-10 - Start/End prairire tyrig4in usb 1 orange prairue3
% 3-11 - LED prairire 5 led usb0 green
% 4-12 - PhotoTrigger pfi 8 prairire 6  daqout1
% 13 - Locomotion prairie7
% daqout0 to user1

channels = [8,9,10,11,12,13];


%% output file name generation
pwd2 = fileparts(which('voltage_recording_NI_DAQ.m'));

save_path = [pwd2 '\output_data\' expdate '\Mice\' mouse '\UnprocessedDaq\' ];
% add time info to saved file name
temp_time = clock;
save_note = '';
time_stamp = ['_', num2str(temp_time(2)), '_', num2str(temp_time(3)), '_', num2str(temp_time(1)), '_', num2str(temp_time(4)), '_', num2str(temp_time(5))];
acquisition_file_path = [save_path, acquisition_file_name, time_stamp];
clear save_path temp_time;
%% run DAQ here
% the data is acquired in buffers of 100ms and dumped into csv files in
% temp data folder
acquireData_YS(channels, recording_length, pwd2);

%% load acquired data from csvs and save as mat
disp('Saving data...');

daq_data.time = csvread([pwd2 '\temp_data\temp_time.csv']);
daq_data.voltage = zeros(length(daq_data.time), length(channels));
for nfile = 1:numel(channels)
    daq_data.voltage(:,nfile) = csvread([pwd2 '\temp_data\temp_volt_data_', num2str(nfile), '.csv']);
end
clear nfile;

% save recorded data
if ~exist(acquisition_file_path,'dir')
    mkdir(acquisition_file_path)
end
save([acquisition_file_path '.mat'], 'daq_data');

%%
% plot everything
figure;
hold on;

plot(daq_data.time, daq_data.voltage(:,1), 'g');
plot(daq_data.time, daq_data.voltage(:,2), 'y');
plot(daq_data.time, daq_data.voltage(:,3), 'k');
plot(daq_data.time, daq_data.voltage(:,4), 'r');
plot(daq_data.time, daq_data.voltage(:,5), 'b');
plot(daq_data.time, daq_data.voltage(:,6), 'c');

legend('vistim', 'photodiode', 'locomotion', 'LED', 'pockelsungacing');

%%
disp('Done');
toc

