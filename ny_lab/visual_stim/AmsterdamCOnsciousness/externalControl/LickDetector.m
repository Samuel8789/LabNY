classdef LickDetector < handle
    %LICKDETECTOR Communicate with an Arduino-based lick detector
    % Objects of class LickDetector allow for communication with the
    % arduino-based lick detector
    properties
        % Default properties for serial connection
        baudRate    = 115200;
        timeOut     = 0.001;
        serialStr   = '/dev/ttyS999';
        serPort;    % serial port object
        
        %Properties for lick detection 
        threshold   = 120; % lick detection threshold (i.e. times the baseline)

        %Properties for water delivery and calibration:
        rewardSizeLeft          = 6;    % default reward size for left port
        rewardSizeRight         = 6;    % default reward size for right port
        rewardDurationLeft      = 120;  % default reward size for left port
        rewardDurationRight     = 120;  % default reward size for right port
        box                     = 2;    % default associated box for calibration
        
        WaterCalibTableLeft = [... % Reward durations / Water Calibration Table:
        %mL 1   2   3   4   5   6   7   8   9   10 ul
            29  47  65  83  101 119 136 154 172 190;... %Box 1
            38  54  70  87  103 119 135 151 167 184;... %Box 2
            45	58	70	82	94	106	118	130	142	155;... %Box 3
            31	49	66	83	100	117	135	152	169	186;... %Box 4
            30  47  64  81  99  116 133 150 167 184;... %Box 5
            30  47  64  81  99  116 133 150 167 184;... %Box 6
            30  47  64  81  99  116 133 150 167 184;... %Box 7
            30  47  64  81  99  116 133 150 167 184;... %Box 8
            39  81  122 164 206 247 289 331 372 414];   %Box 9 = Exp Setup Matthijs

        WaterCalibTableRight = [... 
        %mL 1   2   3   4   5   6   7   8   9   10 ul
            32  47  61  76  91  106 121 136 150 165;... %Box 1
            37  52  67  81  96  110 125 140 154 169;... %Box 2
            27	42	57	71	86	101	115	130	145	160;... %Box 3
            37	53	69	85	100	116	132	148	164	180;... %Box 4
            30  47  64  81  99  116 133 150 167 184;... %Box 5
            30  47  64  81  99  116 133 150 167 184;... %Box 6
            30  47  64  81  99  116 133 150 167 184;... %Box 7
            30  47  64  81  99  116 133 150 167 184;... %Box 8
            40  72  104 136 169 201 233 265 298 330];   %Box 9 = Exp Setup Matthijs

        % The arduino runtime at which the behavioral reftime was started,
        % this used for syncing the arduino time with the behavioral
        % training script time
        arduinoRefTime = 0; 
    end
    
    methods
        %% Constructor
        function obj = LickDetector(varargin)
            % If there are arguments set properties
            if nargin > 0
                % Switch the arguments and set properties accordingly
                for i=1:2:size(varargin,2)
                    if any(strcmp(varargin{i},fieldnames(obj)))
                        obj.(varargin{i}) = varargin{i+1};
                    end
                end
            end
            % Initiate the serial port object
            obj.serPort = serial(obj.serialStr);             
                
            % Set some parameters for serial connection
            set(obj.serPort, 'baudrate', obj.baudRate); % baudrate needs to be the same as arduino
            set(obj.serPort, 'timeout', obj.timeOut);
            
     
            % Open serial connection if in Matlab and not octave
            isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0; % check if we are in octave
            if ~isOctave
              if strcmp(get(obj.serPort, 'status'), 'closed'); fopen(obj.serPort); end;
            end
            pause(0.05);
            % Set default values of arduino:
            set(obj,'threshold',obj.threshold);
%             set(obj,'rewardSizeLeft',obj.rewardSizeLeft);
%             set(obj,'rewardSizeRight',obj.rewardSizeRight);
            
        end
        
        %% Setting method %%
        % 'I' stands for input (i.e. referenced to Arduino)
        function set(obj, prop, value)
            % Set LickDetect property to value            
            obj.(prop) = value;

            % Switch the property to set the corresponding arduino value
            switch prop
                case 'threshold'
                    % Set lick detection threshold
                    inputMessage = ['IF ' num2str(value) 'f'];
                case 'rewardSizeLeft'
                    % Set reward duration for left port:
                    rewardDuration = obj.WaterCalibTableLeft(obj.box,value);
                    inputMessage = ['IL ' num2str(rewardDuration)];
                case 'rewardSizeRight'
                    % Set reward duration for right port:
                    rewardDuration = obj.WaterCalibTableRight(obj.box,value);
                    inputMessage = ['IR ' num2str(rewardDuration)];
                case 'rewardDurationLeft'
                    % Set reward duration for left port:
                    inputMessage = ['IL ' num2str(value)];
                case 'rewardDurationRight'
                    % Set reward duration for right port:
                    inputMessage = ['IR ' num2str(value)];
            end
            % check if we are in octave
            isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0; 

            % try sending the message to the arduino in Octave
            if isOctave;
                srl_write(obj.serPort,inputMessage);
                % Wait for a response
                iAttempt = 1; % var for storing number of attempts
                response = 'none'; % var for storing response
                while ~strcmp(response, 'D') % Stop waiting if response is there
                    % Check the response from arduino
                    %(in octave there is no check for bytes) 
                    response = obj.serial_read_line();                  
                    iAttempt = iAttempt+1; %increment attempts
                    pause(0.001);
                    % After number of attempts without response, try again
                    if iAttempt > 30
                        % try sending the message to the arduino in Octave or Matlab
                        srl_write(obj.serPort,inputMessage);                       
                        iAttempt = 1;
                        disp(['Trouble setting ', prop ' , trying again']);
                    end
                end
                
            % Try sending the message to the arduino in Matlab    
            else
                fprintf(obj.serPort,inputMessage);
                % Wait for a response
                iAttempt = 1; % var for storing number of attempts
                response = 'none'; % var for storing response
            
                while ~strcmp(response, 'D') % Stop waiting if response is there
                    % Check the response from arduino
                    if obj.serPort.BytesAvailable
                        response = fscanf(obj.serPort, '%s');
                    end;
                    iAttempt = iAttempt+1; %increment attempts
                    pause(0.001)
                    % After number of attempts without response, try again
                    if iAttempt > 30
                        % try sending the message to the arduino
                        fprintf(obj.serPort,inputMessage);                
                        iAttempt = 1;
                        disp(['Trouble setting ', prop ' , trying again']);
                    end
                end
            end                   
            % Print to command window that it worked
            fprintf('Succesfully set %s on lickdetector arduino\n',prop);
        end
        
        
        %% Getting method %%
        % Only works for getting runtime Oct 2016
        % 'I' stands for input (i.e. referenced to Arduino)
        function value = get(obj, prop)
            % check if we are in octave
            isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0; 
            
            value = 0;
            % Get LickDetect property to value
            switch prop
                case 'runtime'
                    inputMessage = 'IC ';
            end

            % try sending the message to the arduino in Octave
            if isOctave;
                srl_write(obj.serPort,inputMessage);
                % Wait for a response
                iAttempt = 1; % var for storing number of attempts
                response = 'none'; % var for storing response
                while ~strcmp(response, 'R') % Stop waiting if response is there
                    % Check the response from arduino, use custom serial_read_line function
                    %(in octave there is no check for bytes) 
                    response = obj.serial_read_line();
                    iAttempt = iAttempt+1; %increment attempts
                    pause(0.001);
                    % After number of attempts without response, try again
                    if iAttempt > 30
                        % try sending the message to the arduino in Octave or Matlab

                        srl_write(obj.serPort,inputMessage);                       
                        iAttempt = 1;
                        disp(['Trouble getting ', prop ' , trying again']);
                    end
                end
                % Read the value after 'R' (i.e. second returned line)
                value = str2double(obj.serial_read_line());
    
                
            % Try sending the message to the arduino in Matlab    
            else
                fprintf(obj.serPort,inputMessage);
                % Wait for a response
                iAttempt = 1; % var for storing number of attempts
                response = 'none'; % var for storing response
            
                while ~strcmp(response, 'R') % Stop waiting if response is there
                    % Check the response from arduino
                    if obj.serPort.BytesAvailable
                        response = fscanf(obj.serPort, '%s');
                    end;
                    iAttempt = iAttempt+1; %increment attempts
                    pause(0.001)
                    % After number of attempts without response, try again
                    if iAttempt > 30
                        % try sending the message to the arduino
                        fprintf(obj.serPort,inputMessage);                
                        iAttempt = 1;
                        disp(['Trouble getting ', prop ' , trying again']);
                    end
                end
                % Scan the value byte after 'R'
                if obj.serPort.BytesAvailable
                    value = str2double(fscanf(obj.serPort, '%s'));
                end;
            end     
        end
        
        %% Give reward
        function giveReward(obj,charSide,rewardSize)
            % check if we are in octave
            isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
            
            % Input message for giving a reward
            if rewardSize<15 %Input is meant as ml reward
                if strcmp(charSide,'R')
                rewardDuration = obj.WaterCalibTableRight(obj.box,rewardSize);
            elseif strcmp(charSide,'L')
                rewardDuration = obj.WaterCalibTableLeft(obj.box,rewardSize);
                else
                rewardDuration = obj.WaterCalibTableLeft(obj.box,rewardSize);
            end
            else %Input is meant as ms opening times of the valves
                rewardDuration = rewardSize;
            end
            
            inputMessage = ['IP' charSide num2str(rewardDuration) 'f'];
            % try sending the message to the arduino in Octave
            if isOctave;
                srl_write(obj.serPort,inputMessage);
                % Wait for a response
                iAttempt = 1; % var for storing number of attempts
                response = 'none'; % var for storing response
                while ~strcmp(response, 'D') % Stop waiting if response is there
                    % Check the response from arduino
                    %(in octave there is no check for bytes) 
                    response = obj.serial_read_line();                  
                    iAttempt = iAttempt+1; %increment attempts
                    pause(0.001);
                    % After number of attempts without response, try again
                    if iAttempt > 30
                        % try sending the message to the arduino in Octave or Matlab

                        srl_write(obj.serPort,inputMessage);                       
                        iAttempt = 1;
                        disp('Trouble giving reward, trying again');
                    end
                end
                
            % Try sending the message to the arduino in Matlab    
            else
                fprintf(obj.serPort,inputMessage);
                % Wait for a response
                iAttempt = 1; % var for storing number of attempts
                response = 'none'; % var for storing response
            
                while ~strcmp(response, 'D') % Stop waiting if response is there
                    % Check the response from arduino
                    if obj.serPort.BytesAvailable
                        response = fscanf(obj.serPort, '%s');
                    end;
                    iAttempt = iAttempt+1; %increment attempts
                    pause(0.001)
                    % After number of attempts without response, try again
                    if iAttempt > 30
                        % try sending the message to the arduino
                        fprintf(obj.serPort,inputMessage);                
                        iAttempt = 1;
                        disp('Trouble giving reward, trying again');
                    end
                end
            end
        end
        %% Reset timer for RT to 0
        function resetTimer(obj)
            % check if we are in octave
            isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0; 
            % try sending the message to the arduino in Octave or Matlab
            if isOctave;
                srl_write(obj.serPort,'IS');
            else
                fprintf(obj.serPort,'IS');
            end
            
        end

        
        %% Read information about licks, return side and lick timestamp
        % Return side as 'R' or 'L' and timestamp in seconds
        function [side, timeStamp] = detectLick(obj)
            % there used to be pauses in this part in Ulf's script
            side = ''; % var for returning side
            timeStamp = 0; % var for returning timestamp
            output = ''; % var for storing output of arduino
            
            % check if we are in octave
            isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0; 
            
            if isOctave
                % Scan until 'O' is found, indicating that there is a message sent from arduino
                output = obj.serial_read_line();
                while ~isempty(output) && output ~= 'O'
                    output = obj.serial_read_line();
                end
                % Read in next line
                output = obj.serial_read_line();
                if ~isempty(output)
                    switch output
                    % if output is Y it is a right lick, not after stimulus
                    case 'Y'                       
                        arduinoRunTime = str2double(obj.serial_read_line())*0.001; % in secs
                        timeStamp = arduinoRunTime - obj.arduinoRefTime;
                        side = 'R';
                        % if output is Z it is a left lick, not after stimulus
                    case 'Z'
                        arduinoRunTime = str2double(obj.serial_read_line())*0.001; % in secs
                        timeStamp = arduinoRunTime - obj.arduinoRefTime;
                        side = 'L';
                    end
                end
            else
                %% Check for licks in Matlab
                if obj.serPort.BytesAvailable
                    % scan untill 'O' is found, indicating that there is a
                    % message sent form arduino
                    while ~strcmp(output, 'O') && obj.serPort.BytesAvailable
                        output = fscanf(obj.serPort, '%s');
                    end
                    % Read in the next output byte
                    if obj.serPort.BytesAvailable
                        output = fscanf(obj.serPort, '%s');
                        
                        switch output
                            % If output is X this is the first lick after a stimulus
                            case 'X'
                                if obj.serPort.BytesAvailable
                                    reaction = fscanf(obj.serPort, '%s');
                                end
                                if obj.serPort.BytesAvailable
                                    RT = fscanf(obj.serPort, '%s');
                                end
                                if obj.serPort.BytesAvailable
                                    passfirst = str2double(fscanf(obj.serPort, '%s'));
                                end
                                if obj.serPort.BytesAvailable
                                    wentthrough = str2double(fscanf(obj.serPort, '%s'));
                                end
                                % if output is Y it is a right lick, not after stimulus
                            case 'Y'
                                if obj.serPort.BytesAvailable
                                    arduinoRunTime = str2double(fscanf(obj.serPort, '%s'))*0.001; % in secs
                                    timeStamp = arduinoRunTime - obj.arduinoRefTime;
                                    side = 'R';
                                end
                                % if output is Z it is a left lick, not after stimulus
                            case 'Z'
                                if obj.serPort.BytesAvailable
                                    arduinoRunTime = str2double(fscanf(obj.serPort, '%s'))*0.001; % in secs
                                    timeStamp = arduinoRunTime - obj.arduinoRefTime;
                                    side = 'L';
                                end
                        end
                    end
                end
            end
            
        end
        %% Flush all arduino output on the serial connection
        function flushSerial(obj)
            % check if we are in octave
            isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0; 
            % Flush the serial connection
            if isOctave
                srl_flush(obj.serPort);
            else
                while obj.serPort.BytesAvailable
                    fscanf(obj.serPort, '%s');
                end
            end             
        end
        %% Sync the arduino time with behavioral training script time
        function syncTime(obj,refTime)
           if obj.arduinoRefTime == 0
              % Get the current runtime and convert to seconds
              obj.arduinoRefTime = obj.get('runtime')*0.001;
           else
              tasktime            = toc(refTime);
              arduinotasktime     = obj.get('runtime')*0.001 - obj.arduinoRefTime;
              obj.arduinoRefTime  = obj.arduinoRefTime - (tasktime - arduinotasktime); %Correct arduino time for diff
           end
           
        end
        %% Destructor method
        % Is being called when using delete(obj)
        function delete(obj)
            % Close the serial connection
            fclose(obj.serPort);
            
        end;
        
        %% read a line from the serial port (only necessary in octave) 
        function returnString = serial_read_line(obj)
            returnString = []; % the string that is returned
            responseByte = uint8(0);
            buffer = [];
            while responseByte ~= 10 && ~isempty(responseByte)
                % Check the response from arduino (read 1 byte)
                 responseByte = srl_read(obj.serPort, 1);
                 % Store in buffer
                 if ~isempty(responseByte)
                    buffer(end+1) = responseByte;
                 end
            end           
            % Remove carriage return and newline characters with strsplit trick
            splittedBuffer = strsplit(char(buffer));           
            % Return the string without carriage return or newline
            returnString = splittedBuffer{1};
        end
        
        
    end
end
