classdef PiezoController < handle
    %PiezoController communicate with an arduino piezo controller.
    % Objects of class PiezoController allow for communication with the
    % servo
    properties
        % Default properties for serial connection
        baudRate    = 115200;
        timeOut     = 0.001;
        serialStr   = '/dev/ttyS888'; 
        serPort;    % serial port object
        box = 5;
        arduinoRefTime = 0; 
        
    end
    
    methods
        %% Constructor
        function obj = PiezoController(varargin)
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
            
            
        end
        
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
     %% Function for moving piezo        
        function movePiezo(obj,side,intensity)
            % check if we are in octave
            isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
            
            inputMessage = ['IP' side num2str(intensity) 'f'];
            
            srl_write(obj.serPort,inputMessage);                       
            pause(0.001);
            response=srl_read(obj.serPort,3);
            iii=1;
            while isempty(response) && iii<5
            pause(0.001);
            response=srl_read(obj.serPort,3);
            end
            if response(1)==68
            % Print to command window that it worked
            fprintf('Delivered %s %d tactile stim\n',side,intensity);
            else 
            disp('no correct response from piezo idk if stim or not')
            end 
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        
        %% Destructor method
        % Is being called when using delete(obj)
        function delete(obj)
            % Close the serial connection

            fclose(obj.serPort);

            
        end
        
    end
end