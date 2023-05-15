classdef ServoController < handle
    %ServoController Communicate with a servo
    % Objects of class ServoController allow for communication with the
    % servo
    %
    %
    % Reinder Dorman okt 2017 UvA
    % Servo controller going via arduino now, because the parallax thing is broken.
    % also arduino has some homemade code which makes it really easy.
    properties
        % Properties for serial connection
        inputBufferSize = 10240; %10240;
        baudRate = 9600; % default for servo
        timeOut = 0.001;
        serialStr = '/dev/ttyS888';
        serPort; % serial port object
        
        % Properties for servo movement
        closePos = 0; % the angle for close position
        farPos = 45 ; % the angle for far position 
        offPos = 90 ;% this is the default postion
                    % the servo will be in when it connects serially
                    % so this is the far position. Otherwise, the servo
                    % will go to this position when initilized, and it
                    % can scare the mouse :( 
        
    end
    
    methods
        %% Constructor
        function obj = ServoController(varargin)
            % If there are arguments set properties
            if nargin > 0
                % Switch the arguments and set properties accordingly
                for i=1:2:size(varargin,2)
                    if any(strcmp(varargin{i},obj.properties))
                        obj.(varargin{i}) = varargin{i+1};
                    end
                end
            end
            % Initiate the serial port object
            obj.serPort = serial(obj.serialStr);
            
            % Set some parameters for serial connection
            set(obj.serPort, 'baudrate', obj.baudRate); % baudrate needs to be the same as arduino
            set(obj.serPort, 'timeout', obj.timeOut);
%            set(obj.serPort, 'Terminator', 'cr'); % terminator is carriage return does not work in octave
            
            % Open serial connection if in Matlab and not octave
            isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0; % check if we are in octave
            if ~isOctave
              if strcmp(get(obj.serPort, 'status'), 'closed'); fopen(obj.serPort); end;
            end
            pause(0.05);
             
            
        end
        %% Move servo using serial port command
        function moveServo(obj,argin)
            % if input argument is a character
            if ischar(argin)
                % if it is a C move servo to close position
                if argin == 'C'
                    obj.moveServo(obj.closePos);
                % if it is a F move servo to far position
                elseif argin == 'F'
                    obj.moveServo(obj.farPos); 
                % if you turn the device off move it back to default position, 
                % so it is in the initialization position
                elseif argin == 'off'
                    obj.moveServo(obj.offPos);
                end
            % if input argument is numeric this is pulseWidth which defines
            % the new position of the servo
            elseif isnumeric(argin)
                % Get pulseWidth in bytes    
                % Write on the serial connection
                % First 3 bytes are "!SC", the preamble. 13 is carriage return      
                isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0; % check if we are in octave
                if isOctave
                    writeS = srl_write(obj.serPort,uint8(argin));
                else
                    fwrite(obj.serPort, argin, 'uint8', 'sync');
                end
                
            data = [];
            while isempty(data); 
              % wait for a reply of the servo that it´s done turning
              data = srl_read(obj.serPort,1);
            end
          
            end
        end
                
        %% Destructor method
        % Is being called when using delete(obj)
        function delete(obj)
            % Close the serial connection
            obj.moveServo('off');
            fclose(obj.serPort);
        end
        
    end
end