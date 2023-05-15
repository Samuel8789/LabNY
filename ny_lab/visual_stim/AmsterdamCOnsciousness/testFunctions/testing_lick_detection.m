trialNum = 10; % number of trials


startTime = tic;
% Get arduino time
arduinoTime = ldObj.get('runtime');
% How long did that take?
display(toc(startTime));


% % Start the arduino trial-based timer
% ldObj.resetTimer;

% for iTrial = 1:trialNum
%     [left, right] = ldObj.detectLick;
%     if left > 0
%         fprintf('left: %f\n',left);
%     end
%     if right > 0
%         fprintf('right: %f\n',right);
%     end
%    
% end
% 
while toc(startTime) < 10
    [side, timeStamp] = ldObj.detectLick;
    if strcmp(side, 'left')
        fprintf('time is: %d\n',timeStamp - arduinoTime*0.001);
    elseif strcmp(side, 'right')
        fprintf('time is: %d\n', timeStamp - arduinoTime*0.001);
    end
    
    
    
end