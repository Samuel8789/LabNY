%% Loop over reward delivery at left and right side
% Create LickDetector instance
ldObj = LickDetector;

RewDurRight = input("Right side: Enter reward delivery time: ");
RewDurLeft = input("Left side: Enter reward delivery time: ");
nRep = input("Enter the number of repetitions: ");

try
    for i = 1:nRep
        if RewDurRight~=0
        ldObj.giveReward('R',RewDurRight);
        pause(0.2 + RewDurRight/1000)
        end
        if RewDurLeft~=0
        ldObj.giveReward('L',RewDurLeft);
        pause(0.2 + RewDurLeft/1000); 
        end
    end
catch errorMessage
    % delete LickDetector object
    delete(ldObj); clear ldObj
    %Show error
    rethrow(errorMessage);
    
end    

delete(ldObj); clear ldObj
