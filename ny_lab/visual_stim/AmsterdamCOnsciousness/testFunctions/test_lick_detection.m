function test_lick_detection(ldObj)
figure;
ldObj.flushSerial();
nSamples = 100;
outputArray = zeros(1,nSamples);
while true
    for i = 1:nSamples
        output = ldObj.serial_read_line();
        if ~isempty(output)
            outputArray(i) = str2double(output);
        else
            outputArray(i) = -1;
        end
    end
    outputArray(outputArray == -1) = [];
    plot(outputArray);
    drawnow
end
