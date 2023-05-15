datpath='C:\Users\sp3660\Documents\20230407\Mixed\Analog_2.dat'
dataType='uint16'
fID = fopen(datpath);
hSize = fread(fID,1,'double'); %header size
header = fread(fID,hSize,'double') %Metadata. Default is: 1 = Time of Acquisition onset, 2 = Number of channels, 3 = number of values per channel
data = fread(fID,[header(end-1),header(end)],[dataType '=>' dataType]); %get data. Last 2 header values should contain the size of the data array.



figure
hold on
for i=3:5
    plot(rescale(data(i,:),0,1))
end
legend
ylim([0 1.1])