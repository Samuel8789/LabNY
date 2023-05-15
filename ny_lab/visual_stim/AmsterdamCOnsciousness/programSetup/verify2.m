function [Object999, Object888]=verify2
%First S999
SerPort999 = serial('/dev/ttyS999',115200,10); %Load object in ttyS999
pause(1);
srl_flush(SerPort999);
R=[]; i=1;

while isempty(R) && i<4
srl_write(SerPort999,'IV'); %send verify query
pause(0.5);
R=srl_read(SerPort999,3);
i=i+1;
if isempty(R) 
disp(['No response'])
else
R=R(1);
if R==76 %if lickdetector on 999. 80 is P:piezo, 76 is L:lickdetector
  disp(['Lick Detector on tty999'])
  Object999='Lick Detector';
  elseif R==80 %if piezo
  disp('BAD: Piezo Controller on tty999')
  Object999='Piezo Controller';
  else 
  disp('Unknown object on tty999')
  Object999='Unknown object';
end
end
end

%Then S888
SerPort888 = serial('/dev/ttyS888',115200,10); %Load object in ttyS999
pause(1);
srl_flush(SerPort888);
R=[]; i=1;

while isempty(R) && i<4
srl_write(SerPort888,'IV'); %send verify query
pause(0.5);
R=srl_read(SerPort888,3);
i=i+1;
if isempty(R) 
disp(['No response'])
else
R=R(1);
if R==80 %if piezo on 888. 80 is P:piezo, 76 is L:lickdetector
  disp(['Piezo Controller on tty888'])
  Object888='Piezo Controller';
  elseif R==76 %if piezo
  disp('BAD: Lick Detector on tty888')
  Object888='Lick Detector';
  else 
  disp('Unknown object on tty888')
  Object888='Unknown object';
end
end
end

%Clean 
srl_flush(SerPort999);
fclose(SerPort999);
srl_flush(SerPort888);
fclose(SerPort888);
end