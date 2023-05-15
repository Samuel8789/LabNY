function [Par] = create_trialtypes_pie(Par)
Par.Stim            = struct;

%Session structure
if Par.UseTrialBlocks    
  if size(Par.BlockLength,2)==2       
  block1(1:Par.BlockLength(1)) = 'L';
  block2(1:Par.BlockLength(2)) = 'R';
  else 
  block1(1:Par.BlockLength(1)) = 'L';
  block2(1:Par.BlockLength(1)) = 'R';
  end
  block = [block1 block2];
  Par.Stim.vecSide = repmat(block,1,ceil(Par.intTrialNum/length(block))); %Could use floor+mod, but it also works to overshoot.

else 
SubBlockSize=10; %the trials will be built chaining blocks of 10 so that maximum 5 L/R are aligned (not really but helps)
Par.Stim.vecSide = []; SB=[];
for sb=1:ceil(Par.intTrialNum/SubBlockSize)
SB(1:SubBlockSize)= 'R'; SB(randperm(SubBlockSize,round(SubBlockSize*Par.LeftTrialsFreq)))= 'L';
Par.Stim.vecSide = [Par.Stim.vecSide SB];
end
end

%Remark: Trial-type modality is independent of correct side. I could en up with
%very different conditions so maybe better to make then together.

%Make Trial-Types:
%T:tactile V:visual M:multimodal
%C:catch
SubBlockSize=15; 
Par.Stim.vecTrialType = []; SB=[];
for sb=1:ceil(Par.intTrialNum/SubBlockSize)
 SB(1:SubBlockSize)= 'V'; SB(randperm(SubBlockSize,round(SubBlockSize*Par.MultimodalFreq)))= 'M';
 Unim = (1:SubBlockSize)(SB=='V');
 SB(Unim(randperm(length(Unim),round(length(Unim)*Par.TactileTrialsFreq))))='T';
 SB(randperm(SubBlockSize,floor(SubBlockSize*Par.CatchTrialsFreq)))='C';
 Par.Stim.vecTrialType = [Par.Stim.vecTrialType SB];
end          
%For catch trials there is no correct side.
Par.Stim.vecSide(Par.Stim.vecTrialType=='C')='N'; %N for none

%At the moment, the stimulus will be displayed on the correct side (congruence).
Par.Stim.leftCorrect = zeros(1,Par.intTrialNum);
Par.Stim.rightCorrect = zeros(1,Par.intTrialNum);
Par.Stim.leftCorrect(Par.Stim.vecSide == 'L') = 1; %Trial types ¨L¨ are to be licked to the left. I can add more valid trial types.
Par.Stim.rightCorrect(Par.Stim.vecSide == 'R') = 1; %Trial types ¨R¨ are to be licked to the right.

%Here again the stimuli are independent. 

%Used somewhere not to make it bug.
Par.Stim.vecVisualSide(1:Par.intTrialNum)        = 'A'; %All screen, both sides
%Ori : Generates an Ori for each trial, randomly drawn from vecOrientations.
Par.Stim.vecOri = Par.vecOrientations(randi(length(Par.vecOrientations),Par.intTrialNum,1)); 
Par.Stim.vecOri(Par.Stim.vecTrialType=='C')= nan;
Par.Stim.vecOri(Par.Stim.vecTrialType=='T')= nan;
%Contrast
Par.Stim.vecContrast = Par.dblGratingContrast(randi(length(Par.dblGratingContrast),Par.intTrialNum,1));
Par.Stim.vecContrast(Par.Stim.vecTrialType=='C') = 0; 
Par.Stim.vecContrast(Par.Stim.vecTrialType=='T') = 0; 
%Piezo deflections
Par.Stim.vecDeflection = Par.DeflectionIntensities(randi(length(Par.DeflectionIntensities),Par.intTrialNum,1));
Par.Stim.vecDeflection(Par.Stim.vecTrialType=='C') = 0;
Par.Stim.vecDeflection(Par.Stim.vecTrialType=='V') = 0;
end