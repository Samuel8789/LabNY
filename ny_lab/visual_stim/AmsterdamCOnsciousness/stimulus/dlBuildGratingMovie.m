%DBLBUILDGRATINGMOVIE script that builds a grating movie and saves it in a
%mast file

%make these orientations
vecMakeAngles = [0 120 240];

%parameters for different movies
dblSpeed = 1;
frameRate = 60;

%screen params
intUseScreen = max(Screen('Screens')); %which screen to use
debug = 0;
verbose = 1;
boolForce = 0;

dblScreenDistance_cm = 16; % cm; measured
dblScreenWidth_cm = 34; % cm; measured
dblScreenHeight_cm = 27; % cm; measured

dblScreenWidth_deg = atand((dblScreenWidth_cm / 2) / dblScreenDistance_cm) * 2;
dblScreenHeight_deg = atand((dblScreenHeight_cm / 2) / dblScreenDistance_cm) * 2;

amplitude = [0 0.5]; %amplitude of intensity oscillation around background (dbl, [0 0.5])
bgIntStim = 0.5; %background intensity (dbl, [0 1])
bgInt = round(bgIntStim*255);
	
%open window
% sven changed this part a bit
structScreen.BackColor = bgInt;
structScreen.UseScreen = intUseScreen;
windowRect = [0 0 1024 1024];
[structScreen.window,structScreen.windowRect] = Screen('OpenWindow',intUseScreen,bgInt,windowRect);
% structScreen = Initialize(structScreen,debug); 

ptrWindow = structScreen.window;
intScreenWidth_pix = structScreen.windowRect(3);
intScreenHeight_pix = structScreen.windowRect(4);

for dblOrientation = vecMakeAngles
	
	%stimulus params
	dblStimSizeRetinalDegrees = 50; % retinal degrees
	dblSpatialFrequency = 0.05; %full cycles (black+white) per retinal degree
	dblTotalCycles = dblStimSizeRetinalDegrees * dblSpatialFrequency; %number of full cycles per stimulus
	
	%movie variables
	matMovie = zeros(intScreenHeight_pix,intScreenWidth_pix,3,frameRate,'uint8');
	movieFile = sprintf('gratingmovie_%dfps_Ori%03d_Speed%d.avi',round(frameRate),dblOrientation,round(dblSpeed));
	matFile = sprintf('gratingmovie_%dfps_Ori%03d_Speed%d.mat',round(frameRate),dblOrientation,round(dblSpeed));
	matFileSquare = sprintf('gratingmovie_SQUARE_%dfps_Ori%03d_Speed%d.mat',round(frameRate),dblOrientation,round(dblSpeed));
	width = [];
	height = [];
	movieOptions = ':CodecType=xvidenc EncodingQuality=1.0 :::';
%	moviePtr = Screen('CreateMovie', ptrWindow, movieFile, width, height, frameRate, movieOptions);
	
	%frame variables
	rect = [];
	bufferName = [];
	frameduration = 1;
	
	intFrames = frameRate;
	for intFrame=1:intFrames
		dblPhase = (intFrame/intFrames)*2*pi;
		matStim = buildGrating(dblSpatialFrequency,dblStimSizeRetinalDegrees,[intScreenWidth_pix intScreenHeight_pix],[dblScreenWidth_deg dblScreenHeight_deg],dblPhase);
		stimTex = Screen('MakeTexture', ptrWindow, matStim);
		Screen('DrawTexture',ptrWindow,stimTex,[],[],dblOrientation);
		Screen('Flip',ptrWindow);
		%Screen('AddFrameToMovie', ptrWindow,rect,bufferName,moviePtr,frameduration)
		thisImage = Screen('GetImage', ptrWindow);
		matMovie(:,:,:,intFrame)=thisImage;
		Screen('Close',stimTex);
	end
	%Screen('FinalizeMovie',moviePtr);
	
	%put into square
	squareSize = 1024;
	ySize = size(matMovie,1);
	xSize = size(matMovie,2);
	intChannel = 1;
	
	yDiff = ySize - squareSize;
	xDiff = xSize - squareSize;
	
	ySelect = ((yDiff/2)+1):(ySize-(yDiff/2));
	xSelect = ((xDiff/2)+1):(xSize-(xDiff/2));
	
	matGrating = zeros(squareSize,squareSize,intFrames,'uint8');
	for intFrame=1:intFrames
		thisFrame = squeeze(matMovie(ySelect,xSelect,intChannel,intFrame));
		matGrating(:,:,intFrame) = thisFrame;
	end
	
	%save
	save(matFileSquare,'matGrating');
	%save(matFile,'matMovie');
	
end
Screen('Close',ptrWindow);
Screen('CloseAll');
%end

