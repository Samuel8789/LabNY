%% DBLBUILDGRATINGCHANGEMOVIE 
% script that builds fullscreen gratings and saves these as a mat file 

%make these orientations
vecMakeAngles = 0:1:360;
Par.intScreenHeight_pix = 768;
Par.intScreenWidth_pix = 1366;

%parameters for different grating stimuli
Par.dblSpeed = 1;
Par.intFrameRate = 60;
Par.dblSpatFreq = 0.05; %full cycles (black+white) per retinal degree

%Screen parameters
Par.dblScreenDistance_cm = 26; % cm; measured
Par.dblScreenWidth_cm = 41; % cm; measured
Par.dblScreenHeight_cm = 27; % cm; measured

%Compute in degrees and pixels
dblScreenWidth_deg      = atand((Par.dblScreenWidth_cm / 2) / Par.dblScreenDistance_cm) * 2;
dblScreenHeight_deg     = atand((Par.dblScreenHeight_cm / 2) / Par.dblScreenDistance_cm) * 2;
pixPerDeg               = Par.intScreenHeight_pix / dblScreenHeight_deg; %number of pixels in single retinal degree
degreesPerCycle         = 1/Par.dblSpatFreq; %number of degrees in a single cycle (black-white block)
pixPerCycle             = degreesPerCycle * pixPerDeg; %number of pixels in such a cycle

for dblRotAngle = vecMakeAngles
    
    fullstimgrat = NaN(Par.intScreenHeight_pix,Par.intScreenWidth_pix,Par.intFrameRate);
    for intFrame=1:Par.intFrameRate
        
        dblPhase = (intFrame/Par.intFrameRate)*2*pi;
        
        imSize      = max([Par.intScreenWidth_pix,Par.intScreenHeight_pix]);
        X           = 1:imSize;                     % X is a vector from 1 to imageSize
        X0          = (X / imSize) - .5;            % rescale X -> -.5 to .5
        freq        = imSize/pixPerCycle;           % compute frequency from wavelength
        [Xm,Ym]     = meshgrid(X0, X0);             % 2D matrices
        thetaRad    = (dblRotAngle / 360)*2*pi;     % convert theta (orientation) to radians
        Xt          = Xm * cos(thetaRad);           % compute proportion of Xm for given orientation
        Yt          = Ym * sin(thetaRad);           % compute proportion of Ym for given orientation
        XYt         = Xt + Yt;                      % sum X and Y components
        XYf         = XYt * freq * 2*pi;            % convert to radians and scale by frequency
        thisFrameGrat = sin(XYf + dblPhase);        % make 2D sinewave
        
        thisFrameGrat(thisFrameGrat>0) = 0;   % Convert into squared waves: black
        thisFrameGrat(thisFrameGrat<0) = 255;   % Convert into squared waves: white
        thisFrameGrat = thisFrameGrat(1:Par.intScreenHeight_pix,1:Par.intScreenWidth_pix); % Cut only screen portion of sq grat
        
        fullstimgrat(:,:,intFrame) = thisFrameGrat;
    end
    fullstimgrat = uint8(fullstimgrat); %convert to uint8 and we're done!

    strSaveDir = 'E:\Matlab\dual-lick-training\stimulus\gratingmovies_changedetection\';
    strSaveFilename = sprintf('gratingmat_fullscreen_%dfps_Ori%03d_Speed%d.mat',Par.intFrameRate,dblRotAngle,Par.dblSpeed);

    save(fullfile(strSaveDir,strSaveFilename),'fullstimgrat')
end


