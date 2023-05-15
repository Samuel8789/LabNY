function [vecTex] = buildGratingChange(Par,ptrWindow,dblRotAngle,contrast)

optimizeForDrawAngle = 0; %optimize for upright

dblScreenWidth_deg      = atand((Par.dblScreenWidth_cm / 2) / Par.dblScreenDistance_cm) * 2; %#ok<NASGU>
dblScreenHeight_deg     = atand((Par.dblScreenHeight_cm / 2) / Par.dblScreenDistance_cm) * 2;
pixPerDeg               = Par.intScreenHeight_pix / dblScreenHeight_deg; %number of pixels in single retinal degree
degreesPerCycle         = 1/Par.dblSpatFreq; %number of degrees in a single cycle (black-white block)
pixPerCycle             = degreesPerCycle * pixPerDeg; %number of pixels in such a cycle

vecTex                  = zeros(1,Par.intFrameRate);

if Par.GratingMethod == 1 || Par.GratingMethod == 2
    fullstimgrat = NaN(Par.intScreenHeight_pix,Par.intScreenWidth_pix,Par.intFrameRate);
    
    for intFrame = 1:Par.intFrameRate
        dblPhase = (intFrame/Par.intFrameRate)*2*pi;
        
        if Par.GratingMethod == 1 %Jorrit old method, only 0 degree mov grating
            thisFrameGrat = NaN(Par.intScreenHeight_pix,Par.intScreenWidth_pix);
            
            %create a grid with the size of the required image
            [grid]  = meshgrid(1:Par.intScreenWidth_pix,1:Par.intScreenHeight_pix);
            
            % Build the grating at specified contrast as perc of max
            % Every pixPerCycle pixels the grid flips back to 0 with an offset of dblPhase
            modmat = mod(grid-1+round(pixPerCycle*(dblPhase/(2*pi))),pixPerCycle);
            % Create logicals and build white grating
            thisFrameGrat(modmat > pixPerCycle/2) = Par.bgInt + Par.dblGratingContrast*255/2;
            % Set black gratings to intensity
            thisFrameGrat(modmat < pixPerCycle/2) = Par.bgInt - Par.dblGratingContrast*255/2;
            %             thisFrameGrat = (modmat > pixPerCycle/2)*255; %old jorrit
            
        elseif Par.GratingMethod == 2 %Alternative method that is able to generate angles:
            % Make squared stimulus greater than screen using combinatorial sine
            % waves, then take only screen part.
            
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
            
            thisFrameGrat(thisFrameGrat>0) = 255;   % Convert into squared waves: white
            thisFrameGrat(thisFrameGrat<0) = 0;   % Convert into squared waves: black
            thisFrameGrat = thisFrameGrat(1:Par.intScreenHeight_pix,1:Par.intScreenWidth_pix); % Cut only screen portion of sq grat
            
        end
        fullstimgrat(:,:,intFrame) = thisFrameGrat;
    end
elseif Par.GratingMethod == 3
    strLoadDir = fullfile(Par.pcsettings.mainFolder,'stimulus','gratingmovies_changedetection');
    strLoadFilename = sprintf('gratingmat_fullscreen_%dfps_Ori%03d_Speed%d.mat',Par.intFrameRate,dblRotAngle,Par.dblSpeed);
    load(fullfile(strLoadDir,strLoadFilename),'fullstimgrat');
end

% Take full black/white and set contrast %Range 0-1
fullstimgrat(fullstimgrat > Par.bgInt) = Par.bgInt + contrast*256/2-1;
fullstimgrat(fullstimgrat < Par.bgInt) = Par.bgInt - contrast*256/2;

% Make a mask if required:
maskDim = false(Par.intScreenHeight_pix,Par.intScreenWidth_pix,Par.intFrameRate);
if Par.Stim.vecVisualSide(1:Par.intTrialNum) == 'L'
    maskDim(: , round((Par.StimSpanSides/2+0.5)*Par.intScreenWidth_pix):end,:) = true;
elseif Par.Stim.vecVisualSide(1:Par.intTrialNum) == 'R'
    maskDim(: , 1:round((Par.StimSpanSides/2+0.5)*Par.intScreenWidth_pix),:) = true;
end
fullstimgrat(maskDim) = Par.bgInt; %apply mask

if Par.boolUseSquareTex == 1
    specialFlags = 1; %put into square opengl texture
else
    specialFlags = 0; %use normal texture
end

%Make the textures with the stimuli and store pointers in vecTex
for intFrame = 1:Par.intFrameRate
    vecTex(intFrame) = Screen('MakeTexture', ptrWindow, fullstimgrat(:,:,intFrame), optimizeForDrawAngle, specialFlags);
end

end