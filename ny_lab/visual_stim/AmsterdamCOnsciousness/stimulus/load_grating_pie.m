function vecTex = load_grating_pie(Par,ptrWindow,intThisTrial)

vecTex = [];
optimizeForDrawAngle = 0; %optimize for upright

dblScreenWidth_deg      = atand((Par.dblScreenWidth_cm / 2) / Par.dblScreenDistance_cm) * 2; %#ok<NASGU>
dblScreenHeight_deg     = atand((Par.dblScreenHeight_cm / 2) / Par.dblScreenDistance_cm) * 2;
pixPerDeg = Par.intScreenHeight_pix / dblScreenHeight_deg; %number of pixels in single retinal degree
degreesPerCycle = 1/Par.dblSpatFreq; %number of degrees in a single cycle (black-white block)
pixPerCycle = degreesPerCycle * pixPerDeg; %number of pixels in such a cycle

vecTex = zeros(1,Par.intFrameRate);
    fullstimgrat = NaN(Par.intScreenHeight_pix,Par.intScreenWidth_pix,Par.intFrameRate);
    
    for intFrame = 1:Par.intFrameRate
        dblPhase = (intFrame/Par.intFrameRate)*2*pi;
            % Make squared stimulus greater than screen using combinatorial sine
            % waves, then take only screen part.
            
            imSize      = max([Par.intScreenWidth_pix,Par.intScreenHeight_pix]);
            X           = 1:imSize;                     % X is a vector from 1 to imageSize
            X0          = (X / imSize) - .5;            % rescale X -> -.5 to .5
            freq        = imSize/pixPerCycle;           % compute frequency from wavelength
            [Xm,Ym]     = meshgrid(X0, X0);             % 2D matrices
            thetaRad    = (Par.Stim.vecOri(intThisTrial) / 360)*2*pi;     % convert theta (orientation) to radians
            Xt          = Xm * cos(thetaRad);           % compute proportion of Xm for given orientation
            Yt          = Ym * sin(thetaRad);           % compute proportion of Ym for given orientation
            XYt         = Xt + Yt;                      % sum X and Y components
            XYf         = XYt * freq * 2*pi;            % convert to radians and scale by frequency
            thisFrameGrat = sin(XYf + dblPhase);        % make 2D sinewave
            
            thisFrameGrat(thisFrameGrat>0) = 255;   % Convert into squared waves: white
            thisFrameGrat(thisFrameGrat<0) = 0;   % Convert into squared waves: black
            thisFrameGrat = thisFrameGrat(1:Par.intScreenHeight_pix,1:Par.intScreenWidth_pix); % Cut only screen portion of sq grat
            
        fullstimgrat(:,:,intFrame) = thisFrameGrat;
    end

% Take full black/white and set contrast %Range 0-1
fullstimgrat(fullstimgrat > Par.bgInt) = Par.bgInt + Par.Stim.vecContrast(intThisTrial)*256/2-1;
fullstimgrat(fullstimgrat < Par.bgInt) = Par.bgInt - Par.Stim.vecContrast(intThisTrial)*256/2;

%Make mask
maskDim = false(Par.intScreenHeight_pix,Par.intScreenWidth_pix,Par.intFrameRate);
if isfield(Par.Stim,'vecSide')
    if Par.Stim.vecSide(intThisTrial) == 'L'
        maskDim(: , round(0.5*Par.intScreenWidth_pix):end,:) = true;        
    elseif Par.Stim.vecSide(intThisTrial) == 'R'
        maskDim(: , 1:round(0.5*Par.intScreenWidth_pix),:) = true;       
    end
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
