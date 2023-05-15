%% Load the visual stimulus texture
% Return vecTex which is a vector of textures that are presented in a loop
% to create a movie in the stimulus presentation script
function vecTex = load_texture_change(Par,ptrWindow,intThisTrial)

optimizeForDrawAngle = 0; %optimize for upright

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load texture for gabor patch stimulus
if strcmp(Par.strVisStimType, 'gabor')
    if Par.Stim.vecSide(intThisTrial) == 'R'
        vecGaborCenter = Par.vecCoordinatesRight;
    elseif Par.Stim.vecSide(intThisTrial) == 'L'
        vecGaborCenter = Par.vecCoordinatesLeft;
    end
    
    matGabor = gabor('px',500, 'py',500,'theta',Par.Stim.vecOri(intThisTrial),'freq', ...
        Par.dblSpatialFrequency,'sigma',Par.dblSigma, 'width',Par.intScreenWidth_pix, ...
        'height', Par.intScreenHeight_pix, 'px', vecGaborCenter(1)* Par.intScreenWidth_pix,...
        'py', vecGaborCenter(2)* Par.intScreenHeight_pix);
    vecTex = Screen('MakeTexture', ptrWindow, matGabor, optimizeForDrawAngle);
    
elseif strcmp(Par.strVisStimType, 'movingGrating')
    
    vecTex = []; %make textures for the required orientations
    if Par.Stim.vecOri(intThisTrial) ~= 999
        vecTex              = buildGratingChange(Par,ptrWindow,Par.Stim.vecOri(intThisTrial),Par.Stim.vecVisStimInt(intThisTrial));
    end
    
elseif strcmp(Par.strVisStimType,'fs_grating')
    vecTex = []; %make textures for the required orientations
    if Par.Stim.vecOri(intThisTrial) ~= 999
          vecTex = buildGratingPie(Par,ptrWindow,Par.Stim.vecOri(intThisTrial),Par.Stim.vecContrast(intThisTrial),intThisTrial);
    end
    
elseif strcmp(Par.strVisStimType,'movingGratingChange') %load full screen moving grating stimulus w/ change
    
    vecTex = []; %make textures for the required orientations
    if Par.Stim.vecOriPreChange(intThisTrial) ~= 999
        
        if Par.Stim.visualchange(intThisTrial) %If there is a change:
            vecTexPreChange     = buildGratingChange(Par,ptrWindow,Par.Stim.vecOriPreChange(intThisTrial),Par.dblGratingContrast*Par.StimPreChangeIntensity);
            vecTexPostChange    = buildGratingChange(Par,ptrWindow,Par.Stim.vecOriPostChange(intThisTrial),Par.dblGratingContrast);
            vecTex              = [vecTexPreChange vecTexPostChange];
        else
            vecTex              = buildGratingChange(Par,ptrWindow,Par.Stim.vecOriPreChange(intThisTrial),Par.dblGratingContrast*Par.StimPreChangeIntensity);
        end

    end
end

end