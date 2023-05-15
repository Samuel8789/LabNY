%% Load the visual stimulus texture
% Return vecTex which is a vector of textures that ar presented in a loop
% to create a movie in the stimulus presentation script
function vecTex = load_texture(Par,ptrWindow,intThisTrial)
%retrieve info for current trial

optimizeForDrawAngle = 0; %optimize for upright


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
        
elseif  strcmp(Par.strVisStimType, 'grating')

  vecTex = []; %make textures for the required orientations
    if Par.Stim.vecOriPreChange(intThisTrial) ~= 999
        vecTex = buildGratingChange(Par,ptrWindow,Par.Stim.vecOri(intThisTrial),Par.dblGratingContrast);
    end
else
    
    
%load textures stimulus for moving grating stimulus
    vecTex = [];
    dblRotAngle = Par.Stim.vecOri(intThisTrial);
    if dblRotAngle ~= 999
    
        strGratingFile = sprintf('gratingmovie_SQUARE_%dfps_Ori%03d_Speed1.mat',60,dblRotAngle);
        sLoad = load(strGratingFile);
        matGrating = sLoad.matGrating;
        clear sLoad;
        tLength = size(matGrating,3);
        
        if Par.boolUseSquareTex == 1
           specialFlags = 1; %put into square opengl texture
       else%d
            specialFlags = 0; %use normal texture
        end
        vecTex = zeros(1,tLength);
        for intFrame=1:tLength
            thisFrame = matGrating(:,:,intFrame);
            vecTex(intFrame)=Screen('MakeTexture', ptrWindow, thisFrame, optimizeForDrawAngle, specialFlags);
            clear thisFrame;
       end
        clear matGrating;
        

        
    end
    
        
end



end