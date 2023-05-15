function check_textures(Par)

%% CHECK TEXTURES
% Check whether the textures can be loaded succesfuly
fprintf('\nChecking textures...');
for intOri = Par.vecOrientations
    strGratingFile = sprintf('gratingmovie_SQUARE_%dfps_Ori%03d_Speed1.mat',60,intOri);
    sLoad = load(strGratingFile);
    matGrating = sLoad.matGrating;
    clear sLoad;
    if size(matGrating,3) ~= Par.intFrameRate
        error([mfilename ':TextureError'],'Grating file %s is corrupt or not loaded succesfully. Please check the file',strGratingFile);
    end
    clear matGrating;
end
fprintf('   Done\n');