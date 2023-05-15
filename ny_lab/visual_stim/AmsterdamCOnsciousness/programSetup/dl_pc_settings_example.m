## Author: MouseRig <mouserig@mouserig-OptiPlex-755>
## Created: 2017-03-16
% This function returns a struct with the settings for this pc. Make sure this 
% function is located in your path, but do not push it to git as it should be 
% different on each pc.
function PcSettings = dl_pc_settings ()
    PcSettings = struct;
    %Main folder with all training related files under GIT control:
    PcSettings.mainFolder = '~/Documents/OCTAVE/dual-lick-training';
    % Folder at which trainingscripts are located
    PcSettings.protocolFolder = fullfile(PcSettings.mainFolder,'trainingScripts');
    % Folder at which parameter files are located (this should be in the path)
    PcSettings.parametersFolder = fullfile(PcSettings.mainFolder,'parameterFiles');
    % DropBox Folder:
    PcSettings.dropboxFolder = '~/Dropbox';
    % Serial Port with LickDetector
    PcSettings.ldserialStr = '/dev/ttyS999';
    % Serial Port with PiezoController
    PcSettings.pzserialStr = '/dev/ttyS888';
end
