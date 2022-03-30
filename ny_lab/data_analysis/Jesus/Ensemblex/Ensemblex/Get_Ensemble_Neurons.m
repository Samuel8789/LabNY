function [structure,structure_belongingness,structure_p,ensemble_activity_binary,vectors,indices] = ...
    Get_Ensemble_Neurons(raster,vectorID,sequence)
% Get a network from each ensemble
%
%       [structure,structure_belongingness,structure_p,ensemble_activity_binary,vectors,indices] = ...
%   Get_Ensemble_Neurons(raster,vectorID,sequence)
%
% By Jesus Perez-Ortega, Oct 2021

% Get number of ensembles
ensembles = length(unique(sequence));

% Get ensemble network
for i = 1:ensembles
    
    % Get raster ensemble
    peaks = find(sequence==i);
    peak_indices = [];
    for j = 1:length(peaks)
        peak_indices = [peak_indices; find(vectorID==peaks(j))];
    end
    vectors{i} = raster(:,peak_indices);
    indices{i} = peak_indices;
    
    % Get ensemble activity
    activity = false(1,size(raster,2));
    activity(peak_indices) = true; 
    ensemble_activity_binary(i,:) = activity;
    
    % Detect neurons significantly active with the ensemble
    [h1,belongingness,~,p] = Get_Evoked_Neurons(raster,activity);
    p(h1) = 1;

    structure(i,:) = h1;
    structure_belongingness(i,:) = belongingness;
    structure_p(i,:) = p;
end



