function [structure_sorted,neuron_id,ensemble_id,avg_weights_sorted] = ...
    Sort_Ensemble_Weights(structure,weight_threshold)
% Sort ensemble structure weighted
%
%   [structureSorted,neuronsID,ensemblesID,avgWeights] = 
%       Sort_Ensemble_Weights(structure,weight_threshold)
%
%       default: weight_threshold = 0;
%
% Jesús Pérez-Ortega, Aug 2021
% Modified Oct 2021

if nargin == 1
    weight_threshold = 0;
end

% Get the number of 
[nEnsembles,nNeurons] = size(structure);

% Sort ensembles
avgWeights = mean(structure,2);
[~,ensemble_id] = sort(avgWeights,'descend');

% Structure sorted by ensembles
structure_sorted = structure(ensemble_id,:);
structure_sorted(structure_sorted<weight_threshold) = 0;

% Sort neurons
neuron_id = 1:nNeurons;
for i = nEnsembles:-1:1
   [~,id] = sort(structure_sorted(i,:),'descend');
   structure_sorted = structure_sorted(:,id);
   neuron_id = neuron_id(id); 
end

% Set structure sorted
structure_sorted = structure(ensemble_id,neuron_id);
avg_weights_sorted = avgWeights(ensemble_id);