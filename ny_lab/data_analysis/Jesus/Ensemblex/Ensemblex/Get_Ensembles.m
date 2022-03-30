function analysis = Get_Ensembles(raster,options)
% Unsupervised extraction of ensembles given the raster activity.
%
%       analysis = Get_Ensembles(raster,options)
%
%       default:    Significant network options
%                     options.Network.Bin = 1;
%                     options.Network.Iterations = 1000;
%                     options.Network.Alpha = 0.05;
%                     options.Network.NetworkMethod = 'coactivity';
%                     options.Network.ShuffleMethod = 'time_shift';
%                     options.Network.SingleThreshold = true;
% 
%                   Threshold of coactive neurons in a single frame
%                     coactive_neurons_threshold = 2;
% 
%                   Clustering options
%                     options.Clustering.SimilarityMeasure = 'jaccard';
%                     options.Clustering.LinkageMethod = 'ward';
%                     options.Clustering.EvaluationIndex = 'contrast_excluding_one';
%                     options.Clustering.Range = 3:10;
% 
%                   Ensemble significance  options
%                     options.Ensemble.Iterations = 1000;
%                     options.Ensemble.Alpha = 0.05;
%
% By Jesus Perez-Ortega, Oct 2021

% get initial time point
t_initial = tic;

% If no options, set default options
if nargin == 1
    % Significant network options
    options.Network.Bin = 1;
    options.Network.Iterations = 1000;
    options.Network.Alpha = 0.05;
    options.Network.NetworkMethod = 'coactivity';
    options.Network.ShuffleMethod = 'time_shift';
    options.Network.SingleThreshold = true;
    
    % Threshold of coactive neurons in a single timepoint
    coactive_neurons_threshold = 2;
    
    % Clustering
    options.Clustering.SimilarityMeasure = 'jaccard';
    options.Clustering.LinkageMethod = 'ward';
    options.Clustering.EvaluationIndex = 'contrast_excluding_one';
    options.Clustering.Range = 3:10;
    
    % Ensemble significance
    options.Ensemble.Iterations = 1000;
    options.Ensemble.Alpha = 0.05;
end

%% Read options
% Significant network options
bin = options.Network.Bin;
iterations_network = options.Network.Iterations;
alpha_network = options.Network.Alpha;
network_method = options.Network.NetworkMethod;
shuffle_network = options.Network.ShuffleMethod;
single_th = options.Network.SingleThreshold;

% Clustering
similarity_measure = options.Clustering.SimilarityMeasure;
linkage_method = options.Clustering.LinkageMethod;
clustering_index = options.Clustering.EvaluationIndex;
clustering_range = options.Clustering.Range;

% Ensemble significance
iterations_ensemble = options.Ensemble.Iterations;
alpha_ensemble = options.Ensemble.Alpha;
%% Get ensembles
disp('Neuronal ensemble extraction from raster...')

% Get significant network
disp('   Identifying functional network connectivity...')
rng(0); % for repeatable results
network = Get_Significant_Network_From_Raster(raster,bin,iterations_network,...
    alpha_network,network_method,shuffle_network,single_th); 

% Remove noisy spikes based on functional connections
disp('   Filtering non-significant coactivations...')
[raster_filtered,spikes_fraction_removed] = Filter_Raster_By_Network(raster,network);

% Detect coactivations above 2 active neurons
disp(['   Finding coactivation peaks greater than ' ...
    num2str(coactive_neurons_threshold) ' neurons ...'])
vector_id = Find_Peaks(sum(raster_filtered),coactive_neurons_threshold,false);

% Return if no data
if isempty(vector_id)
    warning('    There is not enough coactivity to extract ensembles.')
    analysis = [];
    return
end

% Get neural vectors
disp('   Getting vectors...')
raster_vectors = Get_Peak_Vectors(raster,vector_id,'binary');

% Get similarity
disp('   Getting similarity...')
similarity = Get_Peaks_Similarity(raster_vectors,similarity_measure);
warning off % turn off warning of ward linkage
tree = linkage(squareform(1-similarity,'tovector'),linkage_method);
warning on
% this is the slowest part i commented it
% try
%     figure; [~,~,treeID] = dendrogram(tree,0); close
% catch
%     treeID = 'It was not possible to get the tree ID.';
%     warning(treeID)
% end

% Get recommended number of ensembles
disp(['   Finding optimum number of clusters (based on ' clustering_index ' index)...'])
[n_ensembles,clustering_indices] = Cluster_Test(tree,similarity,clustering_index,...
   'hierarchical',clustering_range);

% Get hierarchical clustering with recommended clusters
disp(['   Extracting ' num2str(n_ensembles) ' ensembles...'])
sequence = cluster(tree,'maxclust',n_ensembles);

% Get ensemble structure
disp('   Identifying significant neurons during ensemble activation...')
[structure_pre,structure_belongingness,structure_p,ensemble_activity_binary,ensemble_vectors,...
    ensemble_indices] = Get_Ensemble_Neurons(raster,vector_id,sequence);

% Get ensemble networks
disp('   Getting ensemble networks...')
all_ensemble_networks = zeros(size(network));
for i = 1:n_ensembles
    % Get the ensemble neurons
    neurons_i = find(structure_pre(i,:));
    [~,id] = sort(structure_belongingness(i,neurons_i),'descend');
    ensemble_neurons{i} = neurons_i(id);
    
    % Get the connections from the ensemble neurons
    network_i = zeros(size(network));
    network_i(neurons_i,neurons_i) = network(neurons_i,neurons_i);
    ensemble_networks{i} = network_i;
    all_ensemble_networks = all_ensemble_networks|ensemble_networks{i};
end  

% Get final ensemble neurons (this step removes neurons that were significantly active during 
% ensemble activation but were not significantly connected)
disp('   Setting functional connected neurons to ensembles...')
for i = 1:n_ensembles
    structure(i,:) = sum(ensemble_networks{i})>0;    
end  

% Get ensemble activity
disp('   Getting ensemble activity...')
[neurons,frames] = size(raster); 
n_ensembles = length(ensemble_networks);
ensemble_activity = zeros(n_ensembles,frames);
for i = 1:n_ensembles
    % Get activity weights
    id = ensemble_indices{i};
    jaccard = 1-pdist2(double(structure(i,:)>0),double(ensemble_vectors{i}'),'Jaccard');
    ensemble_activity(i,id) = jaccard;
    
    % Get structure weights
    structure_weights(i,:) = mean(ensemble_vectors{i},2);
end
structure_weights_significant = structure_weights.*(structure>0);

% Number of ensemble activation and duration
disp('   Getting ensemble durations...')
[widths,peaks_count] = Get_Ensembles_Length(vector_id,sequence);

% Evaluate similarity within ensemble vectors
disp('   Identifying significant ensembles...')
% Get similarity within rasters
[within,vector_count] = Similarity_Within_Rasters(ensemble_vectors);
ensemble_p = Test_Ensemble_Similarity(similarity,within,peaks_count,iterations_ensemble);
h_ensemble = ensemble_p<alpha_ensemble;

% Get IDs of ensembles and non-ensembles
id_ensemble = find(h_ensemble);
id_nonensemble = find(~h_ensemble);

% Get number of significant and non significant ensembles
n_ensembles = length(id_ensemble);
n_nonensembles = length(id_nonensemble);
if n_nonensembles
    % Get new sequence (only significant ensembles)
    sequence_new = zeros(size(sequence));
    for i = 1:n_ensembles
        sequence_new(sequence==id_ensemble(i)) = i;
    end
    sequence = sequence_new;
    
    % Non ensembles properties
    nonensemble_activity = ensemble_activity(id_nonensemble,:);
    nonensemble_activity_binary = ensemble_activity_binary(id_nonensemble,:);
    nonensemble_networks = ensemble_networks(id_nonensemble);
    nonensemble_vectors = ensemble_vectors(id_nonensemble);
    nonensemble_indices = ensemble_indices(id_nonensemble);
    nonensemble_within = within(id_nonensemble);
    nonensemble_vector_count = vector_count(id_nonensemble);
    nonensemble_structure = structure(id_nonensemble,:);
    nonensemble_structure_belongingness = structure_belongingness(id_nonensemble,:);
    nonensemble_structure_p = structure_p(id_nonensemble,:);
    nonensemble_structure_weights = structure_weights(id_nonensemble,:);
    nonensemble_structure_weights_significant = structure_weights_significant(id_nonensemble,:);
    nonensemble_neurons = ensemble_neurons(id_nonensemble);
    nonensemble_widths = widths(id_nonensemble);
    nonensemble_peaks_count = peaks_count(id_nonensemble);
    nonensemble_p = ensemble_p(id_nonensemble);
    
    % Ensembles 
    ensemble_activity = ensemble_activity(id_ensemble,:);
    ensemble_activity_binary = ensemble_activity_binary(id_ensemble,:);
    ensemble_networks = ensemble_networks(id_ensemble);
    ensemble_vectors = ensemble_vectors(id_ensemble);
    ensemble_indices = ensemble_indices(id_ensemble);
    within = within(id_ensemble);
    vector_count = vector_count(id_ensemble);
    structure = structure(id_ensemble,:);
    structure_belongingness = structure_belongingness(id_ensemble,:);
    structure_p = structure_p(id_ensemble,:);
    structure_weights = structure_weights(id_ensemble,:);
    structure_weights_significant = structure_weights_significant(id_ensemble,:);
    ensemble_neurons = ensemble_neurons(id_ensemble);
    widths = widths(id_ensemble);
    peaks_count = peaks_count(id_ensemble);
    ensemble_p = ensemble_p(id_ensemble);
end
disp(['      ' num2str(n_ensembles) ' significant ensembles.'])
disp(['      ' num2str(n_nonensembles) ' non significant ensembles.'])

% Sort ensembles
disp('   Sorting ensembles from high to low participation...')
[structure_weights_sorted,neuron_id,ensemble_id_sorted,ensemble_avg_weights] =...
    Sort_Ensemble_Weights(structure_weights_significant);
ensemble_activity = ensemble_activity(ensemble_id_sorted,:);
ensemble_activity_binary = ensemble_activity_binary(ensemble_id_sorted,:);
ensemble_networks = ensemble_networks(ensemble_id_sorted);
ensemble_vectors = ensemble_vectors(ensemble_id_sorted);
ensemble_indices = ensemble_indices(ensemble_id_sorted);
within = within(ensemble_id_sorted);
vector_count = vector_count(ensemble_id_sorted);
structure = structure(ensemble_id_sorted,:);
structure_belongingness = structure_belongingness(ensemble_id_sorted,:);
structure_p = structure_p(ensemble_id_sorted,:);    
structure_weights = structure_weights(ensemble_id_sorted,:);
structure_weights_significant = structure_weights_significant(ensemble_id_sorted,:);
ensemble_neurons = ensemble_neurons(ensemble_id_sorted);
widths = widths(ensemble_id_sorted);
peaks_count = peaks_count(ensemble_id_sorted);
ensemble_p = ensemble_p(ensemble_id_sorted);

% Get id of vectors sorted
vectors_id = [];
for i = 1:n_ensembles
        vectors_id = [vectors_id; ensemble_indices{i}];
end
for i = 1:n_nonensembles
    vectors_id = [vectors_id; nonensemble_indices{i}];
end

% Add to analysis structure
disp('   Adding results to ''analysis'' variable output...')
analysis.Options = options;

analysis.Raster = raster;
analysis.Neurons = neurons;
analysis.Frames = frames;
analysis.Network = network;

analysis.Filter.RasterFiltered = raster_filtered;
analysis.Filter.SpikesFractionRemoved = spikes_fraction_removed;
analysis.Filter.RasterVectors = raster_vectors;
analysis.Filter.VectorID = vector_id;

analysis.Clustering.Similarity = similarity;
analysis.Clustering.Tree = tree;
analysis.Clustering.RecommendedClusters = n_ensembles;
analysis.Clustering.ClusteringIndex = clustering_index;
analysis.Clustering.ClusteringRange = clustering_range;
analysis.Clustering.ClusteringIndices = clustering_indices;
% analysis.Clustering.TreeID = treeID;

analysis.Ensembles.Count = n_ensembles;
analysis.Ensembles.ActivationSequence = sequence;
analysis.Ensembles.Activity = ensemble_activity;
analysis.Ensembles.ActivityBinary = ensemble_activity_binary;
analysis.Ensembles.Networks = ensemble_networks;
analysis.Ensembles.AllEnsembleNetwork = all_ensemble_networks;
analysis.Ensembles.Vectors = ensemble_vectors;
analysis.Ensembles.Indices = ensemble_indices;
analysis.Ensembles.Similarity = within;
analysis.Ensembles.VectorCount = vector_count;
analysis.Ensembles.Structure = structure;
analysis.Ensembles.StructureBelongingness = structure_belongingness;
analysis.Ensembles.StructureP = structure_p;
analysis.Ensembles.StructureWeights = structure_weights;
analysis.Ensembles.StructureWeightsSignificant = structure_weights_significant;
analysis.Ensembles.StructureSorted = structure_weights_sorted;
analysis.Ensembles.Weights = ensemble_avg_weights;
analysis.Ensembles.EnsembleNeurons = ensemble_neurons;
analysis.Ensembles.NeuronID = neuron_id;
analysis.Ensembles.VectorID = vectors_id;
analysis.Ensembles.Durations = widths;
analysis.Ensembles.PeaksCount = peaks_count;
analysis.Ensembles.Probability = ensemble_p;
analysis.Ensembles.AlphaEnsemble = alpha_ensemble;

if n_nonensembles
    analysis.NonEnsembles.Count = n_nonensembles;
    analysis.NonEnsembles.Activity = nonensemble_activity;
    analysis.NonEnsembles.ActivityBinary = nonensemble_activity_binary;
    analysis.NonEnsembles.Networks = nonensemble_networks;
    analysis.NonEnsembles.Vectors = nonensemble_vectors;
    analysis.NonEnsembles.Indices = nonensemble_indices;
    analysis.NonEnsembles.Similarity = nonensemble_within;
    analysis.NonEnsembles.VectorCount = nonensemble_vector_count;
    analysis.NonEnsembles.Structure = nonensemble_structure;
    analysis.NonEnsembles.StructureBelongingness = nonensemble_structure_belongingness;
    analysis.NonEnsembles.StructureP = nonensemble_structure_p;
    analysis.NonEnsembles.StructureWeights = nonensemble_structure_weights;
    analysis.NonEnsembles.StructureWeightsSignificant = nonensemble_structure_weights_significant;
    analysis.NonEnsembles.EnsembleNeurons = nonensemble_neurons;
    analysis.NonEnsembles.Durations = nonensemble_widths;
    analysis.NonEnsembles.PeaksCount = nonensemble_peaks_count;
    analysis.NonEnsembles.Probability = nonensemble_p;
end

% Display the total time
t_final = toc(t_initial);
disp(['You are all set! (total time: ' num2str(t_final) ' seconds)'])