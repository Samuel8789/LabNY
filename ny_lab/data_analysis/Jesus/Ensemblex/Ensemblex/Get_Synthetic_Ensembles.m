function [raster,raw,events,all_events] = Get_Synthetic_Ensembles(p_ensemble,samples,...
    event_percentage,ensemble_length,noise,jitter_sd,min_separation,seed)
% Generate a synthetic raster with simulated ensemble activity
%
%       [raster,raw,events,all_events] = Get_Synthetic_Ensembles(p_ensemble,samples,...
%    event_percentage,ensemble_length,noise,jitter_sd,min_separation,seed)
%
%       default: p_ensemble = []; samples = 3000; event_percentage = 0.05; ensemble_length = 5;
%                noise = 0.1; jitter_sd = 1; min_separation = 10; seed = [];
%
%                p_ensemble, matrix of probabilities of each neuron to participate on each ensemble
%                            (num ensembles x num neurons)
%                samples, number of samples to simulate 
%                event_percentage, percentage of total ensemble events
%                                  based on the number of samples
%                ensemble_length, number of consecutive samples of the ensemble event
%                noise, probability of adding activity to all neurons
%                jitter_sd, number of the standard deviation of frames jittered in every event
%                min_separation, number of 'refractary' samples before a following ensemble event
%                seed, seed to generate random variables (fix a value for reproducibility)
%
% By Jesus Perez-Ortega, Oct 2021

switch nargin
    case 0
        p_ensemble = [];
        samples = 3000;
        event_percentage = 0.05;
        ensemble_length = 5;
        noise = 0.1;
        jitter_sd = 1;
        min_separation = 10;
        seed = [];
    case 1
        samples = 3000;
        event_percentage = 0.05;
        ensemble_length = 5;
        noise = 0.1;
        jitter_sd = 1;
        min_separation = 10;
        seed = [];
    case 2
        event_percentage = 0.05;
        ensemble_length = 5;
        noise = 0.1;
        jitter_sd = 1;
        min_separation = 10;
        seed = [];
    case 3
        ensemble_length = 5;
        noise = 0.1;
        jitter_sd = 1;
        min_separation = 10;
        seed = [];
    case 4
        noise = 0.1;
        jitter_sd = 1;
        min_separation = 10;
        seed = [];
    case 5
        jitter_sd = 1;
        min_separation = 10;
        seed = [];
    case 6
        min_separation = 10;
        seed = [];
    case 7
        seed = [];
end

% Set random probability of participation in ensemble events
if isempty(p_ensemble)
    % By default 4 ensembles and 80 neurons (20 neurons/ensemble without sharing)
    p_ensemble = zeros(4,80);
    
    % Normal distribution probability of neuron participation
    ps = normrnd(0.5,0.5,[20 1]);
    ps(ps<0) = 0;
    ps(ps>1) = 1;
    
    % Assign probability to each ensemble
    for i = 1:4
        p_ensemble(i,(i-1)*20+1:20*i) = ps;
    end
end

% Total number of neurons and ensembles
[n_ensembles,n_neurons] = size(p_ensemble);

% Get number of ensemble activations
activations = samples*event_percentage;

% Get maximum posible activations
max_activations = floor(samples/(min_separation+ensemble_length)+0.5);
if activations>max_activations
    activations = max_activations;
    event_percentage = activations/samples;
    warning(['The given ensemble event percentage will not be achieved. '...
        'The event_percentage was set to ' num2str(event_percentage)])
end

% maximum number of frames between ensemble events
max_separation = round(2*(samples/activations-min_separation-ensemble_length))+1;

% Set same seed
if isempty(seed)
    rng shuffle
else
    rng(seed)
end

%% Create the events
events = zeros(n_ensembles,samples);
n_activations = n_ensembles*activations;

% Generate random activations
rand_activity = randi(max_separation,1,n_activations)+min_separation+ensemble_length;
rand_activity = cumsum(rand_activity);

% Assign activations to ensembles
indx = repmat(1:n_ensembles,1,activations);
indx = indx(randperm(n_activations));
for i = 1:n_ensembles
    events(i,rand_activity(indx==i)) = 1;
end

% Get the ensemble activation events
events = events(:,1:samples);

%% Create the raster
raster = zeros(n_neurons,samples);

% Create events per ensemble
for i = 1:n_ensembles    
    % Get all ensemble events
    ind = find(events(i,:));
    n_events = length(ind);
    neurons_ensemble = find(p_ensemble(i,:));
    n_neurons_ensemble = nnz(p_ensemble(i,:));
    
    % Generate a single event
    for j = 1:n_events
        event_time = zeros(1,samples);
        event_time(ind(j)) = 1;
        
        % Jittered neurons based on jitter standard deviation
        if jitter_sd
            % Get a random jitter based on normal distribution, here
            % jitter_sd is the standard deviation
            shifts = round(normrnd(0,jitter_sd,n_neurons_ensemble,1));
            shifts = shifts-min(shifts)+1;

            % Set consecutive frames after jitter
            rand_activity = [];
            for k = 1:n_neurons_ensemble
                rand_activity(k,shifts(k):shifts(k)+ensemble_length-1) = 1;
            end
            
            % Assign the ensemble activity
            event = zeros(n_neurons,size(rand_activity,2));
            event(neurons_ensemble,:) = rand_activity; 
        else
            % Assign the ensemble activity
            event = zeros(n_neurons,ensemble_length);
            event(neurons_ensemble,:) = ones(n_neurons_ensemble,ensemble_length);
        end

        % Remove neuronal activations of the ensemble with probability p_ensemble
        for k = 1:n_neurons_ensemble
            if p_ensemble(i,neurons_ensemble(k))<rand
                event(neurons_ensemble(k),:) = 0; 
            end
        end
        
        % Add data to raster
        newData = conv2(event_time,event);
        newData = newData(:,1:samples);
        raster = raster + newData;
    end
end

%% Add noise to raster
raster = raster+(rand(size(raster))<noise);
raster = raster-(rand(size(raster))<noise);
raster = raster>0;

%% Generate exponential decays (simulating calcium signals)
filtbio = [zeros(1,10*10) exp(-(1:10*10)/10)]; 
raw = conv2(raster,filtbio,'same');

%% Generate a single vector of all events
all_events = zeros(1,samples);
for i = 1:n_ensembles
    all_events(events(i,:)>0) = i;
end
