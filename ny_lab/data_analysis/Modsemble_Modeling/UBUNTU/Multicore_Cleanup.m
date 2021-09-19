basepath = pwd;

% THIRDPARTY DEPENDENCIES
addpath(fullfile(basepath,'thirdparty'))
addpath(fullfile(basepath,'thirdparty','QPBO-v1.32.src'))
addpath(fullfile(basepath,'thirdparty','glmnet_matlab'))
addpath(fullfile(basepath,'thirdparty','glmnet_matlab','glmnet_matlab'))
addpath(fullfile(basepath,'expt'))

% SOURCE FUNCTIONS

addpath(fullfile(basepath,'src_fun'))
addpath(fullfile(basepath,'src_fun','ANALYSIS'))
addpath(fullfile(basepath,'src_fun','FRAMEWORK'))
addpath(fullfile(basepath,'src_fun','MLE_STRUC'))
addpath(fullfile(basepath,'src_fun','include'))
addpath(fullfile(basepath,'src_fun','STRUCTURE'))

%LOAD PARAMS
%Somewhat unneccessary here, but it's more likely that one runs this
%in-line rather that running many models and THEN picking best model. If
%you are cleaning up a multicore run after subsequent runs than you'd have
%to swap parameters in the base directory with model parameters.

load('parameters.mat');
addpath(params.exptdir);
load(strcat(params.exptdir,'/model_parameters.mat'));

tmp_dir = strcat(params.exptdir,'/tmp');
params.tmp_dir=tmp_dir;

[model_collection] = multicore_merge(params);


[best_model_index] = get_best_model(model_collection);

[best_model] = SingleLoopyModel(model_collection, best_model_index);

%Convert to Structures
warning('off','all');
model_collection = struct(model_collection);
best_model=struct(best_model);

save(strcat(params.exptdir, '/', 'model_collection.mat'), 'model_collection');

save(strcat(params.exptdir, '/', 'best_model.mat'), 'best_model');
