function [] = Multicore_Modeling(a)
basepath = pwd;

a=a+1;

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

load('parameters.mat');
addpath(params.exptdir);
load(strcat(params.exptdir,'/model_parameters.mat'));
fname = strcat(params.exptdir,'/structures',num2str(a),'.mat');
load(fname);

addpath(strcat(params.exptdir,'/tmp'));

models = MDL_POOL(~cellfun(@isempty,MDL_POOL));
TF = isempty(models);

if TF==1
	fprintf('No Models Distributed to Core')
	fprintf('\n')
	exit;
end

%Make Parameter Estimation Obect
model_collection = LoopyModelCollection(models,params);

%Clean-up memory
clear models

fprintf('Parameter Estimation...\n');
%Parameter Estimation
        model_collection = model_collection.do_parameter_estimation(...
            params.BCFW_max_iterations, params.BCFW_fval_epsilon,...
            params.compute_true_logZ, params.reweight_denominator,...
            params.printInterval, params.printTest, params.MaxTime);
        
 warning('off','all');
%Save Collection & Identify Best Model       
tmpName = tempname(strcat(params.exptdir,'/tmp'));
tmpName = strcat(tmpName,'.mat');
warning('off','all');
model_collection = struct(model_collection);
save(tmpName, 'model_collection');

exit;

end
