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
        
        
%Save Collection & Identify Best Model  
[best_model_index] = get_best_model(model_collection);
[best_model] = SingleLoopyModel(model_collection, best_model_index);
%Convert to Structures
model_collection=struct(model_collection);
best_model = struct(best_model);
save(strcat(params.exptdir, '/', 'model_collection.mat'), 'model_collection');
save(strcat(params.exptdir, '/', 'best_model.mat'), 'best_model');
     