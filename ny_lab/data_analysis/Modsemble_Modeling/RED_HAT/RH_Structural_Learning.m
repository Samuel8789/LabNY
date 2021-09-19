% MODSEMBLE 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% See README for papers and proper citations
% See GLOSSARY for documentation of variables and parameters
% Laboratory of Rafael Yuste
% Neurotechnology Center, Columbia University, 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[basepath,params]=RHstartup();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%STRUCTURAL LEARNING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%generate neighborhoods
params.variable_groups = all_but_me(1, size(params.x_train,2), params);

%perform regularization across parameter sequences for each neighborhood
[params]=learn_structures_opt(params);

%calculate structures
for i = 1:params.num_structures
    params.learned_structure{i} = learn_structures(params,params.s_lambda_sequence(i));
end

%feedback
fprintf('\n')
fprintf(strcat(num2str(params.num_structures),' Structures Formed'))
fprintf('\n');

[models] = pre_allocate_models(params);
fprintf(strcat(num2str(numel(models)), ' Models Pre-Allocated for Parameter Estimation'))
fprintf('\n')
fprintf('\n')

save(strcat(params.exptdir, '/', 'structures.mat'), 'models');
save(strcat(params.exptdir, '/', 'model_parameters.mat'),'params');
save('parameters.mat','params');

if params.multicore==1
    Distribute_Model_Pools(params,models);
end
