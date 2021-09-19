%%script testing weird datasets through cyclical coordinated descentfor
%%error testing

%% SELECT FILE
filename = uigetfile('*.mat');
load(filename);

%% INITIALIZE
params.data = data;
%params.stimuli = stimuli;
params.split =0.8;
params.time_span=1;
X = params.data;
        sample_count = size(X,1);
        x_train_base = X(1:floor(params.split*sample_count),:);
        x_test_base = X((floor(params.split*sample_count)+1):sample_count,:);

%% Prep data

        x_train = add_lookback_nodes(x_train_base, params.time_span);
        x_test = add_lookback_nodes(x_test_base, params.time_span);
        %stim_count = size(params.stimuli, 2);
        % Append any stimulus nodes
        %if stim_count > 0
         %   assert(sample_count == size(params.stimuli, 1), ...
          %         'Stimuli and neuron data must have same number of samples.')
           % x_train = [x_train params.stimuli(1:floor(params.split*sample_count),:)];
            %x_test = [x_test params.stimuli((floor(params.split*sample_count)+1):sample_count,:)];
        %end
        
        variable_groups = all_but_me(1, size(x_train, 2));
        
 %% NET STUFF
 label_node = 4;
 feature_nodes = variable_groups{label_node};
        X = x_train(:,feature_nodes);
        Y = x_train(:,label_node);
        CVerr = cvglmnet(X,Y,'binomial');