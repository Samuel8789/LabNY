function [recommended,indices] = Cluster_Test(treeOrData,similarity,metric,clusteringMethod,groups,fig)
% Clustering indexes
% Get indexes for evaluating clustering from hierarchical cluster tree or
% data points to perform 
%
%       [recommended,indices] = Cluster_Test(treeOrData,similarity,metric,clusteringMethod,groups,fig)
%
%       default: metric = 'contrast'; clusteringMethod = 'hierarchical';
%                groups = 2:30; fig = []
%
% Inputs:
%      treeOrData = hierarchical cluster tree, or data for k-means
%      similarity = matrix PxP (P=#peaks) for metrics Dunn &
%                    Contrast; Xpeaks, peaks vectors as matrix PxC for metrics
%                    Connectivity & Davies
%                    (P = #peaks; C=#cells)
%      metric = index to compute ('dunn','connectivity','davies','contrast')
%      clusteringMethod = 'hierarchical' or 'kmeans'
%      groups = range of groups to analize
%      numFig = number of the figure to plot
%
% Outputs:
% indices = clustering indices of 'metric' from the range of 'groups'.
% recommended = recommended number of clusters
%
% ..:: by Jesús E. Pérez-Ortega ::.. Jun-2012
% modified Mar 2018
% modified Apr 2020
% modified Sep 2021

switch nargin
    case 2
        metric = 'contrast';
        clusteringMethod = 'hierarchical';
        groups = 2:30;
        fig = [];
    
    case 3
        clusteringMethod = 'hierarchical';
        groups = 2:30;
        fig = [];
    case 4 
        groups = 2:30;
        fig = [];
    case 5 
        fig = [];
end

dist = 1-similarity;
j = 1;
for i = groups
    switch clusteringMethod
        case 'hierarchical'
            T = cluster(treeOrData,'maxclust',i);
        case 'kmeans'
            T = kmeans(treeOrData,i);
    end
    g = max(T);
    
    switch metric
        case 'dunn'
            indices(j) = Dunn_Index(g,dist,T);
        case 'davies'
            indices(j) = Davies_Index(g,similarity,T);
        case 'contrast'
            indices(j) = Contrast_Index(g,similarity,T);    
        case 'contrast_excluding_one'
            indices(j) = Contrast_Index(g,similarity,T,true);
    end
    j = j+1;
end

switch metric
    case 'contrast_excluding_one'
        if nnz(indices == 1)
            id = find(indices == 1,1,'last');
            recommended = groups(id);
        else
            [~,id] = max(indices);
            recommended = groups(id);
        end
    otherwise
        % Select the best number of groups based on an index
        % If any index is equal to 1 it is a perfect separation, so choose the
        % maximum number of groups
        if nnz(indices == 1)
            id = find(indices == 1,1,'last');
            recommended = groups(id);
        else
            [~,id] = find(diff(indices)>0,1,'first');
            if isempty(id) || id==length(groups)-1
                % The indices are decreasing, so select the first
                recommended = groups(1);
                id = 1;
            else
                % Find the first peak of the indices
                indicesCopy = indices;
                indicesCopy(1:id) = 0;
                [~,id] = find(diff(indicesCopy)<0,1,'first');
                if isempty(id)
                    % If there is no peak find the first sudden increase
                    [~,id] = find(diff(diff(indicesCopy))<0,1,'first');
                    id = id+1;
                end
                recommended = groups(id);
            end
        end
end

% Plot 
if ~isempty(fig)
    plot(groups,indices)
    hold on
    plot(recommended,indices(id),'*r')
    hold off
    title([replace(metric,'_','-') '''s index (' num2str(recommended) ' groups recommended)'])
    xlabel('number of groups')
    ylabel('index value')
end