function contrast_index = Contrast_Index(groups,similarity,indices,exclude)
% Contrast index
% Get the Contrast index for g groups given a similarity matrix.
% (Michelson contrast 1927, Plenz 2004)
%
%       contrast_index = Contrast_Index(groups,similarity,indices,exclude)
%
%       default: exclude = [];
%
% Inputs
% g = number of groups
% sim = similarity as matrix PxP (P = #peaks)
% idx = indexes of group to which each data point belongs
% 
% Outputs
% CstIdx = Contrast index

% by Jesus Perez-Ortega, Apr 2012
% Modified Sep 2021

switch nargin
    case 3
        exclude = false;
end

if exclude && groups<=2
    warning('It can not be an exclusion with 2 or less groups.')
    exclude = false;
end
   
% Remove diagonal values from similarity matrix
sim = similarity-diag(diag(similarity));

for i = 1:groups
    id_in = find(indices==i);
    id_out = find(indices~=i);

    % Similarity average inside group
    avg_in(i) = sum(sum(sim(id_in,id_in)))/numel(id_in)^2;
    
    % Similarity average outside group
    avg_out(i) = sum(sum(sim(id_in,id_out)))/(numel(id_in)*numel(id_out));
end

% Identify the group to exclude
if exclude && groups>2
    [~,group_excluded] = min(avg_in./avg_out);
    ids = setdiff(1:groups,group_excluded);
    
    for i = ids
        % Get indices from inside group and outside group
        id_in = find(indices==i);
        id_out = find(indices~=i);
        
        % Similarity average inside group
        avg_in(i) = sum(sum(sim(id_in,id_in)))/numel(id_in)^2;

        % Similarity average outside group
        avg_out(i) = sum(sum(sim(id_in,id_out)))/(numel(id_in)*numel(id_out));
    end
end

% Sum the similarities
S_in = sum(avg_in);
S_out = sum(avg_out);

% Compute the contrast index
contrast_index = (S_in-S_out)/(S_in+S_out);