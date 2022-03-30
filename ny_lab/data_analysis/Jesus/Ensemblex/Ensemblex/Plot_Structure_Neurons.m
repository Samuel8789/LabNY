function colors_structure = Plot_Structure_Neurons(structure,colors,new_figure)
% Plot the structure of the ensembles, i.e., the neurons that belong to the
% ensembles and their weights
%
%       colors_structure = Plot_Structure_Neurons(structure,colors,newfigure)
%
%       default: colors = []; new_figure = false
%
%       structure: columns represent neurons and rows represent ensembles.
%
% By Jesus Perez-Ortega, Aug 2021
% Modified Sep 2021
% Modified Oct 2021

switch nargin
    case 1
        colors = [];
        new_figure = false;
    case 2
        new_figure = false;
end

% Set colors
structure = structure';
[n_neurons,n_ensembles] = size(structure);

% Get colors
if isempty(colors)
    colors = Read_Colors(n_ensembles);
end

% Get hue 
hsvColors = rgb2hsv(colors);
hue = hsvColors(:,1);
hues = repmat(hue,1,n_neurons)';

% Set saturation to colors
for i = 1:n_ensembles
    sat = hsvColors(i,2);
    saturation(:,i) = structure(:,i)*sat;
end

% Set values
value = hsvColors(:,3);
values = repmat((1-value),1,n_neurons)'.*(1-structure)+repmat(value,1,n_neurons)';

% Create image
color_neurons = hsv2rgb(cat(3,hues,saturation,values));

% Plot structure
if new_figure
    Set_Figure('Structure weigthed',[0 0 40*n_ensembles 300]);
end

% Plot each neuron
for i = 1:n_ensembles
    colors = squeeze(color_neurons(:,i,:));
    for j = n_neurons:-1:1
        if sum(colors(j,:))<3
            plot(i,j,'o','MarkerEdgeColor',colors(j,:)*2/3,...
                'MarkerFaceColor',colors(j,:),'MarkerSize',10); hold on
        end
    end
end
xlim([1 n_ensembles])
xlabel('ensemble #')
ylabel('neuron #')
set(gca,'ydir','normal')

if nargout
    colors_structure = color_neurons;
end