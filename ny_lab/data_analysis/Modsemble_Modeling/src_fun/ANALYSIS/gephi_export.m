function [NODES, EDGES] = gephi_export(best_model,results,coords)
    %this creates gephi export
    
    %created undirected model
    Model = best_model.theta.edge_potentials;
    Model = graph(Model);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%WRITE EDGES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %extract and convert edges
    Edges = Model.Edges;
    Edges = splitvars(Edges);
    Edges = table2array(Edges);
    
    %create export cell
    EDGES = cell((length(Edges)+1),3);
    EDGES{1,1} = 'Source';
    EDGES{1,2} = 'Target';
    EDGES{1,3} = 'Weight';
    
    for i = 1:length(Edges)
        EDGES{1+i,1}=Edges(i,1);
        EDGES{1+i,2}=Edges(i,2);
        EDGES{1+i,3}=Edges(i,3);
    end
    
    %write export file
    writecell(EDGES,'EDGE.csv');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%WRITE NODES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Nodes = best_model.theta.node_potentials;
    Nodes = [Nodes Nodes];
    Nodes(1:end,1)=[1:length(Nodes)];
    
    %fix node str
    epsum = results.epsum;
    epsum(isnan(epsum))=0;
    
    %create nodes table
    NODES = cell((length(Nodes)+1),3);
    NODES{1,1} = 'Id';
    NODES{1,2} = 'Node_Pot';
    NODES{1,3} = 'Node_Str';
    
    for i = 1:length(Nodes)
        NODES{1+i,1}=Nodes(i,1);
        NODES{1+i,2}=Nodes(i,2);
        NODES{1+i,3}=epsum(i);
    end
    
    %extract coordinates
    if nargin > 2
        if size(coords,2) == 2
            NODES{1,4} = 'X';
            NODES{1,5} = 'Y';
            for i = 1:length(Nodes)
                NODES{1+i,4} = coords(i,1);
                NODES{1+i,5} = coords(i,2);
            end
        elseif size(coords,3) == 3
            NODES{1,4} = 'X';
            NODES{1,5} = 'Y';
            NODES{1,6} = 'Z';
            for i = 1:length(Nodes)
                NODE{1+i,4} = coords(i,1);
                NODES{1+i,5} = coords(i,2);
                NODES{1+i,6} = coords(i,3);
            end
        else
            fprintf('ERROR')
        end
    end
    
    %write export file
    writecell(NODES,'NODE.csv');
     
end

    
    
    
