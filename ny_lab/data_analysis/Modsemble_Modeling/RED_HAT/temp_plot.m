%plotting shit

edges = best_model.theta.edge_potentials;
nodes = best_model.theta.node_potentials;
node_norm = normalize(nodes,'range',[-10 -1]);
node_norm=node_norm.*-1;
node_wt = round(node_norm);


MODEL = graph(edges);

figure
MDL = plot(MODEL);
MDL.EdgeCData = MODEL.Edges.Weight/max(MODEL.Edges.Weight);
colorbar;
title('Cores Neurons in Ensemble 1: UDF1=g;CORE=r')

for i = 1:length(nodes)
    highlight(MDL,i,'MarkerSize',node_wt(i));
end

for i = 1:3
    highlight(MDL,i,'NodeColor','r');
end

for i = 11
    highlight(MDL,i,'NodeColor','g');
end
