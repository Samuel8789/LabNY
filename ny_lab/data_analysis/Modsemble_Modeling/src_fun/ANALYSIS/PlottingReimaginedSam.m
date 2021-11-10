num_stim = size(params.UDF,2);
core_crf = results.core_crf;
epsum = results.epsum;
epsum(isnan(epsum))=0;
epsum = normalize(epsum,'range');
good_epsum=epsum(1:end-8,:);
coords=params.coords;

good_structure=best_model.structure(1:end-8,1:end-8);
good_edge_potentials=best_model.theta.edge_potentials(1:end-8,1:end-8);


auc = results.auc;
good_auc=auc(1:end-8,:);
auc_ens = results.auc_ens;

time_span=1;
num_subplots = num_stim;
num_orig_neuron = length(good_epsum);

nodesz = 30;
nsmi = 0;
nsma = 1;
aucmi = 0;
aucma = 1;
[s,t,edge_wt] = vectorize_model(good_structure, good_edge_potentials);
MDL = graph(s,t,normalize(edge_wt,'range',[-1 1]));
MDL.Edges.EdgeColors = MDL.Edges.Weight/max(MDL.Edges.Weight);
MDL.Edges.LWidths = 1*normalize(MDL.Edges.Weight,'range', [0.01 0.99])/max(normalize(MDL.Edges.Weight,'range',[0.01 0.99]));

for ii = 1:num_stim
    f = figure; set(gcf,'color','w')
    f.Name = sprintf('Ensemble %d', ii);
    f.WindowState = 'maximized';
    
    subplot(2,2,1)
    MODEL = plot(MDL,'XData',coords(:,1),'YData',coords(:,2),'ZData',coords(:,3));
    MODEL.EdgeCData = MDL.Edges.EdgeColors;
    MODEL.LineWidth = MDL.Edges.LWidths;
    MODEL.NodeLabel = {};
    MODEL.Marker='o';
    xlabel('X Pixels'); ylabel('Y Pixels'); zlabel('Z Pixels');
    xlim([0 256]);ylim([0 256]);zlim([0 3]);
    title('Pattern Completors Within Network Functional Connectivity Map')
    colormap(winter)
    colorbar
    highlight(MODEL,[1:length(good_auc)], 'NodeColor',[0 0 0]);
    highlight(MODEL,[PCNs{ii}],'NodeColor','r');
    
    
    cur_axes = subplot(1,1,1); hold on
    colormap(cur_axes, autumn)
    %scatter(good_epsum,auc(:,ii),nodesz,0.5*[0 0 1],'filled')
    scatter(good_epsum,good_auc(:,ii),nodesz,'k','linewidth',0.5)
    scatter(good_epsum(PCNs{ii}),good_auc(PCNs{ii},ii),nodesz, [1 0 0], 'filled')
     
    plot([nsmi nsma],mean(auc_ens{ii})*[1 1],'k--');
    plot([nsmi nsma],(mean(auc_ens{ii})+std(auc_ens{ii}))*[1 1],'--',...
            'color',0.7*[1 1 1]);
    plot([nsmi nsma],(mean(auc_ens{ii})-std(auc_ens{ii}))*[1 1],'--',...
            'color',0.7*[1 1 1]);
        
    plot(mean(good_epsum)*[1 1],[aucmi aucma],'k--');
    plot((mean(good_epsum)+std(good_epsum))*[1 1],[aucmi aucma],'--',...
            'color',0.7*[1 1 1]);
    plot((mean(good_epsum)-std(good_epsum))*[1 1],[aucmi aucma],'--',...
            'color',0.7*[1 1 1]);
     xlim([nsmi nsma]); ylim([aucmi aucma])
     xlabel('Node Strength'); ylabel(['AUC (Ensemble: ' num2str(ii)]);
     title(['Pattern Completors of Ensemble: ' num2str(ii)])
     X = [(mean(good_epsum)-std(good_epsum)) (mean(good_epsum)+std(good_epsum)) (mean(good_epsum)+std(good_epsum)) (mean(good_epsum)-std(good_epsum))];
     Y = [(mean(auc_ens{ii})-std(auc_ens{ii})) (mean(auc_ens{ii})-std(auc_ens{ii})) (mean(auc_ens{ii})+std(auc_ens{ii})) (mean(auc_ens{ii})+std(auc_ens{ii}))];
     patch(X,Y,[0.5 0.5 0.5],'FaceAlpha',0.25);

     subplot(2,2,3:4)
     plotGraphHighlight(coords,mod(PCNs{ii}-1, num_orig_neuron)+1, 'red', 1);
     xlabel('X Pixels'); ylabel('Y Pixels'); zlabel('Z Pixels');
     xlim([0 256]);ylim([0 256]);zlim([0 3]);
     title('Coordinates of Pattern Completors for Optogenetic Targeting');
     hold off
end

