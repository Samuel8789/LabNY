function [edge_wt_thresh] = threshold_edge_weights(edge_wt,num_dev)
dev = std(edge_wt);
thresh = num_dev*dev;
edge_wt_thresh = [];

L = length(edge_wt);
i = 1;

for i = 1:L
    if edge_wt(i) < thresh;
        edge_wt_thresh(i) = 0;
        i=i+1;
    else edge_wt(i) >= thresh;
        edge_wt_thresh(i) = edge_wt(i);
        i=i+1;
    end
end
end
