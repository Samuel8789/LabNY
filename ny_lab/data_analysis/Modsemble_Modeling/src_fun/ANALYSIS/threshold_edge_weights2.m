function [edge_wt_thresh] = threshold_edge_weights2(edge_wt,thresh)
dev = std(edge_wt);
edge_wt_thresh = [];

L = length(edge_wt);
i = 1;

for i = 1:L
    if edge_wt(i) < thresh;
        edge_wt_thresh(i) = NaN;
        i=i+1;
    else edge_wt(i) >= thresh;
        edge_wt_thresh(i) = edge_wt(i);
        i=i+1;
    end
end
end
