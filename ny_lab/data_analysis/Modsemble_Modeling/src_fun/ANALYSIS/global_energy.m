function [GE] = global_energy(best_model,data,UDF);

%global energy at state S is defined as
data(data<1)=-1;
%E = -1/2*S^T*W*S
edge_potentials = best_model.theta.edge_potentials(1:size(data,2),1:size(data,2));
GE = [];
for i = 1:size(data,1)
    GE = [GE; -0.5.*sum(sum((transpose(data(i,:)).*edge_potentials.*data(i,:))))];
end

GE2 = []
for i = 1:size(data,1)
    GE2 = [GE2; sum(sum(-0.5*sum(sum((transpose(data(i,:)))))*edge_potentials.*data(i,:)+1*(data(i,:))))];
end

%GE = GE./sum(transpose(data));

UDF_ENERGY = GE.*UDF(:,1);

figure
[C,h] = contour(GG12)
h.LineWidth=3;
h.Fill='on';
hold on
X = [7 8 8 8 8 9 9 9 10 10];
Y = [10 9 8 7 6 5 4 3 2 1];
M=plot(X,Y)
M.LineWidth=5;
hold off
end
    
