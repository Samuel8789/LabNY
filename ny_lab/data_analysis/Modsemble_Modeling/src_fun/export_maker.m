function [NODE,EDGE] = export_maker(best_model,MODEL_STRUCTURE,STATE_TEMP,M,K,coords)
% 39 52 53 58 55 are the MAX PENS]
Ed = best_model.theta.edge_potentials;
MODEL = graph(Ed);
q = MODEL_STRUCTURE.HOP{1,M}.SELECTIONS{1,K}(1,:);
A=[];
D=[];
V=[];


A = MODEL.Edges;
A = splitvars(A);
A = table2array(A);

for i = 1:length(q)
    D = [D; outedges(MODEL,q(i))];
end

PP = STATE_TEMP(:,K);
PP(PP<1)=0;
X = coords(:,1);
X = normalize(X,'range',[1 90]);
Y = coords(:,2);
Y = normalize(Y,'range',[-90 -1]);
X = X.*PP;
Y = Y.*PP;
X(X==0)=[];
Y(Y==0)=[];
PP = PP.*transpose([1:810]);
PP(PP==0)=[];

Edges = A;

ID={};
for i = 1:length(Edges)
    for a = 1:length(q)
        if Edges(i,1)==q(1,a)
            if ismember(Edges(i,2),PP)
                ID{end+1}=Edges(i,:);
            end
        elseif Edges(i,2)==q(1,a)
            if ismember(Edges(i,1),PP)
                ID{end+1}=Edges(i,:);
            end
            
        end
    end
end

ID = transpose(ID);
ID = cell2mat(ID);

V = unique([A(D,1); A(D,2)]);
N = ismember(PP,V);
nodes = PP.*N;
nodes(nodes~=0)=1;
STATE=nodes;

 for i = 1:length(PP)
     for m = 1:length(q)
         if PP(i)==q(m)
             STATE(i)=2;
         end
     end
 end
 
NODE = cell(1+length(PP),4);
NODE{1,1} = 'Id';
NODE{1,2} = 'X';
NODE{1,3} = 'Y';
%unconnected = 0, connected = 1, seeds = 2
NODE{1,4} = 'STATE';


for i = 1:length(PP)
    NODE{1+i,1}=PP(i);
    NODE{1+i,2}=X(i);
    NODE{1+i,3}=Y(i);
    NODE{1+i,4}=STATE(i);

end

  

EDGE  = cell(1+size(ID,1),3);
EDGE{1,1}='Source';
EDGE{1,2}='Target';
EDGE{1,3}='Weight';
for i = 1:length(ID)
    EDGE{1+i,1}=ID(i,1);
    EDGE{1+i,2}=ID(i,2);
    EDGE{1+i,3}=ID(i,3);
end

writecell(NODE,'NODE.csv');
writecell(EDGE,'EDGE.csv');

end




