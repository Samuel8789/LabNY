function [PCNs,PAPS_INDEXED] = PAPS_score(best_model,results,params,p)

%import relevant information
data=params.data;
UDF=params.UDF;
num_neur = size(data,2);
num_ens = params.UDF_Count;
ens = results.core_crf;
ENS_STATE = cell(1,num_ens);

node_str=results.epsum(1:num_neur);
degrees = sum(best_model.structure(1:num_neur,1:num_neur));
auc=results.auc(1:num_neur,:);
edge_potentials = best_model.theta.edge_potentials(1:num_neur,1:num_neur);

for i = 1:num_ens
    ENS_STATE{i}=auc(:,i);
    ENS_STATE{i}=round(ENS_STATE{i});
    size_ens{i}=sum(ENS_STATE{i});
end

if nargin < 4
    p = 0.2;
end

%transition times T
T=cell(1,num_ens);
for i = 1:num_ens
    for ii = 2:size(data,1)
        if UDF(ii,i)==1 & UDF(ii-1,i)==0
            T{i}=[T{i} ii];
        end
    end
end


data=transpose(data);
data(data<1)=-1;

delta_Ei = @(Si,t) - (data(Si,t)-data(Si,t-1))*sum(sum(transpose(edge_potentials(Si,:)).*data(:,t)));

DELTA_U=cell(1,num_ens);

for i = 1:num_ens
    DELTA_U{i}=zeros(length(T{i}),num_neur);
    z=1;
    for ii = T{i}
        for k = 1:num_neur
            DELTA_U{i}(z,k) = delta_Ei(k,ii);
        end
        z=z+1;
    end
end

for i = 1:num_ens
    ENS_STATE{i}=ENS_STATE{i}.*transpose([1:num_neur]);
    ENS_STATE{i}(ENS_STATE{i}==0)=[];
end

DELTA_ENS = cell(1,num_ens);
DELTA_ENS_SUM = cell(1,num_ens);
DELTA_U_SUM=cell(1,num_ens);
for i = 1:num_ens
    %for ii = 1:num_neur
    DELTA_U_SUM{i} = sum(DELTA_U{i});
    DELTA_ENS{i} = DELTA_U{i}(:,[ENS_STATE{i}]);
    DELTA_ENS_SUM{i}=sum(DELTA_ENS{i});
end

DEMA = []; %Max Ei
DEMI = []; %Min Ei
nE = []; %min-max norm Ei
dmax = []; %max deg
dmin = []; %min deg
nD = []; %min-max norm deg
nsmi = []; %min node str
nsma = []; %max node str
nS = []; %min max norm node str
auc2 = []; %auc by ens
PAPS_INDEXED = cell(2,num_ens); %INDEX OF PAPS BY ENS
for i = 1:num_ens
    %remember to flip sign!!!!
    DEMA = min(DELTA_ENS_SUM{i})*-1;
    DEMI = max(DELTA_ENS_SUM{i})*-1;
    nE = ((DELTA_U_SUM{i}([ENS_STATE{i}]).*-1)-DEMI)./(DEMA-DEMI);
    
    dmax = max(degrees([ENS_STATE{i}]));
    dmin = min(degrees([ENS_STATE{i}]));
    nD = (degrees(ENS_STATE{i})-dmin)./(dmax-dmin);
    
    nsmi= min(node_str([ENS_STATE{i}]));
    nsma = max(node_str([ENS_STATE{i}]));
    nS = transpose((node_str([ENS_STATE{i}])-nsmi)./(nsma-nsmi));
        
    auc2 = auc([ENS_STATE{i}],i);
    auc2=transpose(auc2);
    
    PAPS_INDEXED{1,i}=ENS_STATE{i};
    for ii = 1:length(ENS_STATE{i})
        PAPS = @(Ni) (nE(Ni)+nD(Ni)+nS(Ni)+auc2(Ni))/4;
        PAPS_INDEXED{2,i}=[PAPS_INDEXED{2,i} PAPS(ii)];
    end
end

%Now Select Only the top p% of each ensemble
PCNs = cell(1,num_ens);
for i = 1:num_ens
    [V,idx] = maxk(PAPS_INDEXED{2,i},ceil(p*size_ens{i}));
    PCNs{i} = transpose(PAPS_INDEXED{1,i}(idx));
end

end
