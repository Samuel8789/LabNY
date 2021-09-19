function [ext_IDs] = find_IDs(data)

CS = cell(size(data,1),size(data,1));

for i = 1:size(data,1)
    a = 1:size(data,1);
    for b = a
        CS{b,i} = cosim(transpose(data(i,:)),transpose(data(b,:)));
    end
end

CS = cell2mat(CS);

CS(logical(eye(size(CS))))=NaN;

CSt = reshape(CS,[],1);
n_mean = nanmean(CSt);
n_std = nanstd(CSt);
n_std=n_std*3;

UL = n_mean+n_std;

CSm = CS;

for i = 1:size(data,1)
    a = 1:size(data,1);
    for b = a
        if CSm(b,i) <= UL
            CSm(b,i) = NaN;
        end
    end
end

ens_partners = cell(1,size(data,1));

for i = 1:size(CSm,2)
    ens_partners{1,i} = find(isnan(CSm(:,i))==0);
end

UDF = zeros(size(data,1),length(ens_partners));

for i = 1:length(UDF);
    for a = 1:size(data,1)
        if ismember(a,cell2mat(ens_partners(1,i)))
            UDF(a,i)=1;
        end
    end
end

ext_IDs = UDF;

end