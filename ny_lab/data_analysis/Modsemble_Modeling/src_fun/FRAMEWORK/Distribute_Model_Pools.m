function [] = Distribute_Model_Pools(params,models)
%Multiple Core Pre-Allocation and Distribution
%Find size of single-core model pool (model_pool_size)

mps = numel(models)/params.poolsize;
mps=ceil(mps);

for i = 1:(params.poolsize-1);
    MDL_POOL = cell(1,mps);
    m = 1+(mps*(i-1));
    n = mps+(mps*(i-1));
    z=1;
    for p = m:n
        MDL_POOL{1,z}=models{1,p};
        z=z+1;
    end
    fname = strcat(params.exptdir,'/structures',num2str(i),'.mat');
    save(fname);
end

for i = params.poolsize
    MDL_POOL = cell(1,mps);
    m = 1+(mps*(i-1));
    z=1;
    for p = m:numel(models);
        MDL_POOL{1,z}=models{1,p};
        z=z+1;
    end
    fname = strcat(params.exptdir,'/structures',num2str(i),'.mat');
    save(fname);
end
end
