function [model_collection] = multicore_merge(params)
%get filenames & remove empties
    tmp_dir = params.tmp_dir;
    Files = dir(tmp_dir);
    Files = Files(cellfun(@any,{Files.bytes}));
    load(strcat(params.tmp_dir,'/',Files(1,1).name));
    temp_collection=model_collection;
    
    for i = 2:length(Files)
        load(strcat(params.tmp_dir,'/',Files(i,1).name));
        temp_collection.models = [temp_collection.models model_collection.models];
    end
    model_collection.models = temp_collection.models;
end
    
