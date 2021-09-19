rootdir = 'D:\Projects\LabNY\Full_Mice_Data\Mice_Projects';
hdf5files = dir(fullfile(rootdir, '**\*.hdf5')); 
matfilesd = dir(fullfile(rootdir, '**\*.mat')); 
mmapfiles = dir(fullfile(rootdir, '**\*.mmap')); 
% Returns readDataCsv.m, readDataXml.m, readDataTxt.m, etc





mousecll={'SPJA','SPIL','SPJO','SPJP','SPJB','SPJD','SPJF','SPJT','SPJS','SPJU','SPIN','SPJC','SPIK','SPID','SPIC','SPGV','SPGX','SPGW','SPIB','SPIJ','SPIM','SPIH','SPHV','SPHW','SPHX','SPHY','SPHZ','SPGT','SPGH','SPJI','SPHC','SPHM','SPIG','SPIA'};
for z=1:length(mousecll);
    test={};
    for i=1:length(mmapfiles);
        if strfind(mmapfiles(i).folder,mousecll{1,z});   
            test{end+1,1} = mmapfiles(i).folder;
        end
    end
    mousecll{2,z}=test;
end

sumall=0
for i=1:length(mousecll)
    sumall=sumall+length(mousecll{2,i});
end


