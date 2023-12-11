
function [visstimfilepath]=resave_vis_stim_file(visstimfilepath)
    savefile = extractBefore(visstimfilepath, ".");
    load(visstimfilepath);
    if ~isa(full_info, 'struct')
        structArray = cell2struct(full_info(2:end,:), full_info(1,:), 2);
        for i=1:size(structArray,1);
              structArray(i).Paradigms=structArray(i).Paradigms{1};
        
              structArray(i).Trials = cell2struct(structArray(i).Trials(2:end,:), structArray(i).Trials(1,:), 2);
              for j=1:size(structArray(i).Trials,1);
                structArray(i).Trials(j).Phases = cell2struct(structArray(i).Trials(j).Phases(2:end,:), structArray(i).Trials(j).Phases(1,:), 2);
              end
        end 
        full_info=structArray;
        clear structArray;
        save(savefile,'-v7.3');
        display('Cell convrerted to structs');
    else
        display('BVisstim mat file File already processed');

    end

end
