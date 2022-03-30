 



data_path='G:\CodeTempRawData\LabData\Chandelier_Imaging'

close all
listing = dir(data_path);
zz={listing.name};
raw_to_split=zeros(1,length(zz));  
for k=1:length(zz)
   raw_to_split(1,k) =endsWith(zz(k),'rand.mat');
end  
fileNamesRaw=zz(logical(raw_to_split));

for dd=1:length(fileNamesRaw)
    
    fileNameRaw=fileNamesRaw{dd};
    file = append(data_path,'\', fileNameRaw);
    load(append(file(1:end-4),'_stiminfo.mat'))
    load(file)
    rate=(MOVIE_FRAMSE/movie_duration)/60
    duration_spont_seconds=length(random_raster_matrix_spont)/rate
    duration_trials_seconds=length(random_raster_matrix_trials)/rate
    
    new_value=rate*50
    fg1=figure(1);

    set(0,'CurrentFigure',fg1);
    imagesc(logical(random_raster_matrix_spont));
    set(gca,'colormap',[1 1 1; 0 0 0])
    set(gca,'XTick',0:new_value:length(random_raster_matrix_spont))
    set(gca,'XTickLabel',0:new_value/rate:duration_spont_seconds)
    xlabel('Time(s)')
    ylabel('Cells')
    saveas(fg1,append(file(1:end-4),'_spontraster'),'pdf')

    fg2=figure(2);

    set(0,'CurrentFigure',fg2);
    
    trial_length=130
    isi=zeros(5,30)
    stim_per=zeros(5,100)
   
    

    imagesc(logical(random_raster_matrix_trials));
    set(gca,'colormap',[1 1 1; 0 0 0])
    set(gca,'XTick',0:new_value:length(random_raster_matrix_trials))
    set(gca,'XTickLabel',0:new_value/rate:duration_trials_seconds)
    xlabel('Time(s)')
    ylabel('Cells')
    hold on
    for trials=1:5
        isi(trials,:)=(((trials-1)*trial_length)+([1:30]))*rate
        stim_per(trials,:)=(((trials-1)*trial_length)+([31:130]))*rate
        rectX1isi=isi(trials,1)
        rectX2isi=isi(trials,end)
        rectX1trial=stim_per(trials,1)
        rectX2trial=stim_per(trials,end)
        rectY = ylim;
        rectXisi=[rectX1isi rectX2isi]
        rectXtrial=[rectX1trial rectX2trial]
        pch1 = patch( rectXisi([1,2,2,1]), rectY([1 1 2 2]), 'r', ...
        'EdgeColor', 'r', 'FaceColor','none','LineWidth',2); % FaceAlpha controls transparency
    % Copy rectangle to 2nd axes and adjust y limits
        pch2 = patch( rectXtrial([1,2,2,1]), rectY([1 1 2 2]), 'g', ...
        'EdgeColor', 'g', 'FaceColor','none','LineWidth',2); % FaceAlpha controls transparency
    end

    clear random_raster_matrix_spont random_raster_matrix_trials
    saveas(fg2,append(file(1:end-4),'_trialraster'),'pdf')
    close all
  
end
    
    
    
    
   