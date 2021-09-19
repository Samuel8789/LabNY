
x=allplanedfdt(:,60:62800);
imagesc(allplanedfdt);

%%
imagesc(x);
plot(x(1,:));
plot(allplanedfdt(1,:));

%threshold
trace=x;
zzz=sqrt(mean(trace.^2,2 ));
z_thresh=2*zzz;
for jj=1:length(trace)
    for ii=1:271
        if trace(ii, jj)<=z_thresh(ii,1)
         trace(ii, jj)=z_thresh(ii,1);
        end
    end
    trace(:, jj)=trace(:, jj) - z_thresh;
end
imagesc(trace);
plot(trace(1,:));
raster=trace;


%%
matlabdiftingindexoff=diftingindexoff+1;
matlabdiftingindexon=diftingindexon+1;
signalsfile='C:\\Users\\sp3660\\Desktop\\TemporaryProcessing\\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'


% load(vistimfile)
T = readtable(signalsfile);

%%

vistim=T.('VisStim');
LED=T.('LED');
PhotoDiode=T.('PhotoDiode');
Locomotion=T.('Locomotion');
clear T

figure(1)
tiledlayout(4,1)
ax1 = nexttile
plot(vistim)

hold on 
plot(matlabdiftingindexon, vistim(matlabdiftingindexon), 'ro')
plot(matlabdiftingindexoff, vistim(matlabdiftingindexoff), 'rx')

hold off
ax2 = nexttile;
plot(PhotoDiode)
ax3 = nexttile;
plot(LED)
ax4 = nexttile;
plot(Locomotion)
linkaxes([ax1 ax2 ax3 ax4],'x')

movie_rate=16.10383676648614 ;
milisecond_period=1000/movie_rate;
movie_frames_tuning_on=matlabdiftingindexon/milisecond_period;
movie_frames_tuning_off=matlabdiftingindexoff/milisecond_period;
test=reshape(movie_frames_tuning_on,[1,numel(movie_frames_tuning_on)])

imagesc(allplanedfdt);
plot(allplanedfdt(1,:));
hold on
plot(test, zeros(1,numel(movie_frames_tuning_on)),'ro');


movie_frames_tuning_on_bordercuts=movie_frames_tuning_on-60
movie_frames_tuning_off_bordercuts=movie_frames_tuning_off-60
reshaped_movie_frames_tuning_on_bordercuts=reshape(movie_frames_tuning_on_bordercuts',[1,numel(movie_frames_tuning_on_bordercuts)])
reshaped_movie_frames_tuning_off_bordercuts=reshape(movie_frames_tuning_off_bordercuts',[1,numel(movie_frames_tuning_off_bordercuts)])




imagesc(trace);
hold on
plot(movie_frames_tuning_on_bordercuts(1,:), ones(1,15),'ro');
for grating=1:15
    line([movie_frames_tuning_on_bordercuts(1,grating),movie_frames_tuning_on_bordercuts(1,grating)], [0,271], 'Color', 'r');
end
figure
plot(trace(1,:));
hold on
for grating=1:15

    line([movie_frames_tuning_on_bordercuts(1,grating),movie_frames_tuning_on_bordercuts(1,grating)], [0,max(trace(1,:))], 'Color', 'r');
end
%%
count=1;
count2=1;
gratings=zeros(size(trace));

for i=1:length(reshaped_movie_frames_tuning_off_bordercuts);
    gratings(:,reshaped_movie_frames_tuning_on_bordercuts(i):reshaped_movie_frames_tuning_off_bordercuts(i))=count2;
    if count==15;
        count=0;
        count2=count2+1;
        
    end  
    count=count+1;

end

imagesc(gratings)
M = max(gratings,[],'all')



