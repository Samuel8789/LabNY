allenpath='C:\Users\sp3660\Documents\Projects\AllenBrainObservatory\HakimMonitor';
load(fullfile(allenpath,'locally_sparse_noise_1.mat'));
locally_sparse_noise_all_warped_frames_1=locally_sparse_noise_all_warped_frames;
load(fullfile(allenpath,'locally_sparse_noise_2.mat'));
locally_sparse_noise_all_warped_frames_2=locally_sparse_noise_all_warped_frames;
load(fullfile(allenpath,'locally_sparse_noise_3.mat'));
locally_sparse_noise_all_warped_frames_3=locally_sparse_noise_all_warped_frames;
load(fullfile(allenpath,'locally_sparse_noise_4.mat'));
locally_sparse_noise_all_warped_frames_4=locally_sparse_noise_all_warped_frames;
load(fullfile(allenpath,'locally_sparse_noise_5.mat'));
locally_sparse_noise_all_warped_frames_5=locally_sparse_noise_all_warped_frames;
locally_sparse_noise_all_warped_frames_full=cat(1,locally_sparse_noise_all_warped_frames_1, locally_sparse_noise_all_warped_frames_2, locally_sparse_noise_all_warped_frames_3, locally_sparse_noise_all_warped_frames_4, locally_sparse_noise_all_warped_frames_5);
% locally_sparse_noise_all_warped_frames_full=cat(1,locally_sparse_noise_all_warped_frames_1, locally_sparse_noise_all_warped_frames_2, locally_sparse_noise_all_warped_frames_3);

locally_sparse_noise_all_warped_frames_full=permute(locally_sparse_noise_all_warped_frames_full,[2,3,1]);
save( 'locally_sparse_noise_full.mat','locally_sparse_noise_all_warped_frames_full','-v7.3');
clear all
load('locally_sparse_noise_full.mat')
locally_sparse_noise_all_warped_frames_full(locally_sparse_noise_all_warped_frames_full==127)=135;
save( 'locally_sparse_noise_full_135.mat','locally_sparse_noise_all_warped_frames_full','-v7.3');
clear all
% 
allenpath='C:\Users\sp3660\Documents\Projects\AllenBrainObservatory\HakimMonitor';

load(fullfile(allenpath,'natural_movie_one_1.mat'));
natural_movie_one_all_warped_frames=permute(natural_movie_one_all_warped_frames,[2,3,1]);
save('natural_movie_one.mat','natural_movie_one_all_warped_frames');

load(fullfile(allenpath,'natural_movie_two_1.mat'));
natural_movie_two_all_warped_frames=permute(natural_movie_two_all_warped_frames,[2,3,1]);
imshow(squeeze(natural_movie_two_all_warped_frames(:,:,2)))
save('natural_movie_two.mat','natural_movie_two_all_warped_frames');

load(fullfile(allenpath,'natural_scenes_1.mat'));
natural_scenes_all_warped_frames=permute(natural_scenes_all_warped_frames,[2,3,1]);
imshow(squeeze(natural_scenes_all_warped_frames(:,:,1)))
save('natural_scenes.mat','natural_scenes_all_warped_frames');
clear all
allenpath='C:\Users\sp3660\Documents\Projects\AllenBrainObservatory\HakimMonitor';

% % 
load(fullfile(allenpath,'natural_movie_three_1.mat'));
natural_movie_three_all_warped_frames_1=natural_movie_three_all_warped_frames;
load(fullfile(allenpath,'natural_movie_three_2.mat'));
natural_movie_three_all_warped_frames_2=natural_movie_three_all_warped_frames;
load(fullfile(allenpath,'natural_movie_three_3.mat'));
natural_movie_three_all_warped_frames_3=natural_movie_three_all_warped_frames;
load(fullfile(allenpath,'natural_movie_three_4.mat'));
natural_movie_three_all_warped_frames_4=natural_movie_three_all_warped_frames;
natural_movie_three_all_warped_frames_full=cat(1,natural_movie_three_all_warped_frames_1, natural_movie_three_all_warped_frames_2,natural_movie_three_all_warped_frames_3,natural_movie_three_all_warped_frames_4);
% natural_movie_three_all_warped_frames_full=cat(1,natural_movie_three_all_warped_frames_1, natural_movie_three_all_warped_frames_2);

natural_movie_three_all_warped_frames_full=permute(natural_movie_three_all_warped_frames_full,[2,3,1]);
save('natural_movie_three_full.mat','natural_movie_three_all_warped_frames_full','-v7.3');
clear all
allenpath='C:\Users\sp3660\Documents\Projects\AllenBrainObservatory\HakimMonitor';


% load('locally_sparse_noise_full.mat');
% implay(locally_sparse_noise_all_warped_frames_full)
% load('natural_movie_three_full.mat');
% implay(natural_movie_three_all_warped_frames_full)

% permute(natural_movie_one_all_warped_frames,[])
% natural_movie_one_all_warped_frames_doubled=cat(4, natural_movie_one_all_warped_frames, natural_movie_one_all_warped_frames);
% rtest=permute(natural_movie_one_all_warped_frames_doubled,[4,1,2,3]);
% final=reshape(rtest,[2*900,1024,1280]);

% 
load(fullfile(allenpath,'static_gratings.mat'));
image(all_warped_static_gratings(:,:,3,4,3))
save('static_gratings.mat','all_warped_static_gratings');
X = uint8(all_warped_static_gratings);
image(X(:,:,6,4,3))

load(fullfile(allenpath,['drifting_gratings_1.mat']));
all_warped_drifting_gratings_full=all_warped_driting_gratings;
for i=2:5
    load(fullfile(allenpath,['drifting_gratings_' int2str(i) '.mat']));
    all_warped_drifting_gratings_full = cat(4,all_warped_drifting_gratings_full, all_warped_driting_gratings);

end
save('drifiting_gratings_full.mat','all_warped_drifting_gratings_full','-v7.3');


gartingmean=squeeze(mean(gratings,[1,2]));
flatengratmena=reshape(gartingmean,1,[]);
figure
hist(flatengratmena)

imagemeanmean=squeeze(mean(scenes,[1,2]));
figure
hist(imagemean)

movie1mean=squeeze(mean(naturalmovie1big,[1,2]));
figure
hist(movie1mean)

movie2mean=squeeze(mean(naturalmovie2big,[1,2]));
figure
hist(movie2mean)

movie3mean=squeeze(mean(naturalmovie3big,[1,2]));
figure
hist(movie3mean)

noisemean=squeeze(mean(fullnoise,[1,2]));
figure
hist(noisemean)


driftmean=squeeze(mean(driftinggratings,[1,2,5]));
flatendriftmena=reshape(driftmean,1,[]);
figure
hist(flatendriftmena)




gartingfullmean=mean(gratings,'all');
imagemeanfullmean=mean(scenes,'all');
movie1fullmean=mean(naturalmovie1big,'all');
movie2fullmean=mean(naturalmovie2big,'all');
movie3fullmean=mean(naturalmovie3big,'all');
driftfullmean=mean(driftinggratings,'all');
noisefullmean=mean(fullnoise,'all');
