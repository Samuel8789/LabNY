folder_path ='C:\Users\sp3660\Desktop\Volumes\20210917_SPGT_Atlas_1050_50024_607_without_210_250z_5z_1ol-010\New folder\20210917_SPGT_Atlas_1050_50024_607_without_210_250z_5z_1ol-010_Cycle00005_Ch1_000005.ome.tif'


d=dir('\\?\C:\Users\sp3660\Desktop\Volumes\20210917_SPGT_Atlas_1050_50024_607_without_210_250z_5z_1ol-010\New folder\*.tif');
im=imread(fullfile(d(1).folder ,d(1).name));
[Ly, Lx, NT] = size(im);
full=zeros(Ly,Lx ,length(d));
for i=1:numel(d) 
  im=imread(fullfile(d(i).folder ,d(i).name));
  full(:,:,i )=im;
end

data=full;
[Ly, Lx, NT] = size(data);
n_frames = NT;
[data2, bidi_phase2] = f_bidi_shift(data, NT, 1);
[path, name, ~] = fileparts(folder_path);
save_file_name = [path '\' name '_shift.tiff'];
% imwrite(data2,save_file_name)

% for ii=1:15
%     figure;
%     imagesc(data(:,:,ii));
%     title('Pre');
%     figure;
%     imagesc(data2(:,:,ii));
%     title('Post');
% end

% figure;
% plot(bidi_phase2);

%%redo on data2
% 
[data3, bidi_phase3] = f_bidi_shift(data2, NT, 1);
[path, name, ~] = fileparts(folder_path);
% save_file_name = [path '\' name '_shift2.tiff'];
% imwrite(data2,save_file_name)
for iii=1:15
    save_file_name=fullfile(d(iii).folder , [d(iii).name '_shift2.tiff']);
    imwrite(data3(:,:,iii),save_file_name)
end
% for ii=1:15
%     figure;
%     imagesc(data(:,:,ii));
%     title('Pre');
%     figure;
%     imagesc(data3(:,:,ii));
%     title('Post-Post');
% end
% 
% figure;
% plot(bidi_phase3);

%%redo on data2



