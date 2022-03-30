%% signlas analysise
clear all
% signalsfile='C:\\Users\\sp3660\\Documents\\Github\\LabNY\\MainClasses\\Visual Stimulation\\testdoublevoltageparadigm-000\\testdoublevoltageparadigm-000_Cycle00001_VoltageRecording_001.csv';
% vistimfile='C:\\Users\\sp3660\\Documents\\Github\\LabNY\\MainClasses\\Visual Stimulation\\AllenSessionA_7_1_21_stim_data_20h_10m.mat.mat';

signalsfile='C:\\Users\\sp3660\\Desktop\\TemporaryProcessing\\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000\\210702_SPJA_FOV1_3planeAllenA_920_50024_narrow_without-000_Cycle00001_VoltageRecording_001.csv'


% load(vistimfile)
T = readtable(signalsfile);

%%

vistim=T.('VisStim');
LED=T.('LED');
PhotoDiode=T.('PhotoDiode');
Locomotion=T.('Locomotion');
clear T
%%
figure(1)
tiledlayout(4,1)
ax1 = nexttile
plot(vistim)
ax2 = nexttile;
plot(PhotoDiode)
ax3 = nexttile;
plot(LED)
ax4 = nexttile;
plot(Locomotion)
linkaxes([ax1 ax2 ax3 ax4],'x')

%%
t = 0:1/1000:(length(vistim)-1)/1000; % create a time vector
figure(2)
tiledlayout(4,1)
ax1 = nexttile
plot(t,vistim)
ax2 = nexttile;
plot(t,PhotoDiode)
ax3 = nexttile;
plot(t,LED)
ax4 = nexttile;
plot(t,Locomotion)
linkaxes([ax1 ax2 ax3 ax4],'x')

%% get paradigm fragments
figure()
plot(vistim)
indexup=find(vistim>6.5);
tocorrectup=find(diff(indexup)<20);
indexup(tocorrectup)=[];

%%

driftinggrats1=vistim(indexup(1):indexup(2));
subst=diff(driftinggrats1);
allstimdown=find(subst<-0.05);
laststim=allstimdown(end);
driftinggrats1leftcorrected=driftinggrats1(2:end);
% lastidx=length(subst(length(driftinggrats1leftcorrected)-2000:end))-find(subst(length(driftinggrats1leftcorrected)-2000:end)>0.05)+1;
% finalindexing=length(driftinggrats1leftcorrected)-lastidx;
finalcorrecteddriftinggrats1 =driftinggrats1leftcorrected(1:laststim);
startindex=2;
endindex=laststim;
driftinggrats1photodiode=PhotoDiode(indexup(1):indexup(2));
driftinggrats1leftcorrectedphotodiode=driftinggrats1photodiode(startindex:end);
finalcorrecteddriftinggrats1photodiode=driftinggrats1leftcorrectedphotodiode(1:endindex);


driftinggrats2=vistim(indexup(7):indexup(8));
subst=diff(driftinggrats2);
allstimdown=find(subst<-0.05)
laststim=allstimdown(end)
driftinggrats2leftcorrected=driftinggrats2(2:end);
% lastidx=length(subst(length(driftinggrats2leftcorrected)-2000:end))-find(subst(length(driftinggrats2leftcorrected)-2000:end)>0.05)+1;
% finalindexing=length(driftinggrats2leftcorrected)-lastidx;
finalcorrecteddriftinggrats2 =driftinggrats2leftcorrected(1:laststim);
startindex=2
endindex=laststim


driftinggrats3=vistim(indexup(11):end);
subst=diff(driftinggrats3);
allstimdown=find(subst<-0.05)
laststim=allstimdown(end)
driftinggrats3leftcorrected=driftinggrats3(2:end);
% lastidx=length(subst(length(driftinggrats3leftcorrected)-2000:end))-find(subst(length(driftinggrats3leftcorrected)-2000:end)>0.05)+1;
% finalindexing=length(driftinggrats3leftcorrected)-lastidx;
finalcorrecteddriftinggrats3 =driftinggrats3leftcorrected(1:laststim);
startindex=2
endindex=laststim


movie31=vistim(indexup(3):indexup(4))  ;
subst=diff(movie31);
startindex=find(subst(1:25)>0.05)-2
movie31leftcorrected=movie31(startindex:end);
lastidx=length(subst(length(movie31leftcorrected)-20:end))-find(subst(length(movie31leftcorrected)-20:end)>0.05)+1;
finalindexing=length(movie31leftcorrected)-lastidx;
finalcorrectedmovie31 =movie31leftcorrected(1:finalindexing);
endindex=finalindexing




movie32=vistim(indexup(9):indexup(10)) ;
subst=diff(movie32);
startindex=find(subst(1:25)>0.05)-2
movie32leftcorrected=movie32(startindex:end);
lastidx=length(subst(length(movie32leftcorrected)-20:end))-find(subst(length(movie32leftcorrected)-00:end)>0.05)+1;
finalindexing=length(movie32leftcorrected)-lastidx;
finalcorrectedmovie32 =movie32leftcorrected(1:finalindexing);
endindex=finalindexing

movie1=vistim(indexup(5):indexup(6));
subst=diff(movie1);
startindex=find(subst(1:25)>0.05)-2
movie1leftcorrected=movie1(startindex:end);
lastidx=length(subst(length(movie1leftcorrected)-20:end))-find(subst(length(movie1leftcorrected)-20:end)>0.05)+1;
finalindexing=length(movie1leftcorrected)-lastidx;
finalcorrectedmovie1 =movie1leftcorrected(1:finalindexing);
endindex=finalindexing

spontaneous=vistim(indexup(8):indexup(9));
subst=diff(spontaneous);
startindex=find(subst(1:25)>0.05)-1
spontaneousleftcorrected=spontaneous(fstartindex:end);
lastidx=length(subst(length(spontaneous)-20:end))-find(subst(length(spontaneous)-20:end)>0.05)+1;
finalindexing=length(spontaneous)-lastidx;
finalcorrectedspontaneous =spontaneous(1:finalindexing);
endindex=finalindexing

gratings={driftinggrats1leftcorrected driftinggrats2leftcorrected   driftinggrats3leftcorrected};
movies={ movie31leftcorrected movie1leftcorrected movie32leftcorrected };




%%
plot(frames100,movie32)
plot(diff(movie32))

frames100=0:500/(length(frames)-1):500;


peaks=1+find(diff(movie32)>0.005);

plot(diff(peaks))

frames=0:60/1000:(length(movie32)-1)/(1/60)/1000;
t = 0:1/1000:(length(movie32)-1)/1000; % create a time vector


plot(movie32(find(diff(movie32)>0.1)))
length(find(diff(movie31)>0.05))/5

    tops= find(abs(movie31-0.5)<1e-2);  



plot(vistim,'-','Marker','o','MarkerEdgeColor' ,'red', 'MarkerIndices',indexup )



%%

%%
gratins1=finalcorrecteddriftinggrats1;
bottoms=cell(5,41);
tops=cell(5,41);
for angle=1:41
   if angle<41
    tops{1,angle}= find(abs(gratins1-angle*4/40)<1e-2);  
   else 
    tops{1,angle}= find(abs(gratins1-4.5)<1e-2);  
   end
   allbottoms=find(abs(gratins1)<1e-2) ;
       
       
   idxxx=find(diff(tops{1,angle})>1);
   transitionsend=tops{1,angle}(idxxx);
   transitionstart=tops{1,angle}(idxxx+1);
   addedextremes=[tops{1,angle}(1); transitionsend; transitionstart; tops{1,angle}(end)];
   stimulustart=[tops{1,angle}(1); transitionstart];
   stimulusend=[transitionsend; tops{1,angle}(end)];
   tops{2,angle}=stimulustart;
   tops{3,angle}=stimulusend;
   tops{4,angle}=stimulusend-stimulustart;
   
   
   stratindex=find(diff(allbottoms)>1)+1;  
   allisimstartidx=allbottoms(stratindex);
   allisimstartidx(end+1)=1;
   endidx=find(diff(allbottoms)>1);  
   allisimendidx=allbottoms(endidx);
   
   isimstart=zeros(1,numel(stimulustart));
   isimend=zeros(1,numel(stimulustart));

   for grat=1:length(stimulustart)
      dirtyisimstart=stimulustart(grat)-1200;
      if dirtyisimstart<0
          dirtyisimstart=1;
      end
      dirtyisimstart:stimulustart(grat)-1;
      isimstart(grat)=intersect(allisimstartidx,[dirtyisimstart:stimulustart(grat)-1]);
      isimend(grat)=intersect(allisimendidx,[dirtyisimstart:stimulustart(grat)-1]);
      bottom=[gratins1(isimstart(grat):isimend(grat))];
      bottoms{1,angle}=[bottoms{1,angle}; bottom ];
   end
   
   bottoms{2,angle}=isimstart;
   bottoms{3,angle}=isimend;
   bottoms{4,angle}=isimend-isimstart;
end

colors=[ [1 0 0] ;[0 1 0]; [0 0 1]; [1 1 0] ;[0 1 1]    ];
% figure(1)
%     hold on 
% for ang=1:5
%     plot(gratins1,'-','Marker','o','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[bottoms{2,1+((ang-1)*8)}   bottoms{3,1+((ang-1)*8)} ] )
%     plot(gratins1,'-','Marker','+','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[bottoms{2,2+((ang-1)*8)}   bottoms{3,2+((ang-1)*8)} ] )
%     plot(gratins1,'-','Marker','*','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[bottoms{2,3+((ang-1)*8)}   bottoms{3,3+((ang-1)*8)} ] )
%     plot(gratins1,'-','Marker','x','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[bottoms{2,4+((ang-1)*8)}   bottoms{3,4+((ang-1)*8)} ] )
%     plot(gratins1,'-','Marker','^','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[bottoms{2,5+((ang-1)*8)}   bottoms{3,5+((ang-1)*8)} ] )
%     plot(gratins1,'-','Marker','>','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[bottoms{2,6+((ang-1)*8)}   bottoms{3,6+((ang-1)*8)} ] )
%     plot(gratins1,'-','Marker','<','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[bottoms{2,7+((ang-1)*8)}   bottoms{3,7+((ang-1)*8)} ] )
%     plot(gratins1,'-','Marker','d','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[bottoms{2,8+((ang-1)*8)}   bottoms{3,8+((ang-1)*8)} ] )
% 
% end
% plot(gratins1,'-','Marker','s','MarkerEdgeColor' ,[0 0 0], 'MarkerIndices',[bottoms{2,41}   bottoms{3,41}] )

figure(2)
    hold on 
for ang=1:5
    plot(gratins1,'-','Marker','o','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[tops{2,1+((ang-1)*8)}  ; tops{3,1+((ang-1)*8)} ] )
    plot(gratins1,'-','Marker','+','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[tops{2,2+((ang-1)*8)} ;  tops{3,2+((ang-1)*8)} ] )
    plot(gratins1,'-','Marker','*','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[tops{2,3+((ang-1)*8)} ;  tops{3,3+((ang-1)*8)} ] )
    plot(gratins1,'-','Marker','x','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[tops{2,4+((ang-1)*8)} ;  tops{3,4+((ang-1)*8)} ] )
    plot(gratins1,'-','Marker','^','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[tops{2,5+((ang-1)*8)} ;  tops{3,5+((ang-1)*8)} ] )
    plot(gratins1,'-','Marker','>','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[tops{2,6+((ang-1)*8)} ;  tops{3,6+((ang-1)*8)} ] )
    plot(gratins1,'-','Marker','<','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[tops{2,7+((ang-1)*8)}  ; tops{3,7+((ang-1)*8)} ] )
    plot(gratins1,'-','Marker','d','MarkerEdgeColor' ,colors(ang,:), 'MarkerIndices',[tops{2,8+((ang-1)*8)} ;  tops{3,8+((ang-1)*8)} ] )

end
plot(gratins1,'-','Marker','s','MarkerEdgeColor' ,[0 0 0], 'MarkerIndices', tops{1,angle})

%%
for angle=1:41
    tops{5,angle}=zeros(2,length(tops{3,angle}));
    for rep=1:length(tops{3,angle})      
        stimend=tops{3,angle}(rep);
        nextstimstart=stimend+1  ;   
        for ang2=1:41
            isnextstim=intersect(nextstimstart, bottoms{2,ang2});
            if isnextstim  
                repetition=find(bottoms{2,ang2}==isnextstim);
                tops{5,angle}(:,rep)= [ang2; repetition];
           
            end     
       end
    end
end
%%

crosscor=cell(1,40)
close all
for ang=1:41
figure()
hold on
crosscor{ang}=zeros(2,length(bottoms{2,ang}))
    for i=1:length(bottoms{2,ang})
        crosscor{ang}(i)
        sitms=[bottoms{2,ang}(i): tops{3,ang}(i)];
        plot(leftprunedphotodiode(sitms,1));
    end
    hold off
end
%%
ang=8;
i=1;
sitm1=[bottoms{2,ang}(i): tops{3,ang}(i)];
sitm2=[bottoms{2,ang}(i+1): tops{3,ang}(i+1)];

[r,lags] = xcorr(leftprunedphotodiode(sitm1,1),leftprunedphotodiode(sitm2,1));
stem(lags,r);



plot(pwelch(leftprunedphotodiode(sitm1,1)))
xlim([0 18])
ylim([0 3])

%%




S=leftprunedphotodiode(sitm1,1)


S=c
L=length(S)
Y=fft(S)
Fs=1000
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
%% camera



c=[]
%%
plot(c)
plot(pwelch(c))
