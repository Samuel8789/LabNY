%Verify Shepard Tones

% Par = mol_par_ch_det_psy;


Par.vecCenterFreq               = 8000:200:15800; %Center frequency with octaves below and above as well
Par.ShepardTones                = 9; %Number of partial tones above and below center freq also present: Needs to be odd!
Par.ShepardWeights              = gausswin(length(Par.vecCenterFreq) * Par.ShepardTones); %Vector of weights for each tone

%% Show the weights that are attributed to the partial tones:

% freq = 8000;
% allfreqs = 8000;

tones = Par.ShepardTones+1;
i=(1:tones-1)'; %All partial tones
partialtones = [];
for freq = Par.vecCenterFreq
    partialtones            = [partialtones freq * 2.^(i-tones/2)];
end

window                  = Par.ShepardWeights(find(Par.vecCenterFreq == freq):length(Par.vecCenterFreq):Par.ShepardTones*length(Par.vecCenterFreq));

%% Figure

alltonessorted = sort(partialtones(:));
close all;

for freq = [8000 10000 12000 14000 15800]
% for freq = [8000]

figure;

%All possible tones and their weights:
allbars = bar(1:length(Par.ShepardWeights),Par.ShepardWeights,0.5); hold on;
set(gcf,'units','normalized','Position',[0 0 1 1],'color','w')
set(allbars,'FaceColor',[0.6 0.6 0.6]);

%All partial tones and their weights:
partialtones            = freq * 2.^(i-tones/2);
idx                     = find(Par.vecCenterFreq == freq):length(Par.vecCenterFreq):Par.ShepardTones*length(Par.vecCenterFreq);
window                  = Par.ShepardWeights(idx);
barselec                = bar(idx,window,0.05);
set(barselec,'FaceColor',[(freq-8000)/8000 0.4 1-(freq-8000)/8000]);

%Base tone and its weight:
partialtones            = freq * 2.^(i-tones/2);
idx                     = find(Par.vecCenterFreq == freq):length(Par.vecCenterFreq):Par.ShepardTones*length(Par.vecCenterFreq);
window                  = Par.ShepardWeights(idx(5));
singlebar               = bar(idx(5),window,2);
set(singlebar,'FaceColor',[(freq-8000)/8000 1 1-(freq-8000)/8000]);
% set(gca,'XTick',idx,'XTickLabel',(num2cell(partialtones))','xlim',[1 length(Par.ShepardWeights)],'FontSize',12)

set(gca,'XTick',idx,'XTickLabel',(num2cell(partialtones))','xlim',[1 length(Par.ShepardWeights)],'FontSize',12)
title(sprintf('Partial tones and weights at %d Hz',freq),'FontSize',20)
xlabel('Frequency (Hz)','FontSize',15)
ylabel('Weight (au)','FontSize',15)

folder = 'E:\Documents\PhD\Presentations\SupportingFigures\TonesWeights';
saveas(gcf,sprintf('%s%d',folder,freq),'jpg')
close;

end


