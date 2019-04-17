clear;clc;

groupLabels = TS_LabelGroups('HCTSA_orngpnk23-24_stim_split_1000sample2.mat',{'dir','undir'});
TS_normalize('scaledRobustSigmoid',[0.7,1.0],'HCTSA_orngpnk23-24_stim_split_1000sample2.mat')

TS_normalize('scaledRobustSigmoid',[0.7,1.0],'HCTSA_combined12.mat')

%[foldlosses,RL,PL] = TS_classify_label('HCTSA_blk12_split_N.mat','svm_linear','numPCs',10);
[foldlosses,RL,PL,CVMdl] = TS_classify_label('HCTSA_orngpnk23-24_stim_split_1000sample2_N.mat','svm_linear','numPCs',10);

[foldlosses,RL,PL,CVMdl] = TS_classify_label('HCTSA_o102p102_N.mat','svm_linear','numPCs',10);
[foldlosses,RL,PL,CVMdl] = TS_classify_label('HCTSA_combined12_N.mat','svm_linear','numPCs',10);
[foldlosses,RL,PL,CVMdl] = TS_classify_label('HCTSA_blk12_N.mat','svm_linear','numPCs',10);
[foldlosses,RL,PL,CVMdl] = TS_classify_label('HCTSA_red98orng15_N.mat','svm_linear','numPCs',10);
[ScoreCVSVMModel,ScoreParameters] = fitSVMPosterior(CVMdl{1,1});
[~,OOSPostProbs] = kfoldPredict(ScoreCVSVMModel);

kk=1;
for k=1:length(RL)
   if ~isequal(RL(:,k),PL(:,k))
      Pred_idx(kk) =  k;
      kk=kk+1;
   end
end

% load('blk12_split.mat','timeSeriesData','sample_val_Total');
% load('HCTSA_blk12_split_N.mat','TimeSeries');

load('orngpnk23-24_stim_split_1000sample2.mat','timeSeriesData','sample_val_Total');
load('HCTSA_orngpnk23-24_stim_split_1000sample2.mat','TimeSeries');

bins = 50;
% cmap = summer(bins);
cmap = cool(bins);

lastID = 0;
for i=1:length(sample_val_Total)
    
    if ~isequal(sample_val_Total(i).ID,lastID)
        figure;
        lastID = sample_val_Total(i).ID;
    end
    t = linspace(sample_val_Total(i).Beg,sample_val_Total(i).Last,(sample_val_Total(i).Last - sample_val_Total(i).Beg+1));
     cidx = ceil(bins*OOSPostProbs(i,2));
     
    if isempty(find(Pred_idx==i, 1)) % Detection for misclassified segment is in Pred_idex or not
        % Loop is for correctly predicted segments
        if sample_val_Total(i).Rept == 1
           subplot(2,1,1);hold on;
        elseif sample_val_Total(i).Rept == 2
            subplot(2,1,2);hold on;
        end
       plot(t,timeSeriesData{1,i},'color',cmap(cidx,:));
       plot([sample_val_Total(i).Beg sample_val_Total(i).Beg],[-1 1],'k--'); % For vertical lines at beginning and end of each plot
       %plot([sample_val_Total(i).Last sample_val_Total(i).Last],[-1 1],'k');
        % Loop for incorrectly predicted segments
    else
        if sample_val_Total(i).Rept == 1
           subplot(2,1,1);hold on;
        elseif sample_val_Total(i).Rept == 2
           subplot(2,1,2);hold on;
        end
      % patch([sample_val_Total(i).Beg sample_val_Total(i).Last sample_val_Total(i).Last sample_val_Total(i).Beg],[1 1 -1 -1 ],'m');
       patch([sample_val_Total(i).Beg sample_val_Total(i).Last sample_val_Total(i).Last sample_val_Total(i).Beg],[1 1 -1 -1 ],[0 0 0 0],'FaceColor',[211/255 211/255 211/255],'EdgeColor','none');
       plot(t,timeSeriesData{1,i},'color',cmap(cidx,:));
       plot([sample_val_Total(i).Beg sample_val_Total(i).Beg],[-1 1],'k--');% For vertical lines at beginning and end of each plot
      
    end
    
    title(TimeSeries(i).Name(1:(strfind(TimeSeries(i).Name,'.')-1))); 
 end