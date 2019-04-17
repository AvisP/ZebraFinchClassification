clear;clc

% cd('F:\data_for_avishek\blkorng_new')
% cd('F:\data_for_avishek\bluwht4-38')
% cd('F:\data_for_avishek\o13p14')
% cd('F:\data_for_avishek\orngpnk23-24_stim');
% cd('F:\data_for_avishek\yelred');
% cd('F:\data_for_avishek\blk17');
% cd('F:\data_for_avishek\Dir_Undir\done\blu50blu58');
% cd('F:\data_for_avishek\Dir_Undir\done\prpred');
% cd('F:\data_for_avishek\Dir_Undir\done\red98orng15');
cd('F:\data_for_avishek\Dir_Undir\done\o102p102');
% cd('F:\data_for_avishek\Dir_Undir\done\o121p122');
% cd('F:\data_for_avishek\Dir_Undir\done\o122p123');

% dirf('*.wav','batch.txt');
fileID = fopen('motifbatch.txt','r');
list = textscan(fileID,'%s \n');
fclose(fileID);

temp_list = list{1,1};
C = cellfun(@(s)strfind(temp_list,s),{'undir'},'UniformOutput',false);
sorted_list_idx = ~cellfun('isempty',vertcat(C{:}));

for i=1:length(temp_list)
    
    [data,~] = audioread(char(temp_list(i)));
    timeSeriesData(i) = {data};
%     labels(i) = {temp_list(i)};
  %  keywords(i) = {'blk12'};
    if(sorted_list_idx(i))
        keywords(i) = {'undir'};
    else
        keywords(i) = {'dir'};
    end
end
labels = temp_list';

TS_init('o122p123.mat','INP_mops.txt','INP_ops.txt',0,'HCTSA_o122p123.mat');
sample_runscript_matlab(1,1,'HCTSA_o122p123.mat');

TS_init('o121p122.mat','INP_mops.txt','INP_ops.txt',0,'HCTSA_o121p122.mat');
sample_runscript_matlab(1,1,'HCTSA_o121p122.mat');

TS_init('o102p102.mat','INP_mops.txt','INP_ops.txt',0,'HCTSA_o102p102.mat');
sample_runscript_matlab(1,1,'HCTSA_o102p102.mat');
TS_LabelGroups('HCTSA_blublu50-58.mat',{'dir','undir'});
TS_normalize('scaledRobustSigmoid',[0.7,1.0],'HCTSA_blublu50-58.mat')

TS_normalize('scaledRobustSigmoid',[0.7,1.0],'HCTSA_o102p102.mat')

TS_LabelGroups('HCTSA_blk17.mat',{'dir','undir'});
TS_normalize('scaledRobustSigmoid',[0.7,1.0],'HCTSA_blk17.mat')

TS_LabelGroups('HCTSA_o121p122.mat',{'dir','undir'});
TS_normalize('scaledRobustSigmoid',[0.7,1.0],'HCTSA_o121p122.mat')

TS_LabelGroups('HCTSA_o122p123.mat',{'dir','undir'});
TS_normalize('scaledRobustSigmoid',[0.7,1.0],'HCTSA_o122p123.mat')

TS_LabelGroups('HCTSA_prpred.mat',{'dir','undir'});
TS_normalize('scaledRobustSigmoid',[0.7,1.0],'HCTSA_prpred.mat')

TS_LabelGroups('HCTSA_red98orng15.mat',{'dir','undir'});
TS_normalize('scaledRobustSigmoid',[0.7,1.0],'HCTSA_red98orng15.mat')


% timeSeriesTotal = {};
% keywordsTotal = {};
% 
% for j=1:length(temp_list)
% win_size = 20000;start_pos = 1;
% temp = timeSeriesData{j};kk=1;
% 
% for i=1:floor(length(temp)/win_size)
%     timeSeriesData2(kk) = {temp(start_pos+win_size*(i-1):start_pos+win_size*i)};
%     labels(kk) = {strcat(temp_list{1},'_',num2str(kk))};
%     keywords2(kk) = keywords(j);
%     kk=kk+1;
% end
% 
% timeSeriesData2(kk) = {temp(win_size*(i):end)};
% labels(kk) = {strcat(temp_list{1},'_',num2str(kk))};
% keywords2(kk) = keywords(j);
% 
% start_pos = 1+win_size/2;kk=kk+1;
% timeSeriesData2(kk) = {temp(1:start_pos)};
% labels(kk) = {strcat(temp_list{1},'_',num2str(kk))};
% keywords2(kk) = keywords(j);
% kk=kk+1;
% 
% for i=1:floor((length(temp)-start_pos)/win_size)
%     timeSeriesData2(kk) = {temp(start_pos+win_size*(i-1):start_pos+win_size*i)};
%     labels(kk) = {strcat(temp_list{1},'_',num2str(kk))};
%     keywords2(kk) = keywords(j);
%     kk=kk+1;
% end
% timeSeriesData2(kk) = {temp(win_size*(i)+start_pos:end)};
% labels(kk) = {strcat(temp_list{1},'_',num2str(kk))};
% keywords2(kk) = keywords(j);
% 
% timeSeriesTotal = [timeSeriesTotal timeSeriesData2];
% keywordsTotal = [keywordsTotal keywords2];
% 
% clear temp timeSeriesData2
% end


