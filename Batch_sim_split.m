clear;clc

% cd('F:\data_for_avishek\blkorng_new')
% cd('F:\data_for_avishek\bluwht4-38')
 cd('F:\data_for_avishek\o13p14')
% cd('F:\data_for_avishek\orngpnk23-24_stim');
% cd('F:\data_for_avishek\yelred');

% dirf('*.wav','batch.txt');
fileID = fopen('batchsong.txt','r');
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
% labels = temp_list';

timeSeriesTotal = {};
keywordsTotal = {};
labelsTotal = {};

for j=1:length(temp_list)
win_size = 20000;start_pos = 1;
temp = timeSeriesData{j};kk=1;

for i=1:floor(length(temp)/win_size)
    timeSeriesData2(kk) = {temp(start_pos+win_size*(i-1):start_pos+win_size*i)};
    labels(kk) = {strcat(temp_list{j},'_',num2str(kk))};
    keywords2(kk) = keywords(j);
    kk=kk+1;
end

timeSeriesData2(kk) = {temp(win_size*(i):end)};
labels(kk) = {strcat(temp_list{j},'_',num2str(kk))};
keywords2(kk) = keywords(j);

start_pos = 1+win_size/2;kk=kk+1;
timeSeriesData2(kk) = {temp(1:start_pos)};
labels(kk) = {strcat(temp_list{j},'_',num2str(kk))};
keywords2(kk) = keywords(j);
kk=kk+1;

for i=1:floor((length(temp)-start_pos)/win_size)
    timeSeriesData2(kk) = {temp(start_pos+win_size*(i-1):start_pos+win_size*i)};
    labels(kk) = {strcat(temp_list{j},'_',num2str(kk))};
    keywords2(kk) = keywords(j);
    kk=kk+1;
end
timeSeriesData2(kk) = {temp(win_size*(i)+start_pos:end)};
labels(kk) = {strcat(temp_list{j},'_',num2str(kk))};
keywords2(kk) = keywords(j);

timeSeriesTotal = [timeSeriesTotal timeSeriesData2];
keywordsTotal = [keywordsTotal keywords2];
labelsTotal = [labelsTotal labels];
clear temp timeSeriesData2 keywords2 labels
end

clear timeSeriesData keywords
timeSeriesData = timeSeriesTotal;
labels = labelsTotal;
keywords = keywordsTotal;
clear timeSeriesTotal labelsTotal keywordsTotal

% temp = any(~cellfun('isempty',vertcat(C{:})),1);
% temp1 = ~cellfun('isempty',vertcat(C{:}));