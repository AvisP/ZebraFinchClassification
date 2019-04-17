Clear;clc;

TS_init('blk12.mat','INP_mops.txt','INP_ops.txt',0,'HCTSA_blk12_demo.mat');
% beVocal 0 or 1 last input

%TS_compute('HCSTA.mat');
% TS_compute(0,[],[],[],'HCTSA_blk12_demo.mat',1);  %this is an alternative
% code to "sample_runscript_matlab"
sample_runscript_matlab(1,5,'HCTSA_blk12_demo.mat');
% first argumnet is parallel processing
% Ssecond argunet batches number, preferably set so that
% modulo(length(timeSeriesData),batch_num) == 0
% this is because, if you have 108 files and the second argument is 5, it
% will ignore the last three (runs in batches of five)

% Inspect quality of unprocessed data
TS_InspectQuality('full','HCTSA_blk12_demo.mat')

% TS_InspectQuality('summary'); [default] Summarizes the proportion of special-valued outputs in each operation as a bar plot, ordered by the proportion of special-valued outputs.
% TS_InspectQuality('master'); Plots which types of special-valued outputs were encountered for each master operation.
% TS_InspectQuality('full'); Plots the full data matrix (all time series as rows and all operations as columns), and shows where each possible special-valued output can occur (including 'error', 'NaN', 'Inf', '-Inf', 'complex', 'empty', or a 'link error').
% TS_InspectQuality('reduced'); As 'full', but includes only columns where special values occurred.

%% Retreiving particular data information
%TimeSeriesIDs = TS_getIDs(theKeyword,'HCTSA_N.mat');

%Or the IDs of operations tagged with the 'entropy' keyword:

%OperationIDs = TS_getIDs('entropy','norm','ops');

%% merging datasets

% TS_combine(HCTSA_1,HCTSA_2,compare_tsids,merge_features,outputFileName)

%% Labelling groups for classification
%Automatically detect group labels

%groupLabels = TS_LabelGroups('HCTSA.mat',{'periodic','stochastic'});
groupLabels = TS_LabelGroups('HCTSA_blk12.mat',{'dir','undir'});

%% Normalizing
%The TS_normalize function writes the new, filtered, normalized matrix to a 
%local file called HCTSA_N.mat

TS_normalize('scaledRobustSigmoid',[0.7,1.0],'HCTSA_blk12.mat')
TS_InspectQuality('full','HCTSA_blk12_N.mat')
%% More plotting after normalization

TS_plot_DataMatrix('HCTSA_blk12_N.mat')

%% Clustering data
distanceMetricRow = 'euclidean'; % time-series feature distance
linkageMethodRow = 'average'; % linkage method
distanceMetricCol = 'corr_fast'; % a (poor) approximation of correlations with NaNs
linkageMethodCol = 'average'; % linkage method

TS_cluster(distanceMetricRow, linkageMethodRow, distanceMetricCol, linkageMethodCol,[0 0],'HCTSA_blk12_N.mat');

% clustering info saved in 'HCTSA_N.mat' file
% to visualize after clustering 
TS_plot_DataMatrix('HCTSA_blk12_N.mat')

%% Low Dimensional Representation
TS_plot_pca('HCTSA_blk12_N.mat','ts');

TS_FeatureSummary(100,'HCTSA_blk12_N.mat');

TS_TopFeatures();

TS_classify('HCTSA_blk12_N.mat','svm_linear','numPCs',10);

%% Nearby clustering of data
TS_SimSearch('whatPlots',{'matrix'}); %THIS FUNCTION DOES NOT WORK

% plot
for i=1:576
    plot(timeSeriesData{1,i});
    title(labels(i));
    pause(0.1)
end

