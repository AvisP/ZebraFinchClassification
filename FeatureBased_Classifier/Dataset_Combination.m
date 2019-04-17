clear;clc;

TS_LabelGroups('HCTSA_blk12.mat',{'dir','undir'});
TS_LabelGroups('HCTSA_blkorng_new.mat',{'dir','undir'});
TS_LabelGroups('HCTSA_bluwht4-38.mat',{'dir','undir'});
TS_LabelGroups('HCTSA_o13p14.mat',{'dir','undir'});
TS_LabelGroups('HCTSA_yelred.mat',{'dir','undir'});
TS_LabelGroups('HCTSA_orngpnk23-24_stim.mat',{'dir','undir'});

TS_combine('HCTSA_blk12.mat','HCTSA_blk17.mat',0,0,'HCTSA_combined1.mat')
TS_combine('HCTSA_combined1.mat','HCTSA_blkorng_new.mat',0,0,'HCTSA_combined2.mat')
TS_combine('HCTSA_combined2.mat','HCTSA_bluwht4-38.mat',0,0,'HCTSA_combined3.mat')
TS_combine('HCTSA_combined3.mat','HCTSA_o13p14.mat',0,0,'HCTSA_combined4.mat')
TS_combine('HCTSA_combined4.mat','HCTSA_yelred.mat',0,0,'HCTSA_combined5.mat')
TS_combine('HCTSA_combined5.mat','HCTSA_orngpnk23-24_stim.mat',0,0,'HCTSA_combined6.mat')
TS_combine('HCTSA_combined6.mat','HCTSA_blublu50-58.mat',0,0,'HCTSA_combined7.mat')
TS_combine('HCTSA_combined7.mat','HCTSA_o121p122.mat',0,0,'HCTSA_combined8.mat')
TS_combine('HCTSA_combined8.mat','HCTSA_o122p123.mat',0,0,'HCTSA_combined9.mat')
TS_combine('HCTSA_combined9.mat','HCTSA_red98orng15.mat',0,0,'HCTSA_combined10.mat')
TS_combine('HCTSA_combined10.mat','HCTSA_prpred.mat',0,0,'HCTSA_combined11.mat')

%Run this line
TS_combine('HCTSA_combined11.mat','HCTSA_o102p102.mat',0,0,'HCTSA_combined12.mat')


TS_LabelGroups('HCTSA_combined12.mat',{'dir','undir'});
