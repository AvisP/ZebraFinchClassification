clear;clc;
% load HCTSA_yelred_N.mat
load HCTSA_combined12_N.mat
for i=1:length(TimeSeries)
    y(i) = TimeSeries(i).Group;
end
TS_DataMat_table = array2table([TS_DataMat y']);

Blk17 = fitcensemble(TS_DataMat,y','Bag','classification',100,'Tree','KFold',10);
Combined_boost = fitensemble(TS_DataMat,y','AdaBoostM1',100,'Tree','KFold',10);
O102P102_boost = fitensemble(TS_DataMat,y','AdaBoostM1',100,'Tree')%,'KFold',10);

act_label = y';
[pred_label,score_label] =  kfoldPredict(BLK12_boost);
[pred_label,score_label] =  kfoldPredict(BaggedTrees_16Dataset);
[pred_label,score_label] =  kfoldPredict(Combined_boost);
or
pred_label,score_label = predict(Bluwht_4_38_boost,TS_DataMat);

C = confusionmat(act_label,pred_label)

C = confusionmat(act_label,pred_label)

acc = (C(1,1)+C(2,2))/(C(1,1)+C(1,2)+C(2,1)+C(2,2))


rng default
Mdl = fitcensemble(TS_DataMat,y','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
pred_label2 = predict(Mdl,TS_DataMat);
C = confusionmat(act_label,pred_label2)

acc = (C(1,1)+C(2,2))/(C(1,1)+C(1,2)+C(2,1)+C(2,2))


figure;
subplot(1,2,1);plotConfMat(C_SVM,{'dir','undir'});
subplot(1,2,2);plotConfMat(C_RndFrst,{'dir','undir'});

for i=1:Mdl.NumTrained
   ypred_individual(:,i) = predict(Mdl.Trained{i,1},Mdl.X)'; 
end
ytrain_act = Yvals(1:1017)';

ycount = zeros(1017,1);
for i=1:Mdl.NumTrained
    for j=1:1017
        if ypred_individual(j,i)==ytrain_act(j)
            ycount(j) = ycount(j)+1;
        end
    end
end

indx = find(ytrain_act==2);
ycount_undir = ycount(indx);
[B,I] = sort(ycount_undir,'ascend');
ycount_undir_index_sorted = indx(I);

X_train_undir_dataeq = X_train(ycount_undir_index_sorted(1:87),:);
X_train_dataeq = [X_train; X_train_undir_dataeq];

indx_dir = find(ytrain_act==1);
indx_undir = find(ytrain_act==2);
ycount_dir = ycount(indx_dir);
[B,I] = sort(ycount_dir,'ascend');
ycount_dir_index_sorted = indx(I);

X_train_dir = X_train(indx_dir,:);
X_train_undir = X_train(indx_undir,:);
X_train_dir_dataeq = X_train(ycount_undir_index_sorted(1:378),:);
X_train_dataeq2 = [X_train_dir;X_train_dir_dataeq;X_train_undir;X_train_undir];