clear;clc;
load HCTSA_red98orng15_N.mat

for i=1:length(TimeSeries)
    y(i) = TimeSeries(i).Group;
end
TS_DataMat_table = array2table([TS_DataMat y']);

Red98orng15_boost = fitcensemble(TS_DataMat,y','Bag',100,'Tree','KFold',10);

act_label = y';
pred_label =  kfoldPredict(Red98orng15_boost);

C = confusionmat(act_label,pred_label)

acc = (C(1,1)+C(2,2))/(C(1,1)+C(1,2)+C(2,1)+C(2,2))


rng default
Mdl = fitcensemble(TS_DataMat,y','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
pred_label2 = predict(Mdl,TS_DataMat);
pred_label2 = predict(Red98orng15_Mdl_opt,TS_DataMat);
C = confusionmat(act_label,pred_label2)

acc = (C(1,1)+C(2,2))/(C(1,1)+C(1,2)+C(2,1)+C(2,2))