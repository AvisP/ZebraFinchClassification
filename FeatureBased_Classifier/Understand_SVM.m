clear;clc;

% load fisheriris;
% inds = ~strcmp(species,'setosa');
% X = meas(inds,3:4); y=species(inds);
% SVMModel = fitcsvm(X,y);
% 
% 
% %%% Old solution for getting distances from SVM linear
% shift = svm_struct.ScaleData.shift;
% scale = svm_struct.ScaleData.scaleFactor;
% 
% XnewScaled = ( Xnew - shift ) .* scale;
% 
% f = kfun( sv, XnewScaled, kfunargs{:} )' * alphaHat + bias
% 
% sv = svm_struct.SupportVectors;
% alphaHat = svm_struct.Alpha;
% bias = svm_struct.Bias;
% kfun = svm_struct.KernelFunction;
% kfunargs = svm_struct.KernelFunctionArgs;
% f = kfun(sv,Xnew,kfunargs{:})'*alphaHat(:) + bias;


%%%% Dataset generation
rng(1); % For reproducibility
r = sqrt(rand(100,1)); % Radius
t = 2*pi*rand(100,1);  % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points

r2 = sqrt(3*rand(100,1)+1); % Radius
t2 = 2*pi*rand(100,1);      % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points
% Visualize the points
figure;
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
ezpolar(@(x)1);ezpolar(@(x)2);
title('Actual Data');
axis equal
hold off

% Put data in one matrix
data3 = [data1;data2];
theclass = ones(200,1);
theclass(1:100) = -1;

%Train the SVM Classifier
cl = fitcsvm(data3,theclass,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[-1,1]);

% Predict scores over the grid( NEW Data)
d=0.02;
[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),...
    min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(cl,xGrid);

% Plot the data and the decision boundary
figure;
h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
hold on
ezpolar(@(x)1);
h(3) = plot(data3(cl.IsSupportVector,1),data3(cl.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
axis equal
title('SVM Fitted Data');
% plot(xGrid(9750,1),xGrid(9750,2),'k*');
% plot(xGrid(167,1),xGrid(167,2),'k*');
% plot(xGrid(3557,1),xGrid(3557,2),'k*');
% plot(xGrid(3726,1),xGrid(3726,2),'k*');
% plot(xGrid(14726,1),xGrid(14726,2),'k*');
% plot(xGrid(19900,1),xGrid(19900,2),'k*');
% hold off

[~,PosteriorRegion] = predict(cl,xGrid);

% Convert into cross validated moddel
CVSVMModel = crossval(cl);
% Estimate the optimal score function for mapping observation scores to posterior probabilities of an observation
[ScoreCVSVMModel2,ScoreParameters2] = fitSVMPosterior(CVSVMModel);
% Estimate the out-of-sample positive class posterior probabilities. 
% Data is in ScoreCVSVMModel2.X a
[~,OOSPostProbs] = kfoldPredict(ScoreCVSVMModel2);






% Example of probability computation from scores
% load ionosphere
% 
% rng(1) % For reproducibility
% CVSVMModel = fitcsvm(X,Y,'Holdout',0.2,'Standardize',true,...
%     'ClassNames',{'b','g'});
% ScoreCVSVMModel = fitSVMPosterior(CVSVMModel);
% 
% 
% [~,OOSPostProbs] = kfoldPredict(ScoreCVSVMModel);






