%{
==========================================
Homework 3 Question 1 Code
Name       : Chase Lotito
University : Southern Illinois University
Course     : ECE469
==========================================
Description:
Training a SVM classifiers using a 
Gaussian Kernel based on MATLAB
==========================================
%}

% (A) GENERATE 200 POINTS IN UNIT DISK
rng(1);
r = sqrt(rand(200,1));
theta = 2*pi*rand(200,1);
data1 = [r.*cos(theta), r.*sin(theta)];

% (B) GENERATE 200 POINTS IN ANNULUS (washer)
r2 = sqrt(3*rand(200,1)+1);
theta2 = 2*pi*rand(200,1);
data2 = [r2.*cos(theta2), r2.*sin(theta2)];

% (C) PLOT data1, data2, AND CIRCLES OF RADIUS 1 AND 2
figure;
plot(data1(:,1), data1(:,2), 'r.', 'MarkerSize', 15)
hold on
plot(data2(:,1), data2(:,2), 'b.', 'MarkerSize', 15)
ezpolar(@(x)1);
ezpolar(@(x)2);
axis equal
hold off

% (D) COMBINE data1 AND data2 INTO ONE MATRIX TO GENERATE
%     A VECTOR FOR CLASSIFICATIONS
data3 = [data1; data2];
theclass = ones(400,1); % 400x1 vector full of 1s
theclass(1:200) = -1;   % set first 200 1s to -1s

% (E) TRAIN A SVM CLASSIFIER WITH "KernelFunction" SET TO
%     "rbf" AND BoxConstraint SET TO Inf. PLOT THE DECISION
%      BOUNDARY AND FLAG THE SUPPORT VECTORS

% train svm
cl = fitcsvm(data3, theclass, 'KernelFunction', 'rbf', 'BoxConstraint', 1, 'ClassNames', [-1,1]);

% predict scores over grid
d = 0.02;
[x1Grid, x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:), x2Grid(:)];
[~, scores] = predict(cl, xGrid);

% plot data and decision boundary
figure;
h(1:2) = gscatter(data3(:,1), data3(:,2),theclass,'rb','.');
hold on
ezpolar(@(x)1);
h(3) = plot(data3(cl.IsSupportVector,1), data3(cl.IsSupportVector,2), 'ko');
contour(x1Grid, x2Grid, reshape(scores(:,2), size(x1Grid)), [0 0], 'k');
legend(h, {'-1', '+1', 'Support Vectors'});
axis equal
hold off