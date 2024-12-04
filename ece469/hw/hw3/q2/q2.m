%{
==========================================
Homework 3 Question 2 Code
Name       : Chase Lotito
University : Southern Illinois University
Course     : ECE469
==========================================
Description:
Training a SVM classifiers using a custom 
Kernel (Sigmoid kernel) based on MATLAB
==========================================
%}

% (A) GENERATE RANDOM SET OF POINTS IN UNIT CIRCLE.
%     Q1 AND Q3 POINTS POSITIVE CLASS.
%     Q2 AND Q4 POINTS NEGATIVE CLASS.

rng(1);
n = 100;

r1 = sqrt(rand(2*n,1));
t1 = [pi/2*rand(n,1) ; (pi/2*rand(n,1)+pi)];
X1 = [r1.*cos(t1) r1.*sin(t1)];

r2 = sqrt(rand(2*n,1));
t2 = [pi/2*rand(n,1) + pi/2 ; (pi/2*rand(n,1) - pi/2)];
X2 = [r2.*cos(t2) r2.*sin(t2)];

X = [X1 ; X2];
Y = ones(4*n,1);
Y(2*n + 1:end) = -1;

% (B) PLOT DATAPOINTS
figure;
gscatter(X(:,1),X(:,2),Y);
title('Scatter Diagram of Simulated Data');

% (C) USING THE SIGMOID KERNEL. WRITE A TRANSFORMATION
%     FUNCTION WHICH GENERATES THE GRAM MATRIX GIVEN
%     TWO MATRICIES AND THE SIGMOID KERNEL

% DEFINED IN SEPARATE FUNCTION FILEs

% (D) TRAIN SVM CLASSIFIER USING SIGMOID KERNEL
Mdl1 = fitcsvm(X, Y, 'KernelFunction', 'mysigmoid', 'Standardize',true);

% (E) PLOT DATA. IDENTIFY SUPPORT VECTORS AND BOUNDARY
% predict scores over grid
d = 0.02;
[x1Grid, x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:), x2Grid(:)];
[~, scores] = predict(Mdl1, xGrid);

% plot data and decision boundary
figure;
h(1:2) = gscatter(X(:,1), X(:,2),Y);
hold on
ezpolar(@(x)1);
h(3) = plot(X(Mdl1.IsSupportVector,1), X(Mdl1.IsSupportVector,2), 'ko', 'MarkerSize',10);
contour(x1Grid, x2Grid, reshape(scores(:,2), size(x1Grid)), [0 0], 'k');
title('Scatter Diagram with the Decision Boundary');
legend({'-1', '+1', 'Support Vectors'}, 'Location', 'Best');
hold off

% (F) DETERMINE OUT-OF-SAMPLE MISCLASSIFICATION RATE
%     USING 10-FOLD CROSS VALIDATION
CVMdl = crossval(Mdl1);
misclass1 = kfoldLoss(CVMdl1);
misclass1

% (G) WRITE NEW SIGMOID KERNEL W/ gamma=0.5 AND RETRAIN 

% (D) TRAIN SVM CLASSIFIER USING SIGMOID KERNEL
Mdl2 = fitcsvm(X, Y, 'KernelFunction', 'mysigmoid_2', 'Standardize',true);

% (E) PLOT DATA. IDENTIFY SUPPORT VECTORS AND BOUNDARY
% predict scores over grid
d = 0.02;
[x1Grid, x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:), x2Grid(:)];
[~, scores] = predict(Mdl2, xGrid);

% plot data and decision boundary
figure;
h(1:2) = gscatter(X(:,1), X(:,2),Y);
hold on
ezpolar(@(x)1);
h(3) = plot(X(Mdl2.IsSupportVector,1), X(Mdl2.IsSupportVector,2), 'ko', 'MarkerSize',10);
contour(x1Grid, x2Grid, reshape(scores(:,2), size(x1Grid)), [0 0], 'k');
title('Scatter Diagram (Model 2) with the Decision Boundary');
legend({'-1', '+1', 'Support Vectors'}, 'Location', 'Best');

CVMdl2 = crossval(Mdl2);
misclass2 = kfoldLoss(CVMdl2);
misclass2