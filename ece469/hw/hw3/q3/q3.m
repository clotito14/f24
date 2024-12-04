%{
==========================================
Homework 3 Question 3 Code
Name       : Chase Lotito
University : Southern Illinois University
Course     : ECE469
==========================================
Description:
Design of a support vector machine for an 
object recognition system
==========================================
%}

% Load in dataset from ../svm/dataset2.mat
% Dataset is a struct of X (100x4)-matrix
% and Y (100x1)-vector
dataset2 = load('C:\Users\cloti\OneDrive\Desktop\SCHOOL\SIUC\f24\ece469\hw\hw3\svm\svm\dataset2.mat');

% Extract input features X and output target y
X = dataset2.X;
y = dataset2.Y;  % y = {1, 0}


% visualize dataset via heatmap
figure;
correlationMatrix = corr(X);
heatmap(correlationMatrix, 'Colormap', jet, 'ColorbarVisible', 'on');
title('Correlation Matrix of Features');

% Train SVM Classifier
% ((1) Linear Kernel)
mdl1 = fitcsvm(X, y, 'KernelFunction','linear');
% ((2) Polynomial Kernel)
mdl2 = fitcsvm(X, y, 'KernelFunction','polynomial', 'PolynomialOrder', 2);
% ((3) Guassian Kernel)
mdl3 = fitcsvm(X, y, 'KernelFunction','mysigmoid_2', 'Standardize', true);

% Cross-validate model with 4-fold cross-validation
cvMdl1 = crossval(mdl1, 'KFold', 4);    % Linear 4-fold cross-val
misclas1 = kfoldLoss(cvMdl1);           % calc losses
cvMdl2 = crossval(mdl2, 'KFold', 4);    % Poly  4-fold cross-val
misclas2 = kfoldLoss(cvMdl2);           % calc losses
cvMdl3 = crossvUal(mdl3, 'KFold', 4);    % Guassian  4-fold cross-val
misclas3 = kfoldLoss(cvMdl3);           % calc losses