%{
==========================================
Homework 3 Question 5 Code
Name       : Chase Lotito
University : Southern Illinois University
Course     : ECE469
==========================================
Description:
Principal Component Analysis (PCA)
==========================================
%}

% load hald (cement heat) dataset
load('hald.mat');

% (i) PCA X, X ZERO MEAN (SAMPLE MEAN),
%     COVARIANCE MATRIX, EIGENVECTORS
%     OF COVARIANCE MATRIX

% let ingredients be X (13x4)-matrix
X = ingredients;

% get zero mean for X (sample mean)
meanX = mean(X(:));
X = X - meanX;
zero_meanX = mean(X(:));
disp(['meanX = ', num2str(meanX), ' | zero_meanX = ', num2str(zero_meanX)]);

% (4x4) covariance matrix (4 b/c 4 features)
covX = cov(X);

% find the eigenvectors of covariance matrix
% V: eigenvector column matrix
% D: eigenvalue diagonal matrix
[V,D] = eig(covX);

% sort the eigenvalues largest to smallest (noting location)
[D_sort, idx] = sort(diag(D), 'descend');
V_sort = V(:, idx);   % rearrange eigenvector matrix accordingly


% (ii) use built-in MATLAB PCA function
Mat_pc = pca(ingredients);

disp('MANUAL:');
disp(V_sort);
disp('BUILT-IN:');
disp(Mat_pc);

% Run PCA on the input matrix X
X = [ 2 4 5 5 3 2 ; 2 3 4 5 4 3 ];
k = 1;
n = 6;
d = 2;

pcaX = pca(X);

disp('pca(X)');
disp(pcaX);