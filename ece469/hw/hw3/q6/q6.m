%{
==========================================
Homework 3 Question 6 Code
Name       : Chase Lotito
University : Southern Illinois University
Course     : ECE469
==========================================
Description:
Image Compression with PCA
==========================================
%}

[RGB map] = imread('peppers.png');
imshow(RGB)
peppers_gray = rgb2gray(RGB);
save peppers_gray;
figure;
imshow(peppers_gray)
peppers_gray = double(peppers_gray); % convert to double precision
axis off, axis equal
X = peppers_gray; %raw data matrix
[m, n] = size(X);
mean_X = mean(X,2); % compute row mean
reformat_mean = repmat(mean_X,1,n);
tilde_X = X - reformat_mean; % subtract row mean to obtain X
covX = tilde_X*tilde_X'/(n-1); %Sample covariance matrix of X
[U,S] = eig(covX);
[Ordered_eigValue,ind] = sort(diag(S),'descend');
U_od = U(:,ind);
variances = (Ordered_eigValue); % compute variances
figure;
bar(variances(1:30)) % plot of variances
%% Extract first 40 principal components
PCs = 477; % Number of principle components used for compression
U_red = U(:,1:PCs);
Z = U_red'*tilde_X; % project data onto PCs
X_hat = U_red*Z; % convert back to original basis
figure;
imshow(X_hat), axis off; % display results