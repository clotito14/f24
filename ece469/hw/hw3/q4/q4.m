%{
==========================================
Homework 3 Question 4 Code
Name       : Chase Lotito
University : Southern Illinois University
Course     : ECE469
==========================================
Description:
K-Means Clustering
==========================================
%}

load fisheriris
X = meas(:,3:4);  % extract petal lengths and widths

% (A) X IS 2D, VISUALIZE IN 2D SPACE
figure;
plot(X(:,1), X(:,2), 'k*', 'MarkerSize', 5);
title('Fisher''s Iris Data (Petal)');
xlabel('Petal Lengths (cm)');
ylabel('Petal Widths (cm)');

% (B) RUN K=3 K-MEANS
rng(1);   % reproducibility

% idx: vector of predicted cluster id's
%   C: matrix of centroid locations
[idx, C] = kmeans(X,3);  

% (C) COMPUTE CENTROID DISTANCE BY PASSING C INTO KMEANS
% create grid
x1 = min(X(:,1)):0.01:max(X(:,1));
x2 = min(X(:,2)):0.01:max(X(:,2));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:), x2G(:)];
idx2Region = kmeans(XGrid, 3, MaxIter=1,Start=C);

% (D) VISUALIZE CLUSTERS
figure;
gscatter(XGrid(:,1), XGrid(:,2), idx2Region, [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0], '.'); % vector is colors
hold on;
plot(X(:,1), X(:,2), 'k*', 'MarkerSize',5);
title('Fisher''s Iris Data (Petal)');
xlabel('Petal Lengths (cm)');
ylabel('Petal Widths (cm)');
legend('Region 1', 'Region 2', 'Region 3', 'Data', Location='SouthEast');
hold off;

% (E) EXTRACT LENGTH AND WIDTH OF SEPALS AND REPEAT ABOVE
X = meas(:,1:2);  % extract sepal lengths and widths

% idx: vector of predicted cluster id's
%   C: matrix of centroid locations
[idx, C] = kmeans(X,3);  

% (C) COMPUTE CENTROID DISTANCE BY PASSING C INTO KMEANS
% create grid
x1 = min(X(:,1)):0.01:max(X(:,1));
x2 = min(X(:,2)):0.01:max(X(:,2));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:), x2G(:)];
idx2Region = kmeans(XGrid, 3, MaxIter=1,Start=C);

% (D) VISUALIZE CLUSTERS
figure;
gscatter(XGrid(:,1), XGrid(:,2), idx2Region, [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0], '.'); % vector is colors
hold on;
plot(X(:,1), X(:,2), 'k*', 'MarkerSize',5);
title('Fisher''s Iris Data (Sepal)');
xlabel('Sepal Lengths (cm)');
ylabel('Sepal Widths (cm)');
legend('Region 1', 'Region 2', 'Region 3', 'Data', Location='SouthEast');
hold off;