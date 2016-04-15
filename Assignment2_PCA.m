load TRAINTEST2D

cluster1 = TRAIN{1,6}{1,1}; % Green
cluster2 = TRAIN{1,6}{1,2}; % Blue
cluster3 = TRAIN{1,6}{1,3}; % Red
cluster4 = TRAIN{1,6}{1,4}; % Cyan

% Plot the data before dimensionality reduction
figure(1);
scatter(cluster1(1,:), cluster1(2,:), 'g'); 
hold on;
scatter(cluster2(1,:), cluster2(2,:), 'b');
hold on;
scatter(cluster3(1,:), cluster3(2,:), 'r');
hold on;
scatter(cluster4(1,:), cluster4(2,:), 'c');
hold on;
legend('cluster 1','cluster 2','cluster 3','cluster 4');

% Centre the data at origin
cluster1 = bsxfun(@minus,cluster1,sum(cluster1,2)/length(cluster1'));
cluster2 = bsxfun(@minus,cluster2,sum(cluster2,2)/length(cluster2'));
cluster3 = bsxfun(@minus,cluster3,sum(cluster3,2)/length(cluster3'));
cluster4 = bsxfun(@minus,cluster4,sum(cluster4,2)/length(cluster4'));

% Plot the data after centering it around the Origin
figure(2);
scatter(cluster1(1,:), cluster1(2,:), 'g'); 
hold on;
scatter(cluster2(1,:), cluster2(2,:), 'b');
hold on;
scatter(cluster3(1,:), cluster3(2,:), 'r');
hold on;
scatter(cluster4(1,:), cluster4(2,:), 'c');
hold on;
legend('cluster 1','cluster 2','cluster 3','cluster 4');

n = 52;
p = 2;
X = [cluster1';cluster2'; cluster3'; cluster4'];


C = (X'*X)/(n-1);
% we get a column vector of eigen values - the first will be the one with
% the largest magnitude
E = svds(C);
% V will have the eigen vectors
[V,D] = eig(C);

% Thus the axis along which variance is Maximum is
maxVarianceAxis = V(:,2);

% No need to divide by (maxVarianceAxis'*maxVarianceAxis) as
% eig generates orthonormal eigenvectors
projectionMatrix = (maxVarianceAxis*maxVarianceAxis');

projCluster1 = projectionMatrix*cluster1;
projCluster2 = projectionMatrix*cluster2;
projCluster3 = projectionMatrix*cluster3;
projCluster4 = projectionMatrix*cluster4;

% Plot the data after projecting onto the principle component
figure(3);
scatter(projCluster1(1,:), projCluster1(2,:), 'g'); 
hold on;
scatter(projCluster2(1,:), projCluster2(2,:), 'b');
hold on;
scatter(projCluster3(1,:), projCluster3(2,:), 'r');
hold on;
scatter(projCluster4(1,:), projCluster4(2,:), 'c');
hold on;
legend('cluster 1','cluster 2','cluster 3','cluster 4');