% Implementation of K-LDA 

load TRAINTEST2D

cluster1 = TRAIN{1,6}{1,1}; % Green
cluster2 = TRAIN{1,6}{1,2}; % Blue
cluster3 = TRAIN{1,6}{1,3}; % Red
cluster4 = TRAIN{1,6}{1,4}; % Cyan

nT = 52; % Total number of data Vectors
n1 = 13; % # vectors in cluster 1
n2 = 13; % # vectors in cluster 2
n3 = 13; % # vectors in cluster 3
n4 = 13; % # vectors in cluster 4

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

% CONSTRUCT KERNEL MATRIX
gamma = 30;

% Arrange all the clusters together
X = [cluster1';cluster2'; cluster3'; cluster4'];
K = ones(nT,nT);

% Construct the Gram Matrix 
% K(i,:) - gives the ith vector out of ALL vectors
for i = 1:1:nT
    for j = 1:1:nT
        K(i,j) = kernelGauss(X(i,:),X(j,:),gamma);
    end
end

% Okay K(i,j) = K(x_i,x_j) Now as X(1:13,:), X(14:26,:), X(27:39,:), 
% X(40:52,:) are the 4 clusters respectively. We can calculate M_i as
% the numbers are like that due to how X was constructed.

M_1 = ones(nT,1);
M_2 = ones(nT,1);
M_3 = ones(nT,1);
M_4 = ones(nT,1);
M_star = ones(nT,1);

% sum(K(1:13,j)) is the sum of k(x,x_j) over all x belonging to cluster 1
% sum(K(14:26,j)) is the sum of k(x,x_j) over all x belonging to cluster 2
% sum(K(27:39,j)) is the sum of k(x,x_j) over all x belonging to cluster 3
% sum(K(40:52,j)) is the sum of k(x,x_j) over all x belonging to cluster 4
% sum(K(:,j)) is the sum of k(x,x_j) over ALL x (all clusters)

for j = 1:1:nT
    M_1(j) = (1/nT)*sum(K(1:13,j));
end
for j = 1:1:nT
    M_2(j) = (1/nT)*sum(K(14:26,j));
end
for j = 1:1:nT
    M_3(j) = (1/nT)*sum(K(27:39,j));
end
for j = 1:1:nT
    M_4(j) = (1/nT)*sum(K(40:52,j));
end

for j = 1:1:nT
    M_star(j) = (1/nT)*sum(K(:,j));
end

% Thus we can construct M
M = n1*(M_1-M_star)*(M_1-M_star)' + n2*(M_2-M_star)*(M_2-M_star)' + n3*(M_3-M_star)*(M_3-M_star)' + n4*(M_4-M_star)*(M_4-M_star)';

% Now we shall construct the kernel matrices for each of the clusters
K_1 = K(:,1:13);
K_2 = K(:,14:26);
K_3 = K(:,27:39);
K_4 = K(:,40:52);

% Thus we can construct N
N = K_1*(eye(n1,n1)-(1/n1)*ones(n1,n1))*K_1' + K_2*(eye(n2,n2)-(1/n2)*ones(n2,n2))*K_2' + K_3*(eye(n3,n3)-(1/n3)*ones(n3,n3))*K_3' + K_4*(eye(n4,n4)-(1/n4)*ones(n4,n4))*K_4';

% Note that in practice, \mathbf{N} is usually singular and 
% so a multiple of the identity is added to it
N = N + 1*eye(nT,nT);
cond(N) % is N ill conditioned? 

% Now that we have both N and M, we can find all the eigenvectors
% and choose most prominent eigenvectors
P = inv(N)*M;
[V,D] = eig(P);

% the first eigen vector has the largest corresponding eigenvalue - 3.07
% after which 0.4, 0.28, 0.26 Then it drops down...
alpha = V(:,1);

% the projected points
y = ones(nT,1);

for i = 1:1:nT
    y(i) = alpha'*K(:,i);
end

projCluster1 = y(1:13);
projCluster2 = y(14:26);
projCluster3 = y(27:39);
projCluster4 = y(40:52);

% Plot the projected data
figure(3);
scatter(projCluster1, zeros(13,1), 'g'); 
hold on;
scatter(projCluster2, zeros(13,1), 'b');
hold on;
scatter(projCluster3, zeros(13,1), 'r');
hold on;
scatter(projCluster4, zeros(13,1), 'c');
hold on;
legend('cluster 1','cluster 2','cluster 3','cluster 4');