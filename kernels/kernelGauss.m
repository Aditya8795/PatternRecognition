% The adjustable parameter sigma - c plays a major role in the performance of the kernel, 
% and should be carefully tuned to the problem at hand. If overestimated, 
% the exponential will behave almost linearly and the higher-dimensional projection will
% start to lose its non-linear power. In the other hand, if underestimated, the 
% function will lack regularization and the decision boundary will be highly sensitive 
% to noise in training data - OVERFITTING
function [res] = kernelGauss(X1, X2, c)
	res = exp( -(norm(X1-X2)^2) / (2*c*c) );