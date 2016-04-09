% The Polynomial kernel is a non-stationary kernel. Polynomial kernels are well suited for problems 
% where all the training data is normalized.
% Adjustable parameters are the slope alpha, the constant term c and the polynomial degree d.
function [res] = kernelPoly(X1, X2, a, c, d)
	res = (a*X1*X2' + c) ^ d;