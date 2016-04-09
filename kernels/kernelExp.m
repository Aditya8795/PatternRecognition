% The exponential kernel is closely related to the Gaussian kernel, with only the 
% square of the norm left out. It is also a radial basis function kernel.
function [res] = kernelExp(X1, X2, c)
	res = exp( -(norm(X1-X2)) / (2*c*c) );