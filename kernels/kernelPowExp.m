% : Power Exponential kernel - ES Gopi sir gave this one.
function [res] = kernelPowExp(X1, X2, c)
	res = (exp( -(norm(X1-X2)^2) / (c*c) ))^5;