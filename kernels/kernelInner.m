% The Linear kernel is the simplest kernel function. It is given by the inner product <x,y> plus an optional constant c. Kernel
% algorithms using a linear kernel are often equivalent to their non-kernel counterparts, i.e. KPCA with linear kernel is the
% same as standard PCA. ==> NOTE X1 and X2 are row vectors!! <==
function [res] = kernelInner(X1, X2)
	res = X1 * X2';