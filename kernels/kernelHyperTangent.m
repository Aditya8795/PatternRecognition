% The Hyperbolic Tangent Kernel is also known as the Sigmoid Kernel and as the Multilayer 
% Perceptron (MLP) kernel. The Sigmoid Kernel comes from the Neural Networks field, 
% where the bipolar sigmoid function is often used as an activation function for artificial neurons.

% It is interesting to note that a SVM model using a sigmoid kernel function is equivalent to a 
% two-layer, perceptron neural network. This kernel was quite popular for support vector machines 
% due to its origin from neural network theory. Also, despite being only conditionally positive 
% definite, it has been found to perform well in practice. 

% There are two adjustable parameters in the sigmoid kernel, 
% the slope alpha - a and the intercept constant c. A common value for alpha - a is 1/N, 
% where N is the data dimension.
function [res] = kernelHyperTangent(X1, X2, a, c)
	res = tanh(a*X1*X2' + c);