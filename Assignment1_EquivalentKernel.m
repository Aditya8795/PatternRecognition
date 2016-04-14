load('ASSIGNMENT1.mat');

% Training DataSet - 7,500 data samples
inputTrain = DATA(1:7500,1:2); % each row is a 1*2 input vector
outputTrain = DATA(1:7500,3:4); % each row is a 1*2 target vector

% Testing DataSet
inputTest = DATA(7501:10000,1:2);
outputTest = DATA(7501:10000,3:4);

r = 0.1; % Resolution
numberOfDecimals = 1; % needed when we truncate the input

% The total number of inputs (as input is 1*2) and normalized to [0,1]
M = int16( ((1/r)+1)*((1/r)+1) ); 
N = 7500; % 75% of the total 10,000 samples are used as the training set.
modelInputs = zeros(M,2); % Preallocate for speed
%modelInputs = []; Slower way

NormalisedInputs = 0:r:1;

for i = 1:1:length(NormalisedInputs)
    for j = 1:1:length(NormalisedInputs)
        %modelInputs = [modelInputs; [NormalisedInputs(i),NormalisedInputs(j)]];
        modelInputs(j+length(NormalisedInputs)*(i-1),1:2) = [NormalisedInputs(i),NormalisedInputs(j)];
    end
end

% modelInputs has M rows (number of inputs the Model has readymade
% predictions for. M = [(1/r)+1]^2
% each row is a 1*2 input vector.

K_gauss = zeros(M,N);
K_sigmoid = zeros(M,N);
gamma = 50;
alpha = 1;
c = 1;

for i = 1:1:M
    for j = 1:1:N
        K_gauss(i,j) = kernelGauss(modelInputs(i,1:2),inputTrain(j,1:2),gamma);
        K_sigmoid(i,j) = kernelHyperTangent(modelInputs(i,1:2),inputTrain(j,1:2),alpha,c);
    end
end

% Out of the N samples for the Pth training sample we can see how the 
% weights corresponding to all M inputs are localized around the 
% Pth training sample's input. TODO get 3D visualization.

% 14 is the main peak => (0.1006,0.2332) close to (0.1,0.2)
P = 98;  
plot(K_gauss(1:M,P));

% This vector's ith element is the sum of weights used to predict the 
% output for the ith input vector.
sumOfWeights = zeros([M,1]);
for i = 1:1:M
    sumOfWeights(i) = sum(K_gauss(i,1:N));
end

predictions = K_gauss * outputTrain;
% Normalize it
predictions(1:M,1) = predictions(1:M,1)./ sumOfWeights;
predictions(1:M,2) = predictions(1:M,2)./ sumOfWeights;

% Now we are going to test this model - calculate MSE error
L = 10000 - N; % remmaining samples in the data set are used to test
mse = 0;

for i = 1:1:L
    % in this iteration we wish to find the output for inputTest(i,1:2)
    modelInput = inputTest(i,1:2);
    
    % First we need to find which index this input is closest to
    % in modelInputs
    X1 = round(modelInput(1), numberOfDecimals);
    X2 = round(modelInput(2), numberOfDecimals);
    
    % temp FIX for 0.3 not being found - all others (0.1,0.2..)- no prob
    % Note that this fix is only valid IF resolution is 0.1
    if(X1 == 0.3)
        X1 = modelInputs(34);
    end
    if(X2 == 0.3)
        X2 = modelInputs(34);
    end
    
    index1 = find(modelInputs(1:M,1) == X1);
    index2 = find(modelInputs(1:M,2) == X2);
    index = intersect(index1,index2);
    
    % so the corresponding prediction will be
    modelOutput = predictions(index,1:2);
    actualOuput = outputTest(i,1:2);
    
    squaredError = (modelOutput - actualOuput)*(modelOutput - actualOuput)';
    mse = mse + squaredError;
end

% as its the Mean we gotta divide with number of samples.
mse = mse/L;

