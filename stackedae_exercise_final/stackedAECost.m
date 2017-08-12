function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

% You might find these variables useful
numCases= size(data, 2);
groundTruth = full(sparse(labels, 1:numCases, 1));

% Autoencoder
z2 = stack{1}.w * data + repmat(stack{1}.b, 1, size(data,2));
a2 = sigmoid(z2);
z3 = stack{2}.w * a2 + repmat(stack{2}.b, 1, size(a2,2));
a3 = sigmoid(z3);

% Softmax
M = softmaxTheta * a3;
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
M = bsxfun(@rdivide, M, sum(M));

% Cost
cost = - sum(sum(groundTruth .* log(M))) / numCases + lambda/2 * sum(sum(softmaxTheta.^2));

% Gradient
delta4 = -(groundTruth - M);
delta3 = softmaxTheta'*delta4 .*sigmoid_d(z3);
delta2 = stack{2}.w'*delta3 .*sigmoid_d(z2);

softmaxThetaGrad = delta4 * a3'/ numCases + lambda * softmaxTheta;
stackgrad{2}.w = delta3*a2'/numCases;
stackgrad{2}.b = sum(delta3,2)/numCases;
stackgrad{1}.w = delta2*data'/numCases;
stackgrad{1}.b = sum(delta2,2)/numCases;

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
function sigm_d = sigmoid_d(x)
    sigm = sigmoid(x);
    sigm_d = sigm.*(1-sigm);
end
