function grad = q2_gradient(Xtrain, Ytrain, theta)
% Compute the gradient of the log likelihood at theta

% INPUT
%  Xtrain  : [m x n] matrix, where each row is a n-dimensional input example (assume it 
%            already contains the constant feature set to 1)
%  Ytrain  : [m x 1] vector, where the i-th element is the correct label
%                    for the i-th example
%  theta   : [n x 1] vector, the current model parameters

% OUTPUT
%  grad    : [n x 1] vector, the gradient of the log likelihood at theta

[m, n] = size(Xtrain);
[pred_Y, prob_Y] = q2_predict(Xtrain, theta);
partial = ones(m, n);

for i = 1:m
    partial(i,:) = (Ytrain(i) - prob_Y(i)) * Xtrain(i, :); % 1 x n
end

grad = sum(partial); % grad: 1 x n
grad = grad'; % now grad is n x 1

end
