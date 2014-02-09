function lik = q2_loglik(Xtrain, Ytrain, theta)
% Computes the log likelihood value for training data (Xtrain, Ytrain) and parameter theta

% INPUT
%  Xtrain  : [m x n] matrix, where each row is a n-dimensional input example (assume it 
%            already contains the constant feature set to 1)
%  Ytrain  : [m x 1] vector, where the i-th element is the correct label
%                    for the i-th example
%  theta   : [n x 1] vector, the current model parameters

% OUTPUT
%  lik     : [1 x 1] scalar, the computed log likelihood 

m = size(Xtrain, 1); 
[pred_Y, prob_Y] = q2_predict(Xtrain, theta);
%display(theta); 
lik_vector = Ytrain .* log(prob_Y) + (ones(m, 1) - Ytrain) .* log(ones(m, 1) - prob_Y);
lik = sum(lik_vector); 
end
