function [pred_Y, prob_Y] = q2_predict(X, theta)
% Predict the labels and probabilities for the set of examples X using the
% model theta

% INPUT
%  X      : [m x n] matrix, where each row is an n-dimensional input example (please assume it 
%            already contains the constant feature set to 1)
%  theta  : [n x 1] vector, the model parameters used to make predictions

% OUTPUT
%  pred_Y : [m x 1] vector, the predicted labels for the examples in X
%                   note that pred_Y has binary values {0,1} in this case
%  prob_Y : [m x 1] vector, the posterior probabilities produced by the logistic function

[m, n] = size(X);

prob_Y = ones(m, 1); 
pred_Y = ones(m, 1);

for i = 1:m
    prob_Y(i) = 1 / (1 + exp(-1 * X(i, :) * theta));
end
 
pred_Y(prob_Y > .5) = 1; 
pred_Y(prob_Y <= .5) = 0; 

%display('in q2_predict:'); 
%display(prob_Y);
end
