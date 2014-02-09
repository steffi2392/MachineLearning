function [theta, n_iter, loglik] = q2_train_line_search(Xtrain, Ytrain, theta_init, alpha0, tol)
% Train logistic regression using gradient ascent via line search
% 
% INPUT
%  Xtrain    : [m x n] matrix, where each row is a n-dimensional input example (assume it 
%              already contains the constant feature set to 1)
%  Ytrain    : [m x 1] vector, where the i-th element is the correct label
%                    for the i-th example
%  theta_init: [n x 1] vector, the initial parameter vector
%  alpha0    : [1 x 1] scalar, the initial (large) step size used for line search
%  tol       : [1 x 1] scalar, tolerance value used in the stopping condition

% OUTPUT
%  theta   : [n x 1] vector, the learned parameters
%  n_iter  : [1 x 1] scalar, the number of gradient ascent iterations until convergence
%  loglik  : [1 x n_iter] vector containing the log likelihood value at each iteration

% HINTS
%  your program should use the following stopping criterion:
%        while (norm(grad)>tol) && (n_iter < 100000)
%
% where grad is the gradient at the current iteration

alpha = alpha0; 
theta = theta_init;
display(theta_init); 
grad = q2_gradient(Xtrain, Ytrain, theta_init);
n_iter = 0; 
loglik = [];

while (norm(grad) > tol) && (n_iter < 5)
    loglik = [loglik q2_loglik(Xtrain, Ytrain, theta)];
    display(loglik); 
    newTheta = theta + alpha .* grad;
    display(grad); 
    
    display('next loglik:');
    display(q2_loglik(Xtrain, Ytrain, newTheta));
    display('curr loglik:');
    display(q2_loglik(Xtrain, Ytrain, theta)); 
    
    while (q2_loglik(Xtrain, Ytrain, newTheta)) <= q2_loglik(Xtrain, Ytrain, theta)
        alpha = alpha / 2; 
        display(alpha); 
        newTheta = theta + alpha .* grad;
    end
    
    theta = newTheta;
    %display(theta); 
    %display(q2_loglik(Xtrain, Ytrain, theta)); 
    n_iter = n_iter + 1;
    grad = q2_gradient(Xtrain, Ytrain, theta); 
end

end
