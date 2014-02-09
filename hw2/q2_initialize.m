function theta = q2_initialize(Xtrain, Ytrain, opt)
% Initializes the weights for training logistic regression

% INPUT
%  Xtrain  : [m x n] matrix, where each row is a n-dimensional input example (assume it 
%            already contains the constant feature set to 1)
%  Ytrain  : [m x 1] vector, where the i-th element is the correct label
%                    for the i-th example
%  opt     : string, can be either 'random' or 'heuristic' which allows to
%                    choose the initialization between randomly of heuristic

% OUTPUT
%  theta   : [n x 1] the initialized parameter vector

% HINTS
%  We provide the code for random initialization and ask you to implement
%  the case of 'heuristic’, which we have discussed in class.


%n = size(Xtrain,2);
%m = size(Xtrain,1); 
[m, n] = size(Xtrain); 
if strcmp(opt,'random')
    % random initialization
    rand('seed', 0);
    theta = rand(n,1); % generate initial value
else
    % "heuristic" initialization
    Z = ones(m, 1);
    class_1 = find(Ytrain == 1);
    Z(find(Ytrain == 1)) = .95;
    Z(find(Ytrain == 0)) = .05;
    %for i = 1:m
    %    if Ytrain(i) == 1
    %        Z(i) = .95;
    %    else
    %        Z(i) = .05;
    %    end
    %end
    
    Z = log(Z ./ (ones(m, 1) - Z));
    theta = Xtrain \ Z;
end



end
