function feat_selected = q4_rf_split(X, Y, feat_idx, F)
% Find the split that maximizes the information gain for the subset 
% (X, Y) of the training set from a random subset of F features

% INPUT

%  X        : [m x n] matrix, where each row is an n-dimensional input example
%  Y        : [m x 1] vector, where the i-th element is the label for the i-th example
%  feat_idx : [1 x L] vector, indices of features to be potentially considered
%  F        : [1 x 1] scalar, size of the random subset of features to be considered

% OUTPUT
%  feat_selected : [1 x 1] scalar, the feature that maximizes the information gain for (X, Y) 
%                  (this should be one of the numbers stored in feat_idx and an integer between 1 and n). 
%                  Note this value must be set to 0 if there is no feasible split


% INSERT YOUR CODE HERE:
% compute the feasible feature indices
% store them in feasible_idx

% USE THIS AS RANDOM SELECTION OF FEATURE SUBSET
% LEAVE UNCHANGED
% -----------------------------------------------
    indperm = randperm(length(feasible_idx));
    if length(feasible_idx)>F
        feasible_idx = feasible_idx(indperm(1:F));
    end
% -----------------------------------------------

% INSERT YOUR CODE HERE:
% choose within this subset the one with the best gain


end
