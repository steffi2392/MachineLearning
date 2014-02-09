function q3a()

% load the data
S = load('parkinsons.mat');
X = S.trainsetX;
Y = S.trainsetY;
Xt = S.testsetX;
Yt = S.testsetY;

clear S;

% set k parameters
k = 1:2:13;


error = q3_test_error(X, Y, Xt, Yt, k);



close all;

plot(k, error, 'bo-');
ylabel('misclassification rate');
title('kNN test error on parkinsons dataset');
xlabel('k');
grid on;

saveas(gcf, 'q3a.fig');