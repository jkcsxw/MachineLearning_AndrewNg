function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   T
% he output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);



mu = mean(X);  
sigma2 = var(X,1); 


% =============================================================


end
