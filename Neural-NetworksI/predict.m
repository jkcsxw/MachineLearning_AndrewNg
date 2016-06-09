function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m,1) X]; % add 1, m*(n+1)  
  
x_theta1 = X * Theta1'; % m*(n+1) ¡Á (n+1)*k -> m*k  
x_theta1 = sigmoid(x_theta1);  
  
x_theta1 = [ones(m, 1) x_theta1]; % add 1  
  
x_theta2 = x_theta1 * Theta2'; % m*k ¡Á k*(n+1) -> m*(n+1)  
x_theta2 = sigmoid(x_theta2); 

for i=1:m
    maxValue=max(x_theta2(i,:));
    p(i)=find(x_theta2(i,:)==maxValue);
end








% =========================================================================


end
