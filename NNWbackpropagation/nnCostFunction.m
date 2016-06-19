function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


a1 = [ones(m,1) X];   %m*n+1
z2 = Theta1*a1';      %j*m
a2=sigmoid(z2);       
a2=[ones(1,m); a2];    %j+1*m
z3=Theta2*a2;          %10*m
a3=sigmoid(z3);

y_m=zeros(num_labels,m);   %10*m
for i=1:m
     y_m(y(i),i)=1;
end;

%cost function
for i=1:m
    J =J+sum(-1*y_m(:,i).*log(a3(:,i))-(1-y_m(:,i)).*log(1-a3(:,i)));
end
J=J/m;
J = J + lambda*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))/2/m;  


%backpropagation
Delta1=zeros(size(Theta1));
Delta2=zeros(size(Theta2));
for i=1:m
    delta3=a3(:,i)-y_m(:,i);
    a2k=Theta2'*delta3;
    delta2=a2k(2:end,:).*sigmoidGradient(z2(:,i));
    Delta2= Delta2+delta3*a2(:,i)';
    Delta1=Delta1 + delta2*a1(i,:);
end
%gradient
Theta2_grad=Delta2/m;
Theta1_grad=Delta1/m;

%regularization
Theta2_grad(:,2:end)=Theta2_grad(:,2:end) + lambda *Theta2(:,2:end)/m;
Theta1_grad(:,2:end)=Theta1_grad(:,2:end) + lambda *Theta1(:,2:end)/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
