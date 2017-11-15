function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = X*theta;
h = sigmoid(h);

theta1 = [0 ; theta(2:size(theta), :)];

cost = (-y)' * log(h) - (1-y)' * log(1-h);
reg = lambda/(2*m) * sum(theta1.*theta1);
J = 1/m * cost + reg;

grad1 = zeros(m,1);
% grad2 = zeros(m,1);


grad1 = 1/m * X' * (h-y);
grad2 = (lambda/m) * theta1;


grad = grad1+grad2;





% =============================================================

end
