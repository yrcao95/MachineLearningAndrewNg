function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);
thetatemp=theta(2:n,:);
hx=sigmoid(X*theta);
J=-(y'*log(hx)+(1-y)'*log(1-hx))/m+thetatemp'*thetatemp*(lambda/(2*m));
% You need to return the following variables correctly 
thetazero=[0;thetatemp]
grad=X'*(hx-y)/m+thetazero*(lambda/m);


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
