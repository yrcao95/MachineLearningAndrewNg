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

%Calculate hx
a1=[ones(m,1) X]; %a1:5000*401
z2=Theta1*a1'; %z2:25*5000
a2=[ones(1,m);sigmoid(z2)]; %a2:26*5000
z3=Theta2*a2; %z3:10*5000
a3=sigmoid(z3); %a3:10*5000

%Construct Ym
Ym=[];
for i=1:m
    Ym=[Ym;[1:num_labels]==y(i)];
end
Ym=Ym'; %Ym:10*5000

%Calculate J without regularization
J=-(1/m)*sum(sum(Ym.*log(a3)+(1-Ym).*log(1-a3)));

Theta1reg=Theta1;
Theta2reg=Theta2;
Theta1reg(:,1)=[];
Theta2reg(:,1)=[];
reg=(sum(sum(Theta1reg.^2))+sum(sum(Theta2reg.^2)))*lambda/(2*m);
J=J+reg;

%Calculate derivative for theta
Delta3=a3-Ym; %Delta3: 10*5000
Delta2=Theta2'*Delta3.*sigmoidGradient([ones(1,m);z2]); %Delta2:26*5000
%size(Delta2)
Theta2_grad=Delta3*a2'; %Theta2_grad: 10*26
Theta2_grad=Theta2_grad/m;
%size(Theta2_grad)
Delta2=Delta2(2:end,:); %Delta2: 25*5000
%size(Delta2)
Theta1_grad=Delta2*a1; %Theta1_grad: 25*401
Theta1_grad=Theta1_grad/m;


%Regularizing Neural Networks
Theta1_temp=[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta1_grad=Theta1_grad+Theta1_temp*lambda/m;
Theta2_temp=[zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta2_grad=Theta2_grad+Theta2_temp*lambda/m;    
    

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation usi  ng a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
