
y=0;
J=0;
J=((y*log(h))+((1-y)*log(1-h))) * -1;
delta3=h-y;
delta2=((theta2' * delta3) * (z2*(1-z2)))(2:end); %to take the 2nd element of the matrix to the end is what is ment by the (2:end) The reason this is done is because we are not propagating an error back from the bias node.

theta2= theta2-(0.01 * (delta3*A2'));
theta1= theta1-(0.01 * (delta2*A1'));
z2=[1; theta1*A1];

%function[result]=sigmoid(x);
%result=1.0 /(1.0 +exp(-x));

A2=sigmoid(z2);
z3=theta2 * A2;
h= sigmoid(z3);
disp("The cost is:");
disp(h);

