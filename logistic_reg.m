function [ w, e_in ] = logistic_reg( X, y, w_init, max_its, eta )
%LOGISTIC_REG Learn logistic regression model using gradient descent
%   Inputs:
%       X : data matrix (without an initial column of 1s)
%       y : data labels (plus or minus 1)
%       w_init: initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta: learning rate
    
%   Outputs:
%       w : weight vector
%       e_in : in-sample error (as defined in LFD)

X=[ones(length(y),1) X];
stop=10E-3;
iter=0;
gr=5;
w=w_init;

while norm(gr,inf)>stop
    if iter>=max_its
        break
    end
    gr=0;
    for i=1:length(y)
        gr = gr+(y(i)*X(i,:))/(1+exp(y(i)*w.'*X(i,:).'));
    end
    g_t=-(gr/length(y)).';
    v_t=-g_t;
    w=w+eta*v_t;
    iter=iter+1;
end

e_in=0;
for i=1:length(y)
    e_in=e_in+log(1+exp(-y(i)*w.'*X(i,:).'));
end

e_in=e_in/length(y);



    
end

