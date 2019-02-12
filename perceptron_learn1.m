function [ w iterations ] = perceptron_learn( data_in )
%perceptron_learn Run PLA on the input data
%   Inputs: data_in: Assumed to be a matrix with each row representing an
%                    (x,y) pair, with the x vector augmented with an
%                    initial 1, and the label (y) in the last column
%   Outputs: w: A weight vector (should linearly separate the data if it is
%               linearly separable)
%            iterations: The number of iterations the algorithm ran for
d=length(data_in(1,:));
w_i = zeros(d,1);
iter=0;
classi = ones(d,1);

while true
    for i = 1:length(classi)
        if classi(i)~=data_in(i,end)
            iter = iter+1;
            w_i=w_i+data_in(i,end)*data_in(i,:)';
            break;
        end
    end
    w_i(end)=0;
    classi = sign(data_in*w_i);
    if classi==data_in(:,end)
        break;
    end
end

iterations=iter;
w=w_i;
end

