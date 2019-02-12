function [ num_iters bounds] = perceptron_experiment ( N, d, num_samples )
%perceptron_experiment Code for running the perceptron experiment in HW1
%   Inputs: N is the number of training examples
%           d is the dimensionality of each example (before adding the 1)
%           num_samples is the number of times to repeat the experiment
%   Outputs: num_iters is the # of iterations PLA takes for each sample
%            bounds is the theoretical bound on the # of iterations
%              for each sample
%      (both the outputs should be num_samples long)

for j=1:1:num_samples
wstar=rand(d+1,1);
wstar(1)=0;%wstar generated

Train1=-1+2.*rand(N,d);
A=ones(N,1);
Samples=horzcat(A,Train1);
data_in=horzcat(A,Train1,A); 
data_in(:,d+2)=sign(Samples*wstar);%Training data set generated

[~,iterations] = perceptron_learn( data_in );%Output of learning
num_iters(j)=iterations;

n=(norm(wstar)^2);
rho=min(abs(Samples*wstar));
X=data_in(:,1:d);
R=max(sum(abs(X).^2,2));
bounds(j)=(n^2)*(R^2)/(rho^2);%Theoretical bound

end 
numdiff=log(bounds-num_iters);
figure
hist(num_iters,50)
title('Figure 7:Histogram of Iterations');
ylabel('Occurences');
xlabel('Number of Iterations');

figure
hist(numdiff,50)
title('Figure 8:Histogram of Log of Theoretical and Experimental Difference');
xlabel('Log of Difference');
ylabel('Occurences')

 


    





 
end

