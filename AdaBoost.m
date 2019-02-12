function [ train_err, test_err ] = AdaBoost( X_tr, y_tr, X_te, y_te, n_trees )
%AdaBoost: Implement AdaBoost using decision stumps learned
%   using information gain as the weak learners.
%   X_tr: Training set
%   y_tr: Training set labels
%   X_te: Testing set
%   y_te: Testing set labels
%   n_trees: The number of trees to use

num_tr_ex = size(X_tr, 1); %number of training examples

y_tr(find(y_tr < mean(y_tr))) = -1; %all values < mean are -1
y_tr(find(y_tr > mean(y_tr))) = 1; %all values > mean are +1
predicted_tr = zeros( [num_tr_ex 1] ); %predicted classes of training data

y_te(find(y_te < mean(y_te))) = -1; %all values < mean are -1
y_te(find(y_te > mean(y_te))) = 1; %all values > mean are +1
predicted_te = zeros( [size(X_te, 1) 1]); %predicted classes of test data

[alpha_vector, training_errors, test_errors] = deal(zeros( [n_trees 1] ));

D = ones( [num_tr_ex 1]) / num_tr_ex; %initialize vector of weights
predicted_matrix = zeros( [num_tr_ex n_trees] ); %matrix of predicted values for each tree
predicted_matrix_te = zeros( [size(X_te, 1) n_trees] ); %matrix of predicted values for test data
for t = 1:n_trees

    decision_stump = fitctree(X_tr, y_tr, 'Weights', D, 'MaxNumSplits', 1); %create stump
    predicted_values = predict(decision_stump, X_tr); %get predicted values
    weight_error = 0;
    
    %calculate weighted error
    for i = 1:num_tr_ex
        if (predicted_values(i) ~= y_tr(i))
           weight_error = weight_error + D(i); 
        end
    end
    weight_error = weight_error / sum(D);
    alpha_vector(t) = .5 * log( (1-weight_error) / weight_error ); %alpha value

    Z = 2 * sqrt( weight_error * (1 - weight_error) ); %normalization factor
    
    %update the weights D based off of correct/incorrect classification
    for j = 1:num_tr_ex
       if (predicted_values(j) ~= y_tr(j))
          D(j) = ( D(j) * exp(alpha_vector(t))) / Z; 
       else
          D(j) = ( D(j) * exp(-alpha_vector(t))) / Z; 
       end
    end

    predicted_matrix(:, t) = predicted_values;
    predicted_matrix_te(:, t) = predict(decision_stump, X_te);
    
    %get final hypotheses for each example for trees made up to this point
    %calculate training and test errors
    for f = 1:num_tr_ex
       predicted_tr(f) = sign( sum( transpose(alpha_vector(1:t)).*predicted_matrix(f, 1:t)) );
       if (predicted_tr(f) ~= y_tr(f))
           training_errors(t) = training_errors(t) + 1; 
       end
       
       if (f <= size(X_te, 1))
           predicted_te(f) = sign( sum( transpose(alpha_vector(1:t)).*predicted_matrix_te(f, 1:t)) );
           if (predicted_te(f) ~= y_te(f))
               test_errors(t) = test_errors(t) + 1; 
           end
       end
    end
    
    training_errors(t) = training_errors(t) / num_tr_ex;
    test_errors(t) = test_errors(t) / size(X_te, 1);
        
end

train_err = training_errors; %final training error
test_err = test_errors; %final test error




end

