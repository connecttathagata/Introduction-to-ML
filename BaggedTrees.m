function [ oobErr ] = BaggedTrees( X, Y, numBags )
%Determines out of bag classification error of an ensemble of CART decision
%trees on the training set
%   Inputs:
%       X - matrix of training data
%       Y - Vector of training labels
%       numBags - Number of trees to learn in the ensemble

treeCell = cell(1, numBags); % vector of trees
predictionMatrix = zeros( [size(X, 1) numBags] ); %matrix to hold predicted values of examples
predictedLabels = zeros( [size(X, 1) 1] ); %vector to hold predicted labels examples
oobErrors = zeros( [1 numBags] ); % vector to hold out of bag errors

for i = 1:numBags
    
    [newX, idx] = datasample( X, size(X, 1) ); %sample random rows of X with replacement
        %newX is sampled X
        %idx is indices of rows that were chosen  

    treeCell{i} = fitctree( newX, Y(idx(:)) ); %set tree to an index in treeCell
    
    unchosenExamples = setdiff( (1:size(X, 1)), idx ); %get indices of oob examples

    %get tree predictions for each oob example
    predictionMatrix(unchosenExamples(:), i) = predict( treeCell{i}, X(unchosenExamples(:), :) );

    %get predicted labels (mode) from all trees for each example
    for j = 1:size(X, 1)
       row = predictionMatrix(j, 1:i);
       rowClean = row(row ~= 0); %row without zeros
       if (isempty(rowClean))
           predictedLabels(j) = 0;
       else
           predictedLabels(j) = mode(rowClean); %get mode of row
       end
    end
    
    %number of errors
    errors = numel( find( predictedLabels(predictedLabels~=0) ~= Y(predictedLabels~=0) ) );
    oobErrors(i) = errors / sum(predictedLabels~=0); %error for i trees
end

figure
plot(1:numBags,oobErrors)
oobErr = oobErrors(numBags); %final oob error

end

