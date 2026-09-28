function [B_est, C_est, D_est] = vecToOperators(theta, stateDim)
numQuadraticCoefficients = stateDim^2 * (stateDim + 1) / 2;
numLinearCoefficients = stateDim^2;

B_vec = theta(1:numQuadraticCoefficients);
C_vec = theta(numQuadraticCoefficients+1 : ...
    numQuadraticCoefficients+numLinearCoefficients);

if numel(theta) > numQuadraticCoefficients + numLinearCoefficients
    D_vec = theta(numQuadraticCoefficients+numLinearCoefficients+1:end);
else
    D_vec = zeros(stateDim, 1);
end

upperTriTensorIndex = B_index_moi(stateDim);
B_full = zeros(stateDim^3, 1);
B_full(upperTriTensorIndex) = B_vec;

B_est = reshape(B_full, stateDim, stateDim, stateDim);
C_est = reshape(C_vec, stateDim, stateDim).';
D_est = D_vec;
end
