function [B_est, C_est, D_est, diagnostics, identifiedDiffusion] = ...
        fitPlainPolynomialOperators(trainingStates, driftTargets, timeStep)

stateDim = size(trainingStates, 1);
if size(driftTargets, 1) ~= stateDim || ...
        size(driftTargets, 2) ~= size(trainingStates, 2)
    error('fitPlainPolynomialOperators:DimensionMismatch', ...
        'States and targets must have matching dimensions and snapshots.');
end

regressionMatrix = assembleRegressionMatrix(trainingStates, stateDim);
targetVector = driftTargets(:);
thetaEstimate = regressionMatrix \ targetVector;

[B_est, C_est, D_est] = vecToOperators(thetaEstimate, stateDim);
D_est = zeros(stateDim, 1);

residualVector = targetVector - regressionMatrix * thetaEstimate;
residualMatrix = reshape(residualVector, stateDim, []);
noiseCovariance = diag(var(residualMatrix, 0, 2));
identifiedDiffusion = sqrt(max(diag(noiseCovariance) * timeStep, 0));

diagnostics = struct( ...
    'thetaEstimate', thetaEstimate, ...
    'noiseCovariance', noiseCovariance, ...
    'identifiedDiffusion', identifiedDiffusion, ...
    'relativeResidual', norm(residualVector) / max(norm(targetVector), eps), ...
    'structuralRank', sprank(regressionMatrix), ...
    'numColumns', size(regressionMatrix, 2));
end
