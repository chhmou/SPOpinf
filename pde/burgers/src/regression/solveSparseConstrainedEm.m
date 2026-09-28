function [thetaEstimate, noiseCovariance, activeSupport] = solveSparseConstrainedEm( ...
        trainingStates, driftTargets, regressionMatrix, stateDim, dt)
numSnapshots = size(trainingStates, 2);

constraintMatrixMoi = buildEnergyConstraintMatrix(stateDim);
nonzeroIndex        = non_zero_BC(stateDim);
initialTheta        = zeros(size(regressionMatrix, 2), 1);

emTolerance       = 1e-8;
sparsityThreshold = 0.001;
ridgeGamma        = 1e-8;
maxEmIterations   = 100;
maxStlsqPasses    = 10;

[thetaEstimate, noiseCovariance, activeSupport] = em_stochastic_solver_sparse( ...
    trainingStates, driftTargets, stateDim, numSnapshots, ...
    initialTheta, constraintMatrixMoi, nonzeroIndex, emTolerance, dt, ...
    sparsityThreshold, ridgeGamma, maxEmIterations, maxStlsqPasses);
end
