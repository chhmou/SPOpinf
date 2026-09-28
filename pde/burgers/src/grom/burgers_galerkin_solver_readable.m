function [u_rom] = burgers_galerkin_solver_readable(PAR, dW, sigmas_rom, A, B, r, model_name)

if nargin < 7 || isempty(model_name), model_name = 'Galerkin'; end

sigmas = sigmas_rom;
lambda = PAR.lam;
dt = PAR.dt;
viscosity = PAR.nu;
gamma = PAR.gam;
numTimeSteps = PAR.N;
saveStride = PAR.t_skip;

numSavedSnapshots = floor(numTimeSteps / saveStride) + 1;
u_rom = zeros(r, numSavedSnapshots);

currentState = zeros(r, 1);
currentState(1) = 0.1;
currentState(4) = 0.1;
currentState = currentState(1:r);
u_rom(:, 1) = currentState;

implicitMatrix = (1 - lambda * dt) .* eye(r) + dt * viscosity .* A;

noiseIncrement = zeros(r, 1);
progressPercent = 0;
progressInterval = floor(numTimeSteps / 100);
lastSavedIndex = 1;

fprintf(1, '\n %s, Computation Progress: %3d%%\n', model_name, progressPercent);
tic

for timeIndex = 2:numTimeSteps
    if mod(timeIndex, progressInterval) == 0
        progressPercent = 100 * (timeIndex / numTimeSteps);
        fprintf(1, '\b\b\b\b%3.0f%%', progressPercent);
    end

    nonlinearTerm = zeros(r, 1);
    for modeIndex = 1:r
        nonlinearTerm(modeIndex) = ...
            currentState' * B(:, :, modeIndex) * currentState;
    end

    noiseIncrement(1:r) = dW(1:r, timeIndex - 1) .* sigmas(1:r);

    rightHandSide = currentState - gamma * nonlinearTerm * dt + noiseIncrement;
    nextState = implicitMatrix \ rightHandSide;

    currentState = nextState;

    if timeIndex == 2 && saveStride == 1
        u_rom(:, 2) = nextState;
    elseif mod(timeIndex, saveStride) == 0
        saveIndex = floor(timeIndex / saveStride) + 1;
        u_rom(:, saveIndex) = nextState;
        lastSavedIndex = saveIndex;
    end

    if any(~isfinite(currentState))
        u_rom(:, lastSavedIndex+1:end) = NaN;
        warning('%s solution blew up', model_name)
        break;
    end
end

fprintf('\n');

fprintf(1, ' %s, CPU time: %3.2f  s\n', model_name, toc);
end
