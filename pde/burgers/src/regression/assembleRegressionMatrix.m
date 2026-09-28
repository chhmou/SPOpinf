function regressionMatrix = assembleRegressionMatrix(trainingStates, stateDim)
numSnapshots = size(trainingStates, 2);

identity = speye(stateDim);
numQuadraticColumns = stateDim^2 * (stateDim + 1) / 2;

upperTriKronIndex = kronector_upper_tri(stateDim);

allRows = [];
allCols = [];
allVals = [];

for snapshot = 1:numSnapshots
    stateRow = trainingStates(1:stateDim, snapshot).';

    fullKronState = kron(stateRow, stateRow);
    reducedKronState = fullKronState(upperTriKronIndex);

    quadraticBlock = kron(identity, reducedKronState);
    linearBlock    = kron(identity, stateRow);

    [quadRows, quadCols, quadVals] = find(quadraticBlock);
    [linRows, linCols, linVals]    = find(linearBlock);

    rowShift = (snapshot - 1) * stateDim;
    quadRows = quadRows + rowShift;
    linRows  = linRows + rowShift;
    linCols  = linCols + numQuadraticColumns;

    allRows = [allRows; quadRows; linRows];
    allCols = [allCols; quadCols; linCols];
    allVals = [allVals; quadVals; linVals];
end

numColumns = numQuadraticColumns + stateDim^2;
regressionMatrix = sparse(allRows, allCols, allVals, ...
    stateDim * numSnapshots, numColumns);
end
