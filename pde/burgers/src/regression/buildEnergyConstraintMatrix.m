function constraintMatrixMoi = buildEnergyConstraintMatrix(stateDim)

idxBiii = constraint_ten_iii(stateDim);

[idxBiij1, idxBiij2, idxBiij3] = constraint_ten_iij(stateDim);

[idxBijk1, idxBijk2, idxBijk3, ...
    idxBijk4, idxBijk5, idxBijk6] = constraint_ten_ijk(stateDim);

rowOffset = 0;

rowsBiii = (1:stateDim).' + rowOffset;
colsBiii = idxBiii;

rowOffset = rowOffset + stateDim;

numIijRows = stateDim * (stateDim - 1);
rowsBiij = kron(ones(3, 1), (1:numIijRows).') + rowOffset;
colsBiij = [idxBiij1; idxBiij2; idxBiij3];

rowOffset = rowOffset + numIijRows;

numIjkRows = stateDim * (stateDim - 1) * (stateDim - 2) / 6;
rowsBijk = kron(ones(6, 1), (1:numIjkRows).') + rowOffset;
colsBijk = [idxBijk1; idxBijk2; idxBijk3; idxBijk4; idxBijk5; idxBijk6];

constraintRows = [rowsBiii; rowsBiij; rowsBijk];
constraintCols = [colsBiii; colsBiij; colsBijk];
constraintVals = ones(size(constraintRows, 1), 1);

numConstraintRows = stateDim + numIijRows + numIjkRows;
numFullColumns    = stateDim^3 + stateDim^2;

constraintMatrixFull = sparse(constraintRows, constraintCols, constraintVals, ...
    numConstraintRows, numFullColumns);

upperTriQuadraticIndex = vec_index_ten_upper_tri(stateDim);
linearIndex = (stateDim^3 + 1:numFullColumns).';
moiColumns = [upperTriQuadraticIndex; linearIndex];

constraintMatrixMoi = constraintMatrixFull(:, moiColumns);
end
