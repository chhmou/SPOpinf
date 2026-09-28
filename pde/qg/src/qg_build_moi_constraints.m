function [constraint_mat_H_moi, nonzero_index] = ...
    qg_build_moi_constraints(dim)

[vec_index1] = constraint_ten_iii(dim);
[vec_index31, vec_index32, vec_index33] = constraint_ten_iij(dim);
[vec_index61, vec_index62, vec_index63, ...
    vec_index64, vec_index65, vec_index66] = constraint_ten_ijk(dim);

row_pos = 0;
row_Biii = (1:dim)' + row_pos;
col_Biii = vec_index1;

row_pos = row_pos + dim;
identity_iij = kron(ones(3, 1), (1:dim * (dim - 1))');
row_Biij = identity_iij + row_pos;
col_Biij = [vec_index31; vec_index32; vec_index33];

row_pos = row_pos + dim * (dim - 1);
identity_ijk = kron(ones(6, 1), ...
    (1:dim * (dim - 1) * (dim - 2) / 6)');
row_Bijk = identity_ijk + row_pos;
col_Bijk = [vec_index61; vec_index62; vec_index63; ...
    vec_index64; vec_index65; vec_index66];

constraint_row = [row_Biii; row_Biij; row_Bijk];
constraint_col = [col_Biii; col_Biij; col_Bijk];
constraint_val = ones(size(constraint_row));

num_rows = dim + dim * (dim - 1) + ...
    dim * (dim - 1) * (dim - 2) / 6;
num_cols = dim^3 + dim^2;
constraint_mat_H = sparse(constraint_row, constraint_col, ...
    constraint_val, num_rows, num_cols);

ind_vec_upper_tri = vec_index_ten_upper_tri(dim);
ind_vec_col_constraint = [ind_vec_upper_tri; ...
    (dim^3 + 1:num_cols)'];
constraint_mat_H_moi = constraint_mat_H(:, ind_vec_col_constraint);
nonzero_index = non_zero_BC(dim);
end
