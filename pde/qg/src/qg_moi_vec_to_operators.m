function [B, C] = qg_moi_vec_to_operators(theta, dim)

n_quad = dim^2 * (dim + 1) / 2;
n_linear = dim^2;
assert(numel(theta) == n_quad + n_linear, ...
    'Unexpected MOI coefficient-vector length.');

B_vec = theta(1:n_quad);
C_vec = theta(n_quad + 1:end);

ten2vec = B_index_moi(dim);
B_full = zeros(dim^3, 1);
B_full(ten2vec) = B_vec;
B = reshape(B_full, dim, dim, dim);
C = reshape(C_vec, dim, dim)';
end
