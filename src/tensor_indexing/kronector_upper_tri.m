function [ind_vec] = kronector_upper_tri(dim)

one_dim  = 1:dim;

s1 = kron(ones(dim,1),one_dim');
s2 = kron(one_dim',ones(dim,1));

criteria =  (s1<=s2);

mat_j = nonzeros(criteria.*s1);
mat_k = nonzeros(criteria.*s2);

indvec = 1:dim^2;

ind2mat = reshape(indvec,dim,dim);

indsize = length(mat_j);

ind_vec = zeros(indsize,1);

for k_inter = 1: indsize

    ind_vec(k_inter) = ind2mat(mat_j(k_inter),mat_k(k_inter));

end

end
