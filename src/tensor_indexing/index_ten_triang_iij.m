function [ind1,ind2,ind3] = index_ten_triang_iij(r)

dim1 = r;
dim2 = r;
dim3 = r;

one_dim1  = 1:dim1;
one_dim2  = 1:dim2;
one_dim3  = 1:dim3;

s1 = kron(ones(dim3,1),kron(ones(dim2,1),one_dim1'));
s2 = kron(ones(dim3,1),kron(one_dim2',ones(dim1,1)));
s3 =  kron(one_dim3',kron(ones(dim2,1),ones(dim1,1)));

criteria =  (s1==s2).*(s1~=s3).*(s1<=s2);

ind1 = nonzeros(criteria.*s1);
ind2 = nonzeros(criteria.*s2);
ind3 = nonzeros(criteria.*s3);

end
