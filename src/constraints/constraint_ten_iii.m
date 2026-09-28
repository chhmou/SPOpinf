function  [vec_index0] =  constraint_ten_iii(r)

dim1 = r;
dim2 = r;
dim3 = r;

[ind1,ind2,ind3] = index_ten_iii(r);

vec_index0 = tensor2vec(ind1,ind2,ind3,dim1,dim2,dim3);

end
