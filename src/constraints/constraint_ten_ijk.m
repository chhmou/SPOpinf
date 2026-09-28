function [vec_index1,vec_index2,vec_index3,...
    vec_index4,vec_index5,vec_index6] = constraint_ten_ijk(r)

dim1 = r;
dim2 = r;
dim3 = r;

[ind1,ind2,ind3] = index_Bvvtau_ijk(r,r);

vec_index1 = tensor2vec(ind1,ind2,ind3,dim1,dim2,dim3);
vec_index2 = tensor2vec(ind1,ind3,ind2,dim1,dim2,dim3);

vec_index3 = tensor2vec(ind2,ind1,ind3,dim1,dim2,dim3);
vec_index4 = tensor2vec(ind2,ind3,ind1,dim1,dim2,dim3);

vec_index5 = tensor2vec(ind3,ind1,ind2,dim1,dim2,dim3);
vec_index6 = tensor2vec(ind3,ind2,ind1,dim1,dim2,dim3);

end
