function pos=tensor2vec(i,j,k,dim1,dim2,dim3)

pos = i+ (j-1)*dim1+(k-1)*(dim1*dim2);

end
