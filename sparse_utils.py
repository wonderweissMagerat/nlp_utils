import numpy as np
from scipy.sparse import csr_matrix
import read_from_txt

def dict_list_to_csr(dic_list,vocabulary):
    #dic_list = [{"hello":3,"world":1}, {"goodbye":1, "cruel":2, "world":1}]
    indptr = [0]
    indices = []
    data = []
    #vocabulary = {"hello":0,"world":1,"goodbye":2,"cruel":3}
    for d in dic_list:
        '''
        if len(d)==0:
            indices.append(0)
            data.append(0)
        '''
        for key in d:
            index = vocabulary[key]
            indices.append(index)
            data.append(d[key])
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr),shape =(len(dic_list), len(vocabulary)))


if __name__ == '__main__':
    dic_list = [{},{"hello":3,"world":1}, {"goodbye":1, "world":1}] 
    vocabulary = {"hello":0,"world":1,"goodbye":2,"cruel":3}
    a = dict_list_to_csr(dic_list,vocabulary)
    print(a.toarray())
    print(a.get_shape())
    print(a.getrow(0).toarray()[0])
    b = a.getrow(1).multiply(a.getrow(2)).sum()
    print(b)

