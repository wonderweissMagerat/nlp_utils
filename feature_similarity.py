import sys
import numpy as np
import math

def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm) 
    return similiarity

def cosine_distance_sparse(a,b):
    dot_distance = a.multiply(b).sum()
    a_len = math.sqrt(a.multiply(a).sum())
    b_len = math.sqrt(b.multiply(b).sum())
    similarity = dot_distance/(a_len * b_len) if a_len * b_len!=0 else 0
    return similarity

if __name__ == '__main__':
    a = np.array([1,0])
    b = np.array([0,1])
    print(cosine_distance(a, b))