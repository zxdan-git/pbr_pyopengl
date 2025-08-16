import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def det3x3(mat3x3):
    return mat3x3[0, 0] * mat3x3[1, 1] * mat3x3[2, 2] -\
           mat3x3[0, 0] * mat3x3[1, 2] * mat3x3[2, 1] +\
           mat3x3[0, 1] * mat3x3[1, 2] * mat3x3[2, 0] -\
           mat3x3[0, 1] * mat3x3[1, 0] * mat3x3[2, 2] +\
           mat3x3[0, 2] * mat3x3[1, 0] * mat3x3[2, 1] -\
           mat3x3[0, 2] * mat3x3[1, 1] * mat3x3[2, 0]