import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg


'''
H = p^2/2 + x^2/2 + gx^4/2 = T/2 + Q/2 + gG/2
'''

'''
Per risolvere il problema agli autovalori occorre costruire la matrice hamiltoniana
e diagonalizzarla.
'''


def osc_base(g, n):
    T = np.eye(n)
    G = np.eye(n)
    Q = np.eye(n)
    for i in range(0, n):
        Q[i,i] = (2*i+1)/2
        G[i,i] = 3*(2*np.power(i,2) + 2*i + 1)/4
        T[i,i] = (2*i+1)/2
    for i in range(0, n-2):
        Q[i, i+2] = np.sqrt((i+1)*(i+2))/2
        Q[i+2, i] = np.sqrt((i+1)*(i+2))/2
        G[i, i+2] = (2*i+3)*np.sqrt((i+1)*(i+2))/2
        G[i+2, i] = (2*i+3)*np.sqrt((i+1)*(i+2))/2
        T[i, i+2] = -np.sqrt((i+1)*(i+2))/2
        T[i+2, i] = -np.sqrt((i+1)*(i+2))/2
    for i in range(0, n-4):
        G[i, i+4] = np.sqrt((i+1)*(i+2)*(i+3)*(i+4))/4
        G[i+4, i] = np.sqrt((i+1)*(i+2)*(i+3)*(i+4))/4
    H = T/2 + Q/2 + g*G/2
    eigval, eigvec = linalg.eigh(H)
    return eigval, eigvec, T, Q, G, H


w, v, T, Q, G, H = osc_base(1, 10000)
print(np.sort(w)[0:5])

