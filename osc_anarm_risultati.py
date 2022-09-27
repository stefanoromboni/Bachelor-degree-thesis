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



def osc_base(g):
    n = 100
    y = np.ones(len(g))
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
    for i in range(0, len(g)):
        H = T/2 + Q/2 + g[i]*G/2
        eigval, eigvec = linalg.eigh(H)
        y[i] = np.sort(eigval)[0]
    return y



def fond_1(g):
    y = 3/8*g
    return y


def fond_2(g):
    y,y1 = 0,0
    n = 10
    G = np.eye(n)
    for i in range(0, n):
        G[i,i] = 3*(2*np.power(i,2) + 2*i + 1)/4
    for i in range(0, n-2):
        G[i, i+2] = (2*i+3)*np.sqrt((i+1)*(i+2))/2
        G[i+2, i] = (2*i+3)*np.sqrt((i+1)*(i+2))/2
    for i in range(0, n-4):
        G[i, i+4] = np.sqrt((i+1)*(i+2)*(i+3)*(i+4))/4
        G[i+4, i] = np.sqrt((i+1)*(i+2)*(i+3)*(i+4))/4
    V = G/2
    E = np.ones(5)
    for i in range(0, 4):
        E[i] = 1/2 + i
    for i in range(1,4):
        y = g**2*np.power(np.abs(V[0,i]),2)/(E[0]-E[i])+y
    return y

def fond_3(g):
    y,y1,y2 = 0,0,0
    n = 10
    G = np.eye(n)
    for i in range(0, n):
        G[i,i] = 3*(2*np.power(i,2) + 2*i + 1)/4
    for i in range(0, n-2):
        G[i, i+2] = (2*i+3)*np.sqrt((i+1)*(i+2))/2
        G[i+2, i] = (2*i+3)*np.sqrt((i+1)*(i+2))/2
    for i in range(0, n-4):
        G[i, i+4] = np.sqrt((i+1)*(i+2)*(i+3)*(i+4))/4
        G[i+4, i] = np.sqrt((i+1)*(i+2)*(i+3)*(i+4))/4
    V = G/2
    E = np.ones(5)
    for i in range(0, 4):
        E[i] = 1/2 + i
    for i in range(1, 4):
        for j in range(1,4):
            y1 = V[0,j]*V[j,i]*V[i,0]/(E[0]-E[i])/(E[0]-E[j])+y1
        y2 = np.power(np.abs(V[0,i]),2)/np.power((E[0]-E[i]), 2)+y2
    y2 = V[0,0]*y2
    y = (y1 - y2)*g**3
    return y



def osc_anarm3(g):
    n = 1280
    x = np.linspace(-6, 6, n+1)     #punti di definizione della nostra funzione
    h = (12)/n                     #spaziatura del mesh
    y = np.ones(len(g))
    for i in range(0, len(g)):
        T = 2*np.eye(n-1)-np.eye(n-1, k=-1)-np.eye(n-1, k=1)
        Q = np.diag(np.power(x[1:n], 2)+g[i]*np.power(x[1:n], 4))
        K = T/np.power(h,2)/2+Q/2
        eigval, eigvec  = linalg.eigh(K)
        y[i]=np.sort(eigval)[0]
    return y


g = np.linspace(0, 10, 100)

plt.plot(g, 0.5 + fond_1(g), marker='', linestyle='dotted', label='1° ordine pt')
plt.plot(g, 0.5+fond_1(g)+fond_2(g), marker='', linestyle='dashed', label='2° ordine pt')
plt.plot(g, 0.5+fond_1(g)+fond_2(g)+fond_3(g), marker='', linestyle='dashdot', label='3° ordine pt')
#plt.plot(g, osc_base(g), marker='', linestyle='--', label='sol. numerica 1')
plt.plot(g, osc_anarm3(g), marker='', linestyle='solid', label='sol. numerica')




plt.legend()
plt.grid(ls='dashed')
plt.xlabel('g')
plt.ylabel('$E_{fund}(g)$')
plt.rc('font', size='15')

plt.show()






