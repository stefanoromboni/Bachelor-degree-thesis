import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

intervallo = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 6, 10])
eigval = np.ones(9)
eigvalRE = np.ones(9)

def osc_arm3(a, b, n):
    x = np.linspace(a, b, n+1)     #punti di definizione della nostra funzione
    h = (b-a)/n         #spaziatura del mesh
    T = 2*np.eye(n-1)-np.eye(n-1, k=-1)-np.eye(n-1, k=1)
    Q = np.diag(np.power(x[1:n], 2))
    K = T/np.power(h,2)+Q
    eigval, eigvec  = linalg.eigh(K)
    return eigval


# INTERVALLO [-1,1]
w = osc_arm3(-1, 1, 160)
eigval[0] = np.sort(w)[0]
w1 = osc_arm3(-1, 1, 80)
eigvalRE[0] = (4*np.sort(w)[0]-np.sort(w1)[0])/3


# INTERVALLO [-3/2,3/2]
w = osc_arm3(-3/2, 3/2, 160)
eigval[1] = np.sort(w)[0]
w1 = osc_arm3(-3/2, 3/2, 80)
eigvalRE[1] = (4*np.sort(w)[0]-np.sort(w1)[0])/3


# INTERVALLO [-2,2]
w = osc_arm3(-2, 2, 160)
eigval[2] = np.sort(w)[0]
w1 = osc_arm3(-2, 2, 80)
eigvalRE[2] = (4*np.sort(w)[0]-np.sort(w1)[0])/3


# INTERVALLO [-5/2,5/2]
w = osc_arm3(-5/2, 5/2, 160)
eigval[3] = np.sort(w)[0]
w1 = osc_arm3(-5/2, 5/2, 80)
eigvalRE[3] = (4*np.sort(w)[0]-np.sort(w1)[0])/3


# INTERVALLO [-3,3]
w = osc_arm3(-3, 3, 320)
eigval[4] = np.sort(w)[0]
w1 = osc_arm3(-3, 3,160)
eigvalRE[4] = (4*np.sort(w)[0]-np.sort(w1)[0])/3


# INTERVALLO [-7/2,7/2]
w = osc_arm3(-7/2, 7/2, 320)
eigval[5] = np.sort(w)[0]
w1 = osc_arm3(-7/2, 7/2, 160)
eigvalRE[5] = (4*np.sort(w)[0]-np.sort(w1)[0])/3


# INTERVALLO [-4,4]
w = osc_arm3(-4, 4, 640)
eigval[6] = np.sort(w)[0]
w1 = osc_arm3(-4, 4, 320)
eigvalRE[6] = (4*np.sort(w)[0]-np.sort(w1)[0])/3


# INTERVALLO [-6,6]
w = osc_arm3(-6, 6, 640)
eigval[7] = np.sort(w)[0]
w1 = osc_arm3(-6, 6, 320)
eigvalRE[7] = (4*np.sort(w)[0]-np.sort(w1)[0])/3


# INTERVALLO [-10,10]
w = osc_arm3(-10, 10, 640)
eigval[8] = np.sort(w)[0]
w1 = osc_arm3(-10, 10, 320)
eigvalRE[8] = (4*np.sort(w)[0]-np.sort(w1)[0])/3


plt.plot(intervallo, np.abs(1-eigval), color='limegreen', marker='o', linestyle='', label='3pt')
plt.plot(intervallo, np.abs(1-eigvalRE), color='blue', marker='x', linestyle='', label='3pt + estr.')
plt.plot(np.linspace(0, 10, 10), np.ones(10), marker='', linestyle='--', color='tab:red', label='$\lambda_{0}$')

#bellurie
plt.rc('font', size=18)
plt.xlabel('Dimensione intervallo')
plt.ylabel('$\Lambda_{0}$')
plt.legend()
plt.grid(ls='dashed')
plt.show()



