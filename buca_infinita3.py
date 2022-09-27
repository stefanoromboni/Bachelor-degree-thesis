import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit


'''
Risolviamo il problema della buca infinita su un intervallo [0,π].
L'equazione differenziale da risolvere, è:
                    -u" = lu
dove in questo caso l indica l'autovalore. Le BC sono date dalla
fisica del problema, u(0)=u(π)=0.
 '''


'''
Andiamo a definire una funzione chiamata "buca", che prendera come parametri iniziali
l'intervallo di definizione e il numero di punti di discretizzazione:
                        buca(a, b, n)
e restituisce un vettore contenente gli autovalori e gli autovettori trovati
'''

def buca3(a, b, n):
    x = np.linspace(a, b, n-1)     #punti di definizione della nostra funzione
    h = (b-a)/n         #spaziatura del mesh
    T = 2*np.eye(n-1)-np.eye(n-1, k=-1)-np.eye(n-1, k=1)
    K = T/np.power(h,2)
    eigval, eigvec  = linalg.eigh(K)
    return eigval, eigvec, x


'''
Come verifica dell'effettivo funzionamento dell'algoritmo occorre verificare l'andamento
dell'errore, deve essere lineare in h^2. Possiamo verificarlo per l'autovalore fondamentale
'''


fond3 = np.ones(6)
h3 = np.ones(6)
n = np.array([10, 20, 40, 80, 160, 320])


for i in range(0, len(fond3)):
    w, v, x = buca3(0, np.pi, n[i])
    fond3[i]=np.sort(w)[0]
    h3[i]=np.pi/n[i]



plt.figure(num=1)
plt.rc('font', size = 20)
#plt.plot( np.power(h3, 2), fond3, color='g', marker='o', linestyle='')
plt.plot(0, 1, color='r', marker='o', alpha=1)
plt.grid()


'''
Per verificare che effettivamente l'andamento dell'errore sia lineare in h^2 andiamo
ad eseguire un best fit dell'errore con una retta e a vedere a che valore tende la sequenza per h=0
'''

plt.figure(num=1)
####CARICAMENTO DATI

# cambiamo array
x=np.power(h3, 2)
y=fond3
dy=1e-15*np.ones(len(y))


# scatter plot with error bars
plt.plot(x,y,linestyle='-', marker='o', color='tab:blue', label='dati 3pt')



# AT THE FIRST ATTEMPT COMMENT FROM HERE TO THE END

# define the function (linear, in this example)
def ff(x, aa, bb):
    return aa*x+bb

# define the initial values (STRICTLY NEEDED!!!)
init=(-1, 1)

# prepare a dummy xx array (with 2000 linearly spaced points)
xx=np.linspace(min(x),max(x),2000)

# plot the fitting curve computed with initial values
# AT THE SECOND ATTEMPT THE FOLLOWING LINE MUST BE COMMENTED
#ax2.plot(xx,ff(xx,*init), color='blue')

# set the error
sigma=dy
w=1/sigma**2

# call the minimization routine
pars,covm=curve_fit(ff,x,y,init, absolute_sigma=False)

# calculate the chisquare for the best-fit function
chi2 = ((w*(y-ff(x,*pars))**2)).sum()

# determine the ndof
ndof=len(x)-len(init)

# print results on the console
print('pars:',pars)
print('covm:',covm)
print ('chi2, ndof:',chi2, ndof)
aa = pars[0]
bb = pars[1]
daa = np.sqrt(covm[0,0])
dbb = np.sqrt(covm[1,1])
print('aa = %.8f +- %.8f [arb.un.]\n' %(aa, daa))
print('bb = %.8f +- %.8f [arb.un.]\n' %(bb, dbb))

# plot the best fit curve
plt.plot(xx,ff(xx,*pars), color='limegreen', label='$f(x)=ax+b$')

# bellurie
plt.xlabel('$O(h^{2})$')
plt.ylabel('$\Lambda_{0}$')
plt.legend(loc='best')
plt.grid(ls='dashed')
plt.minorticks_on()


'''
# build the array of the normalized residuals
r = (y-ff(x,*pars))/sigma

# bellurie
ax1.set_ylabel('Norm. res.')
ax1.minorticks_on()
# set the vertical range for the norm res
ax1.set_ylim((-2, 2))

# plot residuals as a scatter plot with connecting dashed lines
ax1.plot(x,r,linestyle="--",color='blue',marker='o')
'''



### PROVIAMO A GRAFICARE I PRIMI AUTOVETTORI

'''
plt.figure(num=2)

w, v, x = buca3(0, np.pi, 500)
v = np.transpose(v)
for j in range(0, 499):
    if 0.99<w[j]<1.1:
        plt.plot(x, v[j], marker='', linestyle='-', label='$\lambda_{0}=1$')
    if 3.99<w[j]<4.1:
        plt.plot(x, v[j], marker='', linestyle='-', label='$\lambda_{1}=4$')
    if 8.99<w[j]<9.1:
        plt.plot(x, v[j], marker='', linestyle='-', label='$\lambda_{2}=9$')

plt.legend()
plt.grid(ls='dashed')
plt.xlabel('X')
plt.ylabel('Y')

# show the plot
#plt.show()
'''

### ESTRAPOLAZIONE RICHARDSON

'''
Usiamo l'estrapolazione di Rirchardson per andare a migliorare il nostro risultato.

Proviamo sull'autovalore fondamentale che sappiamo debba essere 1
'''

'''
w1, _, _ = buca3(0, np.pi, 10)
w2, _, _ = buca3(0, np.pi, 20)

w = (4*w2[:9]-w1)/3
'''

fond = np.ones(len(fond3)-1)

for i in range(0, len(fond3)-1):
    fond[i] = (4*fond3[i+1]-fond3[i])/3


'''
Eseguiamo anche qui un best-fit dei dati per vedere che l'errore sia effettivamente
lineare in h^4
'''

plt.figure(num=3)
####CARICAMENTO DATI

# cambiamo array
x=np.power(h3, 4)[1:6]
y=fond
dy=1e-9*np.ones(len(y))


# scatter plot with error bars
plt.plot(x,y,linestyle='-', marker='o', color='tab:red', label='dati 3pt + RE', alpha=0.6)
plt.plot(0, 1, color='r', marker='o', alpha=1)


# AT THE FIRST ATTEMPT COMMENT FROM HERE TO THE END

# define the function (linear, in this example)
def ff(x, aa, bb):
    return aa*x+bb

# define the initial values (STRICTLY NEEDED!!!)
init=(-1, 1)

# prepare a dummy xx array (with 2000 linearly spaced points)
xx=np.linspace(min(x),max(x),2000)

# plot the fitting curve computed with initial values
# AT THE SECOND ATTEMPT THE FOLLOWING LINE MUST BE COMMENTED
#ax2.plot(xx,ff(xx,*init), color='blue')

# set the error
sigma=dy
w=1/sigma**2

# call the minimization routine
pars,covm=curve_fit(ff,x,y,init, absolute_sigma=False)

# calculate the chisquare for the best-fit function
chi2 = ((w*(y-ff(x,*pars))**2)).sum()

# determine the ndof
ndof=len(x)-len(init)

# print results on the console
print('pars:',pars)
print('covm:',covm)
print ('chi2, ndof:',chi2, ndof)
aa = pars[0]
bb = pars[1]
daa = np.sqrt(covm[0,0])
dbb = np.sqrt(covm[1,1])
print('aa = %.16f +- %.16f [arb.un.]\n' %(aa, daa))
print('bb = %.16f +- %.16f [arb.un.]\n' %(bb, dbb))

# plot the best fit curve
plt.plot(xx,ff(xx,*pars), color='limegreen', label='$f(x)=ax+b$')

# bellurie
plt.xlabel('$O(h^{4})$')
plt.ylabel('$\Lambda_{0}$')
plt.legend(loc='best')
plt.grid(ls='dashed')
plt.minorticks_on()


# plt.figure(num=3)
# plt.plot( np.power(h3[0:5], 4), fond, color='b', marker='o', linestyle='--')
# plt.xlabel('$O(h^{4})$')
# plt.ylabel('$\lambda_{0}$')
# plt.legend(loc='best')
# plt.grid(ls='dashed')
# plt.minorticks_on()




plt.show()



'''
Scriviamo la tabella con i valori ottenuti per i primi autovalori.
Li inseriamo tutti in un'unica matrice, dove le colonne corrispondono a diverse
discretizzazioni mentre una singola colonna contiene il valore di vari autovettori
'''
m = np.ones(len(n))
M = np.array([m, m, m, m, m, m, m, m, m])
M = np.transpose(M)

for i in range(0, len(n)):
    w, v, x = buca3(0, np.pi, n[i])
    M[i]=np.sort(w)[:9]

M = np.transpose(M)

######     SCRITTURA SU FILE    ############################################################
#
# file=open('/Users/stefanoromboni/Desktop/Università/Tesi/Codici python/buca infinita/tab_eigval.txt', 'w')
# for i in range (0, len(n)):
#     file.write('$%.5f$ \t&\t $%.5f$ \t&\t $%.5f$ \t&\t $%.5f$ \t&\t $%.5f$ \t&\t $%.5f$ \t&\t $%.5f$ \t&\\\\ \n' %(np.power(i+1,2), M[i,0], M[i,1], M[i,2], M[i,3], M[i,4], M[i,5]) )
# file.close()

plt.figure(num=4)
plt.plot(n, 1-fond3, color='tab:blue', linestyle='--', marker='s', label='3pt')
plt.plot(n[:5], 1-fond, color='limegreen', linestyle='--', marker='o', label='3pt+RE')

plt.xlabel('dimensione reticolo $n$' )
plt.ylabel('$|1-\Lambda_{0}|$')
plt.legend()
plt.grid(ls='dashed')
plt.rc('font', size='15')
plt.show()



