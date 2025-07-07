import interfere_long as inter
import numpy as np
import matplotlib as plt

tlist, dt, freqList, Elw, eList, tauList, xticks, yticks = inter.init_pulse()

def stuff(w):
    
    c = 0.2998 #um/fs
    b1 = 1.03961212
    b2 = 0.231792344
    b3 = 1.01046945
    c1 = 6.00069867e-3
    c2 = 2.00179144e-2
    c3 = 1.03560653e2

    if w == 0:
        return 1 + b1 + b2 + b3

    x = (2*np.pi*c/w)**2

    return 1 + b1*x/(x-c1) + b2*x/(x-c2) + b3*x/(x-c3)

stuff_vec = np.vectorize(stuff)

ns = stuff_vec(freqList)

inter.quick_plot(ns,freqList,file='test2',square=False,xlims=(0.185,0.186))

negatives = []

for i, n in enumerate(ns):
    if n < 0:
        negatives.append(freqList[i])

print(min(negatives), max(negatives))


negatives = []

for i, n in enumerate(ns):
    if n < 1e-3:
        negatives.append(freqList[i])


print(min(negatives), max(negatives))



# H = inter.slab_transfer(inter.n_air,inter.n_BK7,0)
# freqs = np.array([1,2,3,4,5,6,7,8])
# print(H(freqs))
# print(inter.n_BK7(freqs))