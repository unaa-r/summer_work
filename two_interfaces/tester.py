import interfaces_jul9 as inter
import numpy as np
import matplotlib as plt

tlist, dt, freqList, Elw, eList, tauList, xticks, yticks = inter.init_pulse()


print(inter.n_BK7_float(inter.w0))














# L = 64000

# H = inter.slab_transfer(inter.n_air,inter.n_BK7,L)
# disps = H(freqList)[:]
# phases = np.angle(disps)
# norms = np.abs(disps)

# eps = eList[-1]
# disps_old = np.exp(1j * eps * (freqList - inter.w0)**2)

# inter.quick_plot(phases,xvals=freqList,file='test_Hphases',square=False)
# inter.quick_plot(norms,xvals=freqList,file='test_Hnorms',square=False)

# inter.quick_plot(np.angle(disps_old),xvals=freqList,file='test_epsPhases',square=False)
# inter.quick_plot(np.abs(disps_old),xvals=freqList,file='test_epsNorms',square=False)

# r1 = inter.ref_coeff(inter.n_air,inter.n_BK7)
# r2 = inter.ref_coeff(inter.n_BK7,inter.n_air)
# t1 = inter.trans_coeff(inter.n_air,inter.n_BK7)

# first_refs = r1(freqList)
# second_refs = r2(freqList)
# transes = t1(freqList)
# second_terms = second_refs*transes

# inter.quick_plot(first_refs,xvals=freqList,file='test_r1',square=True,ylims=(0,1))
# inter.quick_plot(second_refs,xvals=freqList,file='test_r2',square=True)
# inter.quick_plot(transes,xvals=freqList,file='test_t1',square=True)
# inter.quick_plot(second_terms,xvals=freqList,file='test_secondTerms',square=True,ylims=(0,1))
# inter.quick_plot(first_refs**2+inter.n_BK7(freqList)/inter.n_air(freqList)*transes**2,xvals=freqList,file='test_tot1',square=False,ylims=(0,2))


# phases_old = np.angle(disps_old)

# delays = np.imag(np.gradient(disps)/disps)
# del_delays = np.gradient(delays)
# delays_old = np.imag(np.gradient(disps_old)/disps_old)
# del_delays_old = np.gradient(delays_old)

# inter.quick_plot(delays,xvals=freqList,file='test_delays',square=False,ylims=(-5,5))
# inter.quick_plot(del_delays,xvals=freqList,file='test_delDelays',square=False)
# inter.quick_plot(delays_old,xvals=freqList,file='test_delays_old',square=False)
# inter.quick_plot(del_delays_old,xvals=freqList,file='test_delDelays_old',square=False,ylims=(0,1))



