import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.optimize import minimize, least_squares
from scipy.special import erf
import csv
import os
from multiprocessing import Pool, cpu_count, shared_memory
import pandas as pd

# ---------------------- Constants & Parameters ------------------------

print(f"Top-level running in PID {os.getpid()} (name={__name__})")

Npts = 2**21
tmin = 0.0
tmax = 1600000.0
t0 = 800000.0  # fs
sigma = 10.0   # fs
A = 180337     # fs^2
Aerf = 8300
Aerfsuper = 7450
directpath = os.path.dirname(os.path.abspath(__file__))

deltat = (tmax - tmin) / (Npts - 1)
c = 299792458  # m/s
w0 = 2 * np.pi * c / 800 *  1e-6 # rad/fs

# Spectral ranges (in index space for now)
# low = 299753 - 360
# high = 299833 + 360
# lowplot = 299753
# highplot = 299833


#new and improved spectral ranges
low = 1199168 - 1600
high = 1199168 + 1600
lowplot = 1199168 - 160
highplot = 1199168 + 160
wRange = highplot - lowplot

_shared_shape = None
_shared_dtype = None
_shared_names = None
_shared_E1 = None
_shared_E2 = None
_shared_disp = None
_shared_freq = None

#----------------------- Initializer function cuz ChatGPT said so ------------
def init_worker(shm_names, shape, dtype_str, dtype_str_freq):
    global _shared_shape, _shared_dtype, _shared_dtype_freq, _shared_names
    global _shared_E1, _shared_E2, _shared_disp, _shared_freq
    global _shm_E1_sub, _shm_E2_sub, _shm_disp_sub, _shm_freq_sub

    _shared_shape = shape
    _shared_dtype = np.dtype(dtype_str)
    _shared_dtype_freq = np.dtype(dtype_str_freq)
    _shared_names = shm_names

    _shm_E1_sub = shared_memory.SharedMemory(name=shm_names[0])
    _shm_E2_sub = shared_memory.SharedMemory(name=shm_names[1])
    _shm_disp_sub = shared_memory.SharedMemory(name=shm_names[2])
    _shm_freq_sub = shared_memory.SharedMemory(name=shm_names[3])

    _shared_E1 = np.ndarray(_shared_shape, dtype=_shared_dtype, buffer=_shm_E1_sub.buf)
    _shared_E2 = np.ndarray(_shared_shape, dtype=_shared_dtype, buffer=_shm_E2_sub.buf)
    _shared_disp = np.ndarray(_shared_shape, dtype=_shared_dtype, buffer=_shm_disp_sub.buf)
    _shared_freq = np.ndarray(_shared_shape, dtype=_shared_dtype_freq, buffer=_shm_freq_sub.buf)


#----------------------- Plot function for debugging --------------------------
def quick_plot(data: np.ndarray, xvals=None, show=False, file=None, xlims=None):
    if xvals is None:
        plt.plot(np.abs(data)**2)
    else:
        plt.plot(xvals, np.abs(data)**2)

    if xlims is not None:
        plt.xlim(xlims)
    if file:
        plt.savefig(file + '.png')
    if show:
        plt.show()
    plt.close()

# ---------------------- Time & Frequency Axes ------------------------

#Top level stuff I'm removing
# tlist = np.linspace(tmin, tmax, Npts)
# dt = tlist[1] - tlist[0]


# freqList = 2 * np.pi * fftfreq(Npts, d=dt)  # Angular freq

#Top Level stuff I'm removing
#I don't like fftfreq since I want the positive ones. So I'll just build it myself
# freqList = 2*np.pi*np.arange(Npts)/(Npts*deltat) # rad/s

# ---------------------- Gaussian Pulse ------------------------

def ELaser(t):
    return np.exp(-2 * np.log(2) * ((t - t0) / sigma)**2) * np.exp(1j * w0 * t)

#Top level stuff I'm removing
# El = np.array([ELaser(t) for t in tlist])
# Elw = fft(El)


# ---------------------- Chirp Phase Function ------------------------

# Parameters to be set at runtime:
# b and sigma_s (sigma_s = parameter in chirp width)

# def chirp_phase(w, w0, b, sigma_s):
#     x = 2 * (w - w0) * sigma_s / np.sqrt(2 * np.log(256))
#     return b * ((np.exp(-x**2) - 1) / np.sqrt(np.pi) + x * erf(x))

#Step function group delay
def chirp_phase(w, b, sigma_s):
    return b * np.abs(w - w0)


# ---------------------- L Range Setup ------------------------

Lvals = np.arange(0, 64001, 800)  # fs^2 units or length in mm?
hotLvals = np.arange(0, 64001, 8000)  # for heatmaps
coldLvals = np.arange(0,64001,8000)  # for fit plots



#Top level stuff I'm removing
# l_eps_data = pd.read_csv('eps_vs_L_BK7.csv')
# # eList = l_eps_data[l_eps_data['L'].isin(Lvals)] #Takes wanted epsilons from the csv

# epsilon_dict = dict(zip(l_eps_data['L'], l_eps_data['epsilon']))
# eList = [epsilon_dict[L] for L in Lvals]

# ---------------------- Tau Range & Plotting Axes ------------------------

#Top level stuff I'm removing
# tauList = np.arange(-50, 50, 0.5)
# xticks = [(i, tauList[i]) for i in range(0, len(tauList), 10)]

# yticks = [(i, round(2 * np.pi * c / (freqList[i + (lowplot - 1)]) * 1e-6, 2))
#           for i in range(0, wRange + 10, 20)]

# ---------------------- Fit function ------------------------

def fitfunc(t, a1, a2, T0, sigmaFWHM):
    return a1 * (1 - a2 * np.exp(-4 * np.log(2) * (t - T0)**2 / sigmaFWHM**2))


# ----------------------- Preliminaries -----------------------

#I need: 

def init_pulse():

    tlist = np.linspace(tmin, tmax, Npts)
    dt = tlist[1] - tlist[0]

    freqList = 2*np.pi*np.arange(Npts)/(Npts*deltat)

    El = np.array([ELaser(t) for t in tlist])
    Elw = fft(El)

    l_eps_data = pd.read_csv('eps_vs_L_BK7.csv')
    epsilon_dict = dict(zip(l_eps_data['L'], l_eps_data['epsilon']))
    eList = [epsilon_dict[L] for L in Lvals]

    tauList = np.arange(-50, 50, 0.5)

    xticks = [(i, tauList[i]) for i in range(0, len(tauList), 10)]
    yticks = [(i, round(2 * np.pi * c / (freqList[i + (lowplot - 1)]) * 1e-6, 2))
          for i in range(0, wRange + 10, 20)]
    
    return tlist, dt, freqList, Elw, eList, tauList, xticks, yticks



#--------------- Function for the subprocesses ----------------

def compute_row(tau):
    tdelayList = np.exp(1j * _shared_freq * tau)
    ESFGt = ifft(_shared_E1 * _shared_disp) * ifft(_shared_E2 * tdelayList)
    ESFGw = fft(ESFGt)
    intensity = np.abs(ESFGw)**2
    # print('Doing something with this tau',tau)
    return intensity



# ---------------------- Main Interference Simulation ------------------------

def interfere(rules: dict, filenamedips, filenamewidths, filenamechisqs, params):
    print(f"MID-level running in PID {os.getpid()} (name={__name__})")
    
    tlist, dt, freqList, Elw, eList, tauList, xticks, yticks = params

    b = rules.get('b', 0.0)
    sigma_s = rules.get('sigma_s', sigma)  # default to pulse width if not given

    phiList = chirp_phase(freqList, b, sigma_s)
    Ec = Elw * np.exp(1j * phiList)
    Ea = Elw * np.exp(-1j * phiList)
    E1 = Ec + Ea
    E2 = Ec - Ea

    # SHARED MEMORY SETUP
    shape = E1.shape
    dtype_str = str(E1.dtype)
    dtype_str_freq = str(freqList.dtype)

    shm_E1 = shared_memory.SharedMemory(create=True, size=E1.nbytes)
    shm_E2 = shared_memory.SharedMemory(create=True, size=E2.nbytes)
    shm_disp = shared_memory.SharedMemory(create=True, size=E1.nbytes)  # same shape as others
    shm_freq = shared_memory.SharedMemory(create=True, size=freqList.nbytes)

    E1_shared = np.ndarray(shape, dtype=E1.dtype, buffer=shm_E1.buf)
    E2_shared = np.ndarray(shape, dtype=E2.dtype, buffer=shm_E2.buf)
    disp_shared = np.ndarray(shape, dtype=E1.dtype, buffer=shm_disp.buf)
    freq_shared = np.ndarray(shape, dtype=freqList.dtype, buffer=shm_freq.buf)

    print("E1_shared element: ", E1_shared[0])
    print("E1 element: ", E1[0])


    E1_shared[:] = E1[:]
    E2_shared[:] = E2[:]  # disp_shared[:] will be filled inside the L loop
    freq_shared[:] = freqList[:]

    widths = np.zeros(len(Lvals))
    chisqs = np.zeros(len(Lvals))

    os.makedirs(os.path.join(directpath, 'results', filenamedips), exist_ok=True)

    shm_names = (shm_E1.name, shm_E2.name, shm_disp.name, shm_freq.name)
    print('this thing', shm_E1.name)
    print('ill try to just make an array with this now')
    testfreq=np.ndarray(shape, dtype=np.dtype(dtype_str_freq), buffer=shm_freq.buf)
    print("testfreq element: ", testfreq[0])
    print('did it')

    # return None

    with Pool(initializer=init_worker, initargs=(shm_names, shape, dtype_str, dtype_str_freq)) as pool:

        for k, L in enumerate(Lvals):
            eps = eList[k]
            # dispList = np.exp(1j * eps * (freqList - w0)**2)

            # # Parallelize over tau
            # with Pool() as pool:
            #     args_list = [(tau, E1, E2, dispList) for tau in tauList]
            #     data = pool.map(compute_row, args_list)

            dispList = np.exp(1j * eps * (freqList - w0)**2)
            disp_shared[:] = dispList[:]

            print('starting:',L)

            # args_list = [(tau, shm_names, shape, dtype_str) for tau in tauList]
            data = pool.map(compute_row, tauList)
            data = np.array(data)

            print('finishing',L)

            # Export heatmap for selected L values
            if L in hotLvals:
                from matplotlib.colors import Normalize
                maxI = np.max(data[:, lowplot:highplot])
                norm = Normalize(vmin=0, vmax=maxI)
                plt.figure(figsize=(8, 6))
                plt.imshow(
                    data[:, lowplot:highplot].T / maxI,
                    aspect='auto',
                    cmap='rainbow',
                    origin='lower',
                    extent=[tauList[0], tauList[-1],
                            2 * np.pi * c / freqList[highplot] * 1e-6,
                            2 * np.pi * c / freqList[lowplot] * 1e-6]
                )
                plt.colorbar(label="Normalized Intensity")
                plt.xlabel("τ (fs)")
                plt.ylabel("λ (nm)")
                plt.title(f"L = {L}")
                plt.tight_layout()
                plt.savefig(os.path.join(directpath, 'results', filenamedips, f"heat_{L}.png"))
                plt.close()

            # Dip calculation
            d = np.sum(data[:, low:high], axis=1)
            dip_file = os.path.join(directpath, 'results', filenamedips, f"dip_{L}.txt")
            np.savetxt(dip_file, d, delimiter=",")

            # Fit to extract width
            # def chisq(params):
            #     a1, a2, T0, sigmaFWHM = params
            #     model = fitfunc(tauList, a1, a2, T0, sigmaFWHM)
            #     return np.sum((d - model)**2)

            def residuals(params, taus, ys):
                a1, a2, T0, sigmaFWHM = params
                model = fitfunc(taus, a1, a2, T0, sigmaFWHM)
                return ys - model

            # init_guess = [0.0025, 0.9, 0, 10.0]
            init_guess = [6360,0.9,0,10.0]

            # res = minimize(chisq, init_guess)
            res = least_squares(residuals, init_guess, args=(tauList, d), method='lm', 
                               ftol=1e-15, xtol=1e-15, gtol=1e-15)
            widths[k] = np.abs(res.x[3])  # sigmaFWHM
            # chisqs[k] = res.fun
            chisqs[k] = 2 * res.cost

            

            # Save plot for cold values
            if L in coldLvals:
                fit_curve = fitfunc(tauList, *res.x)
                plt.figure()
                plt.plot(tauList, d, label="Data", color='blue')
                plt.plot(tauList, fit_curve, label=f"Fit (FWHM = {np.abs(res.x[3]):.2f})", color='red')
                plt.title(f"L = {L}, σ = {np.abs(res.x[3]):.2f}")
                plt.legend()
                plt.xlabel("τ (fs)")
                plt.ylabel("Integrated Intensity")
                plt.tight_layout()
                plt.savefig(os.path.join(directpath, 'results', filenamedips, f"dip_{L}.png"))
                plt.close()

    # Export width and chisq data
    np.savetxt(os.path.join(directpath, 'results', filenamewidths), np.column_stack((Lvals, widths)), delimiter=",")
    np.savetxt(os.path.join(directpath, 'results', filenamechisqs), np.column_stack((Lvals, chisqs)), delimiter=",")

    shm_E1.close(); shm_E1.unlink()
    shm_E2.close(); shm_E2.unlink()
    shm_disp.close(); shm_disp.unlink()
    shm_freq.close(); shm_freq.unlink()


# ---------------------- Entry Point for Execution ------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chirped pulse interferometer simulation")
    parser.add_argument("--b", type=float, default=0.0, help="Chirp parameter b (fs^2)")
    parser.add_argument("--sigma_s", type=float, default=sigma, help="Spectral chirp width (fs)")
    parser.add_argument("--output", type=str, default="results", help="Output subfolder name")
    args = parser.parse_args()

    rules = {"b": args.b, "sigma_s": args.sigma_s}
    filenamedips = args.output
    filenamewidths = filenamedips+f"widths_b{args.b:.1f}_s{args.sigma_s:.1f}.csv"
    filenamechisqs = filenamedips+f"chisq_b{args.b:.1f}_s{args.sigma_s:.1f}.csv"

    filenamewidths = filenamedips+"widths.csv"
    filenamechisqs = filenamedips+"chisqs.csv"

    #Moving the heavier toplevel code into init_pulse()

    tlist, dt, freqList, Elw, eList, tauList, xticks, yticks = init_pulse()
    params = tlist, dt, freqList, Elw, eList, tauList, xticks, yticks

    print(f"Running interfere() with b={args.b}, sigma_s={args.sigma_s}")
    interfere(rules, filenamedips, filenamewidths, filenamechisqs, params)

    # rules={'b': 8300,'sigma_s': 10}
    # print("now try the big one")
    # interfere(rules,"testdips","testwidths","testchisqs")
    # print("yay")




if __name__ == "__main__":
    main()
