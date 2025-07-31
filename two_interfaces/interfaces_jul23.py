import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.optimize import minimize, least_squares
from scipy.special import erf, erfc, erfinv
import csv
import os
from multiprocessing import Pool, cpu_count, shared_memory
import pandas as pd
from typing import Callable
import re

# ---------------------- Constants & Parameters ------------------------

print(f"Top-level running in PID {os.getpid()} (name={__name__})")

Npts = 2**21
tmin = 0.0
tmax = 1600000.0
t0 = 800000.0  # fs
sigma = 10.0   # fs
# A = 180337     # fs^2

#fake A
# A = 180337


Aerf = 8300
Aerfsuper = 7450
directpath = os.path.dirname(os.path.abspath(__file__))

deltat = (tmax - tmin) / (Npts - 1)
c = 299792458  # m/s
w0 = 2 * np.pi * c / 800 *  1e-6 # rad/fs

#for plotting the chirped pulse
plot_pulse = False

# Spectral ranges (in index space for now)
# low = 299753 - 360
# high = 299833 + 360
# lowplot = 299753
# highplot = 299833


# #new and improved spectral ranges
# low = 1199168 - 1600
# high = 1199168 + 1600
# lowplot = 1199168 - 160
# highplot = 1199168 + 160
# wRange = highplot - lowplot

# #THESE ARE FAKES FOR TESTING just playing around :)
# low = 1199168 - 16
# high = 1199168 + 16
# lowplot = 1199168 - 16
# highplot = 1199168 + 16

# low = 1199170 - 16
# high = 1199170 + 16
# lowplot = 1199170 - 2000
# highplot = 1199170 + 1001


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
def quick_plot(data: np.ndarray, xvals=None, show=False, file=None, xlims=None, ylims=None, square=True):
    if square:
        data = np.abs(data)**2
    if xvals is None:
        plt.plot(data)
    else:
        plt.plot(xvals, data)

    if xlims is not None:
        plt.xlim(xlims)

    if ylims is not None:
        plt.ylim(ylims)

    if file:
        plt.savefig(file + '.png')
    if show:
        plt.show()
    plt.close()

# ---------------------- Gaussian Pulse ------------------------

def ELaser(t):
    return np.exp(-2 * np.log(2) * ((t - t0) / sigma)**2) * np.exp(1j * w0 * t)

#Linear chirp
def lin_ch(w, params):
    """Linear chirp function. params = A, w0"""
    A, w0 = params
    return A * (w-w0)**2

def erf_ch(w, params):
    """Erf/super erf chirp. For erf, let sigma_s = 10. params = b, sigma_s, w0"""
    b, sigma_s, w0 = params
    x = 2 * (w - w0) * sigma_s / np.sqrt(2 * np.log(256))
    return b * ((np.exp(-x**2) - 1) / np.sqrt(np.pi) + x * erf(x))

# Step function group delay
def step_ch(w, params):
    """Step function group delay chirp. params = b, w0"""
    b, w0 = params
    return b * np.abs(w - w0)


def barc_lin_ch(w, params):
    """Barc linear chirp. params = B, w0"""
    B, w0 = params
    return B * (w - w0) * np.abs(w - w0)

def barc_erf_ch(w, params):
    """Barc erf/super chirp. For erf, let sigma_s = 10. params = b, sigma_s, w0"""
    b, sigma_s, w0 = params
    x = 2 * (w - w0) * sigma_s / np.sqrt(2 * np.log(256))
    return b * ((np.exp(-x**2) - 1) / np.sqrt(np.pi) + x * erf(x)) * np.sign(x)


def barc_erf_sh_ch_flt(w, params):
    """Barc erf/super chirp where the group delay range is centred at zero. For erf, let sigma_s = 10. params = b, sigma_s, w0"""
    b, sigma_s, w0 = params
    x = 2 * (w - w0) * sigma_s / np.sqrt(2 * np.log(256))
    
    if x < 0:
        return 1/2 * b * (-2/np.sqrt(np.pi) * np.exp(-x**2) - 3*x + 2*x*erfc(x) + 2/np.sqrt(np.pi))
    elif x >= 0:
        return 1/2 * b * (2/np.sqrt(np.pi) * np.exp(-x**2) + x - 2*x*erfc(x) - 2/np.sqrt(np.pi))
    else:
        raise Exception("something went wrong with barc_erf_sh_ch. Invalid input?")
    
def barc_erf_sh_ch(w, params):
    """Barc erf/super chirp where the group delay range is centred at zero. For erf, let sigma_s = 10. params = b, sigma_s, w0"""
    b, sigma_s, w0 = params
    x = 2 * (w - w0) * sigma_s / np.sqrt(2 * np.log(256))

    phis = np.empty_like(x, dtype=np.float64)

    mask1 = x < 0
    mask2 = x >= 0

    x1 = x[mask1]
    x2 = x[mask2]

    phis[mask1] = 1/2 * b * (-2/np.sqrt(np.pi) * np.exp(-x1**2) - 3*x1 + 2*x1*erfc(x1) + 2/np.sqrt(np.pi))
    phis[mask2] = 1/2 * b * (2/np.sqrt(np.pi) * np.exp(-x2**2) + x2 - 2*x2*erfc(x2) - 2/np.sqrt(np.pi))

    return phis




def woof_erf_ch_flt(w, params):
    """Woof erf/super chirp where the group delay range is centred at zero. For erf, let sigma_s = 10. params = b, sigma_s, w0"""
    b, sigma_s, w0 = params
    x = 2 * (w - w0) * sigma_s / np.sqrt(2 * np.log(256))

    a = 4*(1-2*np.exp(-erfinv(1/2)**2))/np.sqrt(np.pi)

    if x >= erfinv(1/2):
        return 1/4 * b * (4/np.sqrt(np.pi) * np.exp(-x**2) + x - 4*x*erfc(x) + a)
    elif 0 <= x and x < erfinv(1/2):
        return 1/4 * b * (-4/np.sqrt(np.pi) * np.exp(-x**2) + x - 4*x*erf(x) + 4/np.sqrt(np.pi))
    elif erfinv(-1/2) < x and x < 0:
        return 1/4 * b * (4/np.sqrt(np.pi) * np.exp(-x**2) + x + 4*x*erf(x) - 4/np.sqrt(np.pi))
    elif x <= erfinv(-1/2):
        return 1/4 * b * (-4/np.sqrt(np.pi) * np.exp(-x**2) -7*x + 4*x*erfc(x) - a)
    else:
        raise Exception("something went wrong with woof_erf_ch. Invalid input?")
    
def woof_erf_ch(w, params):
    """Woof erf/super chirp where the group delay range is centred at zero. For erf, let sigma_s = 10. params = b, sigma_s, w0"""
    b, sigma_s, w0 = params
    x = 2 * (w - w0) * sigma_s / np.sqrt(2 * np.log(256))

    a = 4*(1-2*np.exp(-erfinv(1/2)**2))/np.sqrt(np.pi)

    phis = np.zeros_like(x, dtype=np.float64)

    mask1 = x >= erfinv(1/2)
    mask2 = (x >= 0) & (x < erfinv(1/2))
    mask3 = (x > erfinv(-1/2)) & (x < 0)
    mask4 = x <= erfinv(-1/2)

    x1 = x[mask1]
    x2 = x[mask2]
    x3 = x[mask3]
    x4 = x[mask4]

    phis[mask1] = 1/4 * b * (4/np.sqrt(np.pi) * np.exp(-x1**2) + x1 - 4*x1*erfc(x1) + a)
    phis[mask2] = 1/4 * b * (-4/np.sqrt(np.pi) * np.exp(-x2**2) + x2 - 4*x2*erf(x2) + 4/np.sqrt(np.pi))
    phis[mask3] = 1/4 * b * (4/np.sqrt(np.pi) * np.exp(-x3**2) + x3 + 4*x3*erf(x3) - 4/np.sqrt(np.pi))
    phis[mask4] = 1/4 * b * (-4/np.sqrt(np.pi) * np.exp(-x4**2) - 7*x4 + 4*x4*erfc(x4) - a)

    return phis

def woof_lin_ch(w, params):
    """Woof lin chirp where the group delay range is centred at zero. T is the total range over which we fold. params = A, T, w0"""
    
    A, T, w0 = params

    phis = np.zeros_like(w, dtype=np.float64)
    
    mask1 = w-w0 < -T/(8*A)
    mask2 = (w-w0 >= -T/(8*A)) & (w-w0 < 0)
    mask3 = (w-w0 >= 0) & (w-w0 < T/(8*A))
    mask4 = w-w0 >= T/(8*A)

    w1 = w[mask1]
    w2 = w[mask2]
    w3 = w[mask3]
    w4 = w[mask4]

    phis[mask1] = -A*(w1-w0)**2 - 3*T/8 * (w1-w0) - T**2/(32*A)
    phis[mask2] = A*(w2-w0)**2 + T/8 * (w2-w0)
    phis[mask3] = -A*(w3-w0)**2 + T/8 * (w3-w0)
    phis[mask4] = A*(w4-w0)**2 - 3*T/8 * (w4-w0) + T**2/(32*A)

    return phis

def barc_lin_sh_ch(w, params):
    """Barc lin chirp where the group delay range is centred at zero. T is the total range over which we fold. params = A, T, w0"""

    A, T, w0 = params

    phis = np.zeros_like(w, dtype=np.float64)

    mask1 = w-w0 > 0
    mask2 = w-w0 < 0

    w1 = w[mask1]
    w2 = w[mask2]

    phis[mask1] = A * (w1-w0)**2 - T/4 * (w1-w0)
    phis[mask2] = -A * (w2-w0)**2 - T/4 * (w2-w0)

    return phis



# -------------- Interfaces: Replaces the Dispersion ------------

#Two interfaces

def ref_coeff(n1: Callable, n2: Callable) -> Callable:
    """calculates the amplitude reflection coefficient at normal incidence, where n1 is the incident/reflected medium and n2
    is the transmitted medium."""

    def r(w):
        return (n1(w)-n2(w))/(n1(w)+n2(w))

    return r

def trans_coeff(n1: Callable, n2: Callable) -> Callable:
    """calculates the amplitude transmission coefficient at normal incidence, where n1 is the incident/reflected medium
    and n2 is the transmitted medium."""

    def t(w):
        return 2*n1(w)/(n1(w)+n2(w))

    return t

def n_air(w: float) -> float:
    
    return 1

def n_BK7_float(w: float) -> float:
    """Calculates index of refraction of BK7 via Sellmeier equation"""

    c = 0.2998 #um/fs
    b1 = 1.03961212
    b2 = 0.231792344
    b3 = 1.01046945
    c1 = 6.00069867e-3
    c2 = 2.00179144e-2
    c3 = 1.03560653e2

    if w == 0:
        return np.sqrt(1+b1+b2+b3)

    x = (2*np.pi*c/w)**2

    nsq = max(1 + b1*x/(x-c1) + b2*x/(x-c2) + b3*x/(x-c3),1e-3)

    
    return np.sqrt(nsq)

#from una, edited; I just need the derivative
def k_deriv_BK7(w0):
    
    c = 0.2998; #(*um/fs*)

    b1 = 1.03961212
    b2 = 0.231792344
    b3 = 1.01046945
    c1 = 6.00069867e-3
    c2 = 2.00179144e-2
    c3 = 1.03560653e2
    
    k_deriv = np.sqrt(1 +((4*b1*(c**2)*(np.pi**2))/(-c1*(w0**2) + (2*c*np.pi)**2)) + 
                      ((4*b2*(c**2)*(np.pi**2))/(-c2*(w0**2) + (2*c*np.pi)**2)) + 
                      ((4*b3*(c**2)*(np.pi**2))/(-c3*(w0**2) + (2*c*np.pi)**2)))/c + w0*(
                          (32*b1*(c*np.pi)**4)/((w0**5)*((-c1 + ((2*c*np.pi)**2)/(w0**2))**2)) +
                          (32*b2*(c*np.pi)**4)/((w0**5)*((-c2 + ((2*c*np.pi)**2)/(w0**2))**2)) +
                          (32*b3*(c*np.pi)**4)/((w0**5)*((-c3 + ((2*c*np.pi)**2)/(w0**2))**2)) -
                          (8*b1*(c*np.pi)**2)/((w0**3)*(-c1 + ((2*c*np.pi)**2)/(w0**2))) -
                          (8*b2*(c*np.pi)**2)/((w0**3)*(-c2 + ((2*c*np.pi)**2)/(w0**2))) -
                          (8*b3*(c*np.pi)**2)/((w0**3)*(-c3 + ((2*c*np.pi)**2)/(w0**2))))/(
                              2*c*np.sqrt(1 + (b1*(2*np.pi*c/w0)**2)/((2*np.pi*c/w0)**2 - c1) + 
                                          (b2*(2*np.pi*c/w0)**2)/((2*np.pi*c/w0)**2 - c2) + 
                                          (b3*(2*np.pi*c/w0)**2)/((2*np.pi*c/w0)**2 - c3)))

    return k_deriv



#the vectorized function
n_BK7 = np.vectorize(n_BK7_float)


def slab_transfer(n1: Callable, n2: Callable, L: float) -> Callable:
    """n1 is exterior to the slab, usually air, and n2 is inside the material"""

    c = 0.2998 #um/fs

    #The reflection and transmission coeffs:
    r1 = ref_coeff(n1,n2)
    r2 = ref_coeff(n2,n1)
    t1 = trans_coeff(n1,n2)

    def H(w):
        return r1(w)*np.exp(-1j*L*w*n2(w)/c) + t1(w)*r2(w)*np.exp(1j*L*w*n2(w)/c)
    
    return H

def slab_frontheavy(n1: Callable, n2: Callable, L: float) -> Callable:

    c = 0.2998 #um/fs

    r1 = -1
    r2 = 0
    t1 = 0

    def H(w):
        return r1*np.exp(-1j*L*w*n2(w)/c) + t1*r2*np.exp(1j*L*w*n2(w)/c)
    
    return H

def slab_backheavy(n1: Callable, n2: Callable, L: float) -> Callable:

    c = 0.2998 #um/fs

    r1 = 0
    r2 = 1
    t1 = 1

    def H(w):
        return r1*np.exp(-1j*L*w*n2(w)/c) + t1*r2*np.exp(1j*L*w*n2(w)/c)
    
    return H

def slab_lossless(n1: Callable, n2: Callable, L: float) -> Callable:

    c = 0.2998 #um/fs

    r1 = -1/np.sqrt(2)
    r2 = 1/np.sqrt(2)
    t1 = 1

    def H(w):
        return r1*np.exp(-1j*L*w*n2(w)/c) + t1*r2*np.exp(1j*L*w*n2(w)/c)
    
    return H

#For adding bulk dispersion

def dispy_transfer_gen(transfer_generator: Callable, params: tuple, n_bulk: Callable, D: float, w0: float, k_deriv: Callable):
    """Returns the transfer function for a systyem consisting of a thin slab of dispersive material, shielded by some other dispersive material"""


    c = 0.2998 #um/fs

    h = transfer_generator(*params)

    def H(w):
        return np.exp(1j*D*w*n_bulk(w)/c) * h(w) * np.exp(1j*D*w*n_bulk(w)/c) * np.exp(-2j*D*k_deriv(w0)*w)
    
    return H




# ---------------------- L Range Setup ------------------------

# Lvals = np.arange(0, 64001, 32000)  # fs^2 units or length in mm?
# hotLvals = np.arange(0, 64001, 32000)  # for heatmaps
# coldLvals = np.arange(0,64001,32000)  # for fit plots


# Lvals = np.array([0,800,32000,64000])
# Lvals = np.array([32000])
# hotLvals = Lvals
# coldLvals = Lvals

# Lvals = np.arange(0, 64001, 800)  # fs^2 units or length in mm?
# hotLvals = np.arange(0, 64001, 8000)  # for heatmaps
# coldLvals = np.arange(0,64001,8000)

# Lvals = np.arange(10,10.4,0.008)  # fs^2 units or length in mm?

# Lvals = np.arange(1,11)
Lvals = np.array([10])
hotLvals = Lvals # for heatmaps
coldLvals = Lvals

Dvals = np.array([0,3000])
hotDvals = Dvals
coldDvals = hotDvals






# ---------------------- Fit function ------------------------

def fitfunc(t, a1, a2, T0, sigmaFWHM):
    return a1 * (1 - a2 * np.exp(-4 * np.log(2) * (t - T0)**2 / sigmaFWHM**2))


# ----------------------- Preliminaries -----------------------

def init_pulse():

    tlist = np.linspace(tmin, tmax, Npts)
    dt = tlist[1] - tlist[0]

    freqList = 2*np.pi*np.arange(Npts)/(Npts*deltat)

    El = np.array([ELaser(t) for t in tlist])
    Elw = fft(El)

    #for smaller L can't use this
    l_eps_data,epsilon_dict,eList = None,None,None
    # l_eps_data = pd.read_csv('eps_vs_L_BK7.csv')
    # epsilon_dict = dict(zip(l_eps_data['L'], l_eps_data['epsilon']))
    # eList = [epsilon_dict[L] for L in Lvals]

    tauList = np.arange(-200, 200, 1)
    # tauList = np.arange(-200000,200000,1000)
    
    return tlist, dt, freqList, Elw, eList, tauList


#----------------- Heat Map Plotting Helpers -------------------

def fancy_round(x):
    """Rounds x to the greatest lesser or equal number with one sig. fig. ending in a 1, 2, or 5"""

    exp = np.floor(np.log10(x))
    y = x * 10 ** (-exp)
    
    if y >= 1 and y < 2:
        return 10 ** exp
    elif y >= 2 and y < 5:
        return 2 * 10 ** exp
    elif y >= 5 and y < 10:
        return 5 * 10 ** exp
    else:
        raise Exception("fancy_round is fancy_wrong", x, exp, y)


def wavelength_position(lamb, freqList, lowplot, highplot):

    w = 2 * np.pi * c / lamb * 1e-6

    delta_w = 2 * np.pi / (Npts * deltat)

    if delta_w != freqList[1] - freqList[0]:
        raise Exception("ill defined freq increment; these should be equal: ", delta_w, freqList[1] - freqList[0])

    wmin = freqList[lowplot]
    wmax = freqList[highplot]

    return (wmax - w) / (wmax - wmin)



def get_wavelength_ticks(freqList, lowplot, highplot):

    # Frequencies and corresponding wavelengths
    freqs = freqList[lowplot:highplot]
    lambdas = 2 * np.pi * c / freqs * 1e-6  # in nm

    lam_min = lambdas.min()
    lam_max = lambdas.max()

    lam_range = lam_max-lam_min

    inc = fancy_round(lam_range/4)

    tick_vals = np.array([400 + i * inc for i in np.arange(-5,6)])
    tick_vals = np.array([tick for tick in tick_vals if tick >= lam_min and tick <= lam_max])

    tick_locs = [wavelength_position(tick, freqList, lowplot, highplot) for tick in tick_vals]

    tick_labels = [f"{tick:.3f}" for tick in tick_vals]

    return tick_locs, tick_labels


# ------------------------- Bandwidth ----------------------

def bandwidth_cutoff(width, freqList):
    """width: detector bandwidth in nanometres, centred at 400 nm. Returns indices"""

    lmin = 400 - width/2
    lmax = 400 + width/2

    waveList = 2 * np.pi * c / freqList * 1e-6

    # remember the wavelengths decrease along the list
    low = np.argmin(np.abs(waveList-lmax))
    high = np.argmin(np.abs(waveList-lmin)) + 1

    if low > high:
        raise Exception("Something is wrong with these indices", low, high)

    if low == 0 or high == len(waveList) - 1:
        raise Exception("Bandwidths is too large")
    

    return low, high


# ---------------------- Saving Parameters ------------------

def extract_param_names(func):
    """Wisdom of chatgpt; seems to work. It's supposed to get a list of the parameter names as written in the docstring of func."""

    doc = func.__doc__
    if not doc:
        return []

    # Look for a line containing "params ="
    match = re.search(r'params\s*=\s*(.+)', doc)
    if match:
        param_str = match.group(1)
        # Split by comma, strip whitespace
        return [p.strip() for p in param_str.split(',')]
    return []


def save_simulation_metadata(output_dir: str, filename: str, params: tuple, Lvals, setup, Dvals = None):
    """Saves simulation metadata (e.g. chirp and transfer function types, parameters) to a text file."""

    os.makedirs(output_dir, exist_ok=True)

    tlist, dt, freqList, Elw, eList, tauList, det_bandwidth, plot_bandwidth, transfer_generator, chirp, chirp_params = params

    outpath = os.path.join(output_dir, filename)

    with open(outpath, "w") as f:
        f.write("=== Simulation Metadata ===\n\n")

        f.write("Chirp Function:\n")
        f.write(f"  Name: {chirp.__name__}\n")
        f.write(f"  Parameters:\n")
        param_names = extract_param_names(chirp)
        if len(param_names) == len(chirp_params):
            for i, param in enumerate(chirp_params):
                f.write(f"    {param_names[i]}: {param}\n")
        else:
            for param in chirp_params:
                f.write(f"    {param}\n")
        f.write("\n")

        f.write(f"Setup: {setup} \n\n")

        f.write("Transfer Function:\n")
        f.write(f"  Name: {transfer_generator.__name__}\n\n")

        f.write("Detector Bandwidth: {:.3f} nm\n".format(det_bandwidth))
        f.write("Heatmap Bandwidth: {:.3f} nm\n\n".format(plot_bandwidth))

        f.write(f"Time Range: {min(tlist):.3f} -- {max(tlist):.3f} fs\n")
        f.write("Number of Time Points: {}\n".format(len(tlist)))
        f.write("Time Step (dt): {:.3f} fs\n\n".format(dt))
        

        f.write(f"Time Delay Range: {min(tauList):.3f} -- {max(tauList):.3f} fs\n")
        f.write(f"Number of Time Delays: {len(tauList)}\n")
        f.write(f"Time Delay Step: {tauList[min(1,len(tauList)-1)] - tauList[0]:.3f} fs\n\n")

        f.write(f"Frequency Range: {min(freqList):.3f} -- {max(freqList):.3f} rad/fs\n")
        f.write(f"Number of Frequencies: {len(freqList)}\n")
        f.write(f"Frequency Step: {freqList[min(1,len(freqList)-1)] - freqList[0]:.3e} rad/fs\n\n")

        f.write(f"Length Range: {min(Lvals):.2f} -- {max(Lvals):.2f} um\n")
        f.write(f"Number of Lvals: {len(Lvals)}\n")
        f.write(f"L Step, Maybe: {Lvals[min(1,len(Lvals)-1)] - Lvals[0]:.2f} um\n")
        f.write(f"All Lvals, Maybe: ")
        if len(Lvals) < 5:
            for L in Lvals:
                f.write(f"{L:.2f}  ")
        else:
            f.write("N/A")
        f.write("\n\n")

        if Dvals is not None:
            f.write(f"Bulk Length Range: {min(Dvals):.2f} -- {max(Dvals):.2f} um\n")
            f.write(f"Number of Dvals: {len(Dvals)}\n")
            f.write(f"D Step, Maybe: {Dvals[min(1,len(Dvals)-1)] - Dvals[0]:.2f} um\n")
            f.write(f"All Dvals, Maybe: ")
            if len(Dvals) < 5:
                for D in Dvals:
                    f.write(f"{D:.2f}  ")
            else:
                f.write("N/A")
            f.write("\n\n") 

        f.write(f"Initial Pulse Width: {sigma:.3f} fs \n")







#--------------- Function for the subprocesses ----------------

#Ordinary Dispersion
def compute_row(tau):
    tdelayList = np.exp(1j * _shared_freq * tau)
    ESFGt = ifft(_shared_E1 * _shared_disp) * ifft(_shared_E2 * tdelayList)
    ESFGw = fft(ESFGt)
    intensity = np.abs(ESFGw)**2
    # print('Doing something with this tau',tau)
    # if tau == 0:
    #     print('some stuff:')
    #     print('tdelaylist',tdelayList[42])
    #     print('ESFGt',ESFGt[42])
    #     print('ESFGw',ESFGw[42])
    #     print('intensity',intensity[42])
    return intensity


# ---------------------- Main Interference Simulation ------------------------

def interfere(rules: dict, filenamedips, filenamewidths, filenamechisqs, params, fit=True, setup='pm'):
    print(f"MID-level running in PID {os.getpid()} (name={__name__})")
    
    tlist, dt, freqList, Elw, eList, tauList, det_bandwidth, plot_bandwidth, transfer_generator, chirp, chirp_params = params


    low, high = bandwidth_cutoff(det_bandwidth, freqList)
    lowplot, highplot = bandwidth_cutoff(plot_bandwidth, freqList)

    # b = rules.get('b', 0.0)
    # sigma_s = rules.get('sigma_s', sigma)  # default to pulse width if not given


    phiList = chirp(freqList, chirp_params)
    Ec = Elw * np.exp(1j * phiList)
    Ea = Elw * np.exp(-1j * phiList)
    
    if setup == 'pm':
        E1 = Ec + Ea
        E2 = Ec - Ea
    elif setup == 'pp':
        E1 = Ec + Ea
        E2 = Ec + Ea
    elif setup == 'cc':
        E1 = Ec
        E2 = Ec
    elif setup == 'aa':
        E1 = Ea
        E2 = Ea
    else:
        raise Exception("Invalid setup")

    if plot_pulse:
        Ect = ifft(Ec)
        quick_plot(Ect, xvals=tlist, file=filenamedips, xlims = (600000,1000000))
        return None


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

    # print("E1_shared element: ", E1_shared[0])
    # print("E1 element: ", E1[0])


    E1_shared[:] = E1[:]
    E2_shared[:] = E2[:]  # disp_shared[:] will be filled inside the L loop
    freq_shared[:] = freqList[:]

    widths = np.zeros(len(Lvals))
    chisqs = np.zeros(len(Lvals))

    os.makedirs(os.path.join(directpath, 'results', filenamedips), exist_ok=True)

    save_simulation_metadata(os.path.join(directpath,'results', filenamedips), filenamedips+'meta.txt', params, Lvals, setup)

    shm_names = (shm_E1.name, shm_E2.name, shm_disp.name, shm_freq.name)
    # print('this thing', shm_E1.name)
    # print('ill try to just make an array with this now')
    # testfreq=np.ndarray(shape, dtype=np.dtype(dtype_str_freq), buffer=shm_freq.buf)
    # print("testfreq element: ", testfreq[0])
    # print('did it')

    # return None

    with Pool(initializer=init_worker, initargs=(shm_names, shape, dtype_str, dtype_str_freq)) as pool:

        for k, L in enumerate(Lvals):

            #This block is for the simple dispersion case. Comment out as needed
            # eps = eList[k]
            # dispList = np.exp(1j * eps * (freqList - w0)**2)
            # disp_shared[:] = dispList[:]


            #For the multiple interfaces, disp_shared is just the transfer function
            H = transfer_generator(n_air,n_BK7,L)
            disp_shared[:] = H(freqList)[:]

            # print('disp entry', L, disp_shared[40])

            print('starting:',L)

            # args_list = [(tau, shm_names, shape, dtype_str) for tau in tauList]
            data = pool.map(compute_row, tauList)
            data = np.array(data)

            # print('data entry',L,data[40])
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
                    origin='upper',
                    extent=[tauList[0], tauList[-1],
                            0,1], #use generalized extent and then do some transformations as needed
                    interpolation='none'
                )

                plt.axhline(y=wavelength_position(400,freqList,lowplot,highplot), color='black', linewidth=1.2, linestyle='-')

                if det_bandwidth < plot_bandwidth:
                    plt.axhline(y=wavelength_position(400 + det_bandwidth/2,freqList,lowplot,highplot), color='black', linewidth=0.8, linestyle='--')
                    plt.axhline(y=wavelength_position(400 - det_bandwidth/2,freqList,lowplot,highplot), color='black', linewidth=0.8, linestyle='--')


                plt.colorbar(label="Normalized Intensity")
                plt.xlabel("τ (fs)")
                plt.ylabel("λ (nm)")
                plt.title(f"L = {L}")

                tick_locs, tick_labels = get_wavelength_ticks(freqList, lowplot, highplot)
                plt.yticks(tick_locs, tick_labels)

                plt.tight_layout()
                plt.savefig(os.path.join(directpath, 'results', filenamedips, f"heat_{L:.3f}.png"))
                plt.close()

            d = None
            res = None

            # Dip calculation
            d = np.sum(data[:, low:high], axis=1)
            dip_file = os.path.join(directpath, 'results', filenamedips, f"dip_{L:.3f}.txt")
            np.savetxt(dip_file, d, delimiter=",")

            if fit:

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

            else:
                widths[k] = 0
                chisqs[k] = 0

            # Save plot for cold values
            if L in coldLvals:
                if fit:
                    fit_curve = fitfunc(tauList, *res.x)
                plt.figure()
                plt.plot(tauList, d, label="Data", color='blue')
                if fit:
                    plt.plot(tauList, fit_curve, label=f"Fit (FWHM = {np.abs(res.x[3]):.2f})", color='red')
                if fit:
                    plt.title(f"L = {L}, σ = {np.abs(res.x[3]):.2f}")
                else:
                    plt.title(f"L = {L}")
                plt.legend()
                plt.xlabel("τ (fs)")
                plt.ylabel("Integrated Intensity")
                plt.tight_layout()
                plt.savefig(os.path.join(directpath, 'results', filenamedips, f"dip_{L:.3f}.png"))
                plt.close()
            
            #trying to solve memory problems -- didn't work :(
            del data, d

    # Export width and chisq data
    np.savetxt(os.path.join(directpath, 'results', filenamewidths), np.column_stack((Lvals, widths)), delimiter=",")
    np.savetxt(os.path.join(directpath, 'results', filenamechisqs), np.column_stack((Lvals, chisqs)), delimiter=",")

    shm_E1.close(); shm_E1.unlink()
    shm_E2.close(); shm_E2.unlink()
    shm_disp.close(); shm_disp.unlink()
    shm_freq.close(); shm_freq.unlink()



def interfere_dispy(rules: dict, filenamedips, filenamewidths, filenamechisqs, params, fit=True, setup='pm'):
    """Runs the main interferometry sim, where there is a large amount of bulk dispersion of width D before the slab of width L."""
    
    print(f"MID-level running in PID {os.getpid()} (name={__name__})")
    
    tlist, dt, freqList, Elw, eList, tauList, det_bandwidth, plot_bandwidth, transfer_generator, chirp, chirp_params = params


    low, high = bandwidth_cutoff(det_bandwidth, freqList)
    lowplot, highplot = bandwidth_cutoff(plot_bandwidth, freqList)

    # b = rules.get('b', 0.0)
    # sigma_s = rules.get('sigma_s', sigma)  # default to pulse width if not given


    phiList = chirp(freqList, chirp_params)
    Ec = Elw * np.exp(1j * phiList)
    Ea = Elw * np.exp(-1j * phiList)
    
    if setup == 'pm':
        E1 = Ec + Ea
        E2 = Ec - Ea
    elif setup == 'pp':
        E1 = Ec + Ea
        E2 = Ec + Ea
    elif setup == 'cc':
        E1 = Ec
        E2 = Ec
    elif setup == 'aa':
        E1 = Ea
        E2 = Ea
    else:
        raise Exception("Invalid setup")

    if plot_pulse:
        Ect = ifft(Ec)
        quick_plot(Ect, xvals=tlist, file=filenamedips, xlims = (797500,802500))
        return None


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

    # print("E1_shared element: ", E1_shared[0])
    # print("E1 element: ", E1[0])


    E1_shared[:] = E1[:]
    E2_shared[:] = E2[:]  # disp_shared[:] will be filled inside the L loop
    freq_shared[:] = freqList[:]

    widths = np.zeros(len(Lvals))
    chisqs = np.zeros(len(Lvals))

    os.makedirs(os.path.join(directpath, 'results', filenamedips), exist_ok=True)

    save_simulation_metadata(os.path.join(directpath,'results', filenamedips), filenamedips+'meta.txt', params, Lvals, setup, Dvals=Dvals)

    shm_names = (shm_E1.name, shm_E2.name, shm_disp.name, shm_freq.name)
    # print('this thing', shm_E1.name)
    # print('ill try to just make an array with this now')
    # testfreq=np.ndarray(shape, dtype=np.dtype(dtype_str_freq), buffer=shm_freq.buf)
    # print("testfreq element: ", testfreq[0])
    # print('did it')

    # return None

    with Pool(initializer=init_worker, initargs=(shm_names, shape, dtype_str, dtype_str_freq)) as pool:

        for k, L in enumerate(Lvals):
            

            for m, D in enumerate(Dvals):


                #This block is for the simple dispersion case. Comment out as needed
                # eps = eList[k]
                # dispList = np.exp(1j * eps * (freqList - w0)**2)
                # disp_shared[:] = dispList[:]


                #For the multiple interfaces, disp_shared is just the transfer function
                H = dispy_transfer_gen(transfer_generator, (n_air,n_BK7,L), n_BK7, D, w0, k_deriv_BK7)
                disp_shared[:] = H(freqList)[:]

                # print('disp entry', L, disp_shared[40])

                print(f"starting: L: {L}, D: {D}")

                # args_list = [(tau, shm_names, shape, dtype_str) for tau in tauList]
                data = pool.map(compute_row, tauList)
                data = np.array(data)

                # print('data entry',L,data[40])
                print(f"finishing: L: {L}, D: {D}")

                # Export heatmap for selected L values
                if L in hotLvals and D in hotDvals:
                    from matplotlib.colors import Normalize
                    maxI = np.max(data[:, lowplot:highplot])
                    norm = Normalize(vmin=0, vmax=maxI)
                    plt.figure(figsize=(8, 6))
                    plt.imshow(
                        data[:, lowplot:highplot].T / maxI,
                        aspect='auto',
                        cmap='rainbow',
                        origin='upper',
                        extent=[tauList[0], tauList[-1],
                                0,1], #use generalized extent and then do some transformations as needed
                        interpolation='none'
                    )

                    plt.axhline(y=wavelength_position(400,freqList,lowplot,highplot), color='black', linewidth=1.2, linestyle='-')

                    if det_bandwidth < plot_bandwidth:
                        plt.axhline(y=wavelength_position(400 + det_bandwidth/2,freqList,lowplot,highplot), color='black', linewidth=0.8, linestyle='--')
                        plt.axhline(y=wavelength_position(400 - det_bandwidth/2,freqList,lowplot,highplot), color='black', linewidth=0.8, linestyle='--')


                    plt.colorbar(label="Normalized Intensity")
                    plt.xlabel("τ (fs)")
                    plt.ylabel("λ (nm)")
                    plt.title(f"L = {L}, D = {D}")

                    tick_locs, tick_labels = get_wavelength_ticks(freqList, lowplot, highplot)
                    plt.yticks(tick_locs, tick_labels)

                    plt.tight_layout()
                    plt.savefig(os.path.join(directpath, 'results', filenamedips, f"heat_L{L:.3f}_D{D:.3f}.png"))
                    plt.close()

                d = None
                res = None

                # Dip calculation
                d = np.sum(data[:, low:high], axis=1)
                dip_file = os.path.join(directpath, 'results', filenamedips, f"dip_L{L:.3f}_D{D:.3f}.txt")
                np.savetxt(dip_file, d, delimiter=",")

                if fit:

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

                else:
                    widths[k] = 0
                    chisqs[k] = 0

                # Save plot for cold values
                if L in coldLvals and D in hotDvals:
                    if fit:
                        fit_curve = fitfunc(tauList, *res.x)
                    plt.figure()
                    plt.plot(tauList, d, label="Data", color='blue')
                    if fit:
                        plt.plot(tauList, fit_curve, label=f"Fit (FWHM = {np.abs(res.x[3]):.2f})", color='red')
                    if fit:
                        plt.title(f"L = {L}, D = {D}, σ = {np.abs(res.x[3]):.2f}")
                    else:
                        plt.title(f"L = {L}, D = {D}")
                    plt.legend()
                    plt.xlabel("τ (fs)")
                    plt.ylabel("Integrated Intensity")
                    plt.tight_layout()
                    plt.savefig(os.path.join(directpath, 'results', filenamedips, f"dip_L{L:.3f}_D{D:.3f}.png"))
                    plt.close()
                
                #trying to solve memory problems -- didn't work :(
                del data, d

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
    parser.add_argument("--dband", type = float, default=1.0, help="Detector bandwidth (nm)")
    parser.add_argument("--pband", type=float, default=1.0, help="Heatmap wavelength range")
    parser.add_argument("--output", type=str, default="results", help="Output subfolder name")
    args = parser.parse_args()
    rules = {"b": args.b, "sigma_s": args.sigma_s}
    filenamedips = args.output
    filenamewidths = filenamedips+f"widths_b{args.b:.1f}_s{args.sigma_s:.1f}.csv"
    filenamechisqs = filenamedips+f"chisq_b{args.b:.1f}_s{args.sigma_s:.1f}.csv"

    tlist, dt, freqList, Elw, eList, tauList = init_pulse()

    #"Ordinary" params:
    # b1 = 8300
    # b2 = 8300 * 10/11
    # b3 = 8300 * 10/12
    # s1 = 10.0
    # s2 = 11.0
    # s3 = 12.0

    #I'm going to try now with an order of magnitude less chirp
    # b1 = 830
    # b2 = 830 * 10/11
    # b3 = 830 * 10/12

    #Mazurek params:
    A = 2500/2
    b1 = 8300 * A/180337
    b2 = 8300 * 10/11 * A/180337
    b3 = 8300 * 10/12 * A/180337
    s1 = 10.0
    s2 = 11.0
    s3 = 12.0

    dband = args.dband
    pband = args.pband

    filenameLinLossless = filenamedips + "_lossless_lin_"
    filenameErf1Lossless = filenamedips + "_lossless_erf_s10.0_"
    filenameErf2Lossless = filenamedips + "_lossless_erf_s11.0_"
    filenameErf3Lossless = filenamedips + "_lossless_erf_s12.0_"
    filenameVphLossless = filenamedips + "_lossless_vph_"
    filenameBarcLinLossless = filenamedips + "_lossless_barc_lin_"
    filenameBarcErf1Lossless = filenamedips + "_lossless_barc_erf_s10.0_"
    filenameBarcErf2Lossless = filenamedips + "_lossless_barc_erf_s11.0_"
    filenameBarcErf3Lossless = filenamedips + "_lossless_barc_erf_s12.0_"

    filenameBarcErfSh1Lossless = filenamedips + "_lossless_barc_erf_sh_s10.0_"
    filenameBarcErfSh2Lossless = filenamedips + "_lossless_barc_erf_sh_s11.0_"
    filenameBarcErfSh3Lossless = filenamedips + "_lossless_barc_erf_sh_s12.0_"

    filenameBarcLinShLossless = filenamedips + "_lossless_barc_lin_sh_"
    filenameWoofLinLossless = filenamedips + "_lossless_woof_lin_"

    filenameWoof1Lossless = filenamedips + "_lossless_woof_s10.0_"
    filenameWoof2Lossless = filenamedips + "_lossless_woof_s11.0_"
    filenameWoof3Lossless = filenamedips + "_lossless_woof_s12.0_"

    filenameLinRealistic = filenamedips + "_realistic_lin_"
    filenameErf1Realistic = filenamedips + "_realistic_erf_s10.0_"
    filenameErf2Realistic = filenamedips + "_realistic_erf_s11.0_"
    filenameErf3Realistic = filenamedips + "_realistic_erf_s12.0_"
    filenameVphRealistic = filenamedips + "_realistic_vph_"
    filenameBarcLinRealistic = filenamedips + "_realistic_barc_lin_"
    filenameBarcErf1Realistic = filenamedips + "__realistic_barc_erf_s10.0_"
    filenameBarcErf2Realistic = filenamedips + "__realistic_barc_erf_s11.0_"
    filenameBarcErf3Realistic = filenamedips + "__realistic_barc_erf_s12.0_"



    names = [filenameLinLossless,filenameErf1Lossless,filenameErf2Lossless,filenameErf3Lossless,filenameVphLossless, 
             filenameBarcLinLossless, filenameBarcErf1Lossless, filenameBarcErf2Lossless, filenameBarcErf3Lossless, 
             filenameBarcErfSh1Lossless, filenameBarcErfSh2Lossless, filenameBarcErfSh3Lossless,
             filenameBarcLinShLossless, filenameWoofLinLossless, 
             filenameWoof1Lossless, filenameWoof2Lossless, filenameWoof3Lossless,
             filenameLinRealistic,filenameErf1Realistic,filenameErf2Realistic,filenameErf3Realistic,filenameVphRealistic, 
             filenameBarcLinRealistic, filenameBarcErf1Realistic, filenameBarcErf2Realistic, filenameBarcErf3Realistic]


    paramsLinLossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, lin_ch, (A, w0)
    paramsErf1Lossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, erf_ch, (b1, s1, w0)
    paramsErf2Lossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, erf_ch, (b2, s2, w0)
    paramsErf3Lossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, erf_ch, (b3, s3, w0)
    paramsVphLossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, step_ch, (50000, w0)
    
    paramsBarcLinLossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, barc_lin_ch, (2*A, w0)
    paramsBarcErf1Lossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, barc_erf_ch, (2*b1, s1, w0)
    paramsBarcErf2Lossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, barc_erf_ch, (2*b2, s2, w0)
    paramsBarcErf3Lossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, barc_erf_ch, (2*b3, s3, w0)
    
    paramsBarcErfSh1Lossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, barc_erf_sh_ch, (2*b1, s1, w0)
    paramsBarcErfSh2Lossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, barc_erf_sh_ch, (2*b2, s2, w0)
    paramsBarcErfSh3Lossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, barc_erf_sh_ch, (2*b3, s3, w0)

    paramsBarcLinShLossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, barc_lin_sh_ch, (2*A, 200000*A/180337, w0)
    paramsWoofLinLossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, woof_lin_ch, (4*A, 200000*A/180337, w0)

    paramsWoof1Lossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, woof_erf_ch, (4*b1, s1, w0)
    paramsWoof2Lossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, woof_erf_ch, (4*b2, s2, w0)
    paramsWoof3Lossless = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_lossless, woof_erf_ch, (4*b3, s3, w0)

    paramsLinRealistic = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_transfer, lin_ch, (A, w0)
    paramsErf1Realistic = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_transfer, erf_ch, (b1, s1, w0)
    paramsErf2Realistic = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_transfer, erf_ch, (b2, s2, w0)
    paramsErf3Realistic = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_transfer, erf_ch, (b3, s3, w0)
    paramsVphRealistic = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_transfer, step_ch, (50000, w0)
    
    paramsBarcLinRealistic = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_transfer, barc_lin_ch, (2*A, w0)
    paramsBarcErf1Realistic = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_transfer, barc_erf_ch, (2*b1, s1, w0)
    paramsBarcErf2Realistic = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_transfer, barc_erf_ch, (2*b2, s2, w0)
    paramsBarcErf3Realistic = tlist, dt, freqList, Elw, eList, tauList, dband, pband, slab_transfer, barc_erf_ch, (2*b3, s3, w0)



    for k, params in enumerate([
        paramsLinLossless, paramsErf1Lossless, paramsErf2Lossless, paramsErf3Lossless, paramsVphLossless,
        paramsBarcLinLossless, paramsBarcErf1Lossless, paramsBarcErf2Lossless, paramsBarcErf3Lossless,
        paramsBarcErfSh1Lossless, paramsBarcErfSh2Lossless, paramsBarcErfSh3Lossless,
        paramsBarcLinShLossless, paramsWoofLinLossless,
        paramsWoof1Lossless, paramsWoof2Lossless, paramsWoof3Lossless,
        paramsLinRealistic, paramsErf1Realistic, paramsErf2Realistic, paramsErf3Realistic, paramsVphRealistic,
        paramsBarcLinRealistic, paramsBarcErf1Realistic, paramsBarcErf2Realistic, paramsBarcErf3Realistic
    ]):
        if k in []:
            interfere(rules, names[k], filenamewidths, filenamechisqs, params, fit=False)
        if k in []:
            interfere(rules, names[k], filenamewidths, filenamechisqs, params, fit=False, setup='pp')
        if k in []:
            interfere(rules, names[k], filenamewidths, filenamechisqs, params, fit=False, setup = 'cc') #setup doen't affect filenames so be careful
        if k in []:
            interfere_dispy(rules, names[k], filenamewidths, filenamechisqs, params, fit=False)
        if k in []:
            interfere_dispy(rules, names[k]+"pp_", filenamewidths, filenamechisqs, params, fit=False, setup='pp')
        if k in [6,7,9,10]:
            interfere_dispy(rules, names[k]+"cc_", filenamewidths, filenamechisqs, params, fit=False, setup='cc')

    return None

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
    

    filenamedips = args.output + '_realistic_lin_'
    filenamewidths = filenamedips+"widths.csv"
    filenamechisqs = filenamedips+"chisqs.csv"
    # params = tlist, dt, freqList, Elw, eList, tauList, xticks, yticks, slab_transfer, lin_ch, (A, w0)
    # print(f"Running interfere() with b={args.b}, sigma_s={args.sigma_s}, lossless")
    # interfere(rules, filenamedips, filenamewidths, filenamechisqs, params, fit=False)
    
    # filenamedips = args.output + '_realistic_erf_'
    # params = tlist, dt, freqList, Elw, eList, tauList, xticks, yticks, slab_transfer, erf_ch, (args.b,args.sigma_s,w0)
    # interfere(rules, filenamedips, filenamewidths, filenamechisqs, params, fit=False)

    filenamedips = args.output + '_realistic_vph_'
    params = tlist, dt, freqList, Elw, eList, tauList, xticks, yticks, slab_transfer, step_ch, (args.b, w0)
    interfere(rules, filenamedips, filenamewidths, filenamechisqs, params, fit=False)



    # rules={'b': 8300,'sigma_s': 10}
    # print("now try the big one")
    # interfere(rules,"testdips","testwidths","testchisqs")
    # print("yay")


if __name__ == "__main__":
    main()
