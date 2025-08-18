import numpy as np
from scipy.fft import fft, ifft
from scipy.special import erf
from multiprocessing import get_context
import time
import os
import multiprocessing as mp
import argparse
from tqdm import tqdm
import shutil

def Ereffer(Ec, Ea, w, tau):
    return (Ec + Ea) * np.exp(1j * w * tau)

#this phi is the dispersion from going through optical elements
def Esampler(Ec, Ea, phi):
    return (Ec - Ea) * np.exp(1j * phi)


def run_cpi_for_L(chirp_type, L, Ec, Ea, ws, taus, epsilon, integration_range, output_dir):
  
    Esamp_w = Esampler(Ec, Ea, epsilon*L)
    Esamp_t = np.conj(ifft(np.conj(Esamp_w), norm="ortho"))

    SFG_data = []
    for tau in taus:
        Eref_w = Ereffer(Ec, Ea, ws, tau)
        Esfg_t = np.conj(ifft(np.conj(Eref_w), norm="ortho")) * Esamp_t
        Esfg_w = np.conj(fft(np.conj(Esfg_t), norm="ortho"))
        Isfg_w = np.abs(Esfg_w)**2
        SFG_data.append(Isfg_w)

    SFG_data = np.array(SFG_data)

    wavelengths = 2 * np.pi * c * 1e9 / (ws * 1e15)

    SFG_band = SFG_data[:, (wavelengths >= 400 - (integration_range/2)) & (wavelengths <= 400 + (integration_range/2))]
    signal_vs_tau = np.sum(SFG_band, axis=1)
    
   
    out_path = os.path.join(output_dir, f"{chirp_type}_L{L}.txt")
    np.savetxt(out_path, signal_vs_tau, fmt="%.15f")
    #print(f"✅ {chirp_type} L={L} saved.")

def run_cpi_unpack(args):
    return run_cpi_for_L(*args)

def lin_chirp(A, w, w_0):
    return A * ((w - w_0) ** 2)

def erf_chirp(B, w, w_0, sigma):
    x = (w - w_0) * 2 * sigma / np.sqrt(2 * np.log(256))
    return B * ((np.exp(-x**2) - 1) / np.sqrt(np.pi) + x * erf(x))

def superf_chirp(C, w, w_0, sigma_s):
    x = (w - w_0) * 2 * sigma_s / np.sqrt(2 * np.log(256))
    return C * ((np.exp(-x**2) - 1) / np.sqrt(np.pi) + x * erf(x))

#this phi is the chirp you want to apply to the pulse
def chirper(Ews, phi):
    return Ews * np.exp(1j * phi)

# Constants
fwhm = 10 #fs
sigma_s = 1.12 * fwhm
c = 299792458 #m/s
w_0 = 2 * np.pi * c * 1e-15 / 800e-9 #fs^-1
t_0 = 200000 #fs
Npts = 2**19
ts = np.linspace(0, 400000, Npts)
dt = ts[1] - ts[0]


if __name__ == "__main__":
    start = time.time()
    ctx = get_context("spawn")  # safe for Windows/macOS

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, required=True)
    parser.add_argument("--integration_range", type=int, default=1)
    parser.add_argument("--tau_range", type=int, default=100)
    parser.add_argument("--no_overwrite", action="store_true")
    parser.add_argument("--max_L", type=int, default=64001)
    parser.add_argument("--L_stepsize", type=int, default=800)

    args = parser.parse_args()

    # Time-domain field
    Es = np.exp((-2 * np.log(2) * (ts - t_0) ** 2) / fwhm**2) * np.exp(-1j * w_0 * ts)
    Ews = np.conj(fft(np.conj(Es), norm='ortho'))
    ws = 2 * np.pi * np.arange(Npts) / (Npts * dt)

    # Chirps
    A, B, C = 180337, 8300, 7450

    Ec_lin = chirper(Ews, lin_chirp(A, ws, w_0))
    Ea_lin = chirper(Ews, lin_chirp(-A, ws, w_0))
    Ec_erf = chirper(Ews, erf_chirp(B, ws, w_0, fwhm))
    Ea_erf = chirper(Ews, erf_chirp(-B, ws, w_0, fwhm))
    Ec_superf = chirper(Ews, superf_chirp(C, ws, w_0, sigma_s))
    Ea_superf = chirper(Ews, superf_chirp(-C, ws, w_0, sigma_s))

    ###This is what you can modify###

    taus = np.arange(-args.tau_range, args.tau_range + .5, 0.5)
    folder_name = args.folder_name
    integration_range = args.integration_range #units of nm, default 1 nm
    Lvals = np.arange(0, args.max_L, args.L_stepsize) #L values (thickness) in um, default use: np.arange(0, 64001, 800)
    no_overwrite = args.no_overwrite #choose to to not files in folder; good if run times out before you check all cases

    #################################

    if not no_overwrite:
        if os.path.exists(f"results/{folder_name}/linear"):
            shutil.rmtree(f"results/{folder_name}/linear")
        
        if os.path.exists(f"results/{folder_name}/erf"):
            shutil.rmtree(f"results/{folder_name}/erf")
        
        if os.path.exists(f"results/{folder_name}/super_erf"):
            shutil.rmtree(f"results/{folder_name}/super_erf")

    os.makedirs(f"results/{folder_name}/linear", exist_ok=True)
    os.makedirs(f"results/{folder_name}/erf", exist_ok=True)
    os.makedirs(f"results/{folder_name}/super_erf", exist_ok=True)

  
    with open(f"results/{folder_name}/plotting_params.txt", 'w') as f:
        f.write(f"BK7\n")
        f.write(f"{args.tau_range}\n")
        f.write(f"{args.max_L}\n")
        f.write(f"{args.L_stepsize}\n")

    with open(f"results/{folder_name}/run_info.txt", 'w') as f:
        f.write(f"Dispersion type: BK7 - 2nd order only\n")
        f.write(f"Integration range: {integration_range} nm\n")
        f.write(f"Tau range: {args.tau_range} fs\n")
        f.write(f"Maximum L value (exclusive): {args.max_L} um\n")
        f.write(f"L stepsize: {args.L_stepsize} um\n")


    print(f"📊 Sweeping {len(Lvals)} dispersion values")

    # Build task list
    tasks = []

    ws[0] = 1e-6

    epsilon = 0.0223238


    for L in Lvals:

        if no_overwrite:
            if not (os.path.exists(f"./results/{folder_name}/linear/linear_L{L}.txt")):
                tasks.append(("linear", L, Ec_lin, Ea_lin, ws, taus, epsilon, integration_range, f"./results/{folder_name}/linear"))
            if not (os.path.exists(f"./results/{folder_name}/erf/erf_L{L}.txt")):
                tasks.append(("erf", L, Ec_erf, Ea_erf, ws, taus, epsilon, integration_range, f"./results/{folder_name}/erf"))
            if not (os.path.exists(f"./results/{folder_name}/super_erf/super_erf_L{L}.txt")):
                tasks.append(("super_erf", L, Ec_superf, Ea_superf, ws, taus, epsilon, integration_range, f"./results/{folder_name}/super_erf"))
        else:
            tasks.append(("linear", L, Ec_lin, Ea_lin, ws, taus, epsilon, integration_range, f"./results/{folder_name}/linear"))
            tasks.append(("erf", L, Ec_erf, Ea_erf, ws, taus, epsilon, integration_range, f"./results/{folder_name}/erf"))
            tasks.append(("super_erf", L, Ec_superf, Ea_superf, ws, taus, epsilon, integration_range, f"./results/{folder_name}/super_erf"))
        
    # Run in parallel using 4 workers
    print("⚙️ Launching parallel CPI...")
    start = time.time()

    '''
    use this if you dont want progress tracking
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(run_cpi_for_L, tasks)
    '''
        
    with mp.Pool(mp.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(run_cpi_unpack, tasks), total=len(tasks)):
            pass

    print(f"\n✅ All CPI runs complete in {time.time() - start:.2f} s")