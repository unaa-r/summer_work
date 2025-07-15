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
from scipy.stats import binned_statistic

def Ereffer(Ec, Ea, w, tau):
    return (Ec + Ea) * np.exp(1j * w * tau)

#this phi is the dispersion from going through optical elements
def Esampler(Ec, Ea, phi):
    return (Ec - Ea) * np.exp(1j * phi)

# from https://www.coherent.com/resources/tech-notes/lasers/PropagationDispersionMeasurement_of_sub_10fsPulses_08_29_18.pdf
#Choose from "BK7", "Fused Silica", "Sapphire", "CaF2", or "SF10"
def glass_type_epsilon(w, w0, material="BK7"):
    
    c = 0.2998; #(*um/fs*)
    match material:
        case "BK7":
            b1 = 1.03961212 #(*for BK7 glass*)
            b2 = 0.231792344
            b3 = 1.01046945
            c1 = 6.00069867e-3 #(*um^2*)
            c2 = 2.00179144e-2
            c3 = 1.03560653e2
        case "Fused Silica":
            b1 =  0.6961663
            b2 = 0.4079426
            b3 = 0.8974794
            c1 = 0.00467914826
            c2 = 0.0135120631
            c3 = 97.9340025
        case "Sapphire":
            b1 = 1.43134930
            b2 = 0.650547130
            b3 = 5.34140210
            c1 = 0.00527992610
            c2 = 0.0142382647
            c3 = 325.017834
        case "CaF2":
            b1 = 0.5675888
            b2 = 0.4710914
            b3 = 3.8484723
            c1 = 0.00252642999
            c2 = 0.0100783328
            c3 = 1200.555973
        case "SF10":
            b1 = 1.62153902
            b2 = 0.256287842
            b3 = 1.64447552
            c1 = 0.0122241457
            c2 = 0.0595736775
            c3 = 103.560653
        case _:
            print("Error! invalid glass type")
            return 0

    k_w = w*(np.sqrt(1 + (b1*(2*np.pi*c/w)**2)/((2*np.pi*c/w)**2 - c1) + 
                     (b2*(2*np.pi*c/w)**2)/((2*np.pi*c/w)**2 - c2) + 
                     (b3*(2*np.pi*c/w)**2)/((2*np.pi*c/w)**2 - c3)))/c
    
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

    return k_w - k_deriv*(w - w0)


#from https://opg.optica.org/ao/fulltext.cfm?uri=ao-36-16-3785
#https://opg.optica.org/ao/fulltext.cfm?uri=ao-34-18-3477&id=45728
#T must be between 0 and 30
#S is salinity, in parts per thousand (35 is seawater)
def water_epsilon(w, w0, T=20, S=0):
    c = 0.2998; #(*um/fs*)
    
    n0 = 1.31405
    n1 = 1.779e-4
    n2 = -1.05e-6
    n3 = 1.6e-8
    n4 = -2.02e-6
    n5 = 15.868
    n6 = 0.01155
    n7 = -0.00423
    n8 = -4382
    n9 = 1.1455e6

    lambdas = (2 * np.pi * c / w) * 1e3

    lambda_0 = (2*1000*np.pi*c/w0)

    k_w = w*(n0 + (n1 + n2*T + n3*T**2)*S + n4*T**2 +(n5 + n6*S + n7*T)/lambdas + 
             n8/lambdas**2 + n9/lambdas**3)/c
    k_deriv = (n0 + (n1 + n2*T + n3*T**2)*S + n4*T**2 +2*(n5 + n6*S + n7*T)/lambda_0 + 3*n8/lambda_0**2 + 4*n9/lambda_0**3)/c
    

    phi_water= k_w - k_deriv*(w - w0)

    return phi_water


def run_cpi_for_L(L, Ec, Ea, ws, taus, epsilon, integration_range, output_dir):
  
    if np.all(np.abs(Ec) < 1e-12) or np.all(np.abs(Ea) < 1e-12):
        print(f"Warning: Ec or Ea near zero for L = {L}")


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
    
    if np.all(SFG_band == 0):
        print(f"⚠️ SFG_band all zero for L = {L}")
        print(" min wl:", np.min(wavelengths), " max wl:", np.max(wavelengths))
        print(" band range:", 400 - (integration_range / 2), "-", 400 + (integration_range / 2))

    
    out_path = os.path.join(output_dir, f"L{L}.txt")
    with open(out_path, 'w') as f:
        np.savetxt(f, signal_vs_tau, fmt="%.15f")
        f.flush()
        os.fsync(f.fileno())
    print(f"📁 Wrote {out_path} — size: {os.path.getsize(out_path)} bytes")

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

##########################################
# What I'm trying to do:
# 1. Find the indices of ws that correspond to the values closest to +/- 1.5*fwhm while still being divisible
#    by the number of pixels (in this case I'm adding remainder so values will go a bit above 3*fwhm)
#    This is stored in mod_indices
# 2. create freq_mod and phi_mod arrays that are the respective slices of ws and pos_lin within those indices
# 3. Create bin_edges array that holds the value of freq_mod at the edges of each bin (num_pixels amount of bins)
# 4. create edge_indices array that holds the index of each bin edge
# 5. Create phi_binned using scipy.binned_statistic which will return an array the size of num_bins, with each value
# 6. in phi_binned being the average value of the corresponding bin
# 7. Create phi_test, a zeroes array the size of ws
# 8. now set phi_test[edge_indices[i]:edge_indices[i+1]] = phi_binned[i], meaning within the range we defined
#    our mask over, set the value of indices corresponding to each bin to the value of that bin
#

def find_mask_indices(ws, w_0, fwhm, num_pixels):
    #from fwhm t * fwhm f = 2 ln 2/pi (gaussian pulse) and w = 2pi*f
    fwhm_w = 4*np.log(2)/fwhm
    
    # Step 1: Get index of center frequency
    i_center = np.argmin(np.abs(ws - w_0))

    # Step 2: Desired number of modulation points (must be divisible by num_pixels)
    dw = ws[2] - ws[1]
    mod_bw = 3 * fwhm_w  # Total width = ±1.5*fwhm
    N_target = int(np.round(mod_bw / dw))  # number of points in ±3*fwhm

    # Round up to nearest multiple of num_pixels
    remainder = N_target % num_pixels
    if remainder != 0:
        N_target += (num_pixels - remainder)

    # Step 3: Make symmetric indices around w0
    half_N = N_target // 2
    i_start = max(i_center - half_N, 0)
    i_end = i_start + N_target

    # Clip if we exceed array bounds
    if i_end > len(ws):
        i_end = len(ws)
        i_start = i_end - N_target

    return np.arange(i_start, i_end)

def realistic_SLM_mask(mask_indices, ws):
    SLM_mask = np.zeros_like(ws)
    SLM_mask[mask_indices] = 1

    return SLM_mask

def realistic_SLM_phase(ws, phase, mask_indices, num_pixels = 800, bit_depth = 2**8):
  
    ws_mod = ws[mask_indices]
    phi_mod = phase[mask_indices]  # UNWRAPPED phase!
    
    # Step 2: Bin to n pixels (averaging unwrapped phase)
    bin_edges = np.linspace(ws_mod[0], ws_mod[-1], num_pixels + 1)
    edge_indices = np.linspace(mask_indices[0], mask_indices[-1]+ 1, num_pixels + 1, dtype=int)
    phi_binned, _, _ = binned_statistic(ws_mod, phi_mod, statistic='mean', bins=bin_edges)

    phi_test = np.zeros_like(ws)

    for i in range(0, len(edge_indices) - 1):
        phi_test[edge_indices[i]:edge_indices[i+1]] = phi_binned[i]

    # Step 3: Wrap the binned phase
    phi_test = np.mod(phi_test, 2*np.pi)

    #Step 4: Bit crushing
    phi_test = np.round(phi_test/(2*np.pi)*(bit_depth - 1))
    phi_test = phi_test/(bit_depth - 1) * np.pi * 2

    return phi_test


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
    parser.add_argument("--dispersion_type", type=str, default="BK7")
    parser.add_argument("--integration_range", type=int, default=1)
    parser.add_argument("--tau_range", type=int, default=100)
    parser.add_argument("--no_overwrite", action="store_true")
    parser.add_argument("--max_L", type=int, default=64001)
    parser.add_argument("--L_stepsize", type=int, default=800)
    parser.add_argument("--bit_levels", type=int, default=256)
    parser.add_argument("--num_pixels", type=int, default=800)

    args = parser.parse_args()

    # Time-domain field
    Es = np.exp((-2 * np.log(2) * (ts - t_0) ** 2) / fwhm**2) * np.exp(-1j * w_0 * ts)
    Ews = np.conj(fft(np.conj(Es), norm='ortho'))
    ws = 2 * np.pi * np.arange(Npts) / (Npts * dt)

    chirp_params = [1, 10, 100, 1000, 10000, 100000]

    ###This is what you can modify###

    taus = np.arange(-args.tau_range, args.tau_range + .5, 0.5)
    folder_name = args.folder_name
    dispersion_type = args.dispersion_type #choose from seawater, freshwater, or BK7 glass for now (default BK7)
    integration_range = args.integration_range #units of nm, default 1 nm
    Lvals = np.arange(0, args.max_L, args.L_stepsize) #L values (thickness) in um, default use: np.arange(0, 64001, 800)
    no_overwrite = args.no_overwrite #choose to to not files in folder; good if run times out before you check all cases
    bit_levels = args.bit_levels
    num_pixels = args.num_pixels

    #################################

    # Remove existing folders (if not no_overwrite)
    folder = f"results/{folder_name}"

    if not no_overwrite:
            if os.path.exists(folder):
                shutil.rmtree(folder)

    # Create necessary folders
    os.makedirs(f"results/{folder_name}", exist_ok=True)

    with open(f"results/{folder_name}/plotting_params.txt", 'w') as f:
        f.write(f"{dispersion_type}\n")
        f.write(f"{args.tau_range}\n")
        f.write(f"{args.max_L}\n")
        f.write(f"{args.L_stepsize}\n")
        f.write(f"{bit_levels}\n")
        f.write(f"{num_pixels}\n")

    with open(f"results/{folder_name}/run_info.txt", 'w') as f:
        f.write(f"Dispersion type: {dispersion_type}\n")
        f.write(f"Integration range: {integration_range} nm\n")
        f.write(f"Tau range: {args.tau_range} fs\n")
        f.write(f"Maximum L value (exclusive): {args.max_L} um\n")
        f.write(f"L stepsize: {args.L_stepsize} um\n")
        f.write(f"Bit levels: {bit_levels} unique values\n")
        f.write(f"Number of SLM pixels: {num_pixels}\n")

    print(f"📊 Sweeping {len(Lvals)} dispersion values")

    # Build task list
    tasks = []

    ws[0] = 1e-6

    match dispersion_type:
        case "freshwater":
            epsilon = water_epsilon(ws, w_0, S = 0)
        case "seawater":    
            epsilon = water_epsilon(ws, w_0, S = 35)
        case "BK7":
            epsilon = glass_type_epsilon(ws, w_0)
        case "Fused Silica":
            epsilon = glass_type_epsilon(ws, w_0, "Fused Silica")
        case "Sapphire":
            epsilon = glass_type_epsilon(ws, w_0, "Sapphire")
        case "CaF2":
            epsilon = glass_type_epsilon(ws, w_0, "CaF2")
        case "SF10":
            epsilon = glass_type_epsilon(ws, w_0, "SF10")
        case _:
            epsilon = glass_type_epsilon(ws, w_0)        

    epsilon = np.nan_to_num(epsilon, nan=1e-6, posinf=1e-6, neginf=1e-6)
    epsilon = np.clip(epsilon, -1e6, 1e6)  # or tighter limits
    print("🧪 epsilon stats — min:", np.min(epsilon), " max:", np.max(epsilon), " nan:", np.isnan(epsilon).any())


    # Path builder
    def p(chirp_type, c, style):
        return f"./results/{folder_name}/{chirp_type}/chirp_{c}/{style}"
    
    mask_indices = find_mask_indices(ws, w_0, fwhm, num_pixels)
    superf_indices = find_mask_indices(ws, w_0, sigma_s, num_pixels)

    SLM_mask = realistic_SLM_mask(mask_indices, ws)
    superf_SLM_mask = realistic_SLM_mask(superf_indices, ws)

    for chirp in chirp_params:

        # Remove existing folders (if not no_overwrite)
        if not no_overwrite:
            for chirp_type in ["linear", "erf", "super_erf"]:
                for style in ["ideal", "realistic"]:
                    folder = f"results/{folder_name}/{chirp_type}/chirp_{chirp}/{style}"
                    if os.path.exists(folder):
                        shutil.rmtree(folder)

        # Create necessary folders
        for chirp_type in ["linear", "erf", "super_erf"]:
            os.makedirs(f"results/{folder_name}/{chirp_type}/chirp_{chirp}/ideal", exist_ok=True)
            os.makedirs(f"results/{folder_name}/{chirp_type}/chirp_{chirp}/realistic", exist_ok=True)

        # Phase generation
        lin_phase = lin_chirp(chirp, ws, w_0)
        erf_phase = erf_chirp(chirp, ws, w_0, fwhm)
        superf_phase = superf_chirp(chirp, ws, w_0, sigma_s)

        Ec_lin_ideal = chirper(Ews, lin_phase)
        Ea_lin_ideal = chirper(Ews, -lin_phase)
        Ec_erf_ideal = chirper(Ews, erf_phase)
        Ea_erf_ideal = chirper(Ews, -erf_phase)
        Ec_superf_ideal = chirper(Ews, superf_phase)
        Ea_superf_ideal = chirper(Ews, -superf_phase)

        lin_phase_realistic = realistic_SLM_phase(ws, lin_phase, mask_indices, num_pixels, bit_levels)
        erf_phase_realistic = realistic_SLM_phase(ws, erf_phase, mask_indices, num_pixels, bit_levels)
        superf_phase_realistic = realistic_SLM_phase(ws, superf_phase, superf_indices, num_pixels, bit_levels)

        Ec_lin_realistic = chirper(Ews, lin_phase_realistic) * SLM_mask
        Ea_lin_realistic = chirper(Ews, -lin_phase_realistic) * SLM_mask
        Ec_erf_realistic = chirper(Ews, erf_phase_realistic) * SLM_mask
        Ea_erf_realistic = chirper(Ews, -erf_phase_realistic) * SLM_mask
        Ec_superf_realistic = chirper(Ews, superf_phase_realistic) * superf_SLM_mask
        Ea_superf_realistic = chirper(Ews, -superf_phase_realistic) * superf_SLM_mask

        print("🧪 max |Ec_lin_ideal|:", np.max(np.abs(Ec_lin_ideal)))
        print("🧪 max |Ea_lin_ideal|:", np.max(np.abs(Ea_lin_ideal)))
        print("🧪 max |Ec_lin_realistic|:", np.max(np.abs(Ec_lin_realistic)))
        print("🧪 max |Ea_lin_realistic|:", np.max(np.abs(Ea_lin_realistic)))


        for L in Lvals:
            
            if no_overwrite:
                if not os.path.exists(p("linear", c,"ideal") + f"/L{L}.txt"):
                    tasks.append((L, Ec_lin_ideal, Ea_lin_ideal, ws, taus, epsilon, integration_range, p("linear", c, "ideal")))
                if not os.path.exists(p("erf", c, "ideal") + f"/L{L}.txt"):
                    tasks.append((L, Ec_erf_ideal, Ea_erf_ideal, ws, taus, epsilon, integration_range, p("erf", c, "ideal")))
                if not os.path.exists(p("super_erf", c, "ideal") + f"/L{L}.txt"):
                    tasks.append((L, Ec_superf_ideal, Ea_superf_ideal, ws, taus, epsilon, integration_range, p("super_erf", c,"ideal")))

                if not os.path.exists(p("linear", c, "realistic") + f"/L{L}.txt"):
                    tasks.append((L, Ec_lin_realistic, Ea_lin_realistic, ws, taus, epsilon, integration_range, p("linear", c, "realistic")))
                if not os.path.exists(p("erf", c, "realistic") + f"/L{L}.txt"):
                    tasks.append((L, Ec_erf_realistic, Ea_erf_realistic, ws, taus, epsilon, integration_range, p("erf", c, "realistic")))
                if not os.path.exists(p("super_erf", c, "realistic") + f"/L{L}.txt"):
                    tasks.append((L, Ec_superf_realistic, Ea_superf_realistic, ws, taus, epsilon, integration_range, p("super_erf", c, "realistic")))
            else:
                tasks.append((L, Ec_lin_ideal, Ea_lin_ideal, ws, taus, epsilon, integration_range, p("linear", c, "ideal")))
                tasks.append((L, Ec_erf_ideal, Ea_erf_ideal, ws, taus, epsilon, integration_range, p("erf", c, "ideal")))
                tasks.append((L, Ec_superf_ideal, Ea_superf_ideal, ws, taus, epsilon, integration_range, p("super_erf", c, "ideal")))

                tasks.append((L, Ec_lin_realistic, Ea_lin_realistic, ws, taus, epsilon, integration_range, p("linear", c, "realistic")))
                tasks.append((L, Ec_erf_realistic, Ea_erf_realistic, ws, taus, epsilon, integration_range, p("erf", c, "realistic")))
                tasks.append((L, Ec_superf_realistic, Ea_superf_realistic, ws, taus, epsilon, integration_range, p("super_erf", c, "realistic")))

            
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