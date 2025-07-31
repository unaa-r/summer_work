#Asked chatgpt for help iterating through these files, since I messed up the names

import os
import re
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

def flatten_params(params):
    """Flatten nested tuples into a flat NumPy array"""
    return np.concatenate([np.atleast_1d(p).flatten() for p in flatten_structure(params)])

def flatten_structure(obj):
    """Recursively flatten nested tuples/lists into a flat list of scalars or 1D arrays"""
    if isinstance(obj, (tuple, list)):
        return [item for sub in obj for item in flatten_structure(sub)]
    else:
        return [obj]

def unflatten_params(flat, template):
    """Reshape flat array into the structure defined by template"""
    flat = iter(flat)
    def rebuild(template):
        if isinstance(template, (tuple, list)):
            return type(template)(rebuild(sub) for sub in template)
        else:
            return next(flat)
    return rebuild(template)



def const(t, params):
    a, = params
    return a

def gaussian(t, params):
    A, sigmaFWHM, T0 = params
    return A * np.exp(-4 * np.log(2) * (t - T0)**2 / sigmaFWHM**2)

def peaks_with_bckg(t, peak, bckg, peak_params=None, bckg_params=None):
    """n = number of peaks. bckg is the background function. peak_params is 2D tuple of the tuples of params 
    for each peak, bckg_params is a tuple of params for the bckg function"""

    return bckg(t, bckg_params) + sum([peak(t,params) for params in peak_params])


def peak_fitter(data, tauList, init_guess):
    "init_guess should be of the form ((peak1params, ...), backparams) where each individual set of params is a tuple"

    init_guess_flat = flatten_params(init_guess)

    def residuals(params_flat, taus, ys):
        params = unflatten_params(params_flat, init_guess)
        model = peaks_with_bckg(taus, gaussian, const, peak_params = params[0], bckg_params=params[1])
        return ys - model
    
    res = least_squares(residuals, init_guess_flat, args=(tauList, data), method = 'lm')

    return res





# Path to the folder
folder = '/Users/noah.costa/Local Documents/Research 3/CPI_Project/two_interfaces/results/testjul31a_lossless_lin_pp_'
case = 'barc'
tauList = np.arange(-200, 200, 1)

os.makedirs(os.path.join(folder, 'fits'), exist_ok=True)


# Target values (the ones you meant to save)
target_values = np.round(np.arange(10, 10.4, 0.008),3)

#for dispy
target_Ls = np.round(np.array([10]), 3)
target_Ds = np.round(np.array([3000]), 3)




# Regex to find files of the form dip_<float>.txt
pattern = re.compile(r'^dip_([-+]?[0-9]*\.?[0-9]+)\.txt$')

#for the dispy stuff -- comment out as needed
pattern = re.compile(r'^dip_L([-+]?[0-9]*\.?[0-9]+)_D([-+]?[0-9]*\.?[0-9]+)\.txt$')


# Dictionary to store results: {nearest_target_value: np.array([...])}
data = {}

# List all files in the folder
# for fname in os.listdir(folder):
#     # print(fname)
#     match = pattern.match(fname)
#     if match:
#         try:
#             # Extract float from filename
#             raw_value = float(match.group(1))
#             # Find nearest target value
#             nearest = min(target_values, key=lambda x: abs(x - raw_value))
#             # Read the data
#             with open(os.path.join(folder, fname), 'r') as f:
#                 array = np.array([float(line.strip()) for line in f])
#             data[nearest] = array
#         except Exception as e:
#             print(f"Failed to process {fname}: {e}")

#for dispy case
data = {}
for fname in os.listdir(folder):
    # print(fname)
    match = pattern.match(fname)
    if match:
        print(fname)
        try:
            # Extract float from filename
            raw_L = float(match.group(1))
            raw_D = float(match.group(2))
            # Find nearest target value
            nearestL = min(target_Ls, key=lambda x: abs(x - raw_L))
            nearestD = min(target_Ds, key=lambda x: abs(x - raw_D))
            # Read the data
            with open(os.path.join(folder, fname), 'r') as f:
                array = np.array([float(line.strip()) for line in f])
            data[(nearestL,nearestD)] = array
        except Exception as e:
            print(f"Failed to process {fname}: {e}")



for L in data:
    if case == 'lin':
        bckg_tau = 100
        bckg_params = (data[L][np.argmin(np.abs(tauList - bckg_tau))],)
        width = 20
        peak1tau = -55
        peak1params = (data[L][np.argmin(np.abs(tauList - peak1tau))]-bckg_params[0], width, peak1tau)
        peak2tau = 55
        peak2params = (data[L][np.argmin(np.abs(tauList - peak2tau))]-bckg_params[0], width, peak2tau)
        art_tau = 0
        art_params = (data[L][np.argmin(np.abs(tauList - art_tau))]-bckg_params[0], width, art_tau)
        peak_params = (peak1params, peak2params, art_params)

        init_guess = (peak_params, bckg_params)

        res = peak_fitter(data[L], tauList, init_guess)

        opt_params = unflatten_params(res.x, init_guess)

        fit_curve = peaks_with_bckg(tauList, gaussian, const, peak_params=opt_params[0], bckg_params=opt_params[1])

        # plt.figure()
        # plt.plot(tauList, data[L], label="Data", color='blue')
        # plt.plot(tauList, fit_curve, label=f"Fit (FWHM)", color='red')
        # plt.title(f"L = {L}")
        # plt.legend()
        # plt.xlabel("τ (fs)")
        # plt.ylabel("Integrated Intensity")
        # plt.tight_layout()

        # # Format parameters nicely into a multiline string
        # param_text = f"""Peak 1:
        # A = {opt_params[0][0][0]:.3f}
        # Width = {opt_params[0][0][1]:.3f}
        # Tau0 = {opt_params[0][0][2]:.3f}

        # Peak 2:
        # A = {opt_params[0][1][0]:.3f}
        # Width = {opt_params[0][1][1]:.3f}
        # Tau0 = {opt_params[0][1][2]:.3f}

        # Artifact:
        # A = {opt_params[0][2][0]:.3f}
        # Width = {opt_params[0][2][1]:.3f}
        # Tau0 = {opt_params[0][2][2]:.3f}

        # Background:
        # a = {opt_params[1][0]:.3f}
        # """

        # plt.text(0.95, 0.05, param_text, transform=plt.gca().transAxes, 
        #         fontsize=8, verticalalignment='bottom', horizontalalignment='right',
        #         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))


        # plt.savefig(os.path.join(folder, 'fits', f"dip_{L:.3f}.png"))
        # plt.close()




        params_dict = {
            'peak1': {'A': opt_params[0][0][0], 'width': opt_params[0][0][1], 'tau0': opt_params[0][0][2]},
            'peak2': {'A': opt_params[0][1][0], 'width': opt_params[0][1][1], 'tau0': opt_params[0][1][2]},
            'artifact': {'A': opt_params[0][2][0], 'width': opt_params[0][2][1], 'tau0': opt_params[0][2][2]},
            'background': {'a': opt_params[1][0]}
        }

        param_filename = os.path.join(folder, 'fits', f'dip_{L:.3f}_params.json')

        with open(param_filename, 'w') as jsonfile:
            json.dump(params_dict, jsonfile, indent=2)





        fig = plt.figure(figsize=(10, 5))  # wider figure
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # 3:1 plot-to-text ratio

        # Left: main plot
        ax_plot = fig.add_subplot(gs[0])

        ax_plot.plot(tauList, data[L], label="Data", color='blue')
        ax_plot.plot(tauList, fit_curve, label=f"Fit (FWHM)", color='red')

        ax_plot.legend()
        ax_plot.set_xlabel("τ (fs)")
        ax_plot.set_ylabel("Integrated Intensity")

        # Right: text box
        ax_text = fig.add_subplot(gs[1])
        ax_text.axis('off')  # turn off axis

        # Build your param string
        param_text = f"""
        Peak 1:
        A = {opt_params[0][0][0]:.3f}
        Width = {opt_params[0][0][1]:.3f}
        Tau0 = {opt_params[0][0][2]:.3f}

        Peak 2:
        A = {opt_params[0][1][0]:.3f}
        Width = {opt_params[0][1][1]:.3f}
        Tau0 = {opt_params[0][1][2]:.3f}

        Artifact:
        A = {opt_params[0][2][0]:.3f}
        Width = {opt_params[0][2][1]:.3f}
        Tau0 = {opt_params[0][2][2]:.3f}

        Background:
        a = {opt_params[1][0]:.3f}
        """

        ax_text.text(0, 1, param_text, va='top', ha='left', fontsize=9, family='monospace')

        plt.tight_layout()
        plt.savefig(os.path.join(folder, 'fits', f"dip_{L:.3f}.png"))
        plt.close()


    if case == 'barc':

        vals = L
        L = vals[0]
        D = vals[1]

        bckg_tau = 100
        bckg_params = (data[vals][np.argmin(np.abs(tauList - bckg_tau))],)
        width = 20
        peak1tau = -55
        peak1params = (data[vals][np.argmin(np.abs(tauList - peak1tau))]-bckg_params[0], width, peak1tau)
        peak2tau = 55
        peak2params = (data[vals][np.argmin(np.abs(tauList - peak2tau))]-bckg_params[0], width, peak2tau)

        peak_params = (peak1params, peak2params)

        init_guess = (peak_params, bckg_params)

        res = peak_fitter(data[vals], tauList, init_guess)

        opt_params = unflatten_params(res.x, init_guess)

        fit_curve = peaks_with_bckg(tauList, gaussian, const, peak_params=opt_params[0], bckg_params=opt_params[1])

        # plt.figure()
        # plt.plot(tauList, data[L], label="Data", color='blue')
        # plt.plot(tauList, fit_curve, label=f"Fit (FWHM)", color='red')
        # plt.title(f"L = {L}")
        # plt.legend()
        # plt.xlabel("τ (fs)")
        # plt.ylabel("Integrated Intensity")
        # plt.tight_layout()

        # # Format parameters nicely into a multiline string
        # param_text = f"""Peak 1:
        # A = {opt_params[0][0][0]:.3f}
        # Width = {opt_params[0][0][1]:.3f}
        # Tau0 = {opt_params[0][0][2]:.3f}

        # Peak 2:
        # A = {opt_params[0][1][0]:.3f}
        # Width = {opt_params[0][1][1]:.3f}
        # Tau0 = {opt_params[0][1][2]:.3f}

        # Artifact:
        # A = {opt_params[0][2][0]:.3f}
        # Width = {opt_params[0][2][1]:.3f}
        # Tau0 = {opt_params[0][2][2]:.3f}

        # Background:
        # a = {opt_params[1][0]:.3f}
        # """

        # plt.text(0.95, 0.05, param_text, transform=plt.gca().transAxes, 
        #         fontsize=8, verticalalignment='bottom', horizontalalignment='right',
        #         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))


        # plt.savefig(os.path.join(folder, 'fits', f"dip_{L:.3f}.png"))
        # plt.close()




        params_dict = {
            'peak1': {'A': opt_params[0][0][0], 'width': opt_params[0][0][1], 'tau0': opt_params[0][0][2]},
            'peak2': {'A': opt_params[0][1][0], 'width': opt_params[0][1][1], 'tau0': opt_params[0][1][2]},
            'background': {'a': opt_params[1][0]}
        }

        param_filename = os.path.join(folder, 'fits', f'dip_L{L:.3f}_D{D:.3f}_params.json')

        with open(param_filename, 'w') as jsonfile:
            json.dump(params_dict, jsonfile, indent=2)





        fig = plt.figure(figsize=(10, 5))  # wider figure
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # 3:1 plot-to-text ratio

        # Left: main plot
        ax_plot = fig.add_subplot(gs[0])

        ax_plot.plot(tauList, data[vals], label="Data", color='blue')
        ax_plot.plot(tauList, fit_curve, label=f"Fit (FWHM)", color='red')

        ax_plot.legend()
        ax_plot.set_xlabel("τ (fs)")
        ax_plot.set_ylabel("Integrated Intensity")

        # Right: text box
        ax_text = fig.add_subplot(gs[1])
        ax_text.axis('off')  # turn off axis

        # Build your param string
        param_text = f"""
        Peak 1:
        A = {opt_params[0][0][0]:.3f}
        Width = {opt_params[0][0][1]:.3f}
        Tau0 = {opt_params[0][0][2]:.3f}

        Peak 2:
        A = {opt_params[0][1][0]:.3f}
        Width = {opt_params[0][1][1]:.3f}
        Tau0 = {opt_params[0][1][2]:.3f}

        Background:
        a = {opt_params[1][0]:.3f}
        """

        ax_text.text(0, 1, param_text, va='top', ha='left', fontsize=9, family='monospace')

        plt.tight_layout()
        plt.savefig(os.path.join(folder, 'fits', f"dip_L{L:.3f}_D{D:.3f}.png"))
        plt.close()








