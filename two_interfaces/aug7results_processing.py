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

def n_BK7(w):

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




def do_fits(folders, case, target_Ls, target_Ds, tauList=np.arange(-200,200,1)):
    
    for folder in folders:

        #first we'll check if the folder exists, before making the fitting folder
        if os.path.isdir(folder):
            os.makedirs(os.path.join(folder, 'fits'), exist_ok=True)
        else:
            print(f"folder {folder} does not exist, skipping to next folder")
            continue

        pattern = re.compile(r'^dip_L([-+]?[0-9]*\.?[0-9]+)_D([-+]?[0-9]*\.?[0-9]+)\.txt$')

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

                    #only except good matches
                    if abs(nearestL - raw_L) > 0.001 or abs(nearestD - raw_D) > 0.001:
                        print(f"Skipping {fname}: L or D value too far from target")
                        continue

                    # Read the data
                    with open(os.path.join(folder, fname), 'r') as f:
                        array = np.array([float(line.strip()) for line in f])

                    data[(nearestL,nearestD)] = array
                    
                except Exception as e:
                    print(f"Failed to process {fname}: {e}")

        for vals in data:

            if case == 'barc':
                
                c = 0.2998 #um/fs
                w0 = 2 * np.pi * c / 800 *  1e3 # rad/fs
                print(f'w0: {w0:.3f}')


                L = vals[0]
                D = vals[1]

                bckg_tau = 100
                bckg_params = (data[vals][np.argmin(np.abs(tauList - bckg_tau))],)

                bckg_params = (0,)

                width = 20
                peak1tau = -1.51 * L / c
                peak1params = (data[vals][np.argmin(np.abs(tauList - peak1tau))]-bckg_params[0], width, peak1tau)
                peak2tau = 1.51 * L / c
                peak2params = (data[vals][np.argmin(np.abs(tauList - peak2tau))]-bckg_params[0], width, peak2tau)

                peak_params = (peak1params, peak2params)

                init_guess = (peak_params, bckg_params)

                res = peak_fitter(data[vals], tauList, init_guess)

                opt_params = unflatten_params(res.x, init_guess)

                fit_curve = peaks_with_bckg(tauList, gaussian, const, peak_params=opt_params[0], bckg_params=opt_params[1])

                #it would be convenient to calculate the chi^2 (not really a chi^2)
                chisq = sum((fit_curve - data[vals])**2)
                chisq_norm = chisq/len(data[vals])

                #it would also be convenient to calculate the "measured" width:
                print('index', n_BK7(w0), n_BK7(2*w0))
                L_measured = abs(opt_params[0][0][2] - opt_params[0][1][2]) * c / (n_BK7(w0) * 2)
                L_measured = abs(opt_params[0][0][2] - opt_params[0][1][2]) * c / (1.515 * 2)


                #now to save in a json
                params_dict = {
                    'peak1': {'A': opt_params[0][0][0], 'width': opt_params[0][0][1], 'tau0': opt_params[0][0][2]},
                    'peak2': {'A': opt_params[0][1][0], 'width': opt_params[0][1][1], 'tau0': opt_params[0][1][2]},
                    'background': {'a': opt_params[1][0]},
                    'chi^2': {'tot': chisq, 'normalized': chisq_norm},
                    'L (measured)': L_measured
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
                plt.title(f"L = {L}, D = {D}")


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

                Error:
                chi^2 = {chisq:.3f}
                chi^2 per point = {chisq_norm:.3f}

                Results:
                L (measured) = {L_measured:.3f}
                """

                ax_text.text(0, 1, param_text, va='top', ha='left', fontsize=9, family='monospace')

                plt.tight_layout()
                plt.savefig(os.path.join(folder, 'fits', f"dip_L{L:.3f}_D{D:.3f}.png"))
                plt.close()

    return None


def main():

    path = '/Users/noah.costa/Local Documents/Research 3/CPI_Project/two_interfaces/results/'
    folders = [path+f"testaug12{c}_lossless_barc_lin_cc_" for c in ('a','b','c','d','e','f')]
    Ls = np.array([5,10,20])
    Ds = np.arange(0,3001,500)
    
    do_fits(folders,'barc',Ls,Ds)

    return None



if __name__ == "__main__":
    main()











