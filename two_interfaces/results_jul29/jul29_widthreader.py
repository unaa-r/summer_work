import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

#path to folder

def white_light(D):

    eps = 0.0223238 * 2 * D #In Maggie we trust. Also notice the factor of two
    s = 20 #From Maggie's 437 report

    return 2 * s * np.sqrt(1 + (4*np.log(2) * eps / s**2)**2)
    


def width_extraction():
    folder = '/Users/noah.costa/Local Documents/Research 3/CPI_Project/two_interfaces/results/two_interfaces_jul29_lossless_woof_s12.0_cc_'

    out_folder = folder

    out_name = 'widths.csv'


    xvals = np.arange(0,5001,500)
    filenames = [f"dip_L10.000_D{x:.3f}_params.json" for x in xvals]


    widths1 = []
    widths2 = []

    for filename in filenames:
        path = os.path.join(folder,'fits',filename)
        with open(path,'r') as f:
            params = json.load(f)
            widths1.append(params["peak1"]["width"])
            widths2.append(params["peak2"]["width"])


    df = pd.DataFrame({'D': xvals, 'peak1width': widths1, 'peak2width': widths2})
    df.to_csv(os.path.join(out_folder,out_name), index=False)

    return None


def plot_all_widths():

    plot_white = True

    xvals = np.arange(0,5001,500)

    df1 = pd.DataFrame({'D': xvals})
    df2 = pd.DataFrame({'D': xvals})

    path = '/Users/noah.costa/Local Documents/Research 3/CPI_Project/two_interfaces/results'

    names = ['barc_erf_sh_s10.0_cc_', 'barc_erf_sh_s11.0_cc_', 'barc_erf_sh_s12.0_cc_', 'barc_lin_sh_cc_', 
             'erf_s10.0_pp_', 'erf_s11.0_pp_', 'erf_s12.0_pp_', 'lin_pp_', 'woof_lin_cc_', 
             'woof_s10.0_cc_', 'woof_s11.0_cc_', 'woof_s12.0_cc_']

    fullnames = ['two_interfaces_jul29_lossless_' + name for name in names]

    pathnames = [os.path.join(path,fullname,'widths.csv') for fullname in fullnames]

    for i, name in enumerate(pathnames):
        df1[names[i]] = pd.read_csv(name)['peak1width']
        df2[names[i]] = pd.read_csv(name)['peak2width']
    
    for column in df1.columns:
        if column != 'D':
            plt.plot(df1['D'], df1[column], label=column)
    
    if plot_white:
        white = white_light(xvals)
        plt.plot(xvals, white, label='white light', linestyle='--')



    plt.xlabel('D')
    plt.ylabel('Dip widths (fs)')
    plt.title('Dip widths for various pulse shapes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path,'many_widths_white.png'))

    











if __name__ == "__main__":
    plot_all_widths()




