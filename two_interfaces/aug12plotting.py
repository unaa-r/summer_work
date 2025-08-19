import json
import os
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/noah.costa/Local Documents/Research 3/CPI_Project/two_interfaces/results/'
folders = [path+f"testaug12{c}_lossless_barc_lin_cc_" for c in ('a','f')]

Ds = np.arange(0,3001,500)
L = 10

labels = ['A = 2500', 'A = 180337']

for k, folder in enumerate(folders):

    widths = []

    for D in Ds:

        with open(os.path.join(path,folder,'fits',f"dip_L{L:.3f}_D{D:.3f}_params.json"),'r') as jsonfile:
            params = json.load(jsonfile)
            widths.append(params["peak1"]["width"])
    
    plt.plot(Ds, widths,label=labels[k])

plt.title('Signal Widths')
plt.legend()
# plt.savefig(os.path.join(path,folder,'fits',"peak1widths.png"))
plt.savefig(os.path.join(path,"peak1widths.png"))

        