import json
import os
import numpy as np
import matplotlib.pyplot as plt

def white_light(D):

    eps = 2 * D * 0.0223238
    sigmat = 10

    return 2 * sigmat * np.sqrt(1 + (4*np.log(2)*eps/sigmat)**2)






path = '/Users/noah.costa/Local Documents/Research 3/CPI_Project/two_interfaces/results/'
folders = [path+f"interfaces_aug19{c}" for c in 
               [f"a_lossless_barc_erf_s{s}_cc_" for s in ['10.0','11.0','12.0']]+
               [f"b_lossless_erf_s{s}_pp_" for s in ['10.0','11.0','12.0']]+
               ["a_lossless_barc_lin_cc_","b_lossless_lin_pp_"]]

Ds = np.arange(0,10001,100)
L = 10

whites = white_light(Ds)

labels = ['barc erf s=10','barc super erf s=11','barc super erf s=12',
          'plus-plus erf s=10','plus-plus super erf s=11','plus-plus super erf s=12',
          'barc linear', 'plus-plus linear']

for k, folder in enumerate(folders):

    widths = []

    for D in Ds:

        with open(os.path.join(path,folder,'fits',f"dip_L{L:.3f}_D{D:.3f}_params.json"),'r') as jsonfile:
            params = json.load(jsonfile)
            widths.append(params["peak1"]["width"])
    
    plt.plot(Ds, widths,label=labels[k])

plt.plot(Ds,whites,label='white light')

plt.title('Signal Widths')

plt.xlabel('Thickness of Dispersive Material (um)')

plt.legend()


plt.xlim((0,8000))
plt.ylim((0,100))

plt.savefig(os.path.join(path,"peak1widthszoom.png"))

plt.close()

for k, folder in enumerate(folders):

    otherwidths = []

    for D in Ds:

        with open(os.path.join(path,folder,'fits',f"dip_L{L:.3f}_D{D:.3f}_params.json"),'r') as jsonfile:
            params = json.load(jsonfile)
            otherwidths.append(params["peak2"]["width"])
    
    plt.plot(Ds, otherwidths,label=labels[k])

plt.plot(Ds,whites,label='white light')

plt.title('Signal Widths')
plt.xlabel('Thickness of Dispersive Material (um)')
plt.legend()


plt.xlim((0,8000))
plt.ylim((0,100))

plt.savefig(os.path.join(path,"peak2widthszoom.png"))

plt.close()


for k, folder in enumerate(folders):

    Lexps = []

    for D in Ds:

        with open(os.path.join(path,folder,'fits',f"dip_L{L:.3f}_D{D:.3f}_params.json"),'r') as jsonfile:
            params = json.load(jsonfile)
            Lexps.append(params["L (measured)"])
    
    plt.plot(Ds, Lexps,label=labels[k])

plt.title('Measured L (um)')
plt.xlabel('Thickness of Dispersive Material (um)')
plt.legend()


plt.xlim((0,8000))
plt.ylim((9.9,10.1))

plt.savefig(os.path.join(path,"Lexptszoom.png"))

plt.close()


plt.close()


for k, folder in enumerate(folders):

    chis = []

    for D in Ds:

        with open(os.path.join(path,folder,'fits',f"dip_L{L:.3f}_D{D:.3f}_params.json"),'r') as jsonfile:
            params = json.load(jsonfile)
            chis.append(params["chi^2"]["normalized"])
    
    plt.plot(Ds, chis,label=labels[k])

plt.title('Chi squared per point')
plt.xlabel('Thickness of Dispersive Material (um)')
plt.legend()


plt.xlim((0,8000))
# plt.ylim((9.9,10.1))

plt.savefig(os.path.join(path,"chi2zoom.png"))

plt.ylim(0,10000)

plt.savefig(os.path.join(path,"chi2morezoom.png"))


plt.close()

