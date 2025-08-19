import numpy as np
import matplotlib.pyplot as plt
import json
import os
import interfaces_aug6 as inter


def get_dict(filename):
    with open(filename, 'r') as jsonfile:
        d = json.load(jsonfile)
    return d


def get_data(folder, L, Ds):
    data = []
    for D in Ds:
        file = os.path.join(folder,f"dip_L{L:.3f}_D{D:.3f}.json")
        d = get_dict(file)
        data.append([d["D"],d["centre wavelength"]])
    return np.transpose(np.array(data))


def theory_data(Ds, params):
    """Gets the theoretical centre positions via mean and median, params = A, w0, sigmat"""

    D = np.linspace(min(Ds),max(Ds),10000)

    means = inter.barc_lin_centre_mean(D, params)
    medians = inter.barc_lin_centre_median(D, params)

    #refined
    means = inter.barc_lin_centre_mean_refined(D, params)

    return np.array([D,means]), np.array([D,medians])


def plot_data(data, outfile, means=None, medians=None, title=None):
    plt.plot(data[0],data[1], marker='o', linestyle='-', label='numerical')
    if means is not None:
        plt.plot(means[0],means[1],linestyle='--', label = 'mean')
    if medians is not None:
        plt.plot(medians[0],medians[1], linestyle='--', label = 'median')
    plt.xlabel("D (um)")
    plt.ylabel("centre wavelength (nm)")
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile+".png")

def plot_data2(data, outfile, means=None, medians=None, title=None, caption=None):


    import textwrap

    fig, ax = plt.subplots()

    # Main data
    ax.plot(data[0], data[1], marker='o', linestyle='-', label='numerical')

    # Optional plots
    if means is not None:
        ax.plot(means[0], means[1], linestyle='--', label='mean')
    if medians is not None:
        ax.plot(medians[0], medians[1], linestyle='--', label='median')

    # Labels and legend
    ax.set_xlabel("D (um)")
    ax.set_ylabel("centre wavelength (nm)")
    ax.legend()

    # Title
    if title is not None:
        ax.set_title(title)

    # Caption text with automatic wrapping
    if caption is not None:
        caption_text = caption
        wrapped_caption = "\n".join(textwrap.wrap(caption_text, width=80))  # adjust width for wrapping

        fig.text(0.5, 0.01, wrapped_caption, ha='center', fontsize=9)

        # Adjust layout to leave room for caption
        fig.tight_layout(rect=[0, 0.05, 1, 1])  # more bottom margin

    # Save
    fig.savefig(outfile + ".png", dpi=300)
    plt.close(fig)








def main():
    path = '/Users/noah.costa/Local Documents/Research 3/CPI_Project/two_interfaces/results_aug12/testaug12a_lossless_barc_lin_cc_'

    caption = 'Wavelength shift from dispersion, where A = 2500, BK7 glass. Using refined mean from testaug12a'
    outfile = 'aug12ashiftrefined'

    A = 2500
    w0 = inter.w0
    sigmat = inter.sigma


    Ds = np.arange(0, 3001, 500)
    data = get_data(path, 10, Ds)
    means, medians = theory_data(Ds, (A, w0, sigmat))
    plot_data2(data, outfile, means=means, medians=medians, caption=caption)

if __name__ == "__main__":
    main()



