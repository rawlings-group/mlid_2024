from cycler import cycler
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle

## Force this script to start in the main repo directory
import sys
import os
main_dir = os.path.dirname(os.path.abspath(__file__)) + '/..'
os.chdir(main_dir)
sys.path.append(main_dir)

from idtools.plotter import plot_tclab

# Load and unpack data
with open('data/tclab_mle.pickle', 'rb') as handle:
    data = pickle.load(handle)
    models = pickle.load(handle)

t = data['t']
u = data['u']
y = data['y']

labels = models.keys()

# Plot results
with PdfPages('figures/tclab_mle_plot.pdf') as pdf:

    fig, axes = plot_tclab(t, u, y, yf={label: model['yf'] for (label, model)
                                        in models.items()})

    for ax in axes.flatten():
        ax.grid()
    fig.tight_layout(pad=1)

    pdf.savefig()
    plt.close()

    # times
    times = [model['t'] for model in models.values()]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(list(models.keys()), times, color='b')
    ax.bar_label(bars, fmt='{:,.2f}', padding=5)
    ax.set_xlabel(r"wall time (s)")
    ax.set_xlim(right=1.1*np.amax(times))

    plt.tight_layout(pad=1)

    ax.set_xscale('log')
    ax.set_xlim([10.**(-1.), 10.**2.])
    pdf.savefig()
    plt.close()
