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

## Plotting defauls
default_cycler = (
    cycler(color=['r', 'g', 'b', 'y', 'c', 'm', 'y', 'k']) +
    cycler(linestyle=['-', '--', ':', '-.', '-', '--', ':', '-.'])
)
plt.rc('axes', prop_cycle=default_cycler)
scatter_args = {
    'label': "data",
    'facecolors': "none",
    'edgecolors': "k",
    'marker': '.',
    's': 0.5
}

# Load and unpack data
with open('data/siso_mle.pickle', 'rb') as handle:
    data = pickle.load(handle)
    models = pickle.load(handle)

t = data['t']
y = data['y']

labels = models.keys()

# Plot results
with PdfPages('figures/siso_mle_plot.pdf') as pdf:

    # k step ahead plot
    plt.figure(figsize=(7, 3.5))

    for (label, model) in models.items():
        plt.plot(t, model['yf'], label=label)
    plt.scatter(t, y[0, :], **scatter_args)

    plt.legend()
    plt.ylabel(r"$y$", rotation=0, labelpad=10)
    plt.xlabel(r"time")
    plt.tight_layout(pad=1)

    pdf.savefig()
    plt.close()

    # times
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(labels, [model['t'] for model in models.values()], color='b')
    ax.set_ylabel(r"$t$ (s)", rotation=0, labelpad=10)

    plt.tight_layout(pad=1)
    ax.set_yscale('log')
    pdf.savefig()
    plt.close()


    # errors
    barlabels = [r'$\|A-\hat A\|/\|A\|$', r'$\|B-\hat B\|/\|B\|$',
                 r'$\|K-\hat K\|/\|K\|$', r'$\|R_e-\hat R_e\|/\|R_e\|$']
    err = np.array([model['err'] for (label, model) in models.items()])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    barWidth = 0.2
    place = np.arange(len(labels),dtype='float64')
    for (i, bar_label) in enumerate(barlabels):
        ax.bar(place, err[:, i], width=barWidth, label=bar_label)
        place += barWidth
    ax.set_xticks([r + barWidth for r in range(len(labels))], labels)
    ax.legend()

    plt.tight_layout(pad=1)
    ax.set_yscale('log')
    pdf.savefig()
    plt.close()


    # errors (no arx)
    labels = [label for (label, model) in models.items() if label not in ['ARX', 'EM']]
    err = np.array([model['err'] for (label, model) in models.items()
                    if label not in ['ARX', 'EM']])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    barWidth = 0.2
    place = np.arange(len(labels),dtype='float64')
    for (i, bar_label) in enumerate(barlabels):
        ax.bar(place, err[:, i], width=barWidth, label=bar_label)
        place += barWidth
    ax.set_xticks([r + barWidth for r in range(len(labels))], labels)
    ax.legend()
    ax.set_yscale('log')
    pdf.savefig()
    plt.close()
