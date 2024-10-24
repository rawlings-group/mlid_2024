# [depends] tclab_dmle_stable.pickle %LIB%/plotter.py
from cycler import cycler
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle
import plottools

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

## Force this script to start in the main repo directory
import sys
import os
main_dir = os.path.dirname(os.path.abspath(__file__)) + '/..'
os.chdir(main_dir)
sys.path.append(main_dir)

from idtools.plotter import plot_tclab

eigs_cycler = cycler(color=['b', 'g', 'r', 'm', 'c', 'y', 'orange', 'k'],
                     marker=['o','s','+','x','1','2','3','4'],
                     mfc=['none', 'none', 'r', 'm', 'c', 'y', 'orange', 'k'])

## Figsizes
if plottools.isPoster():
    FIGSIZE = (8, 4)
    EFIGSIZE = (7, 4)
    LFIGSIZE = (4, 5)
elif plottools.isPaper:
    FIGSIZE = (10, 3)
    EFIGSIZE = (8, 2.5)
    LFIGSIZE = (5, 5)
else:
    FIGSIZE = (8, 4)
    EFIGSIZE = (7, 4)
    LFIGSIZE = (5, 5)

def augment_system(A, B, C, Bd, Cd):
    m = B.shape[1]
    p, n = C.shape

    Aaug = np.block([[A, Bd], [np.zeros((p, n)), np.eye(p)]])
    Baug = np.vstack([B, np.zeros((p, m))])
    Caug = np.hstack([C, Cd])

    return Aaug, Baug, Caug

# Load and unpack data
with open('data/tclab_dmle.pickle', 'rb') as handle:
    data = pickle.load(handle)
    models = pickle.load(handle)

t = data['t']
u = data['u']
y = data['y']

labels = models.keys()

# Plot results
with PdfPages('figures/tclab_dmle_plot.pdf') as pdf:

    fig, axes = plot_tclab(t, u, y, yf={label: model['yf'] for (label, model)
                                        in models.items()}, figsize=FIGSIZE)
    if plottools.isPaper():
        axes[0, 0].legend(bbox_to_anchor=(0, 1, 2.15, 0), loc="lower left",
                          mode="expand", ncol=5)
    else:
        axes[0, 0].legend(bbox_to_anchor=(0, 1, 2.15, 0), loc="lower left",
                          mode="expand", ncol=5)


    for ax in axes.flatten():
        ax.grid()
    fig.tight_layout(pad=1)

    pdf.savefig()
    plt.close()

    # times
    times = [model['t'] for model in models.values()]
    fig, ax = plt.subplots(figsize=LFIGSIZE)
    bars = ax.barh(list(models.keys()), times, color='b')
    ax.bar_label(bars, fmt='{:,.2f}', padding=5)
    ax.set_xlabel(r"wall time (s)")
    ax.set_xlim(right=1.1*np.amax(times))

    plt.tight_layout(pad=1)
    pdf.savefig()

    # times (log)
    ax.set_xscale('log')
    ax.set_xlim([10.**(-2.), 10.**3.])
    pdf.savefig()
    plt.close()

    # eigs
    A = {label: model['est']['A'] for (label, model) in models.items()}
    AK = dict()
    for (label, model) in models.items():
        est = model['est']
        K = est['K']
        Aaug, _, Caug = augment_system(est['A'], est['B'], est['C'], est['Bd'],
                                       est['Cd'])
        AK[label] = Aaug - K@Caug

    fig, axes = plt.subplots(1, 2, figsize=EFIGSIZE)
    for ax in axes.flatten():
        ax.set_prop_cycle(eigs_cycler)
        circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None', lw=1, zorder=1, label='Unit circle')
        ax.add_patch(circ)
        # ax.grid()


    # Configure axins
    axins = [[zoomed_inset_axes(axes[0], 8, loc=6, borderpad=5)],
             [zoomed_inset_axes(axes[1], 30, loc=9, borderpad=0.5),
              # zoomed_inset_axes(axes[1], 20, loc=3, borderpad=2),
              zoomed_inset_axes(axes[1], 20, loc=8, borderpad=2)
              ]]
    for axin_list in axins:
        for ax in axin_list:
            ax.set_prop_cycle(eigs_cycler)
            ax.grid(zorder=1)
            circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None', lw=1, zorder=2)
            ax.add_patch(circ)

    ## Add the LMI region
    max_eig = 0.998
    alpha = 0.3
    axes[0].fill_between(np.array([0,0]), 0., 0., color='b', alpha=0.1,
                         label=rf'$\mathcal{{D}}_1({alpha:.1f})\cap\mathcal{{D}}_2({max_eig:.3f},0)$')
    x = np.linspace(alpha, max_eig)
    y = np.sqrt(max_eig**2 - np.power(x,2))
    axes[1].fill_between(x, y, -y, color='b', alpha=0.1)

    # Plot
    for (label, Ai) in A.items():
        eigs = np.linalg.eigvals(Ai)
        for ax in [axes[0]] + axins[0]:
            ax.plot(np.real(eigs), np.imag(eigs), label=label, ls='')
    for (label, Ai) in AK.items():
        eigs = np.linalg.eigvals(Ai)
        for ax in [axes[1]] + axins[1]:
            ax.plot(np.real(eigs), np.imag(eigs), label=label, ls='')

    # Axes labels
    axes[0].set_xlabel(r'$\textnormal{Re}(\lambda_i(A))$')
    axes[0].set_ylabel(r'$\textnormal{Im}(\lambda_i(A))$')
    axes[1].set_xlabel(r'$\textnormal{Re}(\lambda_i(A_K))$')
    axes[1].set_ylabel(r'$\textnormal{Im}(\lambda_i(A_K))$')

    # Limits
    plt.tight_layout(pad=1)

    xmin = 0.86
    xmax = 1.02
    bbox = ax.get_position()
    ymax = (xmax-xmin)*(bbox.width/bbox.height)/2
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(-ymax, ymax)
    axes[0].hlines(0, xmin, xmax, color='k', alpha=0.25)
    axes[0].vlines(0, -ymax, ymax, color='k', alpha=0.25)

    xmin = -0.05
    xmax = 1.05
    bbox = ax.get_position()
    ymax = (xmax-xmin)*(bbox.width/bbox.height)/2
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(-ymax, ymax)
    axes[1].hlines(0, xmin, xmax, color='k', alpha=0.25)
    axes[1].vlines(0, -ymax, ymax, color='k', alpha=0.25)

    # axes[0].set_xlim([0.86, 1.02])
    # axes[0].set_ylim([-0.04, 0.04])
    # axes[1].set_xlim([0, 1.1])
    # axes[1].set_ylim([-0.275, 0.275])

    # Inset axes
    axins[0][0].set_xlim([0.988, 0.997])
    axins[0][0].set_ylim([-0.002, 0.002])
    mark_inset(axes[0], axins[0][0], loc1=1, loc2=4, ec="silver")

    axins[1][0].set_xlim([0.9875, 1.0025])
    axins[1][0].set_ylim([-0.002, 0.002])
    x = np.linspace(0.9875, max_eig)
    y = np.minimum(np.sqrt(max_eig**2 - np.power(x,2)), 0.002)
    axins[1][0].fill_between(x, -y, y, color='b', alpha=0.1)
    mark_inset(axes[1], axins[1][0], loc1=1, loc2=4, ec="silver")

    # axins[1][1].set_xlim([0.14, 0.15])
    # axins[1][1].set_ylim([-0.002, 0.002])
    # mark_inset(axes[1], axins[1][1], loc1=1, loc2=2, ec="silver")
    # plt.setp(axins[1][1].get_yticklabels(), visible=False)

    axins[1][1].set_xlim([0.64, 0.66])
    axins[1][1].set_ylim([-0.0025, 0.0025])
    mark_inset(axes[1], axins[1][1], loc1=1, loc2=2, ec="silver")


    # Legend
    if plottools.isPaper():
        axes[0].legend(bbox_to_anchor=(0, 1, 2.2, 0), loc="lower left",
                       mode="expand", ncol=5)
    else:
        axes[0].legend(bbox_to_anchor=(0, 1, 2.15, 0), loc="lower left",
                       mode="expand", ncol=5)

    plt.tight_layout(pad=1)
    pdf.savefig()
    plt.close()
