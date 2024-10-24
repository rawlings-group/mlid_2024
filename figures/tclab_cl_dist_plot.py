# [depends] %LIB%/tclab_cl_dist_20240131_013507_newlabels.pickle %LIB%/plotter.py
from cycler import cycler
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle

import scipy as sp

## Force this script to start in the main repo directory
import sys
import os
main_dir = os.path.dirname(os.path.abspath(__file__)) + '/..'
os.chdir(main_dir)
sys.path.append(main_dir)

from idtools.plotter import plot_tclab

## Have to copy the default linestyle cycle because mpl removed access to it in 3.8
linestyles = plt.rcParams['axes.prop_cycle'].by_key()["linestyle"]

## Figures
FIGSIZE = (8, 4)
LFIGSIZE = (4, 5)
DFIGSIZE = (4, 5)
T = [1000] # T for computing the KPI
TDIST = [10, 100] # T for comparing e.T@invRe@e distributions

# Load and unpack data
with open('data/tclab_cl_dist.pickle', 'rb') as handle:
    experiments = pickle.load(handle)

labels = list(experiments.keys())

# Plot results
with PdfPages('figures/tclab_cl_dist_plot.pdf') as pdf:
    exp = list(experiments.values())[0]
    t = exp['t']
    ysp = exp['ysp']
    y0 = exp['y0']
    dout = exp['dout']
    din = exp['din']

    ## Plot side by side?
    fig, axes = plot_tclab(
        t,
        u={label: exp['u'] for (label, exp) in experiments.items()},
        y={label: exp['y'] for (label, exp) in experiments.items()},
        ysp=ysp,
        dout=dout,
        din=din,
        figsize=FIGSIZE
    )
    legend = axes[0, 0].legend(bbox_to_anchor=(0, 1, 2.3, 0), loc="lower left",
                               mode="expand", ncol=5, fancybox=True)

    ymin = np.amin(dout,1) + y0[:,0]
    ymax = np.amax(dout,1) + y0[:,0]
    Dy = ymax - ymin
    axes[0,0].set_ylim(np.array([ymin[0],ymax[0]]) + 0.1*Dy[0]*np.array([-1,1]))
    axes[1,0].set_ylim(np.array([ymin[1],ymax[1]]) + 0.1*Dy[1]*np.array([-1,1]))
    axes[0,1].set_ylim([-5,105])
    axes[1,1].set_ylim([-5,105])

    for ax in axes.flatten():
        ax.grid()
    fig.tight_layout(pad=1)

    pdf.savefig(transparent=True)
    plt.close()

    ## Plot each and collect some extra CL performance data
    cldata = {}
    for label, exp in experiments.items():
        t = exp['t']
        u = exp['u']
        y = exp['y']
        u0 = exp['u0']
        y0 = exp['y0']
        ysp = exp['ysp']
        dout = exp['dout']
        din = exp['din']
        Q = exp['Qs']
        R = exp['Rs']

        m = u.shape[0]
        p = y.shape[0]

        ## Get CI
        A, B, C, Bd, Cd, K_lqe, Re = exp['model']['est'].values()
        n, nd = Bd.shape
        naug = n + nd

        Aaug = np.block([[A, Bd], [np.zeros((nd, n)), np.eye(nd)]])
        Baug = np.vstack([B, np.zeros((nd, m))])
        Caug = np.hstack([C, Cd])
        invRe = np.linalg.inv(Re)

        N = y.shape[1]
        xaug = np.zeros((n+nd, N+1))
        yhat = np.zeros((p, N))
        e = np.zeros((p, N))
        for i in range(N):
            yhat[:, i] = y0[:, 0] + Caug@xaug[:, i]
            e[:, i] = y[:, i] - yhat[:, i]
            xaug[:, i+1] = Aaug@xaug[:, i] + Baug@(u[:, i] - u0[:,0]) + K_lqe@e[:, i]

        ## Extra CL performance data
        ell = np.zeros((N, ))
        kpi = np.zeros((N, len(T)))
        evar = np.zeros((N, ))
        kpi_kf = np.zeros((N, len(T)))
        q = np.zeros((N, ))
        kpi_q = np.zeros((N, len(TDIST)))

        for k in range(N):
            ## u penalty reveals ML advantage but isn't quite honest.
            ell[k] = (y[:, k] - ysp[:, k]).T @ Q @ (y[:, k] - ysp[:, k]) #\
            # + (u[:, k] - u0[:, 0]).T @ R @ (u[:, k] - u0[:, 0])

            ## Error covariances and the Log likelihood variability changes a lot during output
            ## disturbances is hard to understand. Maybe skip this.
            evar[k] = (y[:, k] - yhat[:, k]).T @ (y[:, k] - yhat[:, k])
            q[k] = (y[:, k] - yhat[:, k]).T @ invRe @ (y[:, k] - yhat[:, k])
            for (i,Ti) in enumerate(T):
                if k>=Ti:
                    kpi[k, i] = np.mean(ell[k-Ti+1:k+1])
                    kpi_kf[k, i] = np.mean(evar[k-Ti+1:k+1])
                else:
                    kpi[k, i] = np.mean(ell[:k+1])
                    kpi_kf[k, i] = np.mean(evar[:k+1])
            for (i,Ti) in enumerate(TDIST):
                if k>=Ti:
                    kpi_q[k, i] = np.mean(q[k-Ti+1:k+1])
                else:
                    kpi_q[k, i] = np.mean(q[:k+1])

        cldata[label] = {'t': t, 'yhat': yhat, 'ell': ell, 'kpi': kpi, 'evar':
                         evar, 'kpi_kf': kpi_kf, 'q': q, 'kpi_q': kpi_q,
                         'avgell': np.cumsum(ell) / (np.arange(len(ell))+1),
                         'avgevar': np.cumsum(evar) / (np.arange(len(evar))+1),
                         'avgq': np.cumsum(q) / (np.arange(len(q))+1)}

    # Summary (running averages)
    fig, axes = plt.subplots(1, 2, figsize=(5, 2))

    for (label, data) in cldata.items():
        t = data['t']
        avgell = data['avgell']
        avgevar = data['avgevar']

        axes[0].plot(t, avgell, label=label)
        axes[1].plot(t, avgevar, label=label)

    axes[0].set_ylabel(r'$\langle\ell\rangle_k$', rotation=0,
                       va="center", ha="right")
    axes[0].set_ylim([0, 1.1*np.amax(cldata['Augmented PCA']['avgell'])])
    axes[0].set_xlabel(r'time (s)')
    axes[0].set_xlim([0, t[-1]])

    axes[1].set_ylabel(r'$\langle e^\top e\rangle_k$', rotation=0,
                       va="center", ha="right")
    axes[1].set_ylim([0, 1.1*np.amax(cldata['Augmented ARX']['avgevar'][100:])])
    axes[1].set_xlabel(r'time (s)')
    axes[1].set_xlim([0, t[-1]])

    axes[0].legend(bbox_to_anchor=(0, 1, 2.45, 0), loc="lower left",
                   mode="expand", ncol=4, handlelength=2, fontsize=6)

    fig.tight_layout(pad=1)
    pdf.savefig(transparent=True)
    plt.close()

    # overlapping histogram q plot (a few)
    q_compare = [r'Augmented PCA', r'Augmented ARX', 'Unregularized ML', r'Reg. \& Cons. ML']
    fig, axes = plt.subplots(len(TDIST)+1, 1, figsize=DFIGSIZE, sharex=True)

    bins = np.linspace(0, 8, 51)
    patches = []
    for (i, (label, data)) in enumerate(cldata.items()):
        q = data['q']
        w = np.zeros(len(q)) + 1. / (bins[1] * len(q))
        if label in q_compare:
            *_, p1 = axes[0].hist(q, density=True, bins=bins, weights=w,
                                  histtype='step', ls=linestyles[i], lw=1)
            *_, p2 = axes[0].hist(q, density=True, bins=bins, weights=w,
                                  histtype='stepfilled',
                                  color=p1[0].get_facecolor(), alpha=0.5)
            patches += [(p1[0],p2[0])]
        else:
            axes[0].hist(np.nan*q, density=True, bins=bins)

    axes[0].plot(bins, sp.stats.chi2.pdf(bins, p), c='k', label=rf'$\chi^2(2T)/T$ density')
    axes[0].set_ylabel(r'Frequency')
    pchi2 = sp.stats.chi2.pdf(q, p)
    axes[0].text(6, max(pchi2), '$T=1$')

    for (i, Ti) in enumerate(TDIST):
        binmax = np.ceil(
            np.amax(np.hstack([data['kpi_q'][:, i] for data in cldata.values()]))
        )
        bins = np.linspace(0, 8, 51)
        for (j, (label, data)) in enumerate(cldata.items()):
            kpi = data['kpi_q']
            w = np.zeros(len(kpi[:,i])) + 1. / (bins[1] * len(kpi[:,i]))
            if label in q_compare:
                *_, p1 = axes[i+1].hist(kpi[:,i], density=True, bins=bins,
                                        weights=w, histtype='step', ls=linestyles[j], lw=1)
                *_, p2 = axes[i+1].hist(kpi[:,i], density=True, bins=bins,
                                        weights=w, histtype='stepfilled',
                                        color=p1[0].get_facecolor(), alpha=0.5)
                patches += [(p1[0],p2[0])]
            else:
                axes[i+1].hist(np.nan*kpi[:, i], density=True, bins=bins)
        q = np.linspace(0, 10, 1001)
        pchi2 = sp.stats.chi2.pdf(q*Ti, Ti*p)*Ti
        axes[i+1].plot(q, pchi2, c='k', ls='-', label=r'$\chi^2(2T)/T$ density')
        axes[i+1].set_ylabel(r'Frequency')
        axes[i+1].set_ylim([0, 1.2*max(pchi2)])
        axes[i+1].text(6, 0.2*max(pchi2), rf'$T={Ti}$')
    axes[-1].set_xlabel(r'$\langle e^\top R_e^{-1}e\rangle_T$')
    axes[-1].set_xlim([0, max(bins)])

    axes[0].legend(patches, q_compare, ncol=1, loc='upper right')

    fig.tight_layout(pad=1)
    pdf.savefig(transparent=True)
    plt.close()
