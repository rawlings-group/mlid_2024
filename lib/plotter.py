from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np

default_cycler = (
    cycler(color=['b', 'g', 'r', 'm', 'c', 'y', 'orange', 'k']) +
    cycler(linestyle=['-', '--', ':', '-.', '-', '--', ':', '-.'])
)
eigs_cycler = cycler(marker=['o','s','p','*','^','>','v','<'])
plt.rc('axes', prop_cycle=default_cycler)
data_args = dict(lw=1)
# setpoint_args = dict(lw=1, c='k', ls='-')
setpoint_args = dict(label='Setpoints', lw=0.5, c='k', ls='--')
disturbance_args = dict(label='Disturbances', lw=0.5, c='r', ls=':')
confidence_args = dict(label=r'$3\sigma$ confidence interval', alpha=0.2)

def plot_tclab(t, u, y, ysp=None, yf=None, ylb=None, yub=None, din=None,
               dout=None, legend=True, figsize=(10, 5), data_label='Data',
               fig=None, axes=None):
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    nleg = 0

    if ysp is not None:
        axes[0, 0].step(t, ysp[0, :], **setpoint_args)
        axes[1, 0].step(t, ysp[1, :], **setpoint_args)
        nleg += 1

    if dout is not None:
        axes[0, 0].step([], [], **disturbance_args)
        axout = [ax.twinx() for ax in axes[:, 0]]
        axout[0].step(t, dout[0, :], **disturbance_args)
        axout[1].step(t, dout[1, :], **disturbance_args)
        axout[0].set_ylabel(r"$p_1$", rotation=0, labelpad=10)
        axout[1].set_ylabel(r"$p_2$", rotation=0, labelpad=10)
        nleg += 1
    elif dout is None and isinstance(u, dict):
        axes[0, 0].step(np.NaN, np.NaN, '-', color='none', label=' ')

    if din is not None:
        axin = [ax.twinx() for ax in axes[:, 1]]
        axin[0].step(t, din[0, :], **disturbance_args)
        axin[1].step(t, din[1, :], **disturbance_args)
        axin[0].set_ylabel(r"$m_1$", rotation=0, labelpad=10)
        axin[1].set_ylabel(r"$m_2$", rotation=0, labelpad=10)

    if not isinstance(u, dict):
        axes[0, 1].step(t, u[0, :], **data_args, label=data_label, c='k', ls='-')
        axes[1, 1].step(t, u[1, :], **data_args, label=data_label, c='k', ls='-')
    else:
        for (label, utmp) in u.items():
            axes[0, 1].step(t, utmp[0, :], **data_args, label=label)
            axes[1, 1].step(t, utmp[1, :], **data_args, label=label)

    if not isinstance(y, dict):
        axes[0, 0].step(t, y[0, :], **data_args, label=data_label, c='k', ls='-')
        axes[1, 0].step(t, y[1, :], **data_args, label=data_label, c='k', ls='-')
        nleg += 1
        axes[0, 0].step(np.NaN, np.NaN, '-', color='none', label=' ')
    else:
        for (label, ytmp) in y.items():
            axes[0, 0].plot(t, ytmp[0, :], **data_args, label=label)
            axes[1, 0].plot(t, ytmp[1, :], **data_args, label=label)
            nleg += 1

    ## Plot estimates
    if yf is not None:
        if not isinstance(yf, dict):
            yf = {'Predictions': yf}
        for (label, ytmp) in yf.items():
            axes[0, 0].plot(t, ytmp[0, :], label=label)
            axes[1, 0].plot(t, ytmp[1, :], label=label)
            nleg += 1

    ## Plot CI
    if ylb is not None and yub is not None:
        axes[0, 0].fill_between(t, ylb[0, :], yub[0, :], **confidence_args)
        axes[1, 0].fill_between(t, ylb[1, :], yub[1, :], **confidence_args)
        nleg += 1

    if legend:
        axes[0, 0].legend(bbox_to_anchor=(0, 1, 2.15, 0), loc="lower left",
                          mode="expand", ncol=5)

    axes[0, 1].set_ylabel(r"$u_1$", rotation=0, labelpad=10)
    axes[1, 1].set_ylabel(r"$u_2$", rotation=0, labelpad=10)
    axes[0, 0].set_ylabel(r"$y_1$", rotation=0, labelpad=10)
    axes[1, 0].set_ylabel(r"$y_2$", rotation=0, labelpad=10)
    axes[1, 0].set_xlabel(r"time (s)")
    axes[1, 1].set_xlabel(r"time (s)")
    axes[1, 0].set_xlim([0, max(t)+1])
    axes[1, 1].set_xlim([0, max(t)+1])

    return fig, axes

def plot_siso(t, u, y, ysp=None, dist=None, yf=None, legend=True, figsize=(8, 6)):
    if u is not None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]

    nleg = 0

    if ysp is not None:
        axes[0].step(t, ysp[0, :], **setpoint_args)
        nleg += 1

    if dist is not None:
        axes[0].step(t, dist[0, :], **disturbance_args)
        nleg += 1

    if u is not None:
        if not isinstance(u, dict):
            axes[1].step(t, u[0, :], **data_args, label='Data', c='k')
        else:
            for (label, utmp) in u.items():
                axes[1].step(t, utmp[0, :], **data_args, label=label)

    if not isinstance(y, dict):
        axes[0].step(t, y[0, :], **data_args, label='Data', c='k')
        nleg += 1
    else:
        for (label, ytmp) in y.items():
            axes[0].plot(t, ytmp[0, :], **data_args, label=label)
            nleg += 1

    if yf is not None:
        if not isinstance(yf, dict):
            yf = {'Predictions': yf}
        for (label, ytmp) in yf.items():
            axes[0].plot(t, ytmp[0, :], label=label)
            nleg += 1

    if legend:
        # axes[0].legend(bbox_to_anchor=(0, 1, 2.15, 0), loc="lower left",
                          # mode="expand", ncol=nleg)
        axes[0].legend(ncol=3)

    axes[0].set_ylabel(r"$y$", rotation=0, labelpad=10)
    axes[0].set_xlim([min(t), max(t)+1])
    if u is not None:
        axes[1].set_ylabel(r"$u$", rotation=0, labelpad=10)
        axes[1].set_xlabel(r"time (s)")
    else:
        axes[0].set_xlabel(r"time (s)")

    return fig, axes


class TCLabPlotter():
    def __init__(self, plot_dist=False, figsize=(10, 4.5)):
        # Set up plot and lines
        self.figure, self.axes = plt.subplots(2, 2, figsize=(10, 4.5), sharex=True)
        self.axtwins = [[ax.twinx() for ax in axlist] for axlist in self.axes]
        self.axtwins = np.append(*self.axtwins).reshape(2, 2)

        lu1, = self.axes[0, 1].step([], [], lw=0.5, c='b')
        lu2, = self.axes[1, 1].step([], [], lw=0.5, c='b')
        ly1, = self.axes[0, 0].plot([], [], lw=0.5, c='b')
        ly2, = self.axes[1, 0].plot([], [], lw=0.5, c='b')
        lysp1, = self.axes[0, 0].step([], [], lw=0.5, ls=':', c='k')
        lysp2, = self.axes[1, 0].step([], [], lw=0.5, ls=':', c='k')
        ldout1, = self.axtwins[0, 0].step([], [], lw=0.5, ls='--', c='r')
        ldout2, = self.axtwins[1, 0].step([], [], lw=0.5, ls='--', c='r')
        ldin1, = self.axtwins[0, 1].step([], [], lw=0.5, ls='--', c='r')
        ldin2, = self.axtwins[1, 1].step([], [], lw=0.5, ls='--', c='r')
        self.lines = np.array([[lu1, ly1, lysp1, ldout1, ldin1],
                               [lu2, ly2, lysp2, ldout2, ldin2]])

        # Labels
        self.axes[0, 1].set_ylabel(r"$u_1$", rotation=0, labelpad=10)
        self.axes[1, 1].set_ylabel(r"$u_2$", rotation=0, labelpad=10)
        self.axtwins[0, 1].set_ylabel(r"$m_1$", rotation=0, labelpad=10)
        self.axtwins[1, 1].set_ylabel(r"$m_2$", rotation=0, labelpad=10)
        self.axes[0, 0].set_ylabel(r"$y_1$", rotation=0, labelpad=10)
        self.axes[1, 0].set_ylabel(r"$y_2$", rotation=0, labelpad=10)
        self.axtwins[0, 0].set_ylabel(r"$p_1$", rotation=0, labelpad=10)
        self.axtwins[1, 0].set_ylabel(r"$p_2$", rotation=0, labelpad=10)
        self.axes[1, 0].set_xlabel(r"time (s)")
        self.axes[1, 1].set_xlabel(r"time (s)")

        # Turn off disturbance plotting
        if not plot_dist:
            for ax in self.axtwins.flatten():
                ax.axis('off')

        # Autoscale
        for ax in np.append(self.axes.flatten(), self.axtwins.flatten()):
            ax.set_autoscalex_on(True)
            ax.set_autoscaley_on(True)

        self.figure.tight_layout()


    def __call__(self, u, y, ysp=None, dout=None, din=None):
        # Update data
        signals = (u, y, ysp, dout, din)
        for (k, signal) in enumerate(signals):
            if signal is not None:
                t = np.arange(signal.shape[1])
                self.lines[0, k].set_xdata(t)
                self.lines[1, k].set_xdata(t)
                self.lines[0, k].set_ydata(signal[0, :])
                self.lines[1, k].set_ydata(signal[1, :])

        # Rescale
        for ax in np.append(self.axes, self.axtwins).flatten():
            ax.relim()
            ax.autoscale_view()

        # Draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

def plot_eigs(A, legend=True, figsize=(4, 3)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_prop_cycle(eigs_cycler, default_cycler)

    # Unit circle
    circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
    ax.add_patch(circ)

    # Eigs
    if not isinstance(A, dict):
        eigs = np.linalg.eigvals(A)
        ax.scatter(np.real(eigs), np.imag(eigs))
    else:
        for (label, Ai) in A.items():
            eigs = np.linalg.eigvals(Ai)
            ax.plot(np.real(eigs), np.imag(eigs), label=label, ls='')

    # Legend
    ax.grid()
    if isinstance(A, dict) and legend:
        ax.legend(ncol=len(A.keys()))

    return fig, ax
