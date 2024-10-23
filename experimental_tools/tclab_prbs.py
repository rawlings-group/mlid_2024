import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.linalg import block_diag, solve_discrete_are
from plotter import TCLabPlotter
import tclab

import time

## experiment setup
m = 2
p = 2

Du = 10*np.eye(m)
u0 = 30*np.ones((m, 1))
Nstep = 300 # in seconds

u = np.array([[1., 1., 1., 0., 0., 1, 0., 0.],
              [0., 1., 0., 1., 1., 1., 0., 0.]])
u = Du@(2*u-1)
u = np.kron(u, np.array([[0, 1]]))
u = np.append(u, np.zeros((2, 1)), axis=1)
u = np.kron(u, np.ones((1, Nstep)))
u += u0

## load TCLab and warm up
with tclab.TCLab() as lab:
    Nwarmup = 1200
    lab.Q1(u0[0, 0])
    lab.Q2(u0[1, 0])

    def measure(lab, N=1, pwm=False):
        if pwm:
            return np.array([[lab.U1], [lab.U2]])
        else:
            return np.array([[lab.T1], [lab.T2]])

    plotter = TCLabPlotter()
    plotter.figure.show()

    uplot = measure(lab, pwm=True)
    yplot = measure(lab)
    print(f'Warming up TCLab for {Nwarmup} seconds.')
    for t in tclab.clock(Nwarmup):
        lab.Q1(u0[0, 0])
        lab.Q2(u0[1, 0])
        uplot = np.append(uplot, measure(lab, pwm=True), axis=1)
        yplot = np.append(yplot, measure(lab), axis=1)
        plotter(uplot, yplot)

        if t % 60 == 0 and t > 0:
            print(f'Warming up TCLab for {Nwarmup-t} more seconds.')

    y0 = measure(lab)

    ## run experiment
    N = u.shape[1]
    y = np.zeros((p, N))
    print(f'Starting open-loop ID experiment.')
    for (i, t) in enumerate(tclab.clock(N-1)):
        ## Send inputs
        u[0, i] = lab.Q1(u[0, i])
        u[1, i] = lab.Q2(u[1, i])

        ## Measure
        y[:, i] = measure(lab)[:, 0]

        ## Plot and wait
        plotter(u[:, :i], y[:, :i])
        if i % 60 == 0:
            print(f'{N-i} s to go...')

    print(f'Completed experiment.')

    data = dict(t=np.arange(N), u=u, y=y, u0=u0, y0=y0)

## Export data
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f'tclab_prbs_{timestamp}.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=-1)
