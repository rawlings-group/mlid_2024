import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.linalg import block_diag, solve_discrete_are
from plotter import TCLabPlotter
import tclab

import mpctools as mpc

from datetime import datetime

## import models
with open('../build/tclab_dmle_stable.pickle', 'rb') as handle:
    data = pickle.load(handle)
    models = pickle.load(handle)

## Base model
base_label, base_model = list(models.items())[-1]

## experiment setup
m = data['u'].shape[0]
p = data['y'].shape[0]

Dy = np.diag([5, 5]) # in degrees C
# u0 = data['us0']
# y0 = data['ys0']
u0 = np.array([[50], [50]]) # in % of max power
y0 = np.array([[54], [47]]) # in degrees C
Nstep = 300 # in seconds

# ysp = np.array([[1., 1., 0., 0.],
                # [0., 1., 0., 1.]])
# ysp = np.array([[1., 1., 0., 0., 0., 1, 0., 0.],
#                 [0., 1., 0., 1., 1., 1., 0., 0.]])
# ysp = Dy@(2*ysp-1)
# ysp = np.kron(ysp, np.array([[0, 1]]))
# ysp = np.append(ysp, np.zeros((2, 1)), axis=1)
# ysp = np.kron(ysp, np.ones((1, Nstep)))

Ddin = np.diag([20, 20]) # in % of max power
Ddout = np.diag([5, 5]) # in degrees C
dout = np.array([[-1.,  0., -1., 1.],
                [0., 1., -1., -1.]])
din = dout
dout = Ddout@np.append(dout, np.zeros(dout.shape), axis=1)
din = Ddin@np.append(np.zeros(din.shape), din, axis=1)

dout = np.kron(dout, np.array([[0, 1]]))
dout = np.append(dout, np.zeros((2, 1)), axis=1)
dout = np.kron(dout, np.ones((1, Nstep)))

din = np.kron(din, np.array([[0, 1]]))
din = np.append(din, np.zeros((2, 1)), axis=1)
din = np.kron(din, np.ones((1, Nstep)))

ysp = 0*din
_, N = ysp.shape

Qs = np.eye(p)
Rs = 1e-2*np.eye(m)

## load TCLab and warm up
with tclab.TCLab() as lab:
# with open('../build/tclab_dmle.pickle', 'rb') as lab:
    Nwarmup = 600
    lab.Q1(u0[0, 0])
    lab.Q2(u0[1, 0])

    def measure(lab, N=1, pwm=False):
        # return np.array([[0.], [0.]])
        if pwm:
            return np.array([[lab.U1], [lab.U2]])
        else:
            return np.array([[lab.T1], [lab.T2]])

    plotter = TCLabPlotter(plot_dist=True)
    plotter.figure.show()

    def run_experiment(model, u0, y0, ysp, dout, din, Qs, Rs, label=None):
        ## Get model params
        A, B, C, Bd, Cd, K_lqe, _ = model['est'].values()
        n, nd = Bd.shape
        naug = n + nd

        Aaug = np.block([[A, Bd], [np.zeros((nd, n)), np.eye(nd)]])
        Baug = np.vstack([B, np.zeros((nd, m))])
        Caug = np.hstack([C, Cd])

        ## LQR gain
        Pi = solve_discrete_are(A, B, C.T@Qs@C, Rs, balanced=False)
        K_lqr = np.linalg.solve(B.T@Pi@B + Rs, B.T@Pi@A)

        ## SSTP gain
        T_sstp = np.linalg.solve(
            np.block([[np.eye(n)-A, -B], [C, np.zeros((p, m))]]),
            np.block([[Bd, np.zeros((n, nd))], [-Cd, np.eye(p)]])
        )

        ## Control loop
        N = ysp.shape[1]
        xaug = np.zeros((n+nd, N+1))
        u = np.zeros((m, N))
        y = np.zeros((p, N))
        times = np.zeros((N,))
        print(f'Starting {label} experiment.')
        for (i, t) in enumerate(tclab.clock(N-1)):
            ## Steady-state targets, zs=[xs;us]
            zs = T_sstp @ np.append(xaug[n:, i], ysp[:, i] - y0[:, 0])

            ## LQR
            utmp = -K_lqr@(xaug[:n, i] - zs[:n]) + zs[n:] + u0[:, 0]

            ## Implement control action
            u[0, i] = np.maximum(0,np.minimum(100,utmp[0]))
            u[1, i] = np.maximum(0,np.minimum(100,utmp[1]))
            lab.Q1(u[0, i] + din[0, i])
            lab.Q2(u[1, i] + din[1, i])

            # Measurement
            times[i] = t
            y[:, i] = measure(lab)[:, 0] + dout[:, i]

            ## Kalman filter, xaug=[xhat;dhat]
            e = y[:, i] - y0[:, 0] - Caug@xaug[:, i]
            xaug[:, i+1] = Aaug@xaug[:, i] + Baug@(u[:, i] - u0[:,0]) + K_lqe@e

            ## Plot and wait
            plotter(u[:, :i], y[:, :i], ysp[:, :i], dout[:, :i], din[:, :i])
            if i % 60 == 0:
                print(f'{label} experiment will last {N-i} more seconds.')

        print(f'Completed {label} experiment.')

        return times, u, y

    ## Warm up TCLab with base model
    run_experiment(base_model, u0, y0, np.zeros((p, Nwarmup)) + y0,
                   np.zeros((p, Nwarmup)), np.zeros((m, Nwarmup)), Qs, Rs,
                   label=f'Warmup (to T={y0[:,0]} with {base_label})')

    ## Warm up TCLab with a hold
    u = measure(lab, pwm=True)
    y = measure(lab)
    ytmp = y0
    print(f'Warming up TCLab for {Nwarmup} seconds.')
    for t in tclab.clock(Nwarmup):
    # for t in range(Nwarmup):
        lab.Q1(u0[0, 0])
        lab.Q2(u0[1, 0])
        u = np.append(u, measure(lab, pwm=True), axis=1)
        y = np.append(y, measure(lab), axis=1)
        ytmp = np.append(ytmp, y0, axis=1)
        plotter(u, y, ytmp)

        if t % 60 == 0 and t > 0:
            print(f'Warming up TCLab for {Nwarmup-t} more seconds.')

    y0 = measure(lab)
    ysp += y0

    ## run experiments
    experiments = dict()
    for (label, model) in models.items():
        ## Warm up TCLab with base model
        run_experiment(base_model, u0, y0, np.zeros((p, Nwarmup)) + y0,
                       np.zeros((p, Nwarmup)), np.zeros((m, Nwarmup)), Qs, Rs,
                       label=f'{label} warmup (with {base_label})')

        ## Run experiment and save data
        times, u, y = run_experiment(model, u0, y0, ysp, dout, din, Qs, Rs,
                                     label=label)

        experiments[label] = dict(t=np.arange(N), u=u, y=y, ysp=ysp, dout=dout,
                                  din=din, u0=u0, y0=y0, model=model, Qs=Qs,
                                  Rs=Rs)


## Export data
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f'tclab_cl_dist_{timestamp}.pickle', 'wb') as handle:
    pickle.dump(experiments, handle, protocol=-1)
