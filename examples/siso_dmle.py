from control import c2d, ss, tf, dlqe
from control.matlab import lsim

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.linalg import block_diag, solve_discrete_are
from scipy.optimize import minimize
from scipy.signal import dlsim
from time import time

## Force this script to start in the main repo directory
import sys
import os
main_dir = os.path.dirname(os.path.abspath(__file__)) + '/..'
os.chdir(main_dir)
sys.path.append(main_dir)

from idtools.arx import ARX
from idtools.ssid import *
from idtools.ssmle import *
from idtools.ssmle import _canonical_transform

###################
## Script config ##
###################
labels = [
    f'ARX',
    f'Ho-Kalman',
    'maximum likelihood'
]

# System
k = 1
tau = 1

num = k
den = [tau, 1]
Delta = 0.1

sys = c2d(ss(tf(num, den)), Delta)
A = sys.A
B = sys.B
C = sys.C
n, m = B.shape
p = C.shape[0]

nrep = 2
tfin = 5*tau
nsam = int(np.floor(tfin/Delta)) + 1
varu = 0.1
vary = 0.001

# choose a seed that leads to a nice looking comparison of step responses
np.random.seed(4)
u = np.ones([1, nsam])
u = np.hstack([u, np.zeros([1, nsam])])
u = np.hstack([u, -u])
u = np.hstack([u for i in range(nrep)])
u = np.hstack([np.zeros([1, 5]), u])
nsim = u.shape[1]
print(nsim)
tsim = np.hstack([np.array(list(range(-5, 0)))*Delta, Delta*np.arange(nsim-5)])
x = np.zeros([n, nsim])
y = np.zeros([p, nsim])
xp = np.zeros([n, 1])
for i in range(nsim):
    x[:, i] = xp
    y[:, i] = C@xp + np.sqrt(vary)*np.random.randn()
    xp = A@xp + B@u[:, i] + B*np.sqrt(varu)*np.random.randn()

yhat0 = A@y + B@u
r0 = y[:, 1:] - yhat0[:, :-1]
ssq0 = r0@r0.T

Qw = np.atleast_2d(B*varu*B)
Rv = np.atleast_2d(vary)
K, P, AK = dlqe(A, np.eye(n), C, Qw, Rv)
Re = C*P*C + vary
print(f"Plant params: \t A ={A[0,0]:.4f}, B={B[0,0]:.4f},")
print(f"\t\t Qw={(B*varu*B)[0, 0]:.4f},   Rv={vary:.4f},")
print(f"\t\t K ={K[0, 0]:.4f},   Re={Re[0, 0]:.4f}.")

theta0 = (A, B, np.vstack([K, 0]), Re)
plant = {
    'A': A,
    'B': B,
    'C': C,
    'Bd': np.zeros((n, 0)),
    'Cd': np.zeros((p, 0)),
    'K' : K,
    'Re' : Re,
    'Sd': block_diag(Qw, Rv)
}

## Model fit settings
Bd = np.zeros((n, p))
Cd = np.eye(p)
npast = 5

model_settings = dict(initial_state=False, feedthrough=False, cross_cov=False,
                      initial_state_cov=False)
opt_settings = dict(rho=0e-6, mu=1e-6, method='ipopt', verbosity=4,
                    max_iter=100, with_jit=False)


## Helper functions for augmented disturbance models
## TODO offload these to package
def augment_system(A, B, C, Bd, Cd):
    m = B.shape[1]
    p, n = C.shape

    Aaug = np.block([[A, Bd], [np.zeros((p, n)), np.eye(p)]])
    Baug = np.vstack([B, np.zeros((p, m))])
    Caug = np.hstack([C, Cd])

    return Aaug, Baug, Caug


def disturbance_filter(A, B, C, Bd, Cd, Sd):
    p, n = C.shape

    Aaug, Baug, Caug = augment_system(A, B, C, Bd, Cd)
    Qaug = Sd[:n+p, :n+p]
    Saug = Sd[:n+p, n+p:]
    Rv = Sd[n+p:, n+p:]

    P = solve_discrete_are(Aaug.T, Caug.T, Qaug, Rv, s=Saug, balanced=False)
    K = np.linalg.solve(Caug@P@Caug.T+Rv, Caug@P@Aaug.T).T
    Re = Caug@P@Caug.T + Rv

    return K, Re


def estimate_disturbance_cov(X, U, Y, A, B, C, Bd, Cd):
    m = B.shape[1]
    p, n = C.shape

    ## Get long-range prediction error
    _, yf, _ = dlsim((A, B, C, np.zeros((p, m)), 1), U.T)
    e = Y - yf.T

    ## TODO check if output disturbance model
    ## If output, d = e
    d = e
    ## TODO Otherwise, we have to do a big linear regression to find d.

    ## Get noise cov
    ish = Y.shape[1] - X.shape[1] + 1
    Ns = Y.shape[1] - ish - 1

    Aaug, Baug, Caug = augment_system(A, B, C, Bd, Cd)
    Theta = np.block([[Aaug, Baug], [Caug, np.zeros((p, m))]])
    smat = np.vstack([X[:, 1:-1], d[:, ish+1:], Y[:, ish:-1]])
    tmat = np.vstack([X[:, :-2], d[:, ish:-1], U[:, ish:-1]])
    resid = smat - Theta@tmat
    Sd = resid@resid.T / Ns

    return Sd


## Collect model fitting data
data = {'t': tsim, 'u': u, 'y': y}
models = dict()
for label in labels:
    C1 = np.eye(n)
    Qw1 = None
    Qd1 = None
    K1 = None

    t0 = time()
    if label == f'ARX':
        Theta1, Sigma1, _, _ = arx(y, u, 1)
        A = Theta1[:, :1]
        B = Theta1[:, 1:]
        C = np.eye(n)
        D = np.zeros((p, m))
        X = y

        ## Augment with disturbance model
        Aaug, Baug, Caug = augment_system(A, B, C, Bd, Cd)
        Sd = estimate_disturbance_cov(X, u, y, A, B, C, Bd, Cd)

        ## Grab final Kalman filter terms
        K, Re = disturbance_filter(A, B, C, Bd, Cd, Sd)
    elif label == f'Ho-Kalman':
        A, B, C, D, Qw, Rv, Swv, X = nucnormid(
            y, u, npast, n=n, rho=0,
            cross_cov=False,
            feedthrough=False,
            return_states=True
        )
        indices, T = _canonical_transform(A, C, indices='best')
        Tinv = np.linalg.inv(T)
        A, B, C, Qw, Swv, X = T@A@Tinv, T@B, C@Tinv, T@Qw@T.T, T@Swv, T@X

        ## Augment with disturbance model
        Aaug, Baug, Caug = augment_system(A, B, C, Bd, Cd)

        Sd = estimate_disturbance_cov(X, u, y, A, B, C, Bd, Cd)

        ## Grab final Kalman filter terms
        K, Re = disturbance_filter(A, B, C, Bd, Cd, Sd)

        ## Produce initial guess for the optimization-based methods
        init_sys = KFModel(Aaug, Baug, Caug, K=K, Re=Re)
    elif label == 'maximum likelihood':
        ## Make and fit model
        sys = observable_canonical_dmodel(init_sys, mtype='kf',
                                          noise_type='sqrt', **model_settings)
        params, stats = sys.fit(u, y, **opt_settings)

        ## Unpack terms
        Aaug, Baug, Caug, _, _, K, ReL, *_ = (M.full() for M in params)
        A = Aaug[:n, :n]
        B = Baug[:n, :]
        C = Caug[:, :n]
        Re = ReL@ReL.T
    t1 = time() - t0

    ## Print some results
    print('Aaug = ')
    print(Aaug)
    print('Baug = ')
    print(Baug)
    print('Caug = ')
    print(Caug)
    print('K = ')
    print(K)
    print('Re = ')
    print(Re)

    print('u->y steady-state gain:')
    print(C@np.linalg.inv(np.eye(A.shape[0])-A)@B)
    AK = Aaug - K@Caug
    print(f'eigs(AK)={np.linalg.eigvals(AK)}')
    print(f'sprad(AK)={np.amax(np.abs(np.linalg.eigvals(AK)))}')

    print(f"Time to fit {label} model: {t1:.4f}s")

    ## Save data
    D = np.zeros((p, m))
    est = (A, B, C, D, Delta)
    _, yf, _ = dlsim(est, u.T)

    AK = Aaug-K@Caug
    BK = np.hstack([Baug-K@D, K])
    DK = np.hstack([D, np.zeros((p, p))])
    est = (AK, BK, Caug, DK, Delta)
    zdat = np.vstack([u, y])
    _, yhat, _ = dlsim(est, zdat.T)

    theta1 = (A, B, K, Re)

    theta_error = np.array([np.linalg.norm(M-M0)/np.linalg.norm(M0) for M, M0 in zip(theta1, theta0)])

    models[label] = {
        'est': dict(A=A, B=B, C=C, Bd=Bd, Cd=Cd, K=K, Re=Re),
        'yf': yf,
        'yhat': yhat,
        't': t1,
        'err': theta_error,
    }

with open('data/siso_dmle.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=-1)
    pickle.dump(models, handle, protocol=-1)
    pickle.dump(plant, handle, protocol=-1)
