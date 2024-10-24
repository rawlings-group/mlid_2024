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

###################
## Script config ##
###################
labels = [
    'ARX',
    f'Ho-Kalman',
    f'NN-ARX',
    'PEM (scipy)',
    'PEM (CasADi)',
    'ML'
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

nrep = 1
tfin = 5*tau
nsam = int(np.floor(tfin/Delta)) + 1
varu = 0.1
vary = 0.01

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

K, P, AK = dlqe(A, np.eye(n), C, B*varu*B, vary)
Re = C*P*C + vary
print(f"Plant params: \t A ={A[0,0]:.4f}, B={B[0,0]:.4f},")
print(f"\t\t Qw={(B*varu*B)[0, 0]:.4f},   Rv={vary:.4f},")
print(f"\t\t K ={K[0, 0]:.4f},   Re={Re[0, 0]:.4f}.")
theta0 = np.array([A[0,0], B[0,0], K[0,0], Re[0,0], (B*varu*B)[0, 0], vary])

## Model fit settings
A0, B0, C0, K0, Re0 = A, B, C, K, Re

initial_guess = dict(A0=A0, B0=B0, C0=C0, K0=K0, Re0=Re0, Qw0=B0*varu*B0,
                     Rv0=vary)
model_settings = dict(initial_state=False, feedthrough=False, cross_cov=False,
                      initial_state_cov=False, stable=False)
opt_settings = dict(rho=0e-1, delta=0e-1, mu=1e-6, method='gn', verbosity=5,
                    max_iter=100, with_jit=False)
settings = dict(**initial_guess, **model_settings, **opt_settings)

npast = 5
bound = np.amax(np.absolute(np.linalg.eigvals(A-K@C)))**npast
print(f'||(A-KC)^np||<={bound:.2e}.')

def pred_error(y, u, theta):
    """Compute output prediction error."""
    yf = lsim(ss(theta[0], theta[1], 1, 0, Delta), u)[0]
    r = y - yf
    return r@r.T

## Collect model fitting data
data = {'t': tsim, 'u': u, 'y': y}
init_sys = KFModel(A0, B0, C0, K=K0, Re=Re0)

models = dict()
for label in labels:
    C1 = np.eye(n)
    Qw1 = None
    Swv1 = None
    K1 = None

    t0 = time()
    if label == 'ARX':
        Theta1, Sigma1, _, _ = arx(y, u, 1)
        B1 = Theta1[:, 1:]
        A1 = Theta1[:, :1]
    elif label == f'Ho-Kalman':
        A1, B1, C1, D1, Qw1, Rv1, Swv1 = nucnormid(
            y, u, npast, n=1, rho=0,
            cross_cov=True,
            feedthrough=True
        )
    elif label == f'NN-ARX':
        A1, B1, C1, D1, Qw1, Rv1, Swv1 = \
            nucnormid(y, u, npast, n=1, rho=[30], cross_cov=True,
                      feedthrough=True)
    elif label == 'PEM (scipy)':
        res = minimize(lambda theta: pred_error(y[0, :], u[0, :], theta),
                       Theta1[0])
        A1 = np.array([[res.x[0]]])
        B1 = np.array([[res.x[1]]])
    elif label == 'PEM (CasADi)':
        sys = observable_canonical(LSSModel(A0, B0, C0), **settings)
        params, stats = sys.fit(u, y, **settings)
        A1, B1, C1, *_ = (M.full() for M in params)
    elif label == 'ML':
        sys = observable_canonical(init_sys, **settings)
        params, stats = sys.fit(u, y, **settings)
        A1, B1, C1, _, _, K1, ReL1, *_ = (M.full() for M in params)
        Re1 = ReL1@ReL1.T

    t1 = time() - t0

    T = C1
    invT = np.linalg.inv(T)
    if Qw1 is not None:
        if Swv1 is None:
            print('Setting Swv')
            Swv1 = np.zeros((n, p))
        print('Getting Kalman filter')
        A1, B1, C1, Qw1, Swv1 = T@A1@invT, T@B1, C1@invT, T@Qw1@T.T, T@Swv1
        P1 = solve_discrete_are(A1.T, C1.T, Qw1, Rv1, s=Swv1, balanced=False)
        K1 = np.linalg.solve(C1@P1@C1.T+Rv1, (A1@P1@C1.T+Swv1).T).T
        Re1 = C1 @ P1 @ C1.T + Rv1
    elif K1 is not None:
        print('Getting QRS matrices')
        A1, B1, C1, K1 = T@A1@invT, T@B1, C1@invT, T@K1
        Qw1 = K1 @ Re1 @ K1.T
        Swv1 = K1 @ Re1
        Rv1 = Re1
    else:
        A1, B1, C1 = T@A1@invT, T@B1, C1@invT

    yhat1 = A1*y + B1*u
    r1 = y[:, 1:] - yhat1[:, :-1]
    ssq1 = r1@r1.T
    est1 = (A1, B1, 1, 0, Delta)
    _, yf1, _ = dlsim(est1, u[0, :])

    print(label + f" params:\t A ={A1[0, 0]:.4f}, B={B1[0, 0]:.4f}, "
          f"err={ssq1[0,0]:.4f}. ({t1:.4f}s)")
    if K1 is not None:
        print(f"\t\t K ={K1[0, 0]:.4f},   Re={Re1[0, 0]:.4f}.")
        theta1 = np.array([A1[0,0], B1[0,0], K1[0,0], Re1[0,0], Qw1[0,0], Rv1[0,0]])
    else:
        theta1 = np.array([A1[0,0], B1[0,0], np.nan, np.nan, np.nan, np.nan])


    models[label] = {
        'yf': yf1,
        'yhat': yhat1,
        't': t1,
        'err': np.absolute(theta0-theta1) / np.absolute(theta0),
    }

with open('data/siso_mle.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=-1)
    pickle.dump(models, handle, protocol=-1)
