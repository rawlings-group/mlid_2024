# [depends] %LIB%/pem.py %LIB%/arx.py %LIB%/ssid.py %LIB%/regression.py

# ARX vs PEM for a first order system
#
# Experiment 1:
# - DONE Small sample SISO ID
# - TODO ~10-100(?) sample trajectories, a model fit on each
# - TODO dynamic (a,b) and stochastic (k,re)/(q,r) biplots for each ID method.
# - TODO time distribution for each ID method (or barplot w/ errors)
from control import c2d, ss, tf, dlqe
from control.matlab import lsim
from arx import arx
from pem import ss_predict, ml4lti, em4lti
from ssid import nucnormid

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.linalg import block_diag, solve_discrete_are
from scipy.optimize import minimize
from scipy.signal import dlsim
from time import time

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
labels = [
    # 'ARX',
    # f'HK-SID ($n_p$={npast})',
    # f'NN-ARX ($n_p$={npast})',
    # 'PEM (scipy)',
    'PEM (CasADi)',
    'SSKF-MLE',
    # 'TVKF-MLE',
    # 'EM'
]
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
    elif label == f'HK-SID ($n_p$={npast})':
        A1, B1, C1, D1, Qw1, Rv1, Swv1 = nucnormid(
            y, u, npast, n=1, rho=0,
            cross_cov=True,
            feedthrough=True
        )
    elif label == f'NN-ARX ($n_p$={npast})':
        A1, B1, C1, D1, Qw1, Rv1, Swv1 = \
            nucnormid(y, u, npast, n=1, rho=[30], cross_cov=True,
                      feedthrough=True)
    elif label == 'PEM (scipy)':
        res = minimize(lambda theta: pred_error(y[0, :], u[0, :], theta),
                       Theta1[0])
        A1 = np.array([[res.x[0]]])
        B1 = np.array([[res.x[1]]])
    elif label == 'PEM (CasADi)':
        A1, B1, C1 = ml4lti(y, u, 1, **settings, model='dlti')
    elif label == 'SSKF-MLE':
        A1, B1, C1, K1, Re1 = ml4lti(y, u, 1, **settings, model='kf')
    elif label == 'TVKF-MLE':
        A1, B1, C1, Qw1, Rv1 = ml4lti(y, u, 1, **settings, model='slti')
    elif label == 'EM':
        initial_guess_em = dict(A=A0, B=B0, C=C0, Qw0=B0*varu*B0, Rv0=vary)
        params = em4lti(y, u, 1, initial_guess=initial_guess_em, mu=1e-6,
                        initial_state=False, initial_state_cov=False,
                        feedthrough=False, cross_cov=False, max_iter=1000)

        A1 = params['A']
        B1 = params['B']
        C1 = params['C']
        Qw1 = params['S'][:1, :1]
        Rv1 = params['S'][1:, 1:]
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

    if Swv1 is not None:
        print('Canceling S matrix')
        L = np.linalg.inv(A1) @ Swv1 @ np.linalg.inv(C1.T)
        Qw1 -= A1@L@A1.T - L
        Rv1 -= C1@L@C1.T
        Swv1 -= A1@L@C1.T
        P1 = solve_discrete_are(A1.T, C1.T, Qw1, Rv1, s=Swv1, balanced=False)
        K1 = np.linalg.solve(C1@P1@C1.T+Rv1, (A1@P1@C1.T+Swv1).T).T
        Re1 = C1 @ P1 @ C1.T + Rv1

    yhat1 = A1*y + B1*u
    r1 = y[:, 1:] - yhat1[:, :-1]
    ssq1 = r1@r1.T
    est1 = (A1, B1, 1, 0, Delta)
    _, yf1, _ = dlsim(est1, u[0, :])

    # print(np.linalg.norm(y - yf1.T, ord='fro')**2)

    print(label + f" params:\t A ={A1[0, 0]:.4f}, B={B1[0, 0]:.4f}, "
          f"err={ssq1[0,0]:.4f}. ({t1:.4f}s)")
    if K1 is not None:
        print(f"\t\t Qw={Qw1[0, 0]:.6f}, Rv={Rv1[0, 0]:.4f}, "
              f"Swv={Swv1[0, 0]:.6f}")
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

with open('siso_mle.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=-1)
    pickle.dump(models, handle, protocol=-1)
