# [depends] %LIB%/tclab_data.mat %LIB%/arx.py %LIB%/dmid.py %LIB%/pem.py %LIB%/arx.py %LIB%/ssid.py %LIB%/regression.py

# Try out direct maximum likelihood estimation on some TClab experimental data.
#
# skuntz, 04/13/2023

# Experiment 1

from control import c2d, ss, tf, dlqe
from control.matlab import lsim
from arx import ARX
from pem import ss_predict, ml4lti
from ssid import *

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.io import loadmat
from scipy.linalg import block_diag, solve_discrete_are
from scipy.optimize import minimize
from scipy.signal import dlsim
from time import time

from ssmle import *

# Import and process TCLab data
with open('../lib/tclab_prbs.pickle', 'rb') as handle:
    data = pickle.load(handle)

tdat = data['t']
udat = data['u']
ydat = data['y']


us0 = np.array([30, 30])[:, np.newaxis]
ys0 = np.mean(ydat, 1)[:, np.newaxis]

m, N = udat.shape
p, _ = ydat.shape

udat = udat - us0
ydat = ydat - ys0

Delta = 1

## Model fitting settings (always start with the ARX->SS number of states)
ny = 2
nu = 1

n = ny*p + nu*m

npast = 10

n = 2
A0, B0, C0, D0, Qw0, Rv0, Swv0 = nucnormid(
    ydat, udat, npast, n=n, rho=0,
    cross_cov=True,
    feedthrough=True
)
# A0, B0, C0, D0, Qw0, Rv0, Swv0 = subspaceid(
#     ydat, udat, npast, n=n,
#     cross_cov=True,
#     feedthrough=True
# )
P0 = solve_discrete_are(A0.T, C0.T, Qw0, Rv0, s=Swv0, balanced=False)
K0 = np.linalg.solve(C0@P0@C0.T+Rv0, (A0@P0@C0.T+Swv0).T).T
Re0 = C0@P0@C0.T + Rv0

initial_guess = dict(A0=A0, B0=B0, C0=C0, K0=K0, Re0=Re0, Qw0=Qw0, Rv0=Rv0)
model_settings = dict(initial_state=False, feedthrough=False, cross_cov=False,
                      initial_state_cov=False, use_prior=True, stable=False)
opt_settings = dict(rho=0e-6, delta=0e-1, mu=0e-6, method='ipopt', verbosity=5,
                    max_iter=100, with_jit=True, rescale=False, casaditype='MX')
settings = dict(**initial_guess, **model_settings, **opt_settings)


## Collect model fitting data
data = {'t': tdat, 'u': udat, 'y': ydat, 'us0': us0, 'ys0': ys0}
labels = [
    # 'ARX',
    # 'HK-ARX',
    # 'CCA',
    # # 'NN-ARX',
    # 'PEM',
    # 'KF-MLE',
    # 'TVKF-MLE',
    # 'PEM canonical',
    'ML',
    r'ML ($\rho(A_L)\leq 0.9$))',
    r'ML ($\rho(A_L)\leq 0.6$))',
    r'ML ($\rho_{0.001}(A_L)\leq 0.9$))',
    r'ML ($\rho_{0.001}(A_L)\leq 0.6$))',
    # 'ML (in det form)',
    # 'ML (in logdet form)',
    # 'TVKF-MLE canonical',
]

init_sys = KFModel(A0, B0, C0, K=K0, Re=Re0)

models = {}
for label in labels:
    print('Fitting ' + label + ' model...')
    Qw1 = None
    Swv1 = None
    K1 = None

    t0 = time()
    if label == 'ARX':
        sys = ARX(ny, nu, feedthrough=False)
        sys.fit(ydat, udat)
        A1, B1, C1, _, S1 = sys.arx2ss()
        ntmp = A1.shape[1]
        Qw1 = S1[:ntmp, :ntmp]
        Swv1 = S1[:ntmp, ntmp:]
        Rv1 = S1[ntmp:, ntmp:]
    elif label == 'HK-ARX':
        A1, B1, C1, D1, Qw1, Rv1, Swv1 = \
            nucnormid(ydat, udat, npast, n=n, rho=0, cross_cov=False,
                      feedthrough=False)
    elif label == 'N4SID':
        A1, B1, C1, D1, Qw1, Rv1, Swv1 = \
            subspaceid(ydat, udat, npast, rho=0, cross_cov=False,
                       feedthrough=False, weights=label)
    elif label == 'CCA':
        A1, B1, C1, D1, Qw1, Rv1, Swv1 = \
            subspaceid(ydat, udat, npast, rho=0, cross_cov=False,
                       feedthrough=False)
    elif label == 'NN-ARX':
        A1, B1, C1, D1, Qw1, Rv1, Swv1 = \
            nucnormid(ydat, udat, npast, n=n, rho=[30], cross_cov=True,
                      feedthrough=True, verbosity=5)
    elif label == 'PEM':
        A1, B1, C1, *_ = ml4lti(ydat, udat, n, **settings)
    elif label == 'KF-MLE':
        A1, B1, C1, K1, Re1, *_ = ml4lti(ydat, udat, n, **settings, model='kf')
    elif label == 'TVKF-MLE':
        A1, B1, C1, Qw1, Rv1, *_ = ml4lti(ydat, udat, n, **settings, model='slti')
    elif label == 'PEM canonical':
        sys = observable_canonical(init_sys, **model_settings, rescale=True,
                          rho=opt_settings['rho'])
        A1, B1, C1, *_ = (M.full() for M in sys.fit(udat, ydat))
    elif label == 'ML':
        sys = observable_canonical(init_sys, mtype='kf', **model_settings)
        params, stats = sys.fit(udat, ydat, rescale=True, rho=opt_settings['rho'])
        A1, B1, C1, _, _, K1, ReL1, *_ = (M.full() for M in params)
        Re1 = ReL1@ReL1.T
    elif label == r'ML ($\rho(A_L)\leq 0.9$))':
        sys = observable_canonical(init_sys, mtype='kf', **model_settings)
        sys.add_eigenvalue_constraint(delta=0.1, alpha=1e-6, beta=100, method='lmi')
        # sys.add_eigenvalue_constraint(delta=0, alpha=1e-6, beta=100, method='lmi', cons_type='continuity')
        params, stats = sys.fit(udat, ydat)
        A1, B1, C1, _, _, K1, ReL1, *_ = (M.full() for M in params)
        Re1 = ReL1@ReL1.T
    elif label == r'ML ($\rho(A_L)\leq 0.6$))':
        sys = observable_canonical(init_sys, mtype='kf', **model_settings)
        sys.theta0[:n**2] *= 0.59/np.amax(np.abs(np.linalg.eigvals(A0-K0@C0)))
        sys.theta0[n**2+n*m:n**2+n*m+n*p] *= 0.59/np.amax(np.abs(np.linalg.eigvals(A0-K0@C0)))

        sys.add_eigenvalue_constraint(delta=0.4, alpha=1e-6, beta=100, method='lmi')
        # sys.add_eigenvalue_constraint(delta=0, alpha=1e-6, beta=100, method='lmi', cons_type='continuity')

        params, stats = sys.fit(udat, ydat)
        A1, B1, C1, _, _, K1, ReL1, *_ = (M.full() for M in params)
        Re1 = ReL1@ReL1.T
    elif label == r'ML ($\rho_{0.001}(A_L)\leq 0.9$))':
        sys = observable_canonical(init_sys, mtype='kf', **model_settings)
        sys.add_eigenvalue_constraint(delta=0.1, alpha=1e-6, beta=100)
        # sys.add_eigenvalue_constraint(delta=0, alpha=1e-6, beta=100, method='lmi', cons_type='continuity')
        params, stats = sys.fit(udat, ydat)
        A1, B1, C1, _, _, K1, ReL1, *_ = (M.full() for M in params)
        Re1 = ReL1@ReL1.T
    elif label == r'ML ($\rho_{0.001}(A_L)\leq 0.6$))':
        sys = observable_canonical(init_sys, mtype='kf', **model_settings)
        sys.theta0[:n**2] *= 0.59/np.amax(np.abs(np.linalg.eigvals(A0-K0@C0)))
        sys.theta0[n**2+n*m:n**2+n*m+n*p] *= 0.59/np.amax(np.abs(np.linalg.eigvals(A0-K0@C0)))
        sys.add_eigenvalue_constraint(delta=0.4, alpha=1e-6, beta=100)
        # sys.add_eigenvalue_constraint(delta=0, alpha=1e-6, beta=100, method='lmi', cons_type='continuity')

        params, stats = sys.fit(udat, ydat)
        A1, B1, C1, _, _, K1, ReL1, *_ = (M.full() for M in params)
        Re1 = ReL1@ReL1.T
    elif label == 'ML (in logdet form)':
        sys = observable_canonical(init_sys, mtype='kf', **model_settings)
        theta_new = sys.theta[:-int(p*(p+1)/2)]
        theta0_new = sys.theta0[:-int(p*(p+1)/2)]
        sys = KFModel(sys.A, sys.B, sys.C, K=sys.K, Re=np.diag([1, 1]), theta=theta_new,
                      theta0=theta0_new, **model_settings)
        params, stats = sys.fit(udat, ydat, **opt_settings)
        A1, B1, C1, _, _, K1, ReL1, *_ = (M.full() for M in params)
        print(ReL1)
        AK1 = A1-K1@C1
        BK1 = np.hstack([B1-K1@D1, K1])
        DK1 = np.hstack([D1, np.zeros((p, p))])
        est1 = (AK1, BK1, C1, DK1, Delta)
        zdat = np.vstack([udat, ydat])
        _, yhat1, _ = dlsim(est1, zdat.T)
        Re1 = (1/N)*(ydat-yhat1.T)@(ydat-yhat1.T).T
        ReL1 = np.linalg.cholesky(Re1)
    elif label == 'ML (in det form)':
        sys = observable_canonical(init_sys, mtype='kf', **model_settings)
        theta_new = sys.theta[:-int(p*(p+1)/2)]
        theta0_new = sys.theta0[:-int(p*(p+1)/2)]
        sys = MinDetModel(sys.A, sys.B, sys.C, K=sys.K, theta=theta_new,
                          theta0=theta0_new, **model_settings)
        params, stats = sys.fit(udat, ydat, rescale=True, rho=opt_settings['rho'])
        A1, B1, C1, _, _, K1, ReL1, *_ = (M.full() for M in params)
        Re1 = ReL1@ReL1.T
    elif label == 'TVKF-MLE canonical':
        init_sys = LGSSModel(A0, B0, C0, K=K0, Re=Re0)
        sys = observable_canonical(init_sys, mtype='slti', noise_type='K',
                          **model_settings, rescale=True, rho=opt_settings['rho'], alpha=1e-3)
        out = sys.likelihood(udat[:, :1040], ydat[:, :1040])
        print(cs.Function('f', [out[1]], [out[0]])(sys.theta0))
        A1, B1, C1, _, _, K1, ReL1 = (M.full() for M in sys.fit(udat, ydat))
        Re1 = ReL1@ReL1.T
    t1 = time() - t0

    D1 = np.zeros((p, m))
    est1 = (A1, B1, C1, D1, Delta)
    _, yf1, _ = dlsim(est1, udat.T)

    print(f'sprad(A)={np.amax(np.abs(np.linalg.eigvals(A1)))}')

    print(C1@np.linalg.inv(np.eye(A1.shape[0])-A1)@B1)

    if Qw1 is not None:
        if Swv1 is None:
            print('Setting Swv')
            Swv1 = np.zeros((n, p))
        print('Getting Kalman filter')
        P1 = solve_discrete_are(A1.T, C1.T, Qw1, Rv1, s=Swv1, balanced=False)
        K1 = np.linalg.solve(C1@P1@C1.T+Rv1, (A1@P1@C1.T+Swv1).T).T
        Re1 = C1@P1@C1.T + Rv1
    elif K1 is not None:
        print('Getting QRS matrices')
        Qw1 = K1@Re1@K1.T
        Swv1 = K1@Re1
        Rv1 = Re1

    if K1 is not None:
        AK1 = A1-K1@C1
        BK1 = np.hstack([B1-K1@D1, K1])
        DK1 = np.hstack([D1, np.zeros((p, p))])
        est1 = (AK1, BK1, C1, DK1, Delta)
        zdat = np.vstack([udat, ydat])
        _, yhat1, _ = dlsim(est1, zdat.T)
    else:
        yhat1 = np.full((N, p), np.nan)

    print(label + f" time: {t1:.4f}s")

    ## Print some results
    print('A = ')
    print(A1)
    print('B = ')
    print(B1)
    print('C = ')
    print(C1)
    print('K = ')
    print(K1)
    print('Re = ')
    print(Re1)

    print('u->y steady-state gain:')
    print(C1@np.linalg.inv(np.eye(A1.shape[0])-A1)@B1)
    print(f'eigs(AK)={np.linalg.eigvals(AK1)}')
    print(f'sprad(AK)={np.amax(np.abs(np.linalg.eigvals(AK1)))}')

    print(f"Time to fit {label} model: {t1:.4f}s")



    models[label] = {
        'yf': yf1.T,
        'yhat': yhat1,
        't': t1,
    }

with open('tclab_mle.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=-1)
    pickle.dump(models, handle, protocol=-1)
