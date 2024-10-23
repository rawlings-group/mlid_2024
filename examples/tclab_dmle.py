import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.io import loadmat
from scipy.linalg import solve_discrete_are
from scipy.signal import dlsim
from time import time

from ssid import subspaceid, nucnormid, jointmle
from ssmle import *
from ssmle import _canonical_transform

# np.set_printoptions(precision=4, suppress=True)


# Import and process TCLab data
with open('../data/tclab_prbs.pickle', 'rb') as handle:
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
n = 2
# npast = 1 # 
# npast = 2 # 
# npast = 50 # 
npast = 100 # 

dmodel = 'output'

model_settings = dict(initial_state=False, feedthrough=False, cross_cov=False,
                      initial_state_cov=False, dmodel=dmodel, indices='best')
opt_settings = dict(rho=0, epsilon=1e-6, method='ipopt', verbosity=5, max_iter=500,
                    use_prior=True)
cons_settings = opt_settings.copy()
cons_settings['beta'] = 1/0.03

## Helper functions for augmented disturbance models
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
    # NOTE: We assume an output disturbance model
    d = Y - yf.T

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

def get_init_sys(est):
    A, B, C, Bd, Cd, K, Re = est.values()
    Aaug, Baug, Caug = augment_system(A, B, C, Bd, Cd)
    return KFModel(Aaug, Baug, Caug, K=K, Re=Re)

## Collect model fitting data
data = {'t': tdat, 'u': udat, 'y': ydat, 'us0': us0, 'ys0': ys0}
labels = [
    ## NOTE: The initial guess model must come before any optimization-based methods.
    'Augmented PCA',
    # 'Augmented CCA',
    # 'Augmented HK',
    'Augmented ARX',
    'Unregularized ML',
    'Regularized ML 1',
    'Regularized ML 2',
    'Constrained ML 1',
    'Constrained ML 2',
    r'Reg. \& Cons. ML',
]

guess = 'Augmented ARX'


models = {}
for label in labels:
    ## Start and time
    print(f'Fitting {label} model...')
    t0 = time()

    if label == 'Augmented PCA':
        ## get stuff out of tclab_kuntz_rawlings_2022.mat
        est = loadmat('../lib/tclab_kuntz_rawlings_2022.mat')['est']
        A = est['A'][0, 0]
        B = est['B'][0, 0]
        C = est['C'][0, 0]
        Bd = est['Bd'][0, 0]
        Cd = est['Cd'][0, 0]
        Sd = est['Swwdv'][0, 0]

        ## Augmented matrices for printing later
        Aaug, Baug, Caug = augment_system(A, B, C, Bd, Cd)

        ## Grab final Kalman filter terms
        K, Re = disturbance_filter(A, B, C, Bd, Cd, Sd)

        sys = KFModel(Aaug, Baug, Caug, K=K, Re=Re)
    elif label == 'Augmented ARX':
        params, *_ = jointmle(ydat, udat[:,:-1], ydat[:,:-1], return_states=True)
        A, B, C, D, Qw, Rv, Swv, X = params
        indices, T = _canonical_transform(A, C, indices='best')
        Tinv = np.linalg.inv(T)
        A, B, C, Qw, Swv, X = T@A@Tinv, T@B, C@Tinv, T@Qw@T.T, T@Swv, T@X

        ## Augment with disturbance model
        Bd = np.zeros((n, p))
        Cd = np.eye(p)
        Aaug, Baug, Caug = augment_system(A, B, C, Bd, Cd)

        Sd = estimate_disturbance_cov(X, udat, ydat, A, B, C, Bd, Cd)

        ## Grab final Kalman filter terms
        K, Re = disturbance_filter(A, B, C, Bd, Cd, Sd)
    elif label == 'Augmented CCA':
        ## Get initial model in observability canonical form
        A, B, C, D, Qw, Rv, Swv, X = subspaceid(
            ydat, udat, 50, n=n, mu=1e-3, nu=1e-3,
            cross_cov=False,
            feedthrough=False,
            return_states=True
        )
        indices, T = _canonical_transform(A, C, indices='best')
        Tinv = np.linalg.inv(T)
        A, B, C, Qw, Swv, X = T@A@Tinv, T@B, C@Tinv, T@Qw@T.T, T@Swv, T@X

        ## Augment with disturbance model
        Bd = np.zeros((n, p))
        Cd = np.eye(p)
        Aaug, Baug, Caug = augment_system(A, B, C, Bd, Cd)

        Sd = estimate_disturbance_cov(X, udat, ydat, A, B, C, Bd, Cd)

        ## Grab final Kalman filter terms
        K, Re = disturbance_filter(A, B, C, Bd, Cd, Sd)
    elif label == 'Augmented HK':
        ## Get initial model in observability canonical form
        A, B, C, D, Qw, Rv, Swv, X = nucnormid(
            ydat, udat, 50, n=n, rho=0,
            cross_cov=False,
            feedthrough=False,
            return_states=True
        )
        indices, T = _canonical_transform(A, C, indices='best')
        Tinv = np.linalg.inv(T)
        A, B, C, Qw, Swv, X = T@A@Tinv, T@B, C@Tinv, T@Qw@T.T, T@Swv, T@X

        ## Augment with disturbance model
        Bd = np.zeros((n, p))
        Cd = np.eye(p)
        Aaug, Baug, Caug = augment_system(A, B, C, Bd, Cd)

        Sd = estimate_disturbance_cov(X, udat, ydat, A, B, C, Bd, Cd)

        ## Grab final Kalman filter terms
        K, Re = disturbance_filter(A, B, C, Bd, Cd, Sd)

        ## table_info
        table_info[4] = 'see text'
    elif label == 'Unregularized ML':
        ## Make and fit model
        sys = observable_canonical_dmodel(
            get_init_sys(models[guess]['est']),
            mtype='kf',
            noise_type='sqrt',
            **model_settings
        )
        tmp_settings = opt_settings.copy()
        params, stats = sys.fit(udat, ydat, **tmp_settings)

        ## Unpack terms
        Aaug, Baug, Caug, _, _, K, ReL, *_ = (M.full() for M in params)
        A = Aaug[:n, :n]
        B = Baug[:n, :]
        C = Caug[:, :n]
        Bd = Aaug[:n, n:]
        Cd = Caug[:, n:]
        Re = ReL@ReL.T
    elif label == 'Regularized ML 1':
        ## Make and fit model
        sys = observable_canonical_dmodel(
            get_init_sys(models[guess]['est']),
            mtype='kf',
            noise_type='sqrt',
            **model_settings
        )
        rho = 2e-3 # Just unstable
        # rho = 5e-2 # Just too much (eigenvalue @ ~0.14 goes to ~0.19)
        tmp_settings = opt_settings.copy()
        tmp_settings['rho'] = rho
        params, stats = sys.fit(udat, ydat, **tmp_settings)

        ## Unpack terms
        Aaug, Baug, Caug, _, _, K, ReL, *_ = (M.full() for M in params)
        A = Aaug[:n, :n]
        B = Baug[:n, :]
        C = Caug[:, :n]
        Bd = Aaug[:n, n:]
        Cd = Caug[:, n:]
        Re = ReL@ReL.T
    elif label == 'Regularized ML 2':
        ## Make and fit model
        sys = observable_canonical_dmodel(
            get_init_sys(models[guess]['est']),
            mtype='kf',
            noise_type='sqrt',
            **model_settings
        )
        tmp_settings = opt_settings.copy()
        # rho = 2e-3 # Just unstable
        # rho = 3e-3 # Just stable
        rho = 5e-3 # Looks good (but oscillatory)
        # rho = 1e-2 # Still good (no longer oscillatory, eigenvalue @ ~0.14 goes to ~0.15)
        # rho = 3e-2 # Still good (eigenvalue @ ~0.14 goes to ~0.16)
        # rho = 5e-2 # Just too much (eigenvalue @ ~0.14 goes to ~0.19)
        # rho = 7e-2 # Just too much (eigenvalue @ ~0.14 goes to ~0.21)
        # rho = 1e-1 # Too much (eigenvalue @ ~0.14 goes to ~0.21)
        # rho = 2e-1 # Too much (eigenvalue @ ~0.14 goes to ~0.26)
        # rho = 3e-1 # Too much (eigenvalue @ ~0.14 goes to ~0.30)
        # rho = 5e-1 # Far too much (eigenvalues look like that of guess)
        tmp_settings['rho'] = rho
        params, stats = sys.fit(udat, ydat, **tmp_settings)

        ## Unpack terms
        Aaug, Baug, Caug, _, _, K, ReL, *_ = (M.full() for M in params)
        A = Aaug[:n, :n]
        B = Baug[:n, :]
        C = Caug[:, :n]
        Bd = Aaug[:n, n:]
        Cd = Caug[:, n:]
        Re = ReL@ReL.T
    elif label == r'Constrained ML 1':
        ## Make and fit model
        sys = observable_canonical_dmodel(
            get_init_sys(models[guess]['est']),
            mtype='kf',
            noise_type='sqrt',
            **model_settings
        )
        max_eig = 0.998
        alpha = 0.3
        tmp_settings = cons_settings.copy()
        sys.add_eigenvalue_constraint(delta=1-max_eig,
                                      epsilon=tmp_settings['epsilon'],
                                      beta=tmp_settings['beta'], method='lmi')
        sys.add_eigenvalue_constraint(delta=alpha,
                                      epsilon=tmp_settings['epsilon'],
                                      beta=tmp_settings['beta'], method='lmi',
                                      cons_type='continuity')
        params, stats = sys.fit(udat, ydat, **tmp_settings)

        ## Unpack terms
        Aaug, Baug, Caug, _, _, K, ReL, *_ = (M.full() for M in params)
        A = Aaug[:n, :n]
        B = Baug[:n, :]
        C = Caug[:, :n]
        Bd = Aaug[:n, n:]
        Cd = Caug[:, n:]
        Re = ReL@ReL.T
    elif label == r'Constrained ML 2':
        ## Make and fit model
        sys = observable_canonical_dmodel(
            get_init_sys(models[guess]['est']),
            mtype='kf',
            noise_type='sqrt',
            **model_settings
        )
        max_eig = 0.999
        alpha = 0.3
        tmp_settings = cons_settings.copy()
        sys.add_eigenvalue_constraint(delta=1-max_eig,
                                      epsilon=tmp_settings['epsilon'],
                                      beta=tmp_settings['beta'], method='lmi')
        sys.add_eigenvalue_constraint(delta=alpha,
                                      epsilon=tmp_settings['epsilon'],
                                      beta=tmp_settings['beta'], method='lmi',
                                      cons_type='continuity')
        params, stats = sys.fit(udat, ydat, **tmp_settings)

        ## Unpack terms
        Aaug, Baug, Caug, _, _, K, ReL, *_ = (M.full() for M in params)
        A = Aaug[:n, :n]
        B = Baug[:n, :]
        C = Caug[:, :n]
        Bd = Aaug[:n, n:]
        Cd = Caug[:, n:]
        Re = ReL@ReL.T
    elif label == r'Constrained ML 3':
        ## Make and fit model
        sys = observable_canonical_dmodel(
            get_init_sys(models[guess]['est']),
            mtype='kf',
            noise_type='sqrt',
            **model_settings
        )
        ## Trying to push a filter eigenvalue negative
        max_eig = 0.996
        tmp_settings = cons_settings.copy()
        tmp_settings['beta'] = 200
        sys.add_eigenvalue_constraint(delta=1-max_eig,
                                      epsilon=tmp_settings['epsilon'],
                                      beta=tmp_settings['beta'], method='lmi')
        params, stats = sys.fit(udat, ydat, **tmp_settings)

        ## Unpack terms
        Aaug, Baug, Caug, _, _, K, ReL, *_ = (M.full() for M in params)
        A = Aaug[:n, :n]
        B = Baug[:n, :]
        C = Caug[:, :n]
        Bd = Aaug[:n, n:]
        Cd = Caug[:, n:]
        Re = ReL@ReL.T
    elif label == r'Reg. \& Cons. ML (old)':
        ## Make and fit model
        sys = observable_canonical_dmodel(
            get_init_sys(models[guess]['est']),
            mtype='kf',
            noise_type='sqrt',
            **model_settings
        )
        max_eig = 0.998
        alpha = 0.3
        rho = 1e-3
        tmp_settings = cons_settings.copy()
        tmp_settings['rho'] = rho
        sys.add_eigenvalue_constraint(delta=1-max_eig,
                                      epsilon=tmp_settings['epsilon'],
                                      beta=tmp_settings['beta'], method='lmi')
        sys.add_eigenvalue_constraint(delta=alpha,
                                      epsilon=tmp_settings['epsilon'],
                                      beta=tmp_settings['beta'], method='lmi',
                                      cons_type='continuity')
        params, stats = sys.fit(udat, ydat, **tmp_settings)

        ## Unpack terms
        Aaug, Baug, Caug, _, _, K, ReL, *_ = (M.full() for M in params)
        A = Aaug[:n, :n]
        B = Baug[:n, :]
        C = Caug[:, :n]
        Bd = Aaug[:n, n:]
        Cd = Caug[:, n:]
        Re = ReL@ReL.T
    elif label == r'Reg. \& Cons. ML':
        ## Make and fit model
        sys = observable_canonical_dmodel(
            get_init_sys(models[guess]['est']),
            mtype='kf',
            noise_type='sqrt',
            **model_settings
        )
        max_eig = 0.998
        alpha = 0.3
        rho = 1e-3
        tmp_settings = cons_settings.copy()
        # tmp_settings['rho'] = rho
        extrapenalty = rho*cs.sumsqr(sys.theta-sys.theta0)

        ## Eigenvalue constraints
        eig_cons_configs = [
            dict(delta=1-max_eig, epsilon=tmp_settings['epsilon'], beta=tmp_settings['beta'], method='lmi'),
            dict(delta=alpha, epsilon=tmp_settings['epsilon'], beta=tmp_settings['beta'], method='lmi', cons_type='continuity'),
        ]
        for config in eig_cons_configs:
            _, _theta, _theta0, *_ = sys.add_eigenvalue_constraint(**config)
            ntmp = config.get('A', sys.A).shape[0]
            nL1 = int(ntmp*(ntmp+1)/2)
            ## NOTE This is the better way of regularizing
            extrapenalty += rho*cs.sumsqr((_theta-_theta0)[:nL1])
        params, stats = sys.fit(udat, ydat, **tmp_settings, extrapenalty=extrapenalty)

        ## Add rho back into settings
        tmp_settings['rho'] = rho

        ## Unpack terms
        Aaug, Baug, Caug, _, _, K, ReL, *_ = (M.full() for M in params)
        A = Aaug[:n, :n]
        B = Baug[:n, :]
        C = Caug[:, :n]
        Bd = Aaug[:n, n:]
        Cd = Caug[:, n:]
        Re = ReL@ReL.T
    else:
        raise ValueError(f'Model {label} not implemented.')

    ## Take time
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

    ## Are we offset-free?
    print(f'cond(Kd) = {np.linalg.cond(K[n:, :])}')
    print(f'cond(ctrb(Aaug,K)) = {np.linalg.cond(ct.ctrb(Aaug,K))}')
    print(f'cond(obsv(Aaug,Caug)) = {np.linalg.cond(ct.obsv(Aaug,Caug))}')
    T = np.block([[np.eye(A.shape[0]) - A, -Bd], [C, Cd]])
    print(f'cond(T) = {np.linalg.cond(T)}')

    ## Open-loop properties
    print(f'eigs(A)={np.linalg.eigvals(A)}')
    print(f'sprad(A)={np.amax(np.abs(np.linalg.eigvals(A)))}')
    print('u->y steady-state gain:')
    print(C@np.linalg.inv(np.eye(A.shape[0])-A)@B)

    ## Closed-loop properties
    AK = Aaug - K@Caug
    print(f'eigs(AK)={np.linalg.eigvals(AK)}')
    print(f'sprad(AK)={np.amax(np.abs(np.linalg.eigvals(AK)))}')


    print(f"Time to fit {label} model: {t1:.4f}s")

    ## Are we satisfying constraints?
    if max_eig is not None:
        Q = np.eye(AK.shape[0])
        Pd = sp.linalg.solve_discrete_lyapunov(AK/max_eig, Q)
        Pc = sp.linalg.solve_continuous_lyapunov(AK, Q)
        print(f'tr(Pd) = {np.trace(Pd)}')
        print(f'eigs(Pd) = {np.linalg.eigvalsh(Pd)}')
        print(f'tr(Pc) = {np.trace(Pc)}')
        print(f'eigs(Pc) = {np.linalg.eigvalsh(Pc)}')

    ## Save data
    D = np.zeros((p, m))
    est = (A, B, C, D, Delta)
    _, yf, _ = dlsim(est, udat.T)

    AK = Aaug-K@Caug
    BK = np.hstack([Baug-K@D, K])
    DK = np.hstack([D, np.zeros((p, p))])
    est = (AK, BK, Caug, DK, Delta)
    zdat = np.vstack([udat, ydat])
    _, yhat, _ = dlsim(est, zdat.T)

    err = yhat.T - ydat
    ll = 0.5*p*N*np.log(2*np.pi) + 0.5*N*np.log(np.linalg.det(Re)) \
        + 0.5*np.trace(np.linalg.inv(Re) @ err @ err.T)
    print(f'log-likelihood = {ll}')

    models[label] = {
        'est': dict(A=A, B=B, C=C, Bd=Bd, Cd=Cd, K=K, Re=Re),
        'yf': yf.T,
        'yhat': yhat.T,
        't': t1,
    }

## Save the models
with open('tclab_dmle.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=-1)
    pickle.dump(models, handle, protocol=-1)
