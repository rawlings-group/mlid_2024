"""
Fit stochastic linear state-space models from input-output data.

lti
lti.fit(method)

control package helpers:
lti.observer() (helper function for Kalman filter)
lti.regulator(Q,R,M) (helper function for LQR)

mpctools package helpers (I could probably do both with scipy QP solver):
lti.mpc(Q,R,M,ub,lb,etc.) (helper function for FHOCP)
lti.mhe(...) (helper function for MHE)

METHODS:
arx (fitted arx realization)
armax (TODO fitted armax realization?)
hkid (HK-reduced fitted arx realization)
pcaid (states are low-dimensional projection of past data)
nucnormid
  mapid
  n2sid
  other basic nuc-norm methods
subspaceid
  cva/cca
  n4sid
  moesp
  plsid (pls on the extended state-space model)
  n2sid (nuclear norm prior on the extended state-space model)
"""
from collections.abc import Iterable

import casadi as cs
import numpy as np
from numpy.linalg import solve, svd, cholesky, qr
import scipy as sp

from linalg import *
from util import *
from util import _check_UY_data, _check_XUY_data, _nlpsol_options

from regression import multiple_regression
from regression import nonlinear_least_squares
from regression import nonlinear_regression

from arx import arx

def mldivide(A, B):
    return np.linalg.solve(A, B)

def mrdivide(A, B):
    return np.linalg.solve(B.T, A.T).T

def jointmle(X, U, Y, mu=0, nu=0, feedthrough=False, cross_cov=False,
             kalman_filter=False, return_states=False):
    """
    Return...
    """
    n, m, p, N = _check_XUY_data(X, U, Y)

    S = np.vstack([X[:, 1:], Y])
    T = np.vstack([X[:, :-1], U])

    if feedthrough:
        Theta, Sigma, resid = multiple_regression(S, T, mu, nu)
    else:
        AB, Q, r1 = multiple_regression(X[:, 1:], T, mu, nu)
        C,  R, r2 = multiple_regression(Y, X[:, :-1], mu, nu)
        Theta = np.vstack([AB, np.hstack([C, np.zeros((p, m))])])
        Sigma = sp.linalg.block_diag(Q, R)
        resid = np.vstack([r1, r2])
        if cross_cov:
            g = lambda Theta, Sigma: \
                np.hstack([np.zeros((p, n)), np.eye(p)]) @ Theta @ \
                np.vstack([np.zeros((n, m)), np.eye(m)])
            Theta[n:, n:] = 0
            Theta, Sigma, resid = multiple_regression(
                S, T, mu, nu, g=g, Theta0=Theta, Sigma0=Sigma
            )

    A = Theta[:n, :n]
    B = Theta[:n, n:]
    C = Theta[n:, :n]
    D = Theta[n:, n:]
    if kalman_filter:
        if not cross_cov:
            Warning('Ignoring cross_cov option because kalman_filter==True.')
        V, s, _ = svd(Sigma, hermitian=True)
        Re = V[n:, :p] @ np.diag(s[:p]) @ V[n:, :p].T
        K = solve(V[n:, :p].T, V[:n, :p].T).T
        Sigma = V[:, :p] @ np.diag(s[:p]) @ V[:, :p].T
        params = (A, B, C, D, K, Re)
    else:
        Q = Sigma[:n, :n]
        R = Sigma[n:, n:]
        if not cross_cov:
            Sigma[:n, n:] = 0
        S = Sigma[:n, n:]
        params = (A, B, C, D, Q, R, S)

    if return_states:
        params += (X,)

    return params, Theta, Sigma, resid


def hokalman(Hfp, n, p, m, nfuture, npast):
    """Return the `nfuture,npast`-balanced realization, `A,B,C`, from the block
    Hankel matrix `Hfp`.
    """
    _check_block_hankel(Hfp, n, p, m, nfuture, npast)

    U, s, Vh = svd(Hfp[:, :-m])
    A = np.diag(s[:n]**(-1/2)) @ U[:, :n].T @ Hfp[:, m:] @ \
        Vh[:n, :].T @ np.diag(s[:n]**(-1/2))
    B = np.diag(s[:n]**(1/2)) @ Vh[:n, :m]
    C = U[:p, :n] @ np.diag(s[:n]**(1/2))

    return A, B, C


def subspaceid(Y, U, npast, nfuture=None, n=None, rho=0, mu=0,
               nu=0, feedthrough=False, cross_cov=False, kalman_filter=False,
               verbosity=0, return_states=False, preestimate=True,
               weights='cca'):
    # initialize
    m, p, N = _check_UY_data(U, Y)
    nz = m + p
    if nfuture is None:
        nfuture = npast

    Z = np.vstack([U, Y])

    # Y correction step
    if preestimate:
        narx = max(npast, nfuture-1)
        # Zp = [[z(0), ..., z(N-narx-1)], ..., [z(narx-1), ..., z(N-2)]]
        Zp = hankel(Z[:N-1], narx, N-narx, dtype=np)
        if feedthrough:
            # Zp = [[zp(narx), ..., zp(N-1)], [u(narx), ..., u(N-1)]]
            Zp = np.vstack([Zp, U[:, narx:]])
        # solve [y(narx) ... y(N-1)] = Ghat@Zp
        Ghat = mrdivide(Y[:, narx:]@Zp.T, Zp@Zp.T+rho*np.eye(Zp.shape[0]))
        # Pad Ghat with some zeros, Ghat = [G(narx) ... G(0)]
        Ghat = np.hstack([Ghat, np.zeros((p, p if feedthrough else m+p))])
        # Ghat = [G(f-1) ... G(0)]
        Ghat = Ghat[:, -nfuture*(m+p):]
        # Ghat = [ G(0)            ]
        #        [ ...    ...      ]
        #        [ G(f-1) ... G(0) ]
        Ghat = toeplitz(Ghat, nfuture, isreversed=True, dtype=np)

        # Zp = [ z(0)   ... z(N-f-p) ]
        #      [ ...        ...      ]
        #      [ z(p-1) ... z(N-f-1) ]
        Zp = hankel(Z[:N-nfuture], npast, N-nfuture-npast+1, dtype=np)
        # Zf = [ z(p)     ... z(N-f) ]
        #      [ ...          ...    ]
        #      [ z(p+f-1) ... z(N-1) ]
        Zf = hankel(Z[:, npast:], nfuture, N-nfuture-npast+1, dtype=np)
        # Yf = [ y(p)     ... y(N-f) ]
        #      [ ...          ...    ]
        #      [ y(p+f-1) ... y(N-1) ]
        Yf = hankel(Y[:, npast:], nfuture, N-nfuture-npast+1, dtype=np)

        # preestimate and remove effect of ARX parameters
        Yf = Yf - Ghat@Zf
    else:
        # Zp = [ z(0)   ... z(N-f-p) ]
        #      [ ...        ...      ]
        #      [ z(p-1) ... z(N-f-1) ]
        Zp = hankel(Z[:, :N-nfuture], npast, N-nfuture-npast+1, dtype=np)
        # Uf = [ u(p)     ... u(N-f) ]
        #      [ ...          ...    ]
        #      [ u(p+f-1) ... u(N-1) ]
        Uf = hankel(Z[:, npast:], nfuture, N-nfuture-npast+1, dtype=np)
        # Yf = [ y(p)     ... y(N-f) ]
        #      [ ...          ...    ]
        #      [ y(p+f-1) ... y(N-1) ]
        Yf = hankel(Y[:, npast:], nfuture, N-nfuture-npast+1, dtype=np)
        # compute nullspace of Uf
        null_Uf = sp.linalg.null_space(Uf)
        # project Yf and Zp onto the nullspace of Uf
        Yf = Yf@null_Uf
        Zp = Zp@null_Uf

    if weights.lower() in ['cca', 'cva']:
        W1 = sp.linalg.sqrtm(Yf@Yf.T)
        W2 = sp.linalg.sqrtm(Zp@Zp.T)
    elif weights.lower() == 'n4sid':
        W1 = np.eye(Yf.shape[0])
        W2 = np.eye(Zp.shape[0])
    else:
    # elif weights.lower() == 'moesp':
    #     W1 = np.eye(Yf.shape[0])
    #     W2 = np.eye(Zp.shape[0])
        raise ValueError(f'Weights {weights} not yet implemented.')

    # gSVD step
    _, s, Vh = svd(mrdivide(mldivide(W1, Yf@Zp.T), W2), full_matrices=False)

    try:
        n = int(n)
    except ValueError:
        nmax = len(s)
        Nd = Zp.shape[1]
        params = np.arange(1, nmax+1)*(2*p+m) + p*m
        sigma_sq = s*s
        sigma_sq_rem = np.cumsum(sigma_sq[::-1])[::-1]
        penalty = np.log(Nd)*params/Nd
        if n is None:
            n = 'SVC'
            print('`n` set to `None`, using SVC criterion by default.')
        if n.lower() == 'svc':
            SVC = sigma_sq + penalty
            n = np.argmin(SVC)
            print(f"Selected n={n} via SVC.")
        elif n.lower() == 'nic':
            NIC = sigma_sq_rem + penalty
            n = np.argmin(NIC)
            print(f"Selected n={n} via NIC.")
        else:
            raise ValueError(f'Unknown order criterion {n}.')

    # Form states
    # Zp = [ z(0)   ... z(N-p) ]
    #      [ ...        ...    ]
    #      [ z(p-1) ... z(N-1) ]
    Zp = hankel(Z, npast, N-npast+1, dtype=np)
    # Zp = np.vstack([Z[:, j:N-npast+j+1] for j in range(npast)])
    # X = [x(p) ... x(N)]
    X = (np.diag(s[:n]**(1/2))@Vh[:n, :]@W2)@Zp

    params, _, _, _ = jointmle(X, U[:, npast:], Y[:, npast:], mu, nu,
                               feedthrough, cross_cov, kalman_filter)
    if return_states:
        params += (X,)

    return params


def nucnormid(Y, U, npast, nfuture=None, n=None, delta=1e-6, rho=1, mu=0, nu=0,
              feedthrough=False, cross_cov=False, kalman_filter=False,
              verbosity=0, return_states=False):
    r"""nucnormid(Y, U, npast[, nfuture, n, rho, feedthrough])

    Return...
    """
    # initialize
    m, p, N = _check_UY_data(U, Y)
    nz = m + p
    if nfuture is None:
        nfuture = npast
    narx = nfuture + npast - 1
    Ns = N-narx
    nTheta = nz*narx
    nTheta += m if feedthrough else 0
    Theta0, Re0, Z, _ = arx(Y, U, narx, narx, feedthrough, weave=True)

    if isinstance(rho, Iterable) or rho != 0:
        if not isinstance(rho, Iterable):
            rho = [rho]

        # define casadi variables
        Theta = cs.SX.sym('Theta', p, nTheta)

        ReLs = cs.SX.sym('ReLs', int(p*(p+1)/2))
        ReL = unvech(ReLs, p)
        Re = ReL@ReL.T

        nL11 = p*nfuture
        L11s = cs.SX.sym('L11s', int(nL11*(nL11+1)/2))
        L11 = unvech(L11s, nL11)

        nL21 = nz*npast
        L21 = cs.SX.sym('L21', nL21, nL11)
        L21s = cs.vec(L21)

        H = hankel(Theta[:, m:] if feedthrough else Theta, nfuture, npast,
                   dtype=cs)

        theta = cs.vertcat(cs.vec(Theta), ReLs, L11s, L21s)

        Re0Ls = vech(cholesky(Re0))
        H0 = hankel(Theta0[:, m:] if feedthrough else Theta0,
                    nfuture, npast, dtype=np)
        qH0T, rH0T = qr(H0.T)
        L110 = rH0T.T
        L110s = vech(L110)
        L210 = qH0T.T
        theta0 = cs.vertcat(cs.vec(Theta0), Re0Ls, L110s, cs.vec(L210))

        nh = nL11*nL21
        g = cs.diag(ReL)
        h = cs.vec(L11 @ L21.T - H)

        r = Y[:, narx:] - Theta @ Z
        invReL = cs.inv(ReL)
        for rhoi in rho:
            print(f"Solving nuclear-norm regularized ID problem with "
                  f"rho={rhoi:.2e}...")
            loglikelihood = 2*Ns*cs.sum1(cs.log(cs.diag(ReL))) \
                + cs.sumsqr(invReL@r) \
                + rhoi*(cs.sumsqr(L11s) + cs.sumsqr(L21s)) \
                + delta*cs.sumsqr(theta)
            nlp = {
                'x': theta,
                'f': loglikelihood,
                'g': cs.vertcat(g, h)
            }
            solver = cs.nlpsol(
                'solver', 'ipopt', nlp, _nlpsol_options(verbosity=verbosity, solver='ma57')
            )
            result = solver(
                x0=theta0,
                lbg=np.hstack([np.zeros((p,)), np.zeros((nh,))]),
                ubg=np.hstack([np.zeros((p,))+1e6, np.zeros((nh,))])
            )

            theta0 = result['x'].full()

            Theta1 = theta0[:p*nTheta].reshape((p, nTheta))
            H1 = hankel(Theta1[:, m:] if feedthrough else Theta1,
                        nfuture, npast, dtype=np)
            U1, s1, Vh1 = svd(H1)
            print(f"eigs={s1}.")
    else:
        Theta1 = Theta0
        Re1 = Re0

    # Hankel matrix SVD
    H1 = hankel(Theta1[:, :-m] if feedthrough else Theta1, nfuture, npast,
                dtype=np)
    U1, s1, Vh1 = svd(H1)

    # Form states
    # Zp = [ z(0)   ... z(N-p) ]
    #      [ ...        ...    ]
    #      [ z(p-1) ... z(N-1) ]
    Z = hankel(np.vstack([U, Y]), npast, N-npast+1, dtype=np)
    # Zp = np.vstack([Z[:, j:N-npast+j+1] for j in range(npast)])
    # X = [x(p) ... x(N)]
    X = np.diag(s1[:n]**(1/2)) @ Vh1[:n, :] @ Z

    params, _, _, _ = jointmle(X, U[:, npast:], Y[:, npast:], mu, nu,
                               feedthrough, cross_cov, kalman_filter)
    if return_states:
        params += (X,)

    return params
