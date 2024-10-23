"""
Fit regression models (for system identification)

mvr
mvrcv
rrr
rrrcv
nnr
nnrcv

nlr
nonlinear_regression

multiple_regression
reduced_rank_regression
nuclear_norm_regression
mutliple_regression_cv
reduced_rank_regression_cv
nuclear_norm_regression_cv
"""
import casadi as cs
import numpy as np
from time import time

from linalg import vech, unvech, logdet

def multiple_regression(Y, X, mu=0, nu=0, g=None, h=None, Theta0=None,
                        Sigma0=None):
    """multiple_regression(Y, X[, mu, nu])

    Find the parameters $(\Theta,\Sigma)$ of the matrix MLE problem:

    .. math:

        \min_{\Theta,\Sigma} & \;\frac{N}{2}\ln|\Sigma| +
        \frac{1}{2}\textrm{tr}(\Sigma^{-1}(Y-\Theta X)(Y-\Theta X)^\top) \\
        \text{s.t.} &\; g(\Theta,\Sigma) = 0, h(\Theta,\Sigma) \geq 0

    Optionally, solve the MAP problem:

    .. math:

        \min_{\Theta,\Sigma} \frac{N+\nu+p+2}{2}\ln|\Sigma| +
        \frac{1}{2}\textrm{tr}(\Sigma^{-1}(Y-\Theta X)(Y-\Theta X)^\top) +
        \frac{\mu}{2}\textrm{tr}(\Sigma^{-1}\Theta\Theta^\top) +
        \frac{\nu}{2}\textrm{tr}(\Sigma^{-1}) \\
        \text{s.t.} &\; g(\Theta,\Sigma) = 0, h(\Theta,\Sigma) \geq 0
    """

    n, N = Y.shape
    p, M = X.shape
    if N != M:
        raise ValueError("Equal x and y samples expected: "
                         f"got Nx={N} and Ny={M}.")
    N0 = N if nu==0 else N + 2*p + 2

    if g is None and h is None:
        Theta = np.linalg.solve(X@X.T + mu*np.eye(p), X@Y.T).T
        resid = Y - Theta@X
        Sigma = (resid@resid.T + mu*Theta@Theta.T + nu*np.eye(n)) / N0
    else:
        Theta = cs.MX.sym('Theta', n, p)
        Ls = cs.MX.sym('Ls', int(n*(n+1)/2))
        theta = cs.vertcat(cs.vec(Theta), Ls)

        L = unvech(Ls, n)
        invL = cs.inv(L)
        Sigma = L@L.T
        f = 2*N0*cs.sum1(cs.log(cs.diag(L))) + cs.norm_fro(invL@(Y-Theta@X))**2
        if nu > 0:
            f += nu*cs.norm_fro(invL)**2
        if mu > 0:
            f += mu*cs.norm_fro(invL @ Theta)**2

        if Theta0 is None or Sigma0 is None:
            Theta0, Sigma0, _ = multiple_regression(Y, X, mu, nu)
        Ls0 = vech(np.linalg.cholesky(Sigma0))
        theta0 = cs.vertcat(cs.vec(Theta0), Ls0)

        cons = cs.diag(L)
        lbg = np.zeros(n) + np.sqrt(1e-8)
        ubg = np.zeros(n) + np.inf
        if g is not None:
            g = cs.vec(g(Theta, Sigma))
            cons = cs.vertcat(cons, g)
            lbg = np.hstack([lbg, np.zeros(g.numel())])
            ubg = np.hstack([ubg, np.zeros(g.numel())])
        if h is not None:
            h = cs.vec(h(Theta, Sigma))
            cons = cs.vertcat(cons, h)
            lbg = np.hstack([lbg, np.zeros(h.numel())])
            ubg = np.hstack([ubg, np.zeros(h.numel())+np.inf])

        nlp = {
            'x': theta,
            'f': f,
            'g': cons
        }
        opts = {
            'ipopt.print_level': 5,
            # 'ipopt.sb': 'yes',
            # 'print_time': 0,
            'ipopt.max_iter': 500
        }
        solver = cs.nlpsol('solver', 'ipopt', nlp, opts)
        result = solver(x0=theta0, lbg=lbg, ubg=ubg)

        theta = result['x'].full()
        Theta = theta[:n*p].reshape((n, p))
        Sigma = unvech(theta[n*p:], p)
        Sigma = Sigma@Sigma.T
        resid = Y - Theta@X

    return Theta, Sigma, resid

def nonlinear_least_squares(y, f, W, theta, theta0=None):
    """nonlinear_least_squares(y, f, W, theta[, theta0])

    Solves the nonlinear regression problem:

    .. math::

        \min_\theta \|y-f(\theta)\|_W^2

    where $f(\cdot)$ is the model and $W$ are the regression weights.

    """
    n = y.shape[0]
    r = y - f
    nlp = {
        'x': theta,
        'f': r.T@W@r
    }
    opts = {
        'ipopt.print_level': 0,
        'ipopt.sb': 'yes',
        'print_time': 0
    }
    solver = cs.nlpsol('solver', 'ipopt', nlp, opts)
    result = solver(x0=theta0)

    return result['x'].full()


def nonlinear_regression(y, f, V, theta, theta0=None, regularizer=0, rho=1e-6, g=None):
    """nonlinear_least_squares(y, f, V, theta[, theta0])

    Returns the maximum likelihood estimator $\hat\theta$ of the model

    .. math::

        y \sim N(f(\theta), V(\theta))

    In other words, this function solves the nonlinear regression problem:

    .. math::

        \min_\theta \|y-f(\theta)\|_{[V(\theta)]^+}^2 \qquad \text{subject to}
        \qquad y - f(\theta) \in \mathcal{R}(V(\theta))

    To avoid expensive computations (i.e. $[V(\theta)]^+$ and $y - f(\theta)
    \in \mathcal{R}(V(\theta))$), we parameterize the constraint as in
    \cite{kuntz:rawlings:2023}:

    .. math::

        \min_{\alpha,\theta} \|\alpha\|_{V(\theta)}^2 \qquad \text{subject to}
        \qquad y = f(\theta) + V\alpha

    where $\alpha$ are equivalent to the Lagrange multipliers up to a sign
    change.
    """
    n = y.shape[0]

    A = cs.SX.sym('A', n, n)
    Q, R = cs.qr(A)
    log_det = cs.trace(cs.log(R))
    F = cs.Function('F', [A], [log_det])

    r = y - f
    f = F(V) + r.T@cs.solve(V, r) + regularizer + rho*theta.T@theta
    nlp = {
        'x': theta,
        'f': f
    }
    if g is not None:
        nlp['g'] = g
    opts = {
        'ipopt.print_level': 0,
        'ipopt.sb': 'yes',
        'print_time': 0
    }
    solver = cs.nlpsol('solver', 'ipopt', nlp, opts)
    if g is None:
        result = solver(x0=theta0)
    else:
        result = solver(x0=theta0, lbg=np.zeros(g.shape), ubg=np.zeros(g.shape)+1e6)

    return result['x'].full()


def reduced_rank_regression(Y, X, n, mu=0, nu=0):
    """reduced_rank_regression(Y, X[, mu, nu])

    Find the parameters $(\Theta,\Sigma)$ of the matrix MLE problem:

    .. math:

        \min_{\textrm{rank}\Theta=n,\Sigma} \frac{N}{2}\ln|\Sigma| +
        \frac{1}{2}\textrm{tr}(\Sigma^{-1}(Y-\Theta X)(Y-\Theta X)^\top)

    Optionally, solve the MAP problem:

    .. math:

        \min_{\textrm{rank}\Theta=n,\Sigma} \frac{N}{2}\ln|\Sigma| +
        \frac{1}{2}\textrm{tr}(\Sigma^{-1}(Y-\Theta X)(Y-\Theta X)^\top) +
        \frac{\mu}{2}\textrm{tr}(\Sigma^{-1}\Theta\Theta^\top) +
        \frac{\nu}{2}\textrm{tr}(\Sigma^{-1})
    """
    pass


def nuclear_norm_regression(Y, X, rho, mu=0, nu=0):
    """nuclear_norm_regression(Y, X[, mu, nu])

    Find the parameters $(\Theta,\Sigma)$ of the matrix MLE problem:

    .. math:

        \min_{\Theta,\Sigma} \frac{N}{2}\ln|\Sigma| +
        \frac{1}{2}\textrm{tr}(\Sigma^{-1}(Y-\Theta X)(Y-\Theta X)^\top) + +
        \rho\|\Theta\|_*

    Optionally, solve the MAP problem:

    .. math:

        \min_{\Theta,\Sigma} \frac{N}{2}\ln|\Sigma| +
        \frac{1}{2}\textrm{tr}(\Sigma^{-1}(Y-\Theta X)(Y-\Theta X)^\top) +
        \rho\|\Theta\|_* +
        \frac{\mu}{2}\textrm{tr}(\Sigma^{-1}\Theta\Theta^\top) +
        \frac{\nu}{2}\textrm{tr}(\Sigma^{-1})

    """
    pass
