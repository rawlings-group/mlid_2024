"""
Fit autoregressive models with exogenous inputs from input-output data.
"""
import numpy as np

from .regression import multiple_regression

from .linalg import *
from .util import _check_UY_data

def delay_coords(Y, U, ny, nu=None, feedthrough=False, weave=False):
    """delay_coords(Y, U, ny[, nu, feedthrough, weave])

    Return the delay coordinates of the given input-output data.

    By default, `nu=ny`, `feedthrough=False`, and `weave=False`. If
    `weave=False`, returns

    .. math::
        Z_p &= \begin{bmatrix} z_p(n) & \ldots & z_p(N) \end{bmatrix} \\
            &= \begin{bmatrix} y(0) & z(1) & \hdots & z(N-n) \\
                               \vdots & \vdots && \vdots \\
                               y(n-1) & y(n) & \ldots & y(N-1) \\
                               u(0) & u(1) & \hdots & u(N-n) \\
                               \vdots & \vdots && \vdots \\
                               u(n-1) & u(n) & \ldots & u(N-1) \end{bmatrix}

    where :math:`n=\max\{n_u,n_y\}`. Otherwise, if `weave=True`, returns

    .. math::
        Z_p &= \begin{bmatrix} z_p(n) & \ldots & z_p(N) \end{bmatrix} \\
            &= \begin{bmatrix} z(0) & z(1) & \hdots & z(N-n) \\
                               z(1) & z(2) & \ldots & z(N-n+1) \\
                               \vdots & \vdots && \vdots \\
                               z(n-1) & z(n) & \ldots & z(N-1) \end{bmatrix}

    where :math:`z=[u^\top,\; y^\top]^\top` and :math:`n=n_u=n_y` (an error is
    thrown if :math:`n_u\neq n_y`).

    If `feedthrough=True`, the current input $u(k)$ is added to the delay
    coordinates, e.g.

    .. math::
        Z \rightarrow \begin{bmatrix} z_p(n) & \ldots & z_p(N-1) \\
            \begin{matrix} u(n) & u(n+1) & \ldots & u(N-1) \end{matrix}
            \end{bmatrix}
    """
    m, p, N = _check_UY_data(U, Y)
    if nu is None:
        nu = ny
    nmax = max(nu, ny)

    if weave:
        if ny != nu:
            raise ValueError(f'Expected `nu=ny` for `weave=True`, got: '
                             f' `nu={nu}`=/=`ny={ny}` instead.')
        Zp = hankel(np.vstack([U, Y]), nmax, N-nmax+1, dtype=np)
    else:
        Zp = np.vstack([hankel(Y[:, nmax-ny:], ny, N-nmax+1),
                        hankel(U[:, nmax-nu:], nu, N-nmax+1)])

    if feedthrough:
        Zp = np.vstack([Zp[:, :-1], U[:, nmax:]])

    return Zp


def arx(Y, U, ny, nu=None, feedthrough=False, weave=False):
    r"""arx(Y, U, ny[, nu, feedthrough, weave])

    If :math:`n_u` is not supplied, it is set to :math:`n_y`.

    .. math::
        y(k) = H_1 y(k-1) + \ldots + H_{n_y} y(k-n_y) + G_0 u(k) + G_1 u(k-1) +
        \ldots + G_{n_u} u(k-n_y) + e(k)

    Returns :math:`(\hat\Theta, \hat\Sigma, Z, \hat R)` where

    .. math::
        \Theta &= \begin{bmatrix} G_n & H_n & \ldots & G_1 & H_1 & G_0
                  \end{bmatrix} \\
        Z &= \begin{bmatrix} z(0) & z(1) & \hdots & z(N-n-1) \\
                             z(1) & z(2) & \ldots & z(N-n) \\
                             \vdots & \vdots && \vdots \\
                             z(n-1) & z(n) & \ldots & z(N-2) \\
                             u(n) & u(n+1) & \ldots & u(N-1) \end{bmatrix}

    if :math:`n=n_u=n_y`, or

    .. math::
        \Theta &= \begin{bmatrix} H_{n_y} & \ldots & H_1 &
                                 G_{n_u} & \ldots & G_1 & G_0 \end{bmatrix} \\
        Z &= \begin{bmatrix} z(0) & z(1) & \hdots & z(N-n-1) \\
                             z(1) & z(2) & \ldots & z(N-n) \\
                             \vdots & \vdots && \vdots \\
                             z(n-1) & z(n) & \ldots & z(N-2) \\
                             u(n) & u(n+1) & \ldots & u(N-1) \end{bmatrix}

    if :math:`n_u\neq n_y`, and :math:`\hat R=Y-\hat\Theta Z` (where :math:`Y`
    is truncated to the appropriate indices).

    If ``feedthrough==False``, then :math:`G_0` is empty and the bottom
    block-row of :math:`Z` is removed.

    Parameters
    ----------
    Y : array-like
    U : array-like
    ny : int
    nu : int, optional
    feedthrough : bool, optional

    Returns
    -------
    Theta : array-like
    Sigma : array-like
    Z : array-like
    resid : array-like

    """
    m, p, N = _check_UY_data(U, Y)
    if nu is None:
        nu = ny
    nmax = max(nu, ny)

    Zp = delay_coords(Y, U, ny, nu=nu, feedthrough=feedthrough, weave=weave)
    if not feedthrough:
        Zp = Zp[:, :-1]

    Theta, Sigma, resid = multiple_regression(Y[:, nmax:], Zp)
    return Theta, Sigma, Zp, resid


class ARX(object):
    r"""ARX(ny[, nu, feedthrough])

    A class for creating, fitting, and simulating autoregressive exogenous input
    models:

    .. math::

        y(k) &= \sum_{i=1}^{n_y} G_i y(k-i) + \sum_{i=1}^{n_u} H_i u(k-i) +
        H_0 u(k) + e(k)
        e(k) &\sim N(0,V)

    where $y(k)$, $u(k)$, and $e(k)$ are the output, input, and error at time
    $k$, respectively.

    Parameters
    ----------
    ny : int
        Length of the output horizon.
    nu : int, optional
        Length of the input horizon. Default is equal to the output horizon
        length (`ny==nu`).
    feedthrough : boolean, optional
        Option for including a nonzero feedthrough term $H_0$. By default,
        `feedthrough=False` and $H_0=0$.

    Attributes
    ----------
    ny : int
        Length of the output horizon.
    nu : int, optional
        Length of the input horizon. Default is equal to the output horizon
        length (`ny==nu`).
    feedthrough : boolean, optional
        Option for including a nonzero feedthrough term $H_0$. By default,
        `feedthrough=False` and $H_0=0$.
    p, m, nz, N, Ns : int
        Number of outputs, inputs, regressors, data points, and regression
        samples, respectively. Populated during data storage and fitting.
    Y, U, Z : 2D arrays, optional
        Optionally stored output, input, and regressor data. Only stored if
        `store_data=True`. The `U` and `Y` arrays must have dimensions `(p, N)`
        and `(m, N)` respectively. The `Z` array has dimension `(nz, Ns)`.
    G, H : 2D arrays
        Model coefficients for AR and X parts, respectively. `G` must have
        dimensions `(p, p*ny)` and `H` must have dimensions `(p, m*nu)` when
    `feedthrough=False` and `(p, m*(nu+1))` otherwise.
    V : 2D array
        Model error covariance. Must have dimensions `(p, p)`.
    """

    def __init__(self, ny, nu=None, feedthrough=False):
        self.ny = ny
        self.nu = (nu if not None else ny)
        self.feedthrough = feedthrough

    def fit(self, Y, U, store_data=True):
        self.store_data = store_data
        self.m, self.p, self.N = _check_UY_data(U, Y)

        Theta, Sigma, Z, R = \
            arx(Y, U, self.ny, self.nu, self.feedthrough)

        self.nz, self.Ns = Z.shape
        if self.store_data:
            self.Z = Z
            self.R = R

        nT = self.ny*self.p
        self.H = Theta[:, :nT]
        self.G = Theta[:, nT:]
        self.V = Sigma

    def arx2ss(self, deterministic=False, states=False):
        """Convert a fitted ARX model to LTI system parameters $(A,B,C,D,S)$."""
        p, m, ny, nu = self.p, self.m, self.ny, self.nu
        n = p*ny + m*nu

        if self.feedthrough:
            D = self.H[:, -m:]
            H = self.H[:, :-m]
        else:
            D = np.zeros([p, m])
            H = self.H

        B = np.vstack([np.zeros([p*(ny-1), m]), D,
                       np.zeros([m*(nu-1), m]), np.eye(m)])
        C = np.hstack([H, self.G])
        A = np.vstack([
            np.hstack([np.zeros([p*(ny-1), p]),
                       np.eye(p*(ny-1)),
                       np.zeros([p*(ny-1), m*nu])]),
            C,
            np.hstack([np.zeros([m*(nu-1), m + p*ny]), np.eye(m*(nu-1))]),
            np.zeros([m, n])
        ])

        params = (A, B, C, D)

        if deterministic == False:
            S = np.zeros([n+p, n+p])
            S[:p, :p] = self.V
            S[:p, -p:] = self.V
            S[-p:, :p] = self.V
            S[-p:, -p:] = self.V
            params += (S,)

        if states == True:
            if not self.store_data:
                raise ValueError("Set `store_data=True` to use `states=True`.")
            X = delay_coords(self.Y, self.U, self.ny, self.nu)
            params += (X,)

        return params

    def set_params(self, G, H, V=None):
        """Set parameters of the ARX model."""
        p,  ng = G.shape
        p0, nh = H.shape
        if p != p0:
            raise ValueError("Expected `G` and `H` to have same dimension 0: "
                             f"got G.shape[0]={p} and H.shape[0]={p0} instead.")
        self.G = G
        self.H = H
        self.V = V


    def sim(self, U, x0=0):
        """Simulate a fitted system. `U` is the input data and `x0` is the
        initial data. Simulates with a zero initial condition by default. To
        properly initialize the ARX model with a nonzero `x0`, it must be of
        dimensions `((p+m)*ish,)` or `((p+m)*ish,1)` where `ish=max(ny, nu)`.
        """

        if np.any((self.G, self.H) is None):
            raise ValueError(
                "Parameters required before simulating model.\n"
                "Run `ARX.fit()` or `ARX.set_params()` to continue."
            )

        n = self.p*self.ny + self.m*self.nu
        if np.all(x0 == 0):
            x0 = np.zeros(n)
        elif len(x0.shape) > 1:
            raise ValueError("Expected x0 to be a 1D array.")
        elif x0.shape[0] != n:
            raise ValueError(f"Expected dimension 0 of x0 to be `n={n}`: "
                             f"got `n={x0.shape[0]}`.")

        m, N = U.shape
        if m != self.m:
            raise ValueError(f"Expected dimension 0 of u to be m={self.m}: "
                             f"got m={m}")

        A, B, C, D = self.arx2ss(deterministic=True)

        Y = np.empty([self.p, N])
        for i in range(N):
            Y[:, i] = C@x0 + D@U[:, i]
            x0 = A@x0 + B@U[:, i]

        return Y
