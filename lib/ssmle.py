"""ssmle.py

Likelihood methods for linear time-invariant state space models.

This file contains several parameterized state space model classes (LSSModel,
KFModel, LGSSModel) with built-in functionality for fitting the model
parameters using maximum likelihood. Also included are helper functions for
constructing useful model types, including linear augmented disturbance models
for offset-free MPC.

Compared to the python-control library, these representations have limited
systems analysis and manipulation ability. The main function of this file is to
provide a framework for constructing CasADi expressions of the likelihood
function and the information matrix.

"""

import casadi as cs
## TODO I don't want to use control unless I inherit their methods/classes
import control as ct
import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp

import itertools

from util import iscasadi, isnumpy, get_dtype, _check_UY_data, elementary, \
    zeros, eye, safeblock, safevertcat, safehorzcat, _nlpsol_options
from linalg import vecs, vech, unvecs, unvech, safeqr, safelq, mldivide, \
    mrdivide, safechol, det, logdet

## TODO:
## How do I call PENNON? Is it feasible to use both PENNON and IPOPT?
##
## How do I deal with constraints? Some will need Q+alpha I and Lii>=0
## constraints and some will need Q(\theta)-I\alpha>=0 constraints.
##
## `Hard-coded` information matrices for analysis and Gauss-Newton
## approximation.
##
## Hook LSSModel up to control library. Inherit from
## ct.iosys.InputOutputSystem?

## TODO class NLSSModel?
## TODO class LGSSModel(LSSModel)
class LSSModel(object):
    """A linear state-space model.

    The ``LSSModel`` class is very different than the
    ``control.statesp.StateSpace`` class. While the parameters are (almost) the
    same, the class can handle parameterized CasADi expressions and do
    prediction error minimization based on the user-provided parameterization.

    So far only discrete-time models have been implemented, and all the
    functions assume a single, uniformly sampled trajectory of data. In the
    future, it should be possible to handle continuous-time models with
    arbitrarily sampled data.

    Parameters
    ----------
    A, B, C : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression (required)
        System, input shaping, and measurement shaping matrices.
    D, x : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression (optional)
        Feedthrough matrix, initial state. Defaults to zero.
    n, m, p : int (optional)
        System dimensions. If not supplied, they are inferred from the first
        dimensions of `(A, B, C)`.
    theta, theta0 : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression
        System parameters and initial guess. Used if any of the matrices
        `(A, B, C, D, x)` were supplied as casadi expressions. Defaults to
        an empty vector. An error is thrown if `(A, B, C, D, x)` are not
        fully defined by `theta`.
    cons : dict
        Constraint dictionary with fields:
        * g: the constraints, a `casadi.SX` or `casadi.MX` expression,
        * lb: the constraint lower bounds, a `numpy` array or `casadi.DM` matrix,
        * ub: the constraint upper bounds, a `numpy` array or `casadi.DM` matrix.
        Defaults to empty vectors/expressions.

    Attributes
    ----------
    A, B, C, D, x : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression
        Matrices defining the model.
    n, m, p : int
        System dimensions.
    theta, theta0 : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression
        System parameters and initial guess.
    cons : dict
        Constraint dictionary. Implemented in the optimizer as
        `cons['lbg']<=cons['g'](theta)<=cons['ubg']`.
    dtype : `numpy` or `casadi` type
        Data type of the system parameters. Can be:
        * `numpy.ndarray` if all the parameters are `numpy` arrays,
        * `casadi.SX` if all the parameters are SX or DM expressions,
        * `casadi.MX` if all the parameters are MX or DM expressions,
        * `casadi.DM` if all the parameters are DM expressions.

    Notes
    -----
    The main advantage of `LSSModel`, over, for example, a
    `control.statesp.StateSpace` instance that we could construct with the
    Python-control package, is the ability to store `casadi` expressions in the
    system matrix attributes. This allows us to easily compute the prediction
    error, compute its derivatives, and fit the parameters of the model. See
    :meth:`likelihood` and :meth:`fit` for more information.

    """
    def __init__(self, *args, **kwargs):
        """LSSModel(A, B, C[, D, x, n, m, p, theta, theta0])

        """
        ## First, we set the parameters.
        self._set_params(*args, **kwargs)
        self._check_dims()

        # Next, for casaditype instances, store the parameter vector, number of
        # parameters, initial guess, and constraints. We do this after all the
        # parameters are set so that we can check if theta fully describes all
        # of them (i.e., no free variables).
        if iscasadi(self.dtype):
            self._set_theta(**kwargs)
            self._set_cons(**kwargs)


    def _set_params(self, A, B, C, **kwargs):
        ## Set the dimensions, defaulting to the height of the required
        ## parameters.
        self.n = kwargs.get('n', A.shape[0])
        self.m = kwargs.get('m', B.shape[1])
        self.p = kwargs.get('p', C.shape[0])

        ## Figure out data type.
        self.dtype = get_dtype(A, B, C, kwargs.get('D', np), kwargs.get('x', np))

        ## Get the optional arguments based on this data type.
        D = kwargs.get('D', zeros((self.p, self.m), dtype=self.dtype))
        x = kwargs.get('x', zeros((self.n, 1), dtype=self.dtype))

        ## Set the parameters
        self.A, self.B, self.C, self.D, self.x = A, B, C, D, x


    def _get_params(self, **kwargs):
        return self.A, self.B, self.C, self.D, self.x


    def _set_theta(self, **kwargs):
        self.theta = kwargs.get('theta', cs.DM(0, 1))
        self.ntheta = self.theta.shape[0]
        self.theta0 = kwargs.get('theta0', zeros((self.ntheta,), dtype=self.dtype))
        self._check_theta()


    def _check_theta(self):
        ## Check that theta can actually describe all the system parameters If
        ## this fails, you probably have a free variable (not in theta). Note
        ## that we have to ignore PL if it is None.
        params = [p for p in self._get_params() if p is not None]
        all_params = cs.Function('all_params', [self.theta], params)

        ## Check that theta and theta0 are the right dimensions
        dims = (self.ntheta, 1)
        for label, M in zip(['theta', 'theta0'], [self.theta, self.theta0]):
            if M.shape != dims:
                raise ValueError(f"Mismatched dimensions: {label} has shape "
                                 f"`{M.shape}`. Expected `{dims}`.")


    ## TODO unused so far.
    def _remove_useless_theta(self):
        """Checks each theta element, in reverse order, and removes those that
        do not influence the system parameters."""
        params = [p for p in self._get_params() if p is not None]
        all_indices = np.arange(self.ntheta)
        for i in reversed(range(self.ntheta)):
            theta_tmp = self.theta[np.delete(all_indices, i)]
            try:
                ## Try to define the system params without self.theta[i]
                params_func = cs.Function('params', [theta_tmp], params)
                ## If it works, remove self.theta[i] and all_indices[i].
                self.theta.remove([i], [])
                self.ntheta -= 1
                if iscasadi(self.theta0):
                    self.theta0.remove([i], [])
                else:
                    self.theta0 = np.concatenate((self.theta0[:i], self.theta0[i+1:]))
            except:
                ## If we throw an error, the  parameter must have been important.
                pass


    def _check_dims(self):
        ## Check A, B, C, D (all 2D arrays)
        for label, M, dims in zip(['A', 'B', 'C', 'D'],
                                  [self.A, self.B, self.C, self.D],
                                  [(self.n, self.n), (self.n, self.m),
                                   (self.p, self.n), (self.p, self.m)]):
            if M.shape != dims:
                raise ValueError(f"Mismatched dimensions: `{label}` has shape "
                                 f"`{M.shape}`. Expected `{dims}`.")

        ## Check x (a 2D array if its casadi, 1D array if its numpy.)
        if iscasadi(self.x) and self.x.shape != (self.n, 1):
            raise ValueError(f"Mismatched dimensions: x has shape "
                             f"`{self.x.shape}`. Expected `({self.n}, 1)`.")
        if isnumpy(self.x) and self.x.shape != (self.n, 1):
            raise ValueError(f"Mismatched dimensions: x has shape "
                             f"`{self.x.shape}`. Expected `({self.n},)`.")

    def _set_cons(self, **kwargs):
        cons_keys = ('g', 'lbg', 'ubg')
        cons = kwargs.get('cons', {key: cs.DM(0, 1) for key in cons_keys})

        extra_keys = set(cons.keys()).difference(set(cons_keys))
        if len(extra_keys) > 0:
            raise ValueError(f"Dictionary mismatch: Extra keys in `cons`: `{extra_keys}`.")

        self.cons = cons


    def fit(self, U, Y, **kwargs):
        f, theta, g, bounds = self.likelihood(U, Y, **kwargs)
        nlp = {'x': theta, 'f': f, 'g': g}
        theta0 = kwargs.get('theta0', self.theta0)
        result, stats = _solve('ipopt', nlp, theta0, bounds=bounds,
                               max_iter=kwargs.get('max_iter', 100),
                               with_jit=kwargs.get('with_jit', True),
                               verbosity=kwargs.get('verbosity', 5))
        self.theta0 = result['x']
        # self.H = cs.Function('Hessian', [theta], [cs.hessian(f, theta)[0]])
        # print(np.linalg.eigvalsh(self.H(self.theta0).full()))
        # print(np.diag(H.full()*U.shape[1]))
        # print(np.sqrt(np.diag(np.linalg.inv(self.H(self.theta0).full()*U.shape[1]))))
        return self.eval_params(**kwargs), stats


    def eval_params(self, theta0=None, **kwargs):
        if theta0 is None:
            theta0 = self.theta0
        params = [p for p in self._get_params(**kwargs) if p is not None]
        all_params = cs.Function('all_params', [self.theta], params)
        return all_params(theta0)


    def likelihood(self, U, Y, rho=0, delta=0, theta0=None, **kwargs):
        """Get the (log-)likelihood.

        Gets the log-likelihood of the model for the given input-output data.
        Also gets constraints and, TODO optionally, the information matrix for
        Fisher's scoring algorithm.

        """
        ## Setting stuff up
        _, _, N = _check_UY_data(U, Y, m=self.m, p=self.p)
        if theta0 is not None:
            # If a new initial guess is provided, replace the old one.
            self._set_theta(theta=self.theta, theta0=theta0)
        extrapenalty = kwargs.get('extrapenalty', None)
        rescale = kwargs.get('rescale', True)
        use_prior = kwargs.get('use_prior', True)

        ## Get the likelihood terms. Namely:
        ## - recursive prediction error generator f,
        ## - the non-recursive parts of the objective f0, and
        ## - the initial (possibly extended) state z0.
        f, f0, z0 = self._get_likelihood_terms(U, Y, **kwargs)

        ## Make an MX version of theta and get the errors.
        theta = cs.MX.sym('theta', self.ntheta)
        F = f.mapaccum('F', N)
        _, errors = F(z0(theta), U, Y, cs.repmat(theta, 1, N))

        ## Get the objective by evaluating f0 at theta and taking half the sum
        ## of the squared prediction errors. Always scale by the number of
        ## samples or we will have trouble converging.
        objective = f0(theta)/N + (0.5/N)*cs.sum2(errors)
        if rho > 0:
            rho = rho if rescale else rho/N
            if use_prior:
                objective += rho*cs.sumsqr(theta-self.theta0)
            else:
                objective += rho*cs.sumsqr(theta)
        if extrapenalty is not None:
            extrapenalty = extrapenalty if rescale else extrapenalty/N
            extrapenalty = cs.Function('extrapenalty', [self.theta], [extrapenalty])
            objective += extrapenalty(theta)

        ## TODO Optionally, get info matrix. Base this on _get_info_matrix from
        ## pem.py
        # hessian, jacobian = self._get_info_matrix(objective*N, theta, errors, rho)

        ## Get the class constraints. Notice these are not custom so we can put
        ## them in the superclass method.
        g = cs.Function('g', [self.theta], [self.cons['g']])
        constraints = g(theta)
        bounds = {'lbg': self.cons['lbg'], 'ubg': self.cons['ubg']}

        return objective, theta, constraints, bounds


    def _get_info_matrix(self, f, theta, err, rho=0):
        """Get the sensitivity (variance) matrix."""
        N = err.shape[1]
        hess = cs.hessian(f, theta)[0]
        grad = cs.jacobian(err, theta)

        ## Break the information matrix into one linear system and take the
        ## outer product.
        info = mldivide(hess, grad)
        info = info@info.T

        ## TODO What to do with rho?

        return info


    def _get_likelihood_terms(self, U, Y, **kwargs):
        """Gets the parts of a prediction error model for `self.likelihood`."""
        ## Setting stuff up
        _, _, N = _check_UY_data(U, Y, m=self.m, p=self.p)

        ## Unpack parameters for convenience
        A, B, C, D, x0, *_ = self._get_params()

        ## Weighting matrices
        W = kwargs.get('W', eye(self.p, dtype=self.dtype))
        W0 = kwargs.get('W0', None)

        ## Put together a SX->MX function that recursively computes successor
        ## states and prediction errors.
        x = cs.SX.sym('x', self.n)
        u = cs.SX.sym('u', self.m)
        y = cs.SX.sym('y', self.p)

        xp = A@x + B@u
        e = y - C@x - D@u

        output = [xp, cs.sumsqr(safechol(W)@e)]
        ## If we are computing the information matrix, we will also need the
        ## errors themselves.
        # if kwargs.get('errors', False):
            # output += [e]
        f = cs.Function('f', [x, u, y, self.theta], output)

        ## Get the (optional) initial state penalty.
        if W0 is not None:
            f0 = cs.sumsqr(np.linalg.cholesky(W0)@x)
        else:
            f0 = 0
        f0 = cs.Function('f0', [self.theta], [f0])

        ## Finally, we need to grab the initial state as a function of the
        ## parameters because they change across each problem.
        x0 = cs.Function('x0', [self.theta], [x0])

        return f, f0, x0


    def transform(self, T, **kwargs):
        """Apply the similarity transform `T` to the given system. Takes and
        ignores keyword arguments for lower-level compatibility."""
        self.A = T@mrdivide(self.A, T)
        self.B = T@self.B
        self.C = mrdivide(self.C, T)
        self.x = T@self.x


    def add_eigenvalue_constraint(self, **kwargs):
        _cons, _theta, _theta0, _bounds = self._get_eigenvalue_constraint(**kwargs)
        self.theta = cs.veccat(self.theta, _theta)
        self.ntheta += _theta.numel()
        self.theta0 = cs.veccat(self.theta0, _theta0)
        self.cons['g'] = cs.veccat(self.cons['g'], _cons)
        self.cons['lbg'] = cs.veccat(self.cons['lbg'], _bounds['lbg'])
        self.cons['ubg'] = cs.veccat(self.cons['ubg'], _bounds['ubg'])
        return _cons, _theta, _theta0, _bounds


    def _get_eigenvalue_constraint(self, A=None, cons_type='stability',
                                   **kwargs):
        if A is None or A == 'open loop':
            A = self.A
        if cons_type == 'stability':
            return self.stability_constraint(A, **kwargs)
        elif cons_type == 'continuity':
            return self.continuity_constraint(A, **kwargs)
        elif cons_type == 'conic':
            return self.conic_constraint(A, **kwargs)
        elif cons_type == 'strip':
            return self.strip_constraint(A, **kwargs)
        else:
            raise ValueError(f'Unknown constraint type: `{cons_type}`.')


    def continuity_constraint(self, A, delta=0., method='smooth', **kwargs):
        if method == 'smooth':
            ## Init
            n = A.shape[0]
            epsilon = kwargs.get('epsilon', 1e-6)
            beta = kwargs.get('beta', 1e3)
            Qmin = kwargs.get('Qmin', (1/beta)*np.eye(n))

            ## The Moritz way
            _theta = cs.SX.sym('Ls', int(n*(n+1)/2))
            L = unvech(_theta, n)
            P = L @ L.T

            ## The magic constraints!
            Ashift = A - delta*cs.DM.eye(n)
            _cons = vech(Ashift@P + P@Ashift.T - Qmin)
            _bounds = {'lbg': cs.DM(_cons.numel() + n + 1, 1),
                       'ubg': cs.veccat(cs.DM(_cons.numel(), 1),
                                        cs.DM(n + 1, 1) + cs.inf)}
            _cons = cs.veccat(_cons, beta - cs.trace(P), cs.diag(L) - epsilon)

            ## Initial matrices
            A0 = cs.Function('A', [self.theta], [A])(self.theta0).full()
            P0 = sp.linalg.solve_continuous_lyapunov(A0-delta*np.eye(n), Qmin)
            if np.trace(P0) > beta:
                raise ValueError(f'Initial conditions are stable but produce an '
                                 f'ill-conditioned `P0`: `tr(P0)={np.trace(P0)}`.')
            _theta0 = vech(safechol(P0))

        elif method == 'lmi':
            ## Generating matrices
            M0 = -2*delta*np.eye(1)
            M1 = np.eye(1)
            _cons, _theta, _theta0, _bounds = \
                self._LMI_region_constraint(A, M0, M1, **kwargs)

        return _cons, _theta, _theta0, _bounds


    def stability_constraint(self, A, delta=0., method='smooth', Qmin=None, **kwargs):
        """Make stability constraint of the form :math:$\rho(A)\leq 1-\delta$."""
        n = A.shape[0]
        beta = kwargs.get('beta', 1e3)
        if Qmin is None:
            Qmin = (1/beta)*np.eye(n)
        if method == 'smooth':
            ## Init
            epsilon = kwargs.get('epsilon', 1e-6)

            ## The Moritz way
            _theta = cs.SX.sym('L', int(n*(n+1)/2))
            L = unvech(_theta, n)
            P = L @ L.T

            ## The magic constraints!
            _cons = vech(P - A@P@A.T/(1-delta)**2 - Qmin)
            _bounds = {'lbg': cs.DM(_cons.numel() + n + 1, 1),
                       'ubg': cs.veccat(cs.DM(_cons.numel(), 1),
                                        cs.DM(n + 1, 1) + cs.inf)}
            _cons = cs.veccat(_cons, beta - cs.trace(P), cs.diag(L) - epsilon)

            ## Initial matrices
            A0 = cs.Function('A', [self.theta], [A])(self.theta0).full()
            P0 = sp.linalg.solve_discrete_lyapunov(A0 / (1-delta), Qmin)
            if np.trace(P0) > beta:
                raise ValueError(f'Initial conditions are stable but produce an '
                                 f'ill-conditioned `P0`: `tr(P0)={np.trace(P0)}`.')
            _theta0 = vech(safechol(P0))

        elif method == 'lmi':
            ## Generating matrices
            M0 = (1-delta)*np.eye(2)
            M1 = np.array([[0., 1.], [0., 0.]])
            Qmin = sp.linalg.block_diag(Qmin, Qmin)# np.zeros((n, n)))
            _cons, _theta, _theta0, _bounds = \
                self._LMI_region_constraint(A, M0, M1, **kwargs)

        return _cons, _theta, _theta0, _bounds

    def conic_constraint(self, A, delta=1., x0=0., method=None, **kwargs):
        """Make conic constraint."""
        if method not in [None, 'lmi']:
            raise Warning(f'Ignoring `method={method}` option. Conic constraints are only implemented in LMI form.')
        ## Generating matrices
        M0 = -2*delta*x0*np.eye(2)
        M1 = np.array([[delta, -1.], [1., delta]])
        return self._LMI_region_constraint(A, M0, M1, **kwargs)

    def strip_constraint(self, A, delta=1., method=None, **kwargs):
        """Make horizontal strip constraint."""
        if method not in [None, 'lmi']:
            raise Warning(f'Ignoring `method={method}` option. Horizontal strip constraints are only implemented in LMI form.')
        ## Generating matrices
        M0 = 2*delta*np.eye(2)
        M1 = np.array([[0.,1.], [-1.,0.]])

        return self._LMI_region_constraint(A, M0, M1, **kwargs)

    def _LMI_region_constraint(self, A, M0, M1, epsilon=1e-6, beta=1000,
                               Qmin=None, **kwargs):

        ## Initialize dims and Qmin if not given
        n = A.shape[0]
        m = M0.shape[0]
        if Qmin is None:
            Qmin = (1/beta)*np.eye(m*n)
        # if Pmin is None:
            # Pmin = (1/beta)*np.eye(n)

        ## Make the `slack` variables
        L1s = cs.SX.sym('L1', int(n*(n+1)/2))
        L1 = unvech(L1s, n)
        P = L1 @ L1.T #+ Pmin
        L2s = cs.SX.sym('L2', int(m*n*(m*n+1)/2))
        L2 = unvech(L2s, m*n)
        Q = L2 @ L2.T + Qmin
        _theta = cs.veccat(L1s, L2s)

        ## The magic constraints!
        AP = A@P
        M = cs.kron(M0, P) + cs.kron(M1, AP) + cs.kron(M1.T, AP.T)
        _cons = cs.veccat(vech(M - Q))
        _eigs = cs.veccat(beta - cs.trace(P), cs.diag(L1) - epsilon,
                          cs.diag(L2) - epsilon)
        _bounds = {'lbg': cs.DM(_cons.numel() + _eigs.numel(), 1),
                   'ubg': cs.veccat(cs.DM(_cons.numel(), 1),
                                    cs.DM(_eigs.numel(), 1) + cs.inf)}
        _cons = cs.veccat(_cons, _eigs)

        ## TODO It would be best to find the nearest D-stable \(A\) matrix such
        ## that the constraint holds. For now, we'll just find some feasible
        ## \(P\) and \(Q\) such that the trace of \(P\) is minimized. This will
        ## error if such a \(P\) does not exist.
        A0 = cs.Function('A', [self.theta], [A])(self.theta0).full()
        # P0 = LMI_region_problem(A0, M0, M1, Pmin=Pmin + (epsilon**2)*np.eye(n))
        P0 = LMI_region_problem(A0, M0, M1, Qmin=Qmin + (epsilon**2)*np.eye(m*n))

        if np.trace(P0) > beta:
            message = f'Initial conditions are admissible but produce an ' + \
                f'ill-conditioned `P0`: `tr(P0)={np.trace(P0)}`.'
            raise ValueError(message)

        AP0 = A0@P0
        Q0 = np.kron(M0, P0) + np.kron(M1, AP0) + np.kron(M1.T, AP0.T)
        # _theta0 = cs.veccat(vech(safechol(P0 - Pmin, tol=epsilon)),
                            # vech(safechol(Q0, tol=epsilon)))
        _theta0 = cs.veccat(vech(safechol(P0, tol=epsilon)),
                            vech(safechol(Q0 - Qmin, tol=epsilon)))

        return _cons, _theta, _theta0, _bounds


class KFModel(LSSModel):
    """KFModel(A, B, C[, D, x, K, Re, ...])

    A class for representing Kalman filter models:

    .. math::

          x^+ &= A x + B u + K e \\
            e &= y - C x - D u \sim N(0,R_e)

    where `u` is the input, `y` is the output, `x` is the state, and `e` are the
    innovation errors.

    Parameters
    ----------
    A, B, C : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression (required)
        System, input shaping, and measurement shaping matrices.
    D, x : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression (optional)
        Feedthrough matrix, initial state. Defaults to zero.
    K : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression (optional)
        Filter gain. Defaults to zero. Can be set later using
        `add_filter_params`.
    Re or ReL : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression (optional)
        Innovation error covariance. Can be supplied as a positive definite `Re`
        in standard form, or as a lower triangular Cholesky factor `ReL`.
        Defaults to the identity matrix. Can be set later using
        `add_filter_params`.
    n, m, p : int (optional)
        System dimensions. If not supplied, they are inferred from the first
        dimensions of `(A, B, C)`.
    theta, theta0 : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression
        System parameters and initial guess. Used if any of the matrices
        `(A, B, C, D, x)` were supplied as casadi expressions. Defaults to
        an empty vector. An error is thrown if `(A, B, C, D, x)` are not
        fully defined by `theta`.
    cons : dict
        Constraint dictionary with fields:
        * g: the constraints, a `casadi.SX` or `casadi.MX` expression,
        * lb: the constraint lower bounds, a `numpy` array or `casadi.DM` matrix,
        * ub: the constraint upper bounds, a `numpy` array or `casadi.DM` matrix.
        Defaults to empty vectors/expressions.

    Attributes
    ----------
    A, B, C, D, x, K, ReL : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression
        Matrices defining the model. Note that only the square-root form of the
        covariances are stored.
    n, m, p : int
        System dimensions.
    theta, theta0 : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression
        System parameters and initial guess.
    cons : dict
        Constraint dictionary. Implemented in the optimizer as
        `cons['lbg']<=cons['g'](theta)<=cons['ubg']`.
    dtype : `numpy` or `casadi` type
        Data type of the system parameters. Can be:
        * `numpy.ndarray` if all the parameters are `numpy` arrays,
        * `casadi.SX` if all the parameters are SX or DM expressions,
        * `casadi.MX` if all the parameters are MX or DM expressions,
        * `casadi.DM` if all the parameters are DM expressions.

    Notes
    -----
    The main advantage of `KFModel`, over, for example, a custom
    `control.statesp.StateSpace` instance that we could construct with the
    Python-control package, is the ability to store `casadi` expressions in the
    system matrix attributes. This allows us to easily compute the likelihood
    function, compute its derivatives, and fit the parameters of the model with
    maximum likelihood. See :meth:`likelihood` and :meth:`fit` for more
    information.

    """
    def __init__(self, *args, **kwargs):
        """KFModel(A, B, C[, D, x, K, Re, ReL, n, m, p, theta, theta0])

        Construct a Kalman filter model.

        The default constructor is `KFModel(A, B, C)` which creates a deadbeat
        filter with identity error covariance. To create a more practical
        filter, `K` and `Re` or `ReL` can be supplied as keyword arguments. A
        `LSSModel` to `KFModel` converter has not yet been implemented, so for
        now use the syntax `KFModel(sys.A,sys.B,sys.C,D=sys.D,x=sys.x,...)`.

        """
        ## Nothing new to do here. New stuff is implemented in
        ## _set_params, _get_params, and _check_dims, and gets automatically
        ## called through the superclass constructor.
        super().__init__(*args, **kwargs)


    def _set_params(self, A, B, C, **kwargs):
        ## We inherit the (A, B, C, D, x) setter from the superclass.
        super()._set_params(A, B, C, **kwargs)

        ## We also need K and ReL from kwargs.
        self.K = kwargs.get('K', zeros((self.n, self.p), dtype=self.dtype))
        self.Re, self.ReL = _get_psd_arg('Re', eye(self.p, dtype=self.dtype), **kwargs)


    def _get_params(self, noise_type='sqrt', **kwargs):
        return super()._get_params() + self._get_noise_params(noise_type=noise_type)


    def _get_noise_params(self, noise_type='sqrt'):
        """Gets the noise parameters. For `noise_type='sqrt'`, we use
        `(K,ReL)`, and for `noise_type='posdef`, we use `(K,Re)`. Notice
        these are constructed automatically when `self._set_params(...)` is
        called.

        """
        if noise_type == 'sqrt':
            return self.K, self.ReL
        elif noise_type == 'posdef':
            return self.K, self.Re
        else:
            raise TypeError(f"Unknown noise type: {noise_type}")


    def _check_dims(self):
        ## Check A, B, C, D, x through the superclass method.
        super()._check_dims()

        ## Check K, ReL
        for label, M, dims in zip(['K', 'ReL'],
                                  [self.K, self.ReL],
                                  [(self.n, self.p), (self.p, self.p)]):
            if M is not None and M.shape != dims:
                raise ValueError(f"Mismatched dimensions: `{label}` has shape "
                                 f"`{M.shape}`. Expected `{dims}`.")
            ## TODO how do we safely check that an array is lower triangular?
            # if label != 'K' and (not M.sparsity().is_tril):
            #     raise ValueError(f"Mismatched sparsity: `{label}` is not lower triangular.")


    def _get_likelihood_terms(self, U, Y, **kwargs):
        """Gets the parts of a prediction error model for `self.likelihood`."""
        ## Setting stuff up
        _, _, N = _check_UY_data(U, Y, m=self.m, p=self.p)

        ## Unpack parameters for convenience
        A, B, C, D, x0, K, ReL = self._get_params(noise_type='sqrt')

        ## Put together a SX->MX function that recursively computes successor
        ## states and prediction errors.
        x = cs.SX.sym('x', self.n)
        u = cs.SX.sym('u', self.m)
        y = cs.SX.sym('y', self.p)

        e = y - C@x - D@u
        xp = A@x + B@u + K@e

        output = [xp, cs.sumsqr(mldivide(ReL.T, e))]
        ## If we are computing the information matrix, we will also need the
        ## errors themselves.
        # if kwargs.get('errors', False):
            # output += [e]
        f = cs.Function('f', [x, u, y, self.theta], output)

        ## Get the terms outside the sum, including the (optional) initial
        ## state penalty.
        f0 = N*cs.sum1(cs.log(cs.diag(ReL)))
        f0 = cs.Function('f0', [self.theta], [f0])

        ## Finally, we need to grab the initial state as a function of the
        ## parameters because they change across each problem.
        x0 = cs.Function('x0', [self.theta], [x0])

        ## Optionally add some PEM
        rho_g = kwargs.get('rho_g', 0)
        if rho_g > 0:
            print(f'Adding gain regularizer with rho_g={rho_g}.')
            x_PEM = cs.SX.sym('x_PEM', self.n)
            f_PEM, f0_PEM, x0_PEM = super()._get_likelihood_terms(U, Y, **kwargs)
            xp, err = f(x, u, y, self.theta)
            xp_PEM, err_PEM = f_PEM(x_PEM, u, y, self.theta)
            f = cs.Function('f', [cs.veccat(x, x_PEM), u, y, self.theta], [cs.veccat(xp, xp_PEM), err + 1e-2*err_PEM])
            f0 = cs.Function('f0', [self.theta], [f0(self.theta) + f0_PEM(self.theta)])
            x0 = cs.Function('x0', [self.theta], [cs.veccat(x0(self.theta), x0_PEM(self.theta))])

        return f, f0, x0


    def _get_eigenvalue_constraint(self, A=None, cons_type='stability',
                                   **kwargs):
        if A is None or A == 'closed loop':
            A = self.A - self.K @ self.C
        elif A == 'open loop':
            A = self.A
        if cons_type == 'stability':
            return self.stability_constraint(A, **kwargs)
        elif cons_type == 'continuity':
            return self.continuity_constraint(A, **kwargs)
        elif cons_type == 'conic':
            return self.conic_constraint(A, **kwargs)
        elif cons_type == 'strip':
            return self.strip_constraint(A, **kwargs)
        else:
            raise ValueError(f'Unknown constraint type: `{cons_type}`.')

    def transform(self, T, **kwargs):
        """Apply the similarity transform `T` to the given system."""
        ## Start with (A,B,C,x) using the superclass method.
        super().transform(T)

        ## Wrap up with K.
        self.K = T@self.K


def _get_psd_arg(label, default=None, **kwargs):
    """Gets a positive semidefinite argument from the kwargs."""
    if label in kwargs and label + 'L' in kwargs:
        raise Warning(f"Both `{label}` and `{label}L` supplied as kwargs. "
                      "Defaulting to `{label}L`.")

    if label + 'L' in kwargs:
        ML = kwargs[label + 'L']
        M = ML@ML.T
    elif label in kwargs:
        M = kwargs[label]
        ML = safechol(M)
    else:
        if default is None:
            M = None
            ML = None
        else:
            M = default
            ML = safechol(M)

    return M, ML


class MinDetModel(KFModel):
    """MinDetModel(A, B, C[, D, x, K, ...])

    A class for representing Kalman filter models:

    .. math::

          x^+ &= A x + B u + K e \\
            e &= y - C x - D u \sim N(0,R_e)

    where `u` is the input, `y` is the output, `x` is the state, and `e` are
    the innovation errors.

    The main difference between this class and `KFModel` is that the optimizer
    minimizes the determinant of the predicted sample covariance, rather than
    the negative log-likelihood. The parameters, attributes, and methods are
    the same as `KFModel`, but the `Re` parameter is ignored, and a different
    objective minimized.

    """
    def __init__(self, *args, **kwargs):
        ## Nothing new to do here. New stuff is implemented in _set_params and
        ## gets automatically called through the superclass constructor.
        super().__init__(*args, **kwargs)


    def _set_params(self, A, B, C, **kwargs):
        ## We inherit the (A, B, C, D, x) setter from the superclass.
        super()._set_params(A, B, C, **kwargs)

        ## Reset Re and ReL
        self.Re, self.ReL = _get_psd_arg('Re', eye(self.p, dtype=self.dtype))


    def fit(self, U, Y, **kwargs):
        f, theta, g, bounds, Re = self.objective(U, Y, **kwargs)
        nlp = {'x': theta, 'f': f, 'g': g}
        theta0 = kwargs.get('theta0', self.theta0)
        result = _solve('ipopt', nlp, theta0, bounds=bounds,
                        max_iter=kwargs.get('max_iter', 100),
                        with_jit=kwargs.get('with_jit', True),
                        verbosity=kwargs.get('verbosity', 5))
        self.theta0 = result['x']
        self.Re = Re(self.theta0)
        self.ReL = safechol(self.Re)
        return self.eval_params(**kwargs)

        ## The hessian in this context might not work correctly; have to try if
        ## the parameter elimination worked
        # f, theta, g, bounds = super().likelihood(U, Y, **kwargs)
        # nlp = {'x': theta, 'f': f, 'g': g}
        # theta0 = kwargs.get('theta0', self.theta0)
        # result = _solve('ipopt', nlp, theta0, bounds=bounds,
        #                 max_iter=kwargs.get('max_iter', 100),
        #                 with_jit=kwargs.get('with_jit', True),
        #                 verbosity=kwargs.get('verbosity', 5))
        # self.theta0 = result['x']
        # self.H = cs.Function('Hessian', [theta], [cs.hessian(f, theta)[0]])
        # print(np.linalg.eigvalsh(self.H(self.theta0).full()))
        # # print(np.diag(H.full()*U.shape[1]))
        # print(np.sqrt(np.diag(np.linalg.inv(self.H(self.theta0).full()*U.shape[1]))))
        # return self.eval_params(**kwargs)


    def _get_objective_terms(self, U, Y, **kwargs):
        """Gets the parts of a prediction error model for `self.likelihood`."""
        ## Setting stuff up
        _, _, N = _check_UY_data(U, Y, m=self.m, p=self.p)

        ## Unpack parameters for convenience
        A, B, C, D, x0, K, *_ = self._get_params()

        ## Put together a SX->MX function that recursively computes successor
        ## states and prediction errors.
        x = cs.SX.sym('x', self.n)
        u = cs.SX.sym('u', self.m)
        y = cs.SX.sym('y', self.p)

        e = y - C@x - D@u
        xp = A@x + B@u + K@e

        output = [xp, e]
        f = cs.Function('f', [x, u, y, self.theta], output)

        ## Finally, we need to grab the initial state as a function of the
        ## parameters because they change across each problem.
        x0 = cs.Function('x0', [self.theta], [x0])

        return f, x0


    def objective(self, U, Y, rho=0, delta=0, theta0=None, **kwargs):
        """Get the determinant of the sample covariance matrix.

        """
        ## Setting stuff up
        _, _, N = _check_UY_data(U, Y, m=self.m, p=self.p)
        if theta0 is not None:
            # If a new initial guess is provided, replace the old one.
            self._set_theta(theta=self.theta, theta0=theta0)
        extrapenalty = kwargs.get('extrapenalty', None)
        rescale = kwargs.get('rescale', True)
        use_prior = kwargs.get('use_prior', True)

        ## Get the likelihood terms. Namely:
        ## - recursive prediction error generator f,
        ## - the non-recursive parts of the objective f0, and
        ## - the initial (possibly extended) state z0.
        f, z0 = self._get_objective_terms(U, Y, **kwargs)

        ## Make an MX version of theta and get the errors.
        theta = cs.MX.sym('theta', self.ntheta)
        F = f.mapaccum('F', N)
        _, errors = F(z0(theta), U, Y, cs.repmat(theta, 1, N))

        ## Turn Re into the sample covariance
        Re = errors@errors.T/N

        ## Get the objective by evaluating f0 at theta and taking half the sum
        ## of the squared prediction errors. Always scale by the number of
        ## samples or we will have trouble converging.
        objective = logdet(self.p)(Re)
        if rho > 0:
            rho = rho if rescale else rho/N
            if use_prior:
                objective += 2*rho*cs.sumsqr(theta-self.theta0)
            else:
                objective += 2*rho*cs.sumsqr(theta)
        if extrapenalty is not None:
            extrapenalty = extrapenalty if rescale else extrapenalty/N
            extrapenalty = cs.Function('extrapenalty', [self.theta], [extrapenalty])
            objective += 2*extrapenalty(theta)

        ## TODO Optionally, get info matrix. Base this on _get_info_matrix from
        ## pem.py
        # hessian, jacobian = self._get_info_matrix(objective*N, theta, errors, rho)

        ## Get the class constraints. Notice these are not custom so we can put
        ## them in the superclass method.
        g = cs.Function('g', [self.theta], [self.cons['g']])
        constraints = g(theta)
        bounds = {'lbg': self.cons['lbg'], 'ubg': self.cons['ubg']}

        ## Turn Re into a function
        Re = cs.Function('Re', [theta], [Re])

        return objective, theta, constraints, bounds, Re


class LGSSModel(LSSModel):
    """LGSSModel(A, B, C[, D, x, noise_type, ..., n, m, p, theta, theta0])

    Parameters
    ----------
    A, B, C : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression (required)
        System, input shaping, and measurement shaping matrices.
    D, x : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression (optional)
        Feedthrough matrix, initial state. Defaults to zero.
    noise_type : string (optional)
        Noise parameterization type. Allowed types are as follows:
        * The `'G'` parameterization uses `Gw`, `Gv`, and optionally `Gwv`.
        * The `'K'` parameterization uses `K`, and either `Re` or `ReL`.
        * The `'S'` parameterization uses `Qw`, `Rv`, and optionally `Swv` (or infers them from `S`).
        * The `'SL'` parameterization uses `'SL'` (not recommended).
        If not given, inferred from the keyword arguments by first checking each
        parameterization in order for all the required arguments, then checking
        each parameterization for any argument.
    P or PL : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression (optional)
        Initial state covariance. Can be supplied as a positive definite `P` in
        standard form, or as a lower triangular Cholesky factor `PL`. Defaults
        to zero, in which case it is not used. Can be set later using
        `add_noise_params`.
    n, m, p : int (optional)
        System dimensions. If not supplied, they are inferred from the first
        dimensions of `(A, B, C)`.
    theta, theta0 : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression
        System parameters and initial guess. Used if any of the matrices
        `(A, B, C, D, x)` were supplied as casadi expressions. Defaults to
        an empty vector. An error is thrown if `(A, B, C, D, x)` are not
        fully defined by `theta`.
    cons : dict
        Constraint dictionary with fields:
        * g: the constraints, a `casadi.SX` or `casadi.MX` expression,
        * lb: the constraint lower bounds, a `numpy` array or `casadi.DM` matrix,
        * ub: the constraint upper bounds, a `numpy` array or `casadi.DM` matrix.
        Defaults to empty vectors/expressions.

    Attributes
    ----------
    A, B, C, D, x : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression
        Matrices defining the determinstic part of the model.
    noise_type : string
        Noise parameterization type. In all cases, the `'G'` parameters are
        computed and stored. Only in the `'K'` case can the Kalman filter
        parameterization be recovered symbolically.
    n, m, p : int
        System dimensions.
    (Gw, Gv, Gwv) and/or (K, Re, ReL) and/or (Qw, Rv, Swv) and/or SL : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression
        Matrices defining the noise model.
    theta, theta0 : `numpy.ndarray` or `casadi.SX` or `casadi.MX` expression
        System parameters and initial guess.
    cons : dict
        Constraint dictionary. Implemented in the optimizer as
        `cons['lbg']<=cons['g'](theta)<=cons['ubg']`.
    dtype : `numpy` or `casadi` type
        Data type of the system parameters. Can be:
        * `numpy.ndarray` if all the parameters are `numpy` arrays,
        * `casadi.SX` if all the parameters are SX or DM expressions,
        * `casadi.MX` if all the parameters are MX or DM expressions,
        * `casadi.DM` if all the parameters are DM expressions.

    Notes
    -----
    The main advantage of `LGSSModel`, over, for example, a
    `control.statesp.StateSpace` subclass is the ability to store `casadi`
    expressions in the system matrix attributes. This allows us to easily
    compute the likelihood function, compute its derivatives, and fit the
    parameters of the model with maximum likelihood. See :meth:`likelihood` and
    :meth:`fit` for more information.

    """
    def __init__(self, *args, **kwargs):
        """LGSSModel(A, B, C[, D, x, noise_type, ..., n, m, p, theta, theta0])

        Construct a linear Gaussian state-space model.

        The default constructor is `LGSSModel(A, B, C)` which creates a
        deadbeat model with identity noise covariance. More practical noise
        models can be passed as keyword arguments; see `'noise_type'` for more
        information.

        """
        super().__init__(*args, **kwargs)


    def _set_params(self, A, B, C, **kwargs):
        ## We inherit the (A, B, C, D, x) setter from the superclass.
        super()._set_params(A, B, C, **kwargs)

        ## Get (Gw, Gv, Gwv) from kwargs. First, we need to figure out what
        ## parameter set we are going to use.
        noise_type = kwargs.get('noise_type', self._get_noise_type(**kwargs))
        ## Going through each case:
        if noise_type == 'G':
            self.Gw = kwargs.get('Gw', kwargs.get('QwL', eye(self.n, dtype=self.dtype)))
            self.Gwv = kwargs.get('Gwv', zeros((self.n, self.p), dtype=self.dtype))
            self.Gv = kwargs.get('Gv', kwargs.get('RvL', eye(self.p, dtype=self.dtype)))
        elif noise_type == 'K':
            self.K = kwargs.get('K', zeros((self.n, self.p), dtype=self.dtype))
            self.Re, self.ReL = _get_psd_arg('Re', eye(self.p, dtype=self.dtype), **kwargs)
        elif noise_type == 'S':
            if 'S' in kwargs:
                self.Qw = kwargs['S'][:self.n, :self.n]
                self.Swv = kwargs['S'][:self.n, self.n:]
                self.Rv = kwargs['S'][self.n:, :self.n:]
            else:
                self.Qw = kwargs.get('Qw', eye(self.n, dtype=self.dtype))
                self.Swv = kwargs.get('Swv', zeros((self.n, self.p)))
                self.Rv = kwargs.get('Rv', eye(self.p, dtype=self.dtype))
        elif noise_type == 'SL':
            SL = kwargs.get('SL', eye(self.n+self.p, dtype=self.dtype))
        else:
            raise TypeError(f"Unknown noise model type: {noise_type}.")

        self.P, self.PL = _get_psd_arg('P', None, **kwargs)

        self.noise_type = noise_type


    def _check_dims(self):
        ## Check A, B, C, D, x through the superclass method.
        super()._check_dims()

        ## Check other params.
        n, p = self.n, self.p
        other_config = [(label, getattr(self, label), dims) for label, dims in
                        zip(['Gwv', 'K', 'Swv'],
                            [(n, p), (n, p), (n, p)]) if hasattr(self, label)]
        symm_config = [(label, getattr(self, label), dim) for label, dim in
                       zip(['Re', 'S', 'Qw', 'Rv', 'P'],
                           [(p, p), (n + p, n + p), (n, n), (p, p), (n, n)]) if hasattr(self, label)]
        tril_config = [(label, getattr(self, label), dim) for label, dim in
                       zip(['Gw', 'Gv', 'ReL', 'SL', 'PL'],
                           [(n, n), (p, p), (p, p), (n+p, n+p), (n, n)]) if hasattr(self, label)]

        for label, M, dims in other_config + symm_config + tril_config:
            if M is not None and M.shape != dims:
                raise ValueError(f"Mismatched dimensions: `{label}` has shape "
                                 f"`{M.shape}`. Expected `{dims}`.")

        # for label, M, dims in symm_config:
        #     try:
        #         safechol(M)
        #     except:
        #         raise ValueError(f"Mismatched structure: `{label}` is not positive definite.")

        ## TODO how do we safely check that an array is tril?
        # for label, M, dim in tril_config:
        #     if not M.sparsity().is_tril():
        #         raise ValueError(f"Mismatched structure: `{label}` is not lower triangular.")


    def _get_noise_type(self, **kwargs):
        """Infer the noise model type from the kwargs."""
        ## Lists to keep track if which required parameters are available.
        isG = [key in ('Gw', 'Gv') for key in kwargs]
        isGalt = [key in ('QwL', 'RvL') for key in kwargs]
        isK = [key in ('K', 'Re') for key in kwargs]
        isKalt = [key in ('K', 'ReL') for key in kwargs]
        isS = [key in ('Qw', 'Rv') for key in kwargs]

        ## First round, check each in order for having all required parameters.
        if all(isG) or all(isGalt):
            return 'G'
        elif all(isK) or all(isKalt):
            return 'K'
        elif 'S' in kwargs or all(isS):
            return 'S'
        elif 'SL' in kwargs:
            return SL
        ## Next round, check each in order for having any required parameter.
        elif any(isG) or any(isGalt):
            return 'G'
        elif any(isK) or any(isKalt):
            return 'K'
        elif 'S' in kwargs or any(isS):
            return 'S'
        ## Finally, give up and default to 'G'.
        else:
            return 'G'


    def _get_params(self, noise_type=None, **kwargs):
        if noise_type is None:
            noise_type = self.noise_type
        return super()._get_params() + self._get_noise_params(noise_type=noise_type)


    def _get_noise_params(self, noise_type='G'):
        """Gets the noise parameters of a given type, computing and storing
        them if unavailable."""
        ## Unpack dims for convenience
        n, p = self.n, self.p

        ## Go through each target noise type.
        if noise_type == 'G':
            try:
                ## Try to return all the parameters
                return self.Gw, self.Gwv, self.Gv, self.PL
            except AttributeError:
                ## Compute the missing parameters if unavailable.
                if self.noise_type == 'K':
                    self.Gw = zeros((n, n), dtype=self.dtype)
                    self.Gwv = self.K
                    self.Gv = self.ReL
                elif self.noise_type == 'S':
                    L = safechol(safeblock([[self.Rv, self.Swv.T],
                                            [self.Swv, self.Qw]]))
                    L11, L21, L22 = L[:p, :p], L[p:, :p], L[p:, p:]
                    self.Gv = L11
                    self.Gwv = mrdivide(L21, L11)
                    self.Gw = safelq(safehorzcat([L21 - self.Gwv@L11, L22]), mode='l')
                elif self.noise_type == 'SL':
                    ## TODO implement this one. It is a pain because you don't get any
                    ## of the G matrices for free like the 'S' type, so some tricks are
                    ## needed to get Gw.
                    SL = kwargs['SL']
                    Gv = safelq(SL[n:, :], mode='l')
                    raise ValueError("Noise model type `'SL'` not yet implemented.")
                ## Return them
                return self.Gw, self.Gwv, self.Gv, self.PL

        elif noise_type == 'K':
            try:
                ## Try to return all the parameters
                return self.K, self.ReL, self.PL
            except:
                ## Otherwise, raise an error that we haven't implemented this
                ## conversion yet. This is a pain for casaditypes because we
                ## have to solve a DARE symbolically.
                raise TypeError(f"Conversion from `'{self.noise_type}'` to  `'{noise_type}'` not supported.")

        elif noise_type == 'S':
            try:
                ## Try to return all the parameters
                return self.Qw, self.Rv, self.Swv, self.P
            except:
                ## Compute the missing parameters
                if self.noise_type == 'S':
                    ## This shouldn't happen, but just in case.
                    S = self.S
                elif self.noise_type == 'G':
                    SL = safeblock([[self.Gw, self.Gwv@self.Gv],
                                    [zeros((p, n), dtype=self.dtype),
                                     self.Gv]])
                    S = SL@SL.T
                elif self.noise_type == 'K':
                    SL = safevertcat([self.K, eye(p, dtype=self.dtype)]) @ self.ReL
                    S = SL@SL.T
                elif self.noise_type == 'SL':
                    S = self.SL @ self.SL.T
                self.Qw, self.Rv, self.Swv = S[:n, :n], S[n:, n:], S[:n, n:]

                return self.Qw, self.Rv, self.Swv, self.P

        elif noise_type == 'SL':
            try:
                return self.SL, self.PL
            except:
                raise TypeError(f"Conversion from `'{self.noise_type}'` to  `'{noise_type}'` not supported.")
        else:
            raise TypeError(f"Unknown noise model type: {noise_type}.")


    def _get_likelihood_terms(self, U, Y, **kwargs):
        """Gets the parts of a prediction error model for `self.likelihood`."""
        ## Unpack parameters for convenience
        A, B, C, D, x0, Gw, Gwv, Gv, PL0 = self._get_params(noise_type='G')

        ## Put together a SX->MX function that recursively computes successor
        ## states and prediction errors.
        z = cs.SX.sym('z', self.n+int(self.n*(self.n+1)/2))
        x = z[:self.n]
        PL = unvech(z[self.n:], self.n)
        u = cs.SX.sym('u', self.m)
        y = cs.SX.sym('y', self.p)

        ## Sqrt Kalman filter magic:
        ## 1. A, B corrections
        A = A - Gwv@C
        B = B - Gwv@D
        ## 2. ReL and K computation
        ReL = safelq(safehorzcat([C@PL, Gv]), mode='l')
        K = A @ (PL @ (PL.T @ mrdivide(mrdivide(C.T, ReL.T), ReL)))
        ## 3. Filter and predict state
        e = y - C@x - D@u
        xp = A@x + B@u + Gwv@y + K@e
        ## 4. Filter and predict covariance
        PLp = safelq(safehorzcat([(A-K@C)@PL, K@Gv, Gw]), mode='l')
        ## 5. Stack them
        zp = cs.veccat(xp, vech(PLp))

        output = [zp, cs.sumsqr(cs.solve(ReL, e)) + 2*cs.sum1(cs.log(cs.diag(ReL)))]
        ## If we are computing the information matrix, we will also need the
        ## errors and error covariances themselves.
        # if kwargs.get('errors', False):
            # output += [e, vech(ReL)]
        f = cs.Function('f', [z, u, y, self.theta], output)

        ## Get the terms outside the sum.
        f0 = 0.5*cs.sumsqr(cs.mldivide(PL0, x0)) + cs.sum1(cs.log(cs.diag(PL0)))
        f0 = cs.Function('f0', [self.theta], [f0])

        ## Finally, we need to grab the initial state as a function of the
        ## parameters because they change across each problem.
        z0 = cs.Function('z0', [self.theta], [cs.veccat(x0, vech(PL0))])

        return f, f0, z0


    def transform(self, T, L=None, noise_type=None):
        """Apply the similarity transform `T` to the given system."""
        ## Get the current noise type if not specified.
        if noise_type is None:
            noise_type = self.noise_type

        ## Start with (A,B,C,x) using the superclass method.
        super().transform(T)

        ## Do the Kalman filter gain. L does not affect this model.
        if hasattr(self, 'K'):
            self.K = T@self.K

        ## Next, do P and PL.
        if self.P is None:
            pass
        else:
            tmp = T@self.PL
            P = tmp@tmp.T
            if L is not None:
                P -= L
            self.PL = safechol(P, dtype=self.dtype)
            self.P = P

        ## Finally, do the other noise models. These are hard to generalize, so
        ## we do the following:

        ## Grab the 'S' parameterization.
        Qw, Rv, Swv, _ = self._get_noise_params(noise_type='S')

        ## Transform the variables.
        Qw, Swv = T@Qw@T.T, T@Swv
        if L is not None:
            Qw += A@L@A.T - L
            Swv += A@L@C.T
            Rv += C@L@C.T

        ## For safety purposes, delete all the noise parameters, and then
        ## re-set with the desired type.
        for label in ('S', 'Qw', 'Swv', 'Rv', 'Gw', 'Gv', 'Gwv', 'SL', 'Re', 'ReL'):
            if hasattr(self, label):
                delattr(self, label)

        if noise_type == 'K':
            pass
        elif noise_type in ('S', 'G', 'SL'):
            self.Qw, self.Rv, self.Swv = Qw, Rv, Swv
            self.noise_type = 'S'
            self._get_noise_params(noise_type=noise_type)
        else:
            raise TypeError(f"Unknown noise model type: {noise_type}.")


## Realization generation functions. Note that each of these are produce either
## by feeding one helper function into another, or they are produced by feeding
## a helper function into a decorator.
def full_params(init_sys, **kwargs):
    """full_params(init_sys, **kwargs)

    Construct a fully parameterized `LSSModel` (or `KFModel`/`LGSSModel`) using
    the given initial system and keyword argument configuration.

    """
    args = _full_params(init_sys, **kwargs)
    return _finish_params(*args[:-1], **kwargs)


def observable_canonical(init_sys, **kwargs):
    args = _observable_canonical(init_sys, **kwargs)
    return _finish_params(*args[:-1], **kwargs)


def observable_companion(init_sys, **kwargs):
    args = _observable_companion(init_sys, **kwargs)
    return _finish_params(*args[:-1], **kwargs)


def full_disturbance_model(init_sys, **kwargs):
    """full_disturbance_model(init_sys, **kwargs)

    Construct a fully parameterized disturbance model (of type
    `LSSModel`/`KFModel`/`LGSSModel`) using the given initial system and
    keyword argument configuration.

    """
    fn = _full_params
    return _get_disturbance_model(fn)(init_sys, **kwargs)


def observable_canonical_dmodel(init_sys, **kwargs):
    fn = _observable_canonical
    return _get_disturbance_model(fn)(init_sys, **kwargs)


def observable_companion_dmodel(init_sys, **kwargs):
    fn = _observable_companion
    return _get_disturbance_model(fn)(init_sys, **kwargs)


## Realization helper functions:
def _full_params(init_sys, **kwargs):
    """Helper function for constructing a fully parameterized model."""
    ## Unpack dims
    n, m, p = init_sys.n, init_sys.m, init_sys.p

    ## Make and store (A,B,C) and the corresponding initial guess.
    A = cs.SX.sym('A', n, n)
    B = cs.SX.sym('B', n, m)
    C = cs.SX.sym('C', p, n)
    theta = cs.veccat(A, B, C)
    theta0 = cs.veccat(init_sys.A, init_sys.B, init_sys.C)
    params = {'A': A, 'B': B, 'C': C}

    ## Return stuff (incl an identity matrix for T)
    return init_sys, params, theta, theta0, eye(n, dtype=init_sys.dtype)


def _controllable_canonical(init_sys, **kwargs):
    """Helper function for constructing a model in controllable canonical
    form."""
    ## Unpack dims
    n, m, p = init_sys.n, init_sys.m, init_sys.p
    if n < m:
        raise ValueError(f"Controllable canonical form requires n >= m.")

    ## First, transform initial system to the right realization.
    indices, T = _canonical_transform(init_sys.A.T, init_sys.B.T,
                                      kwargs.get('indices', 'first'),
                                      kwargs.get('tol', 1e-8))
    init_sys.transform(T.T)

    ## Next, make and store (A,B,C) and the corresponding initial guess.
    A = cs.vertcat(cs.SX(1, n), cs.SX.eye(n)[:-1, :])
    A[:, indices] = cs.SX.sym('A', n, m)
    B = cs.SX.eye(n)[:, np.append(0, np.array(indices[:-1])+1)]
    C = cs.SX.sym('C', p, n)
    theta = cs.veccat(A[:, indices], C)
    theta0 = cs.veccat(init_sys.A[:, indices], init_sys.C)
    params = {'A': A, 'B': B, 'C': C}

    return init_sys, params, theta, theta0, T



def _observable_canonical(init_sys, **kwargs):
    """Helper function for constructing a model in observable canonical form."""
    ## Unpack dims
    n, m, p = init_sys.n, init_sys.m, init_sys.p
    if n < p:
        raise ValueError(f"Observable canonical form requires n >= p.")

    ## First, transform initial system to the right realization.
    indices, T = _canonical_transform(init_sys.A, init_sys.C,
                                      kwargs.get('indices', 'first'),
                                      kwargs.get('tol', 1e-8))
    init_sys.transform(T)

    ## Next, make and store (A,B,C) and the corresponding initial guess.
    A = cs.vertcat(cs.SX.eye(n)[1:, :], cs.SX(1, n))
    A[indices, :] = cs.SX.sym('A', p, n)
    B = cs.SX.sym('B', n, m)
    C = cs.SX.eye(n)[np.append(0, np.array(indices[:-1])+1), :]
    theta = cs.veccat(A[indices, :], B)
    theta0 = cs.veccat(init_sys.A[indices, :], init_sys.B)
    params = {'A': A, 'B': B, 'C': C}

    return init_sys, params, theta, theta0, T


def _canonical_transform(A, C, indices='first', tol=1e-8):
    """Get the transform and indices for a canonical transform (Denham, 1974).
    By default, finds the first such transformation with condition number below
    1/tol. Otherwise, we can supply the indices directly in the `indices`
    argument or choose the best (lowest condition number) among all (n-1)
    choose (p-1) combinations with `indices='best'`.

    """
    ## First, get the dims and WO
    p, n = C.shape
    WO = ct.obsv(A, C)

    ## Going through each generation case:
    if indices in ('first', 'best'):
        ## Loop through the possible rows of A we can fill.
        T_pairs = []
        Ncombos = sp.special.comb(n-1, p-1)
        print(f'Searching through {Ncombos} combinations for the {indices} index set.')
        if p>1:
            for i, indices in enumerate(itertools.combinations(np.arange(n-1), p-1)):
                ## Always use the last row
                indices += (n-1,)
                ## Get T and condT
                T = WO[_obsvind(indices), :]
                condT = np.linalg.cond(T)
                T_pairs += [(indices, T, condT)]
                ## For 'first' strategy, terminate if we reach tolerance
                if indices == 'first' and condT*tol < 1:
                    break
        else:
            i = 1
            indices = (n-1, )
            T =  WO[_obsvind(indices), :]
            condT = np.linalg.cond(T)
            T_pairs += [(indices, T, condT)]
        ## For 'best' strategy, pick the minimum condition number transform
        if indices == 'best':
            indices, T, condT = min(T_pairs, key=lambda x: x[-1])
        print(f'Found the index set {indices} after searching through {i+1} combinations.')
    elif len(indices) == p and indices[-1] == n-1:
        ## Get a custom index
        T = WO[_obsvind(indices), :]
        condT = np.linalg.cond(T)
    else:
        raise ValueError('Please supply a valid index type for `indices`. Allowed index types are:'
                         '* the strings \'first\' or \'best\' to return the first or best valid indices,'
                         '* array-like object containing n integers, the nth being equal to n-1.')

    ## Check singularity and return
    if condT*tol > 1:
        raise RuntimeError(f"Chosen transformation is singular to precision 1/cond(T)={1/condT}>{tol}.")
    return indices, T


def _obsvind(indices):
    """Get the indices of the observability matrix rows (equiv, controlability
    matrix columns) corresponding to a canonical form given by ind."""
    ## Infer dimensions
    n, p = indices[-1] + 1, len(indices)

    ## Check that the indices are valid
    # if not all(isinstance(i, int) for i in indices):
        # raise ValueError('Indices must be all integers.')
    if any(indices[i] >= indices[i+1] for i in range(p-1)):
        raise ValueError('Indices of A must be strictly ordered.')
    if n < p:
        raise ValueError('There cannot be more indices than states.')

    ## Convert to Aii block sizes
    indices = np.array(indices)
    sizes = indices - np.append(-1, indices[:-1])

    ## Convert to multiindices and then to obsv matrix indices.
    multiindex = [(i,j) for j in range(p) for i in range(sizes[j])]
    obsvindex = [i*p+j for (i,j) in multiindex]

    return obsvindex


def _observable_companion(init_sys, **kwargs):
    """Helper function for constructing a model in companion canonical form."""
    ## Unpack dims
    n, m, p = init_sys.n, init_sys.m, init_sys.p

    ## Starting with the similarity transform to companion form
    T = ct.obsv(init_sys.A, elementary(p, 0, dtype=init_sys.dtype).T @ init_sys.C)
    init_sys.transform(T)

    ## Next, make and store (A,B,C) and the corresponding initial guess.
    A = cs.vertcat(eye(n, dtype=cs.SX)[1:, :], cs.SX.sym('a', 1, n))
    B = cs.SX.sym('B', n, m)
    C = cs.SX.sym('C', p, n)
    C[0, :] = elementary(n, 0, dtype=cs).T
    theta = cs.veccat(A[-1, :], B, C[1:, :])
    theta0 = cs.veccat(init_sys.A[-1, :], init_sys.B, init_sys.C[1:, :])
    params = {'A': A, 'B': B, 'C': C}

    return init_sys, params, theta, theta0, T


## Helper functions and helpers-of-helpers for the realization generators:
def _finish_params(init_sys, params, theta, theta0, **kwargs):
    """Fill out a params from the starting `(A,B,C)` realization, and
    return the resulting model object. Namely, get the `(D,x)` terms and
    (optionally) the noise model, each according to the supplied keyword
    arguments.

    """
    ## Unpacking kwargs
    feedthrough = kwargs.get('feedthrough', False)
    initial_state = kwargs.get('initial_state', False)

    mtype = kwargs.get('mtype', _obj2mtype(init_sys))
    noise_type = kwargs.get('noise_type', None)

    epsilon = kwargs.get('epsilon', 1e-6)

    n, m, p = init_sys.n, init_sys.m, init_sys.p

    ## First, make D and x and add them to theta if necessary
    if feedthrough:
        D = cs.SX.sym('D', p, m)
        theta = cs.veccat(theta, D)
        theta0 = cs.veccat(theta0, init_sys.D)
    else:
        D = init_sys.D
    params['D'] = D

    if initial_state:
        x = cs.SX.sym('x', n, 1)
        theta = cs.veccat(theta, x)
        theta0 = cs.veccat(theta0, init_sys.x)
    else:
        x = cs.SX(n, 1)
    params['x'] = x

    ## Now get the stochastic parts
    if mtype == 'lti':
        eigs = cs.DM(0, 1)
    elif mtype == 'kf':
        theta, theta0, params, eigs = \
            _add_kf_model(init_sys, params, theta, theta0, **kwargs)
    elif mtype == 'slti':
        theta, theta0, params, eigs = \
            _add_noise_model(init_sys, params, theta, theta0, **kwargs)
    else:
        raise TypeError(f'Unknown model type {mtype}.')

    ## Construct the constraints, add the parameters.
    params['cons'] = {'g': eigs, 'lbg': cs.DM(eigs.numel(), 1) + epsilon, 'ubg':
                      cs.DM(eigs.numel(), 1) + cs.inf}
    params['theta'] = theta
    if theta0 is not None:
        params['theta0'] = theta0

    ## Finally, construct and return the system
    return _mtype2obj(mtype)(**params)


def _add_kf_model(init_sys, params, theta, theta0, **kwargs):
    """Add a Kalman filter gain and innovation error covariance to a parameter
    vector, dictionary, and initial guess.

    """
    noise_type = kwargs.get('noise_type', 'K')
    initial_state_cov = kwargs.get('initial_state_cov', False)
    cross_cov = kwargs.get('cross_cov', False)
    epsilon = kwargs.get('epsilon', 1e-6)

    n, m, p = init_sys.n, init_sys.m, init_sys.p

    ## Make and store (K, ReL)
    K = cs.SX.sym('K', n, p)
    ReL = cs.tril(cs.SX.sym('ReL', p, p))
    params['K'] = K
    params['ReL'] = ReL
    theta = cs.veccat(theta, K, vech(ReL))
    eigs = cs.diag(ReL)

    ## Initialization part
    K0, ReL0 = init_sys._get_noise_params(noise_type='sqrt')
    theta0 = cs.veccat(theta0, K0, vech(ReL0))

    return theta, theta0, params, eigs


def _add_noise_model(init_sys, params, theta, theta0, **kwargs):
    """Adds a noise model to a parameter vector, dictionary, and initial
    guess."""
    ## Unpack dims and process kwargs
    n, m, p = init_sys.n, init_sys.m, init_sys.p
    noise_type = kwargs.get('noise_type', 'G')
    initial_state_cov = kwargs.get('initial_state_cov', False)
    cross_cov = kwargs.get('cross_cov', False)
    epsilon = kwargs.get('epsilon', 1e-6)

    if noise_type in ('sqrt', 'full'):
        noise_type = 'K'

    ## Go through each target noise type
    if noise_type == 'G':
        Gw = cs.tril(cs.SX.sym('Gw', n, n))
        Gv = cs.tril(cs.SX.sym('Gv', p, p))
        theta = cs.veccat(theta, vech(Gw), vech(Gv))
        if cross_cov:
            Gwv = cs.SX.sym('Gwv', n, p)
            theta = cs.veccat(theta, Gwv)
        else:
            Gwv = cs.SX(n, p)
        params['Gw'] = Gw
        params['Gwv'] = Gwv
        params['Gv'] = Gv
        eigs = cs.veccat(cs.diag(Gw), cs.diag(Gv))
    elif noise_type == 'K':
        K = cs.SX.sym('K', n, p)
        ReL = cs.tril(cs.SX.sym('ReL', p, p))
        params['K'] = K
        params['ReL'] = ReL
        theta = cs.veccat(theta, K, vech(ReL))
        eigs = cs.diag(ReL)
    elif noise_type in ('S', 'SL', 'posdef'):
        raise TypeError(f'Noise model type {noise_type} not yet implemented.')
    else:
        raise TypeError(f'Unknown noise model type {noise_type}.')

    if initial_state_cov:
        PL = cs.tril(cs.SX.sym('PL', n, n))
        theta = cs.veccat(theta, vech(PL))
        params['PL'] = PL
        eigs = cs.veccat(eigs, cs.diag(PL))
    else:
        PL = epsilon*cs.SX.eye(n)
        params['PL'] = PL

    ## Extract the relevant parameters from the initial system.
    noise_params = init_sys._get_noise_params(noise_type=noise_type)
    if noise_type == 'G':
        Gw, Gwv, Gv, PL = noise_params
        theta0 = cs.veccat(theta0, vech(Gw), vech(Gv))
        if cross_cov:
            theta0 = cs.veccat(theta0, Gwv)
    elif noise_type in ('K', 'sqrt'):
        K, ReL, PL = noise_params
        theta0 = cs.veccat(theta0, K, vech(ReL))
    elif noise_type in ('S', 'SL', 'posdef'):
        raise TypeError(f'Noise model type {noise_type} not yet implemented.')
    else:
        raise TypeError(f'Unknown noise model type {noise_type}.')

    if initial_state_cov:
        theta0 = cs.veccat(theta0, vech(PL))

    return theta, theta0, params, eigs


def _obj2mtype(sys):
    if isinstance(sys, LGSSModel):
        return 'slti'
    elif isinstance(sys, KFModel):
        return 'kf'
    elif isinstance(sys, LSSModel):
        return 'lti'
    else:
        raise TypeError(f'Unknown model type {type(sys)}.')


def _mtype2obj(mtype):
    if mtype == 'lti':
        return LSSModel
    elif mtype == 'kf':
        return KFModel
    elif mtype == 'slti':
        return LGSSModel
    else:
        raise TypeError(f'Unknown model type {mtype}.')


## Function decorator and other helper functions for the disturbance model
## generators.
def _get_disturbance_model(_disturbance_free_model):
    """Returns a function decorator for disturbance model generation. Takes a
    function that generates disturbance-free models as inputs and returns a
    function that generates the corresponding linear augmented disturbance
    model as an output.

    The function `_disturbance_free_model` should take the form:

    ```
    def _disturbance_free_model(init_subsys, **kwargs):
        ...
        return init_subsys, params_subsys, theta, theta0, T
    ```

    We don't use the decorator @func syntax anywhere because the
    disturbance-free generators tend to be used within other functions. The
    decorator should be seen as a function wrapper, providing the necessary
    processing before and after the `(A,B,C)` subsystem generation. This is
    done to make sure the disturbance model is in the right form and contains
    all the parameters we want.

    """
    def _deco(init_sys, **kwargs):
        ## Unpack dims and model type.
        naug, m, p = init_sys.n, init_sys.m, init_sys.p
        dmodel = kwargs.get('dmodel', 'output')

        ## Process the disturbance model kwarg and make sure init_sys is in the
        ## right realization.
        init_sys, _, _, nd = _transform_dmodel(init_sys, dmodel=dmodel)
        n = naug - nd

        ## Get the symbolic and initial (A,B,C) subsystems, including the
        ## similarity transform required. We can just use an LTI model because
        ## the noise system isn't affected.
        init_subsys = \
            LSSModel(init_sys.A[:n, :n], init_sys.B[:n, :m], init_sys.C[:p, :n])
        kwargs_subsys = kwargs.copy()
        kwargs_subsys['mtype'] = 'lti'
        kwargs_subsys['_return_T'] = True
        init_subsys, params_subsys, theta, theta0, T = \
            _disturbance_free_model(init_subsys, **kwargs_subsys)
        A, B, C = params_subsys['A'], params_subsys['B'], params_subsys['C']

        ## Apply the similarity transform to the *full* initial system
        init_sys.transform(safeblock([[T, zeros((n, nd), dtype=init_sys.dtype)],
                                      [zeros((nd, n), dtype=init_sys.dtype),
                                       eye(nd, dtype=init_sys.dtype)]]))

        ## Made (Bd,Cd)
        Bd, Cd, _ = _get_dmodel_params(init_sys, B, dmodel=dmodel)

        ## Get the *full* system params dictionary
        params = {'A': safeblock([[A, Bd], [cs.SX(nd, n), cs.SX.eye(nd)]]),
                  'B': safevertcat([B, cs.SX(nd, m)]),
                  'C': safehorzcat([C, Cd])}

        ## Finally, process and return the remaining parameters
        return _finish_params(init_sys, params, theta, theta0, **kwargs)
    return _deco

## TODO Helper function to put an arbitrary system with nd simple integrators
## into the standard linear augmented disturbance model form. This might be
## tricky unless we restrict ourselves to diagonalizable systems. Otherwise,
## try doing nd steps of the Schur form construction?

def _get_dmodel_params(sys, B=None, dmodel='output', dtype=None):
    """Gets disturbance model parameters `(Bd,Cd)`, of the desired type
    `dmodel`, based on the dimensions of the reference model `sys` and an
    optional matrix `B`.

    """
    ## Unpack dims
    naug, m, p = sys.n, sys.m, sys.p
    if B is None:
        B = sys.B
    if dtype is None:
        dtype = get_dtype(B)

    ## Process disturbance model string options into indices
    if dmodel == 'output':
        dmodel = (np.zeros((0,), dtype=int), np.arange(p))
        nd = p
    elif dmodel == 'input':
        dmodel = (np.arange(m), np.zeros((0,), dtype=int))
        nd = m
    elif len(dmodel) == 3 and dmodel[0] == 'custom':
        _, Bd, Cd = dmodel
        nd = Bd.shape[1]
        if Bd.shape[1] != Cd.shape[1]:
            raise ValueError("Expected `(Bd,Cd)` to have the same second dimension: "
                             f"got {Bd.shape[1]} and {Cd.shape[1]} instead.")
        if Cd.shape[0] != p:
            raise ValueError("Expected `Cd` to have first dimension `p={p}`: "
                             f"got {Cd.shape[0]} instead.")
        return Bd, Cd, nd
    elif len(dmodel) != 2:
        raise ValueError("Invalid disturbance model format. Valid formats are:\n"
                         "* a string with value 'output' or 'input',\n"
                         "* a triple `('custom', Bd, Cd)` with `(Bd, Cd)` of proper dimensions,\n"
                         "* a tuple `(Bd_indices, Cd_indices)` with total length `nd` among the elements.")

    ## Get disturbance model dims
    nd_in = len(dmodel[0])
    nd_out = len(dmodel[1])
    nd = nd_in + nd_out
    n = naug - nd

    ## Get Bd and Cd from the indices and return.
    Bd = safehorzcat([B[:n, dmodel[0]], zeros((n, nd_out), dtype=dtype)])
    Cd = safehorzcat([zeros((p, nd_in), dtype=dtype),
                      eye(p, dtype=dtype)[:, dmodel[1]]])
    return Bd, Cd, nd


def _transform_dmodel(sys, dmodel='output', dtype=None, tol=1e-8):
    """Transform `sys` into a disutrbance model of the desired type `dmodel`."""
    ## Unpack dims. Initial _get_dmodel_params call to get the indices.
    naug, m, p = sys.n, sys.m, sys.p
    _, _, nd = _get_dmodel_params(sys, dmodel=dmodel)
    n = naug - nd
    if dtype is None:
        dtype = sys.dtype

    ## TODO put system into linear augmented disturbance model form.
    ## For now, however, we just check that the bottom nd rows of A look right.
    err = cs.sum1(cs.vec(
        cs.DM(sys.A[n:, :] - safehorzcat([zeros((nd, n), dtype=dtype),
                                          eye(nd, dtype=dtype)]))
    )).full()[0, 0]
    if err >= tol:
        raise ValueError("Lower block of `Aaug` differs from expected `[0, I]`.\n"
                         f"Got `norm_inf(Aaug-[0, I])={err}`, but we require <={tol}.")

    ## Get the current state model (A0,C0) and disturbance model (Bd0,Cd0).
    A0, C0 = sys.A[:n, :n], sys.C[:, :n]
    Bd0, Cd0 = sys.A[:n, n:], sys.C[:, n:]

    ## Form target model (Bd1,Cd1). Real _get_dmodel_params call.
    Bd1, Cd1, _ = _get_dmodel_params(sys, dmodel=dmodel)

    ## Get the transformation from (Bd0,Cd0) to (Bd1,Cd1). Implicitly checks
    ## detectability.
    T1 = safevertcat([cs.DM.eye(n), cs.DM(nd, n)])
    T2 = mldivide(safeblock([[A0 - cs.DM.eye(n), Bd1], [C0, Cd1]]),
                  safevertcat([Bd0, Cd0]))
    T = safehorzcat([T1, T2])
    sys.transform(T)

    return sys, Bd1, Cd1, nd

import cvxpy as cp
def discrete_stability_problem(A, delta=0, epsilon=1):
    n = A.shape[0]
    P = cp.Variable((n, n), symmetric=True)
    constraints = [P >> epsilon*np.eye(n)]
    constraints += [P - A@P@A.T/(1-delta)**2 >> np.eye(n)]
    prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
    prob.solve()
    return P.value

def continuous_stability_problem(A, delta=0, epsilon=1):
    n = A.shape[0]
    P = cp.Variable((n, n), symmetric=True)
    constraints = [P >> epsilon*np.eye(n)]
    Ashift = A - delta*np.eye(n)
    constraints += [Ashift@P + P@Ashift.T >> np.eye(n)]
    prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
    prob.solve()
    return P.value

def conic_stability_problem(A, delta=1, M0=None):
    n = A.shape[0]
    if M0 is None:
        M0 = 1e-3*np.eye(2*n)
    P = cp.Variable((n, n), symmetric=True)
    constraints = [P >> 0]
    AP = A@P
    APp = AP+AP.T
    APm = AP-AP.T
    M = cp.bmat([[delta*APp, APm], [APm.T, delta*APp]])
    constraints += [M >> M0]
    prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
    prob.solve(verbose=True)
    return P.value

def strip_stability_problem(A, delta=1, M0=None):
    n = A.shape[0]
    if M0 is None:
        M0 = 1e-3*np.eye(2*n)
    P = cp.Variable((n, n), symmetric=True)
    constraints = [P >> 0]
    AP = A@P
    APm = AP-AP.T
    M = cp.bmat([[delta*P, APm], [APm.T, delta*P]])
    constraints += [M >> M0]
    prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
    prob.solve(verbose=True)
    return P.value

# def LMI_region_problem(A, M0, M1, Pmin=None):
#     n = A.shape[0]
#     m = M0.shape[0]
#     if Pmin is None:
#         Pmin = 1e-3*np.eye(n)
#     P = cp.Variable((n, n), symmetric=True)
#     constraints = [P >> Pmin]
#     Q = cp.kron(M0, P) + cp.kron(M1, A@P) + cp.kron(M1.T, P@A.T)
#     constraints += [Q >> 0]
#     prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
#     prob.solve(verbose=True)
#     return P.value

def LMI_region_problem(A, M0, M1, Qmin=None):
    n = A.shape[0]
    m = M0.shape[0]
    if Qmin is None:
        Qmin = 1e-3*np.eye(n*m)
    P = cp.Variable((n, n), symmetric=True)
    constraints = [P >> 0]
    Q = cp.kron(M0, P) + cp.kron(M1, A@P) + cp.kron(M1.T, P@A.T)
    constraints += [Q >> Qmin]
    prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
    prob.solve(verbose=True)
    return P.value

def _solve(method, nlp, x0, bounds={}, hessian=None, max_iter=None, tol=None,
           with_jit=True, verbosity=5):
    opts = _nlpsol_options(max_iter=max_iter, tol=tol, with_jit=with_jit,
                           verbosity=verbosity)
    if method == 'ipopt':
        pass
    elif method.lower() in ['limited-memory', 'limited memory', 'bfgs', 'lbfgs',
                            'l-bfgs']:
        opts['ipopt.hessian_approximation'] = 'limited-memory'
    elif method.lower() in ['gauss-newton', 'gauss newton', 'gn',
                            'fisher scoring', 'scoring', 'fsa']:
        sigma = type(hessian).sym('sigma')
        hessLag = cs.Function(
            'nlp_hess_l',
            {'x': nlp['x'], 'lam_f': sigma,
             'hess_gamma_x_x': sigma*cs.triu(hessian)},
            ['x', 'p', 'lam_f', 'lam_g'],
            ['hess_gamma_x_x']
        )
        opts['hess_lag'] = hessLag
    else:
        raise ValueError(f'Unknown method ``{method}``.')
    solver = cs.nlpsol('solver', 'ipopt', nlp, opts)
    result = solver(x0=x0, **bounds)
    stats = solver.stats()
    return result, stats
