"""
Utility and data processing functions for other modules.

delay_coords
"""
from collections.abc import Iterable, Sequence

import casadi as cs
import numpy as np

CASADI_TYPES = (cs, cs.SX, cs.MX, cs.DM)
NUMPY_TYPES = (np, np.ndarray, np.matrix)

def iscasadi(M):
    """Check if a variable or type is (of) a casadi type."""
    try:
        return (M in CASADI_TYPES) or (type(M) in CASADI_TYPES)
    except:
        return type(M) in CASADI_TYPES
    # return (M in CASADI_TYPES) or (type(M) in CASADI_TYPES)

def isnumpy(M):
    """Check if a variable or type is (of) a numpy type."""
    try:
        return (M in NUMPY_TYPES) or (type(M) in NUMPY_TYPES)
    except:
        return type(M) in NUMPY_TYPES
    # return (M in NUMPY_TYPES) or (type(M) in NUMPY_TYPES)

def get_dtype(*args):
    """get_dtype(arg1, arg2, ...)

    Gets the datatype from the arguments. Returns
      - an error if there are both SX and MX expressions in the arguments,
      - `casadi.SX` if any argument is a SX expressions,
      - `casadi.MX` if any argument is a MX expressions,
      - `casadi.DM` if all arguments are `numpy` types or DM matrices.
      - `numpy.ndarray` otherwise.

    """
    if any(iscasadi(M) for M in args):
        isSX = any(isinstance(M, cs.SX) for M in args)
        isMX = any(isinstance(M, cs.MX) for M in args)
        if isSX and isMX:
            raise TypeError("Unexpected types in arguments: Cannot mix SX and MX types.")
        elif isSX:
            return cs.SX
        elif isMX:
            return cs.MX
        else:
            return cs.DM
    else:
        return np.ndarray
    ## OLD: Breaks if scalars are included.
    # else:
    #     ## Get the index of the first invalid argument.
    #     index = np.nonzero([not (iscasadi(M) or isnumpy(M)) for M in args])[0][0]
    #     ## Return a message about that argument.
    #     raise TypeError(f"Invalid type encountered: {type(args[index])} at argument {index}.")


def zeros(dims, dtype=None):
    """Safe wrapper for numpy's zeros function. Returns
    - cs.DM(dims[0], dims[1]) if dtype is a casadi type.
    - np.zeros(dims) if dtype is None or a numpy type.
    """
    if iscasadi(dtype):
        if not isinstance(dims, Iterable):
            return cs.DM(dims, 1)
        elif len(dims) == 1:
            return cs.DM(*dims, 1)
        else:
            return cs.DM(*dims)
    elif isnumpy(dtype) or dtype is None:
        return np.zeros(dims)
    else:
        raise TypeError(f"Unknown data type: {dtype}.")


def eye(dim, dtype=None):
    """Safe wrapper for generating identity matrices. Returns
    - cs.DM(dims[0], dims[1]) if dtype is a casadi type.
    - np.zeros(dims) if dtype is None or a numpy type.
    """
    if iscasadi(dtype):
        if dtype in (cs.SX, cs.MX, cs.DM):
            return dtype.eye(dim)
        else:
            return cs.DM.eye(dim)
    elif isnumpy(dtype) or dtype is None:
        return np.eye(dim)
    else:
        raise TypeError(f"Unknown data type: {dtype}.")


def elementary(n, k, dtype=None):
    """Creates the ``k``-th elementary ``n``-vector. Defaults to
    ``np.ndarray``, but can return casadi ``DM`` types by supplying a valid
    casadi type or variable in ``dtype``."""
    if iscasadi(dtype):
        e = cs.DM(n, 1)
    elif isnumpy(dtype) or dtype is None:
        e = np.zeros((n,))
    else:
        if isinstance(dtype, type):
            raise TypeError(f"Unknown data type: {dtype}.")
        else:
            raise TypeError(f"The `dtype` keyword must be supplied a valid type.")
    e[k] = 1
    return e


def safevertcat(M, dtype=None):
    """Safe wrapper for casadi's vertcat function.

    Parameters
    ----------
    M : Sequence
        The sequence of casadi/numpy objects to be concatenated.
    dtype : type (default None)
        Output type. If set to None (default) the type is inferred from M using
        `get_dtype`.

    Returns
    -------
    M : casadi or numpy matrix
        * M = casadi.horzcat(*M) if iscasadi(dtype),
        * M = numpy.hstack(M) otherwise.
    """
    if dtype is None:
        dtype = get_dtype(*M)
    return cs.vertcat(*M) if iscasadi(dtype) else np.vstack(M)


def safehorzcat(M, dtype=None):
    """Safe wrapper for casadi's horzcat function.

    Parameters
    ----------
    M : Sequence
        The sequence of casadi/numpy objects to be concatenated.
    dtype : type (default None)
        Output type. If set to None (default) the type is inferred from M using
        `get_dtype`.

    Returns
    -------
    M : casadi or numpy matrix
        * M = casadi.vertcat(*M) if iscasadi(dtype),
        * M = numpy.vstack(M) otherwise.
    """
    if dtype is None:
        dtype = get_dtype(*M)
    return cs.horzcat(*M) if iscasadi(dtype) else np.hstack(M)


def safeblock(M, dtype=None):
    """Safe wrapper for numpy's block function.

    Parameters
    ----------
    M : Sequence of Sequences (typically of casadi/numpy types)
    dtype : type (default None)
        Output type. If set to None (default) the type is inferred from M using
        `get_dtype`.

    Returns
    -------
    M : casadi or numpy matrix
        * M = casadi.vertcat(cs.horzcat(*Mi) for Mi in M) if iscasadi(dtype),
        * M = numpy.block(M) otherwise.
    """
    if dtype is None:
        # Get dtype if not supplied. First, flatten the Sequence of Sequences.
        Mflat = []
        for Mi in M:
            Mflat.extend(Mi)
        # Next, unpack the flattened list into get_dtype and return
        dtype = get_dtype(*Mflat)
    return cs.vertcat(*[cs.horzcat(*Mi) for Mi in M]) if iscasadi(dtype) \
        else np.block(M)


def safehsplit(M, n, dtype=None):
    """Safe wrapper for numpy's hsplit function (equivalent to casadi's
    horzsplit_n function). Splits a matrix `M` (along the horizontal axis) into
    a list of `n` matrices of equal dimension. Note that dimensions of the
    submatrices are implied from context and mismatched shapes will throw
    errors.

    Parameters
    ----------
    M : Sequence
        The casadi/numpy objects to be split.
    dtype : type (default None)
        Output type. If set to None (default) the type is inferred from M using
        `get_dtype`.

    Returns
    -------
    L : list of casadi/numpy matrices
        * L = casadi.horzsplit_n(M, n) if iscasadi(dtype),
        * L = numpy.hsplit(M, n) otherwise.

    """
    if dtype is None:
        dtype = get_dtype(M)
    return cs.horzsplit_n(M, n) if iscasadi(dtype) else np.hsplit(M, n)


def safevsplit(M, n, dtype=None):
    """Safe wrapper for numpy's vsplit function (equivalent to casadi's
    vertsplit_n function). Splits a matrix `M` (along the vertical axis) into a
    list of `n` matrices of equal dimension. Note that dimensions of the
    submatrices are implied from context and mismatched shapes will throw
    errors.

    Parameters
    ----------
    M : Sequence
        The casadi/numpy objects to be split.
    dtype : type (default None)
        Output type. If set to None (default) the type is inferred from M using
        `get_dtype`.

    Returns
    -------
    L : list of casadi/numpy matrices
        * L = casadi.vertsplit_n(M, n) if iscasadi(dtype),
        * L = numpy.vsplit(M, n) otherwise.

    """
    if dtype is None:
        dtype = get_dtype(M)
    return cs.vertsplit_n(M, n) if iscasadi(dtype) else np.vsplit(M, n)


def safehorzsplit(M, incr, dtype=None):
    """Safe wrapper for casadi's horzsplit function. Splits a matrix `M` (along
    the horizontal axis) by increments `incr`.

    Parameters
    ----------
    M : Sequence
        The casadi/numpy objects to be split.
    dtype : type (default None)
        Output type. If set to None (default) the type is inferred from M using
        `get_dtype`.

    Returns
    -------
    M : list of casadi/numpy matrices
        * M = casadi.horzsplit(M, n) if iscasadi(dtype),
        * M = numpy.hsplit(M, M.shape[1] // incr) otherwise.
    """
    if dtype is None:
        dtype = get_dtype(M)
    N = M.shape[1]
    return cs.horzsplit(M, incr) if iscasadi(dtype) else np.hsplit(M, N // incr)


def safevertsplit(M, incr, dtype=None):
    """Safe wrapper for casadi's vertsplit function. Splits a matrix `M` (along
    the horizontal axis) by increments `incr`.

    Parameters
    ----------
    M : Sequence
        The casadi/numpy objects to be split.
    dtype : type (default None)
        Output type. If set to None (default) the type is inferred from M using
        `get_dtype`.

    Returns
    -------
    M : list of casadi/numpy matrices
        * M = casadi.vertsplit(M, n) if iscasadi(dtype),
        * M = numpy.vsplit(M, M.shape[1] // incr) otherwise.
    """
    if dtype is None:
        dtype = get_dtype(M)
    N = M.shape[0]
    return cs.vertsplit(M, incr) if iscasadi(dtype) else np.vsplit(M, N // incr)


def delay_coords(Y, U, ny, nu=None, feedthrough=False, weave=False):
    """delay_coords(Y, U, ny[, nu, feedthrough, weave])

    Return the delay coordinates of the given input-output data.

    By default, `nu=ny`, `feedthrough=False`, and `weave=False`. If
    `weave=False`, returns

    .. math::
        Z = \begin{bmatrix} y(0) & z(1) & \hdots & z(N-n-1) \\
                            \vdots & \vdots && \vdots \\
                            y(n-1) & y(n) & \ldots & y(N-2) \\
                            u(0) & u(1) & \hdots & u(N-n-1) \\
                            \vdots & \vdots && \vdots \\
                            u(n-1) & u(n) & \ldots & u(N-2) \\
                            u(n) & u(n+1) & \ldots & u(N-1) \end{bmatrix}

    where :math:`n=\max\{n_u,n_y\}`. Otherwise, if `weave=True`, returns

    .. math::
        Z = \begin{bmatrix} z(0) & z(1) & \hdots & z(N-n-1) \\
                            z(1) & z(2) & \ldots & z(N-n) \\
                            \vdots & \vdots && \vdots \\
                            z(n-1) & z(n) & \ldots & z(N-2) \\
                            u(n) & u(n+1) & \ldots & u(N-1) \end{bmatrix}

    where :math:`z=[u^\top,\; y^\top]^\top` and :math:`n=n_u=n_y` (an error is
    thrown if :math:`n_u\neq n_y`).

    If `feedthrough=True`, the current input $u(k)$ is added to the delay
    coordinates, e.g.

    .. math::
        Z \rightarrow \begin{bmatrix} Z \\ \begin{matrix} u(n) & u(n+1) & \ldots
            & u(N-1) \end{matrix} \end{bmatrix}

    """

    n, N = Y.shape
    m, M = U.shape
    if N != M:
        raise ValueError("Equal x and y samples expected: "
                         f"got Nx={N} and Ny={M}.")

    if nu is None:
        nu = ny
    ish = max([nu, ny])

    if weave and ny != nu:
            raise ValueError("For `weave=True`, requires `ny=nu`: "
                             f"got ny={ny} and nu={nu}.")

    if weave:
        Z = np.vstack([U, Y])
        Z = np.vstack([Z[:, ish-i-1:N-i] for i in range(ny)])
    else:
        Z = np.vstack(
            [U[:, ish-i-1:N-i] for i in range(nu)] +
            [Y[:, ish-i-1:N-i] for i in range(ny)]
        )

    if feedthrough:
        Z = np.vstack([U[:, ish:], Z[:, :-1]])

    return Z


def _check_UY_data(U, Y, p=None, m=None):
    m1, Nu, p1, Ny = U.shape + Y.shape
    if Nu != Ny:
        raise ValueError("Expected U and Y to have the same number of samples: "
                         f"got Nu={Nu} and Ny={Ny} instead.")
    if (p is not None) and (p != p1):
        raise ValueError(f"Expected Y to have first dimension {p}: got {p1} instead.")
    if (m is not None) and (m != m1):
        raise ValueError(f"Expected U to have first dimension {m}: got {m1} instead.")
    return m1, p1, Nu


def _check_XUY_data(X, U, Y):
    n, Nx, m, Nu, p, Ny = X.shape + U.shape + Y.shape
    if Nx != Ny + 1 or Ny != Nu:
        raise ValueError("Expected X, U, and Y to have N+1, N, and N samples: "
                         f"got Nx={Nx}, Nu={Nu}, and Ny={Ny} instead.")
    return n, m, p, Nu


def _is_positive_definite(M):
    return np.all(np.isreal(np.linalg.eigvals(M))) and \
        np.all(np.linalg.eigvals(M) > 0)


def _is_positive_semidefinite(M):
    return np.all(np.isreal(np.linalg.eigvals(M))) and \
        np.all(np.linalg.eigvals(M) >= 0)


def _test_delay_coords(p=3, m=2, ny=3, nu=3, N=10):
    U = np.vstack([np.arange(N) + 10*i for i in range(m)])
    Y = np.vstack([np.arange(N) + 10*i + 100 for i in range(p)])

    Z = delay_coords(Y, U, ny, nu)
    z00 = max([ny, nu]) - 1
    zy0 = z00 + 100
    assert Z[0, 0] == z00, (f"Expected z_0(0)=={z00}: "
                            f"got z_0(0)={Z[0, 0]}")
    assert Z[m*nu, 0] == zy0, (f"Expected z_{m*nu}(0)=={zy0}: "
                               f"got z_{m*nu}(0)={Z[m*nu, 0]}")

    Z = delay_coords(Y, U, ny, nu, feedthrough=True)
    z00 = max([ny, nu])
    zy0 = z00 + 99
    assert Z[0, 0] == z00, (f"Expected z_0(0)=={z00}: "
                            f"got z_0(0)={Z[0, 0]}")
    assert Z[m+m*nu, 0] == zy0, (f"Expected z_{m+m*nu}(0)=={zy0}: "
                                 f"got z_{m+m*nu}(0)={Z[m+m*nu, 0]}")


def _func_options(with_jit=True):
    opts = {}

    # jit compilation
    if with_jit:
        if cs.Importer.has_plugin('clang'):
            with_jit = True
            compiler = 'clang'
        elif cs.Importer.has_plugin('shell'):
            with_jit = True
            compiler = 'shell'
        else:
            raise Warning("Could not access the compiler; running without jit."
                          " This may result in very slow evaluation times.")
            with_jit = False
            compiler = ''
        opts['jit'] = with_jit
        opts['compiler'] = compiler

    return opts


def _nlpsol_options(verbosity=None, solver='ma57', max_iter=None, tol=None,
                    with_jit=True):
    opts = {}

    # ipopt options
    if verbosity is not None:
        opts['ipopt.print_level'] = verbosity
        if verbosity == 0:
            opts['print_time'] = 0
            opts['ipopt.sb'] = 'yes'
    if max_iter is not None:
        opts['ipopt.max_iter'] = max_iter
    if tol is not None:
        opts['ipopt.tol'] = tol

    # linear solver info
    if solver=='ma57':
        ## TODO Need a check function for all the coinhsl solvers.
        opts['ipopt.linear_solver'] = 'ma57'
        # opts['ipopt.ma57_automatic_scaling'] = 'yes'

    # TODO Autoscaling on/off switch, esp for MUMPS

    # jit compilation
    if with_jit:
        if cs.Importer.has_plugin('clang'):
            with_jit = True
            compiler = 'clang'
        elif cs.Importer.has_plugin('shell'):
            with_jit = True
            compiler = 'shell'
        else:
            print("WARNING; running without jit. This may result in very slow evaluation times")
            with_jit = False
            compiler = ''
        opts['jit'] = with_jit
        if verbosity is not None and verbosity > 0:
            opts['jit_options.verbose'] = True
        opts['compiler'] = compiler

    return opts
