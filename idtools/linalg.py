"""
Linear algebraic helper functions.
"""
import casadi as cs
import numpy as np

from .util import *


def hankel(M, i, j, isreversed=False, dtype=None):
    """Given a block matrix of the form

    M = [M(0), M(1), ..., M(i+j-2)] (default, isreversed==False)
    M = [M(i+j-2), ..., M(1), M(0)] (isreversed==True)

    returns a block Toeplitz matrix of the form

    H = [[M(0), M(1), ..., M(j-1)],
         [M(1), M(2), ..., M(j)],
         ...,
         [M(i-1), M(i), ..., M(i+j-2)]]
    """
    n = M.shape[1] // (i+j-1)
    return safevertcat([M[:, k*n:(k+j)*n] for k in range(i)], dtype=dtype)


def toeplitz(M, i, isreversed=False, dtype=None):
    """Given a block matrix of the form

    M = [M(0), M(1), ..., M(i+j-1)] (default, isreversed==False)
    M = [M(i+j-1), ..., M(1), M(0)] (isreversed==True)

    returns a block Toeplitz matrix of the form

    G = [[M(0), 0, ..., 0],
         [M(1), M(2), 0, ..., 0],
         ...,
         [M(i-1), M(i-2), ..., M(0)]]
    """
    M = safehsplit(M, i, dtype=dtype)
    # We actually want the time-reversed indexing for stacking purposes
    if not isreversed:
        M = M[::-1]
    m, n = M[0].shape
    return safeblock([M[-k-1:] + [zeros((m, n*(i-k-1)), dtype=dtype)]
                      for k in range(i)], dtype=dtype)


def vech(M):
    """Lower triangular vectorization."""
    i1, i0 = np.triu_indices(M.shape[0])
    if isnumpy(M):
        return M[i0, i1]
    elif iscasadi(M):
        return cs.vertcat(*[M[i0[i],i1[i]] for i in range(i0.shape[0])])


def unvech(x, n):
    """Inverse lower triangular vectorization."""
    i1, i0 = np.triu_indices(n)
    if isinstance(x, cs.SX):
        x_new = cs.SX(n, n)
    elif isinstance(x, cs.MX):
        x_new = cs.MX(n, n)
    elif isinstance(x, np.ndarray):
        x_new = np.zeros((n, n))
    else:
        raise TypeError(f"Type {type(x)} not implemented for `unvech`.")
    for (i, (a, b)) in enumerate(zip(i0, i1)):
        x_new[a, b] = x[i]
    return x_new


def vecs(M):
    """Symmetric vectorization."""
    return vech(M)


def unvecs(x, n):
    """Inverse symmetric vectorization."""
    M = unvech(x, n)
    if type(M) in [np.ndarray, np.matrix]:
        return M + M.T - np.diag(np.diag(M))
    elif type(M) in [cs.SX, cs.MX, cs.DM]:
        return M + M.T - cs.diag(cs.diag(M))
    else:
        raise TypeError(f"Type {type(x)} not implemented for `unvecs`.")


def logdet(n):
    A = cs.SX.sym('A', n, n)
    _, R = cs.qr(A)
    pos_eigs = cs.fmax(cs.diag(R), np.sqrt(np.finfo(float).eps))
    log_det = cs.sum1(cs.log(pos_eigs))
    return cs.Function('F', [A], [log_det])

def det(n):
    A = cs.SX.sym('A', n, n)
    _, R = cs.qr(A)
    pos_eigs = cs.fmax(cs.diag(R), np.sqrt(np.finfo(float).eps))
    det = cs.mtimes([pos_eigs[i] for i in range(pos_eigs.numel())])
    return cs.Function('F', [A], [det])


def safeqr(M, mode='reduced', dtype=None):
    """Safe wrapper for the QR decomposition. Uses `casadi.qr` for casaditypes
    and `numpy.qr` for numpytypes.

    The `mode` mode option can be used to switch between return options:
    * `'reduced'` (default): returns `Q, R` with dimensions `(n, r), (r, m)`,
      where `r` is the rank.
    * `'complete'`: returns `Q, R` with dimensions `(n, n), (n, m)`. Only
      available for numpytypes since `casadi.qr` is missing the left nullspace
      basis vectors.
    * `'r'`: returns only `R` with dimensions `(r, m)`.
    * `'raw'`: returns the Householder reflections and scaling factors. Only
      available for numpytypes. See `numpy.linalg.qr` for more details.
    Note: `casadi.qr` only works on tall `M` with full rank.

    """
    if dtype is None:
        dtype = get_dtype(M)
    if iscasadi(dtype):
        if mode == 'reduced':
            return cs.qr(M)
        elif mode in ('complete', 'raw'):
            raise TypeError(f'For casaditypes, `mode={mode}` is not implemented for `safeqr`.')
        elif mode == 'r':
            return cs.qr(M)[1]
        else:
            raise TypeError(f'Unknown mode option: `{mode}`.')
    elif isnumpy(dtype):
        return np.linalg.qr(M, mode=mode)
    else:
        return TypeError(f"Unexpected type encountered: {dtype}")


def safelq(M, mode='reduced', dtype=None):
    """Safe wrapper for the LQ decomposition. Uses `casadi.qr` for casaditypes
    and `numpy.qr` for numpytypes. See `safeqr` for information on the `mode`
    option. For this method, `mode='l'` can be used as a replacement to
    `mode='r'`.

    Note: `casadi.qr` only works on tall `M` with full rank, so this will only
    work on wide casadi matrices.

    """
    if dtype is None:
        dtype = get_dtype(M)
    if mode == 'l':
        mode = 'r'
    output = safeqr(M.T, mode=mode, dtype=dtype)
    if mode in ('reduced', 'complete'):
        return output[1].T, output[0].T
    elif mode in 'r':
        return output.T
    elif mode == 'raw':
        raise TypeError(f'`mode={mode}` is not implemented for `safelq`.')
    else:
        raise TypeError(f'Unknown mode option: `{mode}`.')


def safechol(M, dtype=None, tol=1e-8):
    """Safe wrapper for Cholesky factorizations. Returns the lower-triangular
    factor `L`. For `casadi` types, this requires `M` to be positive definite,
    but with `numpy` types, we can combine the eigenvalue and LQ decompositions
    to get a lower triangular matrix.

    """
    if dtype is None:
        dtype = get_dtype(M)
    if iscasadi(dtype):
        return cs.chol(M).T
    elif isnumpy(dtype):
        ## Try the Cholesky decomposition; otherwise we need to do an
        ## eigenvalue decomposition to find a square root and then LQ
        ## decomposition to find a semidefinite Cholesky factor.
        try:
            ## Try the Cholesky decomposition; raises a LinAlgError when
            ## semi/indefinite
            return np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            ## Otherwise, compute the symmetric version. Start with the
            ## eigenvalue decomposition M=VDV'
            D, V = np.linalg.eigh(M)
            ## If any eigenvalues are too negative, throw an error.
            if np.amin(D)<-tol:
                raise ValueError(f'Encountered negative eigenvalue {np.amin(D)}'
                                 f'less than tolerance {-tol}.')
            ## Compute a square root M=BB'.
            B = V @ np.diag(np.sqrt(np.maximum(D, np.zeros((D.shape[0],)))))
            ## Compute the LQ decomposition B=LQ. Then M=LL'.
            L, Q = safelq(B, mode='complete')
            ## But the diagonal elements are still wrong so we must fix them.
            L = L @ np.diag(np.sign(np.diag(L)))
            return L
    else:
        return TypeError(f"Unexpected type encountered: {dtype}")


def mldivide(A, B, dtype=None):
    """Safe wrapper for linear solvers. In matlab syntax, returns `X=A\B`."""
    if dtype is None:
        dtype = get_dtype(A, B)
    if iscasadi(dtype):
        return cs.mldivide(A, B)
    elif isnumpy(dtype):
        return np.linalg.solve(A, B)
    else:
        return TypeError(f"Unexpected type encountered: {dtype}")


def mrdivide(A, B, dtype=None):
    """Safe wrapper for linear solvers. In matlab syntax, returns `X=A/B`."""
    return mldivide(B.T, A.T, dtype=dtype).T
