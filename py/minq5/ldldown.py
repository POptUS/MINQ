import numpy as np
from ldlrk1 import ldlrk1
from scipy import sparse


def ldldown(L, d, j):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ldldown.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % function [L,d]=ldldown(L,d,j)
    % downdates LDL^T factorization when j-th row and column are replaced
    % by j-th unit vector
    %
    % d contains diag(D) and is assumed positive
    %
    """

    n = len(d)

    test = 0
    if test:
        print("enter ldldown")
        A = L @ np.diag(d.squeeze()) @ L.T
        A[:, j] = np.zeros(n)
        A[j, :] = np.zeros(n)
        A[j, j] = 1

    if j < n:
        I = np.arange(0, j)
        K = np.arange(j + 1, n)
        LKK, d[K], _ = ldlrk1(L[np.ix_(K, K)].copy(), d[K].copy(), d[j].copy(), L[np.ix_(K, [j])].copy())
        # work around expensive sparse L(K,K)=LKK
        # L=[L[I,:]; sparse.csr_matrix((1, n)); L[K,I],sparse.csr_matrix((n-j,1)),LKK]
        # L = np.vstack((L[I, :], sparse.csr_matrix((1, n)), np.hstack((L[K, I], sparse.csr_matrix((n - j, 1)), LKK))))
        L = np.vstack((L[I, :], np.zeros((1, n)), np.hstack((L[np.ix_(K, I)], np.zeros((n - j - 1, 1)), LKK))))
        L[j, j] = 1
    else:
        L[n, 1 : n - 1] = sparse.csr_matrix((1, n - 1))

    d[j] = 1

    if test:
        A1 = L @ np.diag(d.squeeze()) @ L.T
        quot = np.linalg.norm(A1 - A, 1) / np.linalg.norm(A, 1)
        print("leave ldldown")
    return L, d
