import numpy as np


def ldlrk1(L, d, alp, u):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ldlrk1.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % function [L,d,p]=ldlrk1(L,d,alp,u)
    % computes LDL^T factorization for LDL^T+alp*uu^T
    % if alp>=0 or if the new factorization is definite
    % (both signalled by p=[])
    % otherwise, the original L,d and
    % a direction p of null or negative curvature are returned
    %
    % d contains diag(D) and is assumed positive
    %
    % does not work for dimension 0
    %
    """
    eps = np.finfo(float).eps  # Define machine epsilon

    test = 0
    # only for testing the routine
    if test:
        print("enter ldlrk1")
        if len(d) == 1:
            A = L @ d @ L.T + (alp * u) @ u.T
        else:
            A = L @ np.diag(d.squeeze()) @ L.T + (alp * u) @ u.T

    p = []
    if alp == 0:
        return L, d, p

    n = len(u)
    neps = n * eps

    # save old factorization
    L0 = L.copy()
    d0 = d.copy()

    # update
    for k in np.where(u != 0)[0]:
        del_ = d[k] + alp * u[k]**2
        if alp < 0 and del_ <= neps:
            # update not definite
            p = np.zeros((n, 1))
            p[k] = 1
            p[:k + 1] = np.linalg.solve(L[:k + 1, :k + 1].T, p[:k + 1])
            # restore original factorization
            L = L0
            d = d0
            if test:
                indef = (p.T @ (A @ p)) / (abs(p).T @ (abs(A) @ abs(p)))
                print("leave ldlrk1 at 1")
            return L, d, p

        q = d[k] / del_
        d[k] = del_
        # in C, the following 3 lines would be done in a single loop
        ind = np.arange(k + 1, n)
        c = L[ind, k] * u[k]
        L[ind, k] = L[ind, k] * q + (alp * u[k] / del_) * u[ind, 0]
        u[ind, 0] = u[ind, 0] - c
        alp = alp * q
        if alp == 0:
            break

    if test:
        if len(d) == 1:
            A1 = L @ d @ L.T
        else:
            A1 = L @ np.diag(d.squeeze()) @ L.T

        quot = np.linalg.norm(A1 - A, 1) / np.linalg.norm(A, 1)
        print("leave ldlrk1 at 2")

    return L, d, p
