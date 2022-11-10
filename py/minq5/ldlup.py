import numpy as np
from ldlrk1 import ldlrk1


def ldlup(L, d, j, g):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ldlup.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % function [L,d,p]=ldlup(L,d,j,g)
    % updates LDL^T factorization when a unit j-th row and column
    % are replaced by column g
    % if the new matrix is definite (signalled by p=[]);
    % otherwise, the original L,d and
    % a direction p of null or negative curvature are returned
    %
    % d contains diag(D) and is assumed positive
    % Note that g must have zeros in other unit rows!!!
    %
    """
    eps = np.finfo(float).eps  # Define machine epsilon

    p = []

    test = 0
    if test:
        print("enter ldlup")
        A = L @ np.diag(d.squeeze()) @ L.T
        A[:, [j]] = g
        A[j, :] = g.T

    n = len(d)
    I = np.arange(0, j)
    K = np.arange(j + 1, n)
    if j == 0:
        v = np.zeros((0, 1))
        del_ = g[j]
        if del_ <= n * eps:
            p = np.eye(n)[:, 0]
            if test:
                print(A, p)
                Nenner = abs(p).T @ abs(A) @ abs(p)
                if Nenner == 0:
                    indef1 = 0
                else:
                    indef1 = (p.T @ A @ p) / Nenner
                disp("leave ldlup at 1")
            return L, d, p
        w = g[K] / del_
        L[j, I] = v.T
        d[j] = del_
        if test:
            A1 = L @ np.diag(d.squeeze()) @ L.T
            quot = np.linalg.norm(A1 - A, 1) / np.linalg.norm(A, 1)
            print("leave ldlup at 3")
        return L, d, p

    # now j>1, K nonempty
    LII = L[np.ix_(I, I)]
    u = np.linalg.solve(LII, g[I])
    v = u / d[I]
    del_ = g[j] - u.T @ v
    if del_ <= n * eps:
        p = np.vstack((np.linalg.solve(LII.T, v), -1, np.zeros((n - j - 1, 1))))
        if test:
            indef1 = (p.T @ A @ p) / (abs(p).T @ abs(A) @ abs(p))
            print("leave ldlup at 2")
        return L, d, p

    LKI = L[np.ix_(K, I)]
    w = (g[K] - LKI @ u) / del_
    LKK, d[K], q = ldlrk1(L[np.ix_(K, K)].copy(), d[K].copy(), -del_, w.copy())
    if len(q) == 0:
        # work around expensive sparse L[K,K]=LKK
        L = np.vstack((L[I, :], np.hstack((v.T, np.ones((1, 1)), L[np.ix_([j], K)])), np.hstack((LKI, w, LKK))))
        d[j] = del_
        if test:
            A1 = L @ np.diag(d.squeeze()) @ L.T
            print(A, A1)
            quot = np.linalg.norm(A1 - A, 1) / np.linalg.norm(A, 1)
            print("leave ldlup at 4")
    else:
        # work around expensive sparse L(K,K)=LKK
        L = np.vstack((L[:j + 1, :], np.hstack((LKI, L[np.ix_(K, [j])], LKK))))
        pi = w.T @ q
        p = np.vstack((np.linalg.solve(LII.T, (pi * v - LKI.T @ q)), -pi, q))
        if test:
            indef2 = (p.T @ A @ p) / (abs(p).T @ abs(A) @ abs(p))
            print("leave ldlup at 5")
    return L, d, p
