import numpy as np
import scipy as sp
import ipdb  # Only used when prt is > 2
from getalp import getalp
from pr01 import pr01
from ldldown import ldldown
from ldlup import ldlup


def minqsw(gam, c, G, xu, xo, prt, xx=None):
    mdic = {'gam': gam, 'c':   c, 'G':   G, 'xu':  xu, 'xo':  xo, 'prt': prt, 'xx': xx}
    sp.io.savemat("matlab_matrix.mat", mdic)

    """
    % function [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt,xx)
    % minimizes an affine quadratic form subject to simple bounds,
    % using coordinate searches and reduced subspace minimizations
    % using LDL^T factorization updates
    %    min    fct = gam + c^T x + 0.5 x^T G x
    %    s.t.   x in [xu,xo]    % xu<=xo is assumed
    % where G is a symmetric n x n matrix, not necessarily definite
    % (if G is indefinite, only a local minimum is found)
    %
    % if G is sparse, it is assumed that the ordering is such that
    % a sparse modified Cholesky factorization is feasible
    %
    % prt	printlevel
    % xx	initial guess (optional)
    %
    % x	minimizer (but unbounded direction if ier=1)
    % fct	optimal function value
    % ier	0  (local minimizer found)
    % 	1  (unbounded below)
    % 	99 (maxit exceeded)
    % 	-1 (input error)
    %
    % calls getalp.m, ldl*.m, minqsub.m, pr01.m
    %
    function [x,fct,ier,nsub]=minqsw(gam,c,G,xu,xo,prt,xx)

    % This is minq with changes to:
    % 1) max number of iterations,
    % 2) display option for exceeding maxit
    % Search for 'SW' to find specific lines

    % Translated from Matlab to Python by Jeffrey Larson, 2021
    """
    c = np.atleast_2d(c).T
    xu = np.atleast_2d(xu).T
    xo = np.atleast_2d(xo).T

    # initialization
    convex = 0
    n = G.shape[0]
    eps = np.finfo(float).eps  # Define machine epsilon
    if np.ndim(xu) == 1:
        xu = np.atleast_2d(xu).T
    if np.ndim(xo) == 1:
        xo = np.atleast_2d(xo).T
    if np.ndim(c) == 1:
        c = np.atleast_2d(c).T

    np.set_printoptions(precision=16, linewidth=150)
    # check input data for consistency
    ier = 0
    if G.shape[1] != n:
        ier = -1
        print("minq: Hessian has wrong dimension")
        x = NaN + np.zeros(n)
        fct = NaN
        nsub = -1
        return x, fct, ier, nsub

    if c.shape[0] != n or c.shape[1] != 1:
        ier = -1
        print("minq: linear term has wrong dimension")
    if xu.shape[0] != n or xu.shape[1] != 1:
        ier = -1
        print("minq: lower bound has wrong dimension")
    if xo.shape[0] != n or xo.shape[1] != 1:
        ier = -1
        print("minq: lower bound has wrong dimension")
    if xx is not None:
        if xx.shape[0] != n or xx.shape[1] != 1:
            ier = -1
            print("minq: lower bound has wrong dimension")
    if ier == -1:
        x = NaN + zeros(n)
        fct = NaN
        nsub = -1
        return x, fct, ier, nsub

    maxit = 3 * n  # maximal number of iterations
    maxit = 5 * n  # maximal number of iterations % Changed by SW
    # this limits the work to about 1+4*maxit/n matrix multiplies
    # usually at most 2*n iterations are needed for convergence
    nitrefmax = 3  # maximal number of iterative refinement steps

    # initialize trial point xx, function value fct and gradient g

    if xx is None:
        # cold start with absolutely smallest feasible point
        xx = zeros(n)

    # force starting point into the box
    xx = np.maximum(xu, np.minimum(xx, xo))

    # regularization for low rank problems
    hpeps = 100 * eps  # perturbation in last two digits
    G = G + sp.sparse.spdiags(hpeps * np.diag(G), 0, n, n)

    # initialize LDL^T factorization of G_KK
    K = np.zeros(n, dtype=bool)  # initially no rows in factorization
    if sp.sparse.issparse(G):
        L = sp.sparse.eye(n)
    else:
        L = np.eye(n)
    dd = np.ones((n, 1))

    # dummy initialization of indicator of free variables
    # will become correct after first coordinate search
    free = np.zeros(n, dtype=bool)
    nfree = 0
    nfree_old = -1

    fct = np.inf  # best function value
    nsub = 0  # number of subspace steps
    unfix = 1  # allow variables to be freed in csearch?
    nitref = 0  # no iterative refinement steps so far
    improvement = 1  # improvement expected

    ########################################################################
    # main loop: alternating coordinate and subspace searches
    while 1:
        if prt > 1:
            print("enter main loop")

        if np.linalg.norm(xx, np.inf) == np.inf:
            error("infinite xx in minq.m")

        g = G * xx + c
        fctnew = gam + 0.5 * xx.T @ (c + g)
        if not improvement:
            # good termination
            if prt:
                print("terminate: no improvement in coordinate search")
            ier = 0
            break
        elif nitref > nitrefmax:
            # good termination
            if prt:
                print("terminate: nitref>nitrefmax")
            ier = 0
            break
        elif nitref > 0 and nfree_old == nfree and fctnew >= fct:
            # good termination
            if prt:
                print("terminate: nitref>0 & nfree_old==nfree & fctnew>=fct")
            ier = 0
            break
        elif nitref == 0:
            x = xx
            fct = np.minimum(fct, fctnew)
            if prt > 1:
                print("fct ", fct)
            if prt > 2:
                X = x.T
                print("X ", X)
                print("fct ", fct)
        else:  # more accurate g and hence f if nitref>0
            x = xx
            fct = fctnew
            if prt > 1:
                print(fct)
            if prt > 2:
                X = x.T
                print("X ", X)
                print("fct ", fct)

        if nitref == 0 and nsub >= maxit:
            if prt:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!           minq          !!!!!")
                print("!!!!! incomplete minimization !!!!!")
                print("!!!!!   too many iterations   !!!!!")
                print("!!!!!     increase maxit      !!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            else:
                # Changed by SW
                print("iteration limit exceeded")
            ier = 99
            break

        ######################################################################
        # coordinate search
        count = 0  # number of consecutive free steps
        k = -1  # current coordinate searched
        while 1:
            while count <= n:
                # find next free index (or next index if unfix)
                count = count + 1
                if k == n - 1:
                    k = -1
                k = k + 1
                if free[k] or unfix:
                    break
            if count > n:
                # complete sweep performed without fixing a new active bound
                break

            q = G[:, k]
            alpu = xu[k] - x[k]
            alpo = xo[k] - x[k]  # bounds on step

            # find step size
            [alp, lba, uba, ier] = getalp(alpu, alpo, g[k][0, 0], q[k][0, 0])
            if ier:
                x = np.zeros((n, 1))
                if lba:
                    x[k] = -1
                else:
                    x[k] = 1

                if prt:
                    gTp = g[k], pTGp = q[k], quot = pTGp / np.linalg.norm(G[:], np.inf)
                    print("minq: function unbounded below in coordinate direction")
                    print("      unbounded direction returned")
                    print("      possibly caused by roundoff")

                if prt > 1:
                    print("f(alp*x)=gam+gam1*alp+gam2*alp^2/2, where")
                    gam1 = c.T @ x
                    gam2 = x.T @ (G @ x)
                    ddd = diag(G)
                    min_diag_G = min(ddd)
                    max_diag_G = max(ddd)

                return x, fct, ier, nsub
            xnew = x[k] + alp
            if prt and nitref > 0:
                print(xnew, alp)

            if lba or xnew <= xu[k]:
                # lower bound active
                if prt > 2:
                    print(str(k) + " at lower bound")
                if alpu != 0:
                    x[k] = xu[k]
                    g = g + alpu[0] * q
                    count = 0
                free[k] = 0
            elif uba or xnew >= xo[k]:
                # upper bound active
                if prt > 2:
                    print(str(k) + " at upper bound")
                if alpo != 0:
                    x[k] = xo[k]
                    g = g + alpo[0] * q
                    count = 0
                free[k] = 0
            else:
                # no bound active
                if prt > 2:
                    print(str(k) + " free")
                if alp != 0:
                    if prt > 1 and not free[k]:
                        print("unfixstep ", x[k], alp)
                    x[k] = xnew

                    g = g + alp * q
                    free[k] = 1
        # end of coordinate search

        ######################################################################
        nfree = np.sum(free)
        if unfix and nfree_old == nfree:
            # in exact arithmetic, we are already optimal
            # recompute gradient for iterative refinement
            g = G * x + c
            nitref = nitref + 1
            if prt > 0:
                print("optimum found; iterative refinement tried")
        else:
            nitref = 0

        nfree_old = nfree
        gain_cs = fct - gam - 0.5 * x.T @ (c + g)
        improvement = gain_cs > 0 or not unfix

        if prt:
            # print (0,1) profile of free and return the number of nonzeros
            nfree = pr01("csrch ", free)

        if prt:
            print("gain_cs ", gain_cs)
        if prt > 2:
            X = x.T
            print("X ", X)

        # subspace search
        xx = x
        if not improvement or nitref > nitrefmax:
            # optimal point found - nothing done
            pass
        elif nitref > nitrefmax:
            # enough refinement steps - nothing done
            pass
        elif nfree == 0:
            # no free variables - no subspace step taken
            if prt > 0:
                print("no free variables - no subspace step taken")

            unfix = 1
        else:
            ######################
            # take a subspace step
            ######################
            zero_dir = 0
            end_subspace_search = 0

            nsub = nsub + 1

            if prt > 0:
                fct_cs = gam + 0.5 * x.T @ (c + g)
                formatstr = "*** nsub = %4.0f fct = %15.6e fct_cs = %15.6e\n"
                print(formatstr % (nsub, fct, fct_cs))

            # downdate factorization
            for j in np.where(free < K)[0]:  # list of newly active indices
                L, dd = ldldown(L.copy(), dd.copy(), j)
                K[j] = 0
                if prt > 10:
                    print("downdate")
                    print(np.nonzero(K)[0])

            # update factorization or find indefinite search direchtion
            definite = 1
            for j in np.where(free > K)[0]:  # list of newly freed indices
                # later: speed up the following by passing K to ldlup.m!
                p = np.zeros((n, 1))
                if n > 1:
                    p[K] = G[np.nonzero(K)[0], j]
                p[j] = G[j, j]
                L, dd, p = ldlup(L.copy(), dd.copy(), j, p.copy())
                definite = len(p) == 0
                if not definite:
                    if prt:
                        print("indefinite or illconditioned step")
                    break
                K[j] = 1
                if prt > 10:
                    print("update")
                    print(np.nonzero(K)[0])

            if definite:
                # find reduced Newton direction
                p = np.zeros((n, 1))
                p[K] = g[K]
                p = np.linalg.solve(-L.T, (np.linalg.solve(L, p) / dd))
                if prt > 10:
                    print("reduced Newton step")
                    print(np.nonzero(K)[0])

            if prt > 2:
                ipdb.set_trace()
                print("continue with c")

            # set tiny entries to zero
            p = (x + p) - x
            ind = np.where(p != 0)[0]
            if len(ind) == 0:
                # zero direction
                if prt:
                    print("zero direction")
                unfix = 1
                zero_dir = 1

            if not zero_dir:
                # find range of step sizes
                pp = p[ind]
                oo = (xo[ind] - x[ind]) / pp
                uu = (xu[ind] - x[ind]) / pp

                # alpu = np.max(np.vstack((oo[pp < 0], uu[pp > 0], -np.inf)))
                alpu = -np.inf
                if len(oo[pp < 0]):
                    tmp = np.max(oo[pp < 0])
                    alpu = max(tmp, alpu)
                if len(uu[pp > 0]):
                    tmp = np.max(uu[pp > 0])
                    alpu = max(tmp, alpu)

                # alpo = np.min(np.vstack((oo[pp > 0], uu[pp < 0], np.inf)))
                alpo = np.inf
                if len(oo[pp > 0]):
                    tmp = np.min(oo[pp > 0])
                    alpo = min(tmp, alpo)
                if len(uu[pp < 0]):
                    tmp = np.min(uu[pp < 0])
                    alpo = min(tmp, alpo)

                if alpo <= 0 or alpu >= 0:
                    sys.exit("programming error: no alp")

                # find step size
                gTp = g.T @ p
                agTp = abs(g).T @ abs(p)
                if abs(gTp) < 100 * eps * agTp:
                    # linear term consists of roundoff only
                    gTp = 0

                pTGp = p.T @ (G @ p)
                if convex:
                    pTGp = max(0, pTGp)

                if not definite and pTGp > 0:
                    if prt:
                        print("tiny pTGp = " + str(pTGp) + " set to zero")
                    pTGp = 0

                [alp, lba, uba, ier] = getalp(alpu, alpo, gTp, pTGp)

                if ier:
                    x = np.zeros((n, 1))
                    if lba:
                        x = -p
                    else:
                        x = p

                    if prt:
                        qg = gTp / agTp

                        ipdb.set_trace()
                        qG = pTGp / (np.linalg.norm(p, 1) ** 2 * np.linalg.norm(G[:], inf))
                        lam = eig(G)
                        lam1 = min(lam) / max(abs(lam))
                        print("minq: function unbounded below")
                        print("  unbounded subspace direction returned")
                        print("  possibly caused by roundoff")
                        print("  regularize G to avoid this!")

                    if prt > 1:
                        print("f(alp*x)=gam+gam1*alp+gam2*alp^2/2, where")
                        gam1 = c.T @ x
                        rel1 = gam1 / (abs(c).T @ abs(x))
                        gam2 = x.T @ (G @ x)
                        if convex:
                            gam2 = max(0, gam2)
                        rel2 = gam2 / (abs(x).T @ (abs(G) @ abs(x)))
                        ddd = np.diag(G)
                        min_diag_G = min(ddd)
                        max_diag_G = max(ddd)
                    end_subspace_search = 1

                if not end_subspace_search:
                    unfix = not (lba or uba)
                    # allow variables to be freed in csearch?

                    # update of xx
                    for k in range(len(ind)):
                        # avoid roundoff for active bounds
                        ik = ind[k]
                        if alp == uu[k]:
                            xx[ik] = xu[ik]
                            free[ik] = 0
                        elif alp == oo[k]:
                            xx[ik] = xo[ik]
                            free[ik] = 0
                        else:
                            xx[ik] = xx[ik] + alp * p[ik]

                        if abs(xx[ik]) == np.inf:
                            print(ik, alp, p[ik])
                            error("infinite xx in minq.m")

                    nfree = sum(free)
                    subdone = 1
                ######################
                # done with subspace step
                ######################

            if ier:
                return x, fct, ier, nsub

        if prt > 0:
            # print (0,1) profile of free and return the number of nonzeros
            nfree = pr01("ssrch ", free)
            print(" ")
            if unfix and nfree < n:
                print("bounds may be freed in next csearch")

    # end of main loop
    if prt > 0:
        print(fct)
        print("################## end of minq ###################")

    return x, fct, ier, nsub
