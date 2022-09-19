# MINQ
A Matlab Program for General Definite and Bound Constrained Indefinite Quadratic Programming

This repository seeks to mirror and archive software hosted at https://arnold-neumaier.at/software/minq/index.html
===

MINQ is a Matlab program for bound constrained indefinite quadratic programming based on rank 1 modifications. It finds a local optimizer of the optimization problem

     min    fct = c^T x + 0.5 x^T G x 
     s.t.   x in [xu,xo]   (componentwise)

where G is a symmetric n x n matrix, not necessarily semidefinite, and infinite bounds are allowed. (If G is positive semidefinite, any local optimizer is global, so it finds the global optimum.)

The method consists of a combination of coordinate searches and subspace minimization steps (safeguarded equality-constrained QP-steps when the coordinate searches no longer change the active set). Rank 1 updates (in a format suited for both the dense and sparse case) are used to keep the linear algebra cheap.

In the sparse case, it is assumed that the variables are already ordered such that the symbolic factorization of G is sparse.

MINQ can also be used for general definite quadratic programming since the dual is simply constrained. Based on this, the Matlab program MINQDEF solves the optimization problem

    min    fct = c^T x + 0.5 x^T G x 
    s.t.   A x >= b, with equality at indices with eq=1

where G is a positive definite symmetric n x n matrix, and the Matlab program MINQSEP solves the same problem with definite diagonal G. (In the sparse case, it is now assumed that the variables are already ordered such that the symbolic factorizations of G and AG^(-1)A^T are sparse.)

As an application, a robust least squares solver RLS is included. RLS solves a linear least squares problem

    min    ||Ax-b||_2^2  
    s.t.   |x-x0|<=r

If r has the default value, RLS it can be used to regularize least squares problems. In ill-conditioned cases, it yields much better approximate solutions than x=A\b, though at a time penalty.

All versions of MINQ are licenced. The required m-files can be downloaded as the gzipped tar file minq8.tar.gz (for Matlab 8; 255K). The individual files may be viewed from the minq8 directory. The theory behind MINQ8 is described here.

The Matlab 5 version minq5.tar.gz (for Matlab 5; 48K) and the Matlab 4 version (minq.tar.gz; 43K; no sparse facilities, no RLS) are no longer supported.

Source: http://arnold-neumaier.at/software/minq/

If you want to reference MINQ, you may use the following format:

W. Huyer and A. Neumaier, MINQ8 - General Definite and Bound Constrained Indefinite Quadratic Programming, Manuscript (2017). http://arnold-neumaier.at/software/minq/

For the global optimization of indefinite quadratic programs, one may try to use MINQ repeatedly with multiple random starting points. Alternatively, try [not developed by us] SCIP (free for academics) or Gurobi (commercial).

Contact the lead author Arnold Neumaier at (Arnold.Neumaier@univie.ac.at) or via (http://arnold-neumaier.at)
