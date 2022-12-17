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

## Versions
The current version is minq8.
- The m-file version of `minq8` is available:
  - as the gzipped tar file [minq8.tar.gz (for Matlab 8; 255K)](https://arnold-neumaier.at/software/minq/minq8.tar.gz).
  - in this repository from the [m/minq8](https://github.com/POptUS/MINQ/tree/main/m/minq8) directory.
- The m-file version of `minq5` is available:
  - as the gzipped tar file [minq5.tar.gz (for Matlab 5; 48K)](https://arnold-neumaier.at/software/minq/minq5.tar.gz).
  - in this repository from the [m/minq5](https://github.com/POptUS/MINQ/tree/main/m/minq5) directory.
- The python version of `minq5` is available:
  - in this repository from the [py/minq5](https://github.com/POptUS/MINQ/tree/main/py/minq5) directory.

## References
The primary reference for the original source is: http://arnold-neumaier.at/software/minq/

The theory behind MINQ8 is described in [this paper](http://arnold-neumaier.at/ms/minq8.pdf).

To reference MINQ, please cite [doi:10.1007/s10589-017-9949-y](https://doi.org/10.1007/s10589-017-9949-y):

    W. Huyer and A. Neumaier, 
    MINQ8 - General Definite and Bound Constrained Indefinite Quadratic Programming, 
    Computational Optimization and Applications 69 (2018), 351--381.

For the global optimization of indefinite quadratic programs, one may try to use MINQ repeatedly with multiple random starting points. Alternatively, try (*not developed by the minq team*) [SCIP](http://scip.zib.de/) (free for academics) or [Gurobi](http://www.gurobi.com/) (commercial).

## License
All versions of MINQ are licensed and open source, with the particular form of license for each version contained in the top-level subdirectories of [m/](/m/) and [py/](/py/).  If such a subdirectory does not contain a LICENSE file, then it is automatically licensed as described in the otherwise encompassing POPTUS [LICENSE](/LICENSE).


## Resources
Contact the lead author Arnold Neumaier at (Arnold.Neumaier@univie.ac.at) or via (http://arnold-neumaier.at).

Contributions to this repository are welcome, please take a look at [CONTRIBUTING](/CONTRIBUTING.rst).

To seek support or report issues with this repository, e-mail:

 * ``poptus@mcs.anl.gov``

