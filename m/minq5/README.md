This is the Readme included with minqv5; see also the primary MINQ Readme at [../../README.md](https://github.com/POptUS/MINQ/tree/main)

MINQ5 bound constrained quadratic optimization

MINQ minimizes an affine quadratic form subject to simple bounds,
using coordinate searches and reduced subspace minimizations
```
   min    fct = gamma + c^T x + 0.5 x^T G x
   s.t.   x in [xu,xo]    % xu<=xo is assumed
```
where G is a symmetric n x n matrix, not necessarily definite
(if G is indefinite, only a local minimum is found).
If G is sparse, it is assumed that the ordering is such that
a sparse modified Cholesky factorization is feasible.

Main routines:
- `minqdef.m`		general definite QP solver   (calls minq.m)
- `minqsep.m`		definite separable QP solver (calls minq.m)
- `minq.m`		bound constrained QP solver
- `minqsub.m`		patch for minq.m containing the subspace search
- `minq_test.m`		small test program for minq.m
- `rls.m`			robust least squares solver
- `rls_test.m`		small test program for rls.m

`minq.m` calls the following subroutines:
- `getalp.m`		exact quadratic line search
- `ldldown.m`		LDL^T factorization downdate
- `ldlrk1.m`		LDL^T factorization rank 1 change
- `ldlup.m`		LDL^T factorization update
- `ldltest.m`		LDL^T factorization update
- `pr01.m`		print characteristic vecor of activities

`minqsw.m` is a slight modification of `minq.m` that:
- Slightly increases `maxit`
- Does not print to screen when `maxit` is exceeded
- Uses the "short circuiting" versions of logical and `&&` and logical or `||`
