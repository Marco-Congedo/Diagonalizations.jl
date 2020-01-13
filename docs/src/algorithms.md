# Algorithms

The advanced diagonalization methods implemented in *Diagonalizations.jl*,
namely, [gMCA](@ref), [gCCA](@ref), [AJD](@ref) and [mAJD](@ref)
are solved by iterative algorithms, here also called *solvers*.

Solvers differ from one another in several fashions:
- the restriction they impose on the solution(s) (orthogonal/unitary or non-singular)
- whether they support complex data or only real data
- the diagonalization criterion they minimize
- the optimization method they employ
- the diagonalization methods they support
- their initialization

To date four solvers are implemented:

- **Orthogonal joint blind source separation** (OJOB: Congedo et al., 2012[🎓](@ref))
- **Non-orthogonal joint blind source separation** (NoJOB: Congedo et al., 2012[🎓](@ref))
- **Log-likelyhood** (LogLike: Pham, 2001[🎓](@ref))
- **Log-likelyhood Real** (LogLikeR: Pham, 2001[🎓](@ref))

Their main characteristics and domain of application are summarized in the following table:

| Algorithm  | Solution | Complex data | Criterion | Supported Methods |
|:-----------|:---------|:-------------|:----------|:--------|
| OJoB       | Orthogonal/Unitary| yes       | least-squares | gMCA, gCCA, AJD, mAJD |
| NoJoB      | non-singular| yes       | least-squares | gMCA, AJD, mAJD |
| LogLike    | non-singular| yes       | log-likelihood | AJD |
| LogLikeR   | non-singular| no        | log-likelihood | AJD |

## using solvers

A solver is to find ``m`` matrices, where ``m`` depends on the
sought diagonalization method (see [Overview](@ref)).
All algorithms by default are initialized by ``m`` identity
matrices, with the exception of *NoJoB*,
for which, following Congedo et al. (2011)[🎓](@ref), the ``m^{th}`` solution
matrix is initalized by the eigenvectors of

``\frac{1}{mk}\sum_{j=1}^m\sum_{l=1}^k C_{mjk}C_{mjk}^H``,

where ``C_{mjk}`` is the cross-covariance matrix between dataset
``m`` and dataset ``j`` for observation ``k``. When ``m=1``,
for instance in the case of *AJD*, those are just the ``k`` covariance matrices
to be jointly diagonalized.

A matrix (if ``m=1``) or a vector of matrices (if ``m>1``) can be passed with the `init` argument in order to initialize
the solver differently.

`tol` is the tolerance for convergence of the solver.
By default it is set to the square root of `Base.eps` of the nearest real type of the data input. If the solver encounters difficulties in converging, try setting `tol` in between 1e-6 and 1e-3.

Argument `maxiter` is the maximum number of iterations allowed to the solver. The default values are given in the following table:

| Algorithm  | max iterations |
|:-----------|:---------------|
| OJoB       | 1000 for real data, 3000 for complex data |
| NoJoB      | 1000 for real data, 3000 for complex data |
| Log-Like   | 60 for real data, 180 for complex data |
| Log-LikeR  | 40 (real data only) |

If the maximum number of iteration
is attained, a warning will be printed in the REPL.
In this case, try increasing `maxiter` and/or `tol`.
