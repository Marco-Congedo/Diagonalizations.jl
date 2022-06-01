# Algorithms

The advanced diagonalization methods implemented in *Diagonalizations.jl*,
namely, [gMCA](@ref), [gCCA](@ref), [AJD](@ref) and [mAJD](@ref)
are solved by iterative algorithms, here also called *solvers*.

Solvers differ from one another in several fashions:
- the restriction they impose on the solution(s) (orthogonal/unitary or non-singular)
- whether they support complex data or only real data
- whether they take as input only positive-definite (PD) matrices or all symmetric/Hermitian matrices
- the diagonalization criterion they minimize
- the optimization method they employ
- the diagonalization methods they support
- their initialization
- whether they support multi-threading or not

To date nine solvers are implemented:

- **Orthogonal joint blind source separation** (OJOB: Congedo et al., 2012[🎓](@ref))
- **Non-orthogonal joint blind source separation** (NoJOB: Congedo et al., 2012[🎓](@ref))
- **Log-likelihood** (LogLike: Pham, 2001[🎓](@ref))
- **Log-likelihood Real** (LogLikeR: Pham, 2001[🎓](@ref))
- **Quasi-Newton Log-likelihood** (QNLogLike: Ablin et **al.**, 2019[🎓](@ref))
- **Joint Approximate Diagonalization of Eigenmatrices** (JADE: Cardoso and Souloumiac, 1996[🎓](@ref))
- **Joint Approximate Diagonalization of Eigenmatrices max** (JADEmax: Usevich, Li and Comon, 2020[🎓](@ref))
- **Gauss Approximate Joint Diagonalization** (GAJD, unpublished, from the author)
- **Gauss Log-Likelihood** (GLogLike, unpublished, from the author, still experimental)


Their main characteristics and domain of application are summarized in the following table:

| Algorithm  | Solution | Complex data | Only PD | Criterion | Multi-threaded | Supported Methods |
|:---------|:---------|:-------------|:----------|:--------|:--------|:--------|
| OJoB     | Orthogonal/Unitary| yes | no| least-squares | yes | gMCA, gCCA, AJD, mAJD |
| NoJoB    | non-singular| yes       | no |least-squares | yes | gMCA, AJD, mAJD |
| LogLike  | non-singular| yes       | yes| log-likelihood | no | AJD |
| LogLikeR | non-singular| no        | yes| log-likelihood | no | AJD |
| QNLogLike| non-singular| no        | yes| log-likelihood | no | AJD |
| JADE     | Orthogonal/Unitary| yes | no | least-squares | no | AJD |
| JADEmax  | Orthogonal/Unitary| yes | no | least-squares | no | AJD |
| GAJD     | non-singular| no        | no | least-squares | no | AJD |
| GLogLike | non-singular| no        | yes | log-likelihood | no | AJD |



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
the solver differently. Note that for *LogLike* and *LogLikeR* algorithms
the actual approximate joint diagonalizer will be given by `init*B`, where `B` is the output of the algorithms.

`tol` is the tolerance for convergence of the solver.
By default it is set to the square root of `Base.eps` of the nearest real type of the data input. If the solver encounters difficulties in converging, try setting `tol` in between 1e-6 and 1e-3.

Argument `maxiter` is the maximum number of iterations allowed to the solver.
For all algorithms, by default `maxiter` is set to 1000 for real data and to
3000 for complex data.

If the maximum number of iteration
is attained, a warning is printed in the REPL.
In this case, try increasing `maxiter` and/or `tol`.

## Multi-threading

The OJoB, NoJOB and QNLogLike algorithms supports multi-threading.
The methods' constructors feature the `threaded` optional keyword argument,
which is true by default.
For OJoB and NoJoB, the algorithms run in multi-threaded
mode paralellising several comptations over the dimension of the
input matrices ``n``, if `threaded` is true, ``2n>x`` and ``x>1``,
where ``x`` is the number of threads Julia is instructed to use.
For QNLogLike, the algorithms run in multi-threaded
mode paralellising several computations over the number of matrices ``k``,
if `threaded` is true, ``2k>x`` and ``x>1``.

Before running these methods you may want to set the number of threades
Julia is instructed to use to the number of logical CPUs of your machine.
Besides being optionally multi-threaded, OJoB and NoJoB algorithms heavely use BLAS.
Before using these methods you may want to
set `BLAS.set_num_threads(Sys.CPU_THREADS)` to the number of logical CPUs
of your machine. If you are `using` any of the
package written by the author, all this is done automatically. See
[these notes](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Threads-1).
