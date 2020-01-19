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

To date six solvers are implemented:

- **Orthogonal joint blind source separation** (OJOB: Congedo et al., 2012[🎓](@ref))
- **Non-orthogonal joint blind source separation** (NoJOB: Congedo et al., 2012[🎓](@ref))
- **Log-likelyhood** (LogLike: Pham, 2001[🎓](@ref))
- **Log-likelyhood Real** (LogLikeR: Pham, 2001[🎓](@ref))
- **Joint Approximate Diagonalization of Eigenmatrices** (JADE: Cardoso and Souloumiac, 1996[🎓](@ref))
- **Gauss Approximate Joint Diagonalization** (GAJD, unpublished, from the author)

Their main characteristics and domain of application are summarized in the following table:

| Algorithm  | Solution | Complex data | Only PD | Criterion | Multi-threaded | Supported Methods |
|:---------|:---------|:-------------|:----------|:--------|:--------|:--------|
| OJoB     | Orthogonal/Unitary| yes | no| least-squares | yes | gMCA, gCCA, AJD, mAJD |
| NoJoB    | non-singular| yes       | no |least-squares | yes | gMCA, AJD, mAJD |
| LogLike  | non-singular| yes       | yes| log-likelihood | no | AJD |
| LogLikeR | non-singular| no        | yes| log-likelihood | no | AJD |
| JADE     | Orthogonal/Unitary| yes | no | least-squares | no | AJD |
| GAJD     | non-singular| no        | no | least-squares | no | AJD |


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

Argument `maxiter` is the maximum number of iterations allowed to the solver. The default values are given in the following table:

| Algorithm  | max iterations |
|:-----------|:---------------|
| OJoB       | 1000 for real data, 3000 for complex data |
| NoJoB      | 1000 for real data, 3000 for complex data |
| Log-Like   | 60 for real data, 180 for complex data |
| Log-LikeR  | 40 (real data only) |
| JADE       | 60 for real data, 180 for complex data |
| GAJD       | 120 (real data only) |


If the maximum number of iteration
is attained, a warning will be printed in the REPL.
In this case, try increasing `maxiter` and/or `tol`.

## Multi-threading

Besides being the most versitile (they support all methods),
the OJoB and NoJOB algorithms also supports multi-threading.
The methods' constructors feature the `threaded` optional keyword argument,
which is true by default. If `threaded` is true and ``n>x`` and ``x>1``,
where ``x`` is the number of threads Julia is instructed to use and ``n``
is the dimension of the input matrices, the algorithms run in multi-threaded
mode paralellising several comptations over ``n``.

Besides being optionally multi-threaded, these algorithms heavely use BLAS.
Before running these methods you may want to set the number of threades
Julia is instructed to use to the number of logical CPUs of your machine
and set `BLAS.set_num_threads(Sys.CPU_THREADS)`. If you are `using` any of the
package written by the author, this is done automatically. See
[these notes](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Threads-1).

## Speed

Indicatively, the tables here below reports the median execution
time in ms required to converge on a regular laptop computer with a tolerence
set to 1e-6 in the case of a set of ``k`` exactly diagonalizable real matrices
of dimension ``n·n``:

| n, k       | GAJD    |  NoJoB  | NoJoB single-thread |LogLike |
|:-----------|:--------|:--------|:--------------------|:-------|
| 5, 5       | 3       | N.A.    | 6                   | 0.5    |
| 5, 50      | 0.73    | N.A.    | 8.7                 | 1.2    |
| 5, 500     | 1.85    | N.A.    | 49                  | 8.2    |
| 5, 5000    | 21      | N.A.    | 524                 | 80     |
|            |         |         |                     |        |
| 25, 5      | 256     | 606     | 1080                | 47     |
| 25, 50     | 147     | 63      | 112                 | 227    |
| 25, 500    | 356     | 355     | 602                 | 1917   |
| 25, 5000   | 4886    | 3955    | 5306                | 36963  |
|            |         |         |                     |        |
| 50, 5      | 1584    | 8576    | 4095                | 310    |
| 50, 50     | 1056    | 575     | 455                 | 2180   |
| 50, 500    | 4337    | 1109    | 2286                | 21100  |


| n, k       | OJoB    |  OJoB single-thread | JADE   |
|:-----------|:--------|:--------------------|:-------|
| 5, 5       | N.A     | 0.3                 | 0.34   |
| 5, 50      | N.A     | 0.91                | 0.4    |
| 5, 500     | N.A     | 6.3                 | 2.3    |
| 5, 5000    | N.A     | 58                  | 24     |
|            |         |                     |        |
| 25, 5      | 2       | 3                   | 17.1   |
| 25, 50     | 5.4     | 7.3                 | 50.2   |
| 25, 500    | 36.5    | 46.1                | 452    |
| 25, 5000   | 349     | 464                 | 9937   |
|            |         |                     |        |
| 50, 5      | 9.6     | 11.2                | 106    |
| 50, 50     | 17.6    | 26.39               | 683    |
| 50, 500    | 102     | 192                 | 7315   |
| 50, 5000   | 1214    | 2045                | 136469 |

Among the non-singular AJD algorithms
the fastest algorithm per iteration is by far GAJD.
The fastest convergence is achieved by LogLike.
The GAJD algorithm scales well over the number of input matrices (``k``).
NoJoB scales very well over the dimension of the input matrices (``n``).
For small ``n`` and small ``k``, logLike performs better.
For small ``n`` and ``k`` large, GAJD performs better.

Among the orthogonal/unitary AJD algorithms, OJoB outperforms JADE
in all situations, but for ``n`` small.

In a summary, OJoB and NoJoB offers excellent speed in the most common
situations. The only cases where they do not perform best is when ``k<n``
(because of their slow convergence with such settings)
or when either ``k``  or ``n`` are small.
The following table reports suggestions on how to choose
an algorithm depending on ``n`` and ``k``:

| n     | k       | Orhogonal/unitary | non-singular |
|:------|:--------|:------------------|:-------------|
| small | small   | LogLike           | JADE         |
| small | large   | GAJD              | JADE         |
| large | small   | LogLike           | OJoB         |
| large | large   | NoJOB             | OJoB         |
