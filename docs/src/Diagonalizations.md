# Diagonalizations.jl

## dependencies

| standard Julia packages |     external packages    |
|:-----------------------:|:-----------------------:|
| [LinearAlgebra](https://bit.ly/2W5Wq8W) |  [CovarianceEstimation](https://github.com/mateuszbaran/CovarianceEstimation.jl)|
| [Statistics](https://bit.ly/2Oem3li) | [PosDefManifold](https://github.com/Marco-Congedo/PosDefManifold.jl) |
| [StatsBase](https://github.com/JuliaStats/StatsBase.jl) |  |

## types

```
abstract type LinearFilters end
```

is the abstract type for all filters.

All filters are instances of the following immutable structure:

### LinearFilter

```
struct LinearFilter <: LinearFilters
   F     :: AbstractArray
   iF    :: AbstractArray
   D     :: Diagonal
   eVar  :: Float64
   ev    :: Vec
   arev  :: Vec
   name  :: String
```

**Fields**:

`.F`: for simple filters ([PCA](@ref), [Whitening](@ref), [CSP](@ref),
[AJD](@ref)), this is an ``n⋅p`` matrix, where ``n`` is the number
of variables in the data the filter has been derived from and ``p`` is the
[subspace dimension](@ref). For composite filters this is a vector of ``m``
of such ``n⋅p`` matrices, where
- ``m=2`` for [MCA](@ref), [CCA](@ref) and [CSTP](@ref) filters
- ``m≥2`` for [gMCA](@ref), [gCCA](@ref) and [mAJD](@ref) filters.

`.iF`: the ``p⋅n`` left-inverse of the filter(s) in `.F`, that is,
multiplying on the right all matrices in `.iF` by the corresponding
matrices in `.F` yields the ``p⋅p`` identity matrix.   

The following fields are populated by default, but may be set altogether to
`nothing` by all constructors using the `simple` optional keyword
argument:

`.D`: a ``p⋅p`` [Diagonal](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Diagonal)
matrix holding the (generalized) eigenvalues or singular values, depending on the
filter, of the last diagonalization that has been used to derive the filter.
Since `.D` is in diagonal form, this matrix can be used directly in
algebraic computations. For example, if `a` is a PCA filter
computed from covariance matric ``C`` and ``p=n``, then
`C≈a.F*a.D*a.iF` is true.
In a similar way, if `b` is a MCA filter
computed from cross-covariance matric ``C_{xy}`` and ``p=n``, then
`C_xy≈b.F[1]*b.D*b.F[2]'` is true.

`.eVar`: the actual explained variance or variance ratio of the filter(s)
in `.F`. This depends on the filter, see the documentation of each filter
for details.

`.ev`: a vector holding all the `n` (generalized) eigenvalues or singular values,
depending on the filter, of the last diagonalization that has been used
to derive the filter. If ``p=n``, this is a vector holding the diagonal
of `.D`, otherwise `.D` holds in diagonal form only the first
`p` elements of `.ev`.

`arev.`: a vector holding an accumulated regularized function of `.ev` used
to find the [subspace dimension](@ref). This depends on the filter,
see the documentation of each filter for details.

## LinearFilter methods

The [LinearFilter](@ref) structure supports the following methods:

```
size(f::LF)
```
Return the size of `f.F` if it is a matrix, an iterator
over the sizes of all matrices in `f.F` if it is a vector
of matrices.

```
length(f::LF)
```
Return 1 if `f.F` is a matrix, the number of matrices in
`f.F` if it is a vector of matrices. Referring to Table 1 and Fig. 1
(see [Overview](@ref)), this is the number of datasets ``m``.

```
eltype(f::LF)
```
Return the element type of the matrix(ces) in `f.F`.

```
==(f::LF, g::LF), ≈(f::LF, g::LF)
```
Return `true` if all fields of LinearFilter `f` and `g` are equivalent, `false`
otherwise. All but the `.F` and `.iF` fields are requested to be equal,
where for vector fields approximate equality is ascetrained using the `≈`
function. For the equivalence of the matrices in fields `.F` and `.iF`,
it is requested that the mean [`spForm`](@ref) index of the matrices
in `f.iF` times the corresponding matrices in `g.F` and
of the matrices in `g.iF` times the corresponding matrices in `f.F`
is smaller then 0.05.

```
≠(f::LF, g::LF), ≉ (f::LF, g::LF)
```
The negation of `==`.

```
function cut(f::LinearFilter, p::Int64)
```
Create another [LinearFilter](@ref) object with a smaller
[subspace dimension](@ref) given by argument `p`.
This applies to the matrix(ces) in fields `.F`, `.iF` and `.D`.
All other fields remain the same.


## data input

All filter constructors take as input either data matrices
or covariance matrices. Covariance matrices must be flagged by Julia as either
[Symmetric](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Symmetric),
if they are real, or
[Hermitian](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Hermitian),
if they are real or complex, e.g.,

```
X=randn(100, 30)
C=(X'*X)/100
p=pca(Symmetric(C))
# p=pca(C) will throw an ArgumentError
```

the above call to the pca constructor is equivalent to

```
X=randn(100, 30)
p=pca(X)
```

Some methods take as input a vector of Hermitian matrices,
of type [ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1),
see [typecasting matrices](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#typecasting-matrices-1).

Using **real data matrices** as input, shrinked covariance
matrix estimators can be used for several filters
(e.g., [PCA](@ref), [Whitening](@ref), [CSP](@ref), [CSTP](@ref)).
See here below.

## covariance matrix estimations

By default, when data matrices are used as [data input](@ref),
*Diagonalizations.jl* computes covariance matrices
along the larger dimension of the data matrices. That is, for ``r⋅c``
data matrix ``X``, if ``r>c`` ``\frac{1}{r}X^{T}X`` is computed, otherwise
``\frac{1}{c}XX^{T}`` is computed.
Hence, the default behavior assumes that the number of observations
is larger than the number of variables, as it is usually appropriate.
Covariance matrices can be computed along a specific dimension
using optional keyword argument `dims`,
as in [StatsBase](https://github.com/JuliaStats/StatsBase.jl).

Many filter constructors allow to use *shrinked* covariance matrix estimations (only for real data) by means of the
[CovarianceEstimation](https://github.com/mateuszbaran/CovarianceEstimation.jl)
package.   
The following constants are provided to allow quick access
to the most popular choices among the many estimators implemented
therein:

```
SCM=SimpleCovariance()
```

This is the *default* for all constructors and corresponds to the standard
*sample covariance matrix (SCM)* estimation.

```
LShrLW=LShr(ConstantCorrelation())
```

This corresponds to the SCM estimator shrinked by the linear method
of Ledoit and Wolf (2004) [🎓](@ref).

```
LShr=LinearShrinkage
```

This is a shortcut for requesting other types of linear Shrinkage.
See the [CovarianceEstimation](https://github.com/mateuszbaran/CovarianceEstimation.jl)
package for details.

```
NShrLW=AnalyticalNonlinearShrinkage()
```

This corresponds to the SCM estimator shrinked by the analytical
non-linear method of Ledoit and Wolf (2018) [🎓](@ref).

Also, several filter constructors allow to use a `mean` and `w`
keyword arguments:

`mean` can be used to subtract the mean from the variables of data matrices
(e.g., data matrix `X`).
- if `mean=0`, the mean will not be subtracted (default);
- if `mean=nothing`, the mean will be computed and subtracted;
- `mean` can be a vector of means to be subtracted:
  * it must have length=size(`X`, 2) if `dims=1`, length=size(`X`, 2) if `dims=2`;
- `mean` can also be a matrix of means to be subtracted:
  * it must have size=(1, size(`X`, 2)) if `dims=1`, size=(size(`X`, 1), 1) if `dims=2`.

!!! note "Nota Bene"
    For filter constructors taking as input sets of data matrices,
    the `mean` argument can be set only to `0` (default) or `nothing`.

`w` can be `nothing` (default) or a
[StatsBase.AbstractWeights](https://github.com/JuliaStats/StatsBase.jl/blob/master/docs/src/weights.md) object to weights the samples of data matrices
(e.g., data matrix `X`). It must have length=size(`X`, `dims`),
where `dims` by default is set to the larger dimesnion of `X`.
For some constructors `w` can also be a function. This is
documented in the concerned constructors.

Note that if several data matrices can be given as input to filter constructors,
for example `X`, `Y`,..., then you will find arguments named
such as `meanX`, `meanY`,... and `wX`, `wY`,... to differentiate
the mean ad weights of the several input data matrices.

**Examples**:

```
using Diagonalizations

X=randn(100, 30) # X is 'tall'
p=pca(X)
```

The call here above uses the default SCM estimator and computes the PCA
from the ``30⋅30`` covariance matrix ``\frac{1}{100}X^{T}X``.
The 'filter' `p.F` is ``30⋅p``, where ``p`` is the [subspace dimension](@ref). For complex data the call is the same:

```
Xc=randn(ComplexF64, 100, 30)
pc=pca(Xc)
```


This call

```
p=pca(X; covEst=LShrLW)
```

uses the linear shrinked estimator of Ledoit and Wolf (2004).

The call

```
p=pca(X; dims=2)
```

uses the default SCM estimator and compute the PCA
from the ``100⋅100`` (rank-deficient) covariance matrix
``\frac{1}{30}XX^{T}``. The 'filter' `p.F` is in this case ``100⋅p``.


## mean covariance matrix estimations

Some filters can take as input data a set of data matrices
(a vector of matrices). In this case a covariance matrix is estimated
for each data matrix in the set and then a mean of these covariance matrices
is estimated.

If the covariance matrices are actually *cross-covariance* matrices,
no option is provided and the usual arithmetic mean is computed.
If they are `Symmetric`
or `Hermitian` covariance matrices, the
[mean](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#Statistics.mean) function of package
[PosDefManifold.jl](https://github.com/Marco-Congedo/PosDefManifold.jl)
is used, since those matrices may be positive definite
by construction, hence a mean using a metric acting on the Riemannian
manifold of positive definite matrices may be used.

The constructors using this feature employ the following optional
keyword arguments for regulating the computation of the mean:

`metric`: the metric used to compute the mean. *PosDefManifold.jl*
supports 10 [metrics](https://marco-congedo.github.io/PosDefManifold.jl/dev/introToRiemannianGeometry/#metrics-1), nine of which can be used here
(all but the `VonNeumann` metric). Of particular interest are
the following
1. `Fisher`: the natural affine invariant metric, possessing all good properties of a mean.
2. `logEuclidean`, `Jeffrey `: computationally cheaper alternatives to 1), but not possessing all good properties of a mean.
3. `invEuclidean`: leading to the matrix harmonic mean.
4. `Euclidean`: (default) leading to the usual matrix arithmetic mean, thus applying also if the input matrices are not positive-definite.
5. `Wasserstein`: a metric widely adopted in statistics, optimal transport and quantum physics (also known as Bures-Hellinger), also applying if the input matrices are not positive-definite.

Note that, since by default covariance matrices are computed along
the larger dimension of data matrices, the covariance matrices
will be positive definite as long as the number of observations
is sufficiently larger then the number of variables.

See the documentation of the [mean](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#Statistics.mean) function for arguments `w`, `✓w`, `init`,
`tol` and `verbose`. Note that the name of the `w` (and `init`) arguments may actually be `wCx`, `w₁` (`initCx`, `init₁`) and similar. This is to allow using them for different data matrices used to construct the filter.   

## subspace dimension

For a 'wide' ``n⋅t`` data matrix ``X``, where ``n`` is
the number of variables and ``t>n`` the number of samples,
with ``n⋅n`` covariance matrix ``C``, the transformed data
is given by ``F^{H}X`` and the transformed covariance matrix by ``F^{H}CF``.

In the above, ``F`` is the ``n⋅p`` filter matrix and
``p`` is named the *subspace dimension*.
The data filtered in this subspace is given by

``\widetilde{X}=F^{-H}F^{H}X``

and the filtered covariance by

``\widetilde{C}=F^{-H}F^{H}CFF^{-1}``.

If the matrix ``X`` is available in the 'tall' form ``t⋅n`` (default),
the tranformed data is given by ``XF`` and the data filtered
in the subspace is given by

``\widetilde{X}=XFF^{-1}``.

The expressions for the transformed and filtered covariance matrix
are the same as before.

For all filters *Diagonalizations.jl* allows to set the subspace dimension
``p`` using the `eVar` and `eVarMeth` optional keyword arguments.
Ultimately, ``p`` will be en integer ``∈[1, n]`` representing the subspace
dimension. The user may set ``p``:

- **manually**,

 either setting `eVar` explicitly to an integer ``∈[1, n]``, or
 setting `eVar` to the desired explained variance in the subspace filtered
 data, as a float ``∈(0, 1]``, where ``1`` corresponds to the total variance.

- **automatically** (default),

 according to the `.arev` (accumulated regularized eigenvalues) vector
 that is computed by the filter constructors. This vector is
 non-decreasing and the last element is always ``1.0``. The way it is
 computed depends on the filter. Please refer to the documentation of
 each filter for details on how the `.arev` vector is defined.

When `eVar` is given as a float or when such float ia allowed to be chosen
automatically (default), the function passed as the `eVarMeth` argument
determines the subspace dimension ``p`` so as to explain an amount of
variance as close as possible to the desierd `eVar`.
In fact `.arev` holds only ``n`` discrete possible values of explained variance.
Therefore, the `.arev` vector is passed to the `eVarMeth` function.
By default, `eVarMeth` is set to the Julia standard
[searchsortedfirst](https://docs.julialang.org/en/v1/base/sort/#Base.Sort.searchsortedfirst) function, which will select the smallest ``p``
allowing at least `eVar` explained variance. This amounts
to rounding up the desired `eVar` variance.
Another useful choice is the Julia standard
[searchsortedlast](https://docs.julialang.org/en/v1/base/sort/#Base.Sort.searchsortedlast) function, which will select the largest ``p``
allowing at most `eVar` explained variance.
This amounts to rounding down the desired `eVar` variance.  

You can pass a user-defined function as `eVarMeth`.
The function you define will take the `.arev` vector computed by the
filter constructor as input and will return an integer,
which will be automatically clamped to be ``∈[1, n]``.

Note that once the filter has been constructed, its `.eVar` field
will hold the actual explained variance, not the desired one
that has been passed to the constructor using the `eVar` argument.

Note also that for some filter constructors you will find the
`eVar` optional keyword argument and also other arguments with
simialr name, such as `eVarCx` and `eVarCy`. These arguments act in a similar
way as the main `eVar` argument, but apply to determine the subspace dimension
of intermediate diagonalization procedures, typically, pre-whitening procedures. See also [notation & nomenclature](@ref) and [covariance matrix estimations](@ref).

## scale and permutation

Let ``F`` be a diagonalizer of matrix ``C``, i.e.,

``F^{H}CF=Λ``

with ``Λ`` a diagonal matrix. Let ``P`` a
[permutation matrix](https://en.wikipedia.org/wiki/Permutation_matrix)
and ``O`` a diagonal matrix whose entries are either ``1`` or ``-1``.
It is easy to verify then that any matrix ``FPO`` is an *identical* diagonalizer
of ``C``, since ``OP^{H}F^{H}CFPO=Λ``.
This implies that the filter matrices found by an exact diagonalization
procedures are arbitrary up to **sign and permutation** of their columns.

If ``D`` is a generic diagonal matrix, it is easy to verify then
that any matrix ``FPD`` is an **equivalent** diagonalizer of ``C``
(Belouchrani et al., 1997 [🎓](@ref)),
since ``DP^{H}F^{H}CFPD`` is also diagonal,
albeit different from ``Λ``.
This implies that there exist infinite equivalent exact diagonalizers
and that the solution is arbitrary up to **scale and permutation**
of the columns.
Of course, the scale ambiguity implies the sign ambiguity,
but not vice versa.  
All exact diagonalization procedures
implicitly constraint the solution to find ``P`` and ``D`` such that
``Λ`` possesses a desired property. For example, in principal component analysis
the elements of ``Λ`` are the maximum values that can be attained constraining  ``F`` to be orthogonal.

The same ambiguity applies to **approximate joint diagonalization**.
Let ``F`` be an approximate joint diagonalizer of matrix set ``{C_1,...,C_K}``,
i.e.,

``F^{H}C_lF≈Λ_k`` for ``l∈[1, k]``

and let ``D`` be a diagonal matrix, then it is easy to verify that any
matrix ``FPD`` is an *equivalent* approximate joint diagonalizer of the set
``C``. To check if two diagonalizers are equivaent, you can use the
[`spForm`](@ref) function.

## notation & nomenclature

Throughout the code and documentation of this package the following
notation is followed:

- **scalars** and **vectors** are denoted using lower-case letters, e.g., `x`, `y`,
- **matrices** using upper case letters, e.g., `X`, `Y`,
- **sets (vectors) of matrices** using bold upper-case letters, e.g., `𝐗`, `𝐘`.
- superscripts *H* and *T* denote matrix complex conjugate-transpose and transpose.

The following nomenclature is used consistently:

- ``X``, ``Y``: **data matrices**
- ``𝐗``, ``𝐘``: **vectors of data matrices**
- ``C``: a **covariance matrix**
- ``𝐂``: a **vector of covariance matrices**
- ``C_x``: the **covariance matrix** of data matrix ``X``
- ``C_{xy}``: the **cross-covariance matrix** of ``X`` and ``Y``
- ``U``, ``V``: **orthogonal matrices** of eigenvectors or the left and right singular vectors
- ``λ``: **vector** of eigenvalues, singular values or a function thereof
- ``Λ``: **diagonal matrix** of eigenvalues, singular values or a function thereof
- ``B``, ``F``: **non-singular matrices**

In the examples, bold upper-case letters are replaced by
upper case letters in order to allow reading in the REPL.

## acronyms

- AJD: Approximate Joint Diagonalization (Cardoso & Souloumiac, 1996; Flury & Gautschi, 1986)
- AJEVD: Approximate Joint Eigenvalue-Eigenvector Decomposition
- AJSVD: Approximate Joint Singular Value Decomposition (Congedo et al., 2011)
- AMUSE: Algorithm for Multiple Source Extraction (Molgedey & Schuster, 1994; Tong et al., 1991)
- BSS: Blind Source Separation
- CCA: Canonical Correlation Analysis (Hotelling, 1936)
- CSP: Common Spatial Pattern (Fukunaga, 1990)
- CSTP: Common Spatio-Temporal Pattern (Congedo et al., 2016)
- EEG: Electroencephalography
- ERP: Event-Related Potentials
- FOBI: Fourth-Order Blind Identification (Cardoso, 1989)
- gCCA: generalized CCA
- gMCA: generalized MCA
- JADE: Joint Diagonalization of Eigenmatrices (Cardoso & Souloumiac, 1993)
- LShrLW: Linear Shrinkage of Ledoit and Wolf (2004)
- NShrLW: Non-linear Shrinkage of Ledoit and Wolf (2018)
- MCA: Maximum Covariance Analysis
- NoJoB: Non-orthogonal Joint BSS (Congedo et al., 2012)
- OJoB: Orthogonal Joint BSS (Congedo et al., 2012)
- PCA: Principal Component Analysis (Pearson, 1901)
- SCM: Sample Covariance Matrix
- SOBI: Second-Order Blind Identification (Belouchrani et al., 1997)

For the references see [🎓](@ref).
