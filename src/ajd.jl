#   Unit "ajd.jl" of the Diagonalization.jl Package for Julia language
#
#   MIT License
#   Copyright (c) 2019,
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements Approximate Joint Diagonalization.

"""
```
(1)
function ajd(𝐂::ℍVector;
             trace1    :: Bool   = false,
             w         :: Union{Tw, Function} = ○,
          algorithm :: Symbol = :NoJoB,
          preWhite  :: Bool   = false,
          sort      :: Bool   = true,
          init      :: Mato   = ○,
          tol       :: Real   = 0.,
          maxiter   :: Int    = 2000,
          verbose   :: Bool   = false,
        eVar     :: TeVaro    = _minDim(𝐂),
        eVarC    :: TeVaro    = ○,
        eVarMeth :: Function  = searchsortedfirst,
        simple   :: Bool      = false)

(2)
function ajd(𝐗::VecMat;
             covEst     :: StatsBase.CovarianceEstimator = SCM,
             dims       :: Into = ○,
             meanX      :: Into = 0,
          trace1     :: Bool = false,
          w          :: Union{Tw, Function} = ○,
       algorithm :: Symbol = :NoJoB,
       preWhite  :: Bool = false,
       sort      :: Bool = true,
       init      :: Mato = ○,
       tol       :: Real = 0.,
       maxiter   :: Int  = 2000,
       verbose   :: Bool = false,
     eVar     :: TeVaro    = _minDim(𝐗),
     eVarC    :: TeVaro    = ○,
     eVarMeth :: Function  = searchsortedfirst,
     simple   :: Bool      = false)

```

Return a [LinearFilter](@ref) object:

**(1) Approximate joint diagonalization** of the set of ``k``
symmetric or Hermitian matrices `𝐂`, of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1)
using the given solving `algorithm` (*NoJoB* by default).

If `trace1` is true, all matrices in the set
`𝐂` are normalized so as to have trace equal to 1.
It is false by default.

if `w` is a `StatsBase.AbstractWeights`, the weights are applied to the set
`𝐂`. If `w` is a `Function`, the weights are found passing each matrix
in the set to such function. An appropriate choice for AJD algorithms
minimizing a least-squares criterion, like *OJoB* and *NoJoB*, is the
[`nonDiagonality`](@ref) function (Congedo et al.(2008)[🎓](@ref)).
By default, no weights are applied.

If `preWhite` is true the solution is found by the two-step procedure
described here above in section [pre-whitening for AJD](@ref).
By default, it is false.
Dimensionality reduction can be obtained at this stage using arguments
`eVarC` and `eVarMeth`, in the same way they are used to find
the [subspace dimension](@ref) ``p``, but using the accumulated
regularized eigenvalues of

``\\frac{1}{k}\\sum_{l=1}^kC_k``.

The default values are:
- `eVarC` is set to 0.999
- `eVarMeth=searchsortedfirst`.

If `sort` is true (default), the vectors in `.F` are permuted as explained
here above in [permutation for AJD](@ref), otherwise they will be in arbitrary order.

A matrix can be passed with the `init` argument in order to initialize
the matrix ``F`` to be found by the AJD algorithm.
If `nothing` is passed (default), ``F`` is initialized as per this table:

| Algorithm   | Initialization of ``F``|
|:----------|:----------|
| OJoB | eigevector matrix of ``\\frac{1}{k}\\sum_{l=1}^kC_k^2`` (Congedo et al., 2011)|
| NoJoB | identity matrix |

`tol` is the tolerance for convergence of the solving algorithm.
By default it is set to the square root of `Base.eps` of the nearest real type of the data
input. This corresponds to requiring the relative change across two successive
iterations of the average squared norm of the column vectors of ``F`` to vanish for
about half the significant digits. If the solving algorithm encounter difficulties in converging,
try setting `tol` in between 1e-6 and 1e-3.

`maxiter` is the maximum number of iterations allowed to the solving
algorithm (2000 by default). If this maximum number of iteration
is attained, a warning will be printed in the REPL. In this case,
try increasing `maxiter` and/or `tol`.

If `verbose` is true (false by default), the convergence attained
at each iteration will be printed in the REPL.

`eVar` and `eVarMeth` are used to define a
[subspace dimension](@ref) ``p`` using the accumulated regularized
eigenvalues in Eq. [ajd.7].

The default values are:
- `eVar` is set to the dimension of the matrices in `𝐂`
- `eVarMeth=searchsortedfirst`.

Note that passing `nothing` or a real nummber as `eVar` (see
[subspace dimension](@ref)) is meningful only if `sort` is set to true
(default) and if the input matrices ``{C_1,...,C_k}`` are
positive definite.

If `simple` is set to `true`, ``p`` is set equal to the dimension
of the matrices ``{C_1,...,C_k}`` and only the fields `.F` and `.iF`
are written in the constructed object.
This corresponds to the typical output of AJD algorithms.

**(2) Approximate joint diagonalization**
with a set of ``k`` data matrices `𝐗` as input; the
covariance matrices of the set are estimated using arguments
`covEst`, `dims` and `meanX`
(see [covariance matrix estimations](@ref))
and passed to method (1) with the
remaining arguments of method (2).

**See also:** [PCA](@ref), [CSP](@ref), [mAJD](@ref).

**Examples:**

```
using Diagonalizations, LinearAlgebra, PosDefManifold, Test


# method (1) real
t, n, k=50, 10, 4
A=randn(n, n) # mixing matrix in model x=As
Xset = [genDataMatrix(t, n) for i = 1:k]
Xfixed=randn(t, n)./1
for i=1:length(Xset) Xset[i]+=Xfixed end
Cset = ℍVector([ℍ((Xset[s]'*Xset[s])/t) for s=1:k])
aC=ajd(Cset; simple=true)

# method (1) complex
t, n, k=50, 10, 4
Ac=randn(ComplexF64, n, n) # mixing matrix in model x=As
Xcset = [genDataMatrix(ComplexF64, t, n) for i = 1:k]
Xcfixed=randn(ComplexF64, t, n)./1
for i=1:length(Xcset) Xcset[i]+=Xcfixed end
Ccset = ℍVector([ℍ((Xcset[s]'*Xcset[s])/t) for s=1:k])
aCc=ajd(Ccset; algorithm=:OJoB, simple=true)


# method (2) real
aX=ajd(Xset; simple=true)
@test aX≈aC

# method (2) complex
aXc=ajd(Xcset; algorithm=:OJoB, simple=true)
@test aXc≈aCc


# create 20 REAL random commuting matrices
# they all have the same eigenvectors
Cset2=PosDefManifold.randP(3, 20; eigvalsSNR=Inf, commuting=true)

# estimate the approximate joint diagonalizer (ajd)
a=ajd(Cset2; algorithm=:OJoB)

# the ajd must be equivalent to the eigenvector matrix of any of the matrices in Cset
@test spForm(a.F'*eigvecs(Cset2[1]))+1. ≈ 1.0

# the same thing using the NoJoB algorithm. Here we just do a sanity check
# as the NoJoB solution is not constrained in the orthogonal group
a=ajd(Cset2; algorithm=:NoJoB)
@test spForm(a.F'*eigvecs(Cset2[1]))<0.01


# create 20 COMPLEX random commuting matrices
# they all have the same eigenvectors
Ccset2=PosDefManifold.randP(ComplexF64, 3, 20; eigvalsSNR=Inf, commuting=true)

# estimate the approximate joint diagonalizer (ajd)
ac=ajd(Ccset2; algorithm=:OJoB)

# the ajd must be equivalent to the eigenvector matrix of any of the matrices in Cset
# just a sanity check as rounding errors appears for complex data
@test spForm(ac.F'*eigvecs(Ccset2[1]))<0.001

# the same thing using the NoJoB algorithm. Here we just do a sanity check
# as the NoJoB solution is not constrained in the orthogonal group
ac=ajd(Ccset2; algorithm=:NoJoB)
@test spForm(ac.F'*eigvecs(Ccset2[1]))<0.01

# REAL data:
# normalize the trace of input matrices,
# give them weights according to the `nonDiagonality` function
# apply pre-whitening and limit the explained variance both
# at the pre-whitening level and at the level of final vector selection
Cset=PosDefManifold.randP(8, 20; eigvalsSNR=10, SNR=2, commuting=false)

a=ajd(Cset; trace1=true, w=nonD, preWhite=true, eVarC=8, eVar=0.99)

using Plots
# plot the original covariance matrices
# and their transformed counterpart
CMax=maximum(maximum(abs.(C)) for C ∈ Cset);
 h1 = heatmap(Cset[1], clim=(-CMax, CMax), title="C1", yflip=true, c=:bluesreds);
 h2 = heatmap(Cset[2], clim=(-CMax, CMax), title="C2", yflip=true, c=:bluesreds);
 h3 = heatmap(Cset[3], clim=(-CMax, CMax), title="C3", yflip=true, c=:bluesreds);
 h4 = heatmap(Cset[4], clim=(-CMax, CMax), title="C4", yflip=true, c=:bluesreds);
 📈=plot(h1, h2, h3, h4, size=(700,400))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigAJD1.png")

Dset=[a.F'*C*a.F for C ∈ Cset];
 DMax=maximum(maximum(abs.(D)) for D ∈ Dset);
 h5 = heatmap(Dset[1], clim=(-DMax, DMax), title="F'*C1*F", yflip=true, c=:bluesreds);
 h6 = heatmap(Dset[2], clim=(-DMax, DMax), title="F'*C2*F", yflip=true, c=:bluesreds);
 h7 = heatmap(Dset[3], clim=(-DMax, DMax), title="F'*C3*F", yflip=true, c=:bluesreds);
 h8 = heatmap(Dset[4], clim=(-DMax, DMax), title="F'*C4*F", yflip=true, c=:bluesreds);
 📉=plot(h5, h6, h7, h8, size=(700,400))
# savefig(📉, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigAJD2.png")

```

 ![Figure AJD1](assets/FigAJD1.png)

 ![Figure AJD2](assets/FigAJD2.png)

```
# COMPLEX data:
# normalize the trace of input matrices,
# give them weights according to the `nonDiagonality` function
# apply pre-whitening and limit the explained variance both
# at the pre-whitening level and at the level of final vector selection
Ccset=PosDefManifold.randP(3, 20; eigvalsSNR=10, SNR=2, commuting=false)

# run OJoB
ac=ajd(Ccset; trace1=true, w=nonD, preWhite=true,
       algorithm=:OJoB, eVarC=8, eVar=0.99)

# run NoJoB
ac=ajd(Ccset; eVarC=8, eVar=0.99)
```

"""
function ajd(𝐂::ℍVector;
             trace1    :: Bool   = false,
             w         :: Union{Tw, Function} = ○,
          algorithm :: Symbol = :NoJoB,
          preWhite  :: Bool   = false,
          sort      :: Bool   = true,
          init      :: Mato   = ○,
          tol       :: Real   = 1e-06,
          maxiter   :: Int    = 2000,
          verbose   :: Bool   = false,
        eVar     :: TeVaro    = _minDim(𝐂),
        eVarC    :: TeVaro    = ○,
        eVarMeth :: Function  = searchsortedfirst,
        simple   :: Bool      = false)

   args=("Approximate Joint Diagonalization", false)
   k, n=length(𝐂), size(𝐂[1], 1)

   if     algorithm ∈(:OJoB, :NoJoB)
          U, V, λ, iter, conv=JoB(reshape(𝕄Vector(𝐂), (k, 1, 1)), 1, k, :c, algorithm, eltype(𝐂[1]);
               trace1=trace1, w=w, preWhite=preWhite, sort=sort,
                  init=init, tol=tol, maxiter=maxiter, verbose=verbose,
               eVar=eVarC, eVarMeth=eVarMeth)
   #=
   elseif algorithm==:JADE
          U, V, λ, iter, conv=JADE(𝐂, :c;
               trace1=trace1, w=w, preWhite=preWhite, sort=sort,
                  init=init, tol=tol, maxiter=maxiter, verbose=verbose,
               eVar=eVarC, eVarMeth=eVarMeth)
   =#
   else
      throw(ArgumentError(📌*", ajd constructor: invalid `algorithm` argument: $algorithm"))
   end

   λ = _checkλ(λ) # make sure no imaginary noise is present (for complex data)

   simple ? LF(U, V, Diagonal(λ), ○, ○, ○, args...) :
   begin
      # println(λ)
      p, arev = _getssd!(eVar, λ, n, eVarMeth) # find subspace
      LF(U[:, 1:p], V[1:p, :], Diagonal(λ[1:p]), arev[p], λ, arev, args...)
   end
end


function ajd(𝐗::VecMat;
             covEst     :: StatsBase.CovarianceEstimator = SCM,
             dims       :: Into = ○,
             meanX      :: Into = 0,
          trace1     :: Bool = false,
          w          :: Union{Tw, Function} = ○,
       algorithm :: Symbol = :NoJoB,
       preWhite  :: Bool   = false,
       sort      :: Bool   = true,
       init      :: Mato   = ○,
       tol       :: Real   = 1e-06,
       maxiter   :: Int    = 2000,
       verbose   :: Bool   = false,
     eVar     :: TeVaro   = _minDim(𝐗),
     eVarC    :: TeVaro   = ○,
     eVarMeth :: Function = searchsortedfirst,
     simple   :: Bool     = false)

   if dims===○ dims=_set_dims(𝐗) end
   (n, t)=dims==1 ? reverse(size(𝐗[1])) : size(𝐗[1])
   args=("Approximate Joint Diagonalization", false)

   if     algorithm ∈(:OJoB, :NoJoB)
          U, V, λ, iter, conv=JoB(𝐗, 1, length(𝐗), :d, algorithm, eltype(𝐗[1]);
               covEst=covEst, dims=dims, meanX=meanX,
               trace1=trace1, preWhite=preWhite, sort=sort,
                  init=init, tol=tol, maxiter=maxiter, verbose=verbose,
               eVar=eVarC, eVarMeth=eVarMeth)
   elseif algorithm==:JADE
          U, V, λ, iter, conv=JADE(𝐗, :d;
               covEst=covEst, dims=dims, meanX=meanX,
               trace1=trace1, w=w, preWhite=preWhite, sort=sort,
                  init=init, tol=tol, maxiter=maxiter, verbose=verbose,
               eVar=eVarC, eVarMeth=eVarMeth)
   else
      throw(ArgumentError(📌*", ajd constructor: invalid `algorithm` argument"))
   end

   λ = _checkλ(λ) # make sure no imaginary noise is present (for complex data)

   simple ? LF(U, V, Diagonal(λ), ○, ○, ○, args...) :
   begin
      p, arev = _getssd!(eVar, λ, n, eVarMeth) # find subspace
      LF(U[:, 1:p], V[1:p, :], Diagonal(λ[1:p]), arev[p], λ, arev, args...)
   end
end



"""
```
function majd(𝑿::VecVecMat;
              covEst     :: StatsBase.CovarianceEstimator = SCM,
              dims       :: Into    = ○,
              meanX      :: Into    = 0,
          algorithm :: Symbol    = :NoJoB,
          fullModel :: Bool      = false,
          preWhite  :: Bool      = false,
          sort      :: Bool      = true,
          init      :: VecMato   = ○,
          tol       :: Real      = 0.,
          maxiter   :: Int       = 2000,
          verbose   :: Bool      = false,
        eVar     :: TeVaro   = _minDim(𝑿),
        eVarC    :: TeVaro   = ○,
        eVarMeth :: Function = searchsortedfirst,
        simple   :: Bool     = false)

```

Return a [LinearFilter](@ref) object.

**Multiple Approximate Joint Diagonalization** of the ``k`` sets
of ``m`` data matrices `𝐗`
using the given solving `algorithm` (*NoJoB* by default).

If `fullModel` is true, the [gmca.3] problem here above is solved,
otherwise (default), the [gmca.2] problem here above is solved.

If `preWhite` the two-step procedure explained here above in the
section [pre-whitening for MAJD](@ref) is used.
Dimensionality reduction can be obtained at this stage using arguments
`eVarC` and `eVarMeth`.

The default values are:
- `eVarC` is set to 0.999
- `eVarMeth=searchsortedfirst`.

If `sort` is true (default), the column vectors of the matrices ``F_1,...,F_m``
are signed and permuted
as explained here above in [permutation for MAJD](@ref),
otherwise they will have arbitrary sign and will be in arbitrary order.

A vector of matrices can be passed with the `init` argument in order
to initialize the matrices ``F_1,...,F_m`` to be found by the MAJD algorithm.
If `nothing` is passed (default), ``F_i`` is initialized as per this table:

| Algorithm   | `fullModel` | Initialization of ``F_i``|
|:----------|:----------|:----------|
| OJoB | true | eigevector matrix of ``\\frac{1}{m}\\sum_{l=1}^k\\sum_{j=1}^m C_{lij}C_{lij}^H``
 (Congedo et al., 2011)|
| OJoB | false | eigevector matrix of ``\\frac{1}{m}\\sum_{l=1}^k\\sum_{j≠i, j=1}^m C_{lij}C_{lij}^H``
 (Congedo et al., 2011)|
| NoJoB | true or false | identity matrix |

`tol` is the tolerance for convergence of the solving algorithm.
By default it is set to the square root of `Base.eps` of the nearest real
type of the data input. This corresponds to requiring the relative
change across two successive iterations of the average squared norm
of the column vectors of matrices ``F_1,...,F_m`` to vanish for about
half the significant
digits. If the solving algorithm encounter difficulties in converging,
try setting `tol` in between 1e-6 and 1e-3.

`maxiter` is the maximum number of iterations allowed to the solving
algorithm (2000 by default). If this maximum number of iteration
is attained, a warning will be printed in the REPL. In this case,
try increasing `maxiter` and/or `tol`.

If `verbose` is true (false by default), the convergence attained
at each iteration will be printed in the REPL.

`eVar` and `eVarMeth` are used to define a
[subspace dimension](@ref) ``p`` using the accumulated regularized
eigenvalues in Eq. [gmca.7]

The default values are:
- `eVar` is set to the minimum dimension of the matrices in `𝐗`
- `eVarMeth=searchsortedfirst`.

If `simple` is set to `true`, ``p`` is set equal to the dimension
of the covariance matrices that are computed on the matrices in `𝐗`,
which depends on the choice of `dims`,
and only the fields `.F` and `.iF`
are written in the constructed object.
This corresponds to the typical output of approximate diagonalization
algorithms.

**See also:** [gMCA](@ref), [gCCA](@ref), [AJD](@ref).

**Examples:**

```
using Diagonalizations, LinearAlgebra, PosDefManifold, Test

##  Create data for testing the case k>1, m>1 ##
# `t` is the number of samples,
# `m` is the number of datasets,
# `k` is the number of observations,
# `n` is the number of variables,
# `noise` must be smaller than 1.0. The smaller the noise, the more data are correlated
# Output k vectors of m data data matrices
function getData(t, m, k, n, noise)
    # create m identical data matrices and rotate them by different
    # random orthogonal matrices V_1,...,V_m
    𝐕=[randU(n) for i=1:m] # random orthogonal matrices
    # variables common to all subjects with unique variance profile across k
    X=[(abs2.(randn(n))).*randn(n, t) for s=1:k]
    # each subject has this common part plus a random part
    𝐗=[[𝐕[i]*((1-noise)*X[s] + noise*randn(n, t)) for i=1:m] for s=1:k]
    return 𝐗, 𝐕
end

function getData(::Type{Complex{T}}, t, m, k, n, noise) where {T<:AbstractFloat}
    # create m identical data matrices and rotate them by different
    # random orthogonal matrices V_1,...,V_m
    𝐕=[randU(ComplexF64, n) for i=1:m] # random orthogonal matrices
    # variables common to all subjects with unique variance profile across k
    X=[(abs2.(randn(n))).*randn(ComplexF64, n, t) for s=1:k]
    # each subject has this common part plus a random part
    𝐗=[[𝐕[i]*((1-noise)*X[s] + noise*randn(ComplexF64, n, t)) for i=1:m] for s=1:k]
    return 𝐗, 𝐕
end


# REAL data
# do joint blind source separation of non-stationary data
t, m, n, k, noise = 200, 5, 4, 6, 0.1
Xset, Vset=getData(t, m, k, n, noise)
𝒞=Array{Matrix}(undef, k, m, m)
for s=1:k, i=1:m, j=1:m 𝒞[s, i, j]=(Xset[s][i]*Xset[s][j]')/t end

aX=majd(Xset; fullModel=true, algorithm=:OJoB)
# the spForm index of the estimated demixing matrices times the true
# mixing matrix must be low
@test mean(spForm(aX.F[i]'*Vset[i]) for i=1:m)<0.1

# test the same using NoJoB algorithm
aX=majd(Xset; fullModel=true, algorithm=:NoJoB)
@test mean(spForm(aX.F[i]'*Vset[i]) for i=1:m)<0.1

# plot the original cross-covariance matrices and the rotated
# cross-covariance matrices

# Get all products 𝐔[i]' * 𝒞[l, i, j] * 𝐔[j]
function _rotate_crossCov(𝐔, 𝒞, m, k)
    𝒮=Array{Matrix}(undef, k, m, m)
    @inbounds for l=1:k, i=1:m, j=1:m 𝒮[l, i, j]=𝐔[i]'*𝒞[l, i, j]*𝐔[j] end
    return 𝒮
end

# Put all `k` cross-covariances in a single matrix
# of dimension m*n x m*n for visualization
function 𝒞2Mat(𝒞::AbstractArray, m, k)
    n=size(𝒞[1, 1, 1], 1)
    C=Matrix{Float64}(undef, m*n, m*n)
    for i=1:m, j=1:m, x=1:n, y=1:n C[i*n-n+x, j*n-n+y]=𝒞[k, i, j][x, y] end
    return C
end

using Plots

Cset=[𝒞2Mat(𝒞, m, s) for s=1:k]
 Cmax=maximum(maximum(abs.(C)) for C ∈ Cset)
 h1 = heatmap(Cset[1], clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="all cross-cov, k=1")
 h2 = heatmap(Cset[2], clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="all cross-cov, k=2")
 h3 = heatmap(Cset[2], clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="all cross-cov, k=3")
 h4 = heatmap(Cset[2], clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="all cross-cov, k=4")
 h5 = heatmap(Cset[2], clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="all cross-cov, k=5")
 h6 = heatmap(Cset[2], clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="all cross-cov, k=6")
 📈=plot(h1, h2, h3, h4, h5, h6, size=(1200,550))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigmAJD1.png")

𝒮=_rotate_crossCov(aX.F, 𝒞, m, k)
 Sset=[𝒞2Mat(𝒮, m, s) for s=1:k]
 Smax=maximum(maximum(abs.(S)) for S ∈ Sset)
 h11 = heatmap(Sset[1], clim=(-Smax, Smax), yflip=true, c=:bluesreds, title="rotated cross-cov, k=1")
 h12 = heatmap(Sset[2], clim=(-Smax, Smax), yflip=true, c=:bluesreds, title="rotated cross-cov, k=2")
 h13 = heatmap(Sset[2], clim=(-Smax, Smax), yflip=true, c=:bluesreds, title="rotated cross-cov, k=3")
 h14 = heatmap(Sset[2], clim=(-Smax, Smax), yflip=true, c=:bluesreds, title="rotated cross-cov, k=4")
 h15 = heatmap(Sset[2], clim=(-Smax, Smax), yflip=true, c=:bluesreds, title="rotated cross-cov, k=5")
 h16 = heatmap(Sset[2], clim=(-Smax, Smax), yflip=true, c=:bluesreds, title="rotated cross-cov, k=6")
 📉=plot(h11, h12, h13, h14, h15, h16, size=(1200,550))
# savefig(📉, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigmAJD2.png")

```
![Figure mAJD1](assets/FigmAJD1.png)

![Figure mAJD2](assets/FigmAJD2.png)

In the bottom figures here above, the rotated cross-covariance matrices have the
expected *strip-diagonal* form, that is, each block
``F_i^T\\frac{1}{T}(X_{li}X_{lj}^T)F_j``,
for ``l∈[1,...,k]``, ``i,j∈[1,...,m]``, is approximately diagonal.

```
# COMPLEX data
# do joint blind source separation of non-stationary data
t, m, n, k, noise = 200, 5, 4, 6, 0.1
Xcset, Vcset=getData(ComplexF64, t, m, k, n, noise)
𝒞=Array{Matrix}(undef, k, m, m)
for s=1:k, i=1:m, j=1:m 𝒞[s, i, j]=(Xcset[s][i]*Xcset[s][j]')/t end

aXc=majd(Xcset; fullModel=true, algorithm=:OJoB)
# the spForm index of the estimated demixing matrices times the true
# mixing matrix must be low
@test mean(spForm(aXc.F[i]'*Vcset[i]) for i=1:m)<0.1

# test the same using NoJoB algorithm
aXc=majd(Xcset; fullModel=true, algorithm=:NoJoB)
@test mean(spForm(aXc.F[i]'*Vcset[i]) for i=1:m)<0.1
```

"""
function majd(𝑿::VecVecMat;
              covEst     :: StatsBase.CovarianceEstimator = SCM,
              dims       :: Into    = ○,
              meanX      :: Into    = 0,
          algorithm :: Symbol    = :NoJoB,
          fullModel :: Bool      = false,
          preWhite  :: Bool      = false,
          sort      :: Bool      = true,
          init      :: VecMato   = ○,
          tol       :: Real      = 0.,
          maxiter   :: Int       = 2000,
          verbose   :: Bool      = false,
        eVar     :: TeVaro   = _minDim(𝑿),
        eVarC    :: TeVaro   = ○,
        eVarMeth :: Function = searchsortedfirst,
        simple   :: Bool     = false)

   if dims===○ dims=_set_dims(𝑿) end
   (n, t)=dims==1 ? reverse(size(𝑿[1][1])) : size(𝑿[1][1])
   k=length(𝑿)
   m=length(𝑿[1])
   args=("Multiple Approximate Joint Diagonalization", false)

   if algorithm ∈(:OJoB, :NoJoB)
      𝐔, 𝐕, λ, iter, conv=JoB(𝑿, m, k, :d, algorithm, eltype(𝑿[1][1]);
               covEst=covEst, dims=dims, meanX=meanX,
               fullModel=fullModel, preWhite=preWhite, sort=sort,
                  init=init, tol=tol, maxiter=maxiter, verbose=verbose,
               eVar=eVarC, eVarMeth=eVarMeth)
   # elseif...
   else
      throw(ArgumentError(📌*", majd constructor: invalid `algorithm` argument"))
   end

   λ = _checkλ(λ) # make sure no imaginary noise is present (for complex data)

   simple ? LF(𝐔, 𝐕, Diagonal(λ), ○, ○, ○, args...) :
   begin
      p, arev = _getssd!(eVar, λ, n, eVarMeth) # find subspace
      LF([𝐔[i][:, 1:p] for i=1:m], [𝐕[i][1:p, :] for i=1:m], Diagonal(λ[1:p]), arev[p], λ, arev, args...)
   end
end
