#   Unit "pca.jl" of the Diagonalization.jl Package for Julia language
#
#   MIT License
#   Copyright (c) 2019, 2020
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements Principal Component Analysys (PCA) and whitening
#   (sphering), that is, filters based on the eigendecomposition.

"""
```julia
(1)
function pca(C :: SorH;
             eVar     :: TeVaro = nothing,
             eVarMeth :: Function = searchsortedfirst,
             simple   :: Bool = false)

(2)
function pca(X::Mat;
             covEst   :: StatsBase.CovarianceEstimator = SCM,
             dims     :: Into = ○,
             meanX    :: Tmean = 0,
             wX       :: Tw = ○,
          eVar     :: TeVaro = ○,
          eVarMeth :: Function = searchsortedfirst,
          simple   :: Bool = false)

(3)
function pca(𝐗::VecMat;
             covEst   :: StatsBase.CovarianceEstimator = SCM,
             dims     :: Into = ○,
             meanX    :: Into = 0,
          eVar     :: TeVaro = ○,
          eVarMeth :: Function = searchsortedfirst,
          simple   :: Bool = false,
       metric   :: Metric = Euclidean,
       w        :: Vector = [],
       ✓w       :: Bool = true,
       init     :: SorHo = nothing,
       tol      :: Real = 0.,
       verbose  :: Bool = false)
```

Return a [LinearFilter](@ref) object:

**(1) Principal component analysis**
with real or complex covariance matrix `C` as input.

`C` must be flagged as `Symmetric`, if real, or `Hermitian`,
if either real or complex, see [data input](@ref).

`eVar` and `evarMeth` are keyword optional arguments for defining the
[subspace dimension](@ref) ``p`` using the `.arev` vector given by Eq. [pca.6].
The default values are:
- `eVar=0.999`
- `evarMeth=searchsortedfirst`

If `simple` is set to `true`, ``p`` is set equal to ``n`` and only the
fields `.F` and `.iF` are written in the constructed object.
This option is provided for low-level work when you don't need to define
a subspace dimension or you want to define it by your own methods.

**(2) Principal component analysis**
with a real or complex data matrix `X` as input.

`CovEst`, `dims`, `meanX`, `wX` are optional keyword arguments to
regulate the estimation of the covariance matrix of `X`.
See [covariance matrix estimations](@ref).

Once the covariance matrix estimated, method (1) is invoked
with optional keyword arguments `eVar`, `eVarMeth` and `simple`.

**(3) Principal component analysis**
with a vector of real or complex data matrices `𝐗` as input.

`CovEst`, `dims` and `meanX` are optional keyword arguments to
regulate the estimation of the covariance matrices for
all data matrices in `𝐗`. See [covariance matrix estimations](@ref).

A mean of these covariance matrices is computed using
optional keywords arguments `metric`, `w`, `✓w`, `init`, `tol` and `verbose`.
See [mean covariance matrix estimations](@ref).
By default, the arithmetic mean is computed.

Once the mean covariance matrix estimated, method (1) is invoked
with optional keyword arguments `eVar`, `eVarMeth` and `simple`.

**See also:** [Whitening](@ref), [CSP](@ref), [MCA](@ref), [AJD](@ref).

**Examples:**
```julia
using Diagonalizations, LinearAlgebra, PosDefManifold, Test

# Method (1) real
n, t=10, 100
X=genDataMatrix(n, t)
C=(X*X')/t
pC=pca(Hermitian(C); simple=true)
# or, shortly
pC=pca(ℍ(C); simple=true)

# Method (1) complex
Xc=genDataMatrix(ComplexF64, n, t)
Cc=(Xc*Xc')/t
pCc=pca(Hermitian(Cc); simple=true)


# Method (2) real
pX=pca(X; simple=true)
@test C≈pC.F*pC.D*pC.iF
@test C≈pC.F*pC.D*pC.F'
@test pX≈pC

# Method (2) complex
pXc=pca(Xc; simple=true)
@test Cc≈pCc.F*pCc.D*pCc.iF
@test Cc≈pCc.F*pCc.D*pCc.F'
@test pXc≈pCc


# Method (3) real
k=10
Xset=[genDataMatrix(n, t) for i=1:k]

# pca on the average covariance matrix
p=pca(Xset)

# ... selecting subspace dimension allowing an explained variance = 0.5
p=pca(Xset; eVar=0.5)

# ... averaging the covariance matrices using the logEuclidean metric
p=pca(Xset; metric=logEuclidean, eVar=0.5)

# ... giving weights `w` to the covariance matrices
p=pca(Xset; metric=logEuclidean, w=abs2.(randn(k)), eVar=0.5)

# ... subtracting the mean
p=pca(Xset; meanX=nothing, metric=logEuclidean, w=abs2.(randn(k)), eVar=0.5)

# pca on the average of the covariance matrices computed along dims 1
p=pca(Xset; dims=1)

# explained variance
p.eVar

# name of the filter
p.name

using Plots
# plot regularized accumulated eigenvalues
plot(p.arev)

# plot the original covariance matrix and the rotated covariance matrix
 Cmax=maximum(abs.(C));
 h1 = heatmap(C, clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="C");
 D=pC.F'*C*pC.F;
 Dmax=maximum(abs.(D));
 h2 = heatmap(D, clim=(0, Dmax), yflip=true, c=:amp, title="F'*C*F");
 📈=plot(h1, h2, size=(700, 300))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigPCA.png")
```

 ![Figure PCA](assets/FigPCA.png)

```julia
# Method (3) complex
k=10
Xcset=[genDataMatrix(ComplexF64, n, t) for i=1:k]

# pca on the average covariance matrix
pc=pca(Xcset)

# ... selecting subspace dimension allowing an explained variance = 0.5
pc=pca(Xcset; eVar=0.5)

# ... averaging the covariance matrices using the logEuclidean metric
pc=pca(Xcset; metric=logEuclidean, eVar=0.5)

# ... giving weights `w` to the covariance matrices
pc=pca(Xcset; metric=logEuclidean, w=abs2.(randn(k)), eVar=0.5)

# ... subtracting the mean
pc=pca(Xcset; meanX=nothing, metric=logEuclidean, w=abs2.(randn(k)), eVar=0.5)

# pca on the average of the covariance matrices computed along dims 1
pc=pca(Xcset; dims=1)

# explained variance
pc.eVar

# name of the filter
pc.name
```


"""
function pca(C :: SorH;
             eVar     :: TeVaro = ○,
             eVarMeth :: Function = searchsortedfirst,
             simple   :: Bool = false)

   args=("Principal Component Analysis", false)

   λ, U = eig(C) # get evd
   λ = _checkλ(λ) # make sure no imaginary noise is present (for complex data)

   simple ? LF(U, Matrix(U'), Diagonal(λ), ○, ○, ○, args...) :
   begin
      eVar, D, U, p, arev=_ssd!(eVar, λ, U, _minDim(C), eVarMeth) # find subspace
      LF(U, Matrix(U'), D, eVar, λ, arev, args...)
   end
end


# Principal Component Analysis with data as input: API
function pca(X :: Mat;
             covEst   :: StatsBase.CovarianceEstimator = SCM,
             dims     :: Into = ○,
             meanX    :: Tmean = 0,
             wX       :: Tw = ○,
          eVar     :: TeVaro = ○,
          eVarMeth :: Function = searchsortedfirst,
          simple   :: Bool = false)

   dims===○ && (dims=_set_dims(X))
   _check_data(X, dims, covEst, meanX, wX)===○ && return
   args=("Principal Component Analysis", false)

   LF(_getEVD(X, covEst, dims, meanX, wX, eVar, eVarMeth, simple)..., args...)
end


function pca(𝐗 :: VecMat;
             covEst   :: StatsBase.CovarianceEstimator = SCM,
             dims     :: Into = ○,
             meanX    :: Into = 0,
          eVar     :: TeVaro = ○,
          eVarMeth :: Function = searchsortedfirst,
          simple   :: Bool = false,
       metric   :: Metric = Euclidean,
       w        :: Vector = [],
       ✓w       :: Bool = true,
       init     :: SorHo = nothing,
       tol      :: Real = 0.,
       verbose  :: Bool = false)

    Metric==VonNeumann && throw(ArgumentError(📌*", pca function: A solution for the mean is not available for the Von Neumann metric. Use another metric as `metric` argument"))
    dims===○ && (dims=_set_dims(𝐗))
    _check_data(𝐗, dims, covEst, meanX, ○)===○ && return

    𝐂=_cov(𝐗; covEst=covEst, dims = dims, meanX = meanX)

    pca(mean(metric, 𝐂;
             w = w, ✓w = ✓w,
             init = init===○ ? ○ : Hermitian(init), #just init here when you upfate PodDefManifold
             tol = tol, verbose = verbose),
        eVar = eVar, eVarMeth = eVarMeth, simple = simple)
end




"""
```julia
(1)
function whitening(C :: SorH;
                   eVar     :: TeVaro=○,
                   eVarMeth :: Function=searchsortedfirst,
                   simple   :: Bool=false)

(2)
function whitening(X::Mat;
                   covEst   :: StatsBase.CovarianceEstimator = SCM,
                   dims     :: Into = ○,
                   meanX    :: Tmean = 0,
                   wX       :: Tw = ○,
                eVar     :: TeVaro = ○,
                eVarMeth :: Function = searchsortedfirst,
                simple   :: Bool = false)

(3)
function whitening(𝐗::VecMat;
                   covEst   :: StatsBase.CovarianceEstimator = SCM,
                   dims     :: Into = ○,
                   meanX    :: Into = 0,
                eVar     :: TeVaro = ○,
                eVarMeth :: Function = searchsortedfirst,
                simple   :: Bool = false,
             metric   :: Metric = Euclidean,
             w        :: Vector = [],
             ✓w       :: Bool = true,
             init     :: SorHo = nothing,
             tol      :: Real = 0.,
             verbose  :: Bool = false)

```

Return a [LinearFilter](@ref) object:

**(1) Whitening**
with real or complex covariance matrix `C` as input.

`C` must be flagged as `Symmetric`, if real or `Hermitian`,
if real or complex, see [data input](@ref).

`eVar` and `evarMeth` are keyword optional arguments for defining the
[subspace dimension](@ref) ``p`` using the `.arev` vector given by Eq. [pca.6],
see [PCA](@ref).
The default values are:
- `eVar=0.999`
- `evarMeth=searchsortedfirst`

If `simple` is set to `true`, ``p`` is set equal to ``n`` and only the
fields `.F` and `.iF` are written in the constructed object.
This option is provided for low-level work when you don't need to define
a subspace dimension or you want to define it by your own methods.

**(2) Whitening**
with a real or complex data matrix `X` as input.

`CovEst`, `dims`, `meanX`, `wX` are optional keyword arguments to
regulate the estimation of the covariance matrix of `X`.
See [covariance matrix estimations](@ref).

Once the covariance matrix estimated, method (1) is invoked
with optional keyword arguments `eVar`, `eVarMeth` and `simple`.

**(3) Whitening**
with a vector of real or complex data matrices `𝐗` as input.

`CovEst`, `dims` and `meanX` are optional keyword arguments to
regulate the estimation of the covariance matrices for
all data matrices in `𝐗`. See [covariance matrix estimations](@ref).

A mean of these covariance matrices is computed using
optional keywords arguments `metric`, `w`, `✓w`, `init`, `tol` and `verbose`.
See [mean covariance matrix estimations](@ref).
By default, the arithmetic mean is computed.

Once the mean covariance matrix estimated, method (1) is invoked
with optional keyword arguments `eVar`, `eVarMeth` and `simple`.

**See also:** [PCA](@ref), [CCA](@ref).

**Examples:**
```julia
using Diagonalizations, LinearAlgebra, PosDefManifold, Test

# Method (1) real
n, t=10, 100
X=genDataMatrix(n, t)
C=(X*X')/t
wC=whitening(Hermitian(C); simple=true)
# or, shortly
wC=whitening(ℍ(C); simple=true)

# Method (1) complex
Xc=genDataMatrix(ComplexF64, n, t)
Cc=(Xc*Xc')/t
wCc=whitening(Hermitian(Cc); simple=true)


# Method (2) real
wX=whitening(X; simple=true)
@test wC.F'*C*wC.F≈I
@test wX.F'*C*wX.F≈I
@test wX≈wC

# Method (2) complex
wXc=whitening(Xc; simple=true)
@test wCc.F'*Cc*wCc.F≈I
@test wXc.F'*Cc*wXc.F≈I
@test wXc≈wCc


# Method (3) real
k=10
Xset=[genDataMatrix(n, t) for i=1:k]

# whitening on the average covariance matrix
w=whitening(Xset)

# ... selecting subspace dimension allowing an explained variance = 0.5
w=whitening(Xset; eVar=0.5)

# ... averaging the covariance matrices using the logEuclidean metric
w=whitening(Xset; metric=logEuclidean, eVar=0.5)

# ... giving weights `w` to the covariance matrices
w=whitening(Xset; metric=logEuclidean, w=abs2.(randn(k)), eVar=0.5)

# ... subtracting the mean
w=whitening(Xset; meanX=nothing, metric=logEuclidean, w=abs2.(randn(k)), eVar=0.5)

# whitening on the average of the covariance matrices computed along dims 1
w=whitening(Xset; dims=1)

# explained variance
w.eVar

# name of the filter
w.name

using Plots
# plot regularized accumulated eigenvalues
plot(w.arev)

# plot the original covariance matrix and the whitened covariance matrix
 Cmax=maximum(abs.(C));
 h1 = heatmap(C, clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="C");
 D=wC.F'*C*wC.F;
 h2 = heatmap(D, clim=(0, 1), yflip=true, c=:amp, title="F'*C*F");
 📈=plot(h1, h2, size=(700, 300))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigWhitening.png")
```

 ![Figure Whitening](assets/FigWhitening.png)

```julia

# Method (3) complex
k=10
Xcset=[genDataMatrix(ComplexF64, n, t) for i=1:k]

# whitening on the average covariance matrix
wc=whitening(Xcset)

# ... selecting subspace dimension allowing an explained variance = 0.5
wc=whitening(Xcset; eVar=0.5)

# ... averaging the covariance matrices using the logEuclidean metric
wc=whitening(Xcset; metric=logEuclidean, eVar=0.5)

# ... giving weights `w` to the covariance matrices
wc=whitening(Xset; metric=logEuclidean, w=abs2.(randn(k)), eVar=0.5)

# ... subtracting the mean
wc=whitening(Xcset; meanX=nothing, metric=logEuclidean, w=abs2.(randn(k)), eVar=0.5)

# whitening on the average of the covariance matrices computed along dims 1
wc=whitening(Xcset; dims=1)

# explained variance
wc.eVar

# name of the filter
wc.name
```

"""
function whitening(C :: SorH;
                   eVar     :: TeVaro=○,
                   eVarMeth :: Function=searchsortedfirst,
                   simple   :: Bool=false)

   args=("Whitening", false)

   λ, U = eig(C) # get evd
   λ = _checkλ(λ) # make sure no imaginary noise is present (for complex data)
   if simple
     if eltype(C)<:Real
		  ispos(λ; tol=eps(eltype(C)), rev=true, 🔔=true,
		  msg="negative or almost zero eigenvalue") || throw(ArgumentError("A `simple` linear filter cannot be created. See the warning that has been printed in Julia's REPL"))
	 end
     D=Diagonal(λ)
     LF(U*D^-0.5, D^0.5*Matrix(U'), D, ○, ○, ○, args...)
   else
     eVar, D, U, p, arev=_ssd!(eVar, λ, U, _minDim(C), eVarMeth) # subspace dimension
     LF(U*D^-0.5, D^0.5*Matrix(U'), D, eVar, λ, arev, args...)
   end
end


function whitening(X::Mat;
                   covEst   :: StatsBase.CovarianceEstimator = SCM,
                   dims     :: Into = ○,
                   meanX    :: Tmean = 0,
                   wX       :: Tw = ○,
                eVar     :: TeVaro = ○,
                eVarMeth :: Function = searchsortedfirst,
                simple   :: Bool = false)

   dims===○ && (dims=_set_dims(X))
   _check_data(X, dims, covEst, meanX, wX)===○ && return
   args=("Whitening", false)

   LF(_getWhi(X, covEst, dims, meanX, wX, eVar, eVarMeth, simple)..., args...)
end


function whitening(𝐗::VecMat;
                   covEst   :: StatsBase.CovarianceEstimator = SCM,
                   dims     :: Into = ○,
                   meanX    :: Into = 0,
                eVar     :: TeVaro = ○,
                eVarMeth :: Function = searchsortedfirst,
                simple   :: Bool = false,
             metric   :: Metric = Euclidean,
             w        :: Vector = [],
             ✓w       :: Bool = true,
             init     :: SorHo = nothing,
             tol      :: Real = 0.,
             verbose  :: Bool = false)

   Metric==VonNeumann && throw(ArgumentError(📌*", whitening function: A solution for the mean is not available for the Von Neumann metric. Use another metric as `metric` argument"))
   dims===○ && (dims=_set_dims(𝐗))
   _check_data(𝐗, dims, covEst, meanX, ○)===○ && return

   whitening(mean(metric, _cov(𝐗; covEst=covEst, dims=dims, meanX=meanX);
                  w = w, ✓w = ✓w,
                  init = init===○ ? ○ : Hermitian(init), #jet init here when you upfate PodDefManifold
                  tol = tol, verbose = verbose),
             eVar = eVar, eVarMeth = eVarMeth, simple = simple)
end
