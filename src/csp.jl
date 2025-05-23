#   Unit "csp.jl" of the Diagonalization.jl Package for Julia language
#
#   MIT License
#   Copyright (c) 2019-2025,
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements Common Spatial Pattern (CSP) filters,
#   that is, filters based on the generalized eigendecomposition


"""
```julia
(1)
function csp(C₁ :: SorH, C₂ :: SorH;
             eVar     :: TeVaro = ○,
             eVarC    :: TeVaro = ○,
             eVarMeth :: Function = searchsortedfirst,
             selMeth  :: Symbol = :extremal,
             simple   :: Bool = false)

(2)
function csp(X₁ :: Mat, X₂ :: Mat;
             covEst   :: StatsBase.CovarianceEstimator = SCM,
             dims     :: Into = ○,
             meanX₁   :: Tmean = 0,
             meanX₂   :: Tmean = 0,
             wX₁      :: Tw = ○,
             wX₂      :: Tw = ○,
          eVar     :: TeVaro = ○,
          eVarC    :: TeVaro = ○,
          eVarMeth :: Function = searchsortedfirst,
          selMeth  :: Symbol = :extremal,
          simple   :: Bool = false)

(3)
function csp(𝐗₁::VecMat, 𝐗₂::VecMat;
             covEst   :: StatsBase.CovarianceEstimator = SCM,
             dims     :: Into = ○,
             meanX₁   :: Into = 0,
             meanX₂   :: Into = 0,
          eVar     :: TeVaro = ○,
          eVarC    :: TeVaro = ○,
          eVarMeth :: Function = searchsortedfirst,
          selMeth  :: Symbol = :extremal,
          simple   :: Bool = false,
       metric   :: Metric = Euclidean,
       w₁       :: Vector = [],
       w₂       :: Vector = [],
       ✓w       :: Bool = true,
       init₁    :: SorHo = nothing,
       init₂    :: SorHo = nothing,
       tol      :: Real = 0.,
       verbose  :: Bool = false)
```

Return a [LinearFilter](@ref) object:

**(1) Common spatial pattern**
with covariance matrices `C_1` and `C_2` of dimension
``n⋅n`` as input. The subscript of the covariance matrices refers to the `dims`
used to compute it (see above).

`eVar`, `eVarC` and `eVarMeth` are keyword optional arguments
for defining the [subspace dimension](@ref) ``p``. Particularly:
-  By default, the two-step procedure described above is used to find the
   solution. In this case `eVarC` is used for defining the subspace dimension of
   the whitening step. If `eVarC=0.0` is passed (not to be confused with
   `eVarC=0` ), the solution will be find by the generalized
   eigenvalue-eigenvector procedure.
- `eVar` is the keyword optional argument for defining the
   [subspace dimension](@ref) ``p`` using the `.arev` vector
   given by [csp.5].
- `eVarMeth` applies to both `eVarC` and `eVar`. The default value is
   `evarMeth=searchsortedfirst`.

if `selMeth=:extremal` (default) use case **a) Separating two classes**
described above is considered. Any other symbol for `selMeth` will instruct
to consider instead the use case **b) Enhance the signal-to-noise ratio**.

If `simple` is set to `true`, ``p`` is set equal to ``n``
and only the fields `.F` and `.iF` are written in the constructed object.
This option is provided for low-level work when you don't need to define
a subspace dimension or you want to define it by your own methods.

**(2) Common spatial pattern**
with data matrices `X₁` and `X₂` as input.

`X₁` and `X₂` are real or complex data matrices.

`covEst`, `dims`, `meanX₁`, `meanX₂`,  `wX₁` and `wX₂` are optional
keyword arguments to regulate the estimation of the
covariance matrices ``(C_1, C_2)`` of (`X₁`, `X₂`).
Particularly (See [covariance matrix estimations](@ref)),
- `meanX₁` is the `mean` argument for data matrix `X₁`.
- `meanX₂` is the `mean` argument for data matrix `X₂`.
- `wX₁` is the `w` argument for estimating a weighted covariance matrix for `X₁`.
- `wX₂` is the `w` argument for estimating a weighted covariance matrix for `X₂`.
- `covEst` applies to the estimations of both covariance matrices.

Once the two covariance matrices ``C_1`` and ``C_2`` estimated,
method (1) is invoked with optional keyword arguments
`eVar`, `eVarC`, `eVarMeth`, `selMeth` and `simple`.
See method (1) for details.


**(3) Common spatial pattern**
with two vectors of data matrices
`𝐗₁` and `𝐗₂` as input.

`𝐗₁` and `𝐗₂` do not need to hold the same number
of matrices and the number of samples in the matrices they contain
is arbitrary.

`covEst`, `dims`, `meanX₁` and `meanX₂` are optional
keyword arguments to regulate the estimation of the
covariance matrices for all matrices in `𝐗₁` and `𝐗₂`.
See method (2) and [covariance matrix estimations](@ref).

A mean covariance matrix is computed separatedly from the covariance matrices
computed from the data matrices in `𝐗₁` and `𝐗₂`,
using optional keywords arguments
`metric`, `w₁`, `w₂`, `✓w`,
`init₁`, `init₂`, `tol` and `verbose`. Particularly
(see [mean covariance matrix estimations](@ref)),
- `w₁` are the weights for the covariance matrices computed from `𝐗₁`,
- `w₂` are the weights for the covariance matrices computed from `𝐗₂`,
- `init₁` is the initialization for the mean of the covariance matrices computed from `𝐗₁`,
- `init₂` is the initialization for the mean of the covariance matrices computed from `𝐗₂`.
By default, the arithmetic mean is computed.

**See also:** [CSTP](@ref), [PCA](@ref), [AJD](@ref), [mAJD](@ref).

**Examples:**

```julia
using Diagonalizations, LinearAlgebra, PosDefManifold, Test

# Method (1) real
t, n=50, 10
X1=genDataMatrix(n, t)
X2=genDataMatrix(n, t)
Cx1=Symmetric((X1*X1')/t)
Cx2=Symmetric((X2*X2')/t)
C=Cx1+Cx2
cC=csp(Cx1, Cx2; simple=true)
Dx1=cC.F'*Cx1*cC.F
@test norm(Dx1-Diagonal(Dx1))+1≈1.
Dx2=cC.F'*Cx2*cC.F
@test norm(Dx2-Diagonal(Dx2))+1≈1.
@test cC.F'*C*cC.F≈I
@test norm(Dx1-(I-Dx2))+1≈1.

# Method (1) complex
t, n=50, 10
X1c=genDataMatrix(ComplexF64, n, t)
X2c=genDataMatrix(ComplexF64, n, t)
Cx1c=Hermitian((X1c*X1c')/t)
Cx2c=Hermitian((X2c*X2c')/t)
Cc=Cx1c+Cx2c
cCc=csp(Cx1c, Cx2c; simple=true)
Dx1c=cCc.F'*Cx1c*cCc.F
@test norm(Dx1c-Diagonal(Dx1c))+1. ≈ 1.
Dx2c=cCc.F'*Cx2c*cCc.F
@test norm(Dx2c-Diagonal(Dx2c))+1. ≈ 1.
@test cCc.F'*Cc*cCc.F≈I
@test norm(Dx1c-(I-Dx2c))+1. ≈ 1.


# Method (2) real
c12=csp(X1, X2, simple=true)
Dx1=c12.F'*Cx1*c12.F
@test norm(Dx1-Diagonal(Dx1))+1≈1.
Dx2=c12.F'*Cx2*c12.F
@test norm(Dx2-Diagonal(Dx2))+1≈1.
@test c12.F'*C*c12.F≈I
@test norm(Dx1-(I-Dx2))+1≈1.
@test cC==c12

# Method (2) complex
c12c=csp(X1c, X2c, simple=true)
Dx1c=c12c.F'*Cx1c*c12c.F
@test norm(Dx1c-Diagonal(Dx1c))+1. ≈ 1.
Dx2c=c12c.F'*Cx2c*c12c.F
@test norm(Dx2c-Diagonal(Dx2c))+1. ≈ 1.
@test c12c.F'*Cc*c12c.F≈I
@test norm(Dx1c-(I-Dx2c))+1. ≈ 1.
@test cCc==c12c


# Method (3) real
# CSP of the average covariance matrices
k=10
Xset=[genDataMatrix(n, t) for i=1:k]
Yset=[genDataMatrix(n, t) for i=1:k]

c=csp(Xset, Yset)

# ... selecting subspace dimension allowing an explained variance = 0.9
c=csp(Xset, Yset; eVar=0.9)

# ... subtracting the mean from the matrices in Xset and Yset
c=csp(Xset, Yset; meanX₁=nothing, meanX₂=nothing, eVar=0.9)

# csp on the average of the covariance and cross-covariance matrices
# computed along dims 1
c=csp(Xset, Yset; dims=1, eVar=0.9)

# name of the filter
c.name

using Plots
# plot regularized accumulated eigenvalues
plot(c.arev)


# plot the original covariance matrices and the transformed counterpart
# example when argument `selMeth` is `extremal` (default): 2-class separation
 cC=csp(Cx1, Cx2)
 Cx1Max=maximum(abs.(Cx1));
 h1 = heatmap(Cx1, clim=(-Cx1Max, Cx1Max), title="Cx1", yflip=true, c=:bluesreds);
 h2 = heatmap(cC.F'*Cx1*cC.F, clim=(0, 1), title="F'*Cx1*F", yflip=true, c=:amp);
 Cx2Max=maximum(abs.(Cx2));
 h3 = heatmap(Cx2, clim=(-Cx2Max, Cx2Max), title="Cx2", yflip=true, c=:bluesreds);
 h4 = heatmap(cC.F'*Cx2*cC.F, clim=(0, 1), title="F'*Cx2*F", yflip=true, c=:amp);
 CMax=maximum(abs.(C));
 h5 = heatmap(C, clim=(-CMax, CMax), title="Cx1+Cx2", yflip=true, c=:bluesreds);
 h6 = heatmap(cC.F'*C*cC.F, clim=(0, 1), title="F'*(Cx1+Cx2)*F", yflip=true, c=:amp);
 📈=plot(h1, h3, h5, h2, h4, h6, size=(800,400))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigCSP1.png")
```

 ![Figure CSP1](assets/FigCSP1.png)

```julia
# example when argument `selMeth` is different from `extremal`: enhance snr
 cC=csp(Cx1, Cx2; selMeth=:enhaceSNR)
 Cx1Max=maximum(abs.(Cx1));
 h1 = heatmap(Cx1, clim=(-Cx1Max, Cx1Max), title="Cx1", yflip=true, c=:bluesreds);
 h2 = heatmap(cC.F'*Cx1*cC.F, clim=(0, 1), title="F'*Cx1*F", yflip=true, c=:amp);
 Cx2Max=maximum(abs.(Cx2));
 h3 = heatmap(Cx2, clim=(-Cx2Max, Cx2Max), title="Cx2", yflip=true, c=:bluesreds);
 h4 = heatmap(cC.F'*Cx2*cC.F, clim=(0, 1), title="F'*Cx2*F", yflip=true, c=:amp);
 CMax=maximum(abs.(C));
 h5 = heatmap(C, clim=(-CMax, CMax), title="Cx1+Cx2", yflip=true, c=:bluesreds);
 h6 = heatmap(cC.F'*C*cC.F, clim=(0, 1), title="F'*(Cx1+Cx2)*F", yflip=true, c=:amp);
 📉=plot(h1, h3, h5, h2, h4, h6, size=(800,400))
# savefig(📉, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigCSP2.png")

```

 ![Figure CSP2](assets/FigCSP2.png)

```julia
# Method (3) complex
# CSP of the average covariance matrices
k=10
Xsetc=[genDataMatrix(ComplexF64, n, t) for i=1:k]
Ysetc=[genDataMatrix(ComplexF64, n, t) for i=1:k]

cc=csp(Xsetc, Ysetc)

# ... selecting subspace dimension allowing an explained variance = 0.9
cc=csp(Xsetc, Ysetc; eVar=0.9)

# ... subtracting the mean from the matrices in Xset and Yset
cc=csp(Xsetc, Ysetc; meanX₁=nothing, meanX₂=nothing, eVar=0.9)

# csp on the average of the covariance and cross-covariance matrices
# computed along dims 1
cc=csp(Xsetc, Ysetc; dims=1, eVar=0.9)

# name of the filter
cc.name
```
"""
function csp(C₁ :: SorH, C₂ :: SorH;
             eVar     :: TeVaro = ○,
             eVarC    :: TeVaro = ○,
             eVarMeth :: Function = searchsortedfirst,
             selMeth  :: Symbol = :extremal,
             simple   :: Bool = false)

  # checks
  size(C₁, 1)==size(C₁, 2) || throw(ArgumentError(📌*", csp function: Matrix `C₁` must be square"))
  size(C₂, 1)==size(C₂, 2) || throw(ArgumentError(📌*", csp function: Matrix `C₂` must be square"))
  size(C₁)==size(C₂) || throw(ArgumentError(📌*", csp function: Matrices `C₁` and `C₂` must have the same size"))
  eVar isa Int && eVarC isa Int && eVar>eVarC && throw(ArgumentError(📌*", csp function: `eVar` cannot be larger than `eVarC`"))

  args=("Common Spatial Pattern", false)

  if eVarC≠○ && eVarC≈0. # use gevd, which also actually whitens C1+C2
     λ, U = eig(C₁, C₁+C₂)
     λ = _checkλ(λ) # make sure no imaginary noise is present (for complex data)

     simple ? LF(U, inv(U), Diagonal(λ), ○, ○, ○, args...) :
     begin
        eVar, D, U, p, arev=_ssdcsp!(eVar, λ, U, _minDim(C₁, C₂), eVarMeth, selMeth) # subspace dimension
        LF(U, pinv(U), D, eVar, λ, arev, args...)
     end
  else
     w=whitening(C₁+C₂; eVar=eVarC, eVarMeth=eVarMeth)

     # alert user if eVar passed as an integer exceeds the whitening dim
     whiteDim=size(w.F, 2)
     if eVar isa Int && eVar>whiteDim
        @warn(📌*", csp function: the whitening step reduced the rank to $(whiteDim); `eVar` has been lowered to this value.")
        eVar=whiteDim
     end

     λ, U = eig(Hermitian(w.F'*C₁*w.F)) # get evd of whitened C1
     # Hermitian is necessary for complex data
     λ = _checkλ(λ) # make sure no imaginary noise is present (for complex data)

     simple ? LF(w.F*U, U'*w.iF, Diagonal(λ), ○, ○, ○, args...) :
     begin
        eVar, D, U, p, arev=_ssdcsp!(eVar, λ, U, _minDim(C₁, C₂), eVarMeth, selMeth) # subspace dimension
        LF(w.F*U, U'*w.iF, D, eVar, λ, arev, args...)
     end
  end
end



function csp(X₁ :: Mat, X₂ :: Mat;
             covEst   :: StatsBase.CovarianceEstimator = SCM,
             dims     :: Into = ○,
             meanX₁   :: Tmean = 0,
             meanX₂   :: Tmean = 0,
             wX₁      :: Tw = ○,
             wX₂      :: Tw = ○,
          eVar     :: TeVaro = ○,
          eVarC    :: TeVaro = ○,
          eVarMeth :: Function = searchsortedfirst,
          selMeth  :: Symbol = :extremal,
          simple   :: Bool = false)

   dims===○ && (dims=_set_dims(X₁, X₂))
   _check_data(X₁, dims, covEst, meanX₁, wX₁)===○ && return
   _check_data(X₂, dims, covEst, meanX₂, wX₂)===○ && return

   C₁=_cov(X₁, covEst, dims, meanX₁, wX₁)
   C₂=_cov(X₂, covEst, dims, meanX₂, wX₂)

   csp(C₁, C₂; eVar=eVar, eVarC=eVarC, eVarMeth=eVarMeth,
       selMeth=selMeth, simple=simple)
end


function csp(𝐗₁::VecMat, 𝐗₂::VecMat;
             covEst   :: StatsBase.CovarianceEstimator = SCM,
             dims     :: Into = ○,
             meanX₁   :: Into = 0,
             meanX₂   :: Into = 0,
          eVar     :: TeVaro = ○,
          eVarC    :: TeVaro = ○,
          eVarMeth :: Function = searchsortedfirst,
          selMeth  :: Symbol = :extremal,
          simple   :: Bool = false,
       metric   :: Metric = Euclidean,
       w₁       :: Vector = [],
       w₂       :: Vector = [],
       ✓w       :: Bool = true,
       init₁    :: SorHo = nothing,
       init₂    :: SorHo = nothing,
       tol      :: Real = 0.,
       verbose  :: Bool = false)

   Metric==VonNeumann && throw(ArgumentError(📌*", csp function: A solution for the mean is not available for the Von Neumann metric. Use another metric as `metric` argument"))
   dims===○ && (dims=_set_dims(𝐗₁, 𝐗₂))
   _check_data(𝐗₁, dims, covEst, meanX₁, ○)===○ && return
   _check_data(𝐗₂, dims, covEst, meanX₂, ○)===○ && return

   𝐂₁= _cov(𝐗₁; covEst=covEst, dims=dims, meanX=meanX₁)
   𝐂₂= _cov(𝐗₂; covEst=covEst, dims=dims, meanX=meanX₂)

   csp(mean(metric, 𝐂₁;
            w = w₁, ✓w = ✓w,
            init = init₁===○ ? ○ : Hermitian(init₁), #just init₁ here when you upfate PodDefManifold
            tol = tol, verbose = verbose),
       mean(metric, 𝐂₂;
            w = w₂, ✓w = ✓w,
            init = init₂===○ ? ○ : Hermitian(init₂), #just init₂ here when you upfate PodDefManifold
            tol = tol, verbose = verbose);
       eVar=eVar, eVarC=eVarC, eVarMeth=eVarMeth,
       selMeth=selMeth, simple=simple)
end



"""
```julia
(1)
function cstp( X :: Mat, C₍₁₎ :: SorH, C₍₂₎ :: SorH;
               eVar     :: TeVaro = ○,
               eVarC    :: TeVaro = ○,
               eVarMeth :: Function = searchsortedfirst,
               simple   :: Bool = false)

(2)
function cstp( 𝐗::VecMat;
               covEst   :: StatsBase.CovarianceEstimator = SCM,
               meanXd₁  :: Into = 0,
               meanXd₂  :: Into = 0,
            eVar     :: TeVaro = ○,
            eVarC    :: TeVaro = ○,
            eVarMeth :: Function = searchsortedfirst,
            simple   :: Bool = false,
         metric   :: Metric = Euclidean,
         w        :: Vector = [],
         ✓w       :: Bool = true,
         init1    :: SorHo = nothing,
         init2    :: SorHo = nothing,
         tol      :: Real = 0.,
         verbose  :: Bool = false)
```

Return a [LinearFilter](@ref) object:

(1)
**Common spatio-temporal pattern**
with ``n⋅m`` mean data matrix `X`,
``m⋅m`` covariance matrices `C₍₁₎` and ``n⋅n`` covariance matrix `C₍₂₎` as input.


`eVar`, `eVarC` and `eVarMeth` are keyword optional arguments
for defining the [subspace dimension](@ref) ``p``. Particularly:
-  `eVarC` is used for defining the subspace dimension of
   the whitening step. The default is 0.999.
- `eVar` is the keyword optional argument for defining the
   [subspace dimension](@ref) ``p`` using the `.arev` vector
   given by [cstp.5]. The default is given in [cstp.6] here above.
- `eVarMeth` applies to both `eVarC` and `eVar`. The default value is
   `evarMeth=searchsortedfirst`.

If `simple` is set to `true`, ``p`` is set equal to ``n``
and only the fields `.F` and `.iF` are written in the constructed object.
This option is provided for low-level work when you don't need to define
a subspace dimension or you want to define it by your own methods.

(2)
**Common spatio-temporal pattern**
with a set of ``k`` data matrices `𝐗` as input.

The ``k`` matrices in `𝐗` are real or complex data matrices.
They must all have the same size.

`covEst`, `meanXd₁` and `meanXd₂` are optional
keyword arguments to regulate the estimation of the
covariance matrices of the data matrices in `𝐗`,
to be used to compute the mean covariance matrices in [cstp.2] here above.
See [covariance matrix estimations](@ref).
`meanXd₁` and `meanXd₂` are the means along dimension 1 and 2, respectively,
of the data matrices in `𝐗`.

The mean covariance matrices ``C_{(1)}`` and ``C_{(1)}`` in [cstp.2]
are computed using optional keywords arguments
`metric`, `w`, `✓w`, `init1`, `init2`, `tol` and `verbose`,
which allow to compute non-Euclidean means.
Particularly (see [mean covariance matrix estimations](@ref)),
- `init1` is the initialization for ``C_{(1)}``,
- `init2` is the initialization for ``C_{(2)}``.
By default, the arithmetic means [cstp.2] are computed.

Once the two covariance matrices ``C_{(1)}`` and ``C_{(2)}`` estimated,
method (1) is invoked with optional keyword arguments
`eVar`, `eVarC`, `eVarMeth` and `simple`.
See method (1) for details.

**Examples:**

```julia
using Diagonalizations, LinearAlgebra, PosDefManifold, Test

# Method (1) real
t, n, k=10, 20, 10
Xset = [genDataMatrix(t, n) for i = 1:k]
Xfixed=randn(t, n)./1
for i=1:length(Xset) Xset[i]+=Xfixed end
C1=Hermitian( mean((X'*X)/t for X∈Xset) )
C2=Hermitian( mean((X*X')/n for X∈Xset) )
Xbar=mean(Xset)
c=cstp(Xbar, C1, C2; simple=true)
@test c.F[1]'*C2*c.F[1]≈I
@test c.F[2]'*C1*c.F[2]≈I
Z=c.F[1]'*Xbar*c.F[2]
n=minimum(size(Z))
@test norm(Z[1:n, 1:n]-Diagonal(Z[1:n, 1:n]))+1. ≈ 1.
cX=cstp(Xset; simple=true)
@test c==cX

# Method (1) complex
t, n, k=10, 20, 10
Xcset = [genDataMatrix(ComplexF64, t, n) for i = 1:k]
Xcfixed=randn(ComplexF64, t, n)./1
for i=1:length(Xcset) Xcset[i]+=Xcfixed end
C1c=Hermitian( mean((Xc'*Xc)/t for Xc∈Xcset) )
C2c=Hermitian( mean((Xc*Xc')/n for Xc∈Xcset) )
Xcbar=mean(Xcset)
cc=cstp(Xcbar, C1c, C2c; simple=true)
@test cc.F[1]'*C2c*cc.F[1]≈I
@test cc.F[2]'*C1c*cc.F[2]≈I
Zc=cc.F[1]'*Xcbar*cc.F[2]
n=minimum(size(Zc))
@test norm(Zc[1:n, 1:n]-Diagonal(Zc[1:n, 1:n]))+1. ≈ 1.
cXc=cstp(Xcset; simple=true)
@test cc==cXc

# Method (2) real
c=cstp(Xset)

# ... selecting subspace dimension allowing an explained variance = 0.9
c=cstp(Xset; eVar=0.9)

# ... giving weights `w` to the covariance matrices
c=cstp(Xset; w=abs2.(randn(k)), eVar=0.9)

# ... subtracting the means
c=cstp(Xset; meanXd₁=nothing, meanXd₂=nothing, w=abs2.(randn(k)), eVar=0.9)

# explained variance
c.eVar

# name of the filter
c.name

using Plots
# plot the original covariance matrices and the transformed counterpart
c=cstp(Xset)

C1Max=maximum(abs.(C1));
 h1 = heatmap(C1, clim=(-C1Max, C1Max), title="C1", yflip=true, c=:bluesreds);
 D1=c.F[1]'*C2*c.F[1];
 D1Max=maximum(abs.(D1));
 h2 = heatmap(D1, clim=(0, D1Max), title="F[1]'*C2*F[1]", yflip=true, c=:amp);
 C2Max=maximum(abs.(C2));
 h3 = heatmap(C2, clim=(-C2Max, C2Max), title="C2", yflip=true, c=:bluesreds);
 D2=c.F[2]'*C1*c.F[2];
 D2Max=maximum(abs.(D2));
 h4 = heatmap(D2, clim=(0, D2Max), title="F[2]'*C1*F[2]", yflip=true, c=:amp);

XbarMax=maximum(abs.(Xbar));
 h5 = heatmap(Xbar, clim=(-XbarMax, XbarMax), title="Xbar", yflip=true, c=:bluesreds);
 DX=c.F[1]'*Xbar*c.F[2];
 DXMax=maximum(abs.(DX));
 h6 = heatmap(DX, clim=(0, DXMax), title="F[1]'*Xbar*F[2]", yflip=true, c=:amp);
 📈=plot(h1, h3, h5, h2, h4, h6, size=(800,400))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigCSTP.png")

```

 ![Figure CSTP](assets/FigCSTP.png)

```julia

# Method (2) complex
cc=cstp(Xcset)

# ... selecting subspace dimension allowing an explained variance = 0.9
cc=cstp(Xcset; eVar=0.9)

# ... giving weights `w` to the covariance matrices
cc=cstp(Xcset; w=abs2.(randn(k)), eVar=0.9)

# ... subtracting the mean
cc=cstp(Xcset; meanXd₁=nothing, meanXd₂=nothing,
        w=abs2.(randn(k)), eVar=0.9)

# explained variance
c.eVar

# name of the filter
c.name

```

"""
function cstp( X :: Mat, C₍₁₎ :: SorH, C₍₂₎ :: SorH;
               eVar     :: TeVaro = ○,
               eVarC    :: TeVaro = ○,
               eVarMeth :: Function = searchsortedfirst,
               simple   :: Bool = false)

   d₍₂₎, d₍₁₎, dₓ=size(C₍₂₎, 1), size(C₍₁₎, 1), size(X)
   (d₍₁₎==dₓ[2] && d₍₂₎==dₓ[1]) || throw(ArgumentError(📌*", cstp function: For n⋅m matrix X, matrix C₍₁₎ must be m⋅m and matrix C₍₂₎ n⋅n"))
   eVar isa Int && eVarC isa Int && eVar>eVarC && throw(ArgumentError(📌*", cstp function: `eVar` cannot be larger than `eVarC`"))
   args=("Common Spatio-Temporal Pattern", false)

   kwargs=(eVar=eVarC, eVarMeth=eVarMeth, simple=false)
   t=whitening(C₍₁₎; kwargs...)
   s=whitening(C₍₂₎; kwargs...)

   # alert user if eVar passed as an integer exceeds the minimum whitening dim
   whiteDim=min(size(t.F, 2), size(t.F, 2))
   if eVar isa Int && eVar>whiteDim
      @warn(📌*", cstp function: the whitening step reduced the rank to $(whiteDim); `eVar` has been lowered to this value.")
      eVar=whiteDim
   end

   U, λ, V = svd(s.F'*X*t.F; full=true)
   λ = _checkλ(λ) # make sure no imaginary noise is present (for complex data)

   simple ? LF([s.F*U, t.F*V], [U'*s.iF, V'*t.iF], Diagonal(λ), ○, ○, ○, args...) :
   begin
     eVar, D, U, V, p, arev=_ssdcstp!(eVar, λ, U, Matrix(V), _minDim(X), eVarMeth) # subspace dimension
     LF([s.F*U, t.F*V], [U'*s.iF, V'*t.iF], D, eVar, λ, arev, args...)
   end
end

# no dims argument since it is the same for cstp
function cstp( 𝐗::VecMat;
               covEst   :: StatsBase.CovarianceEstimator = SCM,
               meanXd₁  :: Into = 0,
               meanXd₂  :: Into = 0,
            eVar     :: TeVaro = ○,
            eVarC    :: TeVaro = ○,
            eVarMeth :: Function = searchsortedfirst,
            simple   :: Bool = false,
         metric   :: Metric = Euclidean,
         w        :: Vector = [],
         ✓w       :: Bool = true,
         init1     :: SorHo = nothing,
         init2     :: SorHo = nothing,
         tol      :: Real = 0.,
         verbose  :: Bool = false)

   Metric==VonNeumann && throw(ArgumentError(📌*", cstp function: A solution for the mean is not available for the Von Neumann metric. Use another metric as `metric` argument"))
   covEst==SCM && metric ∉ (Euclidean, Wasserstein) && throw(ArgumentError(📌*", cstp function: Only the Euclidean and Wasserstein `metric` can be used if the covariance estimator is `SCM`"))
   _check_data(𝐗, 1, covEst, meanXd₁, ○)===○ && return
   _check_data(𝐗, 2, covEst, meanXd₂, ○)===○ && return

   𝐂₍₁₎=_cov(𝐗; covEst=covEst, dims = 1, meanX = meanXd₁)
   𝐂₍₂₎=_cov(𝐗; covEst=covEst, dims = 2, meanX = meanXd₂)

   cstp(PosDefManifold.fVec(mean, Vector{Matrix}(𝐗)), # multi-threaded Euclidean mean
        mean(metric, 𝐂₍₁₎;
             w = w, ✓w = ✓w,
             init = init1===○ ? ○ : Hermitian(init1), #just init here when you update PodDefManifold
             tol = tol, verbose = verbose),
        mean(metric, 𝐂₍₂₎;
             w = w, ✓w = ✓w,
             init = init2===○ ? ○ : Hermitian(init2), #just init here when you update PodDefManifold.jl
             tol = tol, verbose = verbose),
        eVar     = eVar,
        eVarC    = eVarC,
        eVarMeth = eVarMeth,
        simple   = simple)
end
