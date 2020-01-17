#   Unit "tools.jl" of the Diagonalization.jl Package for Julia language
#
#   MIT License
#   Copyright (c) 2019, 2020
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements general tools and internal functions.

"""
```
function eig(A)

function eig(A, B)
```
Call Julia function [eigen](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigen)
and return its output sorted by descending
order of eigenvalues.
"""
function eig(A)
   λ, U=eigen(A)
   return (reverse(λ), reverse(U, dims=2))
end

function eig(A, B)
   λ, U=eigen(A, B)
   return (reverse(λ), reverse(U, dims=2))
end


"""
```
function nonDiagonality(C::Union{Matrix, Diagonal, SorH})
```

Measure of deviancy from diagonality of ``n⋅n`` square matrix `C`, defined as
(Congedo et al., 2008)[🎓](@ref).

``\\frac{\\sum_{i≠j}|c_{ij}|^2}{(n-1)\\sum_{i}|c_{ii}|^2}``

It is equal to ``0`` if ``C`` is diagonal, equal to ``1`` if
``C`` is perfectly uniform.

**Examples:**
```
using Diagonalizations
C=ones(10, 10)                   # uniform matrix
nd=nonDiagonality(C)             # must be 1
D=Diagonal(abs.(randn(10, 10)))  # diagonal matrix
nd=nonDiagonality(D)             # must be 0
```
"""
function nonDiagonality(C::Union{Matrix, Diagonal, SorH})
   n = size(C, 1)
   n ≠ size(C, 2) && throw(ArgumentError("📌, nonDiagonality function: input matrix must be square"))
   ssDiag=sumOfSqrDiag(C)
   return ((sumOfSqr(C)-ssDiag)/ssDiag)/(n-1)
end
nonD=nonDiagonality


"""
```
function spForm(P::Union{Mat, Real, Complex})
```
Measure of deviancy from scaled permutation form of ``n⋅n`` square matrix
`P`, defined as

``\\frac{1}{2(n-1)}\\bigg(\\sum_{row}1-\\beta(row)+\\sum_{col}1-\\beta(col)\\bigg)``,

where for each *row* and *column* of `P`, β is the maximum of the
absolute values divided by the sum of the absolute values.

This index is equal to ``0`` if in each row and column ``P``
has only one non-zero element, that is, if ``P`` is a scaled permutation matrix.
The larger the index, the farther away ``P`` is from this form.

This measure and several existing variants are well-known in the blind source
separation / independent component analysis community,
where it is used to compare approximate joint diagonalization
algorithms on simulated data. In fact, if ``A`` is the inverse of the
approximate joint diagonalizer that is used to generate the data
and ``B`` the approximate joint diagonalizer estimated by an algorithm,
``P=BA`` must be as close as possible to a scaled permutation matrix
(see [scale and permutation](@ref)).

Return 0.0 (zero) if `P` is a real of complex number.

**Examples:**

```
using Diagonalizations, PosDefManifold
# create 20 random commuting matrices
# they all have the same eigenvectors
Cset=randP(3, 20; eigvalsSNR=Inf, commuting=true)
# estimate the approximate joint diagonalizer (ajd)
a=ajd(Cset)
# the ajd must be equivalent to the eigenvector matrix of
# any of the matrices in Cset
spForm(a.F'*eigvecs(Cset[1]))+1.0≈1.0 ? println(" ⭐ ") : println(" ⛔ ")
```

"""
function spForm(P::Union{Mat, Real, Complex})
   if P isa Number return 0.0 end
   r, c=size(P)
   r ≠ c && throw(ArgumentError("📌, spForm function: input matrix must be square"))
   mos(v::AbstractArray)=1.0-(maximum(v)/sum(v)) # 1- max over sum of a vector
   (sum(mos(abs.(p)) for p∈eachcol(P)) + sum(mos(abs.(p)) for p∈eachrow(P)))/(2*(r-1))
end



"""
```
(1)
function genDataMatrix(t::Int, n::Int, A=nothing)

(2)
function genDataMatrix(::Type{Complex{T}},
                       t::Int, n::Int, A=○) where {T<:AbstractFloat}
```

(1)
Generate a ``t⋅n`` random data matrix as ``XA``,
where ``X`` is a ``t⋅n`` matrix with entries randomly drawn
from a Gaussian distribution and ``A`` a ``n⋅n`` symmetric
matrix, which, if not provided as argument `A`,
will be generated with entries randomly drawn from a uniform
distribution ∈[-1, 1].

(2)
as (1), but ``X`` is generated randomly from a complex
Gaussian distribution and ``A``is Hermitian (complex)
which, if not provided as argument `A`,
will be generated with entries randomly drawn from a uniform
distribution ∈[-1-i1, 1+i1].

**Examples:**

A=genDataMatrix(100, 20) # (1), real
A=genDataMatrix(ComplexF64, 100, 20) # (2), complex

"""
function genDataMatrix(t::Int, n::Int, A=○)
   if A===○ A=Symmetric((rand(n, n).-0.5).*2) end
   return randn(t, n)*A
end

function genDataMatrix(::Type{Complex{T}},
                       t::Int, n::Int, A=○) where {T<:AbstractFloat}
  if A===○ A=Hermitian([(rand(T).-(0.5+0.5im)).*2. for i=1:n, j=1:n]) end
  return randn(T, t, n)*A
end
# -------------------------------------------------------- #
# INTERNAL FUNCTIONS #
# -------------------------------------------------------- #


# EigenDecomposition with a covariance matrix as input
function _getEVD(C :: Union{Hermitian, Symmetric, Mat}, eVar::TeVaro,
                 eVarMeth::Function, simple::Bool)

   λ, U = eig(C)
   λ=_checkλ(λ) # make sure no imaginary noise is present (fro complex data)
   simple ? (U, Matrix(U'), Diagonal(λ), ○, ○, ○) :
   begin
     eVar===○ ? eVar=0.999 : ○
     eVar, D, U, p, arev=_ssd!(eVar, λ, U, _minDim(C), eVarMeth)
     (U, Matrix(U'), D, eVar, λ, arev)
   end
end


# EigenDecomposition with a data matrix as input
_getEVD(X::Mat, covEst::StatsBase.CovarianceEstimator, dims::Int64,
        mean::Tmean, w::Tw, eVar::TeVaro, eVarMeth::Function, simple::Bool) =
  _getEVD(_cov(X, covEst, dims, mean, w), eVar, eVarMeth, simple)


# Whitening with a covariance matrix as input
function _getWhi(C :: Union{Hermitian, Symmetric, Mat}, eVar::TeVaro,
                 eVarMeth::Function, simple::Bool)

  U, Uⁱ, D, eVar, λ, arev=_getEVD(C, eVar, eVarMeth, simple)
  simple ? (U*D^-0.5, D^0.5*Uⁱ, D, ○, ○, ○) :
           (U*D^-0.5, D^0.5*Uⁱ, D, eVar, λ, arev)
end

# Whitening with a data matrix as input
_getWhi(X::Mat, covEst::StatsBase.CovarianceEstimator, dims::Int64,
        mean::Tmean, w::Tw, eVar::TeVaro, eVarMeth::Function, simple::Bool) =
   _getWhi(_cov(X, covEst, dims, mean, w), eVar, eVarMeth, simple)


# convert mean vector for compatibility with StatsBase.jl
function _convert_mean(mean::Tmean, dims::Int, argName::String)
  length(mean)≠n && throw(ArgumentError(📌*", _convert_mean internal function: vector "*argName*" must have length $n"))
  return dims==1 ? Matrix(mean') : mean
end

# return `X` or `X` with the mean subtracted, depending on `meanX`
function _deMean(X::Mat, dims::Int, meanX::Tmean)
   if       meanX isa Int
            return X
   elseif   meanX===○
            meanX_=mean(X; dims=dims)
   elseif   meanX isa AbstractVector
            meanX_=_convert_mean(meanX, dims, "meanX")
   end
   #println("dims ", dims, "  sizemeanX_ ", size(meanX_), " sizeX ", size(X))
   if       dims==1
            s=(1, size(X, 2))
   elseif   dims==2
            s=(size(X, 1), 1)
   end
   size(meanX_)≠s && throw(ArgumentError(📌*", _deMean internal function: The size of `meanX_` does not fit input matrix `X` with `dims`=$dims"))
   return X.-meanX_
end

# check arguments for one data matrix input
function _check_data(X::Mat, dims::Int64, covEst::StatsBase.CovarianceEstimator, meanX::Tmean, wX::Tw)
   dims ∈ (1, 2) || throw(ArgumentError(📌*", _check-data internal function: Argument `dims` may be 1 or 2. dims=$dims"))
   wX≠○ && lenght(wX)≠size(X, dims) && throw(ArgumentError(📌*", _check-data internal function: The size of `wX` does not fit input matrix `X` with `dims`=$dims"))
   eltype(X)<:Complex && covEst≠SCM && throw(ArgumentError(📌*", _check-data internal function: Only the `SCM` (sample covariance matrix) `covEst` estimator can be used for complex data"))
   # TODO add check for meanX
   ishermitian(X) && throw(ArgumentError(📌*", _check-data internal function: it looks like
   you want to call a filter constuctor that takes covariance matrices as input,
   but you are actually calling the constructor that takes data matrices as input.
   Solution: flag your covariance matrix(ces) argument(s) as Symmetric or Hermitian,
   for example, `Hermitian(C)`. To do so, you will need to be using LinearAlgebra."))
   return true
end

# check arguments for two data matrices input
function _check_data(X::Mat, Y::Mat, dims::Int64, covEst::StatsBase.CovarianceEstimator, meanX::Tmean, meanY::Tmean, wXY::Tw)
   dims ∈ (1, 2) || throw(ArgumentError(📌*", _check-data internal function: Argument `dims` may be 1 or 2. dims=$dims"))
   size(X, dims)==size(Y, dims) || throw(ArgumentError(📌*", _check-data internal function: The `dims` dimension of argument `X` and `Y` must be the same"))
   wXY≠○ && lenght(wXY)≠size(X, dims) && throw(ArgumentError(📌*", _check-data internal function: The size of `wXY` does not fit input matrix `X` with `dims`=$dims"))
   (eltype(X)<:Complex || eltype(Y)<:Complex) && covEst≠SCM && throw(ArgumentError(📌*", _check-data internal function: Only the `SCM` (sample covariance matrix) `covEst` estimator can be used for complex data"))
   # TODO add check for meanX and meanY
   return true
end


# check arguments for one vector of data matrix input
function _check_data(𝐗::VecMat, dims::Int64, covEst::StatsBase.CovarianceEstimator, meanX::Into, w::Twf)
   dims ∈ (1, 2) || throw(ArgumentError(📌*", _check-data internal function: Argument `dims` may be 1 or 2. dims=$dims"))
   x=collect(size(X, _flip12(dims)) for X ∈ 𝐗)
   all(y->y==x[1], x) ||  throw(ArgumentError(📌*", _check-data internal function: Given `dims`=$dims, all matrices in `𝐗` must have the same dimension $(_flip12(dims))"))
   w≠○ && !(w isa Function) && lenght(w)≠length(𝐗) && throw(ArgumentError(📌*", _check-data internal function: The size of `w` must be equal to the number of matrices in `𝐗`"))
   !isempty(findall(x->eltype(x)<:Complex, 𝐗)) && covEst≠SCM && throw(ArgumentError(📌*", _check-data internal function: Only the `SCM` (sample covariance matrix) `covEst` estimator can be used for complex data"))
   # TODO add check for meanX
   return true
end

# check arguments for two vectors of data matrix input
function _check_data(𝐗::VecMat, 𝐘::VecMat, dims::Int64, covEst::StatsBase.CovarianceEstimator, meanX::Into, meanY::Into, w::Twf)
   dims ∈ (1, 2) || throw(ArgumentError(📌*", _check-data internal function: Argument `dims` may be 1 or 2. dims=$dims"))
   length(𝐗)==length(𝐘) || throw(ArgumentError(📌*", _check-data internal function: The number of matrices in `𝐗` and `𝐘` must be the same"))
   x=collect(size(X, dims)-size(Y, dims) for (X, Y) ∈ (𝐗, 𝐘))
   norm(x)==0 || throw(ArgumentError(📌*", _check-data internal function: The `dims` dimension of all pairs of matrices in `𝐗` and `𝐘` must be the same"))
   w≠○ && !(w isa Function) && lenght(w)≠length(𝐗) && throw(ArgumentError(📌*", _check-data internal function: The size of `w` must be equal to the number of matrices in `𝐗` and `𝐘`"))
   (!isempty(findall(x->eltype(x)<:Complex, 𝐗)) || !isempty(findall(y->eltype(y)<:Complex, 𝐘))) && covEst≠SCM && throw(ArgumentError(📌*", _check-data internal function: Only the `SCM` (sample covariance matrix) `covEst` estimator can be used for complex data"))
   # TODO add check for meanX and meanY
   return true
end

#TODO _check_data(𝑿::VecVecMat, dims::Int64, covEst::StatsBase.CovarianceEstimator, meanX::Into, ○)===○ && return

# call StatsBase.cov within one line with or without weights
# Also, flag the covariance as Symmetric if is real, Hermitian if is complex
# The mean is subtracted separatedly for consistence with the other _cov method
# NB!!! covarianceestimations.jl does not work for complex data input!
function _cov(X::Matrix{R},
              covEst   :: StatsBase.CovarianceEstimator = SCM,
              dims     :: Int64 = 1,
              meanX    :: Tmean = 0,
              wX       :: Tw = ○) where R<:Real
   #T = R isa Real ? Symmetric : Hermitian
   X_=_deMean(X, dims, meanX)
   return wX===○ ? Symmetric(cov(covEst, X_; dims=dims, mean=0)) : # do NOT remove `mean`=0
                   Symmetric(cov(covEst, X_, wX; dims=dims, mean=0)) # "
end

# `covest` is not used but left in for code homogeneity
function _cov(X::Matrix{R},
              covEst   :: StatsBase.CovarianceEstimator = SCM,
              dims     :: Int64 = 1,
              meanX    :: Tmean = 0,
              wX       :: Tw = ○) where R<:Complex
   X_=_deMean(X, dims, meanX)
   wX≠○ ? ( dims==1 ? X__=wX.*X_ : X__=wX'.*X_ ) : X__=X_
   return dims==1 ? Hermitian((X__'*X__)/size(X, 1)) : Hermitian((X__*X__')/size(X, 2))
end

# as before for a vector of data matrices at once
# NB!!! does not work for complex data input as uses the above method!
function _cov(𝐗::VecMat;
              covEst   :: StatsBase.CovarianceEstimator = SCM,
              dims     :: Int64 = 1,
              meanX    :: Into = 0)
   # once PosDefManifold supports vectors of Symmetric matrices
   # T = R===Real ? Symmetric : Hermitian
   # remove `Hermitian` here below and use T instead
   # _cov will automatically flag its output
   𝐂=Vector{Hermitian}(undef, length(𝐗))
   @threads for i=1:length(𝐗)
               𝐂[i]=Hermitian(_cov(𝐗[i], covEst, dims, meanX, ○))
   end
   return 𝐂
end


# cross-covariance within one line with or without weights
# The mean is subtracted separately since there is no crosscov method in StatsBase
function _cov(X::Matrix{R}, Y::Matrix{R},
              dims     :: Int64 = 1,
              meanX    :: Tmean = 0,
              meanY    :: Tmean = 0,
              wXY      :: Tw = ○) where R<:Union{Real, Complex}
   (size(X, dims) ≠ size(Y, dims)) && throw(ArgumentError(📌*", _cov internal function: the size of matrices `X` and `Y` are not conform for computing cross-covariance with $dims as value of `dims`"))
   X_=_deMean(X, dims, meanX)
   Y_=_deMean(Y, dims, meanY)
   return wXY===○ ? ( dims==1 ? (X_'*Y_)/size(X, 1) : (X_*Y_')/size(X_, 2) ) :
                    ( dims==1 ? ((wXY'.*X_')*Y_)/wXY.sum : ((wXY'.*X_)*Y_')/wXY.sum )
end

# The same as above, for 2 vectors of data matrices at once
# the cross-covariance if computed for all corresponding pairs
function _cov(𝐗::VecMat, 𝐘::VecMat;
              dims     :: Int64 = 1,
              meanX    :: Into = 0,
              meanY    :: Into = 0)
   (length(𝐗)≠length(𝐘)) && throw(ArgumentError(📌*", _cov internal function: vectors 𝐗 and 𝐘 must hold the same number of data matrices"))
   𝐂=Vector{Matrix}(undef, length(𝐗))
   @threads for i=1:length(𝐗)
               𝐂[i]=_cov(𝐗[i], 𝐘[i], dims, meanX, meanY, ○)
            end
   return 𝐂
end

# trace normalize and/or apply weights. Accept a function for computing weights
# only for m=1
function _Normalize!(𝐂::Vector{Hermitian},
                     trace1::Bool=false, w::Union{Tw, Function}=○)
   !trace1 && w===○ && return
   k=length(𝐂)

   if trace1
      @inbounds for κ=1:k 𝐂[κ] = tr1(𝐂[κ]) end
   end
   if w isa Function
      w=[w(𝐂[κ]) for κ=1:k]
   end
   if w ≠ ○
      @inbounds for κ=1:k 𝐂[κ] *= w[κ] end
   end
end

# trace normalize and/or apply weights. Accept a function for computing weights
# m>=1, k>=1. 𝒞 is a 3-D Array of matrices (k, i, j), i, j=1:m
function _Normalize!(𝒞::AbstractArray, m::Int, k::Int,
                     trace1::Bool=false, w::Union{Tw, Function}=○)
   !trace1 && w===○ && return

   #for κ=1:k println(tr(𝒞[κ, 1, 1])) end

   if m==1
      if trace1
         @inbounds for κ=1:k 𝒞[κ, 1, 1] = tr1(𝒞[κ, 1, 1]) end
      end
      if w isa Function
         w=[w(𝒞[κ, 1, 1]) for κ=1:k]
      end
      if w ≠ ○
         @inbounds for κ=1:k 𝒞[κ, 1, 1] *= w[κ] end
      end
   else
      for κ=1:k
         if trace1
               t=[1/sqrt(tr(𝒞[κ, i, i])) for i=1:m]
         elseif w ≠ ○
               t=ones(eltype(𝒞[1, 1, 1]), m)
         end
         if     w isa Function
                  @inbounds for i=1:m t[i]*=sqrt(w(𝒞[κ, i, i])) end
         elseif w isa StatsBase.AbstractWeights
                  @inbounds for i=1:m t[i]*=sqrt(w[i]) end
         end
         if trace1 || w ≠ ○
           @inbounds for i=1:m, j=i:m 𝒞[κ, i, j] = 𝒞[κ, i, j]*(t[i]*t[j]) end
         end
      end
   end
end


# if     m=1 𝐗 is a vector of k data matrices.
#           Return a kx1x1 array of their covariance matrices in the k dimension
# elseif k=1 𝐗 is a vector of m data matrices.
#           Return a 1xmxm array of all cross-covariances of 𝐗[i] and 𝐗[j], for i,j=1:m
# elseif 𝐗 is a k-vector of m data matrices.
#           Return a kxmxm array of all cross-covariances of 𝐗[l][i] and 𝐗[l][j], for l=1:k, i,j=1:m
function _crossCov(𝐗, m, k;
                   covEst  :: StatsBase.CovarianceEstimator=SCM,
                   dims    :: Int64 = 1,
                   meanX   :: Into = 0,
                   trace1  :: Bool = false,
                   w       :: Union{Tw, Function}=○)
    𝒞=Array{Matrix}(undef, k, m, m)
    if      m==1
      @inbounds for κ=1:k 𝒞[κ, 1, 1] = _cov(𝐗[κ], covEst, dims, meanX, ○) end
    elseif  k==1
      @inbounds for i=1:m-1, j=i+1:m
                        𝒞[1, i, j] = _cov(𝐗[i], 𝐗[j], dims, meanX, meanX, ○)
                        𝒞[1, j, i] = 𝒞[1, i, j]'
                end
      @inbounds for i=1:m 𝒞[1, i, i] = _cov(𝐗[i], covEst, dims, meanX, ○) end # This is needed for scaling in any case
    else
      @inbounds for κ=1:k, i=1:m-1, j=i+1:m
                        𝒞[κ, i, j] = _cov(𝐗[κ][i], 𝐗[κ][j], dims, meanX, meanX, ○)
                        𝒞[κ, j, i]=𝒞[κ, i, j]'
                end
      @inbounds for κ=1:k, i=1:m 𝒞[κ, i, i] = _cov(𝐗[κ][i], covEst, dims, meanX, ○) end # This is needed for scaling in any case
    end

    # trace normalize
    if trace1 || w ≠ ○ _Normalize!(𝒞, m, k, trace1, w) end

    return 𝒞
end



# get index and value of the
# first value in 𝜆 greater than or equal to eVar (eVarMeth=searchsortedfirst) or
# last value in 𝜆 less than or equal to eVar (eVarMeth=searchsortedlast),
# where 𝜆 is the vector with accumulated regularized (sum-normalized) eigenvalues
# INPUT:
# the desired explained variance (real) of subspace dimension (int) (evar),
# the eigenvalues in descending order (λ),
# the corresponding eigenvectors (U),
# the maximum theoretical rank of the input matrix (r),
# the method (eVarMeth function) for determining the subspace dimension.
# OUTPUT:
# the actual explained variance (evar!),
# the first p eigenvalues (λ!),
# the corresponding first p eigenvectors (U!),
# the subspace dimension (p),
# the vector with the accumulated regularized eigenvalues (arev)

function _getssd!(eVar::TeVaro, λ::Vec, r::Int64, eVarMeth::Function)
   eltype(λ)<:Complex && @warn "📌, internal function `_getssd!`: the `λ` vector is complex, subspace dimension is based on its absolute values."
   eVar===○ ? eVar=0.999 : ○
   λ_=abs.(λ)
   arev = accumulate(+, λ_./sum(λ_))
   return (eVar isa Int64 ? clamp(eVar, 1, r) : clamp(eVarMeth(arev, eVar), 1, r), arev)
end

#see PCA and Whitening
function _ssd!(eVar::TeVaro, λ::Vec, U::Mat, r::Int64, eVarMeth::Function)
   eltype(λ)<:Complex && @warn "📌, internal function `_ssd!`: the `λ` vector is complex, subspace dimension is based on its absolute values."
   p, arev = _getssd!(eVar, λ, r, eVarMeth)
   return p==r ? 1. : arev[p], Diagonal(λ[1:p]), U[:, 1:p], p, arev
end

#see PMCA and CCA
function _ssdxy!(eVar::TeVaro, λ::Vec, U1::Mat, U2::Mat, r::Int64, eVarMeth::Function)
   eltype(λ)<:Complex && @warn "📌, internal function `_ssdxy!`: the `λ` vector is complex, subspace dimension  is based on its absolute values."
   p, arev = _getssd!(eVar, λ, r, eVarMeth)
   return p==r ? 1. : arev[p], Diagonal(λ[1:p]), U1[:, 1:p], U2[:, 1:p], p, arev
end

# see CSP
function _ssdcsp!(eVar::TeVaro, λ::Vec, U::Mat, r::Int64, eVarMeth::Function, selMeth::Symbol)
   eltype(λ)<:Complex && @warn "📌, internal function `_ssdcsp!`: the `λ` vector is complex, subspace dimension  is based on its absolute values."
   λ_=abs.(λ)
   ratio = λ_./(1.0.-λ_)
   d = (log.(ratio)).^2
   h = selMeth==:extremal ? sortperm(d; rev=true) : [i for i=1:length(λ)]
   arev = accumulate(+, d[h]./sum(d))
   if     eVar isa Int
      p = clamp(eVar, 1, r)
   elseif eVar isa Real
      p = clamp(eVarMeth(arev, eVar), 1, r)
   else    #eVar isa nothing, the default
      if selMeth==:extremal
         g=exp(sum(log, d)/length(d))
         p = clamp(searchsortedlast(d[h], g; rev=true), 1, r)
      else
         p = clamp(searchsortedlast(ratio, 1; rev=true), 1, clamp(argmin(d), 1, r))
      end
   end
   return p==r ? 1. : arev[p], Diagonal(λ[h[1:p]]), U[:, h[1:p]], p, arev
end

# see CSTP
function _ssdcstp!(eVar::TeVaro, λ::Vec, U::Mat, V::Mat, r::Int64, eVarMeth::Function)
   eltype(λ)<:Complex && @warn "📌, internal function `_ssdcstp!`: the `λ` vector is complex, subspace dimension is based on its absolute values."
   λ_=abs.(λ)
   arev = accumulate(+, λ_./sum(λ_))
   if     eVar isa Int
      p = clamp(eVar, 1, r)
   elseif eVar isa Real
      p = clamp(eVarMeth(arev, eVar), 1, r)
   else    #eVar isa nothing, the default
      p = clamp(eVarMeth(arev, 0.999), 1, r)
   end
   return p==r ? 1. : arev[p], Diagonal(λ[1:p]), U[:, 1:p], V[:, 1:p], p, arev
end


_flip12(i::Int) =
   if      i==1 return 2
   elseif  i==2 return 1
   else throw(ArgumentError, 📌*", _flip12 internal function: the `dims` argument must be 1 or 2")
   end


_set_dims(X::Mat)=argmax(collect(size(X)))
_set_dims(X::Mat, Y::Mat)=argmax(collect(size(X))+collect(size(Y)))
_set_dims(𝐗::VecMat)=argmax(sum(collect(size(X)) for X ∈ 𝐗))
_set_dims(𝐗::VecMat, 𝐘::VecMat)=
    argmax(sum(collect(size(X)) for X ∈ 𝐗)+sum(collect(size(Y)) for Y ∈ 𝐘))
_set_dims(𝑿::VecVecMat)=argmax(sum(collect(size(𝑿[i][j])) for i=1:length(𝑿) for j=1:length(𝑿[i])) )


_minDim(X::Matrix) = minimum(size(X))
_minDim(X::Matrix, Y::Matrix) = min(minimum(size(X)), minimum(size(Y)))
_minDim(C::SorH) = size(C, 1)
_minDim(𝐂::ℍVector) = minimum(size(C, 1) for C ∈ 𝐂)
_minDim(C1::SorH, C2::SorH) = min(size(C1, 1), size(C2, 1))
_minDim(𝐗::VecMat) = minimum(minimum(size(X)) for X ∈ 𝐗)
_minDim(𝐗::VecMat, 𝐘::VecMat) = min(_minDim(𝐗), _minDim(𝐘))
_minDim(𝑿::VecVecMat) = minimum((minimum(minimum(size(X)) for X ∈ 𝑿[i]) for i=1:length(𝑿)))


### tools for AJD Algorithms ###

# get the maximum number of iterations for each algorithm depending on the
# data input type if the algorithm supports both real and complex data input
_maxiter(algorithm, type) =
   if       algorithm ∈ (:OJoB, :NoJoB)
            return type<:Real ? 1000 : 3000
   elseif   algorithm ∈ (:LogLike, :LogLikeR, :JADE)
            return type<:Real ? 60 : 180
   else throw(ArgumentError("The `algorithm` keyword is uncorrect"))
   end


# take as input the vector `λ` of diagonal elements of transformed diagonalized
# matrices. Check that the imaginary part of λ is close to zero.
# If so, return a vector with the real part of λ,
# otherwise print a warning and return λ.
function _checkλ(λ::Vec)
   rePart=sum(real(λ).^2)
   imPart=sum(imag(λ).^2)
   if imPart/rePart > 1e-6
      @warn "📌, internal function _permute!: sum_κ=1^k trace(imag(Diagonal(F'*C_k*F))^2) / trace(imag(Diagonal(F'*C_k*F))^2) > 1e-6. The elements of fields `D`, `ev` and `arev` of the constructed LinearFilter will be complex. See documentation"
      return λ
   else
      return real(λ)
   end
end


# try to resolve the permutation for the output of AJD algorithms
# for the case m=1
# return a vector holding the n 'average eigenvalues' λ1,...,λn,
# arranging them in average descending order,
# where λη=𝛍_i=1:k(Di[η, η])
function _permute!(U::AbstractArray, D::Diagonal, n::Int)
   type=eltype(D)

   function flipcol!(U::AbstractArray, η::Int, e::Int)
      temp=U[:, e]
      U[:, e]=U[:, η]
      U[:, η]=temp
   end

   for e=1:n  # for all variables find the position of the absolute maximum
      p, max=e, zero(real(type))
      for η=e:n
           absd=abs(D[η, η])
           if  absd > max
               max = absd
               p=η
           end
      end

      # Bring the maximum from position η on top (current e)
      if p≠e
           flipcol!(U, p, e)
           d=D[p, p]
           D[p, p]=D[e, e]
           D[e, e]=d
      end
   end

   return diag(D)
end


function _permute!(U::AbstractArray, 𝐗::AbstractArray,
                   k::Int, input::Symbol;
    covEst   :: StatsBase.CovarianceEstimator=SCM,
    dims     :: Int64 = 1,
    meanX    :: Tmean = 0,
    trace1   :: Bool = false)
    # if n==t the input is assumed to be the covariance matrices
    input==:d ? 𝒞=_crossCov(𝐗, 1, k;
                    covEst=covEst, dims=dims, meanX=meanX, trace1=trace1) :
                𝒞=𝐗
    n=size(𝒞[1, 1, 1], 1)

    D=𝛍(𝔻([U[:, η]'*𝒞[l, 1, 1]*U[:, η] for η=1:n]) for l=1:k)

    return _permute!(U, D, n)
end # function _Permute!



# try to resolve scaling and permutation for the output of mAJD algorithms
# for the case m>1
# return a vector holding the n 'average eigenvalues' λ1,...,λn,
# trying to make them all positive and in descending order as much as possible,
# where λη=𝛍_i≠j=1:m(Dij[η, η])
function _scaleAndPermute!( 𝐔::AbstractArray, 𝐗::AbstractArray,
                            m::Int, k::Int, input::Symbol;
                            covEst   :: StatsBase.CovarianceEstimator=SCM,
                            dims     :: Int64 = 1,
                            meanX    :: Tmean = 0,
                            trace1   :: Bool = false)
    # if input ≠ :d the input is assumed to be the covariance matrices
    input==:d ? 𝒞=_crossCov(𝐗, m, k;
                    covEst=covEst, dims=dims, meanX=meanX, trace1=trace1) :
                𝒞=𝐗
    n=size(𝒞[1, 1, 1], 1)

    𝑫=𝔻Vector₂(undef, m)
    for i=1:m 𝑫[i]=𝔻Vector([𝛍(𝔻([𝐔[i][:, η]'*𝒞[l, i, j]*𝐔[j][:, η] for η=1:n]) for l=1:k) for j=1:m]) end
    p, type=(1, 1, 1), eltype(𝑫[1][1])

    function flipcol!(𝐔::AbstractArray, m::Int, η::Int, e::Int)
        for i=1:m
            temp=𝐔[i][:, e]
            𝐔[i][:, e]=𝐔[i][:, η]
            𝐔[i][:, η]=temp
        end
    end

    for e=1:n  # for all variables  (e.g., electrodes)
        # find the position of the absolute maximum
        max=zero(real(type))
        for i=1:m-1, j=i+1:m, η=e:n
            absd=abs(𝑫[i][j][η, η])
            if  absd > max
                max = absd
                p=(i, j, η)
            end
        end

        # flip sign of 𝐔[j][η, η] if abs max is negative
        i=p[1]; j=p[2]; η=p[3]
        if real(𝑫[i][j][η, η])<0
            𝐔[j][:, η] *= -one(type)
        end

        # flip sign of 𝐔[j] for all j≠i:1:m if their corresponding element is negative
        for x=1:m
            if x≠j
                if real(𝑫[i][x][η, η])<0
                    𝐔[x][:, η] *= -one(type)
                end
            end
        end

        # Bring the maximum from position η on top (current e)
        if η≠e flipcol!(𝐔, m, η, e) end

        # compute 𝑫 again
        for i=1:m 𝑫[i]=𝔻Vector([𝛍(𝔻([𝐔[i][:, η]'*𝒞[l, i, j]*𝐔[j][:, η] for η=1:n]) for l=1:k) for j=1:m]) end
    end

    return diag(𝛍(𝑫[i][j] for i=1:m for j=1:m if i≠j))
end # function _scaleAndPermute!
