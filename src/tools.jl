#   Unit "tools.jl" of the Diagonalization.jl Package for Julia language
#
#   MIT License
#   Copyright (c) 2019,
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

where for each *row* and *column* of `P`, β is the maximum of the absolute values
divided by the sum of the absolute values.

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
function genDataMatrix(t::Int, n::Int, A=nothing)
```

Generate a ``t⋅n`` random data matrix as ``XA``,
where ``X`` is a ``t⋅n`` matrix with entries randomly drawn
from a Gaussian distribution and ``A`` a ``n⋅n`` symmetric
matrix, which, if not provided as argument `A`,
will be generated with entries randomly drawn from a uniform
distribution ∈[-1, 1].
"""
function genDataMatrix(t::Int, n::Int, A=○)
   if A===○ A=Symmetric((rand(n, n) .-0.5).*2) end
   return randn(t, n)*A
end


# -------------------------------------------------------- #
# INTERNAL FUNCTIONS #
# -------------------------------------------------------- #


# EigenDecomposition with covariance matrix as input
function _getEVD(C :: Union{Hermitian, Symmetric, Mat}, eVar::TeVaro,
                 eVarMeth::Function, simple::Bool)

   λ, U = eig(C)
   simple ? (U, Matrix(U'), Diagonal(λ), ○, ○, ○) :
   begin
     eVar===○ ? eVar=0.999 : ○
     eVar, D, U, p, arev=_ssd!(eVar, λ, U, _minDim(C), eVarMeth)
     (U, Matrix(U'), D, eVar, λ, arev)
   end
end


# EigenDecomposition with data as input
_getEVD(X::Mat, covEst::StatsBase.CovarianceEstimator, dims::Int64,
        mean::Tmean, w::Tw, eVar::TeVaro, eVarMeth::Function, simple::Bool) =
  _getEVD(_cov(X, covEst, dims, mean, w), eVar, eVarMeth, simple)


# Whitening with covariance matrix as input
function _getWhi(C :: Union{Hermitian, Symmetric, Mat}, eVar::TeVaro,
                 eVarMeth::Function, simple::Bool)

  U, Uⁱ, D, eVar, λ, arev=_getEVD(C, eVar, eVarMeth, simple)
  simple ? (U*D^-0.5, D^0.5*Uⁱ, D, ○, ○, ○) :
           (U*D^-0.5, D^0.5*Uⁱ, D, eVar, λ, arev)
end

# Whitening with data as input
_getWhi(X::Mat, covEst::StatsBase.CovarianceEstimator, dims::Int64,
        mean::Tmean, w::Tw, eVar::TeVaro, eVarMeth::Function, simple::Bool) =
   _getWhi(_cov(X, covEst, dims, mean, w), eVar, eVarMeth, simple)


# convert mean vector for compatibility with StatsBase.jl
function _convert_mean(mean::Tmean, dims::Int, argName::String)
  length(mean)≠n && throw(ArgumentError(📌*", _convert_mean internal function: vector "*argName*" must have length $n"))
  return dims==1 ? Matrix(mean') : mean
end


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

# check arguments and rewrite `mean` if necessary
function _check_data(X::Mat, dims::Int64, meanX::Tmean, wX::Tw)
   dims ∈ (1, 2) || throw(ArgumentError(📌*", _check-data internal function: Argument `dims` may be 1 or 2. dims=$dims"))
   wX≠○ && lenght(wX)≠size(X, dims) && throw(ArgumentError(📌*", _check-data internal function: The size of `wX` does not fit input matrix `X` with `dims`=$dims"))
   ishermitian(X) && throw(ArgumentError(📌*", _check-data internal function: it looks like
   you want to call a filter constuctor that takes covariance matrices as input,
   but you are actually calling the constructor that takes data matrices as input.
   Solution: flag your covariance matrix(ces) argument(s) as Symmetric or Hermitian,
   for example, `Hermitian(C)`. To do so, you will need to be using LinearAlgebra."))
   return true
end


function _check_data(X::Mat, Y::Mat, dims::Int64, meanX::Tmean, meanY::Tmean, wXY::Tw)
   dims ∈ (1, 2) || throw(ArgumentError(📌*", _check-data internal function: Argument `dims` may be 1 or 2. dims=$dims"))
   size(X, dims)==size(Y, dims) || throw(ArgumentError(📌*", _check-data internal function: The `dims` dimension of argument `X` and `Y` must be the same"))
   wXY≠○ && lenght(wXY)≠size(X, dims) && throw(ArgumentError(📌*", _check-data internal function: The size of `wXY` does not fit input matrix `X` with `dims`=$dims"))
   # Since cross-covariance is not implemented in StatsBase.jl, we subtract the means here
   return true
end


# call StatsBase.cov within one line with or without weights
# Also, flag the covariance as Symmetric if is real, Hermitian if is complex
# The mean is subtracted separatedly for consistence with the other _cov method
function _cov(X::Matrix{R},
              covEst   :: StatsBase.CovarianceEstimator = SCM,
              dims     :: Int64 = 1,
              meanX    :: Tmean = 0,
              wX       :: Tw = ○) where R<:Union{Real, Complex}
   T = R===Real ? Symmetric : Hermitian
   X_=_deMean(X, dims, meanX)
   return wX===○ ? T(cov(covEst, X_; dims=dims, mean=0)) : # do NOT remove `mean`=0
                   T(cov(covEst, X_, wX; dims=dims, mean=0)) # "
end


function _cov(𝐗::VecMat;
              covEst   :: StatsBase.CovarianceEstimator = SCM,
              dims     :: Int64 = 1,
              meanX    :: Into = 0)
   # once PosDefManifold supports vectors of Symmetric matrices
   # T = R===Real ? Symmetric : Hermitian
   # remove `Hermitian` here below and use T instead
   # _cov will automatically flag its output
   𝐂=Vector{Hermitian}(undef, length(𝐗))
   #@threads
   for i=1:length(𝐗)
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


function _Normalize!(𝒞::AbstractArray, m::Int, k::Int,
                     trace1::Bool=false, w::Union{Tw, Function}=○)
   !trace1 && w===○ && return

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
                  @inbounds for i=1:m t[i]*=w(𝒞[κ, i, i]) end
         elseif w isa StatsBase.AbstractWeights
                  @inbounds for i=1:m t[i]*=w[i] end
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
   eVar===○ ? eVar=0.999 : ○
   arev = accumulate(+, λ./sum(λ))
   return (eVar isa Int64 ? clamp(eVar, 1, r) : clamp(eVarMeth(arev, eVar), 1, r), arev)
end

function _ssd!(eVar::TeVaro, λ::Vec, U::Mat, r::Int64, eVarMeth::Function)
   p, arev = _getssd!(eVar, λ, r, eVarMeth)
   return p==r ? 1. : arev[p], Diagonal(λ[1:p]), U[:, 1:p], p, arev
end


function _ssdxy!(eVar::TeVaro, λ::Vec, U1::Mat, U2::Mat, r::Int64, eVarMeth::Function)
   p, arev = _getssd!(eVar, λ, r, eVarMeth)
   return p==r ? 1. : arev[p], Diagonal(λ[1:p]), U1[:, 1:p], U2[:, 1:p], p, arev
end


function _ssdcsp!(eVar::TeVaro, λ::Vec, U::Mat, r::Int64, eVarMeth::Function, selMeth::Symbol)
   ratio = λ./(1.0.-λ)
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


function _ssdcstp!(eVar::TeVaro, λ::Vec, U::Mat, V::Mat, r::Int64, eVarMeth::Function)
   arev = accumulate(+, λ./sum(λ))
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
