#  Unit "common.jl" of the Diagonalization.jl Package for Julia language
#
#  MIT License
#  Copyright (c) 2019-2021,
#  Marco Congedo, CNRS, Grenoble, France:
#  https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#  This unit implements common internal functions for AJD and other
#  iterative algorithms.


# get the maximum number of iterations for each algorithm depending on the
# data input type if the algorithm supports both real and complex data input
_maxiter(algorithm, type) =
   if       algorithm ∈ (:OJoB, :NoJoB, :LogLike, :JADE, :JADEmax)
            return type<:Real ? 1000 : 3000
   elseif   algorithm ∈ (:GAJD, :QNLogLike, :LogLikeR, :GLogLike)
            type<:Real ? (return 1000) :
            throw(ArgumentError("The GAJD, QNLogLike, LogLikeR and GLogLike algorithms do not support complex data input"))
   else     throw(ArgumentError("The `algorithm` keyword argument is uncorrect. Valid options are: :OJoB, :JADE, :JADEmax, :NoJoB, :GAJD, :LogLike, :LogLikeR, :GLogLike, :QNLogLike."))
   end


# arrange data in a LowerTriangular matrix of k-length vectors:
# if 𝐂=[C1, C2, .. Ck], then 𝐋[i, j][k]=𝐂[k][i, j], with i≥j
function _arrangeData!(T, n, 𝐂)
   k = length(𝐂)
   𝐋 = LowerTriangular(Matrix(undef, n, n))
   for i=1:n, j=i:n
      𝐋[j, i] = Vector{T}(undef, k)
      for κ=1:k 𝐋[j, i][κ] = 𝐂[κ][j, i] end
   end
   return 𝐋
end


# trace normalize and/or apply weights. Accept a function for computing weights
# only for m=1
function _normalize!(𝐂::Vector{Hermitian},
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
   ○
end


# trace normalize and/or apply weights. Accept a function for computing weights
# m>=1, k>=1. 𝒞 is a 3-D Array of matrices (k, i, j), i, j=1:m
function _normalize!(𝒞::AbstractArray, m::Int, k::Int,
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
                  @inbounds for i=1:m t[i]*=sqrt(w(𝒞[κ, i, i])) end
         elseif w isa StatsBase.AbstractWeights
                  @inbounds for i=1:m t[i]*=sqrt(w[i]) end
         end
         if trace1 || w ≠ ○
           @inbounds for i=1:m, j=i:m 𝒞[κ, i, j] = 𝒞[κ, i, j]*(t[i]*t[j]) end
         end
      end
   end
   ○
end

function _normalizeAndWeight(trace1, w, 𝐂)
   if trace1 || w ≠ ○
      𝐆 = deepcopy(𝐂)
      _normalize!(𝐆, trace1, w)
      return 𝐆
   else
      return deepcopy(𝐂)
   end
end

# if `preWhite` is true the mean is computed according to the specified
# PosDefManifold.Metric, the whitening matrix is found with arguments
# `eVar` and `eVarMeth` using the Diagonalizations.jl package and all
# matrices in `𝐂` are whitened.
# if `init` is a matrix, all matrices in `𝐂` are transfromed as
# C_k <- init' C_k init
# if `preWhite` is true return the whitening object of the diagonalizations.jl
# package, otherwise, return nothing
function _preWhiteOrInit!(𝐂, preWhite, metric, eVar, eVarMeth, init)
   if preWhite
      W = whitening(mean(metric, 𝐂); eVar=eVar, eVarMeth=eVarMeth)
      @threads for κ=1:length(𝐂) 𝐂[κ]=Hermitian(W.F'*𝐂[κ]*W.F) end
      return W
   end
   if init≠○
      @threads for κ=1:length(𝐂) 𝐂[κ]=Hermitian(init'*𝐂[κ]*init) end
   end
   ○
end

# if `preWhite` is true the mean is computed according to the specified
# PosDefManifold.Metric, the whitening matrix is found with arguments
# `eVar` and `eVarMeth` using the Diagonalizations.jl package and
#  1) if `out` == :Hvector a vector of Hermitian matrices 𝐆 is
#     constrcucted with all matrices in `𝐂` whitened
#  2) if `out` == :stacked all whitened matrices are stacked horizontally in 𝐆
# if `init` is a matrix,
#  1) a vector of Hermitian matrices 𝐆 is constrcucted with all matrices
#     in `𝐂` transformed as C_k <- init' C_k init
#  2) all transformed matrices are stacked horizontally in 𝐆
# if `preWhite` is true return the whitening object of the diagonalizations.jl
# package and 𝐆.
# if `init` is a matrix return nothing and 𝐆.
# if neither of the above is true,
#  1) return the input matrices 𝐂 untouched
#  2) return the input matrices stacked horizontally
function _preWhiteOrInit(𝐂, preWhite, metric, eVar, eVarMeth, init, out)
   if preWhite
      W = whitening(mean(metric, 𝐂); eVar=eVar, eVarMeth=eVarMeth)
      if out == :Hvector 𝐆 = congruence(Matrix(W.F'), 𝐂, ℍVector) end
      if out == :stacked 𝐆 = hcat([(W.F'*C*W.F) for C∈𝐂]...) end
      return W, 𝐆
   end
   if init≠○
      if out == :Hvector 𝐆 = congruence(Matrix(init'), 𝐂, ℍVector) end
      if out == :stacked 𝐆 = hcat([(init'*C*init) for C∈𝐂]...) end
      return ○, 𝐆
   end
   if out == :Hvector return nothing, deepcopy(𝐂) end
   if out == :stacked return nothing, hcat(𝐂...) end
end

# General REPEAT-UNTIL iterative loop for algorithms.
# `name` is the name of the algorithm given as a string (for printing messages)
# `sweep!` is a function running one iteration of the algo and returning
# the current covergence
# `maxiter` is the maximum number of iterations allowed.
#     If this number of iterations is reached a warning is printed
# `T` is the type of the data given to the algorithm
# If `tol` is equal or inferior to zero, the tolerance for stopping the algo
#     is set to √eps(real(T)), otherwise the value passed as tol is used
# If `verbose` is true the iteration and convergence is printed at each
#     iteration. Warnings and other info may be printed as well.
function _iterate!(name, sweep!, maxiter, T, tol, verbose)
   iter, conv, 😋 = 1, 1., false
   tolerance = tol≤0. ? √eps(real(T)) : tol
   verbose && @info("Iterating "*name*" algorithm...")
   while true
      conv = real(sweep!()) # `real` make sure complex data does not cause problems here
      verbose && println("iteration: ", iter, "; convergence: ", conv)
      (overRun = iter == maxiter) && @warn(name*" reached the max number of iterations before convergence:", iter)
      (😋 = conv <= tolerance) || overRun==true ? break : iter += 1
   end
   verbose && @info("Convergence has "*(😋 ? "" : "not ")*"been attained.\n\n")
   return iter, conv
end


# take as input the vector `λ` of diagonal elements of transformed diagonalized
# matrices. Check that the imaginary part of λ is close to zero.
# If so, return a vector with the real part of λ,
# otherwise print a warning and return λ.
function _checkλ(λ::Vec)
   rePart=sum(real(λ).^2)
   imPart=sum(imag(λ).^2)
   #@show rePart
   #@show imPart
   if imPart/rePart > 1e-6
      @warn "📌, internal function _checkλ: Be careful, the elements of fields `D`, `ev` and `arev` of the constructed LinearFilter will be complex"
      return λ
   else
      return real(λ)
   end
end


# scale column vectors of B to unit norm and correct the quadratic forms
# provided by D=mean(Diagonal(B'C_kB)) to reflect the unit norm of cols of B.
# This is used for AJD algorithms that do not constraint the norm of the
# columns of the solution to unity before calling _permute!, since
# otherwise the elements of D are arbitrary.
function _scale!(B::AbstractArray, D::Diagonal, n::Int)
   inorms=[inv(norm(B[:, i])) for i=1:n]
   for i=1:n B[:, i]*=inorms[i] end     # unit norm
   for i=1:n D[i, i]*=inorms[i]^2 end   # quadr. forms with unit norm
   return B, D, n
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
function _flipAndPermute!( 𝐔::AbstractArray, 𝐗::AbstractArray,
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
end # function _flipAndPermute!
