#  Unit "Gajd.jl" of the Diagonalization.jl Package for Julia language
#
#  MIT License
#  Copyright (c) 2019,
#  Marco Congedo, CNRS, Grenoble, France:
#  https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#  This unit implements the Gauss AJD algorithm of Congedo (unpublished).

#  The algorithm handles the AJD diagonalization procedure, corresponding
#  to the case m=1, k>1 according to the taxonomy adopted in this package.


# update input matrices and AJD matrix for Gauss Algorithms
# bj <- bj +  θbi (update the jth column with respect to the ith one)
# Cpj <- bp' C (bj +  θbi) for p=1:n (jth row of C)
# Cjq <- (bj +  θbi)'' C bq  for q=1:n (jth column of C)
# the update of C is done only on the lower triangular part
# 𝐋 is a lower triangular matrix of k-length vectors : 𝐋[i, j][k]=C[k][i, j]
@inline function _update1!(j, i, n, θ, θ², 𝐋, B) # i>j
   for p = 1:j-1 𝐋[j, p] += θ*𝐋[i, p] end     # update 𝐂 :
   𝐋[j, j] += θ²*𝐋[i, i] + 2θ*𝐋[i, j]         # write jth row and column
   for p = j+1:i 𝐋[p, j] += θ*𝐋[i, p] end     # only on the lower
   for p = i+1:n 𝐋[p, j] += θ*𝐋[p, i] end     # triangular part.
   B[:, j] += θ*B[:, i]                       # update B
end

# update1! takes care of the udpate if i>j, update2! if j≥i
@inline function _update2!(j, i, n, θ, θ², 𝐋, B) # j>i
   for p = 1:i-1 𝐋[j, p] += θ*𝐋[i, p] end     # update 𝐂 :
   for p = i:j-1 𝐋[j, p] += θ*𝐋[p, i] end     # write jth row and column
   𝐋[j, j] += θ²*𝐋[i, i] + 2θ*𝐋[j, i]         # only on the lower
   for p = j+1:n 𝐋[p, j] += θ*𝐋[p, i] end     # triangular part.
   B[:, j] += θ*B[:, i]                       # update B
end


#  PRIMITIVE GAJD algorithm:
#  It takes as input a lower triangular matrix
#  holding in its elements vectors of k real numbers.
#  From data in matrix form given as k symmetric C_κ matrices of size nxn,
#  we have 𝐋[i, j][κ] = C_κ[i, j], for κ=1:k, i>j=1:n.
#  It finds a non-singular matrix B such that the
#  congruences B'*C_κ*B are as diagonal as possible for all κ=1:k.
#  `tol` is the convergence to be attained.
#  `maxiter` is the maximum number of iterations allowed.
#  if `verbose`=true, the convergence attained at each iteration and other
#  information will be printed.
#  RETURN: B, the number of iterations and the convergence attained (a 3-tuple)
function gajd(𝐋::AbstractArray; tol = 0., maxiter = 60, verbose = false)

   # find optimal theta and update convergence (∡)
   function _gauss!(𝑖, 𝑗, i) # 𝑖 must be < 𝑗
      θ  = h*sum(𝐋[𝑗, 𝑖].*𝐋[i, i])
      θ² = θ^2
      ∡ += θ²
   end

   @inline function congedoSweep!()
      ∡ = T(0.)
      for i = 1:n
         h = -inv(sum(𝐋[i, i].^2)) # h is invariant in the inner loop

         # transform all j≠i columns of B with respect to its ith column:
         for j = 1:i-1
            _gauss!(j, i, i) # find θ, θ² and update ∡
            _update1!(j, i, n, θ, θ², 𝐋, B) # update 𝐋 and B given θ and θ²
         end
         for j = i+1:n
            _gauss!(i, j, i) # find θ, θ² and update ∡
            _update2!(j, i, n, θ, θ², 𝐋, B) # update 𝐋 and B given θ and θ²
         end
      end
      return √(∡ * e) # convergence: average squared theta over all n(n-1) pairs
   end

   # declare variables
   T, n = eltype(𝐋[1, 1]), size(𝐋, 1)
   h, θ, θ², ∡, e = T(0), T(0), T(0), T(0), T(inv(n*(n-1)))
   B = Matrix{T}(I, n, n) # initialization of the AJD matrix

   iter, conv = _iterate!("GOFF", congedoSweep!, maxiter, T, tol, verbose)

   return B, iter, conv
end



#  ADVANCED GAJD algorithm:
#  It takes as input a vector of k real symmetric matrices 𝐂 and finds a
#  non-singular matrix B such that the congruences B'*𝐂_κ*B are as diagonal
#  as possible for all κ=1:k.

#  if `trace1` is true (false by default), all input matrices are normalized
#  so as to have unit trace. Note that the diagonal elements
#  of the input matrices must be all positive.
#
#  `w` is an optional vector of k positive weights for each matrix in 𝐂.
#  if `w` is different from `nothing` (default), the input matrices are
#  weighted with these weights (after trace normalization if `trace1` is true).
#  A function can be passed as the `w` argument, in which case the kth weight
#  is found as the output of the function applied to the kth matrix in 𝐂.
#  A good choice in general is the `nonD` function declared in tools.jl unit.
#
#  if `whitening` = true is passed, the Arithmetic mean of the matrices in 𝐂 is
#  computed (using the PosDefManifold.jl package) and the matrices in 𝐂
#  are pre-transformed using the whitening matrix of the mean.
#  Dimensionality reduction can be obtained at this stage using optional
#  arguments `eVar` and `eVarMeth` (see documentation of the AJD constructors).
#
#  if sort=true (default) the column vectors of the B matrix are normalized
#  to unit norm and permuted so as to sort in descending order the mean over
#  κ=1:k of the diagonal elements of B'*𝐂_κ*B.
#  Note that if `whitening` is true the output B will not have unit norm columns
#  as it is multiplied by the whitener after being scaled and sorted
#  and before being returned.
#
#  if  `whitening` = false (default), a matrix can be provided with the `init`
#  argument in order to initialize B. In this case the actual AJD
#  will be given by init*B, where B is the output of the algorithm.
#
#  `tol` is the convergence to be attained.
#
#  `maxiter` is the maximum number of iterations allowed.
#
#  if `verbose`=true, the convergence attained at each iteration and other
#  information will be printed.
#
#  return: B, its pseudo-inverse, the mean diagonal elements of B'*mean(𝐂)*B,
#          the number of iterations and the convergence attained
#
#  NB: Differently from other AJD algorithms, this algorithm proceeds
#  by transformations of one vector at a time given a pair of vectors of B.
#  A sweep goes over all n*(n+1) ij pairs, with i,j ∈ 1:n, j≠i.
#  The update of the input matrices is RECURSIVE, thus it is not suitable
#  for multi-threading. This algorithm has the lowest complexity per iteration
#  among all algorithms here implemented and scale extremely well over k, i.e.,
#  for n small and k large it offers the best performance.
function gajd( 𝐂::Union{Vector{Hermitian}, Vector{Symmetric}};
               trace1   :: Bool  = false,
               w        :: Twf   = ○,
               preWhite :: Bool  = false,
               sort     :: Bool  = true,
               init     :: Union{Matrix, Nothing} = ○,
               tol      :: Real  = 0.,
               maxiter  :: Int   = 120,
               verbose  :: Bool  = false,
            eVar     :: TeVaro = ○,
            eVarMeth :: Function = searchsortedfirst)

   # trace normalization and weighting
   𝐆 = _normalizeAndWeight(trace1, w, 𝐂)

   # pre-whiten or initialize or nothing
   W = _preWhiteOrInit!(𝐆, preWhite, Euclidean, eVar, eVarMeth, init)

   T, n = eltype(𝐆[1]), size(𝐆[1], 1)

   # arrange data in a LowerTriangular matrix of k-vectors
   𝐋 = _arrangeData!(T, n, 𝐆)

   B, iter, conv = gajd(𝐋; tol=tol, maxiter=maxiter, verbose=verbose)

   # scale and permute the vectors of B
   D=Diagonal([mean(𝐋[i, i]) for i=1:n])
   λ = sort ? _permute!(_scale!(B, D, n)...) : diag(D)

   return preWhite ? (W.F*B, pinv(B)*W.iF, λ, iter, conv) :
                     (B, pinv(B), λ, iter, conv)
end




function gLogLike(𝐋::AbstractArray; tol = 0., maxiter = 60, verbose = false)

   # find optimal theta and update convergence ∡
   function _gauss!(𝑖, 𝑗, i, j) # 𝑖 must be < 𝑗
      #=
      fill!(Π, T(1))
      for l=1:𝑖-1 Π.*=𝐋[l, l] end # much faster without the if!
      for l=𝑖+1:𝑗-1 Π.*=𝐋[l, l] end
      for l=𝑗+1:n Π.*=𝐋[l, l] end
      =#

      Π_=Π./𝐋[j, j]  # 𝐋[𝑗, 𝑖] here below picks from lower triangular part
      θ = -sum(@. 𝐋[𝑗, 𝑖]*𝐋[i, i]*Π_) / sum(@. 𝐋[i, i]^2 *Π_)
      ####θ = -sum(@. lᵢⱼ*lᵢᵢ*Π) / sum(lᵢᵢ² .*Π)
      θ² = θ^2
      ∡ += θ²
   end

   @inline function congedoSweep!()
      ∡ = T(0.)
      for i = 1:n
         # product of the diagonal elements excluding the ith one (for each k)
         fill!(Π, T(1))
         for l = 1:i-1 Π .*= 𝐋[l, l] end
         for l = i+1:n Π .*= 𝐋[l, l] end

         # transform all j≠i columns of B with respect to its ith column:
         for j = 1:i-1
            _gauss!(j, i, i, j) # find θ, θ² and update ∡
            _update1!(j, i, n, θ, θ², 𝐋, B) # update 𝐋 and B given θ and θ²
         end
         for j = i+1:n
            _gauss!(i, j, i, j) # find θ, θ² and update ∡
            _update2!(j, i, n, θ, θ², 𝐋, B) # update 𝐋 and B given θ and θ²
         end
      end
      return √(∡ * e) # convergence: average squared theta over all n(n-1) pairs
   end

   # declare variables
   T, n = eltype(𝐋[1, 1]), size(𝐋, 1)
   Π = Vector{T}(undef,  length(𝐋[1, 1])); Π_= similar(Π)
   θ, θ², ∡, e = T(0), T(0), T(0), inv(n*(n-1))
   B = Matrix{T}(I, n, n) # initialization of the AJD matrix

   iter, conv = _iterate!("GLogLike", congedoSweep!, maxiter, T, tol, verbose)
   return B, iter, conv
end


function gLogLike( 𝐂::Union{Vector{Hermitian}, Vector{Symmetric}};
               w        :: Twf   = ○,
               preWhite :: Bool  = false,
               sort     :: Bool  = true,
               init     :: Union{Matrix, Nothing} = ○,
               tol      :: Real  = 0.,
               maxiter  :: Int   = 120,
               verbose  :: Bool  = false,
            eVar     :: TeVaro = ○,
            eVarMeth :: Function = searchsortedfirst)

   # pre-whiten or initialize or nothing
   W, 𝐆 = _preWhiteOrInit(𝐂, preWhite, Jeffrey, eVar, eVarMeth, init, :Hvector)

   T, n = eltype(𝐆[1]), size(𝐆[1], 1)

   # arrange data in a LowerTriangular matrix of k-vectors
   𝐋 = _arrangeData!(T, n, 𝐆)

   B, iter, conv = gLogLike(𝐋; tol=tol, maxiter=maxiter, verbose=verbose)

   # scale and permute the vectors of B
   D=Diagonal([mean(𝐋[i, i]) for i=1:n])
   λ = sort ? _permute!(_scale!(B, D, n)...) : diag(D)

   return preWhite ? (W.F*B, pinv(B)*W.iF, λ, iter, conv) :
                     (B, pinv(B), λ, iter, conv)
end


# approximation computing in the outer loop the products of the diagonal
# elements discarding the ith elements. This does not discared the jth
# element in the inner loop
function gLogLike_(𝐋::AbstractArray; tol = 0., maxiter = 60, verbose = false)

   # find optimal theta and update convergence ∡
   function _gauss!(𝑖, 𝑗, i, j) # 𝑖 must be < 𝑗
      θ  = sum(@.𝐋[𝑗, 𝑖]*lᵢᵢ) * ω
      θ² = θ^2
      ∡ += θ²
   end

   @inline function congedoSweep!()
      ∡ = T(0.)
      for i ∈ 1:n

         # approximation
         fill!(Π, T(1))
         for l=1:i-1 Π.*=𝐋[l, l] end
         for l=i+1:n Π.*=𝐋[l, l] end
         lᵢᵢ=𝐋[i, i].*Π
         ω=-inv(sum(𝐋[i, i].^2 .*Π))

         for j = 1:i-1
            _gauss!(j, i, i, j) # find θ, θ² and update ∡
            _update1!(j, i, n, θ, θ², 𝐋, B) # update 𝐋 and B given θ and θ²
         end
         for j = i+1:n
            _gauss!(i, j, i, j) # find θ, θ² and update ∡
            _update2!(j, i, n, θ, θ², 𝐋, B) # update 𝐋 and B given θ and θ²
         end
      end
      return √(∡ * e) # convergence: average squared theta over all n(n-1) pairs
   end

   # declare variables
   T, n = eltype(𝐋[1, 1]), size(𝐋, 1)
   Π = Vector{T}(undef,  length(𝐋[1, 1]));
   Π_, lᵢᵢ = similar(Π), similar(Π)
   θ, θ², ∡, ω, e = T(0), T(0), T(0), T(0), inv(n*(n-1))
   B = Matrix{T}(I, n, n) # initialization of the AJD matrix

   iter, conv = _iterate!("GLogLike_", congedoSweep!, maxiter, T, tol, verbose)
   return B, iter, conv
end


function gLogLike_( 𝐂::Union{Vector{Hermitian}, Vector{Symmetric}};
               w        :: Twf   = ○,
               preWhite :: Bool  = false,
               sort     :: Bool  = true,
               init     :: Union{Matrix, Nothing} = ○,
               tol      :: Real  = 0.,
               maxiter  :: Int   = 120,
               verbose  :: Bool  = false,
            eVar     :: TeVaro = ○,
            eVarMeth :: Function = searchsortedfirst)

   # pre-whiten or initialize or nothing
   W, 𝐆 = _preWhiteOrInit(𝐂, preWhite, Jeffrey, eVar, eVarMeth, init, :Hvector)

   T, n = eltype(𝐆[1]), size(𝐆[1], 1)

   # arrange data in a LowerTriangular matrix of k-vectors
   𝐋 = _arrangeData!(T, n, 𝐆)

   # run AJD algorithm
   B, iter, conv = gLogLike_(𝐋; tol=tol, maxiter=maxiter, verbose=verbose)

   # scale and permute the vectors of B
   D=Diagonal([mean(𝐋[i, i]) for i=1:n])
   λ = sort ? _permute!(_scale!(B, D, n)...) : diag(D)

   return preWhite ? (W.F*B, pinv(B)*W.iF, λ, iter, conv) :
                     (B, pinv(B), λ, iter, conv)
end
