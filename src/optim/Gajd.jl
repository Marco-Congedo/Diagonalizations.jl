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


#  PRIMITIVE GAJD algorithm:
#  It takes as input a lower triangular matrix
#  holding in its elements vectors of k real numbers.
#  From data in matrix form given as k symmetric C_κ matrices of size nxn,
#  we have 𝐋[i, j][κ] = C_κ[i, j], for κ=1:k, i>j=1:n.
#  It find a non-singular matrix B such that the
#  congruences B'*C_κ*B are as diagonal as possible for all κ=1:k.
#  `tol` is the convergence to be attained.
#  `maxiter` is the maximum number of iterations allowed.
#  if `verbose`=true, the convergence attained at each iteration and other
#  information will be printed.
#  RETURN: B, the number of iterations and the convergence attained (a 3-tuple)
function gajd(𝐋::AbstractArray; tol = 0., maxiter = 60, verbose = false)

   function congedoSweep!()
      ∡ = T(0.)
      @inbounds for i ∈ 1:n
         lᵢᵢ = 𝐋[i, i]
         h = -inv(sum(lᵢᵢ.^2))

         # transform all other columns of B with respect to its ith column
         for j ∈ filter(x->x≠i, 1:n)
            ⊶ = j>i ? (j, i) : (i, j) # pick from lower triangular part only

            θ  = h*sum(𝐋[(⊶ )...].*lᵢᵢ) # find optimal theta
            θ² = θ^2
            ∡ += θ²         # update convergence (∡)

            # update 𝐂 (lower triangular part only)
            # this is RECURSIVE, hence no multi-threading is possible
            𝐋[j, j] += θ²*lᵢᵢ + (2*θ)*𝐋[(⊶ )...]
            for p = 1:j-1 𝐋[j, p] += i≥p ? θ*𝐋[i, p] : θ*𝐋[p, i] end
            for p = j+1:n 𝐋[p, j] += i≥p ? θ*𝐋[i, p] : θ*𝐋[p, i] end

            B[:, j] += θ*B[:, i]    # update B
         end # for j
      end
      return ∡*e # convergence: average squared theta over all n(n-1) pairs
   end

   T, n = eltype(𝐋[1, 1]), size(𝐋, 1)
   tolerance = tol==0. ? √eps(real(T)) : tol
   iter, conv, 😋, e = 1, 0., false, inv(n*(n-1))

   # initialize AJD
   B=Matrix{T}(I, n, n)

   verbose && @info("Iterating GAJD algorithm...")
   while true
      conv=congedoSweep!()
      verbose && println("iteration: ", iter, "; convergence: ", conv)
      (overRun = iter == maxiter) && @warn("GAJD: reached the max number of iterations before convergence:", iter)
      (😋 = conv <= tolerance) || overRun==true ? break : iter += 1
   end
   verbose && @info("Convergence has "*(😋 ? "" : "not ")*"been attained.\n\n")

   return B, iter, conv
end

#  ADVANCED GAJD algorithm:
#  It takes as input a vector of k real symmetric matrices 𝐂 and find a
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
               init     :: Union{Symmetric, Hermitian, Nothing} = ○,
               tol      :: Real  = 0.,
               maxiter  :: Int   = 120,
               verbose  :: Bool  = false,
            eVar     :: TeVaro = ○,
            eVarMeth :: Function = searchsortedfirst)

   # trace normalization and weighting
   trace1 || w ≠ ○ ? begin
      𝐆 = deepcopy(𝐂)
      _Normalize!(𝐆, trace1, w)
   end : 𝐆 = 𝐂

   # pre-whiten or initialize
   if preWhite
      W = whitening(mean(Euclidean, 𝐆); eVar=eVar, eVarMeth=eVarMeth)
      for κ=1:length(𝐆) 𝐆[κ]=Hermitian(W.F'*𝐆[κ]*W.F) end
   else
      if init≠nothing for κ=1:length(𝐆) 𝐆[κ]=Hermitian(W.F'*𝐆[κ]*W.F) end end
   end

   T, n, k = eltype(𝐆[1]), size(𝐆[1], 1), length(𝐆)

   # arrange data in a LowerTriangular matrix of k-vectors
   𝐋 = LowerTriangular(Matrix(undef, n, n))
   for i=1:n, j=i:n
      𝐋[j, i] = Vector{T}(undef, k)
      for κ=1:k 𝐋[j, i][κ] = 𝐆[κ][j, i] end
   end

   B, iter, conv = gajd(𝐋; tol=tol, maxiter=maxiter, verbose=verbose)

   # scale and permute the vectors of B
   D=Diagonal([mean(𝐋[i, i]) for i=1:n])
   λ = sort ? _permute!(_scale!(B, D, n)...) : diag(D)

   return preWhite ? (W.F*B, pinv(B)*W.iF, λ, iter, conv) :
                     (B, pinv(B), λ, iter, conv)
end
