#  Unit "Jade.jl" of the Diagonalization.jl Package for Julia language
#
#  MIT License
#  Copyright (c) 2019,
#  Marco Congedo, CNRS, Grenoble, France:
#  https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#  This unit implements the orthogonal AJD algorithm of Cardoso and
#  Souloumiac (1996), optimized for both real and complex data input.
#  J.-F. Cardoso, A. Souloumiac (1996) Jacobi angles for simultaneous
#  diagonalization. SIAM Journal on Matrix Analysis and Applications,
#  17(1), 161–164.
#  It is adapted in Julia from code freely made available from the author.

#  The algorithm handles the AJD diagonalization procedure, corresponding
#  to the case m=1, k>1 according to the taxonomy adopted in this package.
#  It handles both real and complex data input.
#  It takes as input a vector of k positive
#  definite matrices 𝐂 and find a non-singular matrix B such that the
#  congruences B'*𝐂_κ*B are as diagonal as possible for all κ=1:k.
#
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
#  if sort=true (default) the column vectors of the B matrix are reordered
#  so as to sort in descending order the diagonal elements of B'*mean(𝐂)*B,
#  where mean(𝐂) is the arithmetic mean of the matrices in 𝐂.
#
#  if  `whitening` = false (default), a matrix can be provided with the `init`
#  argument in order to initialize B. In this case the actual AJD
#  will be given by init*B, where B is the output of the algorithms.
#
#  `tol` is the convergence to be attained.
#
#  `maxiter` is the maximum number of iterations allowed.
#
#  if `verbose`=true, the convergence attained at each iteration and other
#  information will be printed.
#
#  return: B, its pseudo-inverse, the diagonal elements of B'*mean(𝐂)*B,
#          the number of iterations and the convergence attained
#
#  NB: Cardoso and Souloumiac's algorithm proceeds by planar rotations of
#  pairs of vectors of B. A sweep goes over all (n*(n+1))/2 ij pairs, i>j.
#  Thus it can be optimized by multi-threading the optimization of the pairs
#  e.g., using the round-Robin tournament scheme.


function jade( 𝐂::Union{Vector{Hermitian}, Vector{Symmetric}};
               trace1   :: Bool  = false,
               w        :: Twf   = ○,
               preWhite :: Bool  = false,
               sort     :: Bool  = true,
               init     :: Union{Symmetric, Hermitian, Nothing} = ○,
               tol      :: Real  = 0.,
               maxiter  :: Int   = 60,
               verbose  :: Bool  = false,
            eVar     :: TeVaro = ○,
            eVarMeth :: Function = searchsortedfirst)

   # Compute the Givens angle θ (scalar) for real data
   function givensAngles(::Type{T}, p, q, 𝓹, 𝓺) where T <:Real
      e₁ = C[p, 𝓹] - C[q, 𝓺]
      e₂ = C[p, 𝓺] + C[q, 𝓹]
      a = e₁⋅e₁ - e₂⋅e₂
      b = 2. * e₁⋅e₂
      θ = 0.5 * atan(b, a + √(a^2 + b^2))
      s, c = sincos(θ)
      return c, s, abs(s)
   end

   # Compute the Givens angles 𝛉 (vector) for complex data
   function givensAngles(::Type{T}, p, q, 𝓹, 𝓺) where T <:Complex
      e₁ = C[p, 𝓹] - C[q, 𝓺]
      e₂ = C[p, 𝓺]
      e₃ = C[q, 𝓹]
      E = Hermitian(UpperTriangular{T}([e₁⋅e₁ e₂⋅e₁ e₃⋅e₁; 0. e₂⋅e₂ e₃⋅e₂; 0. 0. e₃⋅e₃]))
      𝛉 = eigvecs(real(Γ*E*Γₜ))[:, 3] # Julia sorts the eigvecs; the 3rd<->max ev
      if 𝛉[1]<0. 𝛉 = -𝛉 end
      c = √(0.5 + 𝛉[1]/2.)         # cosine
      s = 0.5*(𝛉[2] - 𝛉[3]im)/c   # sine
      return c, s, abs(s)
   end

   function cardosoSweep!()
      ∡ = 0.
      @inbounds for p = 1:n-1
         𝓹 = p:n:nk
         for q = p+1:n
            𝓺 = q:n:nk
            c, s, 𝓈 = givensAngles(T, p, q, 𝓹, 𝓺)
            # updates U and matrices in C by a Givens rotation
            if 𝓈 > tolerance
               G = [c -conj(s); s c]
               ⊶ = [p, q]  # p,q index pair
               U[:, ⊶] = U[:, ⊶]*G
               C[⊶, :] = G'*C[⊶, :]
               C[:, [𝓹 𝓺]] = [c*C[:, 𝓹]+s*C[:, 𝓺] -conj(s)*C[:, 𝓹]+c*C[:, 𝓺]]
            end
            ∡ = max(∡, 𝓈)    # 𝓈 is abs(sine of the angle)
         end
      end
      return ∡    # convergence: maximum abs(sine of the angle) over all pairs
   end

   T, k = eltype(𝐂[1]), length(𝐂)

   # trace normalization and weighting
   trace1 || w ≠ ○ ? begin
      𝐆=deepcopy(𝐂)
      _Normalize!(𝐆, trace1, w)
   end : 𝐆=𝐂

   # pre-whiten, initialize and stack matrices horizontally
   if preWhite
      W = whitening(mean(Euclidean, 𝐆); eVar=eVar, eVarMeth=eVarMeth)
      C = hcat([(W.F'*G*W.F) for G∈𝐆]...)
   else
      # initialization only if preWhite is false
      init≠nothing ? C = hcat([(init'*G*init) for G∈𝐆]...) : C = hcat(𝐆...)
   end

   (n, nk), 𝟘 = size(C), zeros
   tolerance = tol==0. ? √eps(real(T)) : tol
   iter, conv, 😋 = 1, 0., false

   # pre-allocate memory
   e₁ = 𝟘(T, k)
   e₂ = 𝟘(T, k)
   if T <:Complex
      e₃ = 𝟘(T, k)
      Γ = convert(Matrix{T}, [1 0 0; 0 1 1; 0 -im im])
      Γₜ = Γ'
      𝛉 = 𝟘(T, 3)
   end

   # initialize AJD
   U=Matrix{T}(I, n, n)
   verbose && @info("Iterating JADE algorithm...")
   while true
      conv=cardosoSweep!()
      verbose && println("iteration: ", iter, "; convergence: ", conv)
      (overRun = iter == maxiter) && @warn("JADE: reached the max number of iterations before convergence:", iter)
      (😋 = conv <= tolerance) || overRun==true ? break : nothing
      iter += 1
   end
   verbose && @info("Convergence has "*(😋 ? "" : "not ")*"been attained.\n")
   verbose && println("")

   # sort the vectors of solver
   D=Diagonal([mean(C[i, i:n:nk]) for i=1:n])
   # λ = sort ? _permute!(U, D, n) : diag(D)
   λ = diag(D) # temp !

   return preWhite ? (W.F*U, U'*W.iF, λ, iter, conv) :
                     (U, Matrix(U'), λ, iter, conv)
end
