#  Unit "Jade.jl" of the Diagonalization.jl Package for Julia language
#
#  MIT License
#  Copyright (c) 2019-2025
#  Marco Congedo, CNRS, Grenoble, France:
#  https://sites.google.com/site/marcocongedo/home
#  Konstantin Usevich, CNRS, Nancy, France
#  http://w3.cran.univ-lorraine.fr/konstantin.usevich/?q=content/home


# ? CONTENTS :
#  This unit implements two orthogonal AJD algorithms:
#  1) The famous JADE algorithm of Cardoso and Souloumiac (1996),
#  performing GIVENS ROTATIONS of p,q pairs of vectors according to
#  the CYCLIC SCHEME, i.e., all p,q pairs are rotated in the order
#  for all p=1:n-1, q=i+1:n.
#  The code here below is optimized for both real and complex data input and
#  has been adapted in Julia from code freely made available from the authors.
#  J.-F. Cardoso, A. Souloumiac (1996) Jacobi angles for simultaneous
#  diagonalization. SIAM Journal on Matrix Analysis and Applications,
#  17(1), 161–164.
#  2) The JADEmax algorithm proposed by Usevich, Li and Comon (2020).
#  This uses the same GIVENS ROTATIONS however the p,q pairs are chosen
#  as the pair corresponding to the maximum p,q entry of the Riemannian
#  gradient of the cost function. The convergence is much faster in this case
#  at an extra cost of updating the gradient after each rotation.
#  Simulations and tests on real data indicate that JADEmax is overall faster
#  for n>20.
#
#  NB: JADE may be optimized by multi-threading the optimization of the pairs
#  e.g., using the round-Robin tournament scheme.
#  On the other hand, JADEmax is by definition a sequential procedure.

#  These algorithms handle the AJD diagonalization procedure corresponding
#  to the case m=1, k>1 according to the taxonomy adopted in this package.


## Compute the Givens angle (sine, cosine) for p,q pair
# and corrsponding p,q gradient element for real data
@inline function givensAngle(C::Matrix{T}, p, q, 𝓹, 𝓺,
							  e₁, e₂, e₃, Φ, Φₜ) where T <:Real
  e₁[:] = C[p, 𝓹] - C[q, 𝓺]
  e₂[:] = C[p, 𝓺] + C[q, 𝓹]
  a = e₁⋅e₁ - e₂⋅e₂
  b = 2. * e₁⋅e₂
  θ = 0.5 * atan(b, a + √(a^2 + b^2))
  s, c = sincos(θ)
  return c, s, abs(s), abs(b/4) # b/4 is the gradient
end

# Compute the Givens angle (sine, cosine) for p,q pair
# and corrsponding p,q gradient element for complex data
@inline function givensAngle(C::Matrix{T}, p, q, 𝓹, 𝓺,
								e₁, e₂, e₃, Φ, Φₜ) where T <:Complex
  e₁[:] = C[p, 𝓹] - C[q, 𝓺]
  e₂[:] = C[p, 𝓺]
  e₃[:] = C[q, 𝓹]
  E = Hermitian(UpperTriangular{T}([e₁⋅e₁ e₂⋅e₁ e₃⋅e₁; 0. e₂⋅e₂ e₃⋅e₂; 0. 0. e₃⋅e₃]))
  Γ = real(Φ*E*Φₜ)
  𝛉 = eigvecs(Γ)[:, 3] # Julia sorts the eigvecs; the 3rd<->max ev
  if real(𝛉[1])<0. 𝛉 = -𝛉 end  # added 'real' to prevent error
  c = √(0.5 + 𝛉[1]/2.)         # cosine
  s = 0.5*(𝛉[2] - 𝛉[3]im)/c   # sine
  return c, s, abs(s), abs(Γ[1, 2]+Γ[1, 3]im) # Γ[1, 2]+Γ[1, 3]im is the gradient
end


# Update U and all Cₖ matrices by a Givens rotation (both real and complex data)
@inline function jadeUpdate!(U, C, c, s, p, q, 𝓹, 𝓺)
	G = [c -conj(s); s c]
	⊶ = [p, q]  # p,q index pair
	U[:, ⊶] = U[:, ⊶]*G
	C[⊶, :] = G'*C[⊶, :]
	C[:, [𝓹 𝓺]] = [c*C[:, 𝓹]+s*C[:, 𝓺] -conj(s)*C[:, 𝓹]+c*C[:, 𝓺]]
end


# Do one cyclic sweep: rotate p,q pairs for all p=1:n-1, q=i+1:n
# For each p,q pair update U and all Cₖ if max abs(sine(p,q angle))>tol
# Return max abs(sine(p,q angle)) over all pairs
@inline function cyclicSweepAngle!(C, e₁, e₂, e₃, Φ, Φₜ, U, n, nk, tolerance)
  ∡ = 0.
  @inbounds for p = 1:n-1
	 𝓹 = p:n:nk
	 for q = p+1:n
		𝓺 = q:n:nk
		c, s, 𝓈, g = givensAngle(C, p, q, 𝓹, 𝓺, e₁, e₂, e₃, Φ, Φₜ)
		𝓈 > tolerance ? jadeUpdate!(U, C, c, s, p, q, 𝓹, 𝓺) : nothing
		∡ = max(∡, 𝓈)  # 𝓈 is abs(sine of the p,q Givens angle)
	 end
  end
  return ∡
end

# Do one cyclic sweep: rotate p,q pairs for all p=1:n-1, q=i+1:n
# For each p,q pair update U and all Cₖ if abs(p,q gradient's element)>tol
# Return the maximum of abs(p,q gradient's element) over all pairs
@inline function cyclicSweepGradient!(C, e₁, e₂, e₃, Φ, Φₜ, U, n, nk, tolerance)
  ∇pq = 0.
  @inbounds for p = 1:n-1
	 𝓹 = p:n:nk
	 for q = p+1:n
		𝓺 = q:n:nk
		c, s, 𝓈, g = givensAngle(C, p, q, 𝓹, 𝓺, e₁, e₂, e₃, Φ, Φₜ)
		g > tolerance ? jadeUpdate!(U, C, c, s, p, q, 𝓹, 𝓺) : nothing
		∇pq = max(∇pq, g)  # g is abs(of p,q Riemannian gradient's element)
	 end
  end
  return ∇pq
end


# Compute and return the Riemannian gradient (real data)
@inline function jadeRiemannian∇(C::Matrix{T}, D, n, k, 𝓹₁, 𝓹₂) where T <:Real
	∇=zeros(T, n, n)
	@inbounds @simd for i=1:k
		Cₖ = C[:, 𝓹₁[i]: 𝓹₂[i]]
		D[i][:] = diag(Cₖ)
		∇ += Cₖ.*D[i]'
	end
	return ∇-∇'
end


# Compute and return the Riemannian gradient (Complex data)
@inline function jadeRiemannian∇(C::Matrix{T}, D, n, k, 𝓹₁, 𝓹₂) where T <:Complex
	∇=zeros(T, n, n)
	@inbounds @simd for i=1:k
		Cₖ = C[:, 𝓹₁[i]: 𝓹₂[i]]
		D[i][:] = conj(diag(Cₖ))
		∇ += Cₖ.*(transpose(D[i])) - Cₖ.*D[i] # the @. macro transform all ensuing operations in breadcasting element-wise operations
	end
	return ∇-∇'
end


# Update the p, q rows and columns of the gradient
@inline function jadeRiemannian∇Update!(∇, C::Matrix{T}, D, k, p, q, 𝓹, 𝓺, 𝓹₁, 𝓹₂) where T <:Real
	⊶ = [p, q]  # p,q index pair
	∇[:, ⊶] .= T(0)
	@inbounds @simd for i=1:k
		dp, dq = C[p, 𝓹[i]], C[q, 𝓺[i]]
		D[i][p], D[i][q] = dp, dq
		∇[:, ⊶] += [C[:, 𝓹[i]].*(dp.-D[i]) C[:, 𝓺[i]].*(dq.-D[i])]
	end
	∇[⊶, :]=-∇[:, ⊶]'
end


# Update the p, q rows and columns of the gradient
@inline function jadeRiemannian∇Update!(∇, C::Matrix{T}, D, k, p, q, 𝓹, 𝓺, 𝓹₁, 𝓹₂) where T <:Complex
	⊶ = [p, q]  # p,q index pair
	∇[:, ⊶] .= T(0)
	#@inbounds @simd for i=1:k
	for i=1:k
		dp, dq = conj(C[p, 𝓹[i]]), conj(C[q, 𝓺[i]])
		D[i][p], D[i][q] = dp, dq
		∇[:, ⊶] += [(C[:, 𝓹[i]].*(dp.-D[i]).+conj.(C[p, 𝓹₁[i]:𝓹₂[i]]).*conj.(dp.-D[i])) (C[:, 𝓺[i]].*(dq.-D[i]).+conj.(C[q, 𝓹₁[i]:𝓹₂[i]]).*conj.(dq.-D[i]))]
		#∇[:, ⊶] += [C[:, 𝓹[i]].*(dp-D[i])+conj(C[p, 𝓹₁[i]:𝓹₂[i]])*conj(dp.-D[i]) C[:, 𝓺[i]].*(dq.-D[i])+conj(C[q, 𝓹₁[i]:𝓹₂[i]])*conj(dq.-D[i])]
	end
	∇[⊶, :]=-∇[:, ⊶]'
end


# pre-allocate memory
@inline function jadeAllocateMemory(T, k)
	e₁ = zeros(T, k)
	e₂ = zeros(T, k)
	if T <:Complex
		e₃ = zeros(T, k)
		Φ = convert(Matrix{T}, [1 0 0; 0 1 1; 0 -im im])
		Φₜ = Φ'
	else
		e₃, Φ, Φₜ = nothing, nothing, nothing, nothing
	end
	return e₁, e₂, e₃, Φ, Φₜ
end


## PRIMITIVE JADE (Cyclic) algorithm:
#  It takes as input a n·nk matrix holding k horizontally stacked n·n real or
#  complex matrices, such as C=[C_1...C_k].
#  It finds an orthogoanl/unitary matrix U such that the
#  congruences U'*C_κ*U are as diagonal as possible for all κ=1:k.
#  `tol` is the convergence to be attained.
#  `maxiter` is the maximum number of iterations allowed.
#  if `updateRule`=`:angle` (default), the max abs of the sin of the angle
#  across all pairs is used as stopping criterion, otherwise (any other symbol)
#  the max abs of the Riemannian gradient entry is used.
#  The former is the usual choice of the JADE algorithm, the latter allows to
#  compare directly the execution speed of JADE and JADEmax as JADEmax uses
#  that stopping criterion. Note that the Riemannian gradient criterion for 
#  JADE can be obtained at virtually no extra-cost from the computation of
#  the angle.
#  if `verbose`=true, the convergence attained at each iteration and other
#  information will be printed.
#
#  RETURN: B, the number of iterations and the convergence attained (a 3-tuple)
function jade(C::Matrix{T};
				tol 	 = 0.,
				maxiter = 1000,
				updateRule = :angle,
				verbose = false) where T<:Union{Real, Complex}

	(n, nk) = size(C)
	iter, conv, 😋, k, U = 1, 1., false, nk÷n, Matrix{T}(I, n, n)
	tolerance = tol≤0. ? √eps(real(T)) : tol
	e₁, e₂, e₃, Φ, Φₜ = jadeAllocateMemory(T, k)

	verbose && @info("Iterating JADE algorithm...")
	while true
		conv = updateRule==:angle ? cyclicSweepAngle!(C, e₁, e₂, e₃, Φ, Φₜ, U, n, nk, tolerance) :
			   						cyclicSweepGradient!(C, e₁, e₂, e₃, Φ, Φₜ, U, n, nk, tolerance)
		verbose && println("iteration: ", iter, "; convergence: ", conv)
	   	(overRun = iter == maxiter) && @warn("JADE reached the max number of iterations before convergence:", iter)
	   	(😋 = conv <= tolerance) || overRun==true ? break : iter += 1
	end
	verbose && @info("Convergence has "*(😋 ? "" : "not ")*"been attained.\n\n")

	return U, iter, conv
end



## PRIMITIVE JADEmax algorithm:
#  It takes as input a n·nk matrix holding k horizontally stacked n·n real or
#  complex matrices, such as C=[C_1...C_k].
#  It finds an orthogoanl/unitary matrix U such that the
#  congruences U'*C_κ*U are as diagonal as possible for all κ=1:k.
#  `tol` is the convergence to be attained.
#  `maxiter` is the maximum number of iterations allowed.
#  if `verbose`=true, the convergence attained at each iteration and other
#  information will be printed.
#
#  RETURN: B, the number of iterations and the convergence attained (a 3-tuple)
function jademax(C::Matrix{T};
					tol 	 = 0.,
					maxiter = 1000,
					verbose = false) where T<:Union{Real, Complex}
	(n, nk) = size(C)
	k, N, U, p, q = nk÷n, (n*(n-1))÷2, Matrix{T}(I, n, n), 0, 0
	𝓹₁, 𝓹₂ = 1:n:nk, n:n:nk
	𝓹, 𝓺 = 𝓹₁, 𝓹₂
	tolerance = tol≤0. ? √eps(real(T)) : tol
	iter, itN, itN0, conv, 😋 = 0, 1, 0, 1., false
	e₁, e₂, e₃, Φ, Φₜ = jadeAllocateMemory(T, k)
	D=[Vector{T}(undef, n) for i=1:k]

	verbose && @info("Iterating JADEmax algorithm...")
	# compute the full gradient only once at the beginning
	∇ = jadeRiemannian∇(C, D, n, k, 𝓹₁, 𝓹₂)
    while true
		# find max absolute element of the gradient and its position
		conv, maxpos = T <:Real ? findmax(∇) : findmax(abs2.(∇))
		p, q = maxpos[1], maxpos[2]
		𝓹, 𝓺  = p:n:nk, q:n:nk
		# find angle, update U, all Cₖ matrices and the Riemannian gradient
	  	c, s, ∡ = givensAngle(C, p, q, 𝓹, 𝓺, e₁, e₂, e₃, Φ, Φₜ)
		if conv > tolerance
			jadeUpdate!(U, C, c, s, p, q, 𝓹, 𝓺)
			jadeRiemannian∇Update!(∇, C, D, k, p, q, 𝓹, 𝓺, 𝓹₁, 𝓹₂)
		end

		(itN0=itN%N==0) && (iter+=1)
		verbose && itN0 && println("iteration(m): ", iter, "; convergence: ", conv)
		(overRun = iter == maxiter) && @warn("jademax reached the max number of iterations before convergence:", iter)
       	(😋 = conv <= tolerance) || overRun==true ? break : itN += 1
    end
	verbose && !itN0 && println("iteration(m): ", iter, " + ", itN%N, " sweeps ; convergence: ", conv)
    verbose && @info("Convergence has "*(😋 ? "" : "not ")*"been attained.\n\n")

	return U, iter, conv
end



#  ADVANCED JADE and JADEmax algorithms:
#  They take as input a vector of k real symmetric or complex Hermitian
#  matrices 𝐂 and finds an orthogoanl/Unitary matrix U such that the
#  congruences U'*𝐂_κ*U are as diagonal as possible for all κ=1:k.
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
#  so as to sort in descending order the mean of the diagonal elements
#  of B'*𝐂_κ* over k.
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
#  return: B, its pseudo-inverse, the mean diagonal elements of B'*mean(𝐂)*B,
#          the number of iterations and the convergence attained
function jade( 𝐂::Union{Vector{Hermitian}, Vector{Symmetric}};
               trace1   :: Bool  = false,
               w        :: Twf   = ○,
               preWhite :: Bool  = false,
               sort     :: Bool  = true,
               init     :: Union{Matrix, Nothing} = ○,
               tol      :: Real  = 0.,
               maxiter  :: Int   = 1000,
               verbose  :: Bool  = false,
            eVar     :: TeVaro = ○,
            eVarMeth :: Function = searchsortedfirst)

	# trace normalization and weighting
	𝐆 = _normalizeAndWeight(trace1, w, 𝐂)

	# pre-whiten or initialize and stack matrices horizontally
	W, C = _preWhiteOrInit(𝐂, preWhite, Euclidean, eVar, eVarMeth, init, :stacked)

	(n, nk) = size(C)

	U, iter, conv = jade(C; tol=tol, maxiter=maxiter, verbose=verbose)

	# permute the vectors of U
	D=Diagonal([mean(C[i, i:n:nk]) for i=1:n])
	λ = sort ? _permute!(U, D, n) : diag(D)

	return preWhite ? (W.F*U, U'*W.iF, λ, iter, conv) :
	                  (U, Matrix(U'), λ, iter, conv)
end


function jademax( 𝐂::Union{Vector{Hermitian}, Vector{Symmetric}};
               trace1   :: Bool  = false,
               w        :: Twf   = ○,
               preWhite :: Bool  = false,
               sort     :: Bool  = true,
               init     :: Union{Matrix, Nothing} = ○,
               tol      :: Real  = 0.,
               maxiter  :: Int   = 1000,
               verbose  :: Bool  = false,
            eVar     :: TeVaro = ○,
            eVarMeth :: Function = searchsortedfirst)

	# trace normalization and weighting
	𝐆 = _normalizeAndWeight(trace1, w, 𝐂)

	# pre-whiten or initialize and stack matrices horizontally
	W, C = _preWhiteOrInit(𝐂, preWhite, Euclidean, eVar, eVarMeth, init, :stacked)

	(n, nk) = size(C)

	U, iter, conv = jademax(C; tol=tol, maxiter=maxiter, verbose=verbose)

	# permute the vectors of U
	D=Diagonal([mean(C[i, i:n:nk]) for i=1:n])
	λ = sort ? _permute!(U, D, n) : diag(D)

	return preWhite ? (W.F*U, U'*W.iF, λ, iter, conv) :
	                  (U, Matrix(U'), λ, iter, conv)
end
