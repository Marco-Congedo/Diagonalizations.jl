#  Unit "LogLike.jl" of the Diagonalization.jl package for Julia language
#
#  MIT License
#  Copyright (c) 2019-2023,
#  Marco Congedo, CNRS, Grenoble, France:
#  https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#  This unit implements two Dinh-Tuan Pham's algorithms based on the
#  log-likelyhood (Kullback-Leibler divergence) criterion.
#  D.-T. Pham (2000) Joint approximate diagonalization of positive definite
#  matrices, SIAM Journal on Matrix Analysis and Applications, 22(4), 1136–1152.
#  They are adapted in Julia from code freely made available from the author.
#
#  These algorithms handles the AJD diagonalization procedure, corresponding
#  to the case m=1, k>1 according to the taxonomy adopted in this package.


# function to get the weights from argment `w`
function _logLikeWeights(w, 𝐂, type)
	if 	   w isa Function
		   w=[w(C) for C∈𝐂]
	elseif w===○
		   w=ones(type, length(𝐂))
	end
	return w, sum(w)
end


#  PRIMITIVE LogLike algorithm:
#  It takes as input a n·nk matrix holding k horizontally stacked n·n real or
#  complex matrices, such as C=[C_1...C_k].
#  It finds a non-singular matrix B such that the
#  congruences B'*C_κ*B are as diagonal as possible for all κ=1:k.
#  `tol` is the convergence to be attained.
#  `maxiter` is the maximum number of iterations allowed.
#  if `verbose`=true, the convergence attained at each iteration and other
#  information will be printed.
#
#  NB: Pham's algorithm proceeds by transforming pairs of vectors of B.
#  A sweep goes over all (n*(n+1))/2 ij pairs, i>j. Thus it can be optimized
#  by multi-threading the optimization of the pairs as it is done for
#  algorithms based on Givens rotations (e.g., round-Robin tournament scheme).
#
#  RETURN: B, the number of iterations and the convergence attained (a 3-tuple)
function logLike(C::Matrix{T};
	     		 tol 	 = 0.,
				 maxiter = 1000,
				 verbose = false) where T<:Union{Real, Complex}

	@inline function phamSweep!()
		decr = 0.
		@inbounds for i = 2:n, j = 1:i-1
			c1 = C[i, i:n:nk]
			c2 = C[j, j:n:nk]
			g₁₂ = mean(C[i, j:n:nk]./c1)		# this is g_{ij}
			g₂₁ = mean(C[i, j:n:nk]./c2)		# conjugate of g_{ji}
			𝜔₂₁ = mean(c1./c2)
			𝜔₁₂ = mean(c2./c1)
			𝜔 = √(𝜔₁₂*𝜔₂₁)
			𝜏 = √(𝜔₂₁/𝜔₁₂)
			𝜏₁ = (𝜏*g₁₂ + g₂₁)/(𝜔 + 1.)
			if T<:Real 𝜔=max(𝜔 - 1., e) end
			𝜏₂ = (𝜏*g₁₂ - g₂₁)/𝜔 		#max(𝜔 - 1., e)	# in case 𝜔 = 1
			h₁₂ = 𝜏₁ + 𝜏₂				# this is twice h_{ij}
			h₂₁ = conj((𝜏₁ - 𝜏₂)/𝜏)	# this is twice h_{ji}
			decr += k*(g₁₂*conj(h₁₂) + g₂₁*h₂₁)/2.

			𝜏 = 1. + 0.5im*imag(h₁₂*h₂₁)	# = 1 + (h₁₂*h₂₁ - conj(h₁₂*h₂₁))/4
			𝜏 = 𝜏 + √(𝜏^2 - h₁₂*h₂₁)
			Γ = [1 conj(-h₂₁/𝜏); conj(-h₁₂/𝜏) 1]
			C[[i, j], :] = Γ'*C[[i, j], :]		# new i, j rows of C
			ijInd = vcat(collect(i:n:nk), collect(j:n:nk))
			C[:, ijInd] = reshape(reshape(C[:, ijInd], n*k, 2)*Γ, n, k*2) # new i,j columns of C
			B[:, [i, j]] = B[:, [i, j]]*Γ # update the columns of B
		end
		return decr
	end # phamSweep

	(n, nk) = size(C)
	k, e = nk÷n, T(eps(real(T)))
	B=Matrix{T}(I, n, n) # initialize AJD algorithm

    iter, conv = _iterate!("LogLike", phamSweep!, maxiter, T, tol, verbose)

    return B, iter, conv
end


#  ADVANCED LogLike algorithm:
#  It takes as input a vector of k real symmetric or complex Hermitian
#  matrices 𝐂 and finds a non-singular matrix B such that the
#  congruences B'*𝐂_κ*B are as diagonal as possible for all κ=1:k.
#  It handles both real and complex data input.
#
#  NB: For the moment being the weights are not supported. The `w` argument
#  is left in for syntax homogeneity with other AJD procedures.
#
#  if `whitening` = true is passed, the Jeffrey mean of the matrices in 𝐂 is
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
function logLike(𝐂::Union{Vector{Hermitian}, Vector{Symmetric}};
				 w			:: Union{Tw, Function} = ○,
				 preWhite	:: Bool = false,
				 sort      	:: Bool = true,
				 init		:: Union{Matrix, Nothing} = ○,
				 tol     	:: Real = 0.,
				 maxiter 	:: Int  = 1000,
				 verbose 	:: Bool = false,
			  eVar 	   :: TeVaro = ○,
			  eVarMeth :: Function = searchsortedfirst)

	w===○ || @warn 📌*" package - `loglike` function: argument `w` is not taken into consideration for this AJD algorithm. Uniform weights will be applied."

	# pre-whiten or initialize and stack matrices horizontally
	W, C = _preWhiteOrInit(𝐂, preWhite, Jeffrey, eVar, eVarMeth, init, :stacked)

	(n, nk) = size(C)

	B, iter, conv = logLike(C; tol=tol, maxiter=maxiter, verbose=verbose)

	# scale and permute the vectors of B
    D=Diagonal([mean(C[i, i:n:nk]) for i=1:n])
    λ = sort ? _permute!(_scale!(B, D, n)...) : diag(D)

	return preWhite ? (W.F*B, pinv(B)*W.iF, λ, iter, conv) :
                      (B, pinv(B), λ, iter, conv)
end

######################################################
#  ADVANCED 'c-style' version of the LogLike algorithm
#  It supports only real data.
#  It uses the same API as 'logLike' function.
#  It supports weights:
#  `w` is an optional vector of k non-negative weights for each matrix in 𝐂.
#  Pham's criterion being invariant by scaling, the weights act on the cost
#  function, not as a weights for the entries of the matrices.
#  Notice that in this algorithm the weights can be zero,
#  amounting to ignoring the corresponding matrices.
#  By default all weights are equal to one.
#  A function can be passed as the `w` argument, in which case the kth weight
#  is found as the output of the function applied to the kth matrix in 𝐂.
#  A good choice in general is the `nonD` function declared in tools.jl unit.
#
#  All other arguments are in 'logLike' function
function logLikeR(𝐂::Union{Vector{Hermitian}, Vector{Symmetric}};
				  w 		:: Union{Tw, Function} = ○,
				  preWhite  :: Bool = false,
				  sort      :: Bool = true,
				  init 	 	:: Union{Matrix, Nothing} = ○,
				  tol     	:: Real = 0.,
				  maxiter 	:: Int  = 1000,
				  verbose 	:: Bool = false,
			eVar 	  :: TeVaro = ○,
			eVarMeth  :: Function = searchsortedfirst)

	@inline function phamSweepR!()
	   det, decr, i, ic  = 1., 0., 1, n
	   @inbounds while i<n
			j, jc = 0, 0
			while j<i
				ii, jj, ij = i + ic, j + jc, i + jc
				q1, p2, p, q = 0., 0., 0., 0.
				for κ=1:k
					if w[κ]>e
						wκ, t1, t2, t = w[κ], 𝐜[κ][ii+1], 𝐜[κ][jj+1], 𝐜[κ][ij+1]
						p += wκ*t/t1; q += wκ*t/t2; q1 += wκ*t1/t2; p2 += wκ*t2/t1
					end
				end

				# find rotation
				q1 /= ∑w; p2 /= ∑w; p  /= ∑w; q  /= ∑w; β = 1. - p2*q1
				if (q1 ≤ p2)
					α = p2*q - p
					abs(α)-β<e ? begin β = -1.; γ = p/p2; end : γ = - (p*β + α)/p2 # p1 = 1
					decr += ∑w*(p^2 - α^2/β)/p2
				else
					γ  = p*q1 - q		# p1 = 1
					abs(γ)-β<e ? begin β = -1.; α = q/q1; end : α = -(q*β + γ)/q1	# q2 = 1
					decr += ∑w*(q^2 - γ^2/β)/q1
				end
				t = 2/(β - sqrt(β^2 - 4*α*γ))
				B₁₂, B₂₁ = γ*t, α*t

				for κ=1:k
					if w[κ]>e
					   	ii, jj = i, j
						while ii<ij
							𝜏 = 𝐜[κ][ii+1]
							𝐜[κ][ii+1] = 𝐜[κ][ii+1] + B₁₂*𝐜[κ][jj+1]
							𝐜[κ][jj+1] = 𝐜[κ][jj+1] + B₂₁*𝜏 # at exit ii = ij = i + jc
							ii += n
							jj += n
						end
						𝜏 = 𝐜[κ][i+ic+1]
						𝐜[κ][i+ic+1] += (B₁₂*(2*𝐜[κ][ij+1] + B₁₂*𝐜[κ][jj+1]))
						𝐜[κ][jj+1] += B₂₁*𝐜[κ][ij+1]
						𝐜[κ][ij+1] += B₂₁*𝜏	# element of index j,i */
						while ii<ic
							𝜏 = 𝐜[κ][ii+1]
							𝐜[κ][ii+1] += B₁₂*𝐜[κ][jj+1]
							𝐜[κ][jj+1] += B₂₁*𝜏
							ii += n
							jj += 1
						end

						jj += 1
						ii += 1
						while jj<(jc+n)
							𝜏 = 𝐜[κ][ii+1]
							𝐜[κ][ii+1] += B₁₂*𝐜[κ][jj+1]
							𝐜[κ][jj+1] += B₂₁*𝜏
							jj += 1
							ii += 1
						end
					end
				end

				@inbounds for r=1:n # rotate B
					𝜏 = B[i+1, r]
					B[i+1, r] += B₁₂*B[j+1, r]
					B[j+1, r] += B₂₁*𝜏
				end

				det *= 1. - B₁₂*B₂₁ # compute determinant
				j  += 1
				jc +=  n
			end # while j
			i += 1
			ic += n
	   end # while i

		#= useless computing the logdet crit as we only need its decrement
		ld += (2*∑w*log(det));
		𝜏 = 0.
		for κ=1:k # OK
			if w[κ]>e
				det = 1.
				ii = 1
				while ii≤n^2 # OK
					 det *= 𝐜[κ][ii]
					 ii += (n+1)
				end
				𝜏 += w[κ]*log(det)
			end
		end
		# return 𝜏 - ld
		=#
		return decr
	end # phamSweepR

	n, k, type =size(𝐂[1], 1), length(𝐂), eltype(𝐂[1])
	e = eps(type)*100

	w, ∑w = _logLikeWeights(w, 𝐂, type) # weights and sum of weights

	# pre-whiten, initialize and write matrices in vectorized form
	if preWhite
		W=whitening(PosDefManifold.mean(Jeffrey, 𝐂); eVar=eVar, eVarMeth=eVarMeth)
		𝐜=[(W.F'*C*W.F)[:] for C∈𝐂]
		n=size(W.F, 2) # subspace dimension
	else
		# initialization only if preWhite is false
		init≠nothing ? 𝐜=[(init'*C*init)[:] for C∈𝐂] : 𝐜=[C[:] for C∈𝐂]
	end

	B=Matrix{type}(I, n, n)
    iter, conv = _iterate!("LogLikeR", phamSweepR!, maxiter, type, tol, verbose)
	B = Matrix(B')

	# sort the vectors of solver
	M=mean(𝐂)
	D=Diagonal([PosDefManifold.quadraticForm(B[:, i], M) for i=1:n])
	# scale and permute the vectors of B
    λ = sort ? _permute!(_scale!(B, D, n)...) : diag(D)

	return preWhite ? (W.F*B, pinv(B)*W.iF, λ, iter, conv) :
                      (B, pinv(B), λ, iter, conv)
end
