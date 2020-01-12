#   Unit "LogLike.jl" of the Diagonalization.jl package for Julia language
#
#   MIT License
#   Copyright (c) 2019, 2020
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements Dinh-Tuan Pham's algorithm based on the log-likelyhood
#   (Kullback-Leibler divergence) criterion.
#   D.-T. Pham (2000) Joint approximate diagonalization of positive definite
#   matrices, SIAM Journal on Matrix Analysis and Applications, 22(4), 1136–1152.
#
#  This algorithm handles the AJD diagonalization procedure, corresponding
#  to the case m=1, k>1 according to the taxonomy adopted in this package.
#
#  The algorithm takes as input a vector of k-real positive definite matrices
#  𝐂 and find a non-singular matrix B such that the congruences B'*𝐂_κ*B
#  are as diagonal as possible for all κ=1:k
#
#  `w` is an optional vector of k non-negative weights for each matrix in 𝐂.
#  Pham's criterion being invariant by scaling, the weights act on the cost
#  function, not as a weights for the entris of the matrices.
#  Notice that the weights can be zero, amounting to ignoring the
#  corresponding matrices. By default all weights are equal to one.
#  A function can be passed as the `w` argument, in which case the kth weight
#  is found as the output of this function applied to the kth matrix in 𝐂.
#  A good choice in general is the `nonD` function declared in tools.jl unit.
#
#  if `whitening` = true is passed, the Jeffrey mean of the matrices in 𝐂 is
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
#  argument in order to initialize B.
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
#  NB: Pham's algorithm proceeds by transforming pairs of vectors of B.
#  A sweep goes over all (n*(n+1))/2 ij pairs, i>j. Thus it can be optimized
#  by multi-threading the optimization of the pairs as it is done
#  for algorithms based on Givens rotations.
# """


function logLike(𝐂::Union{Vector{Hermitian}, Vector{Symmetric}};
			      w 		:: Union{Tw, Function} = ○,
				  preWhite  :: Bool = false,
				  sort      :: Bool = true,
				  init 	 :: Union{Symmetric, Hermitian, Nothing} = ○,
				  tol     :: Real = 0.,
				  maxiter :: Int  = 30,
				  verbose :: Bool = false,
			  eVar 	  :: TeVaro = ○,
			  eVarMeth :: Function = searchsortedfirst)

	function phamSweep!()
	   det, decr, i, ic  = 1., 0., 1, n
	   @inbounds while i<n
			j, jc = 0, 0
			@inbounds while j<i
				ii, jj, ij = i + ic, j + jc, i + jc
				q1, p2, p, q = 0., 0., 0., 0.
				@inbounds for κ=1:k
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

				@inbounds for κ=1:k
					if w[κ]>e
				   	ii, jj = i, j
				      @inbounds while ii<ij
							tmp = 𝐜[κ][ii+1]
							𝐜[κ][ii+1] = 𝐜[κ][ii+1] + B₁₂*𝐜[κ][jj+1]
							𝐜[κ][jj+1] = 𝐜[κ][jj+1] + B₂₁*tmp # at exit ii = ij = i + jc
							ii += n
							jj += n
				      end
						tmp = 𝐜[κ][i+ic+1]
						𝐜[κ][i+ic+1] += (B₁₂*(2*𝐜[κ][ij+1] + B₁₂*𝐜[κ][jj+1]))
						𝐜[κ][jj+1] += B₂₁*𝐜[κ][ij+1]
						𝐜[κ][ij+1] += B₂₁*tmp	# element of index j,i */

				      @inbounds while ii<ic
							tmp = 𝐜[κ][ii+1]
							𝐜[κ][ii+1] += B₁₂*𝐜[κ][jj+1]
							𝐜[κ][jj+1] += B₂₁*tmp
							ii += n
							jj += 1
					  end

				      jj += 1
					  ii += 1
				      @inbounds while jj<(jc+n)
							tmp = 𝐜[κ][ii+1]
							𝐜[κ][ii+1] += B₁₂*𝐜[κ][jj+1]
							𝐜[κ][jj+1] += B₂₁*tmp
							jj += 1
							ii += 1
				      end
					end
				end

				@inbounds for r=1:n # rotate B
					tmp = B[i+1, r]
					B[i+1, r] += B₁₂*B[j+1, r]
					B[j+1, r] += B₂₁*tmp
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
		tmp = 0.
		for κ=1:k # OK
			if w[κ]>e
				det = 1.
				ii = 1
				while ii≤n^2 # OK
					 det *= 𝐜[κ][ii]
					 ii += (n+1)
				end
				tmp += w[κ]*log(det)
			end
		end
		# return tmp - ld
		=#
		return decr
	end # phamSweep

	n, k, type =size(𝐂[1], 1), length(𝐂), eltype(𝐂[1])
	tol==0. ? tolerance = √eps(real(type)) : tolerance = tol
	iter, conv, converged, e = 1, 0., false, eps(type)*100

	# weights
	if 		w isa Function
			w=[w(𝐂[κ]) for κ=1:k]
	elseif 	w===○
			w=ones(type, k)
	end
	∑w=sum(w)

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

	verbose && @info("Iterating LogLike algorithm...")
	while true
	   conv=phamSweep!()
		verbose && println("iteration: ", iter, "; convergence: ", conv)
		(overRun = iter == maxiter) && @warn("LogLike: reached the max number of iterations before convergence:", iter)
		(converged = conv <= tolerance) || overRun==true ? break : nothing
		iter += 1
	end
	verbose && @info("Convergence has "*converged ? "" : "not "*"been attained.\n")
	verbose && println("")

	B=Matrix(B') # get B such B'*C[k]*B is diagonal

	# sort vectors of solver
	M=mean(𝐂)
	D=Diagonal([PosDefManifold.quadraticForm(B[:, i], M) for i=1:n])
	λ = sort ? _permute!(B, D, n) : diag(D)

	return preWhite ? (W.F*B, pinv(B)*W.iF, λ, iter, conv) :
					  (B, pinv(B), λ, iter, conv)
end
