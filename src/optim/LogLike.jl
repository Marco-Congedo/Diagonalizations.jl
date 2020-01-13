#   Unit "LogLike.jl" of the Diagonalization.jl package for Julia language
#
#   MIT License
#   Copyright (c) 2019, 2020
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements two Dinh-Tuan Pham's algorithms based on the
#   log-likelyhood (Kullback-Leibler divergence) criterion.
#   D.-T. Pham (2000) Joint approximate diagonalization of positive definite
#   matrices, SIAM Journal on Matrix Analysis and Applications, 22(4), 1136–1152.
#   They are adapted in Julia from code freely made provided from the author.
#
#  These algorithms handles the AJD diagonalization procedure, corresponding
#  to the case m=1, k>1 according to the taxonomy adopted in this package.
#  The first (logLike) handles both real and complex data input,
#  the second (logLikeR) only real data.
#  They take as input a vector of k positive
#  definite matrices 𝐂 and find a non-singular matrix B such that the
#  congruences B'*𝐂_κ*B are as diagonal as possible for all κ=1:k.
#  They have exactly the same API:
#
#  `w` is an optional vector of k non-negative weights for each matrix in 𝐂.
#  Pham's criterion being invariant by scaling, the weights act on the cost
#  function, not as a weights for the entries of the matrices.
#  Notice that the weights can be zero, amounting to ignoring the
#  corresponding matrices. By default all weights are equal to one.
#  A function can be passed as the `w` argument, in which case the kth weight
#  is found as the output of the function applied to the kth matrix in 𝐂.
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
#  NB: Pham's algorithm proceeds by transforming pairs of vectors of B.
#  A sweep goes over all (n*(n+1))/2 ij pairs, i>j. Thus it can be optimized
#  by multi-threading the optimization of the pairs as it is done
#  for algorithms based on Givens rotations (e.g., roun-(Robin tournament scheme)).
# """


# function to get the weights from argment `w`
function _logLikeWeights(w, 𝐂, type)
	if 	   w isa Function
		   w=[w(C) for C∈𝐂]
	elseif w===○
		   w=ones(type, length(𝐂))
	end
	return w, sum(w)
end


function logLike(𝐂::Union{Vector{Hermitian}, Vector{Symmetric}};
				 w			:: Union{Tw, Function} = ○,
				 preWhite	:: Bool = false,
				 sort      	:: Bool = true,
				 init		:: Union{Symmetric, Hermitian, Nothing} = ○,
				 tol     	:: Real = 0.,
				 maxiter 	:: Int  = 60,
				 verbose 	:: Bool = false,
			  eVar 	   :: TeVaro = ○,
			  eVarMeth :: Function = searchsortedfirst)

	function phamSweep!()
	  decr = 0.
	  for i = 2:n
		for j = 1:i-1
		  c1 = 𝐜[i, i:n:nk]
		  c2 = 𝐜[j, j:n:nk]
		  g₁₂ = mean(𝐜[i, j:n:nk]./c1)		# this is g_{ij}
		  g₂₁ = mean(𝐜[i, j:n:nk]./c2)		# conjugate of g_{ji}
		  𝜔₂₁ = mean(c1./c2)
		  𝜔₁₂ = mean(c2./c1)
		  𝜔 = √(𝜔₁₂*𝜔₂₁)
		  𝜏 = √(𝜔₂₁/𝜔₁₂)
		  𝜏₁ = (𝜏*g₁₂ + g₂₁)/(𝜔 + 1.)
		  if type<:Real 𝜔=max(𝜔 - 1., e) end
		  𝜏₂ = (𝜏*g₁₂ - g₂₁)/𝜔 #max(𝜔 - 1., e)	# in case 𝜔 = 1
		  h₁₂ = 𝜏₁ + 𝜏₂					# this is twice h_{ij}
		  h₂₁ = conj((𝜏₁ - 𝜏₂)/𝜏)		# this is twice h_{ji}
		  decr += k*(g₁₂*conj(h₁₂) + g₂₁*h₂₁)/2.

		  𝜏 = 1. + 0.5im*imag(h₁₂*h₂₁)	# = 1 + (h₁₂*h₂₁ - conj(h₁₂*h₂₁))/4
		  𝜏 = 𝜏 + √(𝜏^2 - h₁₂*h₂₁) #
		  T = [1 -h₁₂/𝜏; -h₂₁/𝜏 1]
		  𝐜[[i, j], :] = T*𝐜[[i, j], :]		# new i, j rows of 𝐜
		  ijInd = vcat(collect(i:n:nk), collect(j:n:nk))
		  𝐜[:, ijInd] = reshape(reshape(𝐜[:, ijInd], n*k, 2)*T', n, k*2)		# new i,j columns of 𝐜
		  B[[i, j], :] = T*B[[i, j], :]
		end
	  end
	  return decr
	end # phamSweep

	type, k=eltype(𝐂[1]), length(𝐂)

	w, ∑w = _logLikeWeights(w, 𝐂, type) # weights and sum of weights

	# pre-whiten, initialize and stack matrices horizontally
	if preWhite
		W=whitening(PosDefManifold.mean(Jeffrey, 𝐂); eVar=eVar, eVarMeth=eVarMeth)
		𝐜=hcat([(W.F'*C*W.F) for C∈𝐂]...)
	else
		# initialization only if preWhite is false
		init≠nothing ? 𝐜=hcat([(init'*C*init) for C∈𝐂]...) : 𝐜=hcat(𝐂...)
	end

	(n, nk) = size(𝐜)
	tol==0. ? tolerance = √eps(real(type)) : tolerance = tol
	iter, conv, converged, e = 1, 0., false, type(eps(real(type)))

	B=Matrix{type}(I, n, n)

	verbose && @info("Iterating LogLike2 algorithm...")
	while true
	   conv=real(phamSweep!())
		verbose && println("iteration: ", iter, "; convergence: ", conv)
		(overRun = iter == maxiter) && @warn("LogLike: reached the max number of iterations before convergence:", iter)
		(converged = conv <= tolerance) || overRun==true ? break : nothing
		iter += 1
	end
	verbose && @info("Convergence has "*converged ? "" : "not "*"been attained.\n")
	verbose && println("")

	# get B such B'*C[k]*B is diagonal
	B = preWhite ? W.F*Matrix(B') : Matrix(B')

	# sort the vectors of solver
	M=mean(𝐂)
	D=Diagonal([PosDefManifold.quadraticForm(B[:, i], M) for i=1:n])
	λ = sort ? _permute!(B, D, n) : diag(D)

	return (B, pinv(B), λ, iter, conv)
end



function logLikeR(𝐂::Union{Vector{Hermitian}, Vector{Symmetric}};
				  w 		:: Union{Tw, Function} = ○,
				  preWhite  :: Bool = false,
				  sort      :: Bool = true,
				  init 	 	:: Union{Symmetric, Hermitian, Nothing} = ○,
				  tol     	:: Real = 0.,
				  maxiter 	:: Int  = 60,
				  verbose 	:: Bool = false,
			eVar 	  :: TeVaro = ○,
			eVarMeth  :: Function = searchsortedfirst)

	function phamSweepR!()
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
	tol==0. ? tolerance = √eps(real(type)) : tolerance = tol
	iter, conv, converged, e = 1, 0., false, eps(type)*100

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

	verbose && @info("Iterating logLikeR algorithm...")
	while true
	   conv=phamSweepR!()
		verbose && println("iteration: ", iter, "; convergence: ", conv)
		(overRun = iter == maxiter) && @warn("logLikeR: reached the max number of iterations before convergence:", iter)
		(converged = conv <= tolerance) || overRun==true ? break : nothing
		iter += 1
	end
	verbose && @info("Convergence has "*converged ? "" : "not "*"been attained.\n")
	verbose && println("")

	#=
	B=Matrix(B') # get B such B'*C[k]*B is diagonal

	# sort the vectors of solver
	M=mean(𝐂)
	D=Diagonal([PosDefManifold.quadraticForm(B[:, i], M) for i=1:n])
	λ = sort ? _permute!(B, D, n) : diag(D)

	return preWhite ? 	(W.F*B, pinv(B)*W.iF, λ, iter, conv) :
						(B, pinv(B), λ, iter, conv)
	=#

	# get B such B'*C[k]*B is diagonal
	B = preWhite ? W.F*Matrix(B') : Matrix(B')

	# sort the vectors of solver
	M=mean(𝐂)
	D=Diagonal([PosDefManifold.quadraticForm(B[:, i], M) for i=1:n])
	λ = sort ? _permute!(B, D, n) : diag(D)

	return (B, pinv(B), λ, iter, conv)
end
