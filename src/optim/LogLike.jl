

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
							ii, jj = ii+n, jj+n
				      end
						tmp = 𝐜[κ][i+ic+1]
						𝐜[κ][i+ic+1] += (B₁₂*(2*𝐜[κ][ij+1] + B₁₂*𝐜[κ][jj+1]))
						𝐜[κ][jj+1] += B₂₁*𝐜[κ][ij+1]
						𝐜[κ][ij+1] += B₂₁*tmp	# element of index j,i */

				      @inbounds while ii<ic
							tmp = 𝐜[κ][ii+1]
							𝐜[κ][ii+1] += B₁₂*𝐜[κ][jj+1]
							𝐜[κ][jj+1] += B₂₁*tmp
							ii, jj = ii+n, jj+1
				      end

				      jj, ii = jj+1, ii+1
				      @inbounds while jj<(jc+n)
							tmp = 𝐜[κ][ii+1]
							𝐜[κ][ii+1] += B₁₂*𝐜[κ][jj+1]
							𝐜[κ][jj+1] += B₂₁*tmp
							ii, jj = ii+1, jj+1
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

		#= # useless as we only need decr
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

	# pre-whitening and write matrices in vectorized form
	if preWhite
		W=whitening(PosDefManifold.mean(Jeffrey, 𝐂); eVar=eVar, eVarMeth=eVarMeth)
		𝐜=[(W.F'*C*W.F)[:] for C∈𝐂]
		n=size(W.F, 2) # subspace dimension
	else
		𝐜=[C[:] for C∈𝐂]
	end

	# initialization only if preWhite is false
	init≠nothing && !preWhite ? B=copy(Matrix(init')) : B=Matrix{type}(I, n, n)

	verbose && @info("Iterating LogLike algorithm...")
	while true
	   conv=phamSweep!()
		verbose && println("iteration: ", iter, "; convergence: ", conv)
		(overRun = iter == maxiter) && @warn("LogLike: reached the max number of iterations before convergence:", iter)
		(converged = conv <= tolerance) || overRun==true ? break : nothing
		iter += 1
	end
	verbose ? (converged ? @info("Convergence has been attained.\n") : @warn("Convergence has not been attained.")) : nothing
	verbose && println("")

	B=Matrix(B') # get B such B'*C[k]*B is diagonal

	# sort vectors of solver
	M=mean(𝐂)
	D=Diagonal([PosDefManifold.qf(B[:, i], M) for i=1:n])
	λ = sort ? _permute!(B, D, n) : diag(D)

	return preWhite ? (W.F*B, pinv(B)*W.iF, λ, iter, conv) :
							(B, pinv(B), λ, iter, conv)
end
