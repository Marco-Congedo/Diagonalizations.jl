#  Unit "QnLogLike.jl" of the Diagonalization.jl Package for Julia language
#
#  MIT License
#  Copyright (c) 2020,
#  Marco Congedo°, Ronald Phlypo, CNRS, UGA, Grenoble-INP, France
#  Alexandre Gramfort¨, INRIA, U. Paris Saclay, France
#  ° https://sites.google.com/site/marcocongedo/
#  ¨ http://alexandre.gramfort.net/

# ? CONTENTS :
#  Quasi-Newton Approximate Joint Diagonalization (AJD) algorithm of:
#  P. Ablin, J.F. Cardoso and A. Gramfort. Beyond Pham's algorithm
#  for joint diagonalization. Proc. ESANN 2019.
#  https://hal.archives-ouvertes.fr/hal-01936887v1
#
#  Code adapted in Julia from Python code provided from the authors at:
#  https://github.com/pierreablin/qndiag/blob/master/qndiag/qndiag.py
#
#  NOTATION
#  lower-case letter: a scalar, e.g.: a
#  Upper case letter: a matrix, e.g., A
#  Bold lover case letter: a vector of matrices, e.g., 𝐀
#
#  The algorithm takes as input a vector of k real symmetric matrices
#  𝐂 and find a non-singular matrix B such that the congruences
#  B'*𝐂_κ*B are as diagonal as possible for all κ=1:k.
#
#  `w` is an optional vector of k positive weights for each matrix in 𝐂.
#  if `w` is different from `nothing` (default), the input matrices are
#  weighted with these weights.
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
#  if sort=true (default) the column vectors of the B matrix are normalized
#  to unit norm and permuted so as to sort in descending order the mean over
#  κ=1:k of the diagonal elements of B'*𝐂_κ*B.
#  Note that if `whitening` is true the output B will not have unit norm
#  columns, as it is multiplied by the whitener after being scaled and sorted
#  and before being returned.
#
#  if  `whitening` = false (default), a matrix can be provided with the `init`
#  argument in order to initialize B. In this case the actual AJD
#  will be given by init*B, where B is the output of the algorithm.
#
#  `tol` is the convergence to be attained. It default to the square root of
#  the machine epsilon for the data input type.
#
#  `maxiter` is the maximum number of iterations allowed.
#
#  𝜆min is used to reguarized Hessian coefficients; all coefficients smaller
#  than 𝜆min  will be set to 𝜆min
#
#  lsmax is the maximum number of step in the line search.
#
#  if `verbose`=true, the convergence attained at each iteration and other
#  information will be printed.
#
#  return: B, its pseudo-inverse, the mean diagonal elements of B'*mean(𝐂)*B,
#           the number of iterations and the convergence attained.
#
#  Note on the implementation:
#  GRADIENT:
#  D_1,...,D_n are the matrices in 𝐃;
#  the jth column in D_i is divided by D_i[j, j],
#  the mean of these matrices is then taken and finally
#  the identity is subtracted
#
#  HESSIAN COEFFICIENTS
#  for each COLUMN vector dg_1,...,dg_n
#  (the diagonal part of the n matrices in 𝐃)
#  we form a m·m matrix stacking vertically m copies of these vectors
#  transposed and dividing each of them by the mth element, as
#   _                   _
#  |  (dg_i)'/dg_i[1]   |
#  |  (dg_i)'/dg_i[...] |
#  |  (dg_i)'/dg_i[m]   |
#  _                   _
#  finally we take the mean of all the n matrices created in this way.

# function to get the weights from argment `w`
function _qnlogLikeWeights!(w, 𝐂)
	if w isa Function w=[w(C) for C∈𝐂] end
	return w./mean(w)
end

function qnLogLike( 𝐂::Union{Vector{Hermitian}, Vector{Symmetric}};
                    w           :: Twf   = ○,
                    preWhite    :: Bool = false,
                    sort        :: Bool = true,
                    init        :: Union{Matrix, Nothing} = ○,
                    tol         :: Real = 0.,
                    maxiter     :: Int  = 200,
                    𝜆min        :: Real = 1e-4,
                    lsmax       :: Int  = 10,
                    verbose     :: Bool = false,
                 eVar     :: TeVaro = ○,
                 eVarMeth :: Function = searchsortedfirst)

    # internal functions
    @inline function _linesearch(; StartAt::Real = 1.)
        for i ∈ 1:lsmax
            M = (StartAt * →) + I
			𝐃₊ = HermitianVector([Hermitian(M'*D*M) for D ∈ 𝐃])
			#@threads for j=1:k 𝐃₊[j] = Hermitian(M'*𝐃[j]*M) end
			B₊ = B * M
            iter > 2 && (loss₊ = _getLoss())
			#loss₊ = _getLoss()
			#print("x",)
            loss₊ < loss ? break : StartAt /= 2.0
        end
        return 𝐃₊, B₊, loss₊
    end

	_getLoss() =
		if w===○
			-(logabsdet(B₊)[1]) + 0.5*sum(mean(log, [𝔻(D) for D ∈ 𝐃₊]))
		else
			-(logabsdet(B₊)[1]) + 0.5*sum(mean(log, [𝔻(D*v) for (D, v) ∈ zip(𝐃₊, 𝐯)]))
		end

	# pre-whiten or initialize or just copy
    W, 𝐃 = _preWhiteOrInit(𝐂, preWhite, Jeffrey, eVar, eVarMeth, init, :Hvector)

    # set variables
    n, k, T = size(𝐃[1], 1), length(𝐃), eltype(𝐃[1])
    tol==0. ? tolerance = √eps(real(T)) : tolerance = tol
    iter, conv, loss, loss₊, 😋, sqrtn = 1, Inf, Inf, T(1), false, √n
    B = Matrix{T}(I, n, n)
    B₊, →, M, 𝐃₊ = similar(B), similar(B), similar(B), similar(𝐃)
	if w≠○ 𝐯 = _qnlogLikeWeights!(w, 𝐂) end # if w is `nonD` function, apply it to the original input 𝐂

    # here we go
    verbose && println("Iterating quasi-Newton LogLike algorithm...")

    while true
        diagonals = [diag(D) for D ∈ 𝐃]

		# Gradient
		if w===○
			∇ = mean(d./diagd for (d, diagd) ∈ zip(𝐃, diagonals)) - I
		else
			∇ = mean(v.*(d./diagd) for (v, d, diagd) ∈ zip(𝐯, 𝐃, diagonals)) - I
		end
        conv = norm(∇)/sqrtn # relative norm of ∇ with respect to the identity : ||∇-I||/||I||

        verbose && println("iteration: ", iter, "; convergence: ", conv)
        (overRun = iter > maxiter) && @warn("qnLogLike: reached the max number of iterations before convergence:", iter-1)
        (😋 = conv <= tolerance) || overRun==true ? break : iter += 1

		# Hessian Coefficients
		if w===○
			ℌ = mean(diagd'./diagd for diagd ∈ diagonals)
		else
			ℌ = mean(v.*(diagd'./diagd) for (v, diagd) ∈ zip(𝐯, diagonals))
		end

		# Quasi-Newton Direction →
        → = -(∇' .* ℌ - ∇)./replace(x -> x<𝜆min ? 𝜆min : x, @. (ℌ'*ℌ) - 1.)

        𝐃, B, loss = _linesearch(StartAt=T(1.)) # Line Search
    end

    verbose && @info("Convergence has "*(😋 ? "" : "not ")*"been attained.\n\n")

    # scale and permute the vectors of B
    λ = sort ? _permute!(_scale!(B, mean(𝔻(D) for D ∈ 𝐃), n)...) :
                diag(mean(𝔻(D) for D ∈ 𝐃))

    return preWhite ? (W.F*B, pinv(B)*W.iF, λ, iter, conv) :
                      (B, pinv(B), λ, iter, conv)
end
