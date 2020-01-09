#   Unit "JoB.jl" of the Diagonalization.jl Package for Julia language
#
#   MIT License
#   Copyright (c) 2019,
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements a very general form of the OJoB algoithm
#   (Congedo et al., 2012)

# """
#  Orthogonal Joint Blond Source Separation (OJoB) iterative algorithm and
#  Nonorthogonal Joint Blond Source Separation (NoJoB) iterative algorithm:
#  'Orthogonal and Non-Orthogonal Joint Blind Source Separation in the
#  Least-Squares Sense'
#  by Marco Congedo, Ronald Phlypo, Jonas Chatel-Goldman, EUSIPCO 2012
#  https://hal.archives-ouvertes.fr/hal-00737835/document
#
# if `algo` is :OJoB, runs the OJoB algorithm
# if `algo` is :NoJoB, runs the NoJoB algorithm

#  These algorithms handle several use cases corresponding to several
#  combinations of the number of datasets (m) and number of observations (k):
#  1) m=1, k>2
#  2) m>1, k=1
#  3) m>1, k>1
#
#  They take as input either the data matrices or the covariance and cross-
#  covariance matrices. Argument `input` tells the algos to take as input
#  data matrices (`input=:d`) or cov/cross-cov matrices (`input=:d`)
#
#                                   INPUT
# ---------------------------------------------------------------------------
#             Data Matrices                  Cov/cross-cov matrices
# ---------------------------------------------------------------------------
#  1) a vector of k data matrices     an array (k, 1, 1) of covariance matrices
#  2) a vector of m data matrices     an array (1, m, m) of covariance matrices
#  3) k vectors of m data matrices    an array (k, m, m) of covariance matrices
#
#  If data matrices are given as input, the covariance matrices 𝒞 are computed
#  with optional keyword arguments `covEst`, `dims`, `meanX`, `trace1` and `w`.
#
#  OJoB seeks m orthogonal matrices 𝐔=U_1,...,U_m such that the sum of the
#  squares of the diagonal part of matrices U_i'𝒞(κij)U_j
#  is maximized for all i≠j=1:m and κ=1:k.
#  The covariance is maximized rotating each data matrix as: U_i' X_κi,
#  for all i=1:m and κ=1:k.
#
#  NoJoB seeks m non-singular matrices 𝐔=U_1,...,U_m with vectors
#  u_qi, for all i=1:n scaled such that the sum u_qi'(𝒞(κj)u_qju_qj^H𝒞(κj)^H)u_qi=1
#  where the sum is over κ=1:k and j=1:m, separetedly for all q=1:m
#  The covariance is maximized transforming each data matrix as: U_i' X_κi,
#  for all i=1:m and κ=1:k.
#
#  if `fullModel=true` is passed as optional keyword argument,
#  then also the sum of the squares of the diagonal part of matrices
#  U_i'𝒞(κij)U_j is maximized for all ``i=1:m`` and ``κ=1:k``.
#  This amounts to require the intra-covariance to be diagonalized as well.
#
#  if `whitening` = true is passed as optional keyword argument,
#  for OJoB:
#       the 𝐔=U_1,...,U_m are no longer constrained to be orthogonal;
#       instead they will be invertible and a scaling constraint is imposed on them
#       so as to satisfy mean_κ=1:k U_i'𝒞(κii)U_j=I for all i=1:m.
#  for NoJoB:
#       The same constraint is imposed
#
#  For both algorithms, given m whitening matrices W_i,
#  the covariance matrices will be transformed
#  as 𝒢(κij) = W_i'𝒞(κij)W_j, for all i,j=1:m and κ=1:k.
#  Then, the OJoB or NoJob algos will be run on the set 𝒢(κij)
#  and the final solutions will be given by U_iW_i, for all i=1:m
#  Using OjOB, this amounts to requiring a generalized Canonical Correlation
#  Analysis instead of a generalized Maximum Covariance Analysis.
#  By this pre-whitening the dimension of the problem can be reduced
#  requesting matrices W_i to be rectangular. This is achieved using
#  argument `eVar`. If eVar is an integer p, the matrices in 𝒢(κij) will all
#  be pxp. If it is equal to nothing the explained variance is set to
#  the default level (0.999), otherwise if eVar is a float in [0..1], this
#  will be taken as explained variance.
#  if m=1 the eigenvalues of mean_κ=1^k 𝒞(κ11) is considered, otherwise
#         the eigenvalues of mean_κ=1^k, i=1^m 𝒞(κii) is considered.
#  These eigenvalues are normalized to unit sum and the cumulated sum is
#  considered. Finally p is found on this accumulated regularized ev vector
#  using the `eVarMethod` function.
#
#  if sort=true (default) the column vectors of the 𝐔 matrices are reordered
#  so as to allow positive and sorted in descending order
#  diagonal elements of the products ``U_i'𝒞(κij)U_j``.

#  By passing a matrix if k=1 or a vector of k matrices if k>1 as `init`,
#  you can smartly initialize `𝐔`, for example, when excluding some subjects
#  and running the OJoB again.
#
#  `tol` is the convergence to be attained.
#
#  `maxiter` is the maximum number of iterations allowed.
#
#  if `verbose`=true, the convergence attained at each iteration and other
#  information will be printed.
#
#  These algorithms are not multi_threaded, instead they heavely use BLAS.
#  Before running this function you may want to set:
#  `BLAS.set_num_threads(Sys.CPU_THREADS)`
# """
function JoB(𝐗::AbstractArray, m::Int, k::Int, input::Symbol, algo::Symbol, type;
              covEst   :: StatsBase.CovarianceEstimator=SCM,
              dims     :: Int64 = 1,
              meanX    :: Tmean = 0,
              trace1   :: Bool = false,
              w        :: Union{Tw, Function} = ○,
          fullModel :: Bool = false,
          preWhite  :: Bool = false,
          sort      :: Bool = true,
          init      = nothing,
          tol       :: Real=0.,
          maxiter   :: Int=1000,
          verbose   :: Bool=false,
      eVar     :: TeVaro = ○,
      eVarMeth :: Function = searchsortedfirst)

    k<3 && m<2 && throw(ArgumentError("Either `k` must be equal to at least 3 or `m` must be equal to at least 2"))
    #if m==1 fullModel=false end

    # if input ≠:d the input is supposed to be the cross-covariance matrices
    input==:d ? 𝒞=_crossCov(𝐗, m, k;
                            covEst=covEst, dims=dims, meanX=meanX) : 𝒞=𝐗

    if trace1 || w ≠ ○ _Normalize!(𝒞, m, k, trace1, w) end

    if eVar isa Int && size(𝒞[1, 1, 1], 1) ≤ eVar ≤ 0
        eVar=size(𝒞[1, 1, 1], 1)
    end

    iter, conv, oldconv, converged, conv_ = 1, 0., 1.01, false, 0.
    tol==0. ? tolerance = √eps(real(type)) : tolerance = tol

    # pre-whitening
    if preWhite
        # for the moment being computed on the average across k and m
        # if eVar is not an Int.
        # This way for all i=1:m the dimensionality reduction is fixed
        if !(eVar isa Int)
            C=ℍ(𝛍(𝒞[κ, i, i] for κ=1:k, i=1:m))
            #W=whitening(C; eVar=eVar, eVarMeth=eVarMeth)
            p, arev = _getssd!(eVar, eigvals(C), size(𝒞[1, 1, 1], 1), eVarMeth)
            𝑾=[whitening(ℍ(𝛍(𝒞[κ, i, i] for κ=1:k)); eVar=p) for i=1:m]
        else
            𝑾=[whitening(ℍ(𝛍(𝒞[κ, i, i] for κ=1:k)); eVar=eVar) for i=1:m]
        end

        𝒢=Array{Matrix}(undef, k, m, m)
        #println("k, m: ", k, " ", m)
        if m==1
            @inbounds for κ=1:k
                𝒢[κ, 1, 1] = 𝑾[1].F' * 𝒞[κ, 1, 1] * 𝑾[1].F
            end
        else
            @inbounds for κ=1:k, i=1:m-1, j=i+1:m
                𝒢[κ, i, j] = 𝑾[i].F' * 𝒞[κ, i, j] * 𝑾[j].F
                𝒢[κ, j, i] = 𝒢[κ, i, j]'
            end
            @inbounds for κ=1:k, i=1:m
                𝒢[κ, i, i] = 𝑾[i].F' * 𝒞[κ, i, i] * 𝑾[i].F
            end
        end
    else
        𝒢=𝒞
    end
    n=size(𝒢[1, 1, 1], 1)

    # initialization of 𝐔[i], i=1:m, as the eigenvectors of sum_k,j(𝒢_k,i,j*𝒢_k,i,j')
    # see Eq. 17 of Congedo, Phlypo and Pham, 2011, or with the provided matrices
    # note: gemm supports complex matrices
    ggt(κ::Int, i::Int, j::Int) = BLAS.gemm('N', 'T', 𝒢[κ, i, j], 𝒢[κ, i, j])
    if m>1 #gmca, gcca, majd
        if init === nothing
            if algo==:NoJoB
                𝐔 = [Matrix{type}(I, n, n) for i=1:m]
            else
                if fullModel
                    𝐔 = [eigvecs(𝛍(ggt(κ, i, j) for j=1:m, κ=1:k)) for i=1:m]
                else
                    𝐔 = [eigvecs(𝛍(ggt(κ, i, j) for j=1:m, κ=1:k if j≠i)) for i=1:m]
                end
            end
        else
            𝐔 = init
        end
    else  #ajd
        if init === nothing
            if algo==:NoJoB
                𝐔 = [Matrix{type}(I, n, n)]
            else
                𝐔 = [eigvecs(𝛍(ggt(κ, 1, 1) for κ=1:k))]
            end
        else
            𝐔 = [init]
        end
    end

    function updateR!(η, i, j)  # 𝐑[η] += (𝒢[κ, i, j] * 𝐔[j][:, η]) times its transpose
        #println("k, i, j ", k, " ", i, " ", j)
        # both gemv and gemm supports complex input
        @inbounds for κ=1:k
            Ω[:, κ] = BLAS.gemv('N', 𝒢[κ, i, j], 𝐔[j][:, η])
        end
        𝐑[η] += BLAS.gemm('N', 'T', Ω, Ω) # (Ω * Ω')
    end

    𝐑 = [Matrix{type}(undef, n, n) for η=1:n]
    Ω = Matrix{type}(undef, n, k)

    if algo==:OJoB
        verbose && @info("Iterating OJoB algorithm...")
        while true
            conv_ =0.
            @inbounds for i=1:m # m optimizations for updating 𝐔[1]...𝐔[m]
                for η=1:n
                    fill!(𝐑[η], zero(type))
                    if m==1
                        updateR!(η, 1, 1)
                    else
                        for j=1:m i≠j ? updateR!(η, i, j) : nothing end # j ≠ i
                        fullModel ? updateR!(η, i, i) : nothing         # j = i
                    end
                    # 1 power iteration
                    𝐔[i][:, η] = BLAS.gemv('N', 𝐑[η], 𝐔[i][:, η])
                end
                conv_ += PosDefManifold.ss(𝐔[i])/n # square of the norms of power iteration vectors

                # Lodwin Orthogonalization and update 𝐔[i]<-UV', with svd(𝐔[i])=UwV'
                𝐔[i] = PosDefManifold.nearestOrth(𝐔[i])
            end

            conv_ =sqrt(conv_ /m)
            iter==1 ? conv=1. : conv = abs((conv_-oldconv)/oldconv)  # relative change

            verbose && println("iteration: ", iter, "; convergence: ", conv)
            (diverging = conv < 0) && verbose && @warn("OJoB diverged at:", iter)
            (overRun = iter == maxiter) && @warn("OJoB: reached the max number of iterations before convergence:", iter)
            (converged = 0. <= conv <= tolerance) || overRun==true ? break : nothing
            oldconv=conv_
            iter += 1
        end # while
    else
        verbose && @info("Iterating NoJoB algorithm...")
        while true
            conv_ =0.
            @inbounds for e2=1:2, i=1:m # m optimizations for updating 𝐔[1]...𝐔[m]
                for η=1:n      # double loop to avoid oscillating convergence
                    fill!(𝐑[η], type(0))
                    if m==1
                        updateR!(η, 1, 1)
                    else
                        for j=1:m i≠j ? updateR!(η, i, j) : nothing end # j ≠ i
                        fullModel ? updateR!(η, i, i) : nothing         # j = i
                    end
                end

                #1 power iteration
                cho=cholesky(Hermitian(sum(𝐑))) # Cholesky LL'of 𝐑[1]+...+𝐑[n]
                for η=1:n
                    # solve Lx=𝐑[η]*𝐔[i][:, η] for x and L'y=x for y
                    y=cho.U\(cho.L\(𝐑[η]*𝐔[i][:, η]))
                    # 𝐔[i][:, η] <- y/sqrt(y'𝐑[η]t)
                    𝐔[i][:, η]=y*inv(sqrt(quadraticForm(y, 𝐑[η])))
                end
                conv_+=PosDefManifold.ss(𝐔[i])
            end

            conv_=conv_/(2*n^2*m) # 2 because of the double loop
            iter==1 ? conv=1. : conv = abs((conv_-oldconv)/oldconv)  # relative change

            verbose && println("iteration: ", iter, "; convergence: ", conv)
            (diverging = conv < 0) && verbose && @warn("NoJoB diverged at:", iter)
            (overRun = iter == maxiter) && @warn("NoJoB: reached the max number of iterations before convergence:", iter)
            (converged = 0. <= conv <= tolerance) || overRun==true ? break : nothing
            oldconv=conv_
            iter += 1
        end # while
    end

    verbose ? (converged ? @info("Convergence has been attained.\n") : @warn("Convergence has not been attained.")) : nothing
    verbose && println("")

    # auto-sort the eigenvectors
    if sort
        λ = m==1 ? _permute!(𝐔[1], 𝒢, k, :c) :
                   _scaleAndPermute!(𝐔, 𝒢, m, k, :c)
    else
        λ = m==1 ? 𝛍([𝐔[1][:, η]'*𝒢[l, 1, 1]*𝐔[1][:, η] for η=1:n] for l=1:k) :
                   𝛍([𝐔[i][:, η]'*𝒢[l, i, j]*𝐔[j][:, η] for η=1:n] for l=1:k, j=1:m, i=1:m if i≠j)
    end

    if preWhite
        algo==:OJoB ? 𝐕=[𝐔[i]'*𝑾[i].iF for i=1:m] :
                      𝐕=[pinv(𝐔[i])*𝑾[i].iF for i=1:m]
        for i=1:m 𝐔[i] = 𝑾[i].F * 𝐔[i] end
    else
        algo==:OJoB ? 𝐕=[Matrix(𝐔[i]') for i=1:m] :
                      𝐕=[pinv(𝐔[i]) for i=1:m]
    end

    return m>1 ? (𝐔, 𝐕, λ, iter, conv) : (𝐔[1], 𝐕[1], λ, iter, conv)
end
