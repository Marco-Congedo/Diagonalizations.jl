#   Unit "JoB.jl" of the Diagonalization.jl Package for Julia language
#
#   MIT License
#   Copyright (c) 2019-2025,
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements a very general form of the OJoB and NoJoB algoithms
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
#  data matrices (`input=:d`) or cov/cross-cov matrices (`input=:c`)
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
#  diagonal elements of the products ``U_i'𝒞(κij)U_j`` as much as possible.
#  If the NoJoB algorithm is used the column vectors of the matrices U_i
#  are normalized to unit norm.
#  Note that if `whitening` is true the output matrices U_i will not have
#  unit norm columns as they are multiplied by the whiteners after being
#  scaled and sorted and before being returned.

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
#  if `threaded`=true (default) and n>x and x>1, where x is the number
#  of threads Julia is instructed to use (the output of Threads.nthreads())
#  the algorithms run in multithreaded mode paralellising several comptations
#  over n.
#
#  Besides optionally multi-threaded, these algorithms heavely use BLAS.
#  Before running this function you may want to set the number of threades
#  Julia is instructed to use to the number of logical CPUs of your machine
#  and set `BLAS.set_num_threads(Sys.CPU_THREADS)`. See:
#  https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Threads-1
# """
function JoB(𝐗::AbstractArray, m::Int, k::Int, input::Symbol, algo::Symbol, type;
              covEst   :: StatsBase.CovarianceEstimator=SCM,
              dims     :: Int64 = 1,
              meanX    :: Tmean = 0,
              trace1   :: Bool = false,
              w        :: Twf = ○,
          fullModel :: Bool = false,
          preWhite  :: Bool = false,
          sort      :: Bool = true,
          init      :: Union{Matrix, Nothing} = ○,
          tol       :: Real = 0.,
          maxiter   :: Int  = 1000,
          verbose   :: Bool = false,
          threaded  :: Bool = true,
      eVar     :: TeVaro = ○,
      eVarMeth :: Function = searchsortedfirst)

    k<3 && m<2 && throw(ArgumentError("Either `k` must be equal to at least 2 or `m` must be equal to at least 2"))
    #if m==1 fullModel=false end

    # if input ≠:d the input is supposed to be the cross-covariance matrices
    input==:d ? 𝒞=_crossCov(𝐗, m, k;
                            covEst=covEst, dims=dims, meanX=meanX) : 𝒞=𝐗

    𝒢=deepcopy(𝒞)
    if trace1 || w ≠ ○ _normalize!(𝒢, m, k, trace1, w) end

    if eVar isa Int && size(𝒢[1, 1, 1], 1) ≤ eVar ≤ 0
        eVar=size(𝒢[1, 1, 1], 1)
    end

    iter, conv, oldconv, 😋, conv_ = 1, 0., 1.01, false, 0.
    tol==0. ? tolerance = √eps(real(type)) : tolerance = tol

    # pre-whitening
    if preWhite
        # for the moment being computed on the average across k and m
        # if eVar is not an Int.
        # This way for all i=1:m the dimensionality reduction is fixed
        # for the case m=1 this is the classical whitening
        if !(eVar isa Int)
            G=ℍ(mean(𝒢[κ, i, i] for κ=1:k, i=1:m))
            p, arev = _getssd!(eVar, eigvals(G), size(𝒢[1, 1, 1], 1), eVarMeth)
            𝑾=[whitening(ℍ(mean(𝒢[κ, i, i] for κ=1:k)); eVar=p) for i=1:m]
        else
            𝑾=[whitening(ℍ(mean(𝒢[κ, i, i] for κ=1:k)); eVar=eVar) for i=1:m]
        end

        if m==1
            @inbounds for κ=1:k
                𝒢[κ, 1, 1] = 𝑾[1].F' * 𝒢[κ, 1, 1] * 𝑾[1].F
            end
        else
            # off-diagonal blocks
            @inbounds for κ=1:k, i=1:m-1, j=i+1:m
                𝒢[κ, i, j] = 𝑾[i].F' * 𝒢[κ, i, j] * 𝑾[j].F
                𝒢[κ, j, i] = 𝒢[κ, i, j]'
            end
            # diagonal blocks
            @inbounds for κ=1:k, i=1:m
                𝒢[κ, i, i] = 𝑾[i].F' * 𝒢[κ, i, i] * 𝑾[i].F
            end
        end
    end
    n=size(𝒢[1, 1, 1], 1)

    # determie whether running in multi-threaded mode
    ⏩ = n>=Threads.nthreads() && Threads.nthreads()>1 && threaded

    # initialization of 𝐔[i], i=1:m, as the eigenvectors of sum_k,j(𝒢_k,i,j*𝒢_k,i,j')
    # see Eq. 17 of Congedo, Phlypo & Pham(2011), or with the provided matrices
    # note: BLAS.gemm supports complex matrices
    ggt(κ::Int, i::Int, j::Int) = BLAS.gemm('N', 'T', 𝒢[κ, i, j], 𝒢[κ, i, j])
    if m>1 # gmca, gcca, majd
        if init === nothing
            if algo==:NoJoB
                𝐔 = [Matrix{type}(I, n, n) for i=1:m]
            else
                if fullModel
                    𝐔 = [eigvecs(mean(ggt(κ, i, j) for j=1:m, κ=1:k)) for i=1:m]
                else
                    𝐔 = [eigvecs(mean(ggt(κ, i, j) for j=1:m, κ=1:k if j≠i)) for i=1:m]
                end
            end
        else
            𝐔 = init
        end
    else  # ajd
        if init === nothing
            if algo==:NoJoB
                𝐔 = [Matrix{type}(I, n, n)]
            else
                𝐔 = [eigvecs(mean(ggt(κ, 1, 1) for κ=1:k))]
            end
        else
            𝐔 = [init]
        end
    end

    function updateR!(η, i, j)
        # 𝐑[η] += (𝒢[κ, i, j] * 𝐔[j][:, η]) times its transpose
        # don't use BLAS for complex data
        if ⏩ # if threaded don't share memory for Ω but use 𝛀
            if type<:Real
                @inbounds for κ=1:k
                    𝛀[η][:, κ] = BLAS.gemv('N', 𝒢[κ, i, j], 𝐔[j][:, η])
                end
                𝐑[η] += Hermitian(BLAS.gemm('N', 'T', 𝛀[η], 𝛀[η])) # (Ω * Ω')
            else
                @inbounds for κ=1:k
                    𝛀[η][:, κ] = 𝒢[κ, i, j] * 𝐔[j][:, η]
                end
                𝐑[η] += Hermitian(𝛀[η] * 𝛀[η]') # (Ω * Ω')
            end
        else
            if type<:Real
                @inbounds for κ=1:k
                    Ω[:, κ] = BLAS.gemv('N', 𝒢[κ, i, j], 𝐔[j][:, η])
                end
                𝐑[η] += Hermitian(BLAS.gemm('N', 'T', Ω, Ω)) # (Ω * Ω')
            else
                @inbounds for κ=1:k
                    Ω[:, κ] = 𝒢[κ, i, j] * 𝐔[j][:, η]
                end
                𝐑[η] += Hermitian(Ω * Ω') # (Ω * Ω')
            end
        end
    end

    function update!(m, n, i)

        function udR1!(η) # case m=1
            fill!(𝐑[η], type(0))
            updateR!(η, 1, 1)
        end

        function udRm!(η, i)  # case m>1
            fill!(𝐑[η], type(0))
            for j=1:m i≠j ? updateR!(η, i, j) : nothing end # j ≠ i
            fullModel ? updateR!(η, i, i) : nothing         # j = i
        end

        if m==1
            ⏩ ? (@threads for η=1:n udR1!(η) end) : (for η=1:n udR1!(η) end)
        else
            ⏩ ? (@threads for η=1:n udRm!(η, i) end) : (for η=1:n udRm!(η, i) end)
        end
    end

    # pre-allocate memory
    𝐑 = HermitianVector([Hermitian(zeros(type, n, n)) for η=1:n])
    ⏩ ? (𝛀 = [Matrix{type}(undef, n, k) for η=1:n]) :
         (Ω = Matrix{type}(undef, n, k))

    # # # # # here starts the algorithm

    if algo==:OJoB
        verbose && @info("Iterating OJoB algorithm...")
        while true
            conv_ =0.
            @inbounds for i=1:m # m optimizations for updating 𝐔[1]...𝐔[m]

                update!(m, n, i)

                # do 1 power iteration, not worth threading here
                for η=1:n 𝐔[i][:, η] = 𝐑[η] * 𝐔[i][:, η] end

                # sum of squares of the norm of power iteration vectors
                conv_ += PosDefManifold.ss(𝐔[i])

                # Lodwin Orthogonalization and update 𝐔[i]<-UV', with svd(𝐔[i])=UwV'
                𝐔[i] = PosDefManifold.nearestOrth(𝐔[i])
            end

            conv_ =sqrt(conv_ /(n^2*m))
            iter==1 ? conv=1. : conv = abs((conv_-oldconv)/oldconv)  # relative change

            verbose && println("iteration: ", iter, "; convergence: ", conv)
            (diverging = conv < 0) && verbose && @warn("OJoB diverged at:", iter)
            (overRun = iter == maxiter) && @warn("OJoB: reached the max number of iterations before convergence:", iter)
            (😋 = 0. <= conv <= tolerance) || overRun==true ? break : nothing
            oldconv=conv_
            iter += 1
        end # while
    else # NoJoB
        verbose && @info("Iterating NoJoB algorithm...")

        # solve Lx=𝐑[η]*𝐔[i][:, η] for x and L'y=x for y
        # and scale the ηth column as 𝐔[i][:, η] <- y/sqrt(y'𝐑[η]t)
        function triS!(cho, η,  i)
            y=cho.U\(cho.L\(𝐑[η]*𝐔[i][:, η]))
            𝐔[i][:, η]=y*inv(sqrt(PosDefManifold.qf(y, 𝐑[η])))
        end

        # thread sum of 𝐑[η] and the solutions to triangular systems
        # only if n is at least Threads.nthreads()*2 (no worth otherwise)
        ⏩x = ⏩ && n≥Threads.nthreads()*2

        while true
            conv_ =0.
            @inbounds for e2=1:2, i=1:m # m optimizations for updating 𝐔[1]...𝐔[m]
                                        # double loop to avoid oscillating convergence
                update!(m, n, i)

                # do 1 power iteration
                cho=cholesky(⏩x ? fVec(sum, 𝐑) : sum(𝐑)) # Cholesky LL'of 𝐑[1]+...+𝐑[n]

                ⏩x ? (@threads for η=1:n triS!(cho, η,  i) end) :
                            (for η=1:n triS!(cho, η,  i) end)

                conv_+=PosDefManifold.ss(𝐔[i])
            end

            conv_=conv_/(2*n^2*m) # 2 because of the double loop
            iter==1 ? conv=1. : conv = abs((conv_-oldconv)/oldconv)  # relative change

            verbose && println("iteration: ", iter, "; convergence: ", conv)
            (diverging = conv < 0) && verbose && @warn("NoJoB diverged at:", iter)
            (overRun = iter == maxiter) && @warn("NoJoB: reached the max number of iterations before convergence:", iter)
            (😋 = 0. <= conv <= tolerance) || overRun==true ? break : nothing
            oldconv=conv_
            iter += 1
        end # while
    end

    verbose && @info("Convergence has "*(😋 ? "" : "not ")*"been attained.\n\n")


    # scale and permute the vectors of U_1,...,U_m
    if sort
        algo==:NoJoB ? for i=1:m normalizeCol!(𝐔[i], 1:size(𝐔[i], 2)) end : nothing
        λ = m==1 ? _permute!(𝐔[1], 𝒢, k, :c) :
                   _flipAndPermute!(𝐔, 𝒢, m, k, :c)
    else
        λ = m==1 ? mean([𝐔[1][:, η]'*𝒢[l, 1, 1]*𝐔[1][:, η] for η=1:n] for l=1:k) :
                   mean([𝐔[i][:, η]'*𝒢[l, i, j]*𝐔[j][:, η] for η=1:n] for l=1:k, j=1:m, i=1:m if i≠j)
    end

    if preWhite
        algo==:OJoB ? 𝐕=[𝐔[i]'*𝑾[i].iF for i=1:m] :
                      𝐕=[pinv(𝐔[i])*𝑾[i].iF for i=1:m] # algo==:NoJoB
        for i=1:m 𝐔[i] = 𝑾[i].F * 𝐔[i] end
    else
        algo==:OJoB ? 𝐕=[Matrix(𝐔[i]') for i=1:m] :
                      𝐕=[pinv(𝐔[i]) for i=1:m] # algo==:NoJoB
    end

    return m>1 ? (𝐔, 𝐕, λ, iter, conv) : (𝐔[1], 𝐕[1], λ, iter, conv)
end
