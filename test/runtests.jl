using LinearAlgebra, Statistics, PosDefManifold, Test,
      Diagonalizations

# for more tests see the `Examples` folder

const errtol=1e-6

# compare two matrices
function compare(a::AbstractArray, b::AbstractArray)
     @test norm(a-b)/length(a) < errtol
end

# check that cov(F'XX'F)=D arranging terms for the two possible values of `dims`
function compare(d::Int, X::AbstractArray, F::AbstractArray, D::AbstractArray)
    d==1 ? Y=X*F : Y=F'*X
    d==1 ? C=((Y'*Y)/size(Y, 1))[1:size(D, 1), 1:size(D, 2)] :
           C=((Y*Y')/size(Y, 2))[1:size(D, 1), 1:size(D, 2)]
    compare(C, D)
end

# check that D is approximately diagonal
isApproxD(D)=norm(D-Diagonal(D))<errtol

####  Create data for testing the case k=1, m>1
# `t` is the number of samples, e.g.,
    # in Tangent space transfer learning is the number of trials,
# `m` is the number of datasets, e.g., subjects
# `n` is the number of variables, e.g.,
    # in Tangent space transfer learning is the number of elements of the tangent vectors
# `noise` must be smaller than 1.0. The smaller the noise, the more data are correlated
function getData(t, m, n, noise)
    # create m identical data matrices and rotate them by different
    # random orthogonal matrices V_1,...,V_m
    𝐕=[randU(n) for i=1:m] # random orthogonal matrices
    X=randn(n, t)  # data common to all subjects
    # each subject has this common part plus a random part
    𝐗=[𝐕[i]'*((1-noise)*X + noise*randn(n, t)) for i=1:m]
    return 𝐗, 𝐕
end

# As before but output k vectors of data as per the other method
# used for the case m>1, k>1
function getData(t, m, k, n, noise)
    # create m identical data matrices and rotate them by different
    # random orthogonal matrices V_1,...,V_m
    𝐕=[randU(n) for i=1:m] # random orthogonal matrices
    # variables common to all subjects with unique variance profile across k
    X=[(abs2.(randn(n))).*randn(n, t) for s=1:k]
    # each subject has this common part plus a random part
    𝐗=[[𝐕[i]*((1-noise)*X[s] + noise*randn(n, t)) for i=1:m] for s=1:k]
    return 𝐗, 𝐕
end


# Generate data for tests
n, t=3, 10
X=[ 0.0161263  -0.228679  -0.610958;
    0.884571    1.53687    0.569085;
   -1.26195    -1.35456   -0.303478;
   -0.275838    0.775055  -0.461387;
    1.35221    -0.890299  -0.224943;
   -0.204674    0.318228   0.266376;
   -0.203296   -1.83068   -0.869954;
   -0.726922   -0.529861  -0.609168;
   -0.0809636  -0.107088   0.0315276;
    0.177739    0.124636   0.495547]

Y=[ 2.16758   -2.29723   -1.92301;
    3.48593   -0.357984  -0.457085;
    3.02924   -1.5643    -1.38311;
    1.53344   -1.79264   -1.75126;
    0.990636  -0.618866  -0.643236;
    1.37825    0.704042   0.847845;
    0.324096   0.43795    0.142093;
   -3.68015    0.539658   0.436929;
    0.141023  -0.303311  -0.384726;
   -0.903474   1.20838    0.944415]

Cx=Symmetric((X'*X)/t)
Cy=Symmetric((Y'*Y)/t)
Cxy=(X'*Y)/t

𝐗 = [genDataMatrix(t, n) for i = 1:5]
𝐘 = [genDataMatrix(t, n) for i = 1:5]
𝐂x=Vector{Hermitian}([Hermitian((X'*X)/t) for X in 𝐗])
𝐂y=Vector{Hermitian}([Hermitian((Y'*Y)/t) for Y in 𝐘])
𝐂xy=Vector{Matrix}([(X'*Y)/size(X, 1) for (X, Y) in zip(𝐗, 𝐘)])

X_=[X, Matrix(X')]

############  Begin Test  ###################################

@testset "PCA" begin
    ## method (1)
    pC=pca(Cx; simple=true)
    @test Cx≈pC.F*pC.D*pC.iF

    ## method (2)
    for d=1:2
        p=pca(X_[d]; dims=d)
        compare(d, X_[d], p.F, p.D)
        p=pca(X_[d]; dims=d, eVar=3)
        compare(d, X_[d], p.F, p.D)
        p=pca(X_[d]; dims=d, eVar=0.5)
        compare(d, X_[d], p.F, p.D)
        p=pca(X_[d]; dims=d, covEst=LShrLW, eVar=0.5)
        # p=pca(X_[d]; covEst=NShrLW, eVar=0.5)
    end

    p=pca(X; simple=true)
    @test p==pC

    ## method (3)
    p=pca(𝐗; covEst=SCM, eVar=0.5, metric=logEuclidean)
    G=PosDefManifold.mean(logEuclidean, 𝐂x)
    D=p.F'*G*p.F
    @test isApproxD(D)
    @test Diagonal(D)≈Diagonal(reverse(eigvals(G))[1: size(D, 1)])
end



@testset "Whitening" begin
    ## method (1)
    wC=whitening(Cx; simple=true)
    @test wC.F'*Cx*wC.F≈I

    ## method (2)
    for d=1:2
        w=whitening(X_[d]; dims=d)
        compare(d, X_[d], w.F, Diagonal(ones(eltype(w.F), size(w.F, 2))))
        w=whitening(X_[d]; dims=d, eVar=3, eVarMeth=searchsortedlast)
        compare(d, X_[d], w.F, Diagonal(ones(eltype(w.F), size(w.F, 2))))
        w=whitening(X_[d]; dims=d, eVar=0.5, eVarMeth=searchsortedlast)
        compare(d, X_[d], w.F, Diagonal(ones(eltype(w.F), size(w.F, 2))))
        w=whitening(X_[d]; covEst=LShrLW, eVar=0.5)
        # w=whitening(X_[d]; covEst=NShrLW, eVar=0.5)
    end
    w=whitening(X; simple=true)
    @test w==wC

    ## method (3)
    w=whitening(𝐗; covEst=SCM, simple=true, metric=logEuclidean)
    G=PosDefManifold.mean(logEuclidean, 𝐂x)
    @test w.F'*G*w.F≈I
end


@testset "MCA" begin
    ## method (1)
    mC=mca(Cxy, simple=true)
    @test Cxy≈mC.F[1]*mC.D*mC.F[2]'

    ## method (2)
    m=mca(X, Y, simple=true)
    @test isApproxD(m.F[1]'*Cxy*m.F[2])
    @test m==mC

    ## method (3)
    m=mca(𝐗, 𝐘; eVar=0.99)
    m𝐂=mca(mean(𝐂xy); eVar=0.99)
    @test m==m𝐂
end


@testset "CCA" begin
    c=cca(X, Y)
    @test c.F[1]'*Cx*c.F[1]≈I
    @test c.F[2]'*Cy*c.F[2]≈I
    @test isApproxD(c.F[1]'*Cxy*c.F[2])

    # CCA of two vectors of data matrices
    c=cca(𝐗, 𝐘; metric=Euclidean, eVar=0.99)
    c𝐂=cca(Symmetric(mean(𝐂x)), Symmetric(mean(𝐂y)), mean(𝐂xy); eVar=0.99)
    @test c==c𝐂
end



@testset "CSP" begin
    t, n=20, 10
    X=genDataMatrix(t, n)
    Y=genDataMatrix(t, n)
    Cx=Symmetric((X'*X)/t)
    Cy=Symmetric((Y'*Y)/t)

    c=csp(X, Y)
    c=csp(Cx, Cy)
    @test isApproxD(c.F'*Cx*c.F)
    @test isApproxD(c.F'*Cy*c.F)
    @test c.F'*(Cx+Cy)*c.F≈I

    d=csp(Cx, Cy; eVarC=0.)
    @test isApproxD(d.F'*Cx*d.F)
    @test isApproxD(d.F'*Cy*d.F)
    @test d.F'*(Cx+Cy)*d.F≈I

    # the two solutions are identical out of a possible sign ambiguity on the evectors
    @test abs.(c.iF*d.F)≈I
end


@testset "CSTP" begin
    t, n=10, 20
    𝐗 = [genDataMatrix(t, n) for i = 1:10]
    Xfixed=randn(t, n)./1
    for i=1:length(𝐗) 𝐗[i]+=Xfixed end
    C1=Hermitian( mean((X'*X)/t for X∈𝐗) )
    C2=Hermitian( mean((X*X')/n for X∈𝐗) )
    c=cstp(Xfixed, C1, C2; simple=true)
    @test c.F[1]'*C2*c.F[1]≈I
    @test c.F[2]'*C1*c.F[2]≈I
    Z=c.F[1]'*Xfixed*c.F[2]
    n=minimum(size(Z))
    @test isApproxD(Z[1:n, 1:n])
end


@testset "AJD" begin
    t, n, k=50, 10, 50
    A=randn(n, n) # mixing matrix
    𝐗 = [genDataMatrix(t, n) for i = 1:k]
    Xfixed=randn(t, n)./1
    for i=1:length(𝐗) 𝐗[i]+=Xfixed end
    𝐂 = ℍVector([ℍ((𝐗[s]'*𝐗[s])/t) for s=1:k])
    aX=ajd(𝐗; simple=true)
    aC=ajd(𝐂; simple=true)
    @test aX≈aC

    # # # Test orthogonal AJD algorithms
    # create 20 random commuting matrices
    # they all have the same eigenvectors
    𝐂=randP(3, 20; eigvalsSNR=Inf, commuting=true)
    # estimate the approximate joint diagonalizer (ajd) using orthogonal solvers
    # the ajd must be equivalent to the eigenvector matrix of any of the matrices in 𝐂
    a=ajd(𝐂; algorithm=:OJoB)
    @test norm([spForm(a.F'*eigvecs(C)) for C ∈ 𝐂])/20<√errtol
    a=ajd(𝐂; algorithm=:JADE)
    @test norm([spForm(a.F'*eigvecs(C)) for C ∈ 𝐂])/20<√errtol
    a=ajd(𝐂; algorithm=:JADEmax)
    @test norm([spForm(a.F'*eigvecs(C)) for C ∈ 𝐂])/20<√errtol

    # # # Test non-orthogonal AJD algorithms
    # generate positive definite matrices with model A*D_κ*D, where A is an
    # invertible mixing matrix and D_κ, for all κ=1:k, are diagonal matrices.
    # The estimated AJD matrix must be the inverse of A
    # and all transformed matrices bust be diagonal
    n, k=3, 10
    Dest=PosDefManifold.randΛ(eigvalsSNR=Inf, n, k)
    # make the problem identifiable
    for i=1:k Dest[k][1, 1]*=i/(k/2) end
    for i=1:k Dest[k][3, 3]/=i/(k/2) end
    A=randn(n, n) # non-singular mixing matrix
    Cset3=Vector{Hermitian}([Hermitian(A*D*A') for D ∈ Dest])
    a=ajd(Cset3; algorithm=:NoJoB, eVarC=n)
    @test spForm(a.F'*A)<√errtol
    @test mean(nonD(a.F'*Cset3[i]*a.F) for i=1:k)<errtol
    a=ajd(Cset3; algorithm=:LogLike, eVarC=n)
    @test spForm(a.F'*A)<√errtol
    @test mean(nonD(a.F'*Cset3[i]*a.F) for i=1:k)<errtol
    a=ajd(Cset3; algorithm=:LogLikeR, eVarC=n)
    @test spForm(a.F'*A)<√errtol
    @test mean(nonD(a.F'*Cset3[i]*a.F) for i=1:k)<errtol
    a=ajd(Cset3; algorithm=:GAJD, eVarC=n)
    @test spForm(a.F'*A)<√errtol*10 # GAJD has problems sometimes, increase the tolerance
    @test mean(nonD(a.F'*Cset3[i]*a.F) for i=1:k)<errtol*10
    a=ajd(Cset3; algorithm=:QNLogLike, eVarC=n)
    @test spForm(a.F'*A)<√errtol
    @test mean(nonD(a.F'*Cset3[i]*a.F) for i=1:k)<errtol
end


@testset "gMCA" begin
    t, m, n, noise = 200, 2, 6, 0.1
    𝐗, 𝐕 = getData(t, m, n, noise)
    Cx=(𝐗[1]*𝐗[1]')/t
    Cy=(𝐗[2]*𝐗[2]')/t
    Cxy=(𝐗[1]*𝐗[2]')/t

    # check that for the case m=2 GMCA gives the same result as MCA
    gm=gmca(𝐗; simple=true)

    # do MCA
    m=mca(Cxy; simple=true)

    @test (m.F[1]'*Cxy*m.F[2]) ≈ (gm.F[1]'*Cxy*gm.F[2])
    # the following must be the identity matrix out of a possible sign ambiguity
    @test abs.(m.F[1]'*gm.F[1]) ≈ I
    @test abs.(m.F[2]'*gm.F[2]) ≈ I
end


@testset "gCCA" begin
    t, m, n, noise = 200, 2, 6, 0.1
    𝐗, 𝐕 = getData(t, m, n, noise)
    Cx=(𝐗[1]*𝐗[1]')/t
    Cy=(𝐗[2]*𝐗[2]')/t
    Cxy=(𝐗[1]*𝐗[2]')/t

    # check that for the case m=2 GCCA gives the same result as CCA
    gc=gcca(𝐗; simple=true)
    gm=gmca(𝐗; simple=true)
    # do CCA
    c=cca(𝐗[1], 𝐗[2]; simple=true)

    @test isApproxD(gc.F[1]'*Cxy*gc.F[2])

    @test (c.F[1]'*Cxy*c.F[2]) ≈ (gc.F[1]'*Cxy*gc.F[2])
    @test (gc.F[1]'*Cx*gc.F[1]) ≈ I
    @test (gc.F[2]'*Cy*gc.F[2]) ≈ I

    # the following must be the identity matrix out of a possible sign ambiguity
    @test abs.(c.iF[1]*gc.F[1]) ≈ I
    @test abs.(c.iF[2]*gc.F[2]) ≈ I

    # test that gCCA is equivalent to GMCA on pre-whitened data
    w1=whitening(𝐗[1])
    w2=whitening(𝐗[2])
    Y=[w1.F'*𝐗[1], w2.F'*𝐗[2]]
    C1=(Y[1]*Y[1]')/t
    C2=(Y[2]*Y[2]')/t
    C12=(Y[1]*Y[2]')/t
    gm2=gmca(Y; simple=true, algorithm=:OJoB)

    gm2=gcca(Y; simple=true, algorithm=:OJoB)
    @test gm2.F[1]'*C1*gm2.F[1] ≈ I
    @test gm2.F[2]'*C2*gm2.F[2] ≈ I
    @test isApproxD(gm2.F[1]'*C12*gm2.F[2])
    @test gm2.F[1]'*w1.F'*Cx*w1.F*gm2.F[1] ≈ I
    @test gm2.F[2]'*w2.F'Cy*w2.F*gm2.F[2] ≈ I
    @test isApproxD(gm2.F[1]'*w1.F'*Cxy*w2.F*gm2.F[2])
end


@testset "mAJD" begin
    # do joint blind source separation of non-stationary data
    t, m, n, k, noise = 200, 10, 3, 20, 0.01
    𝐗, 𝐕=getData(t, m, k, n, noise)
    aX=majd(𝐗; fullModel=true, algorithm=:OJoB)
    # the spForm index of the estimated demixing matrices times the true
    # mixing matrix must be low
    @test mean(spForm(aX.F[i]'*𝐕[i]) for i=1:m)<0.05
end
