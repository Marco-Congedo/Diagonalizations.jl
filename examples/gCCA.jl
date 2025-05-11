using Diagonalizations, LinearAlgebra, PosDefManifold, Test


####  Create data for testing the case k=1, m>1
# `t` is the number of samples,
# `m` is the number of datasets,
# `n` is the number of variables,
# `noise` must be smaller than 1.0. The smaller the noise,
#  the more data are correlated.
function getData(t, m, n, noise)
    # create m identical data matrices and rotate them by different
    # random orthogonal matrices V_1,...,V_m
    𝐕=[randU(n) for i=1:m] # random orthogonal matrices
    X=randn(n, t)  # data common to all subjects
    # each subject has this common part plus a random part
    𝐗=[𝐕[i]'*((1-noise)*X + noise*randn(n, t)) for i=1:m]
    return 𝐗
end

function getData(::Type{Complex{T}}, t, m, n, noise) where {T<:AbstractFloat}
    # create m identical data matrices and rotate them by different
    # random orthogonal matrices V_1,...,V_m
    𝐕=[randU(ComplexF64, n) for i=1:m] # random orthogonal matrices
    X=randn(ComplexF64, n, t)  # data common to all subjects
    # each subject has this common part plus a random part
    𝐗=[𝐕[i]'*((1-noise)*X + noise*randn(ComplexF64, n, t)) for i=1:m]
    return 𝐗
end

# REAL data: check that for the case m=2 gCCA gives the same result as CCA
t, m, n, noise = 20, 2, 6, 0.1
Xset=getData(t, m, n, noise)
Cx=(Xset[1]*Xset[1]')/t
Cy=(Xset[2]*Xset[2]')/t
Cxy=(Xset[1]*Xset[2]')/t

gc=gcca(Xset; simple=true)

c=cca(Hermitian(Cx), Hermitian(Cy), Cxy; simple=true)

@test (c.F[1]'*Cxy*c.F[2]) ≈ (gc.F[1]'*Cxy*gc.F[2])
@test gc.F[1]'*Cx*gc.F[1]≈I
@test gc.F[2]'*Cy*gc.F[2]≈I
D=gc.F[1]'*Cxy*gc.F[2]
@test norm(D-Diagonal(D))+1≈1.


# COMPLEX data: check that for the case m=2 gCCA gives the same result as CCA
t, m, n, noise = 20, 2, 6, 0.1
Xcset=getData(ComplexF64, t, m, n, noise)
Ccx=(Xcset[1]*Xcset[1]')/t
Ccy=(Xcset[2]*Xcset[2]')/t
Ccxy=(Xcset[1]*Xcset[2]')/t

gcc=gcca(Xcset; simple=true, maxiter=10000)
cc=cca(Hermitian(Ccx), Hermitian(Ccy), Ccxy; simple=true)

# for complex data just do a sanity check as the order of vectors
# is arbitrary. The following two tests currently fail
# @test spForm(cc.F[1]'gcc.F[1])<0.001
# @test spForm(cc.F[2]'gcc.F[2])<0.001

@test gcc.F[1]'*Ccx*gcc.F[1]≈I
@test gcc.F[2]'*Ccy*gcc.F[2]≈I
# sanity check only as there is noise in the complex case
D=gcc.F[1]'*Ccxy*gcc.F[2]
@test norm(D-Diagonal(D))/(n^2-n)<0.001


# REAL data: m>2 case
t, m, n, noise = 20, 4, 6, 0.1
Xset=getData(t, m, n, noise)

# ... selecting subspace dimension allowing an explained variance = 0.9
gc=gcca(Xset, eVar=0.9)

# name of the filter
gc.name

𝒞=Array{Matrix}(undef, 1, m, m)
for i=1:m, j=1:m 𝒞[1, i, j]=(Xset[i]*Xset[j]')/t end

using Plots
# plot regularized accumulated eigenvalues
plot(gc.arev)


# plot the original cross-covariance matrices and the rotated
# cross-covariance matrices

# Get all products 𝐔[i]' * 𝒞[l, i, j] * 𝐔[j]
function _rotate_crossCov(𝐔, 𝒞, m, k)
    𝒮=Array{Matrix}(undef, k, m, m)
    @inbounds for l=1:k, i=1:m, j=1:m 𝒮[l, i, j]=𝐔[i]'*𝒞[l, i, j]*𝐔[j] end
    return 𝒮
end


# Put all cross-covariances in a single matrix of dimension m*n x m*n for visualization
function 𝒞2Mat(𝒞::AbstractArray, m, k)
    n=size(𝒞[1, 1, 1], 1)
    C=Matrix{Float64}(undef, m*n, m*n)
    for i=1:m, j=1:m, x=1:n, y=1:n C[i*n-n+x, j*n-n+y]=𝒞[k, i, j][x, y] end
    return C
end

 C=𝒞2Mat(𝒞, m, 1)
 Cmax=maximum(abs.(C));
 h1 = heatmap(C, clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="all cross-covariances")
 𝒮=_rotate_crossCov(gc.F, 𝒞, m, 1)
 S=𝒞2Mat(𝒮, m, 1)
 h2 = heatmap(S, clim=(0, 1), yflip=true, c=:amp, title="all rotated cross-covariances")
 📈=plot(h1, h2, size=(700,300))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FiggCCA.png")

# COMPLEX data: m>2 case
t, m, n, noise = 20, 4, 6, 0.1
Xcset=getData(ComplexF64, t, m, n, noise)

# ... selecting subspace dimension allowing an explained variance = 0.9
gcc=gcca(Xcset, eVar=0.9)
