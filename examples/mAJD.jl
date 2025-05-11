using Diagonalizations, LinearAlgebra, PosDefManifold, Test

##  Create data for testing the case k>1, m>1 ##
# `t` is the number of samples,
# `m` is the number of datasets,
# `k` is the number of observations,
# `n` is the number of variables,
# `noise` must be smaller than 1.0. The smaller the noise, the more data are correlated
# Output k vectors of m data data matrices
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

function getData(::Type{Complex{T}}, t, m, k, n, noise) where {T<:AbstractFloat}
    # create m identical data matrices and rotate them by different
    # random orthogonal matrices V_1,...,V_m
    𝐕=[randU(ComplexF64, n) for i=1:m] # random orthogonal matrices
    # variables common to all subjects with unique variance profile across k
    X=[(abs2.(randn(n))).*randn(ComplexF64, n, t) for s=1:k]
    # each subject has this common part plus a random part
    𝐗=[[𝐕[i]*((1-noise)*X[s] + noise*randn(ComplexF64, n, t)) for i=1:m] for s=1:k]
    return 𝐗, 𝐕
end


# REAL data
# do joint blind source separation of non-stationary data
t, m, n, k, noise = 200, 5, 4, 6, 0.1
Xset, Vset=getData(t, m, k, n, noise)
𝒞=Array{Matrix}(undef, k, m, m)
for s=1:k, i=1:m, j=1:m 𝒞[s, i, j]=(Xset[s][i]*Xset[s][j]')/t end

aX=majd(Xset; fullModel=true, algorithm=:OJoB)
# the spForm index of the estimated demixing matrices times the true
# mixing matrix must be low
@test mean(spForm(aX.F[i]'*Vset[i]) for i=1:m)<0.1

# test the same using NoJoB algorithm
aX=majd(Xset; fullModel=true, algorithm=:NoJoB)
@test mean(spForm(aX.F[i]'*Vset[i]) for i=1:m)<0.1

# plot the original cross-covariance matrices and the rotated
# cross-covariance matrices

# Get all products 𝐔[i]' * 𝒞[l, i, j] * 𝐔[j]
function _rotate_crossCov(𝐔, 𝒞, m, k)
    𝒮=Array{Matrix}(undef, k, m, m)
    @inbounds for l=1:k, i=1:m, j=1:m 𝒮[l, i, j]=𝐔[i]'*𝒞[l, i, j]*𝐔[j] end
    return 𝒮
end

# Put all `k` cross-covariances in a single matrix
# of dimension m*n x m*n for visualization
function 𝒞2Mat(𝒞::AbstractArray, m, k)
    n=size(𝒞[1, 1, 1], 1)
    C=Matrix{Float64}(undef, m*n, m*n)
    for i=1:m, j=1:m, x=1:n, y=1:n C[i*n-n+x, j*n-n+y]=𝒞[k, i, j][x, y] end
    return C
end

using Plots

Cset=[𝒞2Mat(𝒞, m, s) for s=1:k]
 Cmax=maximum(maximum(abs.(C)) for C ∈ Cset)
 h1 = heatmap(Cset[1], clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="all cross-cov, k=1")
 h2 = heatmap(Cset[2], clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="all cross-cov, k=2")
 h3 = heatmap(Cset[2], clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="all cross-cov, k=3")
 h4 = heatmap(Cset[2], clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="all cross-cov, k=4")
 h5 = heatmap(Cset[2], clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="all cross-cov, k=5")
 h6 = heatmap(Cset[2], clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="all cross-cov, k=6")
 📈=plot(h1, h2, h3, h4, h5, h6, size=(1200,550))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigmAJD1.png")

𝒮=_rotate_crossCov(aX.F, 𝒞, m, k)
 Sset=[𝒞2Mat(𝒮, m, s) for s=1:k]
 Smax=maximum(maximum(abs.(S)) for S ∈ Sset)
 h11 = heatmap(Sset[1], clim=(-Smax, Smax), yflip=true, c=:bluesreds, title="rotated cross-cov, k=1")
 h12 = heatmap(Sset[2], clim=(-Smax, Smax), yflip=true, c=:bluesreds, title="rotated cross-cov, k=2")
 h13 = heatmap(Sset[2], clim=(-Smax, Smax), yflip=true, c=:bluesreds, title="rotated cross-cov, k=3")
 h14 = heatmap(Sset[2], clim=(-Smax, Smax), yflip=true, c=:bluesreds, title="rotated cross-cov, k=4")
 h15 = heatmap(Sset[2], clim=(-Smax, Smax), yflip=true, c=:bluesreds, title="rotated cross-cov, k=5")
 h16 = heatmap(Sset[2], clim=(-Smax, Smax), yflip=true, c=:bluesreds, title="rotated cross-cov, k=6")
 📉=plot(h11, h12, h13, h14, h15, h16, size=(1200,550))
# savefig(📉, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigmAJD2.png")

# COMPLEX data
# do joint blind source separation of non-stationary data
t, m, n, k, noise = 200, 5, 4, 6, 0.1
Xcset, Vcset=getData(ComplexF64, t, m, k, n, noise)
𝒞=Array{Matrix}(undef, k, m, m)
for s=1:k, i=1:m, j=1:m 𝒞[s, i, j]=(Xcset[s][i]*Xcset[s][j]')/t end

aXc=majd(Xcset; fullModel=true, algorithm=:OJoB)
# the spForm index of the estimated demixing matrices times the true
# mixing matrix must be low
@test mean(spForm(aXc.F[i]'*Vcset[i]) for i=1:m)<0.1

# test the same using NoJoB algorithm
aXc=majd(Xcset; fullModel=true, algorithm=:NoJoB)
@test mean(spForm(aXc.F[i]'*Vcset[i]) for i=1:m)<0.1
