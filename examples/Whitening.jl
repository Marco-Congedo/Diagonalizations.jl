using Diagonalizations, LinearAlgebra, PosDefManifold, Test

# Method (1)
n, t=10, 100
X=genDataMatrix(n, t)
C=(X*X')/t
wC=whitening(Hermitian(C); simple=true)
# or, shortly
wC=whitening(ℍ(C); simple=true)

# Method (2)
pX=whitening(X; simple=true)
@test wC.F'*C*wC.F≈I

# Method (3)
k=10
Xset=[genDataMatrix(n, t) for i=1:k]

# whitening on the average covariance matrix
w=whitening(Xset)

# ... selecting subspace dimension allowing an explained variance = 0.5
w=whitening(Xset; eVar=0.5)

# ... averaging the covariance matrices using the logEuclidean metric
w=whitening(Xset; metric=logEuclidean, eVar=0.5)

# ... giving weights `w` to the covariance matrices
w=whitening(Xset; metric=logEuclidean, w=abs2.(randn(k)), eVar=0.5)

# ... subtracting the mean
w=whitening(Xset; meanX=nothing, metric=logEuclidean, w=abs2.(randn(k)), eVar=0.5)

# whitening on the average of the covariance matrices computed along dims 1
w=whitening(Xset; dims=1)

# explained variance
w.eVar

# name of the filter
w.name

using Plots
# plot regularized accumulated eigenvalues
plot(w.arev)

# plot the original covariance matrix and the whitened covariance matrix
 Cmax=maximum(abs.(C));
 h1 = heatmap(C, clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="C");
 D=wC.F'*C*wC.F;
 h2 = heatmap(D, clim=(0, 1), yflip=true, c=:amp, title="F'*C*F");
 📈=plot(h1, h2, size=(700, 300))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigWhitening.png")
