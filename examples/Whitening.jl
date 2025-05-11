using Diagonalizations, LinearAlgebra, PosDefManifold, Test

# Method (1) real
n, t=10, 100
X=genDataMatrix(n, t)
C=(X*X')/t
wC=whitening(Hermitian(C); simple=true)
# or, shortly
wC=whitening(ℍ(C); simple=true)

# Method (1) complex
Xc=genDataMatrix(ComplexF64, n, t)
Cc=(Xc*Xc')/t
wCc=whitening(Hermitian(Cc); simple=true)


# Method (2) real
wX=whitening(X; simple=true)
@test wC.F'*C*wC.F≈I
@test wX.F'*C*wX.F≈I
@test wX≈wC

# Method (2) complex
wXc=whitening(Xc; simple=true)
@test wCc.F'*Cc*wCc.F≈I
@test wXc.F'*Cc*wXc.F≈I
@test wXc≈wCc


# Method (3) real
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


# Method (3) complex
k=10
Xcset=[genDataMatrix(ComplexF64, n, t) for i=1:k]

# whitening on the average covariance matrix
wc=whitening(Xcset)

# ... selecting subspace dimension allowing an explained variance = 0.5
wc=whitening(Xcset; eVar=0.5)

# ... averaging the covariance matrices using the logEuclidean metric
wc=whitening(Xcset; metric=logEuclidean, eVar=0.5)

# ... giving weights `w` to the covariance matrices
wc=whitening(Xset; metric=logEuclidean, w=abs2.(randn(k)), eVar=0.5)

# ... subtracting the mean
wc=whitening(Xcset; meanX=nothing, metric=logEuclidean, w=abs2.(randn(k)), eVar=0.5)

# whitening on the average of the covariance matrices computed along dims 1
wc=whitening(Xcset; dims=1)

# explained variance
wc.eVar

# name of the filter
wc.name
