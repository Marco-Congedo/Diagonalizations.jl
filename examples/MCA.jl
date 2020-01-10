using Diagonalizations, LinearAlgebra, PosDefManifold, Test

# Method (1) real
n, t=10, 100
X=genDataMatrix(n, t)
Y=genDataMatrix(n, t)
Cx=Symmetric((X*X')/t)
Cy=Symmetric((Y*Y')/t)
Cxy=(X*Y')/t
mC=mca(Cxy, simple=true)
@test Cxy≈mC.F[1]*mC.D*mC.F[2]'
D=mC.F[1]'Cxy*mC.F[2]
@test norm(D-Diagonal(D))+1. ≈ 1.

# Method (1) complex
Xc=genDataMatrix(ComplexF64, n, t)
Yc=genDataMatrix(ComplexF64, n, t)
Cxc=Symmetric((Xc*Xc')/t)
Cyc=Symmetric((Yc*Yc')/t)
Cxyc=(Xc*Yc')/t
mCc=mca(Cxyc, simple=true)
@test Cxyc≈mCc.F[1]*mCc.D*mCc.F[2]'
Dc=mCc.F[1]'Cxyc*mCc.F[2]
@test norm(Dc-Diagonal(Dc))+1. ≈ 1.


# Method (2) real
mXY=mca(X, Y, simple=true)
D=mXY.F[1]'*Cxy*mXY.F[2]
@test norm(D-Diagonal(D))+1≈1.
@test mXY==mC

# Method (2) complex
mXYc=mca(Xc, Yc, simple=true)
Dc=mXYc.F[1]'*Cxyc*mXYc.F[2]
@test norm(Dc-Diagonal(Dc))+1. ≈ 1.
@test mXYc==mCc


# Method (3) real
# maximum covariance analysis of the average covariance and cross-covariance
k=10
Xset=[genDataMatrix(n, t) for i=1:k]
Yset=[genDataMatrix(n, t) for i=1:k]

m=mca(Xset, Yset)

# ... selecting subspace dimension allowing an explained variance = 0.5
m=mca(Xset, Yset; eVar=0.5)

# ... subtracting the mean from the matrices in Xset and Yset
m=mca(Xset, Yset; meanX=nothing, meanY=nothing, eVar=0.5)

# mca on the average of the covariance and cross-covariance matrices
# computed along dims 1
m=mca(Xset, Yset; dims=1, eVar=0.5)

# name of the filter
m.name

using Plots
# plot regularized accumulated eigenvalues
plot(m.arev)

# plot the original cross-covariance matrix and the rotated
# cross-covariance matrix
 Cmax=maximum(abs.(Cxy));
 h1 = heatmap(Cxy, clim=(-Cmax, Cmax), yflip=true, c=:bluesreds, title="Cxy");
 D=mC.F[1]'*Cxy*mC.F[2];
 Dmax=maximum(abs.(D));
 h2 = heatmap(D, clim=(0, Dmax), yflip=true, c=:amp, title="F[1]'*Cxy*F[2]");
 📈=plot(h1, h2, size=(700,300))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigMCA.png")

# Method (3) complex
# maximum covariance analysis of the average covariance and cross-covariance

k=10
Xcset=[genDataMatrix(ComplexF64, n, t) for i=1:k]
Ycset=[genDataMatrix(ComplexF64, n, t) for i=1:k]

mc=mca(Xcset, Ycset)

# ... selecting subspace dimension allowing an explained variance = 0.5
mc=mca(Xcset, Ycset; eVar=0.5)

# ... subtracting the mean from the matrices in Xset and Yset
mc=mca(Xcset, Ycset; meanX=nothing, meanY=nothing, eVar=0.5)

# mca on the average of the covariance and cross-covariance matrices
# computed along dims 1
mc=mca(Xcset, Ycset; dims=1, eVar=0.5)

# name of the filter
mc.name
