using Diagonalizations, LinearAlgebra, PosDefManifold, Test

n, t=10, 100
X=genDataMatrix(n, t)
Y=genDataMatrix(n, t)

Cx=Symmetric((X*X')/t)
Cy=Symmetric((Y*Y')/t)
Cxy=(X*Y')/t

# Method (1)
mC=mca(Cxy, simple=true)
@test Cxy≈mC.F[1]*mC.D*mC.F[2]'
D=mC.F[1]'Cxy*mC.F[2]
@test norm(D-Diagonal(D))+1≈1.

# Method (2)
mXY=mca(X, Y, simple=true)
D=mXY.F[1]'*Cxy*mXY.F[2]
@test norm(D-Diagonal(D))+1≈1.
@test mXY==mC

k=10
Xset=[genDataMatrix(n, t) for i=1:k]
Yset=[genDataMatrix(n, t) for i=1:k]

# Method (3)
# maximum covariance analysis of the average covariance and cross-covariance
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
