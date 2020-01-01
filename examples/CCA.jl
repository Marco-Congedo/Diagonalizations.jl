using Diagonalizations, LinearAlgebra, PosDefManifold, Test

n, t=10, 100
X=genDataMatrix(n, t)
Y=genDataMatrix(n, t)

Cx=Symmetric((X*X')/t)
Cy=Symmetric((Y*Y')/t)
Cxy=(X*Y')/t

# Method (1)
cC=cca(Cx, Cy, Cxy, simple=true)
@test cC.F[1]'*Cx*cC.F[1]≈I
@test cC.F[2]'*Cy*cC.F[2]≈I
D=cC.F[1]'*Cxy*cC.F[2]
@test norm(D-Diagonal(D))+1≈1.

# Method (2)
cXY=cca(X, Y, simple=true)
@test cXY.F[1]'*Cx*cXY.F[1]≈I
@test cXY.F[2]'*Cy*cXY.F[2]≈I
D=cXY.F[1]'*Cxy*cXY.F[2]
@test norm(D-Diagonal(D))+1≈1.

@test cXY==cC

k=10
Xset=[genDataMatrix(n, t) for i=1:k]
Yset=[genDataMatrix(n, t) for i=1:k]

# Method (3)
# canonical correlation analysis of the average covariance and cross-covariance
c=cca(Xset, Yset)

# ... selecting subspace dimension allowing an explained variance = 0.9
c=cca(Xset, Yset; eVar=0.9)

# ... subtracting the mean from the matrices in Xset and Yset
c=cca(Xset, Yset; meanX=nothing, meanY=nothing, eVar=0.9)

# cca on the average of the covariance and cross-covariance matrices
# computed along dims 1
c=cca(Xset, Yset; dims=1, eVar=0.9)

# name of the filter
c.name

using Plots
# plot regularized accumulated eigenvalues
plot(c.arev)

# plot the original covariance and cross-covariance matrices
# and their transformed counterpart
 CxyMax=maximum(abs.(Cxy));
 h1 = heatmap(Cxy, clim=(-CxyMax, CxyMax), title="Cxy", yflip=true, c=:bluesreds);
 D=cC.F[1]'*Cxy*cC.F[2];
 Dmax=maximum(abs.(D));
 h2 = heatmap(D, clim=(0, Dmax), title="F1'CxyF2", yflip=true, c=:amp);
 CxMax=maximum(abs.(Cx));
 h3 = heatmap(Cx, clim=(-CxMax, CxMax), title="Cx", yflip=true, c=:bluesreds);
 h4 = heatmap(cC.F[1]'*Cx*cC.F[1], clim=(0, 1), title="F1'CxF1", yflip=true, c=:amp);
 CyMax=maximum(abs.(Cy));
 h5 = heatmap(Cy, clim=(-CyMax, CyMax), title="Cy", yflip=true, c=:bluesreds);
 h6 = heatmap(cC.F[2]'*Cy*cC.F[2], clim=(0, 1), title="F2'CyF2", yflip=true, c=:amp);
 📈=plot(h3, h5, h1, h4, h6, h2, size=(800,400))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigCCA.png")
