using Diagonalizations, LinearAlgebra, PosDefManifold, Test

# Method (1)
t, n=50, 10
X1=genDataMatrix(n, t)
X2=genDataMatrix(n, t)
Cx1=Symmetric((X1*X1')/t)
Cx2=Symmetric((X2*X2')/t)
C=Cx1+Cx2
cC=csp(Cx1, Cx2; simple=true)
Dx1=cC.F'*Cx1*cC.F
@test norm(Dx1-Diagonal(Dx1))+1≈1.
Dx2=cC.F'*Cx2*cC.F
@test norm(Dx2-Diagonal(Dx2))+1≈1.
@test cC.F'*C*cC.F≈I
@test norm(Dx1-(I-Dx2))+1≈1.


# Method (2)
c12=csp(X1, X2, simple=true)
Dx1=c12.F'*Cx1*c12.F
@test norm(Dx1-Diagonal(Dx1))+1≈1.
Dx2=c12.F'*Cx2*c12.F
@test norm(Dx2-Diagonal(Dx2))+1≈1.
@test c12.F'*C*c12.F≈I
@test norm(Dx1-(I-Dx2))+1≈1.

@test cC==c12


k=10
Xset=[genDataMatrix(n, t) for i=1:k]
Yset=[genDataMatrix(n, t) for i=1:k]

# Method (3)
# CSP of the average covariance matrices
c=csp(Xset, Yset)

# ... selecting subspace dimension allowing an explained variance = 0.9
c=csp(Xset, Yset; eVar=0.9)

# ... subtracting the mean from the matrices in Xset and Yset
c=csp(Xset, Yset; meanX₁=nothing, meanX₂=nothing, eVar=0.9)

# csp on the average of the covariance and cross-covariance matrices
# computed along dims 1
c=csp(Xset, Yset; dims=1, eVar=0.9)

# name of the filter
c.name

using Plots
# plot regularized accumulated eigenvalues
plot(c.arev)


# plot the original covariance matrices and the transformed counterpart
# example when argument `selMeth` is `extremal` (default): 2-class separation
 cC=csp(Cx1, Cx2)
 Cx1Max=maximum(abs.(Cx1));
 h1 = heatmap(Cx1, clim=(-Cx1Max, Cx1Max), title="Cx1", yflip=true, c=:bluesreds);
 h2 = heatmap(cC.F'*Cx1*cC.F, clim=(0, 1), title="F'*Cx1*F", yflip=true, c=:amp);
 Cx2Max=maximum(abs.(Cx2));
 h3 = heatmap(Cx2, clim=(-Cx2Max, Cx2Max), title="Cx2", yflip=true, c=:bluesreds);
 h4 = heatmap(cC.F'*Cx2*cC.F, clim=(0, 1), title="F'*Cx2*F", yflip=true, c=:amp);
 CMax=maximum(abs.(C));
 h5 = heatmap(C, clim=(-CMax, CMax), title="Cx1+Cx2", yflip=true, c=:bluesreds);
 h6 = heatmap(cC.F'*C*cC.F, clim=(0, 1), title="F'*(Cx1+Cx2)*F", yflip=true, c=:amp);
 📈=plot(h1, h3, h5, h2, h4, h6, size=(800,400))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigCSP1.png")

# example when argument `selMeth` is different from `extremal`: enhance snr
 cC=csp(Cx1, Cx2; selMeth=:enhaceSNR)
 Cx1Max=maximum(abs.(Cx1));
 h1 = heatmap(Cx1, clim=(-Cx1Max, Cx1Max), title="Cx1", yflip=true, c=:bluesreds);
 h2 = heatmap(cC.F'*Cx1*cC.F, clim=(0, 1), title="F'*Cx1*F", yflip=true, c=:amp);
 Cx2Max=maximum(abs.(Cx2));
 h3 = heatmap(Cx2, clim=(-Cx2Max, Cx2Max), title="Cx2", yflip=true, c=:bluesreds);
 h4 = heatmap(cC.F'*Cx2*cC.F, clim=(0, 1), title="F'*Cx2*F", yflip=true, c=:amp);
 CMax=maximum(abs.(C));
 h5 = heatmap(C, clim=(-CMax, CMax), title="Cx1+Cx2", yflip=true, c=:bluesreds);
 h6 = heatmap(cC.F'*C*cC.F, clim=(0, 1), title="F'*(Cx1+Cx2)*F", yflip=true, c=:amp);
 📉=plot(h1, h3, h5, h2, h4, h6, size=(800,400))
# savefig(📉, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigCSP2.png")
