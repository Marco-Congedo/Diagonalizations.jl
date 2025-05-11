using Diagonalizations, LinearAlgebra, PosDefManifold, Test

# Method (1) real
t, n, k=10, 20, 10
Xset = [genDataMatrix(t, n) for i = 1:k]
Xfixed=randn(t, n)./1
for i=1:length(Xset) Xset[i]+=Xfixed end
C1=Hermitian( mean((X'*X)/t for X∈Xset) )
C2=Hermitian( mean((X*X')/n for X∈Xset) )
Xbar=mean(Xset)
c=cstp(Xbar, C1, C2; simple=true)
@test c.F[1]'*C2*c.F[1]≈I
@test c.F[2]'*C1*c.F[2]≈I
Z=c.F[1]'*Xbar*c.F[2]
n=minimum(size(Z))
@test norm(Z[1:n, 1:n]-Diagonal(Z[1:n, 1:n]))+1. ≈ 1.
cX=cstp(Xset; simple=true)
@test c==cX

# Method (1) complex
t, n, k=10, 20, 10
Xcset = [genDataMatrix(ComplexF64, t, n) for i = 1:k]
Xcfixed=randn(ComplexF64, t, n)./1
for i=1:length(Xcset) Xcset[i]+=Xcfixed end
C1c=Hermitian( mean((Xc'*Xc)/t for Xc∈Xcset) )
C2c=Hermitian( mean((Xc*Xc')/n for Xc∈Xcset) )
Xcbar=mean(Xcset)
cc=cstp(Xcbar, C1c, C2c; simple=true)
@test cc.F[1]'*C2c*cc.F[1]≈I
@test cc.F[2]'*C1c*cc.F[2]≈I
Zc=cc.F[1]'*Xcbar*cc.F[2]
n=minimum(size(Zc))
@test norm(Zc[1:n, 1:n]-Diagonal(Zc[1:n, 1:n]))+1. ≈ 1.
cXc=cstp(Xcset; simple=true)
@test cc==cXc

# Method (2) real
c=cstp(Xset)

# ... selecting subspace dimension allowing an explained variance = 0.9
c=cstp(Xset; eVar=0.9)

# ... giving weights `w` to the covariance matrices
c=cstp(Xset; w=abs2.(randn(k)), eVar=0.9)

# ... subtracting the means
c=cstp(Xset; meanXd₁=nothing, meanXd₂=nothing, w=abs2.(randn(k)), eVar=0.9)

# explained variance
c.eVar

# name of the filter
c.name

using Plots
# plot the original covariance matrices and the transformed counterpart
c=cstp(Xset)

C1Max=maximum(abs.(C1));
 h1 = heatmap(C1, clim=(-C1Max, C1Max), title="C1", yflip=true, c=:bluesreds);
 D1=c.F[1]'*C2*c.F[1];
 D1Max=maximum(abs.(D1));
 h2 = heatmap(D1, clim=(0, D1Max), title="F[1]'*C2*F[1]", yflip=true, c=:amp);
 C2Max=maximum(abs.(C2));
 h3 = heatmap(C2, clim=(-C2Max, C2Max), title="C2", yflip=true, c=:bluesreds);
 D2=c.F[2]'*C1*c.F[2];
 D2Max=maximum(abs.(D2));
 h4 = heatmap(D2, clim=(0, D2Max), title="F[2]'*C1*F[2]", yflip=true, c=:amp);

XbarMax=maximum(abs.(Xbar));
 h5 = heatmap(Xbar, clim=(-XbarMax, XbarMax), title="Xbar", yflip=true, c=:bluesreds);
 DX=c.F[1]'*Xbar*c.F[2];
 DXMax=maximum(abs.(DX));
 h6 = heatmap(DX, clim=(0, DXMax), title="F[1]'*Xbar*F[2]", yflip=true, c=:amp);
 📈=plot(h1, h3, h5, h2, h4, h6, size=(800,400))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigCSTP.png")

# Method (2) complex
cc=cstp(Xcset)

# ... selecting subspace dimension allowing an explained variance = 0.9
cc=cstp(Xcset; eVar=0.9)

# ... giving weights `w` to the covariance matrices
cc=cstp(Xcset; w=abs2.(randn(k)), eVar=0.9)

# ... subtracting the mean
cc=cstp(Xcset; meanXd₁=nothing, meanXd₂=nothing,
        w=abs2.(randn(k)), eVar=0.9)

# explained variance
c.eVar

# name of the filter
c.name
