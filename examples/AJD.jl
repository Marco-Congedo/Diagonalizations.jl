using Diagonalizations, LinearAlgebra, PosDefManifold, Test


# method (1) real
t, n, k=50, 10, 4
A=randn(n, n) # mixing matrix in model x=As
Xset = [genDataMatrix(t, n) for i = 1:k]
Xfixed=randn(t, n)./1
for i=1:length(Xset) Xset[i]+=Xfixed end
Cset = ℍVector([ℍ((Xset[s]'*Xset[s])/t) for s=1:k])
aC=ajd(Cset; simple=true)

# method (1) complex
t, n, k=50, 10, 4
Ac=randn(ComplexF64, n, n) # mixing matrix in model x=As
Xcset = [genDataMatrix(ComplexF64, t, n) for i = 1:k]
Xcfixed=randn(ComplexF64, t, n)./1
for i=1:length(Xcset) Xcset[i]+=Xcfixed end
Ccset = ℍVector([ℍ((Xcset[s]'*Xcset[s])/t) for s=1:k])
aCc=ajd(Ccset; algorithm=:OJoB, simple=true)


# method (2) real
aX=ajd(Xset; simple=true)
@test aX≈aC

# method (2) complex
aXc=ajd(Xcset; algorithm=:OJoB, simple=true)
@test aXc≈aCc


# create 20 REAL random commuting matrices
# they all have the same eigenvectors
Cset2=PosDefManifold.randP(3, 20; eigvalsSNR=Inf, commuting=true)

# estimate the approximate joint diagonalizer (ajd)
a=ajd(Cset2; algorithm=:OJoB)

# the ajd must be equivalent to the eigenvector matrix of any of the matrices in Cset
@test spForm(a.F'*eigvecs(Cset2[1]))+1. ≈ 1.0

# the same thing using the NoJoB algorithm. Here we just do a sanity check
# as the NoJoB solution is not constrained in the orthogonal group
a=ajd(Cset2; algorithm=:NoJoB)
@test spForm(a.F'*eigvecs(Cset2[1]))<0.01


# create 20 COMPLEX random commuting matrices
# they all have the same eigenvectors
Ccset2=PosDefManifold.randP(ComplexF64, 3, 20; eigvalsSNR=Inf, commuting=true)

# estimate the approximate joint diagonalizer (ajd)
ac=ajd(Ccset2; algorithm=:OJoB)

# the ajd must be equivalent to the eigenvector matrix of any of the matrices in Cset
# just a sanity check as rounding errors appears for complex data
@test spForm(ac.F'*eigvecs(Ccset2[1]))<0.001

# the same thing using the NoJoB algorithm. Here we just do a sanity check
# as the NoJoB solution is not constrained in the orthogonal group
ac=ajd(Ccset2; algorithm=:NoJoB)
@test spForm(ac.F'*eigvecs(Ccset2[1]))<0.01

# REAL data:
# normalize the trace of input matrices,
# give them weights according to the `nonDiagonality` function
# apply pre-whitening and limit the explained variance both
# at the pre-whitening level and at the level of final vector selection
Cset=PosDefManifold.randP(8, 20; eigvalsSNR=10, SNR=2, commuting=false)

a=ajd(Cset; trace1=true, w=nonD, preWhite=true, eVarC=8, eVar=0.99)

using Plots
# plot the original covariance matrices
# and their transformed counterpart
CMax=maximum(maximum(abs.(C)) for C ∈ Cset);
 h1 = heatmap(Cset[1], clim=(-CMax, CMax), title="C1", yflip=true, c=:bluesreds);
 h2 = heatmap(Cset[2], clim=(-CMax, CMax), title="C2", yflip=true, c=:bluesreds);
 h3 = heatmap(Cset[3], clim=(-CMax, CMax), title="C3", yflip=true, c=:bluesreds);
 h4 = heatmap(Cset[4], clim=(-CMax, CMax), title="C4", yflip=true, c=:bluesreds);
 📈=plot(h1, h2, h3, h4, size=(700,400))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigAJD1.png")

Dset=[a.F'*C*a.F for C ∈ Cset];
 DMax=maximum(maximum(abs.(D)) for D ∈ Dset);
 h5 = heatmap(Dset[1], clim=(-DMax, DMax), title="F'*C1*F", yflip=true, c=:bluesreds);
 h6 = heatmap(Dset[2], clim=(-DMax, DMax), title="F'*C2*F", yflip=true, c=:bluesreds);
 h7 = heatmap(Dset[3], clim=(-DMax, DMax), title="F'*C3*F", yflip=true, c=:bluesreds);
 h8 = heatmap(Dset[4], clim=(-DMax, DMax), title="F'*C4*F", yflip=true, c=:bluesreds);
 📉=plot(h5, h6, h7, h8, size=(700,400))
# savefig(📉, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigAJD2.png")


# COMPLEX data:
# normalize the trace of input matrices,
# give them weights according to the `nonDiagonality` function
# apply pre-whitening and limit the explained variance both
# at the pre-whitening level and at the level of final vector selection
Ccset=PosDefManifold.randP(3, 20; eigvalsSNR=10, SNR=2, commuting=false)

# run OJoB
ac=ajd(Ccset; trace1=true, w=nonD, preWhite=true,
       algorithm=:OJoB, eVarC=8, eVar=0.99)

# run NoJoB
ac=ajd(Ccset; eVarC=8, eVar=0.99)
