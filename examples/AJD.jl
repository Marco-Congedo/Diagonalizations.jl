using Diagonalizations, LinearAlgebra, PosDefManifold, Test

# method (1) real
t, n, k=50, 10, 4
A=randn(n, n) # mixing matrix in model x=As
Xset = [genDataMatrix(t, n) for i = 1:k]
Xfixed=randn(t, n)./1
for i=1:length(Xset) Xset[i]+=Xfixed end
Cset = ℍVector([ℍ((Xset[s]'*Xset[s])/t) for s=1:k])
aC=ajd(Cset; algorithm=:OJoB, simple=true)
aC2=ajd(Cset; algorithm=:NoJoB, simple=true)
aC3=ajd(Cset; algorithm=:LogLike, simple=true)
aC4=ajd(Cset; algorithm=:LogLikeR, simple=true)

# method (1) complex
t, n, k=50, 10, 4
Ac=randn(ComplexF64, n, n) # mixing matrix in model x=As
Xcset = [genDataMatrix(ComplexF64, t, n) for i = 1:k]
Xcfixed=randn(ComplexF64, t, n)./1
for i=1:length(Xcset) Xcset[i]+=Xcfixed end
Ccset = ℍVector([ℍ((Xcset[s]'*Xcset[s])/t) for s=1:k])
aCc=ajd(Ccset; algorithm=:OJoB, simple=true)
aCc2=ajd(Ccset; algorithm=:NoJoB, simple=true)
aCc3=ajd(Ccset; algorithm=:LogLike, simple=true)


# method (2) real
aX=ajd(Xset; algorithm=:OJoB, simple=true)
aX2=ajd(Xset; algorithm=:NoJoB, simple=true)
aX3=ajd(Xset; algorithm=:LogLike, simple=true)
aX4=ajd(Xset; algorithm=:LogLikeR, simple=true)
@test aX≈aC
@test aX2≈aC2
@test aX3≈aC3
@test aX4≈aC4

# method (2) complex
aXc=ajd(Xcset; algorithm=:OJoB, simple=true)
aXc2=ajd(Xcset; algorithm=:NoJoB, simple=true)
aXc3=ajd(Xcset; algorithm=:LogLike, simple=true)
@test aXc≈aCc
@test aXc2≈aCc2
@test aXc3≈aCc3

# create 20 REAL random commuting matrices
# they all have the same eigenvectors
Cset2=PosDefManifold.randP(3, 20; eigvalsSNR=Inf, commuting=true)
# estimate the approximate joint diagonalizer (AJD)
a=ajd(Cset2; algorithm=:OJoB)
# the orthogonal AJD must be equivalent to the eigenvector matrix
# of any of the matrices in Cset
@test [spForm(a.F'*eigvecs(C))+1. for C ∈ Cset2] ≈
      ones(eltype(Cset2[1]), length(Cset2))

# generate positive definite matrices with model A*D_κ*D, where
# A is the mixing matrix and D_κ, for all κ=1:k, are diagonal matrices.
# The estimated AJD matrix must be the inverse of A
n, k=3, 10
Dest=PosDefManifold.randΛ(eigvalsSNR=100, n, k)
A=randn(n, n) # mixing matrix
Cset3=Vector{Hermitian}([Hermitian(A*D*A') for D ∈ Dest])
a=ajd(Cset3; algorithm=:NoJoB, eVarC=n)
@test spForm(a.F'*A)<0.001
a=ajd(Cset3; algorithm=:LogLike, eVarC=n)
@test spForm(a.F'*A)<0.001
a=ajd(Cset3; algorithm=:LogLikeR, eVarC=n)
@test spForm(a.F'*A)<0.001

# repeat the test adding noise; now the model is no more exactly identifiable
for k=1:length(Cset3) Cset3[k]+=randP(n)/1000 end
a=ajd(Cset3; algorithm=:NoJoB, eVarC=n)
@test spForm(a.F'*A)<0.1
a=ajd(Cset3; algorithm=:LogLike, eVarC=n)
@test spForm(a.F'*A)<0.1
a=ajd(Cset3; algorithm=:LogLikeR, eVarC=n)
@test spForm(a.F'*A)<0.1

# create 20 COMPLEX random commuting matrices
# they all have the same eigenvectors
Ccset2=PosDefManifold.randP(ComplexF64, 3, 20; eigvalsSNR=Inf, commuting=true)
# estimate the approximate joint diagonalizer (AJD)
ac=ajd(Ccset2; algorithm=:OJoB)
# he AJD must be equivalent to the eigenvector matrix of any of the matrices in Cset
# just a sanity check as rounding errors appears for complex data
@test norm([spForm(ac.F'*eigvecs(C)) for C ∈ Ccset2])/3<0.01

# the same thing using the NoJoB and LogLike algorithms. Require less precision
# as the NoJoB solution is not constrained in the orthogonal group
ac=ajd(Ccset2; algorithm=:NoJoB)
@test norm([spForm(ac.F'*eigvecs(C)) for C ∈ Ccset2])/3<0.01
ac=ajd(Ccset2; algorithm=:LogLike)
@test norm([spForm(ac.F'*eigvecs(C)) for C ∈ Ccset2])/3<0.01

# REAL data:
# normalize the trace of input matrices,
# give them weights according to the `nonDiagonality` function
# apply pre-whitening and limit the explained variance both
# at the pre-whitening level and at the level of final vector selection
Cset=PosDefManifold.randP(8, 20; eigvalsSNR=10, SNR=2, commuting=false)

a=ajd(Cset; trace1=true, w=nonD, preWhite=true, eVarC=4, eVar=0.99)

a=ajd(Cset; algorithm=:LogLike, w=nonD, preWhite=true, eVarC=4, eVar=0.99)

a=ajd(Cset; algorithm=:LogLikeR, w=nonD, preWhite=true, eVarC=4, eVar=0.99)


# AJD for plots below
a=ajd(Cset; algorithm=:LogLike, w=nonD, preWhite=true)

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

# run logLike
ac=ajd(Ccset; algorithm=:LogLike, eVarC=8, eVar=0.99)
