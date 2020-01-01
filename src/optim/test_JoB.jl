using Statistics, LinearAlgebra, BenchmarkTools, Plots, PosDefManifold


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~  UTILITIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# Get all products 𝐔[i]' * 𝒞[l, i, j] * 𝐔[j]
function _rotate_crossCov(𝐔, 𝒞, m, k)
    𝒮=Array{Matrix}(undef, k, m, m)
    if m==1
        @inbounds for l=1:k 𝒮[l, 1, 1]=𝐔'*𝒞[l, 1, 1]*𝐔 end
    else
        @inbounds for l=1:k, i=1:m, j=1:m 𝒮[l, i, j]=𝐔[i]'*𝒞[l, i, j]*𝐔[j] end
    end
    return 𝒮
end


# Put all cross-covariances in a single matrix of dimension m*n x m*n for visualization
function 𝒞2Mat(𝒞::AbstractArray, m, k)
    n=size(𝒞[1, 1, 1], 1)
    if m==1
        C=Matrix{Float64}(undef, n, k*n)
        for i=1:k, x=1:n, y=1:n C[x, i*n-n+y]=𝒞[i, 1, 1][x, y] end
    else
        C=Matrix{Float64}(undef, m*n, m*n)
        for i=1:m, j=1:m, x=1:n, y=1:n C[i*n-n+x, j*n-n+y]=𝒞[k, i, j][x, y] end
    end
    return C
end


####  Create data for testing the case k=1, m>1
# `t` is the number of samples, e.g.,
    # in Tangent space transfer learning is the number of trials,
# `m` is the number of datasets, e.g., subjects
# `n` is the number of variables, e.g.,
    # in Tangent space transfer learning is the number of elements of the tangent vectors
# `noise` must be smaller than 1.0. The smaller the noise, the more data are correlated
function getData(t, m, n, noise)
    # create m identical data matrices and rotate them by different
    # random orthogonal matrices V_1,...,V_m
    𝐕=𝕄Vector([randU(n) for i=1:m]) # random orthogonal matrices
    X=randn(n, t)  # data common to all subjects
    # each subject has this common part plus a random part
    𝐗=𝕄Vector([𝐕[i]'*((1-noise)*X + noise*randn(n, t)) for i=1:m])
    return 𝐗
end
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #



###############################################################################
#
# test: case m=1, k>1. (AJD)
#
###############################################################################
t, m, k, n=10, 1, 10, 20
𝐗 = 𝕄Vector([genDataMatrix(n, t) for i = 1:k])
Xfixed=randn(n, t)./0.1
for i=1:k 𝐗[i]+=Xfixed end

U, V, Λ, iter, conv=OJoB(𝐗, 1, k, :d, eltype(𝐗[1]);
                         dims    = 2,
                         sort    = false,
                         tol     = 1e-6,
                         verbose = true)

U, V, Λ, iter, conv=OJoB(𝐗, 1, k, :d, eltype(𝐗[1]);
                         dims    = 1,
                         sort    = true,
                         tol     = 1e-6,
                         verbose = true)




# Visualize cross-covariances BEFORE GMCA
𝒞 = _crossCov(𝐗, m, k; dims=1)
heatmap(𝒞2Mat(𝒞, m, k), yflip=true, c=:bluesreds)
# the diagonalization is approximated. Here is the diagonaliation error:
error=𝛍(nonD(U'*𝒞[i, m, m]*U) for i=1:k)

# Visualize cross-covariances AFTER GMCA, but BEFORE scaling and permutation
𝒮=_rotate_crossCov(U, 𝒞, m, k)
heatmap(𝒞2Mat(𝒮, m, k), yflip=true, c=:bluesreds)

# Visualize cross-covariances AFTER GMCA, scaling and permutation
eigvalues=_permute!(U, 𝐗, k, :d; dims=1)
𝒮=_rotate_crossCov(U, 𝒞, m, k)
heatmap(𝒞2Mat(𝒮, m, k), yflip=true, c=:amp)


###############################################################################
#
# test: case m>1, k=1.
#
###############################################################################
# t trials, m subjects, tang. vec. of dimension n
t, m, k, n, noise = 200, 5, 1, 6, 0.01

𝐗 = getData(t, m, n, noise)

# run GMCA
𝐔, 𝐕, Λ, iter, conv = OJoB(𝐗, m, k, :d, eltype(𝐗[1]);
                        dims=2,
                        verbose=true,
                        sort=false,
                        tol=1e-6)

# the diagonalization is approximated. Here is the diagonaliation error:
error=𝚺(nonD(𝐔[i]'*((𝐗[i]*𝐗[j]')/t)*𝐔[j]) for i=1:m, j=1:m if i≠j)/(m^2-m)

######  Visualize the result  ######

# Visualize cross-covariances BEFORE GMCA
𝒞 = _crossCov(𝐗, m, k; dims=2)
heatmap(𝒞2Mat(𝒞, m, k), yflip=true, c=:bluesreds)

# Visualize cross-covariances AFTER GMCA, but BEFORE scaling and permutation
𝒮=_rotate_crossCov(𝐔, 𝒞, m, 1)
heatmap(𝒞2Mat(𝒮, m, k), yflip=true, c=:bluesreds)

# Visualize cross-covariances AFTER GMCA, scaling and permutation
eigvalues=_scaleAndPermute!(𝐔, 𝐗, m, k, :d; dims=2)

𝒮=_rotate_crossCov(𝐔, 𝒞, m, 1)
heatmap(𝒞2Mat(𝒮, m, 1), yflip=true, c=:amp)


# compare with the same computations given covariance matrices
𝐔2, 𝐕2, Λ, iter, conv = OJoB(𝒞, m, k, :c, eltype(𝒞[1][1, 1]);
                          dims=2,
                          verbose=true,
                          sort=false,
                          tol=1e-6)

# the diagonalization is approximated. Here is the diagonaliation error:
#𝒞 = _crossCov(𝐗, m, k, t)

error2=𝚺(nonD(𝐔2[i]'*𝒞[1, i, j]*𝐔2[j]) for i=1:m, j=1:m if i≠j)/(m^2-m)
error≈error2 ? println(" ⭐ ") : println(" ⛔ ")

# Visualize cross-covariances AFTER GMCA, but BEFORE scaling and permutation
𝒮=_rotate_crossCov(𝐔2, 𝒞, m, 1)
heatmap(𝒞2Mat(𝒮, m, k), yflip=true, c=:bluesreds)

# Visualize cross-covariances AFTER GMCA, scaling and permutation
eigvalues=_scaleAndPermute!(𝐔2, 𝒞, m, k, :c)
𝒮=_rotate_crossCov(𝐔2, 𝒞, m, 1)
heatmap(𝒞2Mat(𝒮, m, 1), yflip=true, c=:amp)




# get a suitable dimension for projecting data via GMCA
# (the first `d` column vectors of matrices 𝐔[1],...,𝐔[m] should be retained)
# d=findDimension(eigvalues, 0.01)

#################################################################
# test 2: check that for m=2, k=1 the result is the same as the MCA
#################################################################

# Get data with m=2 (2 subjects)
t, m, n, k=200, 2, 4, 1
𝐗 = getData(t, m, n, 0.8)

# run GMCA
𝐔, 𝐕, Λ, iter, conv = OJoB(𝐗, m, k, :d, eltype(𝐗[1]);
                        dims=2,
                        verbose=true,
                        sort=false,
                        tol=1e-6)

eigvalues=_scaleAndPermute!(𝐔, 𝐗, m, k, :d; dims=2)
#d=findDimension(_scaleAndPermute!(𝐔, 𝐗, m, k, n, t), 0.01)

# do MCA
XY=(𝐗[1]*𝐗[2]')/t
M=mca(XY; simple=true)

crossCov1 = M.F[1]'*XY*M.F[2]
crossCov2 = 𝐔[1]'*XY*𝐔[2]
crossCov1 ≈ crossCov2 ? println(" ⭐ ") : println(" ⛔ ")

# the following must be the identity matrix out of a possible sign ambiguity
checkU = M.F[1]'*𝐔[1]
abs.(checkU) ≈ I ? println(" ⭐ ") : println(" ⛔ ")
heatmap(checkU, yflip=true, c=:bluesreds)

checkV = M.F[2]'*𝐔[2]
abs.(checkV) ≈ I ? println(" ⭐ ") : println(" ⛔ ")
heatmap(checkV, yflip=true, c=:bluesreds)

###############################################################################
#
# test: case m>1, k>1
#
###############################################################################
t, m, n, k, noise = 200, 5, 6, 20, 0.01
𝐗=𝕄Vector₂(undef, k)
for s=1:k 𝐗[s] = getData(t, m, n, noise) end

𝒞 = _crossCov(𝐗, m, k; dims=2)

𝐔, 𝐕, Λ, iter, conv = OJoB(𝒞, m, k, :c, eltype(𝐗[1][1]);
                        dims=2, verbose=true, tol=1e-4)
error=𝚺(nonD(𝐔[i]'*𝒞[l, i, j]*𝐔[j]) for l=1:k, i=1:m, j=1:m if i≠j)/((m^2-m)*k)

𝐔, 𝐕, Λ, iter, conv = OJoB(𝒞, m, k, :c, eltype(𝐗[1][1]);
                        dims=2, verbose=true, tol=1e-4, sort=false)

𝐔, 𝐕, Λ, iter, conv = OJoB(𝐗, m, k, :d, eltype(𝐗[1][1]);
                        dims=2, verbose=true, tol=1e-4)
error=𝚺(nonD(𝐔[i]'*𝒞[l, i, j]*𝐔[j]) for l=1:k, i=1:m, j=1:m if i≠j)/((m^2-m)*k)

𝐔, 𝐕, Λ, iter, conv = OJoB(𝐗, m, k, :d, eltype(𝐗[1][1]);
                        dims=2, verbose=true, tol=1e-4, fullModel=true)
error=𝚺(nonD(𝐔[i]'*𝒞[l, i, j]*𝐔[j]) for l=1:k, i=1:m, j=1:m)/((m^2)*k)

𝐔, 𝐕, Λ, iter, conv = OJoB(𝐗, m, k, :d, eltype(𝐗[1][1]);
                        dims=2, verbose=true, tol=1e-4, preWhite=true)
error=𝚺(nonD(𝐔[i]'*𝒞[l, i, j]*𝐔[j]) for l=1:k, i=1:m, j=1:m if i≠j)/((m^2-m)*k)

𝐔, 𝐕, Λ, iter, conv = OJoB(𝐗, m, k, :d, eltype(𝐗[1][1]);
                        dims=2, verbose=true, tol=1e-4, fullModel=true, preWhite=true)
error=𝚺(nonD(𝐔[i]'*𝒞[l, i, j]*𝐔[j]) for l=1:k, i=1:m, j=1:m)/((m^2)*k)

eigvalues=_scaleAndPermute!(𝐔, 𝐗, m, k, :d; dims=2)
