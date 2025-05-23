# Diagonalizations.jl

| **Documentation**  |
|:---------------------------------------:|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://Marco-Congedo.github.io/Diagonalizations.jl/stable) |

**Diagonalizations.jl** is a [**Julia**](https://julialang.org/) signal processing package implementing several *closed form* and *iterative* diagonalization procedures for both *real* and *complex* data input:

| Acronym   | Full Name | Datasets ( *m* ) | Observations ( *k* ) |
|:----------|:---------:|:---------:|:---------:|
| PCA | Principal Component Analysis | 1 | 1 |
| Whitening | Whitening (Sphering) | 1 | 1 |
| MCA | Maximum Covariance Analysis | 2 | 1 |
| CCA | Canonical Correlation Analysis | 2 | 1 |
| gMCA | generalized MCA | >1 | 1 |
| gCCA | generalized CCA | >1 | 1 |
| CSP | Common Spatial Pattern | 1 | 2 |
| CSTP | Common Spatio-Temporal Pattern | 1 | >1 |
| AJD | Approximate Joint Diagonalization | 1 | >1 |
| mAJD | multiple AJD | >1 | >1 |

For example the MCA diagonalizes a cross-covariance matrix, like in this figure:

![](/docs/src/assets/FigMCA.png)

As compared to [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl)
this package supports :
- the `dims` keyword like in the [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl) package
- shrinkage covariance matrix estimations throught package [CovarianceEstimation](https://github.com/mateuszbaran/CovarianceEstimation.jl)
- average covariance estimations using metrics for the manifold of positive definite matrices using the [PosDefManifold](https://github.com/Marco-Congedo/PosDefManifold.jl) package
- facilities to set the subspace dimension upon construction
- diagonalization procedures for the case *m≥2* and *k≥2*.

This package implements state-of-the-art **approximate joint diagonalization** algorithms. For some benchmarking see
[here](https://github.com/Marco-Congedo/STUDIES/tree/master/AJD-Algos-Benchmark).


## Installation

To install the package execute the following command in Julia's REPL:

    ]add CovarianceEstimation PosDefManifold Diagonalizations

## Examples

```

using Diagonalizations, PosDefManifold, Test

n, t=10, 100

# generate an nxt data matrix
X=genDataMatrix(n, t)

# principal component analysis
pX=pca(X)

# the following is an equivalent constructor taking the covariance matrix as input
pC=pca(Symmetric((X*X')/t))

@test pX==pC # the output of the two constructors above is equivalent

@test C≈pC.F*pC.D*pC.F'  

# get only the first p eigenvectors, where p is the smallest integer
# explaining at least 75% of the variance
pX=pca(X; eVar=0.75)

Y=genDataMatrix(n, t)

# maximum covariance analysis
mXY=mca(X, Y)

# canonical correlation analysis
cXY=cca(X, Y)

# approximate joint diagonalization
Xset=randP(5, 20)
aXset=ajd(Xset; algorithm=:JADE)
aXset=ajd(Xset; algorithm=:LogLike)

# etc., etc.

```

## About the Authors

[Marco Congedo](https://sites.google.com/site/marcocongedo), is a Research Director of [CNRS](http://www.cnrs.fr/en) (Centre National de la Recherche Scientifique), working at [UGA](https://www.univ-grenoble-alpes.fr/english/) (University of Grenoble Alpes). **contact**: marco *dot* congedo *at* gmail *dot* com

| **Documentation**  | 
|:---------------------------------------:|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://Marco-Congedo.github.io/Diagonalizations.jl/stable) |
