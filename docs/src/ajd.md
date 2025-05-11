# AJD

*Approximate Joint Diagonalization* (AJD) is a diagonalization prodedure
generalizing the eigenvalue-eigenvector decomposition to more then two
matrices. This corresponds to the situation ``m=1`` (one dataset) and ``k>2`` (number of observations). As such, is a very general procedure with a myriad of potential
applications. It was first proposed by Flury and Gautschi (1986) in statistics
and by Cardoso and Souloumiac(1996) in signal processing [🎓](@ref).
Since, it has become a fundamental tool for solving the [blind source
separation](https://en.wikipedia.org/wiki/Signal_separation#EEG)(BSS) problem.

Let ``{C_1,...,C_k}`` be a set of ``n⋅n`` symmetric or Hermitian matrices.
In BSS typically those are covariance matrices, Fourier cross-spectral matrices,
lagged covariance matrices or slices of 4th order cumulants, where
``n`` is the number of variables.

An AJD algorithm seeks a matrix ``F`` diagonalizing all matrices in the
set as much as possible, according to some diagonalization criterion, that is,
we want to achieve

``F^HC_lF≈Λ_l``, for all ``l∈[1...k]``. ``\hspace{1cm}`` [ajd.1]

In some algorithm, such as *OJoB*, ``F`` is constrained to be orthogonal,
in others, like *NoJoB* only to be non-singular.

#### pre-whitening for AJD

Similarly to the two-step procedures encountered in other filters,
e.g., for the [CCA](@ref), for solving the AJD problem often
pre-whitening is applied: first a whitening matrix ``W`` if found such that

``W^H\Big(\frac{1}{k}\sum_{l=1}^kC_k\Big)W_k=I``, ``\hspace{1cm}`` [ajd.2]

then the following transformed AJD problem if solved for ``U``:

``U^H(W^HC_lW)U≈Λ_l``, for all ``l∈[1...k]``.

Finally, ``F`` is obtained as

``F=WU``. ``\hspace{1cm}`` [ajd.3]

Notice that:
- matrix ``W`` may be taken rectangular so as to engender a dimensionality reduction at this stage. This may improve the convergence behavior of AJD algorithms if the matrices ``{C_1,...,C_k}`` are not well-conditioned.  
- if this two-step procedure is employed, the final solution ``F`` is never orthogonal, even if the solving AJD algorithm constrains the solution within the orthogonal group.

#### permutation for AJD

Approximate joint diagonalizers are arbitrary up to a [scale and permutation](@ref). *Diagonalizations.jl* attempts to solve the
permutation ambiguity by reordering
the columns of ``F`` so as to sort in descending order
the diagonal elements of

``\frac{1}{k}\sum_{l=1}^kF^HC_kF``. ``\hspace{1cm}`` [ajd.4]

This sorting mimics the sorting of exact diagonalization procedures such as
the [PCA](@ref), of which the AJD is a generalization, however it is
meaningful only if the input matrices ``{C_1,...,C_k}`` are positive definite.

In analogy with [PCA](@ref), let

``λ=[λ_1...λ_n]``  ``\hspace{1cm}`` [ajd.5]

be the diagonal elements of [ajd.4] and let

``σ_{TOT}=\sum_{i=1}^nλ_i`` be the total variance.

We denote ``\widetilde{F}=[f_1 \ldots f_p]`` the matrix holding the
first ``p<n`` column vectors of ``F``, where ``p`` is the [subspace dimension](@ref). The *explained variance* is given by

``σ_p=\frac{\sum_{i=1}^pλ_i}{σ_{TOT}}`` ``\hspace{1cm}`` [ajd.6]

and the *accumulated regularized eigenvalues* (arev) by

``σ_j=\sum_{i=1}^j{σ_i}``, for ``j=[1 \ldots n]``. ``\hspace{1cm}`` [ajd.7]

For setting the subspace dimension ``p`` manually, set the `eVar`
optional keyword argument of the MCA constructors
either to an integer or to a real number, this latter establishing ``p``
in conjunction with argument `eVarMeth` using the `arev` vector
(see [subspace dimension](@ref)).
By default, `eVar` is set to 0.999.

**Solution**

There is no closed-form solution to the AJD problem in general.
See [Algorithms](@ref).

**Constructors**

Two constructors are available (see here below). The constructed
[LinearFilter](@ref) object holding the AJD will have fields:

`.F`: matrix ``\widetilde{F}`` with columns holding the first
``p`` eigenvectors in ``F``, or just ``F`` if ``p=n``

`.iF`: the left-inverse of `.F`

`.D`: the leading ``p⋅p`` block of ``Λ``, i.e., the elements [ajd.5]
associated to `.F` in diagonal form.

`.eVar`: the explained variance [ajd.6] for the chosen value of ``p``.

`.ev`: the vector ``λ`` [ajd.5].

`.arev`: the accumulated regularized eigenvalues, defined in [ajd.7].

```@docs
ajd
```
