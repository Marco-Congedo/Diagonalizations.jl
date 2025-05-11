# PCA

*Principal Component Analysis* (PCA) is obtained by
*eigenvalue-eigenvector decomposition*.
It was first conceived by Karl Pearson
(1901)[🎓](@ref) as a way to fit straight lines to a multidimensional cloud of points.
It corresponds to the situation ``m=1`` (one dataset) and ``k=1`` (one observation).

Let ``X`` be a ``n⋅t`` data matrix, where ``n`` is the number of
variables and ``t`` the number of samples and let ``C`` be its ``n⋅n``
covariance matrix. Being ``C`` a positive semi-definite matrix,
its eigenvector matrix ``U`` diagonalizes ``C`` by rotation, as

``U^{H}CU=Λ``. ``\hspace{1cm}`` [pca.1]

The eigenvalues in the diagonal matrix ``Λ`` are all non-negative.
They are all real and positive if ``C`` is positive definite,
which is assumed in the remaining of this exposition.
The linear transformation ``U^{H}X`` yields uncorrelated data with
variance of the ``n^{th}`` component equal to the corresponding eigenvalue
``λ_n``, that is,

``\frac{1}{T}U^{H}XX^{H}U=Λ``. ``\hspace{1cm}`` [pca.2]

In *Diagonalizations.jl*
the diagonal elements of diagonalized matrices are always arranged
by descending order, such as

``λ_1≥\ldots≥λ_n``. ``\hspace{1cm}`` [pca.3]

Then, because of the extremal properties of
eigenvalues (Congedo, 2013, p. 66; Schott, 1997, p. 104-128)[🎓](@ref),
the first component (row) of ``U^{H}X``
holds the linear combination of ``X`` with maximal variance,
the second the linear combination with maximal residual variance and so on,
subject to constraint ``U^{H}U=UU^{H}=I``.

Let ``σ_{TOT}=\sum_{i=1}^nλ_i=tr(C)`` be the total variance and let
``\widetilde{U}=[u_1 \ldots u_p]`` be the matrix holding the first ``p<n``
eigenvectors, where ``p`` is the [subspace dimension](@ref), then

``σ_p=\frac{\sum_{i=1}^pλ_i}{σ_{TOT}}=\frac{tr(\widetilde{U}^HC\widetilde{U})}{tr(C)}``  ``\hspace{1cm}`` [pca.4]

is named the *explained variance* and

``ε_p=σ_{TOT}-σ_p`` ``\hspace{1cm}`` [pca.5]

is named the *representation error*. These quantities are expressed in
proportions, that is, it holds ``σ_p+ε_p=1``.


The *accumulated regularized eigenvalues* (arev) are defined as

``σ_j=\sum_{i=1}^j{σ_i}``, for ``j=[1 \ldots n]``, ``\hspace{1cm}`` [pca.6]

where ``σ_i`` is given by Eq. [pca.4].

For setting the subspace dimension ``p`` manually, set the `eVar`
optional keyword argument of the PCA constructors
either to an integer or to a real number, this latter establishing ``p``
in conjunction with argument `eVarMeth` using the `arev` vector
(see [subspace dimension](@ref)).
By default, `eVar` is set to 0.999.

**Solution**

The PCA solution is given by the
[eigenvalue-eigenvector decoposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix)
of ``C``

``\textrm{EVD}(C)=UΛU^{H}``.

It is worth mentioning that

``\widetilde{U}\widetilde{Λ}\widetilde{U}^H``,

where ``\widetilde{Λ}`` is the leading ``p⋅p`` block of ``Λ``, is the
best approximant to ``C`` with rank ``p``
in the least-squares sense (Good, 1969)[🎓](@ref).


**Constructors**

Three constructors are available (see here below). The constructed
[LinearFilter](@ref) object holding the PCA will have fields:

`.F`: matrix ``\widetilde{U}`` with orthonormal columns holding the first
``p`` eigenvectors in ``U``, or just ``U`` if ``p=n``

`.iF`: the (conjugate) transpose of `.F`

`.D`: the leading ``p⋅p`` block of ``Λ``, i.e., the eigenvalues associated to `.F` in diagonal form.

`.eVar`: the explained variance [pca.4] for the chosen value of ``p``.

`.ev`: the vector `diag(Λ)` holding all ``n`` eigenvalues.

`.arev`: the accumulated regularized eigenvalues in [pca.6].

```@docs
pca
```
