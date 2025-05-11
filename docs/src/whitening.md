# Whitening

*Whitening* (also named *Sphering*) is a standardized version of [PCA](@ref). Possibly it is the most used diagonalization procedure among all. Like PCA, it corresponds to the situation ``m=1`` (one dataset) and ``k=1`` (one observation).

Let ``X`` be a ``n⋅t`` data matrix, where ``n`` is the number of
variables and ``t`` the number of samples and let ``C`` be its ``n⋅n``
covariance matrix. Being ``C`` a positive semi-definite matrix,
its eigenvector matrix ``U`` diagonalizes ``C`` by rotation, as

``U^{H}CU=Λ``. ``\hspace{1cm}`` [whitening.1]

The eigenvalues in the diagonal matrix ``Λ`` are all non-negative.
They are all real and positive if ``C`` is positive definite,
which is assumed in the remaining of this exposition.
The linear transformation ``Λ^{-1/2}U^{H}X`` yields uncorrelated data with
unit variance at all ``n`` components, that is,

``\frac{1}{T}Λ^{-1/2}U^{H}XX^{H}UΛ^{-1/2}=I``. ``\hspace{1cm}`` [whitening.2]

Whitened data remains whitened after whatever further rotation. That is, for
*any* orthogonal matrix ``V``, it holds

``V^H\big(\frac{1}{T}Λ^{-1/2}U^{H}XX^{H}UΛ^{-1/2}\big)V=I``. ``\hspace{1cm}`` [whitening.3]

Hence there exist an infinite number of possible whitening matrices with
general form ``V^HΛ^{-1/2}U^{H}``. Because of this property whitening
plays a fundamental role as a first step in many two-steps diagonalization procedures (e.g., for the [CSP](@ref), [CSTP](@ref) and [CCA](@ref)).
Particularly important among the infinite family of whitening matrices
is the only symmetric one (or Hermitian, if complex),
which is the inverse of the *principal square root* of ``C`` and is found repeatedly in computations on the manifold of positive definite matrices (see
for example
[intro to Riemannian geometry](https://marco-congedo.github.io/PosDefManifold.jl/dev/introToRiemannianGeometry/)).

For setting the subspace dimension ``p`` manually, set the `eVar`
optional keyword argument of the Whitening constructors
either to an integer or to a real number, this latter establishing ``p``
in conjunction with argument `eVarMeth` using the `arev` vector,
which is defined as for the [PCA](@ref) filter, see Eq. [pca.6] therein
and [subspace dimension](@ref) for details.
By default, `eVar` is set to 0.999.

**Solution**

As for PCA, the solution is given by the
[eigenvalue-eigenvector decoposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix)
of ``C``

``\textrm{EVD}(C)=UΛU^{H}``.

**Constructors**

Three constructors are available (see here below), which use exactly
the same syntax as for [PCA](@ref). The constructed
[LinearFilter](@ref) object holding the Whitening will have fields:

`.F`: matrix ``\widetilde{U}\widetilde{Λ}^{-1/2}`` with scaled
orthonormal columns, where ``\widetilde{Λ}`` is
the leading ``p⋅p`` block of ``Λ`` and  ``\widetilde{U}=[u_1 \ldots u_p]`` holds the first ``p`` eigenvectors in ``U``.
If ``p=n``, `.F` is just ``UΛ^{-1/2}``.

`.iF`: ``\widetilde{Λ}^{1/2}\widetilde{U}^H``, the left-inverse of `.F`.

`.D`: ``\widetilde{Λ}``, i.e., the eigenvalues associated to
`.F` in diagonal form.

`.eVar`: the explained variance for the chosen value of ``p``.
This is the same as for the [PCA](@ref), see Eq. [pca.4] therein.

`.ev`: the vector `diag(Λ)` holding all ``n`` eigenvalues.

`.arev`: the *accumulated regularized eigenvalues*.
This is the same as for the [PCA](@ref), see Eq. [pca.6] therein.


```@docs
whitening
```
