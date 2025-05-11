# MCA

*Maximum Covariance Analysis* is obtained by
*singular-value decomposition*. It can be conceived as the multivariate extension of covariance and as the bilinear version of
[PCA](@ref); if PCA diagonalizes the covariance matrix of
a data set, MCA diagonalized the *cross-covariance matrix* of
two data sets. It corresponds to the situation ``m=2`` (two datasets) and ``k=1`` (one observation).

Let ``X`` and ``Y`` be two ``n_x⋅t`` and ``n_y⋅t`` data matrices, where ``n_x`` and ``n_y`` are the number of variables in ``X`` and ``Y``, respectively and ``t`` the number of samples. We assume
that the samples in ``X`` and ``Y`` are synchronized. Let ``C_{xy}=\frac{1}{t}X^HY`` be the ``n_x⋅n_y`` [cross-covariance matrix](https://en.wikipedia.org/wiki/Cross-covariance). MCA seeks two orthogonal matrices ``U`` and ``V``
such that

``U_x^{H}C_{xy}U_y=Λ``, ``\hspace{1cm}`` [mca.1]

where ``Λ`` is a ``n⋅n`` diagonal matrix, with ``n=min(n_x, n_y)``.
The first components (rows) of ``U_x^{H}X`` and ``U_y^{H}Y``
hold the linear combination of ``X`` and ``Y`` with maximal
covariance, the second the linear combination with maximal residual covariance and so on,
subject to constraint ``U_x^{H}U_x=I`` and ``U_y^{H}U_x=I``.
If ``n_x=n_y``, ``U_x`` and ``U_y`` are both square, hence
it holds also ``U_xU_x^{H}=I`` and ``U_yU_y^{H}=I``, otherwise this holds only for one of them.

It should be kept in mind that MCA is sensitive to the amplitude of
each process, since the covariance is maximized and not the correlation as in [CCA](@ref).
Threfore, if the amplitude of the two processes is not homegeneous,
the covariance will be driven by the process with highest amplitude.

The *accumulated regularized eigenvalues* (arev) for the MCA are defined as

``σ_j=\sum_{i=1}^j{σ_i}``, for ``j=[1 \ldots n]``, ``\hspace{1cm}`` [mca.2]

where ``σ_i`` is given by

``σ_p=\frac{\sum_{i=1}^pλ_i}{σ_{TOT}}``  ``\hspace{1cm}`` [mca.3]

and ``λ_i`` are the singular values in ``Λ`` of [mca.1].

For setting the subspace dimension ``p`` manually, set the `eVar`
optional keyword argument of the MCA constructors
either to an integer or to a real number, this latter establishing ``p``
in conjunction with argument `eVarMeth` using the `arev` vector
(see [subspace dimension](@ref)).
By default, `eVar` is set to 0.999.


**Solution**

The MCA solution is given by the
[singular value decoposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
of ``C_{xy}``

``\textrm{SVD}(C_{xy})=U_xΛU_y^{H}``.

It is worth mentioning that

``\widetilde{U}_x\widetilde{Λ}\widetilde{U}_y^H``,

where ``\widetilde{U}_x=[u_{x1} \ldots u_{xp}]`` is the matrix holding the first ``p<n_x`` left singular vectors,
``\widetilde{U}_y=[u_{y1} \ldots u_{yp}]`` is the matrix holding the first ``p<n_y`` right singular vectors and ``\widetilde{Λ}`` is the leading ``p⋅p`` block of ``Λ``, is the
best approximant to ``C_{xy}`` with rank ``p``
in the least-squares sense (Good, 1969)[🎓](@ref).

**Constructors**

Three constructors are available (see here below). The constructed
[LinearFilter](@ref) object holding the MCA will have fields:

`.F[1]`: matrix ``\widetilde{U}_x`` with orthonormal columns holding the first ``p`` left singular vectors in ``U_x``.

`.F[2]`: matrix ``\widetilde{U}_y`` with orthonormal columns holding the first ``p`` right singular vectors in ``U_y``.

`.iF[1]`: the (conjugate) transpose of `.F[1]`

`.iF[2]`: the (conjugate) transpose of `.F[2]`

`.D`: the leading ``p⋅p`` block of ``Λ``, i.e., the singular values associated to `.F` in diagonal form.

`.eVar`: the explained variance for the chosen value of ``p``,
defined in [mca.3].

`.ev`: the vector `diag(Λ)` holding all ``n`` singular values.

`.arev`: the accumulated regularized eigenvalues, defined in [mca.2].

```@docs
mca
```
