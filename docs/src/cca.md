# CCA

*Canonical Correlation Analysis* was first proposed by Hotelling (1936) [🎓](@ref). It can be conceived as the multivariate
extension of Pearson's product-moment correlation, as the bilinear version of [Whitening](@ref) or as the standardized version of
[MCA](@ref); if MCA maximizes the cross-covariance of
two data set, CCA maximizes their correlation.
Like the MCA, the CCA corresponds to the situation ``m=2`` (two datasets) and ``k=1`` (one observation).

As per [MCA](@ref), Let ``X`` and ``Y`` be two ``n_x⋅t`` and ``n_y⋅t`` data matrices, where ``n_x`` and ``n_y`` are the number of variables in ``X`` and ``Y``, respectively and ``t`` the number of samples. We assume here too that the samples in ``X`` and ``Y`` are synchronized. Let ``C_{x}`` be the ``n_x⋅n_x`` covariance matrix of ``X`` and ``C_{y}`` be the ``n_y⋅n_y`` covariance matrix of ``Y``. Let also ``C_{xy}=\frac{1}{t}X^HY`` be the ``n_x⋅n_y`` [cross-covariance matrix](https://en.wikipedia.org/wiki/Cross-covariance). CCA seeks two matrices ``F_x`` and ``F_y`` such that

``\left \{ \begin{array}{rl}F_x^HC_xF_x=I\\F_y^HC_yF_y=I\\F_x^HC_{xy}F_y=Λ \end{array} \right.``, ``\hspace{1cm}`` [cca.1]

where ``Λ`` is a ``n⋅n`` diagonal matrix, with ``n=min(n_x, n_y)``.
The first components (rows) of ``F_x^{H}X`` and ``F_y^{H}Y``
hold the linear combination of ``X`` and ``Y`` with maximal
correlation, the second the linear combination with maximal residual correlation and so on, subject to constraint [cca.1].

In matrix form this can be written such as

``\begin{pmatrix} F_x^H & 0\\0 & F_y^H\end{pmatrix} \begin{pmatrix} C_x & C_{xy}\\C_{xy}^H & C_y\end{pmatrix} \begin{pmatrix} F_x & 0\\0 & F_y\end{pmatrix} = \begin{pmatrix} I & Λ\\Λ & I\end{pmatrix}``. ``\hspace{1cm}`` [cca.2]


If ``n_x=n_y``, from [cca.1] it follows

``\left \{ \begin{array}{rl}F_x^{-H}F_x^{-1}=C_x\\F_y^{-H}F_y^{-1}=C_y\\F_x^{-H}F_y^{-1}=C_{xy} \end{array} \right.``, ``\hspace{1cm}`` [cca.3]


that is, CCA is a special kind of [full-rank factorization](https://marco-congedo.github.io/PosDefManifold.jl/dev/linearAlgebra/#PosDefManifold.frf) of ``C_x`` and ``C_y``.

Equating the left hand sides of the first two expressions in [cca.1] and setting ``E_x=B_xB_y^{-H}``, ``E_y=B_yB_x^{-H}``, it follows

``\left \{ \begin{array}{rl}C_x=E_xC_yE_x^H\\C_y=E_yC_xE_y^H \end{array} \right.``, ``\hspace{1cm}`` [cca.4]

that is, CCA defines a linear transormation linking the covariance
matrices of two data sets.

Since it maximizes the correlation, differently from [MCA](@ref), CCA is not
sensitive to the amplitude of the two processes. This should be kept in mind,
as the correlation of two signals may be high even if the amplitude of one of
the two signal is very low, say, at the noise level. In such a case the
resulting correlation is meaningless.

The *accumulated regularized eigenvalues* (arev) for the CCA are defined as

``σ_j=\sum_{i=1}^j{σ_i}``, for ``j=[1 \ldots n]``, ``\hspace{1cm}`` [cca.5]

where ``σ_i`` is given by

``σ_p=\frac{\sum_{i=1}^pλ_i}{σ_{TOT}}``  ``\hspace{1cm}`` [cca.6]

and ``λ_i`` are the singular values in ``Λ`` of [cca.1].

For setting the subspace dimension ``p`` manually, set the `eVar`
optional keyword argument of the CCA constructors
either to an integer or to a real number, this latter establishing ``p``
in conjunction with argument `eVarMeth` using the `arev` vector
(see [subspace dimension](@ref)).
By default, `eVar` is set to 0.999.

**Solution**

If ``n_x=n_y`` the solutions to the CCA
``B_x`` and ``B_y``
are given by the eigenvector matrix of (non-symmetric) matrices
``C_x^{-1}C_{xy}C_y^{-1}C_{yx}`` and ``C_y^{-1}C_{yx}C_x^{-1}C_{xy}``,
respecively.

A numerically preferable solution, accommodating also the case ``n_x≠n_y``,
is the following two-step procedure, which is the one here implemented:

1. get some whitening matrices ``\hspace{0.1cm}(W_x, W_y)\hspace{0.1cm}`` such that ``\hspace{0.1cm}W_x^HC_xW_x=I\hspace{0.1cm}`` and ``\hspace{0.1cm}W_y^HC_yW_y=I``
2. do ``\hspace{0.1cm}\textrm{SVD}(W_x^HC_{xy}W_y)=U_xΛU_y^{H}``

The solutions are ``\hspace{0.1cm}B_x=W_xU_x\hspace{0.1cm}`` and ``\hspace{0.1cm}B_y=W_yU_y\hspace{0.1cm}``.

**Constructors**

Three constructors are available (see here below). The constructed
[LinearFilter](@ref) object holding the CCA will have fields:

`.F[1]`: matrix ``\widetilde{B}_x=[b_{x1} \ldots b_{xp}]``
with the columns holding the first ``p`` vectors in ``B_x``.

`.F[2]`: matrix ``\widetilde{B}_y=[b_{y1} \ldots b_{yp}]``
with the columns holding the first ``p`` vectors in ``B_y``.

`.iF[1]`: the left-inverse of `.F[1]`

`.iF[2]`: the left-inverse of `.F[2]`

`.D`: the leading ``p⋅p`` block of ``Λ``, i.e., the correlation values
associated to `.F` in diagonal form.

`.eVar`: the explained variance for the chosen value of ``p``,
defined in [cca.6].

`.ev`: the vector `diag(Λ)` holding all ``n`` correlation values.

`.arev`: the accumulated regularized eigenvalues, defined in [cca.5].


```@docs
cca
```
