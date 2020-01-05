# gCCA

*Generalized Canonical Correlation Analysis* (gCCA) is a mutiple approximate
joint diagonalization prodedure generalizing the canonical correlation analysis
([CCA](@ref)) to the situation ``m>2`` (number of datasets),
as for CCA with ``k=1`` (one observation). As the CCA is an [MCA](@ref)
carried out on whitened data, so the gCCA is a [gMCA](@ref)
carried out on whitened data.

Let ``{X_1,...,X_m}`` be a set of ``m`` data matrices of dimension
``n⋅t``, where ``n`` is the number of variables
and ``t`` the number of samples, both common to all datasets. From these
data matrices let us estimate

``C_{ij}=\frac{1}{T}X_iX_j^H``, for all ``i,j∈[1...m]``, ``\hspace{1cm}`` [gcca.1]

i.e., all *covariance* (``i=j``) and *cross-covariance* (``i≠j``) matrices.

The gMCA seeks ``m`` matrices ``F_1,...,F_m``
diagonalizing as much as possible all products

``F_i^H C_{ij} F_j``, for all ``i≠j∈[1...m]``. ``\hspace{1cm}`` [gcca.2]

under costraint

``F_i^H C_{ii} F_i=I``, for all ``i∈[1...m]``. ``\hspace{1cm}`` [gcca.3]

#### permutation for gCCA

Given constraint [gcca.3], the scaling of approximate diagonalizers
``F_1,...,F_m`` are fixed, however there is still a sign and permutation
ambiguity (see [scale and permutation](@ref)). *Diagonalizations.jl* attempts
to solve them by finding signed permutation
matrices for ``F_1,...,F_m`` so as to make all diagonal elements of [gcca.2] positive and sorted in descending order.

Let

``λ=[λ_1...λ_n]``  ``\hspace{1cm}`` [gcca.4]

be the diagonal elements of

``\frac{1}{m^2-m}\sum_{i≠j=1}^m(F_i^H C_{ij} F_j)`` ``\hspace{1cm}`` [gcca.5]

and ``σ_{TOT}=\sum_{i=1}^nλ_i`` be the total correlation.

We denote ``\widetilde{F}_i=[f_{i1} \ldots f_{ip}]`` the matrix holding the
first ``p<n`` column vectors of ``F_i``, where ``p`` is the
[subspace dimension](@ref). The *explained variance*
is given by

``σ_p=\frac{\sum_{i=1}^pλ_i}{σ_{TOT}}`` ``\hspace{1cm}`` [gcca.6]

and the *accumulated regularized eigenvalues* (arev) by

``σ_j=\sum_{i=1}^j{σ_i}``, for ``j=[1 \ldots n]``, ``\hspace{1cm}`` [gcca.7]

where ``σ_i`` is given by Eq. [gcca.6].

For setting the subspace dimension ``p`` manually, set the `eVar`
optional keyword argument of the gCCA constructors
either to an integer or to a real number, this latter establishing ``p``
in conjunction with argument `eVarMeth` using the `arev` vector
(see [subspace dimension](@ref)).
By default, `eVar` is set to 0.999.



**Solution**

There is no closed-form solution to the gCCA problem in general.
*Diagonalizations.jl* implements the following iterative algorithms:

| Algorithm   | Constraint | Reference |
|:----------|:----------|:----------|
| OJoB | ``F`` orthogonal | Congedo et al (2011, 2012); Congedo (2013)|

Note that solving algorithm constraining the solution
to the general linear group, like *NoJoB* do not suit
gCCA as they do not encure constraint [gcca.2].


**Constructors**

One constructor is available (see here below). The constructed
[LinearFilter](@ref) object holding the gCCA will have fields:

`.F`: vector of matrices ``\widetilde{F}_1,...,\widetilde{F}_m``
with columns holding the first ``p`` eigenvectors in
``F_1,...,F_m``, or just ``F_1,...,F_m`` if ``p=n``

`.iF`: the vector of the left-inverses of the matrices in `.F`

`.D`: the leading ``p⋅p`` block of ``Λ``, i.e., the elements [gcca.4]
associated to the matrices in `.F` in diagonal form.

`.eVar`: the explained variance [gcca.6] for the
chosen value of ``p``.

`.ev`: the vector ``λ`` [gcca.4].

`.arev`: the accumulated regularized eigenvalues, defined by [gcca.7]

```@docs
gcca
```
