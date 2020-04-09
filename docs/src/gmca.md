# gMCA

*Generalized Maximum Covariance Analysis* (gMCA) is a multiple approximate joint diagonalization prodedure generalizing the maximum covariance analysis ([MCA](@ref)) to the situation ``m>2`` (number of datasets),
as for MCA with ``k=1`` (one observation).

Let ``{X_1,...,X_m}`` be a set of ``m`` data matrices of dimension
``n⋅t``, where ``n`` is the number of variables
and ``t`` the number of samples, both common to all datasets. From these
data matrices let us estimate

``C_{ij}=\frac{1}{t}X_iX_j^H``, for all ``i,j∈[1...m]``, ``\hspace{1cm}`` [gmca.1]

i.e., all *covariance* (``i=j``) and *cross-covariance* (``i≠j``) matrices.

The gMCA seeks ``m`` matrices ``F_1,...,F_m``
diagonalizing as much as possible all products

``F_i^H C_{ij} F_j``, for all ``i≠j∈[1...m]``. ``\hspace{1cm}`` [gmca.2]

If the MCA (``m=2``) diagonalizes the cross-covariance,
this generalized model (``m>2``) diagonalizes all cross-covariance matrices.

#### alternative model for gMCA

The gMCA constructors also allow to seeks ``m`` matrices ``F_1,...,F_m``
diagonalizing as much as possible all products

``F_i^H C_{ij} F_j``, for all ``i,j∈[1...m]``. ``\hspace{1cm}`` [gmca.3]

As compared to model [gmca.2], this model diagonalizes the covariance
matrices in addition to the cross-covariance matrices.

#### permutation for gMCA

As usual, the approximate diagonalizers ``F_1,...,F_m`` are arbitrary up to a
[scale and permutation](@ref). In gMCA scaling is fixed by
appropriate constraints. For the remaining sign and permutation ambiguities,
*Diagonalizations.jl* attempts to solve them by finding signed permutation
matrices for ``F_1,...,F_m`` so as to make all diagonal elements of [gmca.2] or
[gmca.3] positive and sorted in descending order.

Let

``λ=[λ_1...λ_n]``  ``\hspace{1cm}`` [gmca.4]

be the diagonal elements of

``\frac{1}{m^2-m}\sum_{i≠j=1}^m(F_i^H C_{ij} F_j)`` ``\hspace{1cm}`` [gmca.5]

and ``σ_{TOT}=\sum_{i=1}^nλ_i`` be the total covariance.

We denote ``\widetilde{F}_i=[f_{i1} \ldots f_{ip}]`` the matrix holding the
first ``p<n`` column vectors of ``F_i``, for ``i∈[1...m]``, where ``p`` is the
[subspace dimension](@ref). The *explained variance*
is given by

``σ_p=\frac{\sum_{i=1}^pλ_i}{σ_{TOT}}``, ``\hspace{1cm}`` [gmca.6]

and the *accumulated regularized eigenvalues* (arev) by

``σ_j=\sum_{i=1}^j{σ_i}``, for ``j=[1 \ldots n]``, ``\hspace{1cm}`` [gmca.7]

where ``σ_i`` is given by Eq. [gmca.6].

For setting the subspace dimension ``p`` manually, set the `eVar`
optional keyword argument of the gMCA constructors
either to an integer or to a real number, this latter establishing ``p``
in conjunction with argument `eVarMeth` using the `arev` vector
(see [subspace dimension](@ref)).
By default, `eVar` is set to 0.999.


**Solution**

There is no closed-form solution to the AJD problem in general.
See [Algorithms](@ref).

Note that the solution of the MCA are orthogonal matrices.
In order to mimic this in gMCA use *OJoB*. Using *NoJoB*
will constraint the solution only in the general linear group.

**Constructors**

One constructor is available (see here below). The constructed
[LinearFilter](@ref) object holding the gMCA will have fields:

`.F`: vector of matrices ``\widetilde{F}_1,...,\widetilde{F}_m``
with columns holding the first ``p`` eigenvectors in
``F_1,...,F_m``, or just ``F_1,...,F_m`` if ``p=n``

`.iF`: the vector of the left-inverses of the matrices in `.F`

`.D`: the leading ``p⋅p`` block of ``Λ``, i.e., the elements [gmca.4]
associated to the matrices in `.F` in diagonal form.

`.eVar`: the explained variance [gmca.6] for the
chosen value of ``p``.

`.ev`: the vector ``λ`` [gmca.4].

`.arev`: the accumulated regularized eigenvalues, defined by Eq. [gmca.7].

```@docs
gmca
```
