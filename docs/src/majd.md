# mAJD

*Multiple Approximate Joint Diagonalization* (MAJD) is the utmost
general diagonalization prodedure implemented in *Diagonalizations.jl*.
It generalizes the [AJD](@ref) to the case of multiple datasets (``m>1``)
and the [gMCA](@ref)/[gCCA](@ref) to the case of multiple
observations (``k>1``). Therefore, it suits the situation ``m>2``
(multiple datasets) and ``k>2`` (multiple observations) at once.

Let ``{X_{l1},...,X_{lm}}`` be ``k`` sets of ``m`` data matrices of dimension ``n⋅t`` each, indexed by ``l∈[1...k]``.

From these data matrices let us estimate

``C_{lij}=\frac{1}{T}X_{li}X_{lj}^H``, for all ``l∈[1...k]`` and ``i,j∈[1...m]``, ``\hspace{1cm}`` [majd.1]

i.e., all *covariance* (``i=j``) and *cross-covariance* (``i≠j``) matrices
for all ``l∈[1...k]``.

The MAJD seeks ``m`` matrices ``F_1,...,F_m``
diagonalizing as much as possible all products

``F_i^H C_{lij} F_j``, for all ``l∈[1...k]`` and ``i≠j∈[1...m]`` ``\hspace{1cm}`` [majd.2]

or all products

``F_i^H C_{lij} F_j``, for all ``l∈[1...k]`` and ``i,j∈[1...m]``, ``\hspace{1cm}`` [majd.3]

depending on the chosen model (see argument `fullModel` below).

#### pre-whitening for MAJD

Pre-whitening can be applied. In this case, first ``m`` whitening matrices ``W_1,...,W_m`` are found such that

``W_i^H\Big(\frac{1}{k}\sum_{l=1}^kC_{kii}\Big)W_i=I``, for all ``i∈[1...m]`` ``\hspace{1cm}``

then the following transformed AJD problem if solved for ``U_1,...,U_m``:

``U_i^H(W_i^HC_{lij}W_j)U_j≈Λ_{lij}``, for all ``l∈[1...k]`` and ``i,j∈[1...m]``.

Finally, ``F_1,...,F_m`` are obtained as

``F_i=W_iU_i``, for ``i∈[1...m]``. ``\hspace{1cm}``

Notice that:
- matrix ``W`` may be taken rectangular so as to engender a dimensionality reduction at this stage. This may improve the convergence behavior of AJD algorithms if the matrices ``{C_{lii}}`` are not well-conditioned.  
- if this two-step procedure is employed, the final matrices ``F_1,...,F_m`` are never orthogonal, even if the solving AJD algorithm constrains the solutions within the orthogonal group.

#### permutation for MAJD

As usual, the approximate diagonalizers ``F_1,...,F_m`` are arbitrary up to a [scale and permutation](@ref). in MAJD scaling is fixed by
appropriate constraints. For the remaining sign and permutation ambiguities,
*Diagonalizations.jl* attempts to solve them by finding signed permutation
matrices for ``F_1,...,F_m`` so as to make all diagonal elements of [gmca.2] or [gmca.3] positive and sorted in descending order.

Let

``λ=[λ_1...λ_n]``  ``\hspace{1cm}`` [majd.4]

be the diagonal elements of

``\frac{1}{k(m^2-m)}\sum_{j=1}^k\sum_{i≠j=1}^m(F_i^H C_{lij} F_j)`` ``\hspace{1cm}`` [majd.5]

and ``σ_{TOT}=\sum_{i=1}^nλ_i`` be the total variance.

We denote ``\widetilde{F}_i=[f_{i1} \ldots f_{ip}]`` the matrix holding the
first ``p<n`` column vectors of ``F_i``, where ``p`` is the
[subspace dimension](@ref). The *explained variance*
is given by

``σ_p=\frac{\sum_{i=1}^pλ_i}{σ_{TOT}}`` ``\hspace{1cm}`` [majd.6]

and the *accumulated regularized eigenvalues* (arev) by

``σ_j=\sum_{i=1}^j{σ_i}``, for ``j=[1 \ldots n]``, ``\hspace{1cm}`` [majd.7]

where ``σ_i`` is given by Eq. [majd.6].


**Solution**

There is no closed-form solution to the AJD problem in general.
See [Algorithms](@ref).

**Constructors**

One constructor is available (see here below). The constructed
[LinearFilter](@ref) object holding the MAJD will have fields:

`.F`: vector of matrices ``\widetilde{F}_1,...,\widetilde{F}_m``
with columns holding the first ``p`` eigenvectors in
``F_1,...,F_m``, or just ``F_1,...,F_m`` if ``p=n``

`.iF`: the vector of the left-inverses of the matrices in `.F`

`.D`: the leading ``p⋅p`` block of ``Λ``, i.e., the elements [majd.4]
associated to the matrices in `.F` in diagonal form.

`.eVar`: the explained variance [majd.6] for the
chosen value of ``p``.

`.ev`: the vector ``λ`` [majd.4].

`.arev`: the *accumulated regularized eigenvalues*  in [majd.7].

```@docs
majd
```
