# CSTP

In the [CSP](@ref) one assumes that the multiplicity of data have a common
structure along one dimension of the input data matrices.
For example, in *electroencephalography* ([EEG](https://en.wikipedia.org/wiki/Electroencephalography)) a data matrix ``X``, which is comprised of ``n``
variables corresponding to the spatial locations for the electrodes on the
scalp and ``t`` temporal samples, this is the *spatial* dimension.
The assumption holds because the same brain source engenders a fixed spatial
pattern on the scalp, whereas, in general, the temporal pattern is arbitrary.

The *common spatio-temporal pattern* (CSTP) extends the CSP to situations when
the multiplicity of data have a common structure along *both* dimensions.
In EEG, for example, this is the case of *event-related potentials*
([ERPs](https://en.wikipedia.org/wiki/Event-related_potential)).
The assumption holds because, again, the same brain source engenders a fixed
spatial pattern on the scalp and furthermore ERPs have a quasi-fixed
temporal pattern. As the CSP, the CSTP corresponds to the situation ``m=1`` (one dataset) and ``k=2`` (two observation).

Given a set of ``k`` data matrices ``\{X_1 \ldots X_k\}`` of dimension ``n⋅t``,
with mean ``\bar{X}=\frac{1}{k}\sum_{i=1}^kX_i``, the goal of the CSTP is to
find two matrices ``B_{(1)}`` and ``B_{(2)}`` verifying

``\left \{ \begin{array}{rl}B_{(1)}^TC_{(2)}B_{(1)}=I\\B_{(2)}^TC_{(1)}B_{(2)}=I\\B_{(1)}^T\bar{X}B_{(2)}=Λ \end{array} \right.``, ``\hspace{1cm}`` [cstp.1],

where

``\left \{ \begin{array}{rl}C_{(1)}=\sum_{i=1}^k\frac{1}{t}(X_i^TX_i)\\C_{(2)}=\sum_{i=1}^k\frac{1}{n}(X_iX_i^T) \end{array} \right.``, ``\hspace{1cm}`` [cstp.2]

are the mean covariance matrices along the first and second dimension of the
``X_i`` matrices and ``Λ`` is a diagonal matrix.

In words, the CSTP maximizes the ratio of the variance of the transformed
``\bar{X}`` over the transformed mean covariance matrices ``C_{(1)}`` and
``C_{(2)}``. The CSTP can threfore be used to enhance the signal-to-noise
ratio of data matrices mean estimation. For doing so, we retain the filters ``\widetilde{B}_{(1)}=[b_{(1)1} \ldots b_{(1)p}]`` and
``\widetilde{B}_{(2)}=[b_{(2)1} \ldots b_{(2)p}]``
holding the first ``p`` vectors of ``B_{(1)}`` and ``B_{(2)}``
corresponding to the highest values of the variance ratio ``Λ``.

For the CSTP we define the total variance ratio as

``λ_{TOT}=\sum_{i=1}^nλ_i``,

where the ``λ_i`` are the diagonal elements of ``Λ`` [cstp.1] and
we define the *explained variance* for dimension ``p`` such as

``σ_p=\frac{\sum_{i=1}^pλ_i}{λ_{TOT}}``. ``\hspace{1cm}`` [cstp.3]

The `.arev` field of the CSTP filter
is defined as the vector of accumulated variance ratios

``[σ_1≤\ldots≤σ_n]``, ``\hspace{1cm}`` [cstp.4]

where ``σ_j`` is defined in [cstp.3].

For setting the subspace dimension ``p`` manually, set the `eVar`
optional keyword argument of the CSTP constructors
either to an integer or to a real number, this latter establishing ``p``
in conjunction with argument `eVarMeth` using the `arev` vector
(see [subspace dimension](@ref)).
By default, `eVar` is set to 0.999.

**Solution**

The CSTP solutions ``B_{(1)}`` and ``B_{(2)}`` can be found by a
two-step procedure (Congedo et al., 2016)[🎓](@ref):

1. get two whitening matrices ``\hspace{0.1cm}W_{(1)}\hspace{0.1cm}`` and ``\hspace{0.1cm}W_{(2)}\hspace{0.1cm}`` such that ``\left \{ \begin{array}{rl}W_{(1)}^TC_{(1)}W_{(1)}=I\\W_{(2)}^TC_{(2)}W_{(2)}=I \end{array} \right.``
2. do ``\hspace{0.1cm}\textrm{SVD}(W_{(2)}^T\bar{X}W_{(1)})=UΛV^{T}``

The solutions are ``\hspace{0.1cm}B_{(1)}=W_{(2)}U\hspace{0.1cm}`` and ``\hspace{0.1cm}B_{(2)}=W_{(1)}V``.

**Constructors**

Two constructors are available (see here below). The constructed
[LinearFilter](@ref) object holding the CSTP will have fields:

`.F[1]`: matrix ``\widetilde{B}_{(1)}=[b_{(1)1} \ldots b_{(1)p}]``.
This is the whole matrix ``B_{(1)}`` if ``p=n``.

`.F[2]`: matrix ``\widetilde{B}_{(2)}=[b_{(2)1} \ldots b_{(2)p}]``.
This is the whole matrix ``B_{(2)}`` if ``p=n``

`.iF[1]`: the left-inverse of `.F[1]`

`.iF[2]`: the left-inverse of `.F[2]`

`.D`: the leading ``p⋅p`` block of ``Λ`` in [cstp.1].

`.eVar`: the explained variance for the chosen value of ``p``,
given by the ``p^{th}`` value of [cstp.4].

`.ev`: the vector `diag(Λ)` holding all ``n`` diagonal elements
of matrix ``Λ`` in [cstp.1].

`.arev`: the *accumulated regularized eigenvalues*, defined in [cstp.4].


```@docs
cstp
```
