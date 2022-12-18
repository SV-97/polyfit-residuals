# Description
This package allows to efficiently compute the residual errors (in the least squares sense) of *all* the possible polynomial models on *all* (or all the ones starting at some point) subsegments of some data.
So it minimizes ∑ᵢ (P(xᵢ)-yᵢ)² where i ∈ I and I is a subsegment of 1,...,n where the input data is given by x₁,...,xₙ, y₁,...,yₙ; without actually computing the polynomial P. There is also a function to calculate P explicitly for the given data and some particular degree - however this functionality is not considered the primary focus of this package.

# Algorithm
The algorithm used is essentially a Givens rotation based QR decomposition. For numerical reasons it uses a Newton basis throughout which should yield better results than some other implementations based on the monomial basis / Vandermonde matrix due to being better conditioned. This might incur some additional runtime costs in some places and save some in others.
The algorithm for all segments starting at 0 should be O(nd) and the one for **all** segments should be O(n²d²) where n is the size of the input and d the maximal polynomial degree.

# Performance
Criterion benchmarks run on the authors machine (AMD Ryzen 9 5900X @ ~4.6GHz) may be found [here](BENCHMARKS.md).