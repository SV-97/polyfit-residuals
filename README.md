# Description
This package allows to efficiently compute the residual errors (in the least squares sense) of *all* the possible polynomial models on *all* (or all the ones starting at some point) subsegments of some data. So it solves the regression problem of finding a polynomial P minimizing ∑ᵢ (P(xᵢ)-yᵢ)² for some data x₁,...,xₙ, y₁,...,yₙ; but doesn't actually compute the polynomial and instead only returns the optimal target value.

# Algorithm
The algorithm used is essentially a Givens rotation based QR decomposition. For numerical reasons it uses a Newton basis throughout which should yield better results than some other implementations based on the monomial basis / Vandermonde matrix due to being better conditioned. This might incur some additional runtime costs in some places and save some in others.
The algorithm for all segments starting at 0 should be O(nd) and the one for **all** segments should be O(n²d) where n is the size of the input and d the maximal polynomial degree.

# Performance
On my machine (5900X @ ~4.6GHz) I find the following numbers for some benchmarks (generated using criterion):

For `all_residuals`

| data size | maximal degree | time     |
| --------- | -------------- | -------- |
| 100       | 10             | 1.1404ms |
| 500       | 10             | 30.286ms |
| 1,000     | 10             | 123.22ms |
| 5,000     | 10             | 2.9052s  |
| 1,000     | 100            | 2.1673s  |

For `residuals_from_front`

| data size | maximal degree | time     |
| --------- | -------------- | -------- |
| 50,000    | 2              | 2.7330ms |
| 50,000    | 10             | 11.592ms |
| 5,000     | 100            | 22.791ms |
| 5,000     | 1000           | 1.3186s  |
