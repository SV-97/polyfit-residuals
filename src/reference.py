"""Prototype / reference implementation of algorithm including comparison to numpy based impl."""

from numbers import Integral
import numpy as np


def apply_givens(arr, rhs, i, j):
    """
    Eliminate entry `[i,j]` of `arr` *in place* using a givens rotation and apply the
    same rotation to `rhs`.
    """
    a = arr[j, j]
    b = arr[i, j]
    r = np.hypot(a, b)
    c = a / r  # always non-neg since hypot(a,b) >= 0 and a / sign(a) = |a|
    s = -b / r
    new_j = c * arr[j] - s * arr[i]
    new_i = s * arr[j] + c * arr[i]
    arr[j] = new_j
    arr[i] = new_i

    # modify rhs
    new_y_j = c * rhs[j] - s * rhs[i]
    new_y_i = s * rhs[j] + c * rhs[i]
    rhs[j] = new_y_j
    rhs[i] = new_y_i


def residuals_from_front(xs: np.ndarray, ys: np.ndarray, max_deg: Integral) -> np.ndarray:
    """Compute the residual squared errors for all polynomials of degree at most `max_deg`
    for the data segments `xs[:i]`, `ys[:i]` for all `i`.

    # Returns
    A 2D float array where the first dimension ranges over `i` and the second one over the
    polynomial degree.

    # Note
    This function has linear time and memory complexity in the length of the input and the
    maximal degree.
    """
    if len(xs) != len(ys):
        raise ValueError("Inputs have differing lengths")
    # we'll modify these later on so we make a copy up front
    xs = xs.copy()
    ys = ys.copy()

    max_dofs = max_deg + 1
    base = xs[:max_dofs]  # select one base point for each degree of freedom
    # TODO: integrate this into main matrix and maybe add another
    rhs = ys[:max_dofs]
    # row to lhs and rhs for the elements we wanna eliminate later on
    lhs = np.zeros((max_dofs, max_dofs), dtype=np.float64)
    # set degree 0 column
    lhs[:, 0] = 1.0
    # set all the higher order basis rows
    lhs[1:, 1:] = np.cumprod(base[1:].reshape(-1, 1) - base[:-1], axis=1)

    # first axis is rb, second is deg such that on data[0..=rb] the polynomial of degree deg
    # has the residual error at this index
    residuals = np.full((len(xs), max_dofs), np.nan)

    for i, row in enumerate(residuals[:max_dofs]):
        # with i dofs we can fit i data points exactly and have no residual error
        row[i] = 0.0

    # eliminate base mat downwards such that we can start our later eliminations at deg 0
    for base_idx in range(1, max_dofs):
        # at base_idx i there's the highest nonzero deg term is i
        for deg in range(base_idx):
            apply_givens(lhs, rhs, base_idx, deg)
            residuals[base_idx, deg] = rhs[base_idx]**2 + \
                residuals[base_idx-1, deg]

    # add another row for the data points that aren't in the basis / that we
    # haven't considered yet
    extended_lhs = np.vstack((lhs, np.zeros(max_dofs)))
    extended_rhs = np.hstack((rhs, 0))
    # repeat procedure we already used on the basis entries: eliminate higher and higher
    # orders for each data point in turn while accumulating the residuals along the way
    for i, x, y in zip(range(max_dofs, len(xs)), xs[max_dofs:], ys[max_dofs:]):
        extended_lhs[-1, 0] = 1
        extended_lhs[-1, 1:] = np.cumprod(x - base[:-1])
        extended_rhs[-1] = y
        for deg in range(max_dofs):
            apply_givens(extended_lhs, extended_rhs, -1, deg)
            residuals[i, deg] = extended_rhs[-1]**2 + residuals[i-1, deg]
    return residuals


# Some sample data
xs = np.linspace(1, 10, num=10)
ys = xs**2 + 2*xs  # 3*xs**2 + 2*xs + 1

max_deg = 3

residuals = residuals_from_front(xs, ys, max_deg)
print("Residuals computed using our algorithm:")
print(residuals)


def numpy_residual(xs, ys, degree):
    degrees_of_freedom = degree + 1
    if degrees_of_freedom == len(xs):
        return 0
    elif degrees_of_freedom > len(xs):
        return np.nan
    else:
        return np.polyfit(xs, ys, degree, full=True)[1][0]


np_res = np.array([[numpy_residual(xs[:i], ys[:i], deg)
                  for i in range(1, len(xs)+1)] for deg in range(max_deg+1)]).T
print("Residuals computed using numpy:")
print(np_res)

print("The computed residuals match: ", np.all(np.isclose(
    np_res, residuals) | (np.isnan(np_res) & np.isnan(residuals))))


# Simple reference for weighted case

xs = np.array([0., 0.11111111, 0.22222222, 0.33333333, 0.44444444,
              0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.])
ys = np.array([-4.99719342, -4.76675941, -4.57502219, -4.33219826, -
              4.14968982, -3.98840112, -3.91026478, -3.83464713, -3.93700013, -4.00937516])
ws = np.array([1., 2., 3., 1., 2., 3., 0.5, 0.7, 0.6, 3.])

A = np.diag(ws) @ np.column_stack([xs**0, xs, xs**2, xs**3])
rhs = ws * ys
coef, residual, *_ = np.linalg.lstsq(A, rhs, rcond=None)
poly = np.polynomial.Polynomial(coef)


xs = np.array([1., 2., 3., 4.])
ys = np.array([2., 3., 4., 5.])
ws = np.array([0.1, 0.1, 0.2, 0.6])
A = np.diag(ws) @ np.column_stack([xs**0, xs])
rhs = ws * ys
coef, residual, *_ = np.linalg.lstsq(A, rhs, rcond=None)
poly = np.polynomial.Polynomial(coef)
