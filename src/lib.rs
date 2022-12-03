//! Efficiently compute the residual errors for all possible polynomial models up to some degree
//! for given data.
//!
//! # Example
//! For examples please see [residuals_from_front].
use ndarray::{s, Array2, ArrayView1, ArrayViewMut2, Axis};
use num_traits::real::Real;

use std::fmt::Debug;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum FitError {
    InputsOfDifferentLengths,
    DegreeTooHigh,
}

#[inline]
fn apply_givens<R>(mut arr: ArrayViewMut2<R>, i: usize, j: usize)
where
    R: Real + Clone + Debug,
{
    // Apply a givens rotation eliminating entry [i,j] of the given array *in-place*
    let a = arr[[j, j]];
    let b = arr[[i, j]];
    let r = a.hypot(b);
    let c = a / r;
    let s = -b / r;
    for col_idx in 0..arr.shape()[1] {
        let new_j = c * arr[[j, col_idx]] - s * arr[[i, col_idx]];
        let new_i = s * arr[[j, col_idx]] + c * arr[[i, col_idx]];
        arr[[j, col_idx]] = new_j;
        arr[[i, col_idx]] = new_i;
    }
}

fn generate_system_matrix<R>(ys: ArrayView1<R>, base: ArrayView1<R>) -> Array2<R>
where
    R: Real + Clone + Debug,
{
    let max_dofs = base.len();
    // We create the block matrix
    // [R L]
    // [r l]
    // where R is the triangular matrix corresponding to the newton polynomial evaluations,
    // L are the corresponding right hand sides and r and l are the entries which we'll use
    // later on to store the newton polynomial evaluations and right hand sides for each data
    // point that's not part of the basis.
    let mut system_matrix = Array2::zeros((max_dofs + 1, max_dofs + 1));
    // write `ys` for the basis elements to last column
    system_matrix
        .slice_mut(s![..max_dofs, max_dofs])
        .assign(&ys.slice(s![..max_dofs]));

    // set degree 0 column
    system_matrix.slice_mut(s![.., 0]).fill(R::one());

    // set all the higher order basis rows
    let mut diff = {
        let b1 = base
            .slice(s![1_usize..])
            .to_owned()
            .into_shape([base.len() - 1, 1])
            .unwrap(); //  - base.slice(s![..base.len() - 1]);
                       //lhs[1:, 1:] = np.cumprod(base[1:].reshape(-1, 1) - base[:-1], axis=1)
                       // set degree 0 column
        b1 - base.slice(s![..base.len() - 1])
    };
    // compute cumprod along axis 1
    diff.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr = *curr * prev);

    system_matrix
        .slice_mut(s![1..max_dofs, 1..max_dofs])
        .assign(&diff);
    system_matrix
}

/// Compute the residual squared errors (RSS) for all polynomials of degree at most `max_deg`
/// for the data segments `xs[j..=i]`, `ys[j..=i]` for all `i`, `j`.
///
/// # Returns
/// A vector such that element `j` contains the residuals for `data[j..=i]` for all `i`
/// in the format of [residuals_from_front].
///
/// # Note
/// This function has linear time and memory complexity in the maximal degree and quadratic ones
/// in the length of the input.
/// For further details and an example see [residuals_from_front].
pub fn all_residuals<R>(xs: ArrayView1<R>, ys: ArrayView1<R>, max_degree: usize) -> Vec<Array2<R>>
where
    R: Real + Clone + Debug,
{
    let mut ret = Vec::with_capacity(xs.len());
    for j in 0..xs.len() {
        let max_dof_on_seg = xs.len() - j;
        ret.push(
            residuals_from_front(
                xs.slice(s![j..]),
                ys.slice(s![j..]),
                std::cmp::min(max_dof_on_seg - 1, max_degree),
            )
            .unwrap(),
        );
    }
    ret
}

/// Compute the residual squared errors (RSS) for all polynomials of degree at most `max_deg`
/// for the data segments `xs[0..=i]`, `ys[0..=i]` for all `i`.
///
/// # Returns
/// A 2D float array where the first dimension ranges over `i` and the second one over the
/// polynomial degree. Note that the array will contain 0 at underdetermined indices (for
/// example when fitting a degree 3 polynomial to `data[0..=1]`; with only 2 data point we
/// can't determine 4 unique polynomial coefficients but we can guarantee the error to vanish).
///
/// # Arguments
/// * `xs` - The "inputs values"
/// * `ys` - The "output values" corresponding to the `xs`
/// * `max_degree` - The maximal polynomial degree we wanna calculate the residual errors for.
///
/// # Note
/// This function has linear time and memory complexity in the length of the input and the
/// maximal degree.
/// We consider constant polynomials to have degree 0.
///
/// # Example
/// ```
/// use approx::assert_abs_diff_eq;
/// use ndarray::{arr1, Array2};
/// use polyfit_residuals::residuals_from_front;
///
/// let xs = arr1(&[1., 2., 3., 4., 5.]);
/// let ys = arr1(&[2., 4., 6., 8., 10.]);
/// // We want to fit polynomials with degree <= 2
/// let residuals: Array2<f64> = residuals_from_front(xs.view(), ys.view(), 2).unwrap();
/// // since ys = 2 * xs we expect the linear error to vanish on for example data[0..=2]
/// assert_abs_diff_eq!(residuals[[2, 1]], 0.0)
/// ```
pub fn residuals_from_front<R>(
    xs: ArrayView1<R>,
    ys: ArrayView1<R>,
    max_degree: usize,
) -> Result<Array2<R>, FitError>
where
    R: Real + Clone + Debug,
{
    if xs.len() != ys.len() {
        return Err(FitError::InputsOfDifferentLengths);
    }
    let data_len = xs.len();

    let max_dofs = max_degree + 1;
    if max_dofs > data_len {
        return Err(FitError::DegreeTooHigh);
    }

    // select one base point for each degree of freedom
    let base = xs.slice(s![..max_dofs]).to_owned();
    let mut system_matrix = generate_system_matrix(ys, base.view());
    // first axis is rb, second is deg such that on data[0..=rb] the polynomial of degree deg
    // has the residual error at this index
    let mut residuals: Array2<R> = Array2::zeros([data_len, max_dofs]);

    for (i, mut row) in residuals
        .slice_mut(s![..max_dofs, ..])
        .rows_mut()
        .into_iter()
        .enumerate()
    {
        row[i] = R::zero();
    }

    let last_row_idx = system_matrix.shape()[0] - 1;
    let last_col_idx = system_matrix.shape()[1] - 1;

    // eliminate base mat downwards such that we can start our later eliminations at deg 0
    for base_idx in 1..max_dofs {
        // at base_idx i there's the highest nonzero deg term is i
        for deg in 0..base_idx {
            apply_givens(system_matrix.view_mut(), base_idx, deg);
            residuals[[base_idx, deg]] =
                system_matrix[[base_idx, last_col_idx]].powi(2) + residuals[[base_idx - 1, deg]];
        }
    }

    // repeat procedure we already used on the basis entries: eliminate higher and higher
    // orders for each data point in turn while accumulating the residuals along the way
    for (i, (x, y)) in xs
        .into_iter()
        .zip(ys.into_iter())
        .enumerate()
        .skip(max_dofs)
    {
        // write newton poly evaluation at x into last row of system matrix
        system_matrix[[last_row_idx, 0]] = R::one();
        for col in 1..last_col_idx {
            system_matrix[[last_row_idx, col]] =
                system_matrix[[last_row_idx, col - 1]] * (*x - base[col - 1]);
        }
        // write y into last column of last row of system matrix
        system_matrix[[last_row_idx, last_col_idx]] = *y;
        // eliminate the complete row we just added back to zeros and note down the residuals
        // computed along the way
        for deg in 0..max_dofs {
            apply_givens(system_matrix.view_mut(), last_row_idx, deg);
            residuals[[i, deg]] =
                system_matrix[[last_row_idx, last_col_idx]].powi(2) + residuals[[i - 1, deg]];
        }
    }
    Ok(residuals)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, Array1};

    use super::*;

    #[test]
    fn big_example() {
        let xs = arr1(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        let ys = arr1(&[3., 8., 15., 24., 35., 48., 63., 80., 99., 120.]);
        let our_sol: Array2<f64> = residuals_from_front(xs.view(), ys.view(), 3).unwrap();
        let correct_sol: Array2<f64> = arr2(&[
            [
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
            ],
            [
                1.25000000e+01,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
            ],
            [
                7.26666667e+01,
                6.66666667e-01,
                0.00000000e+00,
                0.00000000e+00,
            ],
            [
                2.49000000e+02,
                4.00000000e+00,
                1.10933565e-31,
                0.00000000e+00,
            ],
            [
                6.54000000e+02,
                1.40000000e+01,
                1.60237371e-31,
                1.52146949e-31,
            ],
            [
                1.45483333e+03,
                3.73333333e+01,
                7.25998552e-30,
                1.53688367e-30,
            ],
            [
                2.88400000e+03,
                8.40000000e+01,
                7.25998552e-30,
                5.54305496e-30,
            ],
            [
                5.25000000e+03,
                1.68000000e+02,
                1.04154291e-29,
                5.54372640e-30,
            ],
            [
                8.94800000e+03,
                3.08000000e+02,
                3.88144217e-29,
                3.18162402e-29,
            ],
            [
                1.44705000e+04,
                5.28000000e+02,
                2.94405355e-28,
                1.11382159e-28,
            ],
        ]);
        assert_abs_diff_eq!(&our_sol, &correct_sol, epsilon = 1e-5);
    }

    #[test]
    fn small_example() {
        let xs = arr1(&[1., 2., 3., 4., 5.]);
        let ys = arr1(&[2., 4., 6., 8., 10.]);
        let correct_sol: Array2<f64> = arr2(&[
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [2.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [8.00000000e+00, 3.74256039e-31, 0.00000000e+00],
            [2.00000000e+01, 1.42772768e-30, 1.53273315e-30],
            [4.00000000e+01, 1.03537994e-30, 1.91026599e-31],
        ]);
        let our_sol: Array2<f64> = residuals_from_front(xs.view(), ys.view(), 2).unwrap();

        assert_abs_diff_eq!(&our_sol, &correct_sol, epsilon = 1e-12);
    }

    #[test]
    fn too_many_dofs() {
        let xs = arr1(&[1., 2.]);
        let ys = arr1(&[2., 4.]);
        assert_eq!(
            residuals_from_front(xs.view(), ys.view(), 2),
            Err(FitError::DegreeTooHigh)
        );
    }

    #[test]
    fn different_lengths() {
        let xs = arr1(&[1., 2.]);
        let ys = arr1(&[2., 4., 6.]);
        assert_eq!(
            residuals_from_front(xs.view(), ys.view(), 2),
            Err(FitError::InputsOfDifferentLengths)
        );
    }

    #[test]
    fn empty_input() {
        let xs: Array1<f64> = arr1(&[]);
        let ys = arr1(&[]);
        assert_eq!(
            residuals_from_front(xs.view(), ys.view(), 2),
            Err(FitError::DegreeTooHigh)
        );
    }

    #[test]
    fn all_residuals_shapes() {
        let xs = arr1(&[1., 2., 3., 4.]);
        let ys = arr1(&[2., 4., 6., 8.]);
        let sol = all_residuals(xs.view(), ys.view(), 2);
        assert_eq!(sol.len(), 4);
        assert_eq!(sol[0].shape(), [4, 3]);
        assert_eq!(sol[1].shape(), [3, 3]);
        assert_eq!(sol[2].shape(), [2, 2]);
        assert_eq!(sol[3].shape(), [1, 1]);
    }
}
