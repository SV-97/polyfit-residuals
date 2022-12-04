//! Basic polynomials in a Newton basis.

use num_traits::One;
use std::iter;
use std::ops::Add;
use std::{
    marker::PhantomData,
    ops::{Mul, Sub},
};

pub type OwnedNewtonPolynomial<C, X> = NewtonPolynomial<C, X, Vec<C>, Vec<X>>;
pub type StaticNewtonPolynomial<C, X, const DEGREE: usize> =
    NewtonPolynomial<C, X, [C; DEGREE + 1], [X; DEGREE]>;
pub type RefNewtonPolynomial<'a, C, X> = NewtonPolynomial<C, X, &'a [C], &'a [X]>;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct NewtonPolynomial<C, X, DataC, DataX>
where
    DataC: AsRef<[C]>,
    DataX: AsRef<[X]>,
{
    coeffs: DataC,
    basis_elems: DataX,
    _phantom: PhantomData<(C, X)>,
}

impl<C, X, DataC, DataX> NewtonPolynomial<C, X, DataC, DataX>
where
    DataC: AsRef<[C]>,
    DataX: AsRef<[X]>,
{
    /// Construct the polynomial
    ///     coeffs[0] + coeffs[1] (x - basis_elems[0]) + coeffs[2] (x - basis_elems[0]) * (x - basis_elems[1]) + ...
    /// with `coeffs.len()` degrees of freedom / of degree `basis_elems.len()`.
    pub fn new(coeffs: DataC, basis_elems: DataX) -> Self {
        assert_eq!(coeffs.as_ref().len(), basis_elems.as_ref().len() + 1);
        assert!(!coeffs.as_ref().is_empty());
        Self {
            coeffs,
            basis_elems,
            _phantom: PhantomData,
        }
    }

    /// Evaluate the polynomial "from the left" as 1ₓα₀ + (x-x₁)(α₁+(x-x₂)(α₂ + (x-x₃)(...)))
    pub fn left_eval<Y>(&self, x: X) -> Y
    where
        C: Clone,
        X: Clone + One + Sub<Output = X> + Mul<C, Output = Y> + Mul<Y, Output = Y>,
        Y: Add<Output = Y>,
    {
        let mut it = self
            .basis_elems
            .as_ref()
            .iter()
            .map(|x_i| x.clone() - x_i.clone())
            .rev()
            .chain(iter::once(X::one()))
            .zip(self.coeffs.as_ref().iter().rev());
        let init = {
            let (x, alpha) = it.next().unwrap();
            x * alpha.clone()
        };
        it.fold(init, |acc, (x_i, alpha_i)| {
            x_i * (X::one() * alpha_i.clone() + acc)
        })
    }

    /// Turn a polynomial into it's constituent coefficients and basis elements
    pub fn into_raw(self) -> (DataC, DataX) {
        let Self {
            coeffs,
            basis_elems,
            ..
        } = self;
        (coeffs, basis_elems)
    }
}

#[cfg(test)]
mod tests {
    use super::NewtonPolynomial;

    #[test]
    fn left_eval() {
        let poly = NewtonPolynomial::new(vec![-1, 2, 3], vec![10, 20]);
        assert_eq!(poly.left_eval(10), -1);
        assert_eq!(poly.left_eval(20), 19);
        assert_eq!(poly.left_eval(15), -66);
        assert_eq!(poly.left_eval(2), 415);
        assert_eq!(poly.left_eval(5), 214);
    }
}
