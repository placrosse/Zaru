use std::ops::{Index, IndexMut, Mul};

use crate::{traits::Number, Matrix, Vector};

impl<T, const R: usize, const C: usize> Index<(usize, usize)> for Matrix<T, R, C> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.0[col][row]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<(usize, usize)> for Matrix<T, R, C> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.0[col][row]
    }
}

// More general `PartialEq` impl than what the derive generates.
impl<T, U, const R: usize, const C: usize> PartialEq<Matrix<U, R, C>> for Matrix<T, R, C>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &Matrix<U, R, C>) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T, const R: usize, const C: usize> Eq for Matrix<T, R, C> where T: Eq {}

/// Matrix * Column Vector.
impl<T, const R: usize, const C: usize> Mul<Vector<T, C>> for Matrix<T, R, C>
where
    T: Number,
{
    type Output = Vector<T, R>;

    fn mul(self, rhs: Vector<T, C>) -> Self::Output {
        Vector::from_fn(|row| (0..C).fold(T::ZERO, |acc, col| acc + self[(row, col)] * rhs[col]))
    }
}

/// Matrix * Matrix.
impl<T, const M: usize, const N: usize, const P: usize> Mul<Matrix<T, N, P>> for Matrix<T, M, N>
where
    T: Number,
{
    type Output = Matrix<T, M, P>;

    fn mul(self, rhs: Matrix<T, N, P>) -> Self::Output {
        Matrix::from_fn(|i, j| (0..N).fold(T::ZERO, |acc, k| acc + self[(i, k)] * rhs[(k, j)]))
    }
}
