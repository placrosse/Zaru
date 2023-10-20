//! Implementations of `std::ops`.

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Index, IndexMut, Mul,
    MulAssign, Neg, Not, Sub, SubAssign,
};

use crate::approx::ApproxEq;

use super::Vector;

impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// More general impl than what the derive generates.
impl<T, U, const N: usize> PartialEq<Vector<U, N>> for Vector<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &Vector<U, N>) -> bool {
        self.0 == other.0
    }
}

impl<T, const N: usize> Eq for Vector<T, N> where T: Eq {}

impl<T, U, const N: usize> PartialEq<[U; N]> for Vector<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &[U; N]) -> bool {
        self.0.eq(other)
    }
}

impl<T, U, const N: usize> PartialEq<Vector<U, N>> for [T; N]
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &Vector<U, N>) -> bool {
        *self == other.0
    }
}

impl<T, U, const N: usize> PartialEq<[U]> for Vector<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &[U]) -> bool {
        self.0.eq(other)
    }
}

impl<T, U, const N: usize> PartialEq<&[U]> for Vector<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &&[U]) -> bool {
        self.0.eq(other)
    }
}

impl<T, const N: usize> ApproxEq for Vector<T, N>
where
    T: ApproxEq,
{
    type Tolerance = T::Tolerance;

    fn abs_diff_eq(&self, other: &Self, abs_tolerance: Self::Tolerance) -> bool {
        self.0.abs_diff_eq(&other.0, abs_tolerance)
    }

    fn rel_diff_eq(&self, other: &Self, rel_tolerance: Self::Tolerance) -> bool {
        self.0.rel_diff_eq(&other.0, rel_tolerance)
    }

    fn ulps_diff_eq(&self, other: &Self, ulps_tolerance: u32) -> bool {
        self.0.ulps_diff_eq(&other.0, ulps_tolerance)
    }
}

/// Element-wise negation.
impl<T, const N: usize> Neg for Vector<T, N>
where
    T: Neg,
{
    type Output = Vector<T::Output, N>;

    fn neg(self) -> Self::Output {
        self.map(T::neg)
    }
}

/// Element-wise logical negation.
impl<T, const N: usize> Not for Vector<T, N>
where
    T: Not,
{
    type Output = Vector<T::Output, N>;

    fn not(self) -> Self::Output {
        self.map(T::not)
    }
}

/// Element-wise addition.
impl<T, const N: usize> Add<Vector<T, N>> for Vector<T, N>
where
    T: Add,
{
    type Output = Vector<T::Output, N>;

    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        self.zip(rhs).map(|(l, r)| l + r)
    }
}

/// Element-wise addition.
impl<T, const N: usize> AddAssign<Vector<T, N>> for Vector<T, N>
where
    T: AddAssign,
{
    fn add_assign(&mut self, rhs: Vector<T, N>) {
        self.as_mut_slice()
            .iter_mut()
            .zip(rhs.into_array())
            .for_each(|(lhs, rhs)| *lhs += rhs);
    }
}

/// Element-wise subtraction.
impl<T, const N: usize> Sub<Vector<T, N>> for Vector<T, N>
where
    T: Sub,
{
    type Output = Vector<T::Output, N>;

    fn sub(self, rhs: Vector<T, N>) -> Self::Output {
        self.zip(rhs).map(|(l, r)| l - r)
    }
}

/// Element-wise subtraction.
impl<T, const N: usize> SubAssign<Vector<T, N>> for Vector<T, N>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, rhs: Vector<T, N>) {
        self.as_mut_slice()
            .iter_mut()
            .zip(rhs.into_array())
            .for_each(|(lhs, rhs)| *lhs -= rhs);
    }
}

/// Element-wise multiplication.
impl<T, const N: usize> Mul<Vector<T, N>> for Vector<T, N>
where
    T: Mul + Copy,
{
    type Output = Vector<T::Output, N>;

    fn mul(self, rhs: Vector<T, N>) -> Self::Output {
        Vector::zip(self, rhs).map(|(a, b)| a * b)
    }
}

/// Element-wise multiplication.
impl<T, const N: usize> MulAssign<Vector<T, N>> for Vector<T, N>
where
    T: MulAssign + Copy,
{
    fn mul_assign(&mut self, rhs: Vector<T, N>) {
        self.as_mut_slice()
            .iter_mut()
            .zip(rhs.into_array())
            .for_each(|(lhs, rhs)| *lhs *= rhs);
    }
}

// NB: above, we choose to support both vector-scalar multiplication as well as element-wise vector-vector multiplication
// This rules out a more generic implementation `Mul<U> for Vector<T, N> where T: Mul<U>`.
// Zaru uses both impls a bunch, but isn't affected much by the lack of genericity, so it seems like
// this tadeoff is worth it?

/// Vector-Scalar multiplication (scaling).
impl<T, const N: usize> Mul<T> for Vector<T, N>
where
    T: Mul + Copy,
{
    type Output = Vector<T::Output, N>;

    fn mul(self, rhs: T) -> Self::Output {
        self.map(|elem| elem * rhs)
    }
}

/// Vector-Scalar multiplication (scaling).
impl<T, const N: usize> MulAssign<T> for Vector<T, N>
where
    T: MulAssign + Copy,
{
    fn mul_assign(&mut self, rhs: T) {
        self.as_mut_slice().iter_mut().for_each(|lhs| *lhs *= rhs);
    }
}

/// Element-wise division.
impl<T, const N: usize> Div<Vector<T, N>> for Vector<T, N>
where
    T: Div + Copy,
{
    type Output = Vector<T::Output, N>;

    fn div(self, rhs: Vector<T, N>) -> Self::Output {
        Vector::zip(self, rhs).map(|(a, b)| a / b)
    }
}

/// Vector-Scalar division (scaling).
impl<T, const N: usize> Div<T> for Vector<T, N>
where
    T: Div + Copy,
{
    type Output = Vector<T::Output, N>;

    fn div(self, rhs: T) -> Self::Output {
        self.map(|elem| elem / rhs)
    }
}

/// Vector-Scalar division (scaling).
impl<T, const N: usize> DivAssign<T> for Vector<T, N>
where
    T: DivAssign + Copy,
{
    fn div_assign(&mut self, rhs: T) {
        self.as_mut_slice().iter_mut().for_each(|lhs| *lhs /= rhs);
    }
}

/// Element-wise bitwise and.
impl<T, const N: usize> BitAnd<Vector<T, N>> for Vector<T, N>
where
    T: BitAnd,
{
    type Output = Vector<T::Output, N>;

    fn bitand(self, rhs: Vector<T, N>) -> Self::Output {
        self.zip(rhs).map(|(l, r)| l & r)
    }
}

/// Element-wise bitwise and.
impl<T, const N: usize> BitAndAssign<Vector<T, N>> for Vector<T, N>
where
    T: BitAndAssign,
{
    fn bitand_assign(&mut self, rhs: Vector<T, N>) {
        self.as_mut_slice()
            .iter_mut()
            .zip(rhs.into_array())
            .for_each(|(lhs, rhs)| *lhs &= rhs);
    }
}

/// Element-wise bitwise or.
impl<T, const N: usize> BitOr<Vector<T, N>> for Vector<T, N>
where
    T: BitOr,
{
    type Output = Vector<T::Output, N>;

    fn bitor(self, rhs: Vector<T, N>) -> Self::Output {
        self.zip(rhs).map(|(l, r)| l | r)
    }
}

/// Element-wise bitwise or.
impl<T, const N: usize> BitOrAssign<Vector<T, N>> for Vector<T, N>
where
    T: BitOrAssign,
{
    fn bitor_assign(&mut self, rhs: Vector<T, N>) {
        self.as_mut_slice()
            .iter_mut()
            .zip(rhs.into_array())
            .for_each(|(lhs, rhs)| *lhs |= rhs);
    }
}

// NB: operations are deliberately not as generic as they could be (eg. using `T: Add<U>`) to allow
// for adding vector-scalar operations in the future.

// NB: a few rarely used ones are omitted (eg. `Rem`) because it is not clear whether elementwise
// or scalar operation is more helpful there.
