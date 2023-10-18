//! Implementations of `std::ops`.

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Index, IndexMut, Mul,
    MulAssign, Neg, Not, Sub, SubAssign,
};

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
impl<T, U, const N: usize> Add<Vector<U, N>> for Vector<T, N>
where
    T: Add<U>,
{
    type Output = Vector<T::Output, N>;

    fn add(self, rhs: Vector<U, N>) -> Self::Output {
        self.zip(rhs).map(|(l, r)| l + r)
    }
}

/// Element-wise addition.
impl<T, U, const N: usize> AddAssign<Vector<U, N>> for Vector<T, N>
where
    T: AddAssign<U>,
{
    fn add_assign(&mut self, rhs: Vector<U, N>) {
        self.as_mut_slice()
            .iter_mut()
            .zip(rhs.into_array())
            .for_each(|(lhs, rhs)| *lhs += rhs);
    }
}

/// Element-wise subtraction.
impl<T, U, const N: usize> Sub<Vector<U, N>> for Vector<T, N>
where
    T: Sub<U>,
{
    type Output = Vector<T::Output, N>;

    fn sub(self, rhs: Vector<U, N>) -> Self::Output {
        self.zip(rhs).map(|(l, r)| l - r)
    }
}

/// Element-wise subtraction.
impl<T, U, const N: usize> SubAssign<Vector<U, N>> for Vector<T, N>
where
    T: SubAssign<U>,
{
    fn sub_assign(&mut self, rhs: Vector<U, N>) {
        self.as_mut_slice()
            .iter_mut()
            .zip(rhs.into_array())
            .for_each(|(lhs, rhs)| *lhs -= rhs);
    }
}

/// Vector-Scalar multiplication (scaling).
impl<T, U, const N: usize> Mul<U> for Vector<T, N>
where
    T: Mul<U>,
    U: Copy,
{
    type Output = Vector<T::Output, N>;

    fn mul(self, rhs: U) -> Self::Output {
        self.map(|elem| elem * rhs)
    }
}

/// Vector-Scalar multiplication (scaling).
impl<T, U, const N: usize> MulAssign<U> for Vector<T, N>
where
    T: MulAssign<U>,
    U: Copy,
{
    fn mul_assign(&mut self, rhs: U) {
        self.as_mut_slice().iter_mut().for_each(|lhs| *lhs *= rhs);
    }
}

/// Vector-Scalar division (scaling).
impl<T, U, const N: usize> Div<U> for Vector<T, N>
where
    T: Div<U>,
    U: Copy,
{
    type Output = Vector<T::Output, N>;

    fn div(self, rhs: U) -> Self::Output {
        self.map(|elem| elem / rhs)
    }
}

/// Vector-Scalar division (scaling).
impl<T, U, const N: usize> DivAssign<U> for Vector<T, N>
where
    T: DivAssign<U>,
    U: Copy,
{
    fn div_assign(&mut self, rhs: U) {
        self.as_mut_slice().iter_mut().for_each(|lhs| *lhs /= rhs);
    }
}

/// Element-wise bitwise and.
impl<T, U, const N: usize> BitAnd<Vector<U, N>> for Vector<T, N>
where
    T: BitAnd<U>,
{
    type Output = Vector<T::Output, N>;

    fn bitand(self, rhs: Vector<U, N>) -> Self::Output {
        self.zip(rhs).map(|(l, r)| l & r)
    }
}

/// Element-wise bitwise and.
impl<T, U, const N: usize> BitAndAssign<Vector<U, N>> for Vector<T, N>
where
    T: BitAndAssign<U>,
{
    fn bitand_assign(&mut self, rhs: Vector<U, N>) {
        self.as_mut_slice()
            .iter_mut()
            .zip(rhs.into_array())
            .for_each(|(lhs, rhs)| *lhs &= rhs);
    }
}

/// Element-wise bitwise or.
impl<T, U, const N: usize> BitOr<Vector<U, N>> for Vector<T, N>
where
    T: BitOr<U>,
{
    type Output = Vector<T::Output, N>;

    fn bitor(self, rhs: Vector<U, N>) -> Self::Output {
        self.zip(rhs).map(|(l, r)| l | r)
    }
}

/// Element-wise bitwise or.
impl<T, U, const N: usize> BitOrAssign<Vector<U, N>> for Vector<T, N>
where
    T: BitOrAssign<U>,
{
    fn bitor_assign(&mut self, rhs: Vector<U, N>) {
        self.as_mut_slice()
            .iter_mut()
            .zip(rhs.into_array())
            .for_each(|(lhs, rhs)| *lhs |= rhs);
    }
}

// NB: a few rarely used ones are omitted (eg. `Rem`) because it is not clear whether elementwise
// or scalar operation is more helpful there.
