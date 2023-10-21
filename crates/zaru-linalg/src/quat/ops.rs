use std::ops::Mul;

use crate::Quat;

impl<T> Mul for Quat<T>
where
    T: Mul + Copy,
{
    type Output = Quat<T::Output>;

    fn mul(self, rhs: Self) -> Self::Output {
        Quat {
            vec: self.vec * rhs.vec,
        }
    }
}
