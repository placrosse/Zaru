use super::ApproxEq;

impl ApproxEq for f32 {
    type Tolerance = Self;

    fn abs_diff_eq(&self, other: &Self, abs_tolerance: Self::Tolerance) -> bool {
        if !self.is_finite() || !other.is_finite() {
            // Ensures that `inf == inf`, `-inf == -inf` and `inf != -inf`.
            return self == other;
        }

        let diff = (self - other).abs();
        diff <= abs_tolerance
    }

    fn rel_diff_eq(&self, other: &Self, rel_tolerance: Self::Tolerance) -> bool {
        if !self.is_finite() || !other.is_finite() {
            // Ensures that `inf == inf`, `-inf == -inf` and `inf != -inf`.
            return self == other;
        }

        let abs_diff = (self - other).abs();
        let abs_self = self.abs();
        let abs_other = other.abs();
        let largest = Self::max(abs_self, abs_other);

        abs_diff <= largest * rel_tolerance
    }

    fn ulps_diff_eq(&self, other: &Self, ulps_tolerance: u32) -> bool {
        if self.is_sign_negative() != other.is_sign_negative() {
            return self == other; // `-0.0` == `+0.0`
        }

        if self.is_nan() || other.is_nan() {
            return false;
        }

        let diff = self.to_bits().abs_diff(other.to_bits());
        diff <= ulps_tolerance
    }
}

impl ApproxEq for f64 {
    type Tolerance = Self;

    fn abs_diff_eq(&self, other: &Self, abs_tolerance: Self::Tolerance) -> bool {
        if !self.is_finite() || !other.is_finite() {
            // Ensures that `inf == inf`, `-inf == -inf` and `inf != -inf`.
            return self == other;
        }

        let diff = (self - other).abs();
        diff <= abs_tolerance
    }

    fn rel_diff_eq(&self, other: &Self, rel_tolerance: Self::Tolerance) -> bool {
        if !self.is_finite() || !other.is_finite() {
            // Ensures that `inf == inf`, `-inf == -inf` and `inf != -inf`.
            return self == other;
        }

        let abs_diff = (self - other).abs();
        let abs_self = self.abs();
        let abs_other = other.abs();
        let largest = Self::max(abs_self, abs_other);

        abs_diff <= largest * rel_tolerance
    }

    fn ulps_diff_eq(&self, other: &Self, ulps_tolerance: u32) -> bool {
        if self.is_sign_negative() != other.is_sign_negative() {
            return self == other; // `-0.0` == `+0.0`
        }

        if self.is_nan() || other.is_nan() {
            return false;
        }

        let diff = self.to_bits().abs_diff(other.to_bits());
        diff <= ulps_tolerance.into()
    }
}

impl<'a, T: ApproxEq<U> + ?Sized, U: ?Sized> ApproxEq<U> for &'a T {
    type Tolerance = T::Tolerance;

    fn abs_diff_eq(&self, other: &U, abs_tolerance: Self::Tolerance) -> bool {
        T::abs_diff_eq(&self, other, abs_tolerance)
    }

    fn rel_diff_eq(&self, other: &U, rel_tolerance: Self::Tolerance) -> bool {
        T::rel_diff_eq(&self, other, rel_tolerance)
    }

    fn ulps_diff_eq(&self, other: &U, ulps_tolerance: u32) -> bool {
        T::ulps_diff_eq(&self, other, ulps_tolerance)
    }
}

impl<T: ApproxEq<U>, U> ApproxEq<[U]> for [T] {
    type Tolerance = T::Tolerance;

    fn abs_diff_eq(&self, other: &[U], abs_tolerance: Self::Tolerance) -> bool {
        for (a, b) in self.iter().zip(other) {
            if !T::abs_diff_eq(a, b, abs_tolerance.clone()) {
                return false;
            }
        }
        true
    }

    fn rel_diff_eq(&self, other: &[U], rel_tolerance: Self::Tolerance) -> bool {
        for (a, b) in self.iter().zip(other) {
            if !T::rel_diff_eq(a, b, rel_tolerance.clone()) {
                return false;
            }
        }
        true
    }

    fn ulps_diff_eq(&self, other: &[U], ulps_tolerance: u32) -> bool {
        for (a, b) in self.iter().zip(other) {
            if !T::ulps_diff_eq(a, b, ulps_tolerance) {
                return false;
            }
        }
        true
    }
}

impl<T: ApproxEq<U>, U, const N: usize> ApproxEq<[U; N]> for [T; N] {
    type Tolerance = T::Tolerance;

    fn abs_diff_eq(&self, other: &[U; N], abs_tolerance: Self::Tolerance) -> bool {
        self.as_slice().abs_diff_eq(other.as_slice(), abs_tolerance)
    }

    fn rel_diff_eq(&self, other: &[U; N], rel_tolerance: Self::Tolerance) -> bool {
        self.as_slice().rel_diff_eq(other.as_slice(), rel_tolerance)
    }

    fn ulps_diff_eq(&self, other: &[U; N], ulps_tolerance: u32) -> bool {
        self.as_slice()
            .ulps_diff_eq(other.as_slice(), ulps_tolerance)
    }
}
