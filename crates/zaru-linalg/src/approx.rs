//! Approximate equality.

mod impls;

use std::{fmt, panic::Location};

/// Types that can be compared for *approximate equality*.
///
/// Compound types implementing this trait are considered *equal* if all of their fields are.
///
/// For more information on the subtleties of approximate floating-point number comparison, see:
/// <https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/>
pub trait ApproxEq<Rhs: ?Sized = Self> {
    /// Type representing the tolerance for absolute and relative comparisons.
    ///
    /// This is almost always either [`f32`] or [`f64`], depending on which one is the underlying
    /// primitive type being compared.
    type Tolerance: DefaultTolerances + Copy;

    /// Performs an *absolute comparison* of `self` and `other`.
    ///
    /// If the absolute difference of the compared values is less than or equal to `abs`, the values
    /// are considered to be equal.
    fn abs_diff_eq(&self, other: &Rhs, abs_tolerance: Self::Tolerance) -> bool;

    /// Performs a *relative comparison* of `self` and `other`.
    ///
    /// If the absolute difference of the compared values is less than or equal to the largest of
    /// the two values times `rel_tolerance`, the values are considered to be equal.
    fn rel_diff_eq(&self, other: &Rhs, rel_tolerance: Self::Tolerance) -> bool;

    /// Performs a comparison of `self` and `other` by counting the number of
    /// [*units in the last place*] (ULPs) between the values.
    ///
    /// If there are at most `ulps` values between the two compared values, they are considered to
    /// be equal.
    ///
    /// `NaN` is never considered equal to anything. `-0.0` and `+0.0` are always considered equal,
    /// other values with differing signs are never considered equal.
    ///
    /// [*units in the last place*]: https://en.wikipedia.org/wiki/Unit_in_the_last_place
    fn ulps_diff_eq(&self, other: &Rhs, ulps_tolerance: u32) -> bool;
}

/// Trait implemented for the `Tolerance` value of [`ApproxEq`] implementations.
///
/// This supplies the default tolerances used by [`assert_approx_eq!`][crate::assert_approx_eq]
/// and [`assert_approx_ne!`][crate::assert_approx_ne].
pub trait DefaultTolerances {
    /// Default tolerance for *absolute comparisons* via [`ApproxEq::abs_diff_eq`].
    const DEFAULT_ABS_TOLERANCE: Self;
    /// Default tolerance for *relative comparisons* via [`ApproxEq::rel_diff_eq`].
    const DEFAULT_REL_TOLERANCE: Self;
    /// Default tolerance for *ULPS comparisons* via [`ApproxEq::ulps_diff_eq`].
    const DEFAULT_ULPS_TOLERANCE: u32;
}

impl DefaultTolerances for f32 {
    const DEFAULT_ABS_TOLERANCE: Self = Self::EPSILON;
    const DEFAULT_REL_TOLERANCE: Self = Self::EPSILON;
    const DEFAULT_ULPS_TOLERANCE: u32 = 4;
}

impl DefaultTolerances for f64 {
    const DEFAULT_ABS_TOLERANCE: Self = Self::EPSILON;
    const DEFAULT_REL_TOLERANCE: Self = Self::EPSILON;
    const DEFAULT_ULPS_TOLERANCE: u32 = 4;
}

/// Assertion guard returned by the [`assert_approx_eq!`][crate::assert_approx_eq]
/// and [`assert_approx_ne!`][crate::assert_approx_ne] macros.
///
/// This type will check the assertion when dropped, and has methods that allow configuring the
/// comparison method and tolerances to use. It supports 3 ways of comparing values that can be
/// enabled by calling the appropriate methods:
///
/// - [`Asserter::abs`] for comparing the value's *absolute difference* via [`ApproxEq::abs_diff_eq`].
/// - [`Asserter::rel`] for comparing the value's *relative difference* via [`ApproxEq::rel_diff_eq`].
/// - [`Asserter::ulps`] for comparing the values by checking how many other values can fit between
///   them via [`ApproxEq::ulps_diff_eq`].
///
/// If more than one of these methods is called, the values will be considered equal if *any*
/// comparison considers them equal (ie. the results are ORed together).
///
/// If none of the methods are called to customize the behavior, a *default comparison* is
/// performed: the values compare equal if an *absolute comparison* with a tolerance of
/// [`DEFAULT_ABS_TOLERANCE`] considers them equal, *or* if a *relative comparison* with a
/// tolerance of [`DEFAULT_REL_TOLERANCE`] considers them equal.
///
/// [`DEFAULT_ABS_TOLERANCE`]: DefaultTolerances::DEFAULT_ABS_TOLERANCE
/// [`DEFAULT_REL_TOLERANCE`]: DefaultTolerances::DEFAULT_REL_TOLERANCE
pub struct Asserter<'a, T>
where
    T: ApproxEq + fmt::Debug,
{
    left: &'a T,
    right: &'a T,
    kind: AssertionKind,
    location: &'static Location<'static>,
    msg: Option<fmt::Arguments<'a>>,
    abs: Option<T::Tolerance>,
    rel: Option<T::Tolerance>,
    ulps: Option<u32>,
}

impl<'a, T> Asserter<'a, T>
where
    T: ApproxEq + fmt::Debug,
{
    #[doc(hidden)]
    #[track_caller]
    pub fn new(
        left: &'a T,
        right: &'a T,
        kind: AssertionKind,
        msg: Option<fmt::Arguments<'a>>,
    ) -> Self {
        Self {
            left,
            right,
            kind,
            location: Location::caller(),
            msg,
            abs: None,
            rel: None,
            ulps: None,
        }
    }

    /// Perform an *absolute comparison* of the values with the given tolerance.
    ///
    /// If the absolute difference of the compared values is less than or equal to `abs`, the values
    /// are considered to be equal.
    ///
    /// This type of comparison is typically a good choice when comparing values that are relatively
    /// close to zero and potentially have opposing signs.
    pub fn abs(&mut self, abs: T::Tolerance) -> &mut Self {
        self.abs = Some(abs);
        self
    }

    /// Perform a *relative comparison* of the values with the given tolerance.
    ///
    /// If the absolute difference of the compared values is less than or equal to the largest of
    /// the two values times `rel`, the values are considered to be equal.
    ///
    /// This type of comparison is a good default for numbers that aren't very close to zero. For
    /// numbers close to zero, a very large relative tolerance might be required (eg. two numbers
    /// close to zero but with opposing signs will only compare equal with a relative tolerance of
    /// at least 2.0; any non-zero number will only compare equal to 0.0 with a relative tolerance
    /// of at least 1.0).
    pub fn rel(&mut self, rel: T::Tolerance) -> &mut Self {
        self.rel = Some(rel);
        self
    }

    /// Perform a comparison by counting the number of [*units in the last place*] between the
    /// values.
    ///
    /// If there are at most `ulps` values between the two compared values, they are considered to
    /// be equal.
    ///
    /// This type of comparison has the nice property of respecting the uneven distribution of
    /// floating-point numbers: for example, floats are much denser between 1.0 and 2.0 than between
    /// 1001.0 and 1002.0. However, it does not work very well with floats that are close to zero
    /// (floats with opposing sign are billions of ULPs apart, even if they are both very close to
    /// 0.0).
    ///
    /// [*units in the last place*]: https://en.wikipedia.org/wiki/Unit_in_the_last_place
    pub fn ulps(&mut self, ulps: u32) -> &mut Self {
        self.ulps = Some(ulps);
        self
    }

    fn equal(&mut self) -> bool {
        if let Some(abs) = self.abs.take() {
            if T::abs_diff_eq(self.left, self.right, abs) {
                return true;
            }
        }
        if let Some(rel) = self.rel.take() {
            if T::rel_diff_eq(self.left, self.right, rel) {
                return true;
            }
        }
        if let Some(ulps) = self.ulps.take() {
            if T::ulps_diff_eq(self.left, self.right, ulps) {
                return true;
            }
        }

        false // (currently unreachable)
    }
}

impl<'a, T> Drop for Asserter<'a, T>
where
    T: ApproxEq + fmt::Debug,
{
    // FIXME: the largest UX issue is that `#[track_caller]` does not work correctly on destructors
    // (the location of `ptr::drop_in_place` is blamed instead of the user code dropping the value)
    //#[track_caller]
    fn drop(&mut self) {
        if self.abs.is_none() && self.rel.is_none() && self.ulps.is_none() {
            // Configure default behavior.
            self.abs = Some(T::Tolerance::DEFAULT_ABS_TOLERANCE);
            self.rel = Some(T::Tolerance::DEFAULT_REL_TOLERANCE);
        }

        let equal = self.equal();
        if (!equal && self.kind == AssertionKind::Eq) || (equal && self.kind == AssertionKind::Ne) {
            assert_failed_inner(self.left, self.right, self.kind, self.location, self.msg);
        }
    }
}

fn assert_failed_inner(
    left: &dyn fmt::Debug,
    right: &dyn fmt::Debug,
    kind: AssertionKind,
    location: &Location<'_>,
    args: Option<fmt::Arguments<'_>>,
) -> ! {
    let op = match kind {
        AssertionKind::Eq => "==",
        AssertionKind::Ne => "!=",
    };
    match args {
        // If the panic output takes you here, you've probably clicked on the wrong location.
        // `#[track_caller]` doesn't work correctly on `drop`, so we manually print the correct
        // location of the assertion.
        Some(args) => panic!(
            r#"assertion `left {op} right` failed at {location}: {args}
  left: {left:?}
 right: {right:?}"#
        ),
        None => panic!(
            r#"assertion `left {op} right` failed at {location}
  left: {left:?}
 right: {right:?}"#
        ),
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AssertionKind {
    Eq,
    Ne,
}

/// Asserts that two expressions are approximately equal to each other (using [`ApproxEq`]).
///
/// This macro functions identically to [`assert_eq!`], except in that it uses the [`ApproxEq`]
/// trait to perform an approximate comparison, and returns an [`Asserter`] that can be used to
/// configure the exact type of comparison, as well as the tolerance values to use.
///
/// Also see [`assert_approx_ne!`].
///
/// # Examples
///
/// Default approximate comparison:
///
/// ```
/// # use zaru_linalg::*;
/// let one = (0..10).fold(0.0, |acc, _| acc + 0.1);
/// assert_approx_eq!(one, 1.0);
/// ```
///
/// Perform absolute and relative comparisons with custom tolerance values:
///
/// ```
/// # use zaru_linalg::*;
/// assert_approx_eq!(100.0, 99.0).abs(1.0);
/// assert_approx_eq!(100.0, 99.0).rel(0.01);
/// ```
///
/// Compare values via ULPs, based on the number of floats that fit between them:
///
/// ```
/// # use zaru_linalg::*;
/// assert_approx_eq!(1.0, 1.0 + f64::EPSILON).ulps(1);
/// ```
#[macro_export]
macro_rules! assert_approx_eq {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::approx::Asserter::new(&$lhs, &$rhs, $crate::approx::AssertionKind::Eq, ::core::option::Option::None)
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::approx::Asserter::new(&$lhs, &$rhs, $crate::approx::AssertionKind::Eq, ::core::option::Option::Some(::core::format_args!($($arg)+)))
    };
}

/// Asserts that two expressions are *not* approximately equal to each other (using [`ApproxEq`]).
///
/// This macro functions identically to [`assert_ne!`], except in that it uses the [`ApproxEq`]
/// trait to perform an approximate comparison, and returns an [`Asserter`] that can be used to
/// configure the exact type of comparison, as well as the tolerance values to use.
///
/// Also see [`assert_approx_eq!`].
///
/// # Examples
///
/// Perform absolute and relative comparisons with custom tolerance values:
///
/// ```
/// # use zaru_linalg::*;
/// assert_approx_ne!(100.0, 99.0).abs(0.5);
/// assert_approx_ne!(100.0, 99.0).rel(0.005);
/// ```
///
/// Compare values via ULPs, based on the number of floats that fit between them:
///
/// ```
/// # use zaru_linalg::*;
/// assert_approx_ne!(1.0, 1.0 + f64::EPSILON + f64::EPSILON).ulps(1);
/// ```
#[macro_export]
macro_rules! assert_approx_ne {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::approx::Asserter::new(
            &$lhs,
            &$rhs,
            $crate::approx::AssertionKind::Ne,
            ::core::option::Option::None
        )
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::approx::Asserter::new(
            &$lhs,
            &$rhs,
            $crate::approx::AssertionKind::Ne,
            ::core::option::Option::Some(::core::format_args!($($args)+))
        )
    };
}

#[cfg(test)]
mod tests {
    #[test]
    #[should_panic(expected = "assertion `left != right` failed")]
    fn fail_ne() {
        assert_approx_ne!(1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn fail_eq() {
        assert_approx_eq!(1.0, 2.0);
    }

    #[test]
    #[should_panic(expected = "my message")]
    fn assertion_message() {
        assert_approx_eq!(1.0, 2.0, "my message");
    }

    #[test]
    fn rel() {
        assert_approx_eq!(1.0, 1.001).rel(0.01);
        assert_approx_eq!(1.0, -1.0).rel(2.0);
        assert_approx_eq!(0.0, 0.00001).rel(1.0);
    }

    #[test]
    fn epsilon() {
        assert_approx_eq!(1.0, 1.0 + f32::EPSILON);
        assert_approx_eq!(1.0, 1.0 + f32::EPSILON).ulps(1);
        assert_approx_ne!(1.0, 1.0 + f32::EPSILON).ulps(0);
    }

    #[test]
    fn negative() {
        assert_approx_ne!(1.0, -1.0);
        assert_approx_ne!(1.0, -1.0).abs(1.0);
        assert_approx_eq!(1.0, -1.0).abs(2.0);
        assert_approx_eq!(-1.0, -1.0).abs(0.0);
        assert_approx_eq!(-1.0, -1.0).rel(0.0);
        assert_approx_eq!(-1.0, -1.0).ulps(0);
    }

    #[test]
    fn nan() {
        assert_approx_ne!(f32::NAN, f32::NAN).abs(0.0);
        assert_approx_ne!(f32::NAN, f32::NAN).rel(0.0);
        assert_approx_ne!(f32::NAN, f32::NAN).ulps(0);
        assert_approx_ne!(f32::NAN, f32::NAN).abs(1.0);
        assert_approx_ne!(f32::NAN, f32::NAN).rel(1.0);
        assert_approx_ne!(f32::NAN, f32::NAN).ulps(100);

        assert_approx_ne!(f32::NAN, 0.0).abs(0.0);
        assert_approx_ne!(f32::NAN, 0.0).rel(0.0);
        assert_approx_ne!(f32::NAN, 0.0).ulps(0);
        assert_approx_ne!(f32::NAN, 0.0).abs(1.0);
        assert_approx_ne!(f32::NAN, 0.0).rel(1.0);
        assert_approx_ne!(f32::NAN, 0.0).ulps(100);
    }

    #[test]
    fn inf() {
        assert_approx_eq!(f32::INFINITY, f32::INFINITY).abs(0.0);
        assert_approx_eq!(f32::INFINITY, f32::INFINITY).rel(0.0);
        assert_approx_eq!(f32::INFINITY, f32::INFINITY).ulps(0);
        assert_approx_ne!(f32::INFINITY, f32::MAX).abs(10000.0);
        assert_approx_ne!(f32::INFINITY, f32::MAX).rel(10000.0);
        assert_approx_ne!(f32::MAX, f32::INFINITY).abs(10000.0);
        assert_approx_ne!(f32::MAX, f32::INFINITY).rel(10000.0);
        assert_approx_ne!(f32::MAX, f32::INFINITY).ulps(0);
        assert_approx_eq!(f32::MAX, f32::INFINITY).ulps(1);

        assert_approx_eq!(f64::INFINITY, f64::INFINITY).abs(0.0);
        assert_approx_eq!(f64::INFINITY, f64::INFINITY).rel(0.0);
        assert_approx_eq!(f64::INFINITY, f64::INFINITY).ulps(0);
        assert_approx_ne!(f64::INFINITY, f64::MAX).abs(10000.0);
        assert_approx_ne!(f64::INFINITY, f64::MAX).rel(10000.0);
        assert_approx_ne!(f64::MAX, f64::INFINITY).abs(10000.0);
        assert_approx_ne!(f64::MAX, f64::INFINITY).rel(10000.0);
        assert_approx_ne!(f64::MAX, f64::INFINITY).ulps(0);
        assert_approx_eq!(f64::MAX, f64::INFINITY).ulps(1);
    }
}
